import Metal
import simd

/// The per-tube stages of a `PBDTubeBatch` substep, batched: gather, integrate, capsule
/// build (+ broadphase boxes), SDF contact with a broadphase, floor, scatter and tube expand,
/// each ONE dispatch over every tube's particles in the batch's flat buffer (DH-1623).
///
/// Before this the batch encoded those stages once PER TUBE — ~10 tiny dispatches per tube
/// per substep, run nearly back to back by the serial compute encoder, so Daydream Home's
/// frank rain paid ~0.16 ms of GPU per frank per substep however little each dispatch did
/// (160 franks ≈ 1 650 dispatches a substep). Now a substep is O(colour groups) dispatches
/// whatever the population. The kernels live in `Shaders/PBDFlatStages.h`; each computes
/// exactly what its per-tube twin did (see that file).
///
/// Used only when the tube set is eligible (`isEligible`): the flat constraint solve is on,
/// every tube is an open chain with ≥ 2 particles and the same radius / damping / floor band
/// (the per-tube integrate and floor uniforms are then one uniform), and no scene pass needs
/// the tubes' own buffers mid-substep (`extraConstraintPass == nil`). Anything else keeps the
/// per-tube encoding, unchanged.
@MainActor
final class PBDTubeBatchFlatStages {

    // Mirrors of the Metal structs in PBDFlatStages.h — byte for byte.
    private struct FlatTube {
        var particles: UInt64
        var start: UInt32
        var count: UInt32
        var colliderBase: UInt32
        var ownerID: UInt32
    }
    private struct BuildUniforms {
        var tubeCount: UInt32
        var radius: Float
        var reach: Float
        var pad: UInt32 = 0
    }
    private struct BroadUniforms {
        var totalParticles: UInt32
        var colliderCount: UInt32
        var selfRadius: Float
        var collideStiffness: Float
        var collideFriction: Float
        var tubeCount: UInt32
        var staticCount: UInt32
        var countTests: UInt32
    }

    /// Keeps the tubes' particle buffers alive across a command buffer's completion handler.
    final class Hold: @unchecked Sendable {
        let buffers: [MTLBuffer]
        init(_ buffers: [MTLBuffer]) { self.buffers = buffers }
    }

    /// Slack added to a tube's broadphase reach so float rounding in the closest-point
    /// computation can never make the box test reject a capsule the contact would have
    /// touched. 0.1 mm — far below any collider radius, far above any rounding.
    static let broadphaseMargin: Float = 1e-4

    private let device: MTLDevice
    private let gather: MTLComputePipelineState
    private let scatter: MTLComputePipelineState
    private let build: MTLComputePipelineState
    private let broad: MTLComputePipelineState
    private let expandBatch: MTLComputePipelineState

    // ── Per tube set (rebuilt by `rebuild` on markDirty; NEW buffers each time, so a command
    //    buffer still in flight keeps reading the set it was encoded against) ──────────────
    private var tubeTable: MTLBuffer?
    private var tubeOf: MTLBuffer?
    private var bounds: MTLBuffer?
    private var vertexBases: MTLBuffer?
    private var expandUniforms: MTLBuffer?
    private var expandOutputs: (position: MTLBuffer, normal: MTLBuffer, color: MTLBuffer)?
    private var expandRings = 0
    private var tubeCount = 0
    private var totalParticles = 0
    /// Every tube's own particle buffer: made resident on each substep's encoder (the kernels
    /// reach them through GPU addresses), and retained by the substep's completion handler.
    private(set) var residentBuffers: [MTLBuffer] = []

    /// Diagnostic contact-test counters: [0] narrow-phase contact evaluations, [1] tube-box
    /// tests. Counted only while `countsTests` is on (an atomic per particle per pass).
    let counters: MTLBuffer
    var countsTests = false
    private(set) var countedPasses = 0

    init?(device: MTLDevice) {
        let cache = SimPipelineCache.shared
        guard let g = cache.pipelineState(name: "pbdFlatGather", device: device),
              let s = cache.pipelineState(name: "pbdFlatScatter", device: device),
              let b = cache.pipelineState(name: "pbdBuildCapsuleCollidersFlat", device: device),
              let c = cache.pipelineState(name: "pbdSDFCollideBroad", device: device),
              let e = cache.pipelineState(name: "pbdTubeExpandBatch", device: device),
              let counters = device.makeBuffer(length: 2 * MemoryLayout<UInt32>.stride,
                                               options: .storageModeShared)
        else { return nil }
        counters.label = "PBDTubeBatch.contactTestCounters"
        memset(counters.contents(), 0, counters.length)
        self.device = device
        gather = g; scatter = s; build = b; broad = c; expandBatch = e
        self.counters = counters
    }

    // MARK: - Eligibility + per-set layout

    /// True when every per-tube stage of `tubes` can run as one dispatch (see the type doc).
    static func isEligible(_ tubes: [PBDTubeRenderer]) -> Bool {
        guard let first = tubes.first else { return false }
        let s0 = first.solver
        return tubes.allSatisfy { t in
            t.solver.ringCount == 0
                && t.solver.particleBuffer.count >= 2
                && t.radius == first.radius
                && t.solver.damping == s0.damping
                && t.solver.floorBand == s0.floorBand
        }
    }

    /// Lay out the tube set: flat ranges (cumulative particle counts, the batch's flat
    /// order), collider bases (`colliderOffsets`, the batch's own cache), owner ids, and —
    /// when every tube expands into ONE shared batched mesh — the one-dispatch expand.
    func rebuild(tubes: [PBDTubeRenderer], colliderOffsets: [Int]) -> Bool {
        let n = tubes.count
        let total = tubes.reduce(0) { $0 + $1.solver.particleBuffer.count }
        guard n > 0, total > 0, colliderOffsets.count == n,
              let table = device.makeBuffer(length: n * MemoryLayout<FlatTube>.stride,
                                            options: .storageModeShared),
              let owner = device.makeBuffer(length: total * MemoryLayout<UInt32>.stride,
                                            options: .storageModeShared),
              let box = device.makeBuffer(length: 2 * n * MemoryLayout<SIMD4<Float>>.stride,
                                          options: .storageModePrivate)
        else { return false }
        table.label = "PBDTubeBatch.flatTubes"
        owner.label = "PBDTubeBatch.tubeOfParticle"
        box.label = "PBDTubeBatch.tubeBounds"
        let tp = table.contents().bindMemory(to: FlatTube.self, capacity: n)
        let op = owner.contents().bindMemory(to: UInt32.self, capacity: total)
        var start = 0
        for (t, tube) in tubes.enumerated() {
            let count = tube.solver.particleBuffer.count
            tp[t] = FlatTube(particles: tube.solver.particleBuffer.buffer.gpuAddress,
                             start: UInt32(start), count: UInt32(count),
                             colliderBase: UInt32(colliderOffsets[t]),
                             ownerID: tube.solver.ownerID)
            for k in 0..<count { op[start + k] = UInt32(t) }
            start += count
        }
        tubeTable = table; tubeOf = owner; bounds = box
        tubeCount = n; totalParticles = total
        residentBuffers = tubes.map { $0.solver.particleBuffer.buffer }

        // One-dispatch expand only when every tube writes a slot of the SAME shared mesh.
        expandOutputs = nil; vertexBases = nil; expandUniforms = nil
        if let ext0 = tubes[0].externalOutput,
           tubes.allSatisfy({ t in
               guard let e = t.externalOutput else { return false }
               return e.position === ext0.position && e.normal === ext0.normal && e.color === ext0.color
           }),
           let vb = device.makeBuffer(length: n * MemoryLayout<UInt32>.stride, options: .storageModeShared),
           let eu = device.makeBuffer(length: n * MemoryLayout<TubeExpandUniforms>.stride,
                                      options: .storageModeShared) {
            vb.label = "PBDTubeBatch.expandVertexBase"
            eu.label = "PBDTubeBatch.expandUniforms"
            let vp = vb.contents().bindMemory(to: UInt32.self, capacity: n)
            for (t, tube) in tubes.enumerated() { vp[t] = UInt32(tube.externalOutput!.vertexBase) }
            vertexBases = vb; expandUniforms = eu
            expandOutputs = (ext0.position, ext0.normal, ext0.color)
            expandRings = tubes.map(\.visualRingCount).max() ?? 0
        }
        return true
    }

    // MARK: - Encoding (one dispatch per stage)

    func makeResident(_ enc: MTLComputeCommandEncoder) {
        enc.useResources(residentBuffers, usage: [.read, .write])
    }

    private func dispatch1D(_ enc: MTLComputeCommandEncoder, _ p: MTLComputePipelineState, _ n: Int) {
        let w = min(n, p.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    /// Every tube's own buffer → the flat buffer (ownerID stamped). `pbdCopyToFlat`, batched.
    func encodeGather(_ enc: MTLComputeCommandEncoder, flat: MTLBuffer) {
        guard let table = tubeTable, let owner = tubeOf else { return }
        var total = UInt32(totalParticles)
        enc.setComputePipelineState(gather)
        enc.setBuffer(flat, offset: 0, index: 0)
        enc.setBuffer(table, offset: 0, index: 1)
        enc.setBuffer(owner, offset: 0, index: 2)
        enc.setBytes(&total, length: 4, index: 3)
        dispatch1D(enc, gather, totalParticles)
    }

    /// The flat buffer → every tube's own buffer (stamp cleared). `pbdCopyFromFlat`, batched.
    func encodeScatter(_ enc: MTLComputeCommandEncoder, flat: MTLBuffer) {
        guard let table = tubeTable, let owner = tubeOf else { return }
        var total = UInt32(totalParticles)
        enc.setComputePipelineState(scatter)
        enc.setBuffer(flat, offset: 0, index: 0)
        enc.setBuffer(table, offset: 0, index: 1)
        enc.setBuffer(owner, offset: 0, index: 2)
        enc.setBytes(&total, length: 4, index: 3)
        dispatch1D(enc, scatter, totalParticles)
    }

    /// `pbdIntegrate` or `pbdFloorCollide` over the whole flat buffer with ONE uniform — the
    /// per-tube uniforms were identical by eligibility, so this is the per-tube pass.
    func encodeParticlePass(_ enc: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState,
                            flat: MTLBuffer, uniforms: PBDUniforms) {
        var u = uniforms
        u.particleCount = UInt32(totalParticles)
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(flat, offset: 0, index: 0)
        enc.setBuffer(flat, offset: 0, index: 1)   // the kernels' unused constraint slot
        enc.setBytes(&u, length: MemoryLayout<PBDUniforms>.stride, index: 2)
        dispatch1D(enc, pipeline, totalParticles)
    }

    /// Every tube's capsules + broadphase box, one thread per tube.
    func encodeBuildColliders(_ enc: MTLComputeCommandEncoder, flat: MTLBuffer,
                              colliders: MTLBuffer, radius: Float, selfRadius: Float) {
        guard let table = tubeTable, let box = bounds else { return }
        var u = BuildUniforms(tubeCount: UInt32(tubeCount), radius: radius,
                              reach: radius + selfRadius + Self.broadphaseMargin)
        enc.setComputePipelineState(build)
        enc.setBuffer(flat, offset: 0, index: 0)
        enc.setBuffer(colliders, offset: 0, index: 1)
        enc.setBuffer(table, offset: 0, index: 2)
        enc.setBuffer(box, offset: 0, index: 3)
        enc.setBytes(&u, length: MemoryLayout<BuildUniforms>.stride, index: 4)
        dispatch1D(enc, build, tubeCount)
    }

    /// SDF contact over every particle, with the per-tube broadphase.
    func encodeSDF(_ enc: MTLComputeCommandEncoder, flat: MTLBuffer, colliders: MTLBuffer,
                   colliderCount: Int, staticCount: Int, params: PBDTubeBatch.Params) {
        guard let table = tubeTable, let owner = tubeOf, let box = bounds, colliderCount > 0 else { return }
        var u = BroadUniforms(totalParticles: UInt32(totalParticles),
                              colliderCount: UInt32(colliderCount),
                              selfRadius: params.selfRadius,
                              collideStiffness: params.collideStiffness,
                              collideFriction: params.collideFriction,
                              tubeCount: UInt32(tubeCount),
                              staticCount: UInt32(staticCount),
                              countTests: countsTests ? 1 : 0)
        if countsTests { countedPasses += 1 }
        enc.setComputePipelineState(broad)
        enc.setBuffer(flat, offset: 0, index: 0)
        enc.setBuffer(colliders, offset: 0, index: 1)
        enc.setBytes(&u, length: MemoryLayout<BroadUniforms>.stride, index: 2)
        enc.setBuffer(table, offset: 0, index: 3)
        enc.setBuffer(owner, offset: 0, index: 4)
        enc.setBuffer(box, offset: 0, index: 5)
        enc.setBuffer(counters, offset: 0, index: 6)
        dispatch1D(enc, broad, totalParticles)
    }

    /// Tube expand for every tube in one dispatch, reading spines out of the flat buffer.
    /// Returns false when the set does not share one mesh (the caller expands per tube).
    func encodeExpand(_ enc: MTLComputeCommandEncoder, flat: MTLBuffer,
                      tubes: [PBDTubeRenderer]) -> Bool {
        guard let out = expandOutputs, let eu = expandUniforms, let vb = vertexBases,
              let table = tubeTable, tubes.count == tubeCount, expandRings > 0 else { return false }
        // Tints and dip state can change without a re-layout; refresh every substep.
        let up = eu.contents().bindMemory(to: TubeExpandUniforms.self, capacity: tubeCount)
        for (t, tube) in tubes.enumerated() { up[t] = tube.expandUniforms }
        enc.setComputePipelineState(expandBatch)
        enc.setBuffer(flat, offset: 0, index: 0)
        enc.setBuffer(out.position, offset: 0, index: 1)
        enc.setBuffer(eu, offset: 0, index: 2)
        enc.setBuffer(out.normal, offset: 0, index: 3)
        enc.setBuffer(out.color, offset: 0, index: 4)
        enc.setBuffer(table, offset: 0, index: 5)
        enc.setBuffer(vb, offset: 0, index: 6)
        let w = min(expandRings, expandBatch.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: expandRings, height: tubeCount, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        return true
    }

    // MARK: - Diagnostics

    func resetCounters() {
        memset(counters.contents(), 0, counters.length)
        countedPasses = 0
    }

    var counted: (narrow: Int, box: Int, passes: Int) {
        let p = counters.contents().bindMemory(to: UInt32.self, capacity: 2)
        return (Int(p[0]), Int(p[1]), countedPasses)
    }
}
