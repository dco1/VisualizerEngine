import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── PBD FIELD SOLVER ─────────────────────────────────────────────────────────
//
// XPBD for *many* short identical chains. The single-chain solver in
// PBDSolver.swift allocates one MTLBuffer and one set of compute encoders
// per body; that's fine for ≤ a dozen hot dogs but collapses at the scales
// grass / fur / hair require (10⁴ blades = 10⁴ buffer allocations × ~6
// encoder dispatches per substep × 10 substeps/frame ≈ 600 000 encoder
// operations per second on the GPU command queue — Metal will not be
// amused).
//
// PBDFieldSolver collapses N chains into ONE big particle buffer + ONE big
// constraint buffer + ONE shared rest-position buffer. Every kernel
// dispatch is global-over-all-particles or global-over-all-constraints, so
// adding the 10001st blade adds ~50 bytes of memory and zero encoder cost.
// This is the "future perf tier" described in PBDSolver.swift's header,
// realised as a sibling instead of a refactor so the existing single-body
// PBD code stays untouched (and so the field case can drop SDF / floor /
// velocity-solve baggage it doesn't need).
//
// CONSTRAINT TOPOLOGY (identical across every chain)
//
// Per chain with `particlesPerChain = M`:
//   • Primary even pairs:  (0,1),(2,3),…        — ⌈(M−1)/2⌉ constraints
//   • Primary odd pairs:   (1,2),(3,4),…        — ⌊(M−1)/2⌋ constraints
//   • Skip-one bending:    (0,2),(1,3),(2,4),…  — split into 3 colours by i%3
//
// All-of-colour-X across all chains forms one disjoint-particle dispatch.
// We lay the constraint buffer out as concatenated colour groups exactly
// like the single-chain solver, so the same `pbdConstraint` kernel from
// PBD.metal handles each subpass with no modification.
//
// PERF TARGETS (M3 Max, debug build)
//   • 5 000 blades × 6 particles  = 30 000 particles
//   • 5 000 × 9 constraints       = 45 000 constraints
//   • 10 substeps × 6 sub-passes  = 60 dispatches/frame
//   • Total dispatches/frame      = 60 (vs ~30 000 in the per-chain world)
//
// EXTENSIONS — not done yet
//   • Spatial-hash inter-blade collision. Grass doesn't need it; fur does.
//   • Per-particle ownerID for SDF colliders. Punted until something here
//     actually collides with a non-floor.
//   • Variable-topology chains. Today every chain has the same M; a tail
//     for trees-with-stems would need either padding or a per-chain offset
//     table. YAGNI.

@MainActor
public final class PBDFieldSolver {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "PBDFieldSolver")

    public let engine: SimEngine
    public var device: MTLDevice { engine.device }

    // ── Pipelines (mostly shared with the single-chain solver) ─────────
    private let integratePipeline:   MTLComputePipelineState
    private let constraintPipeline:  MTLComputePipelineState
    private let floorPipeline:       MTLComputePipelineState
    private let fieldForcesPipeline: MTLComputePipelineState

    // ── Particle / constraint / rest storage ───────────────────────────
    /// Particle buffer holding all chains end-to-end. Chain `b`'s particles
    /// occupy `[b * M, (b+1) * M)`. The same index layout is used by every
    /// downstream consumer (renderer, rest-position lookup).
    public let particleBuffer:   SimBuffer<PBDParticle>
    /// Constraint buffer concatenated as
    /// `[even primary | odd primary | bend A | bend B | bend C]`. Indices
    /// inside each constraint are GLOBAL into `particleBuffer`.
    public let constraintBuffer: SimBuffer<PBDConstraint>
    /// One float4 per particle — xyz = "where this particle wants to be"
    /// position, sampled by `pbdFieldForces` for the restoring spring; w
    /// reserved (currently 0). Must be SIMD4<Float> ↔ float4: Swift's
    /// `SIMD3<Float>` stride is 16 bytes (not 12), so pairing it with a
    /// kernel-side `packed_float3` would tear the lookup for every
    /// particle past id 0 — symptom is a static uniform tilt no slider
    /// can recover. See the ALIGNMENT RULE in PBDSolver.swift.
    public let restBuffer:       SimBuffer<SIMD4<Float>>

    private let lambdaBuffer: MTLBuffer
    private let uniformBuffer: MTLBuffer  // PBDUniforms (shared kernels)
    private let fieldUniformBuffer: MTLBuffer

    // ── Chain layout ───────────────────────────────────────────────────
    public private(set) var chainCount: Int = 0
    public private(set) var particlesPerChain: Int = 0
    public private(set) var primaryEvenCount: Int = 0    // across ALL chains
    public private(set) var primaryConstraintCount: Int = 0
    public private(set) var bendACount: Int = 0
    public private(set) var bendBCount: Int = 0
    public private(set) var bendCCount: Int = 0

    // ── Tunables (Swift-side; pushed into uniforms each substep) ────────
    /// Gravity in m/s². Defaults to 0 — pure gravity pulls grass blades
    /// flat to the floor and they never recover. The restoring-spring +
    /// wind in `pbdFieldForces` is what shapes the motion. Leave 0 for
    /// grass; bump up for kelp or anything whose default state is droopy.
    public var gravity: Float = 0
    public var damping: Float = 0.985
    public var floorY:  Float = 0

    /// Wind acceleration peak (m/s²). 0 disables wind entirely. The actual
    /// per-particle force scales between ±windAmp by the procedural noise
    /// in PBDField.metal::windScalar — see that function for the formula.
    public var windAmp: Float = 0
    /// Spatial frequency of the wind noise (cycles per metre). Higher =
    /// gusts move in tighter patches.
    public var windFreq: Float = 0.4
    /// How fast the wind pattern drifts along (windDirX, windDirZ) in m/s.
    /// 0 = stationary noise (still varies in time but doesn't translate).
    public var windScroll: Float = 0.7
    public var windDir: SIMD2<Float> = SIMD2(1, 0)

    /// Restoring-spring stiffness (1/s²). Acceleration toward rest
    /// position = springK · (rest − pos). Keep it small relative to wind
    /// amplitude or blades become rigid sticks.
    public var springK: Float = 8.0
    /// Per-substep velocity damping in [0, 1] — additional viscous loss
    /// on top of the `damping` multiplier in pbdIntegrate. 0 = none.
    public var springDamp: Float = 0.02

    public var constraintIterations: Int = 4

    /// Fixed substep dt. Field-scale grass simulates fine at 1/120 s.
    private let fixedDt: Float = 1.0 / 120.0
    public var accumulator: Float = 0.0

    // Wall-clock time used as the wind noise seed. Advanced inside
    // `encode(to:wallDt:)` so wind animates whether the substep loop runs
    // 0, 1, or 4 times this frame.
    private var time: Float = 0.0

    // ── Init ───────────────────────────────────────────────────────────

    public init?(engine: SimEngine,
                 maxChains: Int,
                 particlesPerChain: Int) {
        // Reuse the existing PBD pipelines for integrate / constraint /
        // floor — they're already global-over-id and don't care that the
        // buffer holds many chains instead of one.
        guard let pbd = engine.pipelineCache.pbdPipelines(for: engine.device),
              let forces = engine.pipelineCache.pipelineState(
                  name: "pbdFieldForces", device: engine.device)
        else {
            Self.log.error("PBDField pipeline cache failed — check PBDField.metal is in VisualizerRendering/Shaders/")
            return nil
        }

        let device = engine.device

        // Per-chain constraint count is computed from topology — see the
        // header docstring. Capacity covers the max chain count.
        let maxParticles = maxChains * particlesPerChain
        let primaryPerChain = max(0, particlesPerChain - 1)
        let bendPerChain    = max(0, particlesPerChain - 2)
        let perChain = primaryPerChain + bendPerChain
        let maxConstraints = maxChains * perChain

        let lambdaBytes = MemoryLayout<Float>.stride * max(maxConstraints, 1)

        guard
            let pBuf = SimBuffer<PBDParticle>(device: device,
                                              capacity: maxParticles,
                                              label: "PBDField.particles"),
            let cBuf = SimBuffer<PBDConstraint>(device: device,
                                                capacity: maxConstraints,
                                                label: "PBDField.constraints"),
            let rBuf = SimBuffer<SIMD4<Float>>(device: device,
                                                capacity: maxParticles,
                                                label: "PBDField.rest"),
            let uBuf = device.makeBuffer(length: MemoryLayout<PBDUniforms>.stride,
                                         options: .storageModeShared),
            let fBuf = device.makeBuffer(length: MemoryLayout<PBDFieldForcesUniforms>.stride,
                                         options: .storageModeShared),
            let lBuf = device.makeBuffer(length: lambdaBytes,
                                         options: .storageModePrivate)
        else {
            Self.log.error("PBDField buffer allocation failed")
            return nil
        }
        uBuf.label = "PBDField.uniforms"
        fBuf.label = "PBDField.fieldUniforms"
        lBuf.label = "PBDField.lambda"

        self.engine = engine
        self.integratePipeline   = pbd.integrate
        self.constraintPipeline  = pbd.constraint
        self.floorPipeline       = pbd.floor
        self.fieldForcesPipeline = forces
        self.particleBuffer      = pBuf
        self.constraintBuffer    = cBuf
        self.restBuffer          = rBuf
        self.lambdaBuffer        = lBuf
        self.uniformBuffer       = uBuf
        self.fieldUniformBuffer  = fBuf
    }

    // ── Chain population ───────────────────────────────────────────────
    //
    // Caller hands us the rest pose of every chain — particle world
    // positions for chain b at indices [b * M, (b+1) * M). Particle 0 of
    // every chain is automatically pinned (invMass = 0); that matches grass
    // / fur / hair (root anchored to scalp / ground).

    /// Build all chains in one shot. `restPositions.count` must equal
    /// `chainCount * particlesPerChain`. Anchor mode = first-particle pin.
    public func configureChains(chainCount: Int,
                                particlesPerChain: Int,
                                restPositions: [SIMD3<Float>],
                                bendStiffness: Float = 0.4,
                                primaryCompliance: Float = 5e-5) {
        precondition(restPositions.count == chainCount * particlesPerChain,
                     "configureChains: rest array size \(restPositions.count) ≠ chainCount * particlesPerChain (\(chainCount) × \(particlesPerChain))")

        self.chainCount = chainCount
        self.particlesPerChain = particlesPerChain

        var particles = [PBDParticle]()
        particles.reserveCapacity(restPositions.count)
        for (i, pos) in restPositions.enumerated() {
            let isRoot = (i % particlesPerChain) == 0
            var p = PBDParticle(position: pos, invMass: isRoot ? 0 : 1)
            // prevPos initialised to pos so first-frame velocity = 0.
            p.prevPositionAndPad = SIMD4(pos, 0)
            particles.append(p)
        }
        particleBuffer.write(particles)
        restBuffer.write(restPositions.map { SIMD4($0, 0) })

        // ── Constraint generation ──────────────────────────────────────
        //
        // Pattern per chain (same as PBDSolver.configureChain) but indices
        // are GLOBAL across the merged particle buffer:
        //   even primary : (b*M + 0,  b*M + 1), (b*M + 2,  b*M + 3), …
        //   odd  primary : (b*M + 1,  b*M + 2), (b*M + 3,  b*M + 4), …
        //   bend A       : (b*M + 0,  b*M + 2), (b*M + 3,  b*M + 5), …  i%3 == 0
        //   bend B       : (b*M + 1,  b*M + 3), (b*M + 4,  b*M + 6), …  i%3 == 1
        //   bend C       : (b*M + 2,  b*M + 4), (b*M + 5,  b*M + 7), …  i%3 == 2
        //
        // Disjoint-particle within each colour group is preserved chain by
        // chain (the per-chain colouring is the same as the single-chain
        // solver) AND across chains (different chains' particles live in
        // disjoint slices of the buffer). So one global dispatch per colour
        // is race-free.
        let M = particlesPerChain
        let bendCompliance = PBDSolver.stiffnessToCompliance(bendStiffness)

        // Per-chain segment length. With height jitter (or any non-uniform
        // chain rest geometry) the chains have *different* segment lengths
        // even though they share the same particle count. Compute each
        // chain's segLen from its own rest positions — sampling one global
        // segLen for all chains would force every blade to a uniform
        // length, which the constraints then enforce, which yanks the tip
        // particles away from their actual rest positions and the spring
        // fights to pull them back. The visible symptom was that no
        // amount of slider tweaking would settle the blades to vertical.

        var evenC = [PBDConstraint](), oddC = [PBDConstraint]()
        var bendA = [PBDConstraint](), bendB = [PBDConstraint](), bendC = [PBDConstraint]()

        for b in 0..<chainCount {
            let base = UInt32(b * M)
            let segLen = (M >= 2)
                ? simd_length(restPositions[b * M + 1] - restPositions[b * M])
                : 0.1
            // Same colour-group topology as the single-chain solver, offset by
            // this chain's global base index. Generator lives in PBDSolver so
            // the two solvers can never disagree on the layout.
            let g = PBDSolver.openChainColourGroups(
                count: M, base: base, segLength: segLen,
                primaryCompliance: primaryCompliance, bendCompliance: bendCompliance)
            evenC += g.even
            oddC  += g.odd
            bendA += g.bendA
            bendB += g.bendB
            bendC += g.bendC
        }

        var all = [PBDConstraint]()
        all.reserveCapacity(evenC.count + oddC.count + bendA.count + bendB.count + bendC.count)
        all += evenC + oddC + bendA + bendB + bendC
        constraintBuffer.write(all)

        self.primaryEvenCount       = evenC.count
        self.primaryConstraintCount = evenC.count + oddC.count
        self.bendACount             = bendA.count
        self.bendBCount             = bendB.count
        self.bendCCount             = bendC.count
    }

    // ── Per-frame substep loop ─────────────────────────────────────────

    /// Accumulate wallDt and encode 0…4 fixed substeps into the given
    /// command buffer. The caller commits. Wind time advances by wallDt
    /// regardless of how many substeps run, so the noise field animates at
    /// real-time even when the substep budget caps out.
    public func encode(to cb: MTLCommandBuffer, wallDt: Float) {
        time += wallDt
        accumulator += wallDt
        var steps = 0
        while accumulator >= fixedDt && steps < 4 {
            encodeSubstep(to: cb, dt: fixedDt)
            accumulator -= fixedDt
            steps += 1
        }
    }

    private func encodeSubstep(to cb: MTLCommandBuffer, dt: Float) {
        writeUniforms(dt: dt)

        // Phase 1: integrate (gravity + damping + Verlet).
        encodePass(cb, pipeline: integratePipeline,
                   buffers: [particleBuffer.buffer, constraintBuffer.buffer, uniformBuffer],
                   count: particleBuffer.count, label: "PBDField.integrate")

        // Reset XPBD λ for this substep.
        if constraintBuffer.count > 0,
           let blit = cb.makeBlitCommandEncoder() {
            blit.label = "PBDField.lambdaReset"
            let bytes = MemoryLayout<Float>.stride * constraintBuffer.count
            blit.fill(buffer: lambdaBuffer, range: 0..<bytes, value: 0)
            blit.endEncoding()
        }

        // Phase 2: external forces (wind + restoring spring). Modifies
        // prev positions so the next implicit velocity reflects the
        // impulse. Runs AFTER integrate so it sees the freshly-integrated
        // position vs prev for the velocity inference.
        encodeFieldForces(cb, dt: dt)

        // Phase 3: constraint iterations (each iteration cycles all 5
        // colour groups in order, identical to the single-chain solver).
        let layout = constraintLayout
        for i in 0..<constraintIterations {
            encodeConstraintSubpass(cb, start: layout.evenStart,
                                    count: layout.evenCount, label: "PBDField.even[\(i)]")
            encodeConstraintSubpass(cb, start: layout.oddStart,
                                    count: layout.oddCount, label: "PBDField.odd[\(i)]")
            encodeConstraintSubpass(cb, start: layout.bendAStart,
                                    count: layout.bendACount, label: "PBDField.bendA[\(i)]")
            encodeConstraintSubpass(cb, start: layout.bendBStart,
                                    count: layout.bendBCount, label: "PBDField.bendB[\(i)]")
            encodeConstraintSubpass(cb, start: layout.bendCStart,
                                    count: layout.bendCCount, label: "PBDField.bendC[\(i)]")
        }

        // Phase 4: floor clamp. Cheap, prevents wind-driven blades from
        // sinking below their roots. Floor band 0 — we don't have piles to
        // worry about here.
        encodePass(cb, pipeline: floorPipeline,
                   buffers: [particleBuffer.buffer, constraintBuffer.buffer, uniformBuffer],
                   count: particleBuffer.count, label: "PBDField.floor")
    }

    private func writeUniforms(dt: Float) {
        let uPtr = uniformBuffer.contents().bindMemory(to: PBDUniforms.self, capacity: 1)
        uPtr.pointee = PBDUniforms(
            dt: dt, gravity: gravity, damping: damping, floorY: floorY,
            particleCount: UInt32(particleBuffer.count),
            constraintCount: UInt32(constraintBuffer.count),
            floorBand: 0,
            colliderCount: 0,
            ownerID: .max,
            collideStiffness: 0,
            collideFriction: 0,
            selfRadius: 0,
            velRestitution: 0,
            floorRestitution: 0
        )

        let fPtr = fieldUniformBuffer.contents().bindMemory(to: PBDFieldForcesUniforms.self,
                                                            capacity: 1)
        fPtr.pointee = PBDFieldForcesUniforms(
            particleCount: UInt32(particleBuffer.count),
            dt: dt,
            time: time,
            windAmp: windAmp,
            windFreq: windFreq,
            windScroll: windScroll,
            windDirX: windDir.x,
            windDirZ: windDir.y,
            springK: springK,
            springDamp: springDamp
        )
    }

    private func encodeFieldForces(_ cb: MTLCommandBuffer, dt: Float) {
        let count = particleBuffer.count
        guard count > 0,
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "PBDField.forces"
        enc.setComputePipelineState(fieldForcesPipeline)
        enc.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(restBuffer.buffer,     offset: 0, index: 1)
        enc.setBuffer(fieldUniformBuffer,    offset: 0, index: 2)
        let w = min(count, fieldForcesPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodePass(_ cb: MTLCommandBuffer,
                            pipeline: MTLComputePipelineState,
                            buffers: [MTLBuffer],
                            count: Int,
                            label: String) {
        guard count > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(pipeline)
        for (i, b) in buffers.enumerated() {
            enc.setBuffer(b, offset: 0, index: i)
        }
        let w = min(count, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeConstraintSubpass(_ cb: MTLCommandBuffer,
                                         start: Int, count: Int, label: String) {
        guard count > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(constraintPipeline)
        enc.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        let cByteOffset = start * MemoryLayout<PBDConstraint>.stride
        enc.setBuffer(constraintBuffer.buffer, offset: cByteOffset, index: 1)
        enc.setBuffer(uniformBuffer, offset: 0, index: 2)
        let lByteOffset = start * MemoryLayout<Float>.stride
        enc.setBuffer(lambdaBuffer, offset: lByteOffset, index: 3)
        let w = min(count, constraintPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    // ── Constraint layout (per-colour offsets into constraintBuffer) ────

    public struct ConstraintLayout {
        public let evenStart: Int,  evenCount: Int
        public let oddStart:  Int,  oddCount:  Int
        public let bendAStart: Int, bendACount: Int
        public let bendBStart: Int, bendBCount: Int
        public let bendCStart: Int, bendCCount: Int
    }

    public var constraintLayout: ConstraintLayout {
        let evenC  = primaryEvenCount
        let oddC   = primaryConstraintCount - primaryEvenCount
        let bAStart = primaryConstraintCount
        let bBStart = bAStart + bendACount
        let bCStart = bBStart + bendBCount
        return ConstraintLayout(
            evenStart: 0,           evenCount: evenC,
            oddStart:  evenC,       oddCount:  oddC,
            bendAStart: bAStart,    bendACount: bendACount,
            bendBStart: bBStart,    bendBCount: bendBCount,
            bendCStart: bCStart,    bendCCount: bendCCount
        )
    }
}

// ── Field-forces uniform mirror ──────────────────────────────────────────────
//
// Internal — the kernel signature lives in PBDField.metal. Keep field order
// IDENTICAL to that struct or grass will spasm weirdly.
struct PBDFieldForcesUniforms {
    var particleCount: UInt32
    var dt:           Float
    var time:         Float
    var windAmp:      Float
    var windFreq:     Float
    var windScroll:   Float
    var windDirX:     Float
    var windDirZ:     Float
    var springK:      Float
    var springDamp:   Float
}
