import Foundation
import Metal
import simd

// ── POURING INTO A CONTAINER ─────────────────────────────────────────────────
//
// `MLSMPMSolver.emitters` are nozzles: liquid leaves each at an exit velocity, a radius and a volume flow, and the solver does the
// rest — the stream falls, hits the container (`vessel`, or the domain box), splashes, slosh and settles. Everything runs on the GPU:
//
//   • A particle with mass 0 is DEAD. Every particle kernel skips it (one load), so a pool sized for the whole pour costs next to
//     nothing until it is used. `reservePool(totalParticles:)` appends dead particles up to a total.
//   • Each substep a nozzle is owed `flow · dt / particleVolume` particles, counted here with the fraction carried, so the volume that
//     has left a nozzle is the volume asked of it (to one particle) however the frame rate or the substeps fall. They are laid across the
//     nozzle's disc on a low-discrepancy sequence and through the substep's time, each carried on at the exit velocity from the moment
//     it left (`mlsEmit`): an even, continuous stream, not a burst a frame.
//   • `killBelowY` retires particles that have spilled out of the container's world; their slots go back to the nozzles.
//   • `maxExpectedSpeed` makes `encode` take as many substeps as keep every particle under `cflCells` cells a substep: a stream that
//     has fallen a third of a metre moves ~3 m/s, and a particle that skips a cell of the grid in one step misses its contact.
//
// What is poured, what has been dropped for want of a free slot, and what has been retired are counted on the GPU (`pourStats`).

extension MLSMPMSolver {

    /// A nozzle.
    public struct Emitter: Sendable {
        /// The nozzle's centre, sim coordinates (at the END of the frame it is set for: the nozzle may move between frames, and
        /// the particles of every substep leave from where it was then).
        public var position: SIMD3<Float>
        /// The velocity of the liquid as it leaves (world, m/s): the nozzle's own motion included.
        public var velocity: SIMD3<Float>
        /// The stream's radius at the nozzle (m); the disc it fills is perpendicular to `velocity`.
        public var radius: Float
        /// Volume poured a second (m³/s). 0 = shut.
        public var flowRate: Float
        /// Random speed variation, as a fraction of the speed (0 = a clean stream).
        public var velocityJitter: Float
        public init(position: SIMD3<Float>, velocity: SIMD3<Float>, radius: Float, flowRate: Float, velocityJitter: Float = 0) {
            self.position = position; self.velocity = velocity; self.radius = radius
            self.flowRate = flowRate; self.velocityJitter = velocityJitter
        }
        /// The flow that fills a stream of this radius and speed exactly (a plug of liquid: no gaps, no overlap).
        public static func fullFlow(radius: Float, speed: Float) -> Float { .pi * radius * radius * speed }
        /// The speed a stream of this radius must leave at to carry `flowRate` (the same plug of liquid, the other way round):
        /// choose how thick the stream is and how much it pours, and this is how fast.
        public static func fullSpeed(radius: Float, flowRate: Float) -> Float { flowRate / (.pi * max(radius * radius, 1e-10)) }
        /// A stream `radius` thick pouring `flowRate` along `direction`, as fast as that takes.
        public static func stream(from position: SIMD3<Float>, toward direction: SIMD3<Float> = SIMD3(0, -1, 0), radius: Float,
                                  flowRate: Float, velocityJitter: Float = 0) -> Emitter {
            Emitter(position: position, velocity: simd_normalize(direction) * fullSpeed(radius: radius, flowRate: flowRate),
                    radius: radius, flowRate: flowRate, velocityJitter: velocityJitter)
        }
    }

    /// What has happened to the pour so far (counted on the GPU, so up to a frame late).
    public struct PourStats: Sendable {
        /// Particles revived at nozzles.
        public var emitted: Int
        /// Particles a nozzle was owed but had no free slot for (the pool ran out: reserve more).
        public var dropped: Int
        /// Particles retired by the kill plane.
        public var killed: Int
        /// Live particles: seeded + emitted − killed.
        public var live: Int
    }

    /// Volume one particle stands for (m³): a cell's volume over the particles that fill a cell at rest.
    public var particleVolume: Float { cellSize * cellSize * cellSize / max(restDensity, 1e-4) }

    public var pourStats: PourStats {
        let s = pourStatsBuffer.contents().bindMemory(to: UInt32.self, capacity: 16)
        let e = Int(s[Int(PourState.emitted)]), k = Int(s[Int(PourState.killed)])
        return PourStats(emitted: e, dropped: Int(s[Int(PourState.dropped)]), killed: k, live: seededLive + e - k)
    }

    /// Volume of liquid emitted so far (m³).
    public var pouredVolume: Float { Float(pourStats.emitted) * particleVolume }

    /// Append dead particles until the buffer holds `totalParticles` (what is there already — a seeded fill — is kept), the pool the
    /// emitters draw on. Call before the first `encode`; set `restDensity` first (a pour into an empty container: `restDensity =
    /// particlesPerCellAxis³`, as `seedVessel` sets it).
    public func reservePool(totalParticles total: Int) {
        let have = particleBuffer.count
        let n = min(total, particleBuffer.capacity)
        guard n > have else { return }
        var all = [MLSParticle](UnsafeBufferPointer(start: particleBuffer.contents, count: have))
        seededLive = all.reduce(0) { $0 + ($1.positionMass.w > 0 ? 1 : 0) }
        let parked = (boundsMin + boundsMax) / 2
        all.append(contentsOf: (have..<n).map { _ in MLSParticle(position: parked, mass: 0) })
        particleBuffer.write(all)
    }

    /// Make every particle dead (the pool stays; the fill and what was poured are gone).
    public func killAll() {
        let parked = (boundsMin + boundsMax) / 2
        particleBuffer.write((0..<particleBuffer.count).map { _ in MLSParticle(position: parked, mass: 0) })
        seededLive = 0
        let s = pourStatsBuffer.contents().bindMemory(to: UInt32.self, capacity: 16)
        for i in 0..<16 { s[i] = 0 }
        pour?.carry = [Float](repeating: 0, count: 8)
        pour?.previous = Array(repeating: nil, count: PourState.maxEmitters)
    }

    /// Substeps this frame takes: `substepsPerFrame`, or more if `maxExpectedSpeed` needs them to keep to `cflCells`.
    public func effectiveSubsteps(wallDt: Float) -> Int {
        let base = max(1, substepsPerFrame)
        guard maxExpectedSpeed > 0, wallDt > 0 else { return base }
        // Two limits: no particle more than `cflCells` cells a substep, and (the explicit EOS) sound no more than 0.45.
        let flow = maxExpectedSpeed * wallDt / (cflCells * cellSize), sound = soundSpeed * wallDt / (0.45 * cellSize)
        return max(base, Int(max(flow, sound).rounded(.up)))
    }

    /// The speed of sound in the liquid as tuned, √(k·γ/ρ₀) (m/s). A splash needs it well above the speed the liquid moves at:
    /// below that the liquid is squeezed where it should be thrown aside.
    public var soundSpeed: Float { (bulkModulus * gamma / max(restDensity, 1e-4)).squareRoot() }

    /// The speed a particle reaches falling `height` metres from rest with `initial` m/s to start with (g as set).
    public func fallSpeed(height: Float, initial: Float = 0) -> Float {
        (initial * initial + 2 * gravity * max(height, 0)).squareRoot()
    }

    // MARK: Encoding (called from `encode`)

    func beginPourFrame() {
        guard !emitters.isEmpty else { return }
        if pour == nil { pour = PourState(engine: engine, capacity: particleBuffer.capacity) }
        pour?.startFrame(emitters)
    }

    func endPourFrame() { pour?.endFrame(emitters) }

    /// One substep's emission: `from…to` are how far through the frame it starts and ends (the nozzles move between the frame's
    /// starting and ending positions).
    func encodePourEmission(into cb: MTLCommandBuffer, dt: Float, from f0: Float, to f1: Float) {
        guard !emitters.isEmpty, let pour, particleBuffer.count > 0 else { return }
        let owed = pour.owed(emitters, dt: dt, particleVolume: particleVolume, from: f0, to: f1)
        guard owed.total > 0 else { return }
        guard let enc = beginPass(cb, "MLS.emit") else { return }
        var count = UInt32(particleBuffer.count)
        use(enc, pour.beginPipeline)
        bind(enc, pourStatsBuffer, 0)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        use(enc, pour.collectPipeline)
        bind(enc, particleBuffer.buffer, 0)
        bind(enc, pour.freeList, 1)
        bind(enc, pourStatsBuffer, 2)
        enc.setBytes(&count, length: 4, index: 3); boundBuffers[3] = nil
        let w = min(Int(count), pour.collectPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        use(enc, pour.emitPipeline)
        bind(enc, particleBuffer.buffer, 0)
        bind(enc, pour.freeList, 1)
        bind(enc, pourStatsBuffer, 2)
        var u = owed.uniforms
        enc.setBytes(&u, length: MemoryLayout<PourState.EmitUniforms>.stride, index: 3); boundBuffers[3] = nil
        let ew = min(owed.total, pour.emitPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: owed.total, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: ew, height: 1, depth: 1))
        endPass(enc)
    }
}

/// The pour's host-side state: pipelines, the free list, each nozzle's carry of a fraction of a particle and its serial number.
@MainActor
final class PourState {
    static let emitted: UInt32 = 1, dropped: UInt32 = 2, killed: UInt32 = 3
    static let maxEmitters = 8

    struct Nozzle {
        var posRadius: SIMD4<Float>, prevPos: SIMD4<Float>, velJitter: SIMD4<Float>, basisU: SIMD4<Float>, basisV: SIMD4<Float>
        var info: SIMD4<UInt32>
    }
    struct EmitUniforms {
        var meta: SIMD4<UInt32>
        var params: SIMD4<Float>
        var e: (Nozzle, Nozzle, Nozzle, Nozzle, Nozzle, Nozzle, Nozzle, Nozzle)
    }

    let beginPipeline: MTLComputePipelineState
    let collectPipeline: MTLComputePipelineState
    let emitPipeline: MTLComputePipelineState
    let freeList: MTLBuffer
    var carry = [Float](repeating: 0, count: maxEmitters)
    var serial = [UInt32](repeating: 0, count: maxEmitters)
    /// Each nozzle's position at the start of the frame (the last frame's end), nil until it has been seen.
    var previous: [SIMD3<Float>?] = Array(repeating: nil, count: maxEmitters)
    var frameStart: [SIMD3<Float>] = Array(repeating: .zero, count: maxEmitters)

    init?(engine: SimEngine, capacity: Int) {
        guard let b = engine.pipeline("mlsEmitBegin"), let c = engine.pipeline("mlsCollectFree"), let e = engine.pipeline("mlsEmit"),
              let f = engine.device.makeBuffer(length: 4 * max(capacity, 1), options: .storageModePrivate) else { return nil }
        f.label = "MLS.freeList"
        beginPipeline = b; collectPipeline = c; emitPipeline = e; freeList = f
    }

    func startFrame(_ emitters: [MLSMPMSolver.Emitter]) {
        for (i, e) in emitters.prefix(Self.maxEmitters).enumerated() { frameStart[i] = previous[i] ?? e.position }
    }
    func endFrame(_ emitters: [MLSMPMSolver.Emitter]) {
        for (i, e) in emitters.prefix(Self.maxEmitters).enumerated() { previous[i] = e.position }
    }

    /// How many particles each nozzle is owed this substep, and the uniforms that tell the kernel.
    func owed(_ emitters: [MLSMPMSolver.Emitter], dt: Float, particleVolume: Float, from f0: Float, to f1: Float)
        -> (total: Int, uniforms: EmitUniforms) {
        let z = Nozzle(posRadius: .zero, prevPos: .zero, velJitter: .zero, basisU: .zero, basisV: .zero, info: .zero)
        var list = [Nozzle](repeating: z, count: Self.maxEmitters)
        var total = 0, used = 0
        for (i, e) in emitters.prefix(Self.maxEmitters).enumerated() where e.flowRate > 0 && e.radius > 0 {
            carry[i] += e.flowRate * dt / max(particleVolume, 1e-12)
            let n = Int(carry[i])
            carry[i] -= Float(n)
            guard n > 0 else { continue }
            let speed = simd_length(e.velocity)
            let axis = speed > 1e-6 ? e.velocity / speed : SIMD3<Float>(0, -1, 0)
            let u = simd_normalize(simd_cross(axis, abs(axis.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)))
            let v = simd_cross(axis, u)
            let travel: SIMD3<Float> = e.position - frameStart[i]
            let a: SIMD3<Float> = frameStart[i] + travel * f0
            let b: SIMD3<Float> = frameStart[i] + travel * f1
            list[used] = Nozzle(posRadius: SIMD4(b, e.radius), prevPos: SIMD4(a, 0), velJitter: SIMD4(e.velocity, e.velocityJitter),
                                basisU: SIMD4(u, 0), basisV: SIMD4(v, 0), info: SIMD4(serial[i], UInt32(n), UInt32(total), 0))
            serial[i] &+= UInt32(n)
            total += n
            used += 1
        }
        let u = EmitUniforms(meta: SIMD4(UInt32(used), UInt32(total), 0, 0), params: SIMD4(dt, 0, 0, 0),
                             e: (list[0], list[1], list[2], list[3], list[4], list[5], list[6], list[7]))
        return (total, u)
    }
}
