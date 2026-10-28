import Metal
import simd

// ONE vocabulary — and ONE GPU gate — for "this simulation has come to rest", shared by every engine solver that can.
//
// Before this file each solver answered the question its own way or not at all: `CoinDEMSolver` had island sleep,
// `PaperClothSolver` had `stickSpeed` / `anchorBreak` (both OFF by default, so a draped towel crept at ~0.7 mm per
// frame forever), and `PBDSolver` / `PBDFieldSolver` / `ShapeMatchSoftBody` had nothing — a settled cord or grass
// patch simply kept being stepped. A caller wanting "simulate until it stops, then hold" had to invent the
// measurement and the freeze every time.
//
//   RestPolicy   WHAT counts as rest: a speed held for some simulated seconds, plus the static-friction stick speed.
//   SettleGate   the state that decides it. It lives in GPU memory: `simSettleReduce` finds the fastest particle,
//                `simSettleGate` folds that into "seconds spent under the speed" and the `frozen` word
//                (Shaders/SimSettle.metal). There is no CPU loop over particles and no completion handler; the CPU
//                sets the policy and reads one word on demand.
//   SimSettling  the contract a solver adopts: say where its records live (`settleSource`), map the policy onto its
//                own knobs (`applyRestPolicy`), return early from `encode` while frozen, and call
//                `encodeSettling` at the end of `encode`. `isSettled`, `freeze()`, `wake()`, `peakSpeed()` are free.
//
// ONE reduction kernel serves every record layout — PBDParticle (cloth, cords, grass), SMParticle (soft bodies),
// CoinBody (rigid) — because the caller passes the byte stride and field offsets. Frozen means "do not step":
// positions stay as they were and the last packed render data stays valid, so a settled scene costs nothing, as
// `CoinDEMSolver`'s whole-pile sleep already did. (The skip is the same one CoinDEM uses: the CPU reads the frozen
// word at the top of `encode`. That read is a load from shared memory, not a sync.)

/// What "at rest" means. Speeds are m/s of the solver's fastest particle (or awake body).
public struct RestPolicy: Equatable, Sendable {
    /// The fastest particle must stay under this …
    public var speed: Float
    /// … for this many SIMULATED seconds to count as settled.
    public var seconds: Float
    /// Static friction: below this tangential speed a particle resting on a collider or the floor does not slide at
    /// all (the stick regime a retention-rate friction never has). 0 = leave as set. Solvers without it ignore it.
    public var holdSpeed: Float
    /// Stop stepping when the gate trips. False measures and reports but never freezes, so a live scene that must
    /// stay able to react (the app canvas) can adopt the contract and choose its own moment to `freeze()`.
    public var autoFreeze: Bool

    public init(speed: Float, seconds: Float, holdSpeed: Float = 0, autoFreeze: Bool = true) {
        self.speed = speed; self.seconds = seconds; self.holdSpeed = holdSpeed; self.autoFreeze = autoFreeze
    }

    /// Cloth lying on something (a towel on a floor, a duvet on a bed): static friction on, hold until it is still.
    public static let cloth = RestPolicy(speed: 0.05, seconds: 0.25, holdSpeed: 0.08)
    /// A cord or tube chain: no contact-friction model to switch on, so measurement and the freeze only.
    public static let cord = RestPolicy(speed: 0.02, seconds: 0.4)
    /// Soft bodies and shape-matched clusters.
    public static let softBody = RestPolicy(speed: 0.03, seconds: 0.4)
    /// Rigid bodies. `CoinDEMSolver` also maps speed → `sleepLinVel` and seconds → `sleepFrames` for its per-island sleep.
    public static let rigid = RestPolicy(speed: 0.03, seconds: 0.5)
    /// Measure and report, never freeze, never hold. Assign it to watch a solver without stopping it. (An UNASSIGNED gate is
    /// off entirely: adopting the contract costs a solver nothing until a policy is set.)
    public static let observeOnly = RestPolicy(speed: 0.03, seconds: 0.4, holdSpeed: 0, autoFreeze: false)
}

/// Where a solver's records live, for the one reduction kernel.
public struct SettleSource {
    public var buffer: MTLBuffer
    /// Records to read. For a pool with free slots, the high-water mark.
    public var count: Int
    public var byteStride: Int
    /// Byte offset of the float4 position (its `w` is inverse mass: <= 0 is pinned or inactive and is skipped).
    public var posOffset: Int
    /// Byte offset of the float4 previous position — or, with `velocityMode`, of the float4 velocity.
    public var prevOffset: Int
    /// Read every n-th record (a field of 180 k blades does not need all of them to know it is moving).
    public var sampleStride: Int = 1
    /// `prevOffset` names a velocity, not a previous position: speed is its length, not a displacement over dt.
    public var velocityMode = false
    /// One uint per record; non-zero = asleep, skipped.
    public var asleep: MTLBuffer? = nil

    public init(buffer: MTLBuffer, count: Int, byteStride: Int, posOffset: Int, prevOffset: Int,
                sampleStride: Int = 1, velocityMode: Bool = false, asleep: MTLBuffer? = nil) {
        self.buffer = buffer; self.count = count; self.byteStride = byteStride
        self.posOffset = posOffset; self.prevOffset = prevOffset
        self.sampleStride = sampleStride; self.velocityMode = velocityMode; self.asleep = asleep
    }

    /// `PBDParticle` / `SMParticle` layout: float4 position, float4 previous, 32 bytes.
    public static func particles(_ buffer: MTLBuffer, count: Int, sampleStride: Int = 1) -> SettleSource {
        SettleSource(buffer: buffer, count: count, byteStride: 32, posOffset: 0, prevOffset: 16, sampleStride: sampleStride)
    }
}

/// The rest decision for one solver. State and kernels live on the GPU; this object holds the buffers and the policy.
@MainActor
public final class SettleGate {
    /// The policy in force. Until one is assigned the gate is OFF: nothing is encoded and a solver pays nothing. (Measured:
    /// even this two-dispatch pass costs ~1 ms of GPU time on a one-body world, so it must be opt-in, never a default tax.)
    public var policy: RestPolicy = .observeOnly { didSet { enabled = true } }
    /// False until a policy is assigned; an unset gate never encodes.
    public private(set) var enabled = false

    private struct ReduceUniforms {
        var count: UInt32, byteStride: UInt32, posOffset: UInt32, prevOffset: UInt32
        var sampleStride: UInt32, velocityMode: UInt32, useAsleep: UInt32
        var invDt: Float
    }
    private struct GateUniforms { var speed: Float, seconds: Float, autoFreeze: UInt32, elapsed: Float }

    /// uint[4]: [0] seconds under speed (float bits), [1] frozen, [2] last peak speed (float bits), [3] observations
    private var state: MTLBuffer?
    /// atomic uint[2]: [0] max speed bits, [1] a sample landed
    private var peak: MTLBuffer?
    private var emptyAsleep: MTLBuffer?
    private var reduce: MTLComputePipelineState?
    private var gate: MTLComputePipelineState?

    public init() {}

    private func prepare(device: MTLDevice) -> Bool {
        if state != nil { return reduce != nil && gate != nil }
        state = device.makeBuffer(length: 16, options: .storageModeShared)
        peak = device.makeBuffer(length: 8, options: .storageModeShared)
        emptyAsleep = device.makeBuffer(length: 4, options: .storageModeShared)
        reduce = SimPipelineCache.shared.pipelineState(name: "simSettleReduce", device: device)
        gate = SimPipelineCache.shared.pipelineState(name: "simSettleGate", device: device)
        if let s = state { memset(s.contents(), 0, 16) }
        if let p = peak { memset(p.contents(), 0, 8) }
        return reduce != nil && gate != nil && state != nil && peak != nil
    }

    private var words: UnsafeMutablePointer<UInt32>? { state?.contents().bindMemory(to: UInt32.self, capacity: 4) }

    /// True once the solver has stopped, by the gate or by `freeze()`. One word of shared memory, no sync.
    public var frozen: Bool { (words?[1] ?? 0) != 0 }
    /// Simulated seconds the peak speed has stayed under the policy speed, as of the last completed step.
    public var secondsBelow: Float { words.map { Float(bitPattern: $0[0]) } ?? 0 }
    /// Peak speed (m/s) at the last observation; NaN before the first.
    public var lastPeakSpeed: Float {
        guard let w = words, w[3] > 0 else { return .nan }
        return Float(bitPattern: w[2])
    }
    /// Observations folded in so far.
    public var observations: Int { words.map { Int($0[3]) } ?? 0 }
    /// True when the speed has stayed under the policy's for its time.
    public var isUnderSpeed: Bool { secondsBelow >= policy.seconds && observations > 0 }

    /// Stop stepping, keeping the current shape. Call while the GPU is idle for this solver (a store into shared memory).
    public func freeze() { words?[1] = 1 }
    /// Forget the count and resume stepping.
    public func wake() { if let w = words { w[0] = 0; w[1] = 0; w[2] = 0; w[3] = 0 } }

    /// Encode the reduction over `source` and the gate update, after the solver's own passes.
    /// `elapsed` is the simulated time those passes covered; `stepDt` the time of ONE step (for displacement → speed).
    func encode(_ cb: MTLCommandBuffer, source: SettleSource, elapsed: Float, stepDt: Float) {
        guard enabled, elapsed > 0, stepDt > 0, source.count > 0, prepare(device: cb.device),
              let reduce, let gate, let state, let peak, let e = cb.makeComputeCommandEncoder() else { return }
        e.label = "SimSettle"
        let stride = max(1, source.sampleStride)
        var ru = ReduceUniforms(count: UInt32(source.count), byteStride: UInt32(source.byteStride),
                                posOffset: UInt32(source.posOffset), prevOffset: UInt32(source.prevOffset),
                                sampleStride: UInt32(stride), velocityMode: source.velocityMode ? 1 : 0,
                                useAsleep: source.asleep == nil ? 0 : 1, invDt: 1 / stepDt)
        let n = (source.count + stride - 1) / stride
        e.setComputePipelineState(reduce)
        e.setBuffer(source.buffer, offset: 0, index: 0)
        e.setBuffer(peak, offset: 0, index: 1)
        e.setBytes(&ru, length: MemoryLayout<ReduceUniforms>.stride, index: 2)
        e.setBuffer(source.asleep ?? emptyAsleep, offset: 0, index: 3)
        let w = max(1, min(n, reduce.maxTotalThreadsPerThreadgroup))
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))

        var gu = GateUniforms(speed: policy.speed, seconds: policy.seconds, autoFreeze: policy.autoFreeze ? 1 : 0, elapsed: elapsed)
        e.setComputePipelineState(gate)
        e.setBuffer(state, offset: 0, index: 0)
        e.setBuffer(peak, offset: 0, index: 1)
        e.setBytes(&gu, length: MemoryLayout<GateUniforms>.stride, index: 2)
        e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        e.endEncoding()
    }
}

/// A solver that can come to rest. Conformers say where their records are and how the policy maps onto their own
/// knobs; the measurement, the gate and the surface are shared.
@MainActor
public protocol SimSettling: AnyObject {
    /// The gate this solver is stepped through. Declared `let settleGate = SettleGate()` by the conformer.
    var settleGate: SettleGate { get }
    /// Where the solver's particles (or bodies) live. Nil while it has none.
    func settleSource() -> SettleSource?
    /// Map a policy onto this solver's own knobs (cloth `stickSpeed`, rigid `sleepLinVel` / `sleepFrames`, …).
    func applyRestPolicy(_ policy: RestPolicy)
}

public extension SimSettling {
    /// The policy in force. Setting it also maps it onto the solver's own knobs.
    var restPolicy: RestPolicy {
        get { settleGate.policy }
        set { settleGate.policy = newValue; applyRestPolicy(newValue) }
    }
    /// True once the solver has stopped, or has stayed under the policy's speed for its time.
    var isSettled: Bool { settleGate.frozen || settleGate.isUnderSpeed }
    /// Fastest particle (or awake body) speed, m/s, at the last completed observation. NaN before the first.
    func peakSpeed() -> Float { settleGate.lastPeakSpeed }
    /// Stop stepping, keeping the current shape.
    func freeze() { settleGate.freeze() }
    /// Resume stepping and forget the settle count.
    func wake() { settleGate.wake() }
    /// Call at the top of `encode`: true means skip the step.
    var settleFrozen: Bool { settleGate.frozen }
    /// Call at the end of `encode`: measure and update the gate on the GPU. `elapsed` = simulated seconds this call covered,
    /// `stepDt` = the length of one step.
    func encodeSettling(_ cb: MTLCommandBuffer, elapsed: Float, stepDt: Float) {
        guard let src = settleSource() else { return }
        settleGate.encode(cb, source: src, elapsed: elapsed, stepDt: stepDt)
    }
}
