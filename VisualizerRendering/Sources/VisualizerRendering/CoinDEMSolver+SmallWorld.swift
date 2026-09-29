import Foundation
import Metal
import simd

// ── The SMALL-WORLD path (engine plan item 3c, stage B3) ───────────────────────────────────
//
// A constraint-path frame is ~40 dependent dispatches per substep plus ~5 per velocity
// iteration (CoinDEMSolver.encodeConstraintSubstep), and each one drains the GPU before the
// next starts. A small world — the Digital Clock's 47 bodies, ~12 contacts, a 9-joint crane —
// does perhaps a millisecond of arithmetic per frame under 5–9 ms of those bubbles. With
// `smallWorldPath` on, such a frame is ONE dispatch of one threadgroup (coinSmallWorldFrame,
// Shaders/CoinDEMSmallWorld.h) that runs every substep, the sleep update and the transforms
// through the same step functions the multi-dispatch kernels wrap. The result is the same
// simulation (CoinDEMSmallWorldTests compares the two bit for bit); only the dispatch count
// changes.
//
// The path is taken only while the world is small — `smallWorldMaxBodies` bodies (slots up to
// the high-water mark) and `smallWorldMaxContacts` contacts at the last completed frame —
// and falls back to the multi-dispatch path by itself when it outgrows them (or for the legacy
// solver, or `levelingEnabled`). One threadgroup is one GPU core: past the limits the per-colour
// dispatches of the multi-dispatch path spread a big world's solve across the whole GPU.

/// Mirrors `CdSWParams` in Shaders/CoinDEMSmallWorld.h exactly (224 bytes; asserted there and in
/// CoinDEMSmallWorldTests).
struct CoinSmallWorldParamsGPU {
    var steps: UInt32 = 0
    var velocityIterations: UInt32 = 0
    var colorScheme: UInt32 = 0
    var colorRounds: UInt32 = 0
    var warmStart: UInt32 = 0
    var clearHashFirst: UInt32 = 0
    var hashSize: UInt32 = 0
    var torsionOn: UInt32 = 0
    var maxContacts: UInt32 = 0
    var maxPolyPairs: UInt32 = 0
    var jointCount: UInt32 = 0
    var jointTGBodies: UInt32 = 0
    var jointPasses: UInt32 = 1
    var footCount: UInt32 = 0
    var footEveryIteration: UInt32 = 0
    var sleepEnabled: UInt32 = 0
    var sleepFrames: UInt32 = 0
    var islandUnionRounds: UInt32 = 0
    var phaseMask: UInt32 = 0xFFFF_FFFF
    var firstColor: UInt32 = 0          // always 0 (a runtime zero on purpose — see cdSWSolve)
    var offSorted: UInt32 = 0, offCellOffsets: UInt32 = 0, offBodyContacts: UInt32 = 0, offBodyContactCount: UInt32 = 0
    var offPriority: UInt32 = 0, offColor0: UInt32 = 0, offColor1: UInt32 = 0, offBid: UInt32 = 0
    var offColorCount: UInt32 = 0, offColorOffset: UInt32 = 0, offColorContacts: UInt32 = 0, offUncolSub: UInt32 = 0
    var offPolyPairs: UInt32 = 0, offIslandLabel: UInt32 = 0, offIslandMin: UInt32 = 0, offSleepHub: UInt32 = 0
    var offJointedBody: UInt32 = 0, offPolySurv: UInt32 = 0, pad2: UInt32 = 0, pad3: UInt32 = 0
    var footLater = CoinFootUniformsGPU(dt: 0, gravity: 0, globalMu: 0, footCount: 0, colliderCount: 0,
                                        bodyCount: 0, sleepEnabled: 0, lastIteration: 0)
    var footLast = CoinFootUniformsGPU(dt: 0, gravity: 0, globalMu: 0, footCount: 0, colliderCount: 0,
                                       bodyCount: 0, sleepEnabled: 0, lastIteration: 1)
}

/// Per-solver host state of the small-world path (`CoinDEMSolver.smallWorld`).
@MainActor
final class CoinSmallWorldState {
    /// TEST SWITCH for running the whole CoinDEM suite through the path: `VIZ_COINDEM_SMALLWORLD=1`
    /// makes it every solver's DEFAULT (with the usual limits); `=all` also lifts the limits, so
    /// EVERY constraint-path world — a 176-body pile, an 800-egg rain — runs the one-threadgroup
    /// kernel (slow for a big world; a correctness run, not a perf one). Unset or `=0`: off. A
    /// solver that sets `smallWorldPath` / the limits itself keeps its own choice.
    static let environmentMode: String = ProcessInfo.processInfo.environment["VIZ_COINDEM_SMALLWORLD"] ?? "0"
    static let defaultMaxBodies = 64
    static let defaultMaxContacts = 2048

    var enabled = CoinSmallWorldState.environmentMode == "1" || CoinSmallWorldState.environmentMode == "all"
    var maxBodies = CoinSmallWorldState.environmentMode == "all" ? Int.max : CoinSmallWorldState.defaultMaxBodies
    var maxContacts = CoinSmallWorldState.environmentMode == "all" ? Int.max : CoinSmallWorldState.defaultMaxContacts
    /// Threads in the one threadgroup (clamped to the pipeline's limit, 384 on Apple GPUs for
    /// this kernel — the register budget of its heaviest phase, contact generation).
    var threads = 256
    /// TEST SEAM (cost attribution): CD_SW_PH_* bits of phases to run; all in use.
    var phaseMask: UInt32 = 0xFFFF_FFFF
    /// TEST SEAM (cost attribution): when set, a small-world frame is handed to this closure
    /// instead of being encoded as one dispatch; it gets the frame's substep count and an encoder
    /// of (steps, phase mask, pass descriptor) dispatches — so a probe can time each phase in its
    /// own encoder. Never set in a scene.
    var profileHook: ((_ steps: Int, _ encode: (_ steps: Int, _ mask: UInt32, _ pass: MTLComputePassDescriptor?) -> Void) -> Void)?
    var lastFrameUsed = false
    var framesUsed = 0
    var arena: MTLBuffer?
    var layout = CoinSmallWorldParamsGPU()      // the arena offsets (the off* lanes)
    var dummy: MTLBuffer?
    /// `VIZ_COINDEM_SW_PROFILE` (diagnostic): the per-phase timer, made on the first frame when set.
    var profiler: CoinSmallWorldProfiler?
    var profilerTried = false
}

@MainActor
extension CoinDEMSolver {

    // ── Public knobs ─────────────────────────────────────────────────────────────

    /// OPT-IN (default off): run each frame of a SMALL world — at most `smallWorldMaxBodies`
    /// body slots and `smallWorldMaxContacts` contacts — as ONE single-threadgroup dispatch
    /// (see the file header). The simulation is the same (bit for bit on every world
    /// CoinDEMSmallWorldTests steps); only the GPU time changes. Constraint path only; a world
    /// that outgrows the limits takes the multi-dispatch path until it is small again.
    public var smallWorldPath: Bool {
        get { smallWorld.enabled }
        set { smallWorld.enabled = newValue }
    }

    /// Body slots (the high-water mark) up to which the small-world path is taken. Default 64.
    /// The kernel pairs bodies over a one-cell grid (every body against every higher one), so its
    /// cost grows as n² where the multi-dispatch grid's grows as n: measured on an M1 Max
    /// (CoinDEMSmallWorldTests.testSmallWorldCrossover, a mixed pile, sleep off) the small world
    /// takes 0.38× the multi-dispatch GPU time at 16 bodies, 0.49× at 48, 0.57× at 64, 0.85× at
    /// 128 and 1.20× at 192 — crossover ≈ 150. 64 keeps a margin (and one threadgroup is one GPU
    /// core for the whole frame) while holding the Digital Clock's 47-body world.
    public var smallWorldMaxBodies: Int {
        get { smallWorld.maxBodies }
        set { smallWorld.maxBodies = max(0, newValue) }
    }

    /// Contacts (at the last completed frame) up to which the small-world path is taken.
    /// Default 2048: one core's colouring and colour sweep scale with the contacts where the
    /// multi-dispatch path spreads them over the GPU. Measured at the 64-body limit (64 boxes on a
    /// slatted floor, CoinDEMSmallWorldTests.testSmallWorldCrossover): 0.62× the multi-dispatch
    /// GPU time at 256 contacts, 0.81× at 832, 0.88× at 1 664, 0.91× at 2 432 — still ahead, so
    /// 2048 guards the trend with a margin rather than marking a measured loss.
    public var smallWorldMaxContacts: Int {
        get { smallWorld.maxContacts }
        set { smallWorld.maxContacts = max(0, newValue) }
    }

    /// Whether the last `encode` ran the small-world kernel (for tests / instrumentation).
    public var lastFrameUsedSmallWorld: Bool { smallWorld.lastFrameUsed }

    /// Threads of the small-world threadgroup (tests / tuning; clamped to the pipeline).
    var smallWorldThreads: Int {
        get { smallWorld.threads }
        set { smallWorld.threads = max(32, newValue) }
    }

    // ── Hook (called from encode(to:wallDt:)) ────────────────────────────────────

    /// This frame can run as the small-world kernel.
    func smallWorldEligible(bodies: Int) -> Bool {
        guard smallWorld.enabled, solverMode == .constraint, smallWorldPipeline != nil, !levelingEnabled else { return false }
        return bodies <= smallWorld.maxBodies && contactCount <= smallWorld.maxContacts
    }

    /// Encode `steps` substeps + the sleep update + the transforms as one dispatch. The joint
    /// list for the frame is already uploaded (encode(to:wallDt:)), and the uniforms written.
    func encodeSmallWorldFrame(_ cb: MTLCommandBuffer, coinCount bound: Int, steps: Int) {
        if CoinSmallWorldProfiler.path != nil, !smallWorld.profilerTried {
            smallWorld.profilerTried = true
            smallWorld.profiler = CoinSmallWorldProfiler(device: device)
        }
        if let prof = smallWorld.profiler, smallWorld.profileHook == nil {
            prof.encodeFrame(cb, steps: steps, bodies: bound, contacts: contactCount) { n, mask, pass in
                self.encodeSmallWorldDispatch(cb, coinCount: bound, steps: n, phaseMask: mask, pass: pass)
            }
            smallWorld.framesUsed += 1
            return
        }
        if let hook = smallWorld.profileHook {
            hook(steps) { n, mask, pass in self.encodeSmallWorldDispatch(cb, coinCount: bound, steps: n, phaseMask: mask, pass: pass) }
            smallWorld.framesUsed += 1
            return
        }
        encodeSmallWorldDispatch(cb, coinCount: bound, steps: steps, phaseMask: smallWorld.phaseMask, pass: nil)
        smallWorld.framesUsed += 1
    }

    private func encodeSmallWorldDispatch(_ cb: MTLCommandBuffer, coinCount bound: Int, steps: Int,
                                          phaseMask: UInt32, pass: MTLComputePassDescriptor?) {
        guard let pso = smallWorldPipeline, let arena = smallWorldArena(), let dummy = smallWorldDummy() else { return }
        let polyLive = hasPolyBodies && !polyKernelDisabledForTesting
        if polyLive && polyPairCountBuffer == nil {
            polyPairCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
            polyPairCountBuffer?.label = "Coin.polyPairCount"
        }
        // The frame's uniforms over a ONE-CELL grid: every body in cell 0, so each generate
        // thread tests every higher-index body (see Shaders/CoinDEMSmallWorld.h).
        var u = makeUniforms(dt: fixedDt, coinCount: bound)
        u.gridMinX = 0; u.gridMinY = 0; u.gridMinZ = 0; u.invCell = 0
        u.gridResX = 1; u.gridResY = 1; u.gridResZ = 1

        var p = smallWorld.layout
        p.steps = UInt32(steps)
        p.velocityIterations = UInt32(max(0, velocityIterations))
        p.colorScheme = coloringScheme == .speculative ? 1 : 0
        p.colorRounds = UInt32(max(0, coloringScheme == .speculative ? speculativeColorRounds : colorRounds))
        p.warmStart = warmStart ? 1 : 0
        p.clearHashFirst = (warmStart && needsHashClear) ? 1 : 0
        p.hashSize = UInt32(hashSize)
        p.torsionOn = torsionPatchSlots.isEmpty ? 0 : 1
        p.maxContacts = UInt32(maxContacts)
        p.maxPolyPairs = polyLive ? UInt32(maxPolyPairs) : 0
        p.jointCount = UInt32(frameJointCount)
        p.jointTGBodies = UInt32(frameJointTGBodies)
        p.jointPasses = UInt32(max(1, jointInnerPasses))
        let foot = footSystem
        if let f = foot, f.anyActive {
            f.iterationCursor = 0
            f.iterateEveryIteration = f.anyOnDynamicSupport || torsionEnabled
            p.footCount = UInt32(f.highWater)
            p.footEveryIteration = f.iterateEveryIteration ? 1 : 0
            p.footLater = footUniforms(f, colliderCount: colliders.count, last: false)
            p.footLast = footUniforms(f, colliderCount: colliders.count, last: true)
        }
        p.sleepEnabled = sleepEnabled ? 1 : 0
        p.sleepFrames = sleepFrames
        p.islandUnionRounds = UInt32(max(0, islandUnionRounds))
        p.phaseMask = phaseMask

        guard let enc = pass.map({ cb.makeComputeCommandEncoder(descriptor: $0) }) ?? cb.makeComputeCommandEncoder() else { return }
        enc.label = "Coin.cs.smallWorldFrame"
        enc.setComputePipelineState(pso)
        enc.setBuffer(coinBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(biasBuffer, offset: 0, index: 1)
        enc.setBytes(&u, length: MemoryLayout<CoinUniforms>.stride, index: 2)
        enc.setBuffer(asleepBuffer, offset: 0, index: 3)
        enc.setBuffer(colliderBuffer.buffer, offset: 0, index: 4)
        enc.setBuffer(linkBuffer, offset: 0, index: 5)
        enc.setBuffer(contactBuffer, offset: 0, index: 6)
        enc.setBuffer(contactCountBuffer, offset: 0, index: 7)
        enc.setBuffer(hullVertexBuffer, offset: 0, index: 8)
        enc.setBuffer(hullRangeBuffer, offset: 0, index: 9)
        enc.setBuffer(jointBuffer, offset: 0, index: 10)
        enc.setBuffer(jointListBuffer, offset: 0, index: 11)
        enc.setBuffer(colorStatsBuffer, offset: 0, index: 12)
        enc.setBuffer(materialBuffer, offset: 0, index: 13)
        enc.setBuffer(patchRadiusBuffer, offset: 0, index: 14)
        enc.setBuffer(prevContactBuffer, offset: 0, index: 15)
        enc.setBuffer(pairHashBuffer, offset: 0, index: 16)
        enc.setBuffer(jointPrepBuffer ?? dummy, offset: 0, index: 17)
        enc.setBuffer(sleepTimerBuffer, offset: 0, index: 18)
        enc.setBuffer(sleepKeyBuffer, offset: 0, index: 19)
        enc.setBuffer(transformBuffer, offset: 0, index: 20)
        enc.setBuffer(foot?.configBuffer ?? dummy, offset: 0, index: 21)
        enc.setBuffer(foot?.stateBuffer ?? dummy, offset: 0, index: 22)
        enc.setBytes(&p, length: MemoryLayout<CoinSmallWorldParamsGPU>.stride, index: 23)
        enc.setBuffer(arena, offset: 0, index: 24)
        enc.setBuffer(polyLive ? (polyPairCountBuffer ?? dummy) : dummy, offset: 0, index: 25)
        enc.setBuffer(pairHashBuffer, offset: 0, index: 26)
        // The prepared contact rows (opt-in; read only while the uniforms carry CD_FLAG_PREPARED_CONTACTS),
        // else the dummy. Bound unconditionally: asking whether the library has the prepare kernels
        // RESOLVED coinPrepareContacts — a ≈ 53–97 ms cold backend compile on the main actor for every
        // small-world solver, prepared rows or not (stage-B3 verifier). An index a kernel does not
        // declare is simply unused.
        enc.setBuffer(contactPrepBufferForBinding ?? dummy, offset: 0, index: 27)
        // A whole number of SIMD groups: the polytope phase runs one pair per SIMD group
        // (cdPolyNarrowSGBody) and the joint solve its bias chain on the second group.
        let simd = max(1, pso.threadExecutionWidth)
        let tg = max(simd, (min(smallWorld.threads, pso.maxTotalThreadsPerThreadgroup) / simd) * simd)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        if warmStart { needsHashClear = false }
    }

    // ── Scratch ──────────────────────────────────────────────────────────────────

    /// The kernel's transient arrays in ONE private buffer (made on first use, so a solver that
    /// never takes the path never allocates it): per-body contact lists, the colouring's
    /// priorities / colours / bids, the colour buckets, the polytope pair list, the island
    /// labels — everything that lives within a substep or a frame. Persistent state (bodies,
    /// contacts, warm-start snapshot + hash, joint blocks, sleep timers / keys, feet) stays in
    /// the solver's own buffers, so a world can switch paths between any two frames.
    private func smallWorldArena() -> MTLBuffer? {
        if let a = smallWorld.arena { return a }
        var off = 0
        func take(_ bytes: Int) -> UInt32 {
            let o = off
            off = (off + max(bytes, 4) + 15) & ~15
            return UInt32(o)
        }
        var l = CoinSmallWorldParamsGPU()
        let c = maxCoins, k = maxContacts, u4 = MemoryLayout<UInt32>.stride
        l.offSorted = take(c * u4)
        l.offCellOffsets = take(2 * u4)
        l.offBodyContacts = take(c * Self.bodyContactCapacity * u4)
        l.offBodyContactCount = take(c * u4)
        l.offPriority = take(k * u4)
        l.offColor0 = take(k * u4)
        l.offColor1 = take(k * u4)
        l.offBid = take(k * u4)
        l.offColorCount = take((Self.maxColors + 1) * u4)
        l.offColorOffset = take((Self.maxColors + 3) * u4)
        l.offColorContacts = take(k * u4)
        l.offUncolSub = take(k * u4)
        l.offPolyPairs = take(maxPolyPairs * MemoryLayout<SIMD4<UInt32>>.stride)
        l.offIslandLabel = take(c * u4)
        l.offIslandMin = take(c * u4)
        l.offSleepHub = take(c * u4)
        l.offJointedBody = take(c * u4)
        l.offPolySurv = take(maxPolyPairs * MemoryLayout<SIMD4<UInt32>>.stride)   // the SAT pairs past their first face
        guard let b = device.makeBuffer(length: off, options: .storageModePrivate) else { return nil }
        b.label = "Coin.smallWorldArena"
        smallWorld.arena = b
        smallWorld.layout = l
        return b
    }

    /// Bound where the solver has no buffer (no joint blocks yet, no feet, no polytope list):
    /// the kernel never reads it then.
    private func smallWorldDummy() -> MTLBuffer? {
        if let d = smallWorld.dummy { return d }
        let d = device.makeBuffer(length: 1024, options: .storageModePrivate)   // ≥ one element of every pointee (a joint block: 496 B)
        d?.label = "Coin.smallWorldDummy"
        smallWorld.dummy = d
        return d
    }
}
