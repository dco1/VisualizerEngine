import Metal
import OSLog
import simd

/// Shared fixed-step physics dispatcher for a set of `PBDTubeRenderer` objects.
///
/// Owns the flat particle buffer, collider buffer, per-substep constraint layout
/// cache, and the batched-encoder dispatch that was previously duplicated across
/// `HotdogDropPlusController` and `HotdogDropUltraController`. Both controllers
/// call `tick(tubes:dt:params:)` and do not hold any of these resources themselves.
///
/// **Thread safety**: must be called from the main actor (same requirement as the
/// controllers that own it). The completed-handler only calls `DispatchSemaphore
/// .signal()`, which is safe from the GPU-completion queue.
@MainActor
public final class PBDTubeBatch {

    // ── Swift mirror of the Metal PBDSDFBatchUniforms struct ──────────────────
    // 4 × 4-byte fields, 16 bytes total, no padding. Matches PBD.metal exactly.
    private struct SDFBatchUniforms {
        var totalParticles:   UInt32
        var colliderCount:    UInt32
        var selfRadius:       Float
        var collideStiffness: Float
        var collideFriction:  Float
    }

    // ── Per-tick physics params passed by the scene ───────────────────────────
    public struct Params {
        /// Gravitational acceleration (m/s²).
        public var gravity: Float
        /// Stiffness of inter-tube SDF collision response.
        public var collideStiffness: Float
        /// XPBD velocity post-solve restitution. < 1 activates velocity post-solve.
        public var velocityRestitution: Float
        /// Floor restitution (bounciness).
        public var floorRestitution: Float
        /// Capsule radius used as the self-collision margin in the batch SDF kernel.
        /// Normally equals `hotdogRadius` for the scene.
        public var selfRadius: Float
        /// Optional hard-floor plane (m). `nil` keeps the historical behaviour —
        /// the batch clamps every tube to a floor at `tube.radius` (the drop-onto-
        /// concrete scenes). A scene that wants franks to fall PAST y=0 into a void
        /// (Hot Dog Press) passes a value far below the void so the floor never
        /// catches a spilled frank; containment + the resting column come from the
        /// scene's static box colliders instead. HotdogDrop+/Ultra leave it nil →
        /// byte-identical to before.
        public var floorY: Float?
        /// 0…1 tangential-velocity RETENTION on contact (inter-tube AND box walls).
        /// 1.0 = frictionless (old behaviour); lower = stickier → a stuffed column
        /// JAMS and holds (rides up as a mass) instead of sliding/fluidising. Defaults
        /// to 0.70 so capsule-collider scenes (HotdogDrop+/Ultra) are unchanged.
        public var collideFriction: Float

        public init(gravity: Float,
                    collideStiffness: Float,
                    velocityRestitution: Float,
                    floorRestitution: Float,
                    selfRadius: Float,
                    floorY: Float? = nil,
                    collideFriction: Float = 0.70) {
            self.gravity            = gravity
            self.collideStiffness   = collideStiffness
            self.velocityRestitution = velocityRestitution
            self.floorRestitution   = floorRestitution
            self.selfRadius         = selfRadius
            self.floorY             = floorY
            self.collideFriction    = collideFriction
        }
    }

    // ── Fixed-step constants ──────────────────────────────────────────────────
    /// Physics substep duration (s). Both scene controllers use this value.
    public let fixedDt: Float = 1.0 / 120.0
    /// Accumulator cap = 8 substeps. Past this the batch drops physics debt
    /// rather than spiral into a permanent GPU-lag runaway.
    public let maxAccumulator: Float = 1.0 / 120.0 * 8

    // ── Shared state ──────────────────────────────────────────────────────────
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private var accumulator: Float = 0

    /// Back-pressure guard: max 3 in-flight command buffers. If the GPU hasn't
    /// drained the ring by the time the next substep is ready, drop the substep
    /// rather than stall the main thread on `makeCommandBuffer()`.
    private let inflightSemaphore = DispatchSemaphore(value: 3)

    // ── Flat particle buffer (GPU-only, sized for all tubes' particles) ───────
    private var flatBuffer:      MTLBuffer?
    private var boundaryBuffer:  MTLBuffer?   // [start: UInt32, count: UInt32] per tube
    private var sdfBatchUniBuf:  MTLBuffer?   // SDFBatchUniforms, written each substep
    private var flatDirty        = true

    // ── Shared SDF collider buffer ─────────────────────────────────────────────
    private var colliderBuffer:   MTLBuffer?
    private var colliderCapacity: Int = 0

    // ── Flat-batched constraint solve ──────────────────────────────────────────
    /// Scene opt-in. When true AND every tube is an open chain (ringCount == 0),
    /// the constraint + velocity phases run as ONE dispatch per colour group over
    /// a flat cross-tube constraint buffer instead of one dispatch PER TUBE per
    /// colour — collapsing the dominant ~38·N dispatches/substep to ~34. The math
    /// is identical (within a colour, constraints touch disjoint particles). Off
    /// by default so every existing scene is byte-for-byte unchanged. Set by the
    /// scene (HotdogDropUltra) before its first tick.
    public var flatSolveEnabled = false
    /// Flat constraint buffer: all tubes' constraints, GROUPED BY COLOUR then tube,
    /// with `.i`/`.j` remapped to global flat particle indices. Rebuilt on
    /// markDirty (spawn/cull/respawn changes rest lengths).
    private var flatConstraintBuffer:  MTLBuffer?
    private var flatLambdaBuffer:       MTLBuffer?
    private var flatConstraintUniform:  MTLBuffer?
    private var flatConstraintCapacity: Int = 0
    private var flatColourStart = [Int](repeating: 0, count: 6)
    private var flatColourCount = [Int](repeating: 0, count: 6)
    private var flatConstraintTotal = 0
    private var flatConstraintsDirty = true

    // ── Static scene colliders ─────────────────────────────────────────────────
    /// Fixed collider capsules from the scene (e.g. HotdogDropUltra's mustard-tub
    /// wall ring). Written into slots [0, count) of the shared collider buffer
    /// ahead of the per-tube capsules, with `ownerID = .max` recommended so every
    /// tube collides with them. Because `colliderBinding` exposes the whole
    /// buffer, fluid solvers bound to it see the statics too — one wall ring
    /// contains both the franks AND the mustard. Call `markDirty()` after changing.
    public var staticColliders: [PBDCollider] = [] {
        didSet { staticStaging = nil }   // re-stage on next tick
    }
    /// Shared-memory staging copy of `staticColliders`, blitted into the private
    /// collider buffer at the start of every substep CB (cheap — tens of entries).
    private var staticStaging: MTLBuffer?

    /// Optional extra constraint pass, run once per substep on the shared compute
    /// encoder AFTER the final floor pass and BEFORE tube-expand — so the mesh
    /// reflects whatever it corrects. The scene owns the pipeline + any buffers
    /// and dispatches per tube inside the closure (mirroring `dispatchFloor`).
    /// `nil` (the default) ⇒ byte-identical to the historical tick for every
    /// existing caller (HotdogDrop+/Ultra/Press). Used by the Hotdog Waterslide
    /// Descent for its inside-out swept-tube confinement (`pbdSweptTubeConfine`).
    public var extraConstraintPass: ((MTLComputeCommandEncoder, [PBDTubeRenderer]) -> Void)?

    // ── Constraint layout + collider-slot cache ────────────────────────────────
    // Rebuilt only on spawn/cull (when `markDirty()` is called). Reused across
    // every substep within the same tick to avoid heap allocations at 240 Hz.
    private var cachedLayouts:          [PBDSolver.ConstraintLayout] = []
    private var cachedColliderOffsets:  [Int] = []
    private var cachedColliderTotal:    Int = 0
    private var layoutsDirty           = true

    private let log = Logger(subsystem: "com.visualizer-engine", category: "PBDTubeBatch")

    // MARK: - Public accessors for dependent systems

    /// The Metal command queue owned by this batch. Fluid solvers (e.g.
    /// HotdogDropPlus's MLS-MPM mustard) that read the collider buffer
    /// after PBD substeps should submit on this queue; single-queue ordering
    /// guarantees the fluid CB sees the PBD writes without an explicit fence.
    public var commandQueue: MTLCommandQueue { queue }

    /// True when the flat-batched constraint/velocity solve is active for the
    /// current tube set (homogeneous open-chain topology). Exposed so a scene's
    /// perf HUD can report the dispatch-count collapse. Set each tick by `tick`.
    public private(set) var flatSolveActive = false

    /// The current shared SDF collider buffer + count, or nil if no substeps
    /// have run yet. Fluid solvers bind this to expose hot-dog capsule shapes.
    public var colliderBinding: PBDSolver.ColliderBinding? {
        guard let buf = colliderBuffer, cachedColliderTotal > 0 else { return nil }
        return PBDSolver.ColliderBinding(buffer: buf, count: cachedColliderTotal)
    }

    // MARK: - Init

    public enum Error: Swift.Error {
        case noCommandQueue
    }

    public init(device: MTLDevice) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else { throw Error.noCommandQueue }
        q.label = "PBDTubeBatch"
        self.queue = q
    }

    // MARK: - Dirty flag

    /// Mark layout + flat buffer dirty. Call immediately after adding or removing
    /// any tube from the live set so the next `tick` rebuilds cached offsets.
    public func markDirty() {
        layoutsDirty = true
        flatDirty    = true
        flatConstraintsDirty = true
    }

    // MARK: - Tick

    /// Encode one fixed-step physics tick for `tubes`.
    ///
    /// Advances the accumulator by `dt`, then drains it in `fixedDt` substeps
    /// (capped at 4 per tick). Each substep is one `MTLCommandBuffer` that
    /// encodes integrate → build-colliders → constraint-loop → velocity-postsolve
    /// → floor → tube-expand, all in a single `MTLComputeCommandEncoder` with
    /// explicit buffer barriers between phases.
    public func tick(tubes: [PBDTubeRenderer], dt: Double, params: Params) {
        guard !tubes.isEmpty else { return }

        accumulator += Float(dt)
        if accumulator > maxAccumulator { accumulator = maxAccumulator }

        rebuildFlatBufferIfDirty(tubes: tubes)

        var steps = 0
        while accumulator >= fixedDt && steps < 4 {
            if inflightSemaphore.wait(timeout: .now()) == .timedOut {
                accumulator -= fixedDt
                break
            }
            rebuildLayoutsIfDirty(tubes: tubes)
            guard cachedColliderTotal > 0,
                  let colBuf = colliderBuffer,
                  let cb     = queue.makeCommandBuffer()
            else {
                inflightSemaphore.signal()
                break
            }
            cb.label = "PBDTubeBatch[\(steps)]"

            // Broadcast per-substep settings to every solver.
            let binding = PBDSolver.ColliderBinding(buffer: colBuf, count: cachedColliderTotal)
            for tube in tubes {
                let s = tube.solver
                s.gravity             = params.gravity
                s.collideStiffness    = params.collideStiffness
                s.velocityRestitution = params.velocityRestitution
                s.floorRestitution    = params.floorRestitution
                // Default: clamp to a floor at radius above y=0 (drop-onto-concrete
                // scenes). A scene that wants franks to fall into a void overrides
                // params.floorY far below it so the plane never catches a spill.
                s.floorY              = params.floorY ?? tube.radius
                s.colliders           = binding
                s.writeIntegrateUniforms(dt: fixedDt)
            }

            let iters       = tubes[0].solver.constraintIterations
            let vrestActive = params.velocityRestitution < 1.0

            // ── Flat-solve eligibility ────────────────────────────────────────
            // Opt-in + every tube an OPEN chain (the flat constraint layout has no
            // ring-seam handling) + the batch pipelines/flat particle buffer ready.
            let batchPipelines = SimPipelineCache.shared.pbdPipelines(for: tubes[0].solver.device)
            let hasBatchFlat   = batchPipelines != nil && flatBuffer != nil
            let allOpenChains  = tubes.allSatisfy { $0.solver.ringCount == 0 }
            let flatRun = flatSolveEnabled && allOpenChains && hasBatchFlat
                && rebuildFlatConstraintsIfDirty(tubes: tubes)
            flatSolveActive = flatRun
            if flatRun, let ubuf = flatConstraintUniform {
                // Constraint + velocity kernels read only dt, ringCount(0),
                // constraintCount(≥ any colour count), velRestitution from this.
                let u = PBDUniforms(
                    dt: fixedDt, gravity: 0, damping: 0, floorY: 0,
                    particleCount: 0, constraintCount: UInt32(flatConstraintTotal),
                    velRestitution: params.velocityRestitution, ringCount: 0)
                ubuf.contents().bindMemory(to: PBDUniforms.self, capacity: 1).pointee = u
            }

            // Lambda reset can overlap with the first compute pass on hardware
            // that supports blit + compute concurrency; use a separate encoder.
            if let blit = cb.makeBlitCommandEncoder() {
                blit.label = "PBDTubeBatch.lambdaReset"
                if flatRun, let lbuf = flatLambdaBuffer, flatConstraintTotal > 0 {
                    // One fill over the whole flat lambda range replaces N per-tube
                    // fills (XPBD resets λ = 0 at the start of every substep).
                    blit.fill(buffer: lbuf, range: 0..<(flatConstraintTotal * MemoryLayout<Float>.stride), value: 0)
                } else {
                    for tube in tubes { tube.solver.dispatchLambdaReset(into: blit) }
                }
                // Static scene colliders → slots [0, count) of the private
                // collider buffer. Re-blitted every substep so buffer reallocs
                // (capacity growth) never leave stale/garbage static entries;
                // the copy is tens of 48-byte structs — noise next to the
                // constraint dispatches.
                if !staticColliders.isEmpty {
                    if staticStaging == nil {
                        let bytes = MemoryLayout<PBDCollider>.stride * staticColliders.count
                        if let stage = device.makeBuffer(length: bytes,
                                                         options: .storageModeShared) {
                            stage.label = "PBDTubeBatch.staticColliders.staging"
                            staticColliders.withUnsafeBytes { src in
                                stage.contents().copyMemory(
                                    from: src.baseAddress!, byteCount: bytes)
                            }
                            staticStaging = stage
                        }
                    }
                    if let stage = staticStaging {
                        blit.copy(from: stage, sourceOffset: 0,
                                  to: colBuf, destinationOffset: 0,
                                  size: MemoryLayout<PBDCollider>.stride * staticColliders.count)
                    }
                }
                blit.endEncoding()
            }

            // Single compute encoder for all phases: avoids ~1 440 encoder
            // open/close pairs per second at 12 dogs × 12 phases × 2 substeps × 60 Hz.
            guard let enc = cb.makeComputeCommandEncoder() else {
                inflightSemaphore.signal()
                break
            }
            enc.label = "PBDTubeBatch.compute[\(steps)]"

            // Phase 1: integrate (Verlet, writes particle positions).
            for tube in tubes { tube.solver.dispatchIntegrate(into: enc) }
            enc.memoryBarrier(scope: .buffers)

            // Phase 2: build per-tube SDF capsule colliders into the shared buffer.
            for (i, tube) in tubes.enumerated() {
                tube.dispatchBuildColliders(
                    into: enc,
                    target: colBuf,
                    targetOffset: cachedColliderOffsets[i],
                    ownerID: tube.solver.ownerID)
            }
            enc.memoryBarrier(scope: .buffers)

            if flatRun, let flat = flatBuffer, let p = batchPipelines,
               let solver0 = tubes.first?.solver {
                // ── FLAT PATH: particles stay in the flat buffer across the whole
                // iteration loop. Constraint + velocity colours are ONE dispatch
                // each (vs one PER TUBE). Identical op-sequence to the per-tube
                // path (P → constraints → SDF → … → velocity), just coarser
                // dispatches over disjoint-particle colour groups.
                copyToFlat(enc, tubes: tubes, flat: flat, pipelines: p)
                enc.memoryBarrier(scope: .buffers)
                for _ in 0..<iters {
                    for c in 0..<6 { runFlatConstraintColour(enc, solver: solver0, colour: c) }
                    dispatchBatchSDF(enc, tubes: tubes, pipelines: p, params: params)
                    enc.memoryBarrier(scope: .buffers)
                }
                if vrestActive {
                    for c in 0..<6 { runFlatVelocityColour(enc, solver: solver0, colour: c) }
                }
                copyFromFlat(enc, tubes: tubes, flat: flat, pipelines: p)
                enc.memoryBarrier(scope: .buffers)
            } else {
                // ── PER-TUBE PATH (unchanged; every non-opted-in scene) ─────────
                // Phase 3: constraint loop — each colour group is one batch dispatch.
                for _ in 0..<iters {
                    runConstraintColour(enc, tubes: tubes) { l in (l.primaryEvenStart, l.primaryEvenCount) }
                    runConstraintColour(enc, tubes: tubes) { l in (l.primaryOddStart,  l.primaryOddCount)  }
                    runConstraintColour(enc, tubes: tubes) { l in (l.bendAStart,       l.bendACount)       }
                    runConstraintColour(enc, tubes: tubes) { l in (l.bendBStart,       l.bendBCount)       }
                    runConstraintColour(enc, tubes: tubes) { l in (l.bendCStart,       l.bendCCount)       }
                    runConstraintColour(enc, tubes: tubes) { l in (l.longRangeStart,   l.longRangeCount)   }

                    if hasBatchFlat, let flat = flatBuffer, let p = batchPipelines {
                        copyToFlat(enc,   tubes: tubes, flat: flat, pipelines: p)
                        enc.memoryBarrier(scope: .buffers)
                        dispatchBatchSDF(enc, tubes: tubes, pipelines: p, params: params)
                        enc.memoryBarrier(scope: .buffers)
                        copyFromFlat(enc, tubes: tubes, flat: flat, pipelines: p)
                        enc.memoryBarrier(scope: .buffers)
                    } else {
                        // Fallback while the batch pipeline is still warming up.
                        for tube in tubes { tube.solver.dispatchSDFCollide(into: enc) }
                        enc.memoryBarrier(scope: .buffers)
                    }
                }

                // Phase 3a: XPBD velocity post-solve (same colour-group barriers).
                if vrestActive {
                    runVelocityColour(enc, tubes: tubes) { l in (l.primaryEvenStart, l.primaryEvenCount) }
                    runVelocityColour(enc, tubes: tubes) { l in (l.primaryOddStart,  l.primaryOddCount)  }
                    runVelocityColour(enc, tubes: tubes) { l in (l.bendAStart,       l.bendACount)       }
                    runVelocityColour(enc, tubes: tubes) { l in (l.bendBStart,       l.bendBCount)       }
                    runVelocityColour(enc, tubes: tubes) { l in (l.bendCStart,       l.bendCCount)       }
                    runVelocityColour(enc, tubes: tubes) { l in (l.longRangeStart,   l.longRangeCount)   }
                }
            }

            // Phase 3b: floor → post-floor SDF → final floor.
            for tube in tubes { tube.solver.dispatchFloor(into: enc) }
            enc.memoryBarrier(scope: .buffers)
            if hasBatchFlat, let flat = flatBuffer, let p = batchPipelines {
                copyToFlat(enc,   tubes: tubes, flat: flat, pipelines: p)
                enc.memoryBarrier(scope: .buffers)
                dispatchBatchSDF(enc, tubes: tubes, pipelines: p, params: params)
                enc.memoryBarrier(scope: .buffers)
                copyFromFlat(enc, tubes: tubes, flat: flat, pipelines: p)
                enc.memoryBarrier(scope: .buffers)
            } else {
                for tube in tubes { tube.solver.dispatchSDFCollide(into: enc) }
                enc.memoryBarrier(scope: .buffers)
            }
            for tube in tubes { tube.solver.dispatchFloor(into: enc) }
            enc.memoryBarrier(scope: .buffers)

            // Phase 3c: optional scene confinement pass (e.g. swept-tube interior).
            // nil for every existing caller → no-op. Runs after the floor so the
            // expand below meshes the confined positions.
            if let extraConstraintPass {
                extraConstraintPass(enc, tubes)
                enc.memoryBarrier(scope: .buffers)
            }

            // Phase 4: expand tube mesh — reads final positions, writes
            // the Illuminatorama-visible position + normal buffers.
            for tube in tubes { tube.dispatchTubeExpand(into: enc) }
            enc.endEncoding()

            let sema = inflightSemaphore
            cb.addCompletedHandler { cb in
                if let err = cb.error {
                    os_log(.error, "PBDTubeBatch GPU error: %{public}@",
                           err.localizedDescription)
                }
                sema.signal()
            }
            cb.commit()
            accumulator -= fixedDt
            steps += 1
        }
    }

    // MARK: - Private helpers

    @inline(__always)
    private func runConstraintColour(
        _ enc: MTLComputeCommandEncoder,
        tubes: [PBDTubeRenderer],
        _ slice: (PBDSolver.ConstraintLayout) -> (start: Int, count: Int)
    ) {
        for (i, tube) in tubes.enumerated() {
            let s = slice(cachedLayouts[i])
            tube.solver.dispatchConstraintSubpass(into: enc, start: s.start, count: s.count)
        }
        enc.memoryBarrier(scope: .buffers)
    }

    @inline(__always)
    private func runVelocityColour(
        _ enc: MTLComputeCommandEncoder,
        tubes: [PBDTubeRenderer],
        _ slice: (PBDSolver.ConstraintLayout) -> (start: Int, count: Int)
    ) {
        for (i, tube) in tubes.enumerated() {
            let s = slice(cachedLayouts[i])
            tube.solver.dispatchVelocitySubpass(into: enc, start: s.start, count: s.count)
        }
        enc.memoryBarrier(scope: .buffers)
    }

    private func rebuildLayoutsIfDirty(tubes: [PBDTubeRenderer]) {
        guard layoutsDirty else { return }
        layoutsDirty = false

        cachedColliderOffsets.removeAll(keepingCapacity: true)
        cachedColliderOffsets.reserveCapacity(tubes.count)
        // Per-tube capsules start AFTER the static scene colliders.
        var total = staticColliders.count
        for tube in tubes {
            let n = tube.solver.particleBuffer.count
            cachedColliderOffsets.append(total)
            if n >= 2 { total += n - 1 }
        }
        cachedColliderTotal = total

        cachedLayouts.removeAll(keepingCapacity: true)
        cachedLayouts.reserveCapacity(tubes.count)
        for tube in tubes { cachedLayouts.append(tube.solver.constraintLayout) }

        if total > 0,
           colliderBuffer == nil || colliderCapacity < total {
            let newCap  = max(total + 32, colliderCapacity * 2, 256)
            let byteLen = MemoryLayout<PBDCollider>.stride * newCap
            if let buf  = device.makeBuffer(length: byteLen, options: .storageModePrivate) {
                buf.label       = "PBDTubeBatch.colliders"
                colliderBuffer  = buf
                colliderCapacity = newCap
            }
        }
    }

    private func rebuildFlatBufferIfDirty(tubes: [PBDTubeRenderer]) {
        guard flatDirty, !tubes.isEmpty else {
            if tubes.isEmpty { flatDirty = false }
            return
        }
        flatDirty = false

        let totalParticles = tubes.reduce(0) { $0 + $1.solver.particleBuffer.count }
        guard totalParticles > 0 else { return }

        flatBuffer = device.makeBuffer(
            length: totalParticles * MemoryLayout<PBDParticle>.stride,
            options: .storageModePrivate)
        flatBuffer?.label = "PBDTubeBatch.flat"

        let boundaryStride = 2 * MemoryLayout<UInt32>.size
        guard let bb = device.makeBuffer(
            length: tubes.count * boundaryStride,
            options: .storageModeShared) else { return }
        var flatOff = 0
        let ptr = bb.contents().bindMemory(to: UInt32.self, capacity: tubes.count * 2)
        for (i, tube) in tubes.enumerated() {
            ptr[i * 2]     = UInt32(flatOff)
            ptr[i * 2 + 1] = UInt32(tube.solver.particleBuffer.count)
            flatOff += tube.solver.particleBuffer.count
        }
        boundaryBuffer       = bb
        boundaryBuffer?.label = "PBDTubeBatch.boundaries"

        sdfBatchUniBuf = device.makeBuffer(
            length: MemoryLayout<SDFBatchUniforms>.size,
            options: .storageModeShared)
        sdfBatchUniBuf?.label = "PBDTubeBatch.sdfBatchUnis"
    }

    /// Colour-group slices of a tube's `ConstraintLayout`, in the SAME order the
    /// per-tube path dispatches them (so the flat path is order-equivalent).
    private static let colourSlices: [(PBDSolver.ConstraintLayout) -> (Int, Int)] = [
        { ($0.primaryEvenStart, $0.primaryEvenCount) },
        { ($0.primaryOddStart,  $0.primaryOddCount)  },
        { ($0.bendAStart,       $0.bendACount)       },
        { ($0.bendBStart,       $0.bendBCount)       },
        { ($0.bendCStart,       $0.bendCCount)       },
        { ($0.longRangeStart,   $0.longRangeCount)   },
    ]

    /// Rebuild the flat cross-tube constraint buffer (grouped by colour → tube),
    /// remapping each tube's local particle indices to global flat indices. Only
    /// runs on markDirty (spawn/cull/respawn) — never inside the substep loop.
    /// Returns false if it couldn't build (caller falls back to the per-tube path).
    @discardableResult
    private func rebuildFlatConstraintsIfDirty(tubes: [PBDTubeRenderer]) -> Bool {
        guard flatConstraintsDirty else { return flatConstraintTotal > 0 }
        flatConstraintsDirty = false
        flatConstraintTotal = 0

        // Per-tube flat particle base = cumulative particle count.
        var bases = [Int](repeating: 0, count: tubes.count)
        var pacc = 0
        for (t, tube) in tubes.enumerated() { bases[t] = pacc; pacc += tube.solver.particleBuffer.count }

        // Total constraints across all tubes.
        var total = 0
        for tube in tubes { total += tube.solver.constraintBuffer.count }
        guard total > 0 else { return false }

        // (Re)allocate the flat constraint + lambda + uniform buffers.
        if flatConstraintBuffer == nil || flatConstraintCapacity < total {
            let newCap = max(total + 64, flatConstraintCapacity * 2, 256)
            flatConstraintBuffer = device.makeBuffer(
                length: newCap * MemoryLayout<PBDConstraint>.stride, options: .storageModeShared)
            flatConstraintBuffer?.label = "PBDTubeBatch.flatConstraints"
            flatLambdaBuffer = device.makeBuffer(
                length: newCap * MemoryLayout<Float>.stride, options: .storageModePrivate)
            flatLambdaBuffer?.label = "PBDTubeBatch.flatLambda"
            flatConstraintCapacity = newCap
        }
        if flatConstraintUniform == nil {
            flatConstraintUniform = device.makeBuffer(
                length: MemoryLayout<PBDUniforms>.stride, options: .storageModeShared)
            flatConstraintUniform?.label = "PBDTubeBatch.flatConstraintUni"
        }
        guard let cbuf = flatConstraintBuffer else { return false }
        let dst = cbuf.contents().bindMemory(to: PBDConstraint.self, capacity: flatConstraintCapacity)

        // Pack grouped by colour, then tube — each colour becomes one contiguous,
        // independently-dispatchable slice.
        var cursor = 0
        for c in 0..<6 {
            flatColourStart[c] = cursor
            let slice = Self.colourSlices[c]
            for (t, tube) in tubes.enumerated() {
                let layout = tube.solver.constraintLayout
                let (s, cnt) = slice(layout)
                guard cnt > 0 else { continue }
                let src = tube.solver.constraintBuffer.contents
                let base = UInt32(bases[t])
                for k in 0..<cnt {
                    var con = src[s + k]
                    con.i &+= base
                    con.j &+= base
                    dst[cursor] = con
                    cursor += 1
                }
            }
            flatColourCount[c] = cursor - flatColourStart[c]
        }
        flatConstraintTotal = cursor
        return cursor > 0
    }

    /// One constraint colour group as a SINGLE flat dispatch across all tubes.
    @inline(__always)
    private func runFlatConstraintColour(_ enc: MTLComputeCommandEncoder,
                                         solver: PBDSolver, colour: Int) {
        let count = flatColourCount[colour]
        guard count > 0,
              let cbuf = flatConstraintBuffer, let lbuf = flatLambdaBuffer,
              let ubuf = flatConstraintUniform, let pbuf = flatBuffer else { return }
        solver.dispatchConstraintFlat(
            into: enc, particles: pbuf,
            constraints: cbuf, constraintByteOffset: flatColourStart[colour] * MemoryLayout<PBDConstraint>.stride,
            lambda: lbuf, lambdaByteOffset: flatColourStart[colour] * MemoryLayout<Float>.stride,
            uniform: ubuf, count: count)
        enc.memoryBarrier(scope: .buffers)
    }

    /// One velocity colour group as a SINGLE flat dispatch across all tubes.
    @inline(__always)
    private func runFlatVelocityColour(_ enc: MTLComputeCommandEncoder,
                                       solver: PBDSolver, colour: Int) {
        let count = flatColourCount[colour]
        guard count > 0,
              let cbuf = flatConstraintBuffer, let ubuf = flatConstraintUniform,
              let pbuf = flatBuffer else { return }
        solver.dispatchVelocityFlat(
            into: enc, particles: pbuf,
            constraints: cbuf, constraintByteOffset: flatColourStart[colour] * MemoryLayout<PBDConstraint>.stride,
            uniform: ubuf, count: count)
        enc.memoryBarrier(scope: .buffers)
    }

    private func copyToFlat(_ enc: MTLComputeCommandEncoder,
                            tubes: [PBDTubeRenderer],
                            flat: MTLBuffer,
                            pipelines: SimPipelineCache.PBDPipelines) {
        var off = 0
        for tube in tubes {
            tube.solver.dispatchCopyToFlat(into: enc, flatBuffer: flat,
                                           offset: off, pipelines: pipelines)
            off += tube.solver.particleBuffer.count
        }
    }

    private func copyFromFlat(_ enc: MTLComputeCommandEncoder,
                              tubes: [PBDTubeRenderer],
                              flat: MTLBuffer,
                              pipelines: SimPipelineCache.PBDPipelines) {
        var off = 0
        for tube in tubes {
            tube.solver.dispatchCopyFromFlat(into: enc, flatBuffer: flat,
                                             offset: off, pipelines: pipelines)
            off += tube.solver.particleBuffer.count
        }
    }

    private func dispatchBatchSDF(_ enc: MTLComputeCommandEncoder,
                                  tubes: [PBDTubeRenderer],
                                  pipelines: SimPipelineCache.PBDPipelines,
                                  params: Params) {
        guard let flat    = flatBuffer,
              let bounds  = boundaryBuffer,
              let uniBuf  = sdfBatchUniBuf,
              let colBuf  = colliderBuffer,
              cachedColliderTotal > 0,
              !tubes.isEmpty
        else { return }

        let totalParticles = tubes.reduce(0) { $0 + $1.solver.particleBuffer.count }
        guard totalParticles > 0 else { return }

        let uPtr = uniBuf.contents().bindMemory(to: SDFBatchUniforms.self, capacity: 1)
        uPtr.pointee = SDFBatchUniforms(
            totalParticles:   UInt32(totalParticles),
            colliderCount:    UInt32(cachedColliderTotal),
            selfRadius:       params.selfRadius,
            collideStiffness: params.collideStiffness,
            collideFriction:  params.collideFriction
        )
        var tubeCount = UInt32(tubes.count)
        enc.setComputePipelineState(pipelines.sdfBatch)
        enc.setBuffer(flat,   offset: 0, index: 0)
        enc.setBuffer(colBuf, offset: 0, index: 1)
        enc.setBuffer(uniBuf, offset: 0, index: 2)
        enc.setBuffer(bounds, offset: 0, index: 3)
        enc.setBytes(&tubeCount, length: 4, index: 4)
        let w = min(totalParticles, pipelines.sdfBatch.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: totalParticles, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }
}
