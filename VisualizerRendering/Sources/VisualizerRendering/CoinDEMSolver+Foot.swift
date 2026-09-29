import Foundation
import Metal
import simd

// ── CoinDEMSolver + Foot ──────────────────────────────────────────────────────
//
// plausibility: real — the host side of the SPRING-FOOT actuator (engine plan item 5b; kernels
// and the full physics write-up in Shaders/CoinDEMFoot.h). A foot is a massless compliant leg
// hinged at a hip on its body: aimed (cant + yaw) while in the air, pinned to what it lands on,
// pushing along tip → hip with an exactly integrated spring-damper whose reaction goes into
// the support, and turning its body in place through a yaw motor that real pad friction
// bounds. A hop is ½·k·Δ² of stored spring energy; nothing here sets a velocity.
//
// CALL RULE (as CoinDEMSolver+Actuation): every method is @MainActor and writes / reads shared
// buffers the GPU also uses — call them only while no command buffer that encodes this solver
// is in flight.
//
// COST. A solver with no foot, or whose feet are all RETRACTED, dispatches nothing for them
// (the retracted leg is inside the shell: it cannot touch anything). An active foot costs one
// dispatch per substep, one per velocity iteration and one per island-union round.
//
// ONE FOOT PER BODY. Feet are keyed by body slot. `clearAll()` / `despawn` do not know about
// feet: `removeFoot(body:)` first (a foot whose body slot is empty does nothing, but it would
// attach to the next body spawned into that slot).

// ── GPU mirrors (CoinDEMFoot.h; the Metal side static_asserts the sizes) ───────

struct CoinFootGPU {                 // 80 bytes
    var meta: SIMD4<UInt32>          // body, enabled, mode, armSeq
    var mount: SIMD4<Float>          // hip (body-local), extended rest length
    var axis: SIMD4<Float>           // leg direction (body-local, unit), retracted length
    var spring: SIMD4<Float>         // k, c, max force, pad μ (< 0 ⇒ the body's material)
    var spin: SIMD4<Float>           // yaw target rate, yaw max torque, μ_spin, patch radius
}

struct CoinFootStateGPU {            // 160 bytes
    var meta: SIMD4<UInt32>          // planted, support, collider index, firedSeq
    var tipL: SIMD4<Float>
    var nrmL: SIMD4<Float>
    var point: SIMD4<Float>
    var dir: SIMD4<Float>
    var imp: SIMD4<Float>
    var nrmW: SIMD4<Float>
    var row: SIMD4<Float>
    var total: SIMD4<Float>
    var stats: SIMD4<Float>
    static let zero = CoinFootStateGPU(meta: .zero, tipL: .zero, nrmL: .zero, point: .zero, dir: .zero,
                                       imp: .zero, nrmW: .zero, row: .zero, total: .zero, stats: .zero)
}

struct CoinFootUniformsGPU {         // 32 bytes
    var dt: Float
    var gravity: Float
    var globalMu: Float
    var footCount: UInt32
    var colliderCount: UInt32
    var bodyCount: UInt32
    var sleepEnabled: UInt32
    var lastIteration: UInt32
}

// ── Public types ──────────────────────────────────────────────────────────────

/// The static description of one spring foot. Lengths in metres, body-local frame (for an
/// egg / Weeble: origin at the COM, +Y the axis toward the tip).
public struct CoinFootSpec: Sendable, Hashable {
    /// The hip: where the leg is hinged. At the COM (the default) the leg's force has no
    /// moment arm, so a push cannot spin the body however the leg is canted.
    public var hip: SIMD3<Float>
    /// Leg direction at cant 0 (unit; default −Y, straight down out of the bottom).
    public var neutralAxis: SIMD3<Float>
    /// The axis the hop heading is measured about (default +Y) …
    public var yawAxis: SIMD3<Float>
    /// … and the heading at yaw 0 (default +X). A cant θ at yaw ψ points the foot DOWN and
    /// BACK from that heading, a = cosθ·neutral − sinθ·heading(ψ), so it pushes the body up
    /// and forward along it.
    public var headingZero: SIMD3<Float>
    /// Rest length while latched (inside the shell — the leg has no contact then; this is
    /// only where a renderer draws it).
    public var retractedLength: Float
    /// Rest length while released: the distance from the hip to the surface plus the stroke.
    public var extendedLength: Float
    /// Spring stiffness k (N/m) and damping c (N·s/m); the force is clamp(k·δ + c·δ̇, 0, maxForce).
    public var stiffness: Float
    public var damping: Float
    public var maxForce: Float
    /// Pad Coulomb μ, combined with the support's as √(μ·μs); nil ⇒ the body's own material.
    public var padFriction: Float?
    /// Torsional friction of the pad: it resists spin about the support normal with up to
    /// spinFriction · patchRadius · N.
    public var spinFriction: Float
    public var patchRadius: Float
    /// The hip yaw motor's torque limit (N·m) — also its brake while the yaw target is 0.
    /// 0 ⇒ a free hip: the body spins freely over a planted foot.
    public var yawMaxTorque: Float

    public init(hip: SIMD3<Float> = .zero,
                neutralAxis: SIMD3<Float> = SIMD3(0, -1, 0),
                yawAxis: SIMD3<Float> = SIMD3(0, 1, 0),
                headingZero: SIMD3<Float> = SIMD3(1, 0, 0),
                retractedLength: Float, extendedLength: Float,
                stiffness: Float, damping: Float, maxForce: Float = 10,
                padFriction: Float? = nil, spinFriction: Float = 0.5, patchRadius: Float = 0.002,
                yawMaxTorque: Float = 0) {
        self.hip = hip
        self.neutralAxis = neutralAxis
        self.yawAxis = yawAxis
        self.headingZero = headingZero
        self.retractedLength = retractedLength
        self.extendedLength = extendedLength
        self.stiffness = stiffness
        self.damping = damping
        self.maxForce = maxForce
        self.padFriction = padFriction
        self.spinFriction = spinFriction
        self.patchRadius = patchRadius
        self.yawMaxTorque = yawMaxTorque
    }

    /// Hip-to-surface distance along a leg canted by `cant` for a hip `hipHeight` above a
    /// level surface (an upright body): hipHeight / cos(cant). Add the stroke for
    /// `extendedLength`.
    public static func legLength(hipHeight: Float, cant: Float) -> Float {
        hipHeight / max(cos(cant), 1e-3)
    }

    /// The Digital Clock worker's foot (the §B1 Weeble: m 14.63 g, COM 6.63 mm above the
    /// table when upright). Hip at the COM; a 4 mm stroke on k = 230 N/m, c = 0.1 N·s/m stores
    /// ½kΔ² = 1.84 mJ — a 12 mm straight-up hop (11.9 mm after the damper's 0.13 mJ). The
    /// pad (r 2 mm, μ_spin 0.5) turns the worker in place on its 6e-5 N·m yaw motor.
    public static func weebleWorker(hipHeight: Float = 0.006626, stroke: Float = 0.004) -> CoinFootSpec {
        CoinFootSpec(retractedLength: 0.0055, extendedLength: hipHeight + stroke,
                     stiffness: 230, damping: 0.1, maxForce: 3,
                     padFriction: nil, spinFriction: 0.5, patchRadius: 0.002, yawMaxTorque: 6e-5)
    }
}

/// What the leg's rest length does.
public enum CoinFootRest: Sendable, Equatable {
    /// Latched inside the shell: no contact, no force, no GPU work.
    case retracted
    /// Held out at `extendedLength` (a stand for turning in place, or a springy landing).
    case extended
    /// Released ONCE: extended until the foot next lifts off after pushing, then latched
    /// retracted by the GPU itself (no host round trip between the push and the landing).
    case pulse
}

/// What a planted foot stands on.
public enum CoinFootSupport: Sendable, Equatable {
    case none
    case collider(Int)
    case body(Int)
}

/// One foot's state after the last completed frame (its last substep).
public struct CoinFootReading: Sendable {
    public var planted: Bool
    public var support: CoinFootSupport
    /// World hip (from the body's pose now).
    public var hip: SIMD3<Float>
    /// World tip: the planted point, else where the free leg's end is (hip + axis · rest).
    public var tip: SIMD3<Float>
    /// Support normal at the planted tip (world).
    public var normal: SIMD3<Float>
    /// Leg compression δ = L0 − |tip − hip| at the last substep's start (m; 0 in the air).
    public var compression: Float
    /// Leg force over the last substep (N).
    public var force: Float
    /// Push direction on the body (tip → hip, world).
    public var pushDirection: SIMD3<Float>
    /// The rest length now in force (extended while active, else retracted).
    public var restLength: Float
    /// Where the free leg points (world).
    public var axisWorld: SIMD3<Float>
    /// Cumulative leg impulse ON THE BODY since addFoot / resetFootTotals (world, N·s); a
    /// dynamic support received exactly the opposite.
    public var impulseTotal: SIMD3<Float>
    /// Cumulative torsional impulse on the body about the support normal (N·m·s).
    public var torsionTotal: Float
    /// Torsion row of the last substep: its bound (N·m·s) and accumulated impulse.
    public var torsionBound: Float
    public var torsionImpulse: Float
    public var plants: Int
    public var liftoffs: Int
    public var skids: Int
    /// A pulse was armed and has pushed and lifted off (the leg is latched again).
    public var pulseFired: Bool
    public var mode: CoinFootRest
}

// ── The per-solver foot table ─────────────────────────────────────────────────

@MainActor
final class CoinFootSystem {
    static let maxFeet = 64

    struct Record {
        var body: Int
        var spec: CoinFootSpec
        var mode: CoinFootRest = .retracted
        var armSeq: UInt32 = 0
        var cant: Float = 0
        var yaw: Float = 0
        var yawRate: Float = 0
    }

    var records: [Record?] = []
    let configBuffer: MTLBuffer
    let stateBuffer: MTLBuffer
    var substepPipeline: MTLComputePipelineState?
    var iteratePipeline: MTLComputePipelineState?
    var unionPipeline: MTLComputePipelineState?
    /// Velocity iterations encoded since the last substep hook (to flag the last one).
    var iterationCursor = 0
    /// This substep dispatches coinFootIterate on EVERY velocity iteration (a planted foot on a
    /// dynamic support: its rows couple to that body's contacts) or only on the last one.
    var iterateEveryIteration = false
    /// Foot dispatches encoded so far (tests: a retracted foot must cost nothing).
    var dispatchCount = 0

    init?(device: MTLDevice) {
        guard let c = device.makeBuffer(length: MemoryLayout<CoinFootGPU>.stride * Self.maxFeet,
                                        options: .storageModeShared),
              let s = device.makeBuffer(length: MemoryLayout<CoinFootStateGPU>.stride * Self.maxFeet,
                                        options: .storageModeShared) else { return nil }
        c.label = "Coin.feet"
        s.label = "Coin.footState"
        memset(c.contents(), 0, c.length)
        memset(s.contents(), 0, s.length)
        configBuffer = c
        stateBuffer = s
    }

    var config: UnsafeMutablePointer<CoinFootGPU> {
        configBuffer.contents().bindMemory(to: CoinFootGPU.self, capacity: Self.maxFeet)
    }
    var state: UnsafeMutablePointer<CoinFootStateGPU> {
        stateBuffer.contents().bindMemory(to: CoinFootStateGPU.self, capacity: Self.maxFeet)
    }
    var highWater: Int { records.count }

    /// Any foot whose leg is out (extended, or a pulse not yet fired as of the last frame).
    var anyActive: Bool {
        for (i, r) in records.enumerated() {
            guard let r else { continue }
            switch r.mode {
            case .retracted: continue
            case .extended: return true
            case .pulse: if state[i].meta.w != r.armSeq { return true }
            }
        }
        return false
    }

    /// Any foot planted on a DYNAMIC body as of the last completed frame.
    var anyOnDynamicSupport: Bool {
        for (i, r) in records.enumerated() where r != nil {
            let m = state[i].meta
            if m.x != 0 && m.y != 0xFFFF_FFFF { return true }
        }
        return false
    }

    func index(ofBody body: Int) -> Int? {
        records.firstIndex { $0?.body == body }
    }

    /// Body-local leg direction for a cant / yaw.
    static func legAxis(_ s: CoinFootSpec, cant: Float, yaw: Float) -> SIMD3<Float> {
        let a0 = simd_normalize(s.neutralAxis)
        let heading = simd_act(simd_quatf(angle: yaw, axis: simd_normalize(s.yawAxis)), simd_normalize(s.headingZero))
        return simd_normalize(cos(cant) * a0 - sin(cant) * heading)
    }

    func upload(_ i: Int) {
        guard let r = records[i] else { config[i] = CoinFootGPU(meta: .zero, mount: .zero, axis: .zero, spring: .zero, spin: .zero); return }
        let s = r.spec
        let mode: UInt32
        switch r.mode {
        case .retracted: mode = 0
        case .extended: mode = 1
        case .pulse: mode = 2
        }
        config[i] = CoinFootGPU(
            meta: SIMD4(UInt32(r.body), 1, mode, r.armSeq),
            mount: SIMD4(s.hip, s.extendedLength),
            axis: SIMD4(Self.legAxis(s, cant: r.cant, yaw: r.yaw), s.retractedLength),
            spring: SIMD4(s.stiffness, max(s.damping, 0), max(s.maxForce, 0), s.padFriction ?? -1),
            spin: SIMD4(r.yawRate, max(s.yawMaxTorque, 0), max(s.spinFriction, 0), max(s.patchRadius, 0)))
    }
}

// ── The extension ─────────────────────────────────────────────────────────────

@MainActor
extension CoinDEMSolver {

    private static let footKernelNames = ["coinFootSubstep", "coinFootIterate", "coinFootIslandUnion"]

    /// The foot table, made on first use with the kernels from the engine's pipeline cache
    /// (the package metallib) unless a test already installed them.
    private func makeFootSystem(resolveKernels: Bool = true) -> CoinFootSystem? {
        if let f = footSystem { return f }
        guard let f = CoinFootSystem(device: device) else { return nil }
        if resolveKernels {
            f.substepPipeline = engine.pipeline(Self.footKernelNames[0])
            f.iteratePipeline = engine.pipeline(Self.footKernelNames[1])
            f.unionPipeline = engine.pipeline(Self.footKernelNames[2])
        }
        footSystem = f
        return f
    }

    /// Test seam: take the foot kernels from a runtime-compiled CoinDEM library (the SwiftPM
    /// CLI builds no metallib, so `engine.pipeline` has nothing to find there).
    @discardableResult
    func useFootKernels(from library: MTLLibrary) -> Bool {
        guard let f = makeFootSystem(resolveKernels: false) else { return false }
        func pso(_ name: String) -> MTLComputePipelineState? {
            guard let fn = library.makeFunction(name: name) else { return nil }
            return try? device.makeComputePipelineState(function: fn)  // gpu-ok: setup-time, test library seam
        }
        f.substepPipeline = pso(Self.footKernelNames[0])
        f.iteratePipeline = pso(Self.footKernelNames[1])
        f.unionPipeline = pso(Self.footKernelNames[2])
        return f.substepPipeline != nil && f.iteratePipeline != nil && f.unionPipeline != nil
    }

    // ── Host API ──────────────────────────────────────────────────────────────

    /// Give `body` a spring foot (latched retracted, aimed at cant 0). One foot per body;
    /// a second call replaces the spec. False if the table is full, the kernels are missing, or
    /// the solver is not in `.constraint` mode — the foot hooks live only in the constraint
    /// substep, so a foot on the legacy Jacobi path would be accepted and then never move
    /// anything. (Set `solverMode` first; switching back to `.legacy` later leaves feet inert.)
    @discardableResult
    public func addFoot(body: Int, spec: CoinFootSpec) -> Bool {
        guard solverMode == .constraint else { return false }
        guard body >= 0, body < maxCoins, let f = makeFootSystem(),
              f.substepPipeline != nil, f.iteratePipeline != nil, f.unionPipeline != nil else { return false }
        let i: Int
        if let existing = f.index(ofBody: body) { i = existing }
        else if let free = f.records.firstIndex(where: { $0 == nil }) { i = free }
        else if f.records.count < CoinFootSystem.maxFeet { f.records.append(nil); i = f.records.count - 1 }
        else { return false }
        f.records[i] = CoinFootSystem.Record(body: body, spec: spec)
        f.state[i] = .zero
        f.upload(i)
        return true
    }

    public func removeFoot(body: Int) {
        guard let f = footSystem, let i = f.index(ofBody: body) else { return }
        f.records[i] = nil
        f.state[i] = .zero
        f.upload(i)
        while let last = f.records.last, last == nil { f.records.removeLast() }
    }

    public func hasFoot(body: Int) -> Bool { footSystem?.index(ofBody: body) != nil }

    /// Command a foot. Every argument left nil keeps its value.
    ///   • rest — `.pulse` arms ONE release (extended until the next liftoff after pushing,
    ///     then latched by the GPU); `.extended` holds the leg out; `.retracted` latches it
    ///     now (and un-plants it). Anything but `.retracted` wakes the body.
    ///   • extendedLength — hip-to-surface distance + stroke (CoinFootSpec.legLength).
    ///   • cant / yaw — the leg's aim for its NEXT plant (radians; a planted leg swings freely
    ///     about the hip and keeps its tip).
    ///   • yawRate — the hip yaw motor's target (rad/s, the body relative to its support
    ///     about the support normal) while planted; 0 brakes. yawMaxTorque its limit (N·m).
    ///   • hip — OPT-IN (nil keeps it; the default is the spec's): where the leg is hinged, body-local
    ///     (the principal frame, COM at the origin). A SLIP leg pushes through its body's centre of
    ///     mass; a body that CARRIES something (a load held by other bodies jointed to it) has its
    ///     centre of mass off its own, and a leg left at its own COM then pitches the whole assembly
    ///     with every push. The caller moves the hip to the assembly's COM while it carries, and back.
    public func setFoot(body: Int, rest: CoinFootRest? = nil, extendedLength: Float? = nil,
                        cant: Float? = nil, yaw: Float? = nil,
                        yawRate: Float? = nil, yawMaxTorque: Float? = nil, hip: SIMD3<Float>? = nil) {
        guard let f = footSystem, let i = f.index(ofBody: body), var r = f.records[i] else { return }
        var wakeIt = false
        if let rest {
            switch rest {
            case .pulse: r.armSeq &+= 1; wakeIt = true
            case .extended: wakeIt = true
            case .retracted: f.state[i].meta.x = 0     // pulled in: un-plant now (the GPU may not run)
            }
            r.mode = rest
        }
        if let extendedLength { r.spec.extendedLength = max(extendedLength, 0) }
        if let cant { r.cant = cant }
        if let yaw { r.yaw = yaw }
        if let yawRate {
            if yawRate != r.yawRate { wakeIt = true }
            r.yawRate = yawRate
        }
        if let yawMaxTorque { r.spec.yawMaxTorque = max(yawMaxTorque, 0) }
        if let hip { r.spec.hip = hip }
        f.records[i] = r
        f.upload(i)
        if wakeIt && r.mode != .retracted {
            var slots = [body]
            let s = f.state[i]
            if s.meta.x != 0, s.meta.y != 0xFFFF_FFFF, Int(s.meta.y) < maxCoins { slots.append(Int(s.meta.y)) }
            wake(slots)
        }
    }

    /// Zero a foot's cumulative impulse / torsion totals and its counters.
    public func resetFootTotals(body: Int) {
        guard let f = footSystem, let i = f.index(ofBody: body) else { return }
        f.state[i].total = .zero
        f.state[i].stats = SIMD4(0, 0, 0, f.state[i].stats.w)
    }

    /// The foot's state after the last completed frame (nil if `body` has no foot).
    public func footReading(body: Int) -> CoinFootReading? {
        guard let f = footSystem, let i = f.index(ofBody: body), let r = f.records[i] else { return nil }
        let s = f.state[i]
        let c = f.config[i]
        let b = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)[body]
        let q = simd_quatf(ix: b.orient.x, iy: b.orient.y, iz: b.orient.z, r: b.orient.w)
        let com = SIMD3(b.posInvMass.x, b.posInvMass.y, b.posInvMass.z)
        let hip = com + simd_act(q, r.spec.hip)
        let axisW = simd_act(q, SIMD3(c.axis.x, c.axis.y, c.axis.z))
        let fired = r.mode == .pulse && s.meta.w == r.armSeq
        let out = r.mode == .extended || (r.mode == .pulse && !fired)
        let rest = out ? r.spec.extendedLength : r.spec.retractedLength
        let planted = s.meta.x != 0
        let support: CoinFootSupport = !planted ? .none
            : (s.meta.y == 0xFFFF_FFFF ? .collider(Int(s.meta.z)) : .body(Int(s.meta.y)))
        let tipPlanted = SIMD3(s.point.x, s.point.y, s.point.z)
        return CoinFootReading(
            planted: planted, support: support, hip: hip,
            tip: planted ? tipPlanted : hip + axisW * rest,
            normal: SIMD3(s.nrmW.x, s.nrmW.y, s.nrmW.z),
            compression: planted ? s.tipL.w : 0,
            force: planted ? s.point.w : 0,
            pushDirection: SIMD3(s.dir.x, s.dir.y, s.dir.z),
            restLength: rest, axisWorld: axisW,
            impulseTotal: SIMD3(s.total.x, s.total.y, s.total.z),
            torsionTotal: s.total.w,
            torsionBound: s.row.y, torsionImpulse: s.row.w,
            plants: Int(s.stats.x), liftoffs: Int(s.stats.y), skids: Int(s.stats.z),
            pulseFired: fired, mode: r.mode)
    }

    /// Feet dispatches encoded so far (tests / instrumentation).
    var footDispatchCount: Int { footSystem?.dispatchCount ?? 0 }

    // ── Encode hooks (called from CoinDEMSolver.swift; no-ops without an active foot) ──

    func footUniforms(_ f: CoinFootSystem, colliderCount: Int, last: Bool) -> CoinFootUniformsGPU {
        CoinFootUniformsGPU(dt: fixedDt, gravity: gravity, globalMu: frictionCoeff,
                            footCount: UInt32(f.highWater), colliderCount: UInt32(colliderCount),
                            bodyCount: UInt32(highWater), sleepEnabled: sleepEnabled ? 1 : 0,
                            lastIteration: last ? 1 : 0)
    }

    /// Per substep, after coinIntegrateVelocityCS and before contact generation.
    func encodeFootSubstep(_ cb: MTLCommandBuffer, colliders: MTLBuffer, colliderCount: Int, bias: MTLBuffer) {
        guard let f = footSystem else { return }
        f.iterationCursor = 0
        guard f.anyActive, let pso = f.substepPipeline, let enc = cb.makeComputeCommandEncoder() else { return }
        // Contact torsion (stage B2, `setPatchRadius`) puts a second row on a planted body's spin
        // about the support normal — the shell's own patch friction — so the pad's row must
        // converge WITH it (Gauss–Seidel, every iteration), not get the last word once after it.
        f.iterateEveryIteration = f.anyOnDynamicSupport || torsionEnabled
        var u = footUniforms(f, colliderCount: colliderCount, last: false)
        enc.label = "Coin.cs.footSubstep"
        enc.setComputePipelineState(pso)
        enc.setBuffer(coinBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(bias, offset: 0, index: 1)
        enc.setBuffer(f.configBuffer, offset: 0, index: 2)
        enc.setBuffer(f.stateBuffer, offset: 0, index: 3)
        enc.setBuffer(colliders, offset: 0, index: 4)
        enc.setBuffer(materialBuffer, offset: 0, index: 5)
        enc.setBuffer(asleepBuffer, offset: 0, index: 6)
        enc.setBuffer(sleepTimerBuffer, offset: 0, index: 7)
        enc.setBytes(&u, length: MemoryLayout<CoinFootUniformsGPU>.stride, index: 8)
        let w = max(1, min(f.highWater, pso.maxTotalThreadsPerThreadgroup, CoinFootSystem.maxFeet))
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
        f.dispatchCount += 1
    }

    /// Per velocity iteration, after that iteration's contact colours and joint pass. Feet whose
    /// pads all stand on STATICS are dispatched on the last iteration only: a pad's torsion row
    /// about the support normal is then decoupled from every other row (an upright body's own
    /// contacts sit under its COM: their friction has no lever about the normal), so one
    /// accumulated-clamp solve after the contacts is exact — and five dispatches per substep
    /// cheaper. A foot on a dynamic support couples through that body: every iteration. So does
    /// any foot while contact TORSION is on (stage B2): the body's own contacts then carry a
    /// torsion row about the same normal, and a pad row solved once after them always had the
    /// last word — a yaw motor weaker than the shell's patch friction still turned the body
    /// (τ_pad·dt/I per substep: the scene's 0.15 mm-preload turn against a 3 mm shell patch crept
    /// 24° in 1 s at 0.42 rad/s, CoinDEMTorsionTests) instead of stalling.
    func encodeFootIteration(_ cb: MTLCommandBuffer, colliders: MTLBuffer, colliderCount: Int, bias: MTLBuffer) {
        guard let f = footSystem else { return }
        f.iterationCursor += 1
        let last = f.iterationCursor >= velocityIterations
        guard last || f.iterateEveryIteration else { return }
        guard f.anyActive, let pso = f.iteratePipeline, let enc = cb.makeComputeCommandEncoder() else { return }
        var u = footUniforms(f, colliderCount: colliderCount, last: last)
        enc.label = "Coin.cs.footIterate"
        enc.setComputePipelineState(pso)
        enc.setBuffer(coinBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(bias, offset: 0, index: 1)
        enc.setBuffer(f.configBuffer, offset: 0, index: 2)
        enc.setBuffer(f.stateBuffer, offset: 0, index: 3)
        enc.setBuffer(asleepBuffer, offset: 0, index: 4)
        enc.setBytes(&u, length: MemoryLayout<CoinFootUniformsGPU>.stride, index: 5)
        enc.setBuffer(colliders, offset: 0, index: 6)
        // One group, a thread per foot: feet on statics in parallel, dynamic supports serially.
        let w = max(1, min(f.highWater, pso.maxTotalThreadsPerThreadgroup, CoinFootSystem.maxFeet))
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
        f.dispatchCount += 1
    }

    /// Per island-union round of the sleep update.
    func encodeFootIslandUnion(_ cb: MTLCommandBuffer, label: MTLBuffer) {
        guard let f = footSystem, f.anyActive, let pso = f.unionPipeline,
              let enc = cb.makeComputeCommandEncoder() else { return }
        var u = footUniforms(f, colliderCount: 0, last: false)
        enc.label = "Coin.cs.footIslandUnion"
        enc.setComputePipelineState(pso)
        enc.setBuffer(label, offset: 0, index: 0)
        enc.setBuffer(f.configBuffer, offset: 0, index: 1)
        enc.setBuffer(f.stateBuffer, offset: 0, index: 2)
        enc.setBytes(&u, length: MemoryLayout<CoinFootUniformsGPU>.stride, index: 3)
        let n = max(1, f.highWater)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(n, pso.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        enc.endEncoding()
        f.dispatchCount += 1
    }
}
