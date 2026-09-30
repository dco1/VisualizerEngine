import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Engine gate for the SPRING-FOOT actuator (plan item 5b; CoinDEMSolver+Foot.swift and
/// Shaders/CoinDEMFoot.h): the Digital Clock's Weeble worker hopping on a massless compliant
/// leg, run through the real solver at the scene's §G configuration (1/180 s, 6 velocity
/// iterations, sleep on) via the same runtime-compiled-library seam as the other CoinDEM
/// suites. Every physical claim is checked against an independent CPU reference — a
/// continuous RK4 integration (1 µs steps, Double) of the same leg model (a point mass on a
/// one-sided spring-damper leg pinned at its tip, hinged at the COM) followed by exact
/// ballistic flight — not against numbers tuned to the GPU.
///
/// Measured numbers are PRINTed with a `FOOT_` prefix so a run's log is the record.
/// `VIZ_FOOT_GIF=1` additionally records a 4-hop + planted-turn run for the side-view GIF.
@MainActor
final class CoinDEMFootTests: XCTestCase {

    // ── Harness ───────────────────────────────────────────────────────────────

    static var shaderURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
    }

    private static var cached: (SimEngine, MTLLibrary, MTLCommandQueue)?
    private static func shared() throws -> (SimEngine, MTLLibrary, MTLCommandQueue) {
        if let c = cached { return c }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness queue
        guard FileManager.default.fileExists(atPath: shaderURL.path) else { throw XCTSkip("CoinDEM.metal not found") }
        let c = (SimEngine(device: device), try MetalSourceLoader.makeLibrary(device: device, contentsOf: shaderURL), queue)
        cached = c
        return c
    }

    static let g: Float = 9.81

    /// The Digital Clock worker (§B1): 0.9 mm ABS shell + 6.5 mm lead cap fill.
    static let worker = CoinBallastedEgg(fatRadius: 0.0115, tipRadius: 0.0075, centerDistance: 0.017,
                                         shellThickness: 0.0009, shellDensity: 1050,
                                         ballastFillHeight: 0.0065, ballastDensity: 11300)
    static var props: CoinEggMassProperties { CoinDEMSolver.ballastedEggProperties(worker) }
    /// COM height above the table when upright: r1 − d.
    static var hc: Float { worker.fatRadius - props.comBelowFatCenter }

    /// The table: the nightstand top as a static box, its top face at y = 0 (oak, μ 0.5).
    static let table = CoinStaticCollider.box(center: SIMD3(0, -0.011, 0), halfExtents: SIMD3(0.28, 0.011, 0.21),
                                              friction: 0.5, restitution: 0.15)

    /// The scene's solver knobs (§G + the stage-A fixes the clock turns on).
    static func applyClockConfig(_ s: CoinDEMSolver) {
        s.solverMode = .constraint
        s.gravity = g
        s.fixedDt = 1.0 / 180
        s.maxSubsteps = 10
        s.velocityIterations = 6
        s.colorRounds = 16
        s.islandUnionRounds = 6
        s.warmStart = false
        s.linDamping = 0.99995
        s.angDamping = 0.9998
        s.frictionCoeff = 0.5
        s.accumulatedRollingResistance = true
        s.rollingResistance = 0.024
        s.scaleAwareDeadStop = true            // stage A: a slow slew / spin is motion, not creep
        s.restitution = 0.15
        s.restThreshold = 0.14
        s.contactSlop = 1e-4
        s.baumgarteBeta = 0.2
        s.speculativeMargin = 2e-4
        s.maxSpeed = 3
        s.maxHSpeed = 3
        s.maxOmega = 60
        s.floorY = -0.5
        s.sleepEnabled = true
        s.sleepFrames = 20
        s.sleepLinVel = 2e-4
    }

    func makeSolver(library: MTLLibrary? = nil, maxCoins: Int = 16, maxRadius: Float = 0.03,
                    colliders: [CoinStaticCollider]? = nil, feet: Bool = true) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib0, queue) = try Self.shared()
        let lib = library ?? lib0
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: maxCoins,
                                    coinRadius: maxRadius, halfThickness: maxRadius,
                                    boundsMin: SIMD3(-0.4, -0.1, -0.4), boundsMax: SIMD3(0.4, 0.4, 0.4))
        else { throw XCTSkip("solver init failed") }
        Self.applyClockConfig(s)
        s.setColliders(colliders ?? [Self.table])
        if feet { XCTAssertTrue(s.useFootKernels(from: lib), "foot kernels in the CoinDEM library") }
        return (s, queue)
    }

    @discardableResult
    func step(_ s: CoinDEMSolver, _ q: MTLCommandQueue, frames: Int, wallDt: Float = 1.0 / 60,
              perFrame: ((Int) -> Void)? = nil) -> [Double] {
        var gpu: [Double] = []
        for f in 0..<frames {
            guard let cb = q.makeCommandBuffer() else { break }
            s.encode(to: cb, wallDt: wallDt)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            if cb.status == .error { XCTFail("command buffer error: \(String(describing: cb.error))"); break }
            gpu.append(max(0, cb.gpuEndTime - cb.gpuStartTime) * 1000)
            perFrame?(f)
        }
        return gpu
    }

    /// A worker standing upright with its fat sphere on the table at (x, z).
    @discardableResult
    func spawnWorker(_ s: CoinDEMSolver, x: Float = 0, z: Float = 0, y0: Float = 0, lift: Float = 0,
                     tilt: Float = 0) -> Int? {
        s.spawnBallastedEgg(Self.worker, fatCenter: SIMD3(x, y0 + Self.worker.fatRadius + lift, z),
                            orient: simd_quatf(angle: tilt, axis: SIMD3(1, 0, 0)), friction: 0.6, restitution: 0.15)
    }

    static func up(_ q: simd_quatf) -> SIMD3<Float> { simd_act(q, SIMD3<Float>(0, 1, 0)) }
    static func tiltDeg(_ q: simd_quatf) -> Float { acos(min(max(up(q).y, -1), 1)) * 180 / .pi }
    /// Heading of body +X about world +Y (radians; +X → 0, −Z → +π/2).
    static func heading(_ q: simd_quatf) -> Float {
        let h = simd_act(q, SIMD3<Float>(1, 0, 0))
        return atan2(-h.z, h.x)
    }
    static func wrap(_ a: Float) -> Float { atan2(sin(a), cos(a)) }

    // ── CPU reference: continuous SLIP stance + exact ballistic flight ──────────

    struct SlipReference {
        var pushTime: Double      // s
        var vLift: SIMD2<Double>  // (horizontal, vertical) at liftoff, m/s
        var liftRise: SIMD2<Double>
        var damperLoss: Double    // J
        var apex: Double          // COM rise above the start, m
        var range: Double         // horizontal COM travel when the COM is back at its start height, m
        var springEnergy: Double  // ½kΔ², J
    }

    /// A point mass `m` whose COM starts `hc` above level ground, on a massless leg canted by
    /// `cant` (tip behind, pushing forward) pinned at its tip, rest length hc/cos(cant) + stroke,
    /// force clamp(k·δ + c·δ̇, 0, ∞) along tip → COM. RK4, 1 µs.
    static func slipReference(m: Double, hc: Double, k: Double, c: Double, stroke: Double, cant: Double,
                              g: Double = Double(CoinDEMFootTests.g)) -> SlipReference {
        let d0 = hc / cos(cant), l0 = d0 + stroke
        var s = SIMD4<Double>(d0 * sin(cant), d0 * cos(cant), 0, 0)   // (x, y, vx, vy); tip at the origin
        var loss = 0.0, t = 0.0
        let dt = 1e-6
        func force(_ s: SIMD4<Double>) -> (a: SIMD2<Double>, f: Double, dd: Double) {
            let l = (s.x * s.x + s.y * s.y).squareRoot()
            let u = SIMD2(s.x / l, s.y / l)
            let delta = l0 - l, dd = -(s.z * u.x + s.w * u.y)
            let f = delta > 0 ? max(0, k * delta + c * dd) : 0
            return (SIMD2(f * u.x / m, f * u.y / m - g), f, dd)
        }
        func deriv(_ s: SIMD4<Double>) -> SIMD4<Double> {
            let a = force(s).a
            return SIMD4(s.z, s.w, a.x, a.y)
        }
        while true {
            let l = (s.x * s.x + s.y * s.y).squareRoot()
            let fr = force(s)
            if l >= l0 || (fr.f <= 0 && t > 0) { break }
            let k1 = deriv(s), k2 = deriv(s + 0.5 * dt * k1), k3 = deriv(s + 0.5 * dt * k2), k4 = deriv(s + dt * k3)
            s += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if fr.f > 0 { loss += c * fr.dd * fr.dd * dt }
            t += dt
        }
        let rise = SIMD2(s.x - d0 * sin(cant), s.y - hc)
        let apex = rise.y + s.w * s.w / (2 * g)
        // Flight back down to the start height: rise.y + vy·T − g T²/2 = 0.
        let T = (s.w + (s.w * s.w + 2 * g * rise.y).squareRoot()) / g
        return SlipReference(pushTime: t, vLift: SIMD2(s.z, s.w), liftRise: rise, damperLoss: loss,
                             apex: apex, range: rise.x + s.z * T, springEnergy: 0.5 * k * stroke * stroke)
    }

    // ── A hop, measured ───────────────────────────────────────────────────────

    struct HopMeasure {
        var startCOM: SIMD3<Float>
        var liftSubstep = -1
        /// State at the end of the RELEASE substep (the last one the leg was planted in): the
        /// leg's exact integration has put the body on the exact trajectory there.
        var releaseCOM = SIMD3<Float>.zero, releaseVel = SIMD3<Float>.zero
        var impliedApex: Float = 0            // (y − y0) + vy²/2g at release: the energy the leg delivered
        var discreteApex: Float = 0           // highest COM sample − y0 (the engine's kick-drift flight)
        var maxFlightOmega: Float = 0
        var launchTravel = SIMD3<Float>.zero  // exact ballistic touchdown from the release state
        var touchdownTravel = SIMD3<Float>.zero   // the engine's own flight, to the COM back at y0
        var landed = false
        var reading: CoinFootReading?
    }

    /// Fire one pulse from rest and follow the worker, ONE SUBSTEP per encode, to touchdown.
    func hop(_ s: CoinDEMSolver, _ q: MTLCommandQueue, worker w: Int, cant: Float, yaw: Float = 0,
             stroke: Float, maxSubsteps: Int = 120) -> HopMeasure {
        let y0 = s.position(of: w)!.y
        var m = HopMeasure(startCOM: s.position(of: w)!)
        s.setFoot(body: w, rest: .pulse,
                  extendedLength: CoinFootSpec.legLength(hipHeight: y0, cant: cant) + stroke,
                  cant: cant, yaw: yaw)
        var prev: (SIMD3<Float>, SIMD3<Float>)?
        func ballisticTouchdown(_ x: SIMD3<Float>, _ v: SIMD3<Float>) -> SIMD3<Float> {
            let a = -0.5 * Self.g, b = v.y, c = x.y - y0
            let tt = (-b - (b * b - 4 * a * c).squareRoot()) / (2 * a)
            var d = x + v * tt - m.startCOM
            d.y = 0
            return d
        }
        step(s, q, frames: maxSubsteps, wallDt: s.fixedDt) { f in
            guard !m.landed else { return }
            let x = s.position(of: w)!, v = s.velocity(of: w)!, om = s.angularVelocity(of: w)!
            let r = s.footReading(body: w)!
            if m.liftSubstep < 0 {
                if r.planted {
                    m.releaseCOM = x; m.releaseVel = v
                } else if r.pulseFired {
                    m.liftSubstep = f
                    m.impliedApex = (m.releaseCOM.y - y0) + m.releaseVel.y * m.releaseVel.y / (2 * Self.g)
                    m.launchTravel = ballisticTouchdown(m.releaseCOM, m.releaseVel)
                    m.reading = r
                }
            }
            if m.liftSubstep >= 0 {
                m.discreteApex = max(m.discreteApex, x.y - y0)
                if x.y > y0 + 1e-4 || v.y > 0 {
                    m.maxFlightOmega = max(m.maxFlightOmega, simd_length(om))
                    prev = (x, v)
                } else if let (px, pv) = prev {
                    m.touchdownTravel = ballisticTouchdown(px, pv)
                    m.landed = true
                }
            }
        }
        step(s, q, frames: 30)          // let it land and settle at the normal frame rate
        return m
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 1. A retracted foot is inert: egg-only physics bit for bit.
    // ══════════════════════════════════════════════════════════════════════════

    func testRetractedFootLeavesEggPhysicsBitIdentical() throws {
        func run(feet: Bool) throws -> ([[CoinBodyState]], Int, Int) {
            let (s, q) = try makeSolver(feet: feet)
            let a = try XCTUnwrap(spawnWorker(s, x: -0.05, tilt: 12 * .pi / 180))   // rocks, then sleeps
            let b = try XCTUnwrap(spawnWorker(s, x: 0.05, lift: 0.004))             // drops 4 mm, settles
            if feet {
                XCTAssertTrue(s.addFoot(body: a, spec: .weebleWorker()))
                XCTAssertTrue(s.addFoot(body: b, spec: .weebleWorker()))
                s.setFoot(body: a, rest: .retracted, cant: 0.2, yaw: 1.0)            // aimed, but latched
            }
            var trace: [[CoinBodyState]] = []
            var st: [CoinBodyState] = []
            step(s, q, frames: 240) { _ in s.readStates([a, b], into: &st); trace.append(st) }
            return (trace, s.asleepCount, s.footDispatchCount)
        }
        let (plain, sleptPlain, _) = try run(feet: false)
        let (footed, sleptFooted, dispatches) = try run(feet: true)
        var mismatches = 0
        for (fa, fb) in zip(plain, footed) {
            for (x, y) in zip(fa, fb) {
                let same = x.position == y.position && x.velocity == y.velocity
                    && x.angularVelocity == y.angularVelocity && x.orientation.vector == y.orientation.vector
                    && x.asleep == y.asleep
                if !same { mismatches += 1 }
            }
        }
        print("FOOT_1 frames=\(plain.count) mismatchedStates=\(mismatches) asleep plain=\(sleptPlain) footed=\(sleptFooted) footDispatches=\(dispatches)")
        XCTAssertEqual(mismatches, 0, "a retracted foot must not change the egg's physics by a single bit")
        XCTAssertEqual(dispatches, 0, "a retracted foot costs no GPU work")
        XCTAssertEqual(sleptFooted, 2, "both Weebles rock, settle and sleep as before")
        XCTAssertEqual(sleptPlain, 2)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 2. A foot-free world is bit-identical to the library WITHOUT the foot kernels.
    // ══════════════════════════════════════════════════════════════════════════

    func testFootFreeWorldMatchesTheLibraryWithoutFootKernels() throws {
        let (engine, _, _) = try Self.shared()
        let device = engine.device
        let full = try MetalSourceLoader.source(contentsOf: Self.shaderURL)
        // The pre-hook library: the same source text with the inlined CoinDEMFoot.h cut out.
        guard let a = full.range(of: "// ── inlined from CoinDEMFoot.h"),
              let bEnd = full.range(of: "// ── end CoinDEMFoot.h ──") else {
            return XCTFail("CoinDEM.metal no longer includes CoinDEMFoot.h")
        }
        var baseline = full
        baseline.removeSubrange(a.lowerBound..<bEnd.upperBound)
        // Stage B3: the small-world frame kernel (inlined after the hook) runs the foot phases on
        // the foot types, so it leaves with the hook — the pre-hook library never had it, and its
        // pipeline is optional (a solver built from this library simply never takes that path).
        if let c = baseline.range(of: "// ── inlined from CoinDEMSmallWorld.h"),
           let cEnd = baseline.range(of: "// ── end CoinDEMSmallWorld.h ──") {
            baseline.removeSubrange(c.lowerBound..<cEnd.upperBound)
        }
        XCTAssertFalse(baseline.contains("coinFootSubstep"))
        let libBase = try device.makeLibrary(source: baseline, options: nil)
        let libFull = try device.makeLibrary(source: full, options: nil)

        func run(_ lib: MTLLibrary) throws -> [CoinBodyState] {
            let (s, q) = try makeSolver(library: lib, maxCoins: 24, feet: false)
            var slots: [Int] = []
            for i in 0..<4 {       // Weebles dropped tilted, plus spheres, capsules and boxes
                let x = Float(i) * 0.04 - 0.06
                slots.append(try XCTUnwrap(s.spawnBallastedEgg(Self.worker, fatCenter: SIMD3(x, 0.02 + Float(i) * 0.005, 0),
                                                               orient: simd_quatf(angle: 0.3 * Float(i + 1), axis: simd_normalize(SIMD3(1, 0, 1))),
                                                               friction: 0.6, restitution: 0.15)))
                slots.append(try XCTUnwrap(s.spawnSphere(at: SIMD3(x + 0.01, 0.05, 0.03), radius: 0.006, mass: 0.004,
                                                         friction: 0.4, restitution: 0.3)))
                slots.append(try XCTUnwrap(s.spawnCapsule(at: SIMD3(x, 0.07, -0.03), radius: 0.003, halfLength: 0.008,
                                                          orient: SIMD4(0, 0, 0.38, 0.925), mass: 0.003)))
                slots.append(try XCTUnwrap(s.spawnBox(at: SIMD3(x - 0.01, 0.09, 0.0), halfExtents: SIMD3(0.005, 0.004, 0.006),
                                                      orient: SIMD4(0.2, 0.1, 0, 0.975), mass: 0.002)))
            }
            step(s, q, frames: 240)
            var st: [CoinBodyState] = []
            s.readStates(slots, into: &st)
            return st
        }
        let base = try run(libBase)
        let withFeet = try run(libFull)
        var diff = 0
        var maxDx: Float = 0
        for (x, y) in zip(base, withFeet) {
            if !(x.position == y.position && x.velocity == y.velocity && x.angularVelocity == y.angularVelocity
                 && x.orientation.vector == y.orientation.vector && x.asleep == y.asleep) { diff += 1 }
            maxDx = max(maxDx, simd_length(x.position - y.position))
        }
        print("FOOT_2 bodies=\(base.count) differingBodies=\(diff) maxPositionDelta=\(maxDx) m")
        XCTAssertEqual(diff, 0, "adding the foot kernels to the library changes nothing in a foot-free world")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 3. Straight-up hop: the apex is ½kΔ² minus the damper's loss.
    // ══════════════════════════════════════════════════════════════════════════

    func testVerticalHopApexMatchesSpringEnergy() throws {
        let m = Self.props.mass, hc = Self.hc
        for (c, stroke) in [(Float(0), Float(0.004)), (Float(0.1), Float(0.004)), (Float(0.1), Float(0.003))] {
            let (s, q) = try makeSolver()
            let w = try XCTUnwrap(spawnWorker(s))
            var spec = CoinFootSpec.weebleWorker(hipHeight: hc, stroke: stroke)
            spec.damping = c
            XCTAssertTrue(s.addFoot(body: w, spec: spec))
            step(s, q, frames: 40)                                       // settle (and sleep)
            let h = hop(s, q, worker: w, cant: 0, stroke: stroke)
            let ref = Self.slipReference(m: Double(m), hc: Double(h.startCOM.y), k: Double(spec.stiffness),
                                         c: Double(c), stroke: Double(stroke), cant: 0)
            let ideal = ref.springEnergy / (Double(m) * Double(Self.g))
            let net = (ref.springEnergy - ref.damperLoss) / (Double(m) * Double(Self.g))
            let r = try XCTUnwrap(h.reading)
            print(String(format: "FOOT_3 c=%.2f stroke=%.1fmm: implied apex %.3f mm (reference %.3f, ½kΔ²/mg %.3f, minus damper %.3f) engine discrete apex %.3f mm, release v %.4f m/s (ref liftoff %.4f) push %.1f ms (ref) flight|ω|max %.4f rad/s plants %d liftoffs %d skids %d impulse %.4e N·s (m·v %.4e)",
                         c, stroke * 1000, h.impliedApex * 1000, ref.apex * 1000, ideal * 1000, net * 1000,
                         h.discreteApex * 1000, h.releaseVel.y, ref.vLift.y, ref.pushTime * 1000, h.maxFlightOmega,
                         r.plants, r.liftoffs, r.skids, r.impulseTotal.y, m * h.releaseVel.y))
            XCTAssertGreaterThanOrEqual(h.liftSubstep, 0, "the pulse fired and lifted off")
            XCTAssertEqual(Double(h.impliedApex), net, accuracy: net * 0.10, "apex = (½kΔ² − damper loss)/mg within 10%")
            XCTAssertEqual(Double(h.impliedApex), ref.apex, accuracy: ref.apex * 0.03, "apex vs the continuous reference within 3%")
            XCTAssertLessThan(h.maxFlightOmega, 0.2, "a push through the COM leaves no spin")
            XCTAssertEqual(r.skids, 0)
            XCTAssertTrue(h.landed, "it came back down")
            let after = try XCTUnwrap(s.footReading(body: w))
            XCTAssertFalse(after.planted, "the pulse latched the leg: it lands on its round bottom")
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 4. Canted hop through the COM: no flight spin, and it lands where the physics says.
    // ══════════════════════════════════════════════════════════════════════════

    func testCantedHopThroughCOMHasNoSpinAndTravelsThePredictedDistance() throws {
        let m = Self.props.mass
        let stroke: Float = 0.004
        for cant in [Float(10), Float(14)].map({ $0 * .pi / 180 }) {
            let (s, q) = try makeSolver()
            let w = try XCTUnwrap(spawnWorker(s))
            let spec = CoinFootSpec.weebleWorker(hipHeight: Self.hc, stroke: stroke)
            XCTAssertTrue(s.addFoot(body: w, spec: spec))
            step(s, q, frames: 40)
            let h = hop(s, q, worker: w, cant: cant, stroke: stroke)
            let ref = Self.slipReference(m: Double(m), hc: Double(h.startCOM.y), k: Double(spec.stiffness),
                                         c: Double(spec.damping), stroke: Double(stroke), cant: Double(cant))
            let travel = simd_length(h.touchdownTravel), launch = simd_length(h.launchTravel)
            let dirDeg = atan2(-h.touchdownTravel.z, h.touchdownTravel.x) * 180 / .pi
            print(String(format: "FOOT_4 cant=%.1f°: flight |ω|max %.5f rad/s, travel: launch-implied %.3f mm, engine touchdown %.3f mm (reference %.3f) heading %.2f°, implied apex %.3f mm (ref %.3f), engine discrete apex %.3f, release v (%.4f, %.4f), ref liftoff v (%.4f, %.4f), push %.1f ms",
                         cant * 180 / .pi, h.maxFlightOmega, launch * 1000, travel * 1000, ref.range * 1000, dirDeg,
                         h.impliedApex * 1000, ref.apex * 1000, h.discreteApex * 1000,
                         simd_length(SIMD2(h.releaseVel.x, h.releaseVel.z)), h.releaseVel.y, ref.vLift.x, ref.vLift.y,
                         ref.pushTime * 1000))
            XCTAssertTrue(h.landed)
            XCTAssertLessThan(h.maxFlightOmega, 0.2, "the leg's force passes through the COM: no flight spin")
            XCTAssertEqual(Double(launch), ref.range, accuracy: ref.range * 0.03, "launch-implied travel vs the reference within 3%")
            XCTAssertEqual(Double(travel), ref.range, accuracy: ref.range * 0.10, "the engine's own flight lands within 10%")
            XCTAssertEqual(dirDeg, 0, accuracy: 2, "it hops along its heading (+X)")
            XCTAssertEqual(Double(h.impliedApex), ref.apex, accuracy: ref.apex * 0.03)
        }

        // Control: the same hop with the hip 3 mm below the COM — now the leg has a lever arm
        // and the worker leaves spinning (the "through the COM" condition is what kills spin).
        let (s, q) = try makeSolver()
        let w = try XCTUnwrap(spawnWorker(s))
        var spec = CoinFootSpec.weebleWorker(hipHeight: Self.hc, stroke: stroke)
        spec.hip = SIMD3(0, -0.003, 0)
        XCTAssertTrue(s.addFoot(body: w, spec: spec))
        step(s, q, frames: 40)
        let cant: Float = 10 * .pi / 180
        s.setFoot(body: w, rest: .pulse,
                  extendedLength: CoinFootSpec.legLength(hipHeight: Self.hc - 0.003, cant: cant) + stroke, cant: cant)
        var maxOmega: Float = 0
        var lifted = false
        step(s, q, frames: 8) { _ in
            if s.footReading(body: w)!.pulseFired { lifted = true }
            if lifted { maxOmega = max(maxOmega, simd_length(s.angularVelocity(of: w)!)) }
        }
        print(String(format: "FOOT_4c hip 3 mm below the COM: flight |ω|max %.3f rad/s", maxOmega))
        XCTAssertTrue(lifted)
        XCTAssertGreaterThan(maxOmega, 1.0, "an off-COM hip spins the body (control)")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 4b. The hip moved at run time (`setFoot(hip:)`, opt-in) acts exactly as one installed there:
    //     off the COM the push spins the body, back at the COM it does not; the reading follows it.
    // ══════════════════════════════════════════════════════════════════════════

    func testHipMovedAtRunTimeActsLikeAnInstalledHip() throws {
        let stroke: Float = 0.004, cant: Float = 10 * .pi / 180
        func hopSpin(moveHipTo hip: SIMD3<Float>?) throws -> (spin: Float, readingHip: SIMD3<Float>, com: SIMD3<Float>) {
            let (s, q) = try makeSolver()
            let w = try XCTUnwrap(spawnWorker(s))
            XCTAssertTrue(s.addFoot(body: w, spec: CoinFootSpec.weebleWorker(hipHeight: Self.hc, stroke: stroke)))
            step(s, q, frames: 40)
            let hipY = (hip?.y ?? 0)
            s.setFoot(body: w, rest: .pulse, extendedLength: CoinFootSpec.legLength(hipHeight: Self.hc + hipY, cant: cant) + stroke,
                      cant: cant, hip: hip)
            let r0 = try XCTUnwrap(s.footReading(body: w))
            var maxOmega: Float = 0
            var lifted = false, flightFrames = 0
            step(s, q, frames: 8) { _ in
                if s.footReading(body: w)!.pulseFired { lifted = true }
                // In flight only (the first 3 frames after the release — a 10 mm hop flies ≈ 6): a
                // landing's own rock is not the push's spin.
                if lifted, flightFrames < 3 { flightFrames += 1; maxOmega = max(maxOmega, simd_length(s.angularVelocity(of: w)!)) }
            }
            XCTAssertTrue(lifted)
            return (maxOmega, r0.hip, s.position(of: w)!)
        }
        let atCOM = try hopSpin(moveHipTo: nil)
        let below = try hopSpin(moveHipTo: SIMD3(0, -0.003, 0))
        let back = try hopSpin(moveHipTo: .zero)
        print(String(format: "FOOT_4b flight |ω|max: hip at the COM %.4f, moved 3 mm below it at run time %.3f, moved back %.4f rad/s",
                     atCOM.spin, below.spin, back.spin))
        XCTAssertLessThan(atCOM.spin, 0.2)
        XCTAssertGreaterThan(below.spin, 1.0, "a hip moved off the COM at run time spins the body like an installed one")
        XCTAssertLessThan(back.spin, 0.2, "moved back to the COM: no spin")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 5. Planted turn: 90° in place on the yaw motor, COM drift < 0.5 mm.
    // ══════════════════════════════════════════════════════════════════════════

    /// Turn a worker in place by `angle` on a planted stand; returns (turned, max COM drift,
    /// max torsion impulse / bound ratio, frames).
    func plantedTurn(_ s: CoinDEMSolver, _ q: MTLCommandQueue, worker w: Int, angle: Float, rate: Float,
                     standPreload: Float = 0.0005) -> (turned: Float, drift: Float, boundRatio: Float, frames: Int) {
        let start = s.position(of: w)!
        let h0 = Self.heading(s.orientation(of: w)!)
        let iAxis = Self.props.mass / Self.props.invInertiaK.y
        let spec = CoinFootSpec.weebleWorker(hipHeight: Self.hc)
        s.setFoot(body: w, rest: .extended, extendedLength: s.position(of: w)!.y + standPreload,
                  cant: 0, yaw: 0, yawRate: rate)
        var drift: Float = 0, ratio: Float = 0, braked = false, frames = 0, still = 0
        let alpha = spec.yawMaxTorque / iAxis
        step(s, q, frames: 240) { f in
            guard still < 12 else { return }
            frames = f + 1
            let x = s.position(of: w)!
            drift = max(drift, simd_length(SIMD2(x.x - start.x, x.z - start.z)))
            let r = s.footReading(body: w)!
            if r.torsionBound > 0 { ratio = max(ratio, abs(r.torsionImpulse) / r.torsionBound) }
            let turned = Self.wrap(Self.heading(s.orientation(of: w)!) - h0)
            let om = s.angularVelocity(of: w)!.y
            if !braked, abs(turned) >= abs(angle) - (om * om / (2 * alpha) + abs(om) / 60) {
                s.setFoot(body: w, yawRate: 0)
                braked = true
            }
            if braked { still = abs(om) < 0.05 ? still + 1 : 0 }
        }
        let turned = Self.wrap(Self.heading(s.orientation(of: w)!) - h0)
        s.setFoot(body: w, rest: .retracted)
        return (turned, drift, ratio, frames)
    }

    func testPlantedTurnNinetyDegreesWithoutCOMDrift() throws {
        let (s, q) = try makeSolver()
        let w = try XCTUnwrap(spawnWorker(s))
        XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker()))
        step(s, q, frames: 40)
        let r = plantedTurn(s, q, worker: w, angle: .pi / 2, rate: .pi)
        let reading = try XCTUnwrap(s.footReading(body: w))
        print(String(format: "FOOT_5 turned %.2f° in %d frames, max COM drift %.4f mm, torsion |Λ|/bound max %.3f, torsion total %.3e N·m·s, tilt %.2f°",
                     r.turned * 180 / .pi, r.frames, r.drift * 1000, r.boundRatio, reading.torsionTotal,
                     Self.tiltDeg(s.orientation(of: w)!)))
        XCTAssertEqual(r.turned, .pi / 2, accuracy: 4 * .pi / 180, "a 90° turn in place (±4°)")
        XCTAssertLessThan(r.drift, 0.0005, "the COM stays put: < 0.5 mm")
        XCTAssertLessThanOrEqual(r.boundRatio, 1.0001, "the pad's torsion never exceeds μ_spin·r·N")
        // Retracted, it rocks out and sleeps.
        var slept = -1
        step(s, q, frames: 120) { f in if slept < 0 && s.isAsleep(w) { slept = f } }
        print("FOOT_5b asleep \(slept) frames after the retract")
        XCTAssertGreaterThanOrEqual(slept, 0, "a parked, retracted worker sleeps")
    }

    /// The pad's friction bounds the turn: with μ_spin tiny the motor can't accelerate the body
    /// faster than μ_spin·r·N / I.
    func testTurnRateIsBoundedByPadFriction() throws {
        let (s, q) = try makeSolver()
        let w = try XCTUnwrap(spawnWorker(s))
        var spec = CoinFootSpec.weebleWorker()
        spec.spinFriction = 0.02
        spec.yawMaxTorque = 1e-2                                          // a motor far stronger than the grip
        XCTAssertTrue(s.addFoot(body: w, spec: spec))
        step(s, q, frames: 40)
        s.setFoot(body: w, rest: .extended, extendedLength: s.position(of: w)!.y + 0.0005, cant: 0, yawRate: 20)
        var forces: [Float] = []
        var omegas: [Float] = []
        step(s, q, frames: 6) { _ in
            forces.append(s.footReading(body: w)!.force)
            omegas.append(s.angularVelocity(of: w)!.y)
        }
        let iAxis = Self.props.mass / Self.props.invInertiaK.y
        let nMean = forces.dropFirst().reduce(0, +) / Float(max(forces.count - 1, 1))
        let alphaMax = spec.spinFriction * spec.patchRadius * nMean / iAxis
        let t: Float = Float(omegas.count) / 60
        print(String(format: "FOOT_6 μ_spin %.2f: ω after %.3f s = %.3f rad/s, friction-limited bound α·t = %.3f (N %.4f N)",
                     spec.spinFriction, t, omegas.last ?? 0, alphaMax * t, nMean))
        XCTAssertLessThanOrEqual(omegas.last ?? 0, alphaMax * t * 1.15 + 0.02, "the turn is bounded by pad friction")
        XCTAssertGreaterThan(omegas.last ?? 0, alphaMax * t * 0.5, "…and the motor does use that friction")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 6. The support gets the equal and opposite impulse (a dynamic plate, zero g).
    // ══════════════════════════════════════════════════════════════════════════

    func testSupportReactionImpulseEqualsMomentumChange() throws {
        let (s, q) = try makeSolver(colliders: [])
        s.gravity = 0
        s.sleepEnabled = false
        s.linDamping = 1
        s.angDamping = 1
        let plateHalf = SIMD3<Float>(0.02, 0.002, 0.02)
        let plate = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0, 0), halfExtents: plateHalf, mass: 0.05,
                                             friction: 0.6, restitution: 0.15))
        let gap: Float = 0.0005
        let w = try XCTUnwrap(spawnWorker(s, y0: plateHalf.y, lift: gap))
        XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker()))
        let hipHeight = s.position(of: w)!.y - plateHalf.y
        let mw = Self.props.mass, mp: Float = 0.05
        s.setFoot(body: w, rest: .pulse, extendedLength: hipHeight + 0.003, cant: 0)
        var support: CoinFootSupport = .none
        var lifted = false
        step(s, q, frames: 10) { _ in
            let r = s.footReading(body: w)!
            if r.planted { support = r.support }
            if r.pulseFired { lifted = true }
        }
        let r = try XCTUnwrap(s.footReading(body: w))
        let J = r.impulseTotal
        let pw = mw * s.velocity(of: w)!, pp = mp * s.velocity(of: plate)!
        print(String(format: "FOOT_7 support %@ leg impulse J=(%.4e, %.4e, %.4e) worker m·Δv=(%.4e, %.4e, %.4e) plate m·Δv=(%.4e, %.4e, %.4e) Σp=%.3e plate |ω| %.2e",
                     "\(support)", J.x, J.y, J.z, pw.x, pw.y, pw.z, pp.x, pp.y, pp.z, simd_length(pw + pp),
                     simd_length(s.angularVelocity(of: plate)!)))
        XCTAssertTrue(lifted)
        XCTAssertEqual(support, .body(plate), "the leg planted on the dynamic plate")
        XCTAssertGreaterThan(J.y, 0)
        XCTAssertEqual(simd_length(pw - J), 0, accuracy: simd_length(J) * 0.005, "worker: m·Δv = the leg impulse")
        XCTAssertEqual(simd_length(pp + J), 0, accuracy: simd_length(J) * 0.005, "plate: m·Δv = −the leg impulse")
        XCTAssertLessThan(simd_length(pw + pp), simd_length(J) * 1e-3, "momentum is conserved")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 7. Sleep: a slow turn or a pending release keeps the worker awake; parked, it sleeps.
    // ══════════════════════════════════════════════════════════════════════════

    func testSlowTurnAndPendingReleaseKeepAwakeParkedFootSleeps() throws {
        let (s, q) = try makeSolver()
        let w = try XCTUnwrap(spawnWorker(s))
        XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker()))
        step(s, q, frames: 60)
        XCTAssertTrue(s.isAsleep(w), "settled first")
        let h0 = Self.heading(s.orientation(of: w)!)
        // 0.3 rad/s: below the legacy dead-stop (0.6 rad/s) and the sleep test (1 rad/s).
        s.setFoot(body: w, rest: .extended, extendedLength: s.position(of: w)!.y + 0.0005, cant: 0, yawRate: 0.3)
        var everAsleep = false
        step(s, q, frames: 120) { _ in if s.isAsleep(w) { everAsleep = true } }
        let turned = Self.wrap(Self.heading(s.orientation(of: w)!) - h0)
        print(String(format: "FOOT_8 slow turn: %.3f rad in 2 s (commanded 0.6), ever asleep %@", turned, "\(everAsleep)"))
        XCTAssertFalse(everAsleep, "a turning foot keeps its worker awake")
        XCTAssertEqual(turned, 0.6, accuracy: 0.09, "the slow slew is not dead-stopped")
        // Park the way a scene must: BRAKE on the planted pad first (the egg's own point contact
        // has no torsional friction — a worker latched mid-spin keeps spinning), then latch.
        s.setFoot(body: w, yawRate: 0)
        step(s, q, frames: 10)
        XCTAssertLessThan(abs(s.angularVelocity(of: w)!.y), 0.01, "the pad's brake stops the spin")
        s.setFoot(body: w, rest: .retracted)
        var slept = -1
        step(s, q, frames: 120) { f in if slept < 0 && s.isAsleep(w) { slept = f } }
        XCTAssertGreaterThanOrEqual(slept, 0, "parked retracted, it sleeps")
        // A pulse that cannot reach the table (leg shorter than the hip height) is pending: awake.
        s.setFoot(body: w, rest: .pulse, extendedLength: Self.hc - 0.001, cant: 0)
        var asleepWhilePending = false
        step(s, q, frames: 90) { _ in if s.isAsleep(w) { asleepWhilePending = true } }
        s.setFoot(body: w, rest: .retracted)
        var slept2 = -1
        step(s, q, frames: 120) { f in if slept2 < 0 && s.isAsleep(w) { slept2 = f } }
        print("FOOT_8b slept \(slept) frames after parking; pending pulse asleep=\(asleepWhilePending); re-slept \(slept2) frames after cancelling")
        XCTAssertFalse(asleepWhilePending, "a pending release keeps the worker awake")
        XCTAssertGreaterThanOrEqual(slept2, 0)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 8. Grip: a leg canted outside the friction cone skids instead of pushing.
    // ══════════════════════════════════════════════════════════════════════════

    func testLegOutsideFrictionConeSkids() throws {
        let (s, q) = try makeSolver()
        let w = try XCTUnwrap(spawnWorker(s))
        XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker()))
        step(s, q, frames: 40)
        let y0 = s.position(of: w)!.y
        let cant: Float = 40 * .pi / 180                      // tan 40° = 0.84 > μ = √(0.6·0.5) = 0.55
        s.setFoot(body: w, rest: .pulse, extendedLength: CoinFootSpec.legLength(hipHeight: y0, cant: cant) + 0.004, cant: cant)
        var maxRise: Float = 0
        step(s, q, frames: 20) { _ in maxRise = max(maxRise, s.position(of: w)!.y - y0) }
        let r = try XCTUnwrap(s.footReading(body: w))
        print(String(format: "FOOT_9 cant 40°: skids %d plants %d max rise %.4f mm", r.skids, r.plants, maxRise * 1000))
        XCTAssertGreaterThan(r.skids, 0)
        XCTAssertEqual(r.plants, 0)
        XCTAssertLessThan(maxRise, 0.0002, "a skidding leg delivers nothing")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 9. Supports: plane, oriented box, dynamic box and sphere.
    // ══════════════════════════════════════════════════════════════════════════

    func testLegPlantsOnPlaneOrientedBoxAndDynamicBodies() throws {
        // Plane.
        do {
            let (s, q) = try makeSolver(colliders: [.plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.15)])
            let w = try XCTUnwrap(spawnWorker(s))
            XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker()))
            s.setFoot(body: w, rest: .extended, extendedLength: Self.hc + 0.0003)
            step(s, q, frames: 2)
            let r = try XCTUnwrap(s.footReading(body: w))
            XCTAssertEqual(r.support, .collider(0))
            XCTAssertEqual(r.normal.y, 1, accuracy: 1e-5)
        }
        // Oriented box tilted 8° about Z.
        do {
            let tilt = simd_quatf(angle: 8 * .pi / 180, axis: SIMD3(0, 0, 1))
            let (s, q) = try makeSolver(colliders: [.orientedBox(center: SIMD3(0, -0.01, 0), halfExtents: SIMD3(0.1, 0.01, 0.1),
                                                                 orientation: tilt, friction: 0.5, restitution: 0.15)])
            let w = try XCTUnwrap(s.spawnBallastedEgg(Self.worker, fatCenter: SIMD3(0, Self.worker.fatRadius + 0.0003, 0),
                                                       orient: tilt, friction: 0.6, restitution: 0.15))
            XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker()))
            s.setFoot(body: w, rest: .extended, extendedLength: Self.hc + 0.001)
            step(s, q, frames: 1)
            let r = try XCTUnwrap(s.footReading(body: w))
            let nWant = simd_act(tilt, SIMD3<Float>(0, 1, 0))
            print("FOOT_10 obox support \(r.support) normal \(r.normal) want \(nWant)")
            XCTAssertEqual(r.support, .collider(0))
            XCTAssertEqual(simd_dot(r.normal, nWant), 1, accuracy: 1e-4)
        }
        // Dynamic sphere and box (zero g so they stay put).
        for useSphere in [true, false] {
            let (s, q) = try makeSolver(colliders: [])
            s.gravity = 0
            let sup: Int
            if useSphere { sup = try XCTUnwrap(s.spawnSphere(at: SIMD3(0, -0.01, 0), radius: 0.01, mass: 0.1)) }
            else { sup = try XCTUnwrap(s.spawnBox(at: SIMD3(0, -0.005, 0), halfExtents: SIMD3(0.01, 0.005, 0.01), mass: 0.1)) }
            let w = try XCTUnwrap(spawnWorker(s, lift: 0.0003))
            XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker()))
            s.setFoot(body: w, rest: .extended, extendedLength: Self.hc + 0.0003 + 0.0008)
            step(s, q, frames: 1, wallDt: s.fixedDt)          // one substep: the plant's own compression
            let r = try XCTUnwrap(s.footReading(body: w))
            print("FOOT_10 dynamic \(useSphere ? "sphere" : "box") support \(r.support) normal \(r.normal) compression \(r.compression * 1000) mm")
            XCTAssertEqual(r.support, .body(sup))
            XCTAssertEqual(r.normal.y, 1, accuracy: 1e-3)
            XCTAssertEqual(r.compression, 0.0008, accuracy: 0.0002)
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 9b. A pulse whose stance ENDS IN A SKID is spent: exactly one stroke per pulse.
    // ══════════════════════════════════════════════════════════════════════════

    /// Near the friction angle (28.7° for the worker on oak) the leg's line of force drifts
    /// outward during the push and the pad skids at the end of the stroke. That stance is over:
    /// the pulse must latch, or the still-extended leg re-plants at the landing and fires a
    /// second, uncommanded stroke (before the fix: 4 plants, 9 bounces and a 130° tumble from
    /// ONE pulse at 27°).
    func testSkidAtEndOfPushLatchesThePulse() throws {
        for deg in [Float(27), Float(28)] {
            let (s, q) = try makeSolver()
            let w = try XCTUnwrap(spawnWorker(s))
            XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker()))
            step(s, q, frames: 40)
            let y0 = s.position(of: w)!.y
            let cant = deg * .pi / 180
            s.setFoot(body: w, rest: .pulse, extendedLength: CoinFootSpec.legLength(hipHeight: y0, cant: cant) + 0.004, cant: cant)
            var rises = 0, up = false, firedAt = -1
            var maxTilt: Float = 0
            step(s, q, frames: 150, wallDt: s.fixedDt) { f in
                let y = s.position(of: w)!.y - y0
                if !up && y > 0.002 { rises += 1; up = true } else if up && y < 0.0005 { up = false }
                if firedAt < 0 && s.footReading(body: w)!.pulseFired { firedAt = f }
                maxTilt = max(maxTilt, Self.tiltDeg(s.orientation(of: w)!))
            }
            let r = try XCTUnwrap(s.footReading(body: w))
            print(String(format: "FOOT_9b cant %.0f°: plants %d liftoffs %d skids %d, latched at substep %d, rises %d, max tilt %.1f°",
                         deg, r.plants, r.liftoffs, r.skids, firedAt, rises, maxTilt))
            XCTAssertEqual(r.plants, 1, "one pulse, one stance")
            XCTAssertTrue(r.pulseFired)
            XCTAssertGreaterThanOrEqual(firedAt, 0)
            XCTAssertLessThanOrEqual(firedAt, 6, "latched at the end of the push, not after a landing")
            XCTAssertEqual(rises, 1, "one hop — no uncommanded second stroke at the landing")
            XCTAssertLessThan(maxTilt, 60, "it lands on its round bottom instead of being knocked over")
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 9c. Feet exist only on the constraint path.
    // ══════════════════════════════════════════════════════════════════════════

    func testAddFootRefusesTheLegacySolver() throws {
        let (s, _) = try makeSolver()
        // Spawn on the constraint path FIRST: the worker is a ballasted egg, and `spawnEgg`
        // asserts `.constraint` (legacy has no ovoid narrowphase) — switching before the spawn
        // trapped the whole debug test process (Daydream DH-0959).
        let w = try XCTUnwrap(spawnWorker(s))
        s.solverMode = .legacy
        XCTAssertFalse(s.addFoot(body: w, spec: .weebleWorker()), "the legacy substep has no foot hooks: refuse, don't no-op")
        XCTAssertFalse(s.hasFoot(body: w))
        s.solverMode = .constraint
        XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker()))
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 9d. A push wakes an ASLEEP dynamic support and hands it the reaction (sleep on).
    // ══════════════════════════════════════════════════════════════════════════

    func testPushWakesAsleepSupportAndConservesMomentum() throws {
        let (s, q) = try makeSolver(colliders: [])
        s.gravity = 0
        s.linDamping = 1
        s.angDamping = 1
        let plateHalf = SIMD3<Float>(0.02, 0.002, 0.02)
        let plate = try XCTUnwrap(s.spawnBox(at: .zero, halfExtents: plateHalf, mass: 0.05, friction: 0.6, restitution: 0.15))
        let w = try XCTUnwrap(spawnWorker(s, y0: plateHalf.y, lift: 0.0005))
        XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker()))
        step(s, q, frames: 40)
        XCTAssertTrue(s.isAsleep(plate), "the floating plate fell asleep first")
        let hipHeight = s.position(of: w)!.y - plateHalf.y
        s.setFoot(body: w, rest: .pulse, extendedLength: hipHeight + 0.003, cant: 0)
        step(s, q, frames: 6)
        let r = try XCTUnwrap(s.footReading(body: w))
        let J = r.impulseTotal
        let pw = Self.props.mass * s.velocity(of: w)!, pp = 0.05 * s.velocity(of: plate)!
        print(String(format: "FOOT_7b asleep plate: J %.4e worker m·v %.4e plate m·v %.4e Σp %.3e",
                     J.y, pw.y, pp.y, simd_length(pw + pp)))
        XCTAssertEqual(r.plants, 1)
        XCTAssertGreaterThan(J.y, 0)
        XCTAssertEqual(simd_length(pp + J), 0, accuracy: simd_length(J) * 0.005, "the woken plate took −J")
        XCTAssertLessThan(simd_length(pw + pp), simd_length(J) * 1e-3, "momentum is conserved")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 10. Cost: µs per substep with 4 feet.
    // ══════════════════════════════════════════════════════════════════════════

    func testFootCostPerSubstepWithFourFeet() throws {
        // Two identical worlds, 4 workers standing, sleep OFF so nothing skips: feet planted as
        // stands (every foot kernel runs every substep) vs feet retracted (none runs).
        func world(active: Bool) throws -> (CoinDEMSolver, MTLCommandQueue, [Int]) {
            let (s, q) = try makeSolver()
            s.sleepEnabled = false
            var ws: [Int] = []
            for i in 0..<4 {
                let w = try XCTUnwrap(spawnWorker(s, x: Float(i) * 0.05 - 0.075))
                XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker()))
                ws.append(w)
            }
            step(s, q, frames: 30)
            if active { for w in ws { s.setFoot(body: w, rest: .extended, extendedLength: Self.hc + 0.0005, yawRate: 0) } }
            step(s, q, frames: 30)
            return (s, q, ws)
        }
        let (sa, qa, wa) = try world(active: true)
        let (sb, qb, _) = try world(active: false)
        var a: [Double] = [], b: [Double] = []
        let d0 = sa.footDispatchCount
        for _ in 0..<6 {                                                 // interleaved blocks
            a += step(sa, qa, frames: 40)
            b += step(sb, qb, frames: 40)
        }
        let dispatchesPerSubstep = Double(sa.footDispatchCount - d0) / Double(6 * 40 * sa.lastStepCount)
        func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
        let substeps = Double(sa.lastStepCount)
        let perSubstep = (median(a) - median(b)) / substeps * 1000
        let planted = wa.filter { sa.footReading(body: $0)!.planted }.count
        print(String(format: "FOOT_11 4 planted feet: frame p50 %.3f ms vs %.3f ms without → %.1f µs per substep (%d substeps/frame, %d planted, %.2f foot dispatches/substep)",
                     median(a), median(b), perSubstep, Int(substeps), planted, dispatchesPerSubstep))
        XCTAssertEqual(planted, 4)
        XCTAssertLessThan(perSubstep, 250, "4 feet cost well under a quarter millisecond per substep")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 11. (VIZ_FOOT_GIF=1) Record 4 hops across the table with a planted 90° turn.
    // ══════════════════════════════════════════════════════════════════════════

    func testRecordFourHopsWithPlantedTurnForGIF() throws {
        guard ProcessInfo.processInfo.environment["VIZ_FOOT_GIF"] == "1" else {
            throw XCTSkip("set VIZ_FOOT_GIF=1 to record the hop run")
        }
        let out = ProcessInfo.processInfo.environment["VIZ_FOOT_GIF_OUT"]
            ?? "/tmp/foot_hops.json"
        let (s, q) = try makeSolver()
        let w = try XCTUnwrap(spawnWorker(s, x: -0.022, z: 0.004))
        let spec = CoinFootSpec.weebleWorker()
        XCTAssertTrue(s.addFoot(body: w, spec: spec))
        let props = Self.props
        let stroke: Float = 0.004
        let cant: Float = 10 * .pi / 180
        let iAxis = props.mass / props.invInertiaK.y

        // The controller a scene would run: one decision per completed frame, from measured
        // state only. Every action waits for the worker to be SETTLED (upright, still); a
        // landing that leaves it spinning is stopped by planting the foot as a yaw brake.
        enum Action { case hop(Int), turn, finish }
        enum Phase {
            case settle(next: Action, count: Int, elapsed: Int, braking: Bool)
            case hopping(n: Int, elapsed: Int)
            case turning(h0: Float, braked: Bool, still: Int)
            case done(Int)
        }
        var phase = Phase.settle(next: .hop(1), count: 0, elapsed: -30, braking: false)   // stand 0.5 s first
        var label = "settle"
        var frames: [[String: Any]] = []
        var events: [[String: Any]] = []
        var restY = s.position(of: w)!.y
        var hopsDone = 0
        var turnedDeg: Float = 0
        let alpha = spec.yawMaxTorque / iAxis

        func stand(_ y: Float, rate: Float) {
            s.setFoot(body: w, rest: .extended, extendedLength: y + 0.0005, cant: 0, yaw: 0, yawRate: rate)
        }
        for f in 0..<600 {
            let x = s.position(of: w)!, qb = s.orientation(of: w)!, om = s.angularVelocity(of: w)!
            let v = s.velocity(of: w)!
            let r = s.footReading(body: w)!
            switch phase {
            case .settle(let next, let count, let elapsed, let braking):
                let quiet = Self.tiltDeg(qb) < 1.5 && simd_length(om) < 0.25 && simd_length(v) < 0.004
                var braking = braking
                if !braking && abs(om.y) > 0.3 && Self.tiltDeg(qb) < 3 {
                    stand(x.y, rate: 0); braking = true; label = "yaw brake"
                    events.append(["frame": f, "event": "brake plant"])
                } else if braking && abs(om.y) < 0.05 {
                    s.setFoot(body: w, rest: .retracted); braking = false; label = "settle"
                }
                let c = (quiet && !braking) ? count + 1 : 0
                if (c >= 4 && elapsed >= 0) || elapsed >= 90 {
                    if braking { s.setFoot(body: w, rest: .retracted) }
                    switch next {
                    case .hop(let n):
                        restY = x.y
                        s.setFoot(body: w, rest: .pulse,
                                  extendedLength: CoinFootSpec.legLength(hipHeight: restY, cant: cant) + stroke,
                                  cant: cant, yaw: 0)
                        label = "hop \(n)"
                        events.append(["frame": f, "event": "fire \(n)"])
                        phase = .hopping(n: n, elapsed: 0)
                    case .turn:
                        stand(x.y, rate: 0.8 * .pi)
                        label = "planted turn"
                        events.append(["frame": f, "event": "turn start"])
                        phase = .turning(h0: Self.heading(qb), braked: false, still: 0)
                    case .finish:
                        label = "done"
                        phase = .done(0)
                    }
                } else {
                    phase = .settle(next: next, count: c, elapsed: elapsed + 1, braking: braking)
                }
            case .hopping(let n, let elapsed):
                if r.pulseFired && v.y <= 0 && x.y < restY + 0.0005 && elapsed > 3 {
                    hopsDone = n
                    events.append(["frame": f, "event": "landed \(n)"])
                    label = "land \(n)"
                    let next: Action = n == 2 ? .turn : (n < 4 ? .hop(n + 1) : .finish)
                    phase = .settle(next: next, count: 0, elapsed: 0, braking: false)
                } else {
                    phase = .hopping(n: n, elapsed: elapsed + 1)
                }
            case .turning(let h0, let braked, let still):
                let turned = abs(Self.wrap(Self.heading(qb) - h0))
                if !braked, turned >= .pi / 2 - (om.y * om.y / (2 * alpha) + abs(om.y) / 60) {
                    s.setFoot(body: w, yawRate: 0)
                    events.append(["frame": f, "event": "turn brake"])
                    phase = .turning(h0: h0, braked: true, still: 0)
                } else if braked {
                    let st = (abs(om.y) < 0.05 && Self.tiltDeg(qb) < 1.5) ? still + 1 : 0
                    if st >= 6 {
                        turnedDeg = turned * 180 / .pi
                        s.setFoot(body: w, rest: .retracted)
                        events.append(["frame": f, "event": "turned \(turnedDeg)"])
                        label = "settle"
                        phase = .settle(next: .hop(3), count: 0, elapsed: 0, braking: false)
                    } else {
                        phase = .turning(h0: h0, braked: true, still: st)
                    }
                }
            case .done(let n):
                phase = .done(n + 1)
            }
            if case .done(let n) = phase, n > 45 { break }

            step(s, q, frames: 1)
            let x1 = s.position(of: w)!, q1 = s.orientation(of: w)!
            let r1 = s.footReading(body: w)!
            let v1 = s.velocity(of: w)!, w1 = s.angularVelocity(of: w)!
            frames.append([
                "frame": f, "t": Double(f + 1) / 60, "label": label,
                "com": [x1.x, x1.y, x1.z], "quat": [q1.imag.x, q1.imag.y, q1.imag.z, q1.real],
                "vel": [v1.x, v1.y, v1.z], "omega": [w1.x, w1.y, w1.z],
                "planted": r1.planted, "hip": [r1.hip.x, r1.hip.y, r1.hip.z], "tip": [r1.tip.x, r1.tip.y, r1.tip.z],
                "axis": [r1.axisWorld.x, r1.axisWorld.y, r1.axisWorld.z], "rest": r1.restLength,
                "retracted": spec.retractedLength,
                "compression": r1.compression, "force": r1.force,
                "active": r1.mode == .extended || (r1.mode == .pulse && !r1.pulseFired),
                "asleep": s.isAsleep(w),
            ])
        }
        let doc: [String: Any] = [
            "worker": ["r1": Self.worker.fatRadius, "r2": Self.worker.tipRadius, "c": Self.worker.centerDistance,
                       "shell": Self.worker.shellThickness, "ballastFill": Self.worker.ballastFillHeight,
                       "yFat": props.yFat, "yTip": props.yTip, "mass": props.mass],
            "foot": ["k": spec.stiffness, "c": spec.damping, "stroke": stroke, "cant": cant,
                     "muSpin": spec.spinFriction, "patch": spec.patchRadius, "yawMaxTorque": spec.yawMaxTorque],
            "solver": ["dt": s.fixedDt, "iterations": s.velocityIterations],
            "tableTop": 0.0, "frames": frames, "events": events,
        ]
        let data = try JSONSerialization.data(withJSONObject: doc, options: [.prettyPrinted])
        try data.write(to: URL(fileURLWithPath: out))
        print("FOOT_GIF wrote \(frames.count) frames, \(hopsDone) hops, turned \(turnedDeg)°, events \(events) → \(out)")
        XCTAssertEqual(hopsDone, 4, "4 hops completed")
        XCTAssertEqual(turnedDeg, 90, accuracy: 5, "the planted turn")
    }
}
