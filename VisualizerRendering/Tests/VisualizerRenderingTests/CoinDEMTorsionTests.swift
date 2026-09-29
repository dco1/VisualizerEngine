import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Engine gate for TORSIONAL (point) friction — engine plan item 5a, stage B2
/// (CoinDEMSolver+Torsion.swift, Shaders/CoinDEM.metal `cdSolveContactTorsion`) — and for its
/// consistency with the spring foot's pad row (CoinDEMFoot.h). Same runtime-compiled-library
/// seam and the Digital Clock's §G configuration as CoinDEMFootTests (1/180 s, 6 velocity
/// iterations, sleep on), the §B1 Weeble worker on the oak table. Every expectation is the
/// Coulomb law worked by hand — a spin ω about the contact normal stops after I·ω / (μ·r·m·g) —
/// not a number tuned to the GPU. Measured values are PRINTed with a `TORSION_` prefix.
@MainActor
final class CoinDEMTorsionTests: XCTestCase {

    // ── Harness ───────────────────────────────────────────────────────────────

    private static var cached: (SimEngine, MTLLibrary, MTLCommandQueue)?
    private static func shared() throws -> (SimEngine, MTLLibrary, MTLCommandQueue) {
        if let c = cached { return c }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness queue
        let url = CoinDEMFootTests.shaderURL
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("CoinDEM.metal not found") }
        let c = (SimEngine(device: device), try MetalSourceLoader.makeLibrary(device: device, contentsOf: url), queue)
        cached = c
        return c
    }

    static let g: Float = 9.81
    static var worker: CoinBallastedEgg { CoinDEMFootTests.worker }
    static var props: CoinEggMassProperties { CoinDEMFootTests.props }
    /// The worker's moment of inertia about its own axis (kg·m²).
    static var iAxis: Float { props.mass / props.invInertiaK.y }
    /// Worker (0.6) on oak (0.5): the solver's combine rule √(μA·μB).
    static let mu: Float = (0.6 * 0.5 as Float).squareRoot()

    func makeSolver(maxCoins: Int = 16, colliders: [CoinStaticCollider]? = nil,
                    feet: Bool = false) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: maxCoins,
                                    coinRadius: 0.03, halfThickness: 0.03,
                                    boundsMin: SIMD3(-0.4, -0.1, -0.4), boundsMax: SIMD3(0.4, 0.4, 0.4))
        else { throw XCTSkip("solver init failed") }
        CoinDEMFootTests.applyClockConfig(s)
        s.setColliders(colliders ?? [CoinDEMFootTests.table])
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

    /// A worker standing upright on the table at (x, z).
    @discardableResult
    func spawnWorker(_ s: CoinDEMSolver, x: Float = 0, z: Float = 0) -> Int? {
        s.spawnBallastedEgg(Self.worker, fatCenter: SIMD3(x, Self.worker.fatRadius, z),
                            friction: 0.6, restitution: 0.15)
    }

    static func heading(_ q: simd_quatf) -> Float { CoinDEMFootTests.heading(q) }
    static func wrap(_ a: Float) -> Float { atan2(sin(a), cos(a)) }

    /// Spin `w` about +Y at ω0 and step ONE substep per encode; the stop time from the samples:
    /// ω falls linearly while the patch slides (constant Coulomb moment), so the zero crossing
    /// is the last sliding sample plus its ω over the measured deceleration.
    func spinDown(_ s: CoinDEMSolver, _ q: MTLCommandQueue, worker w: Int, omega0: Float,
                  substeps: Int = 24) -> (stop: Float, samples: [Float]) {
        s.setAngularVelocity(ofSlot: w, to: SIMD3(0, omega0, 0))
        var om: [Float] = [omega0]
        step(s, q, frames: substeps, wallDt: s.fixedDt) { _ in om.append(s.angularVelocity(of: w)!.y) }
        let h = s.fixedDt
        var stop: Float = .infinity
        for k in 1..<om.count where om[k] <= 1e-4 {
            let decel = (om[max(k - 2, 0)] - om[k - 1]) / (Float(k - 1 - max(k - 2, 0)) * h)
            stop = Float(k - 1) * h + (decel > 0 ? om[k - 1] / decel : 0)
            break
        }
        return (stop, om)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 1. The spec gate: a Weeble spun at 5 rad/s stops in I·ω/(μ·r·m·g) (±20 %).
    // ══════════════════════════════════════════════════════════════════════════

    /// Plan §6 5a: a 3 mm patch stops a 5 rad/s spin in ≈ 12.6 ms (the Coulomb moment μ·r·m·g
    /// against the worker's axial inertia); a zero patch (the default) leaves it spinning — the
    /// torsion flag is off and the step is the pre-B2 point contact (actuation test #8 keeps
    /// asserting > 70 % after a second).
    func testWeebleSpinStopsInTheCoulombTime() throws {
        let omega0: Float = 5, r: Float = 0.003
        let tPred = Self.iAxis * omega0 / (Self.mu * r * Self.props.mass * Self.g)

        let (s, q) = try makeSolver()
        let w = try XCTUnwrap(spawnWorker(s))
        step(s, q, frames: 40)
        XCTAssertTrue(s.isAsleep(w), "precondition: the idle worker sleeps")
        XCTAssertFalse(s.torsionEnabled)
        s.setPatchRadius(w, r)
        XCTAssertTrue(s.torsionEnabled)
        XCTAssertEqual(s.patchRadius(of: w), r)
        let p0 = s.position(of: w)!, h0 = Self.heading(s.orientation(of: w)!)
        let run = spinDown(s, q, worker: w, omega0: omega0)
        let drift = simd_length(s.position(of: w)! - p0)
        let turned = Self.wrap(Self.heading(s.orientation(of: w)!) - h0)
        let tilt = CoinDEMFootTests.tiltDeg(s.orientation(of: w)!)

        // Control: the same spin with no patch.
        let (s0, q0) = try makeSolver()
        let w0 = try XCTUnwrap(spawnWorker(s0))
        step(s0, q0, frames: 40)
        let run0 = spinDown(s0, q0, worker: w0, omega0: omega0)

        print(String(format: "TORSION_1 r %.1f mm: stop %.3f ms vs I·ω/(μ·r·m·g) %.3f ms (%+.2f %%) | ω per substep %@ | turned %.3f° (continuous ω²I/2τ %.3f°), COM drift %.4f mm, tilt %.3f° | no patch: ω after %d substeps %.4f of %.1f",
                     r * 1000, run.stop * 1000, tPred * 1000, (run.stop / tPred - 1) * 100,
                     run.samples.prefix(5).map { String(format: "%.3f", $0) }.joined(separator: " "),
                     turned * 180 / .pi, omega0 * omega0 * Self.iAxis / (2 * Self.mu * r * Self.props.mass * Self.g) * 180 / .pi,
                     drift * 1000, tilt, run0.samples.count - 1, run0.samples.last!, omega0))
        XCTAssertEqual(run.stop, tPred, accuracy: 0.2 * tPred, "plan 5a: stops within ±20 % of I·ω/(μ·r·m·g)")
        XCTAssertEqual(run.stop, tPred, accuracy: 0.03 * tPred, "…and in fact to a few per cent (the law is exact per substep)")
        XCTAssertLessThan(abs(run.samples.last!), 1e-3, "and stays stopped")
        XCTAssertLessThan(drift, 5e-5, "the spin-down does not move the worker")
        XCTAssertLessThan(tilt, 1.0, "…or tip it")
        XCTAssertGreaterThan(run0.samples.last!, 0.99 * omega0, "control: a point contact keeps its spin")
        XCTAssertFalse(s0.torsionEnabled)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 2. The accumulated clamp: |Λs| ≤ μ·r·Λn·share on every contact of a tumbling pile.
    // ══════════════════════════════════════════════════════════════════════════

    /// Spheres, boxes, capsules and workers dropped spinning onto the table and each other,
    /// every body with a patch: after every frame each contact's accumulated torsional impulse
    /// sits inside its bound — per point for ungrouped contacts, 1/n of it on each point of an
    /// n-point grouped manifold (`manifoldSolve`), and with the warm start carrying it across
    /// substeps. And the torsion rows really run (the pile's spin is not all untouched).
    func testTorsionImpulseStaysInsideItsBound() throws {
        for manifold in [false, true] {
            let (s, q) = try makeSolver(maxCoins: 32)
            s.manifoldSolve = manifold
            s.warmStart = true
            var bodies: [Int] = []
            for i in 0..<16 {
                let x = Float(i % 4) * 0.012 - 0.018, z = Float(i / 4) * 0.012 - 0.018
                let p = SIMD3<Float>(x, 0.01 + Float(i % 3) * 0.008, z)
                let slot: Int?
                switch i % 4 {
                case 0: slot = s.spawnSphere(at: p, radius: 0.004, mass: 0.004, friction: 0.5, restitution: 0.1)
                case 1: slot = s.spawnBox(at: p, halfExtents: SIMD3(0.004, 0.003, 0.005), mass: 0.004, friction: 0.5, restitution: 0.1)
                case 2: slot = s.spawnCapsule(at: p, radius: 0.0025, halfLength: 0.004, mass: 0.003, friction: 0.5, restitution: 0.1)
                default: slot = s.spawnBallastedEgg(Self.worker, fatCenter: p + SIMD3(0, 0.01, 0), friction: 0.6, restitution: 0.15)
                }
                let b = try XCTUnwrap(slot)
                s.setPatchRadius(b, 0.001 + 0.0005 * Float(i % 3))
                s.setAngularVelocity(ofSlot: b, to: SIMD3(0, 6 - Float(i % 5), 2))
                bodies.append(b)
            }
            // The law each contact must have been solved with: μ = √(μA·μB) (the table 0.5),
            // r = max(rA, rB), and a share of 1 (ungrouped) or 1/n (a grouped n-point manifold).
            let mat = s.materialBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: s.maxCoins)
            var rows = 0, loaded = 0, worstRatio: Float = 0, grouped = 0, badShare = 0
            let cptr = s.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: s.maxContacts)
            step(s, q, frames: 90) { _ in
                for i in 0..<min(s.contactCount, s.maxContacts) {
                    let c = cptr[i]
                    guard c.ext.y > 0 else { continue }
                    rows += 1
                    let bound = c.ext.y * max(c.rA.w, 0)
                    if bound > 0 {
                        loaded += 1
                        worstRatio = max(worstRatio, abs(c.ext.x) / bound)
                    } else {
                        XCTAssertEqual(c.ext.x, 0, "an unloaded contact holds no torsion")
                    }
                    let a = Int(c.meta.x), bStatic = c.meta.y == 0xFFFF_FFFF
                    let muB: Float = bStatic ? 0.5 : mat[Int(c.meta.y)].x
                    let rP = max(s.patchRadius(of: a), bStatic ? 0 : s.patchRadius(of: Int(c.meta.y)))
                    let share = c.ext.y / ((mat[a].x * muB).squareRoot() * rP)
                    if [Float(1), 0.5, 1.0 / 3, 0.25].allSatisfy({ abs(share - $0) > 1e-4 }) { badShare += 1 }
                    if share < 0.99 { grouped += 1 }
                }
            }
            print(String(format: "TORSION_2 manifold=%@: torsion rows %d (loaded %d, 1/n share %d, off-law %d), worst |Λs|/(μ·r·Λn·share) %.6f",
                         "\(manifold)", rows, loaded, grouped, badShare, worstRatio))
            XCTAssertGreaterThan(loaded, 100, "the torsion rows ran on loaded contacts")
            XCTAssertLessThanOrEqual(worstRatio, 1.0001, "|Λs| ≤ μ·r·Λn·share on every contact")
            XCTAssertEqual(badShare, 0, "every row used μ·r·share with share 1 or 1/n")
            if !manifold { XCTAssertEqual(grouped, 0, "ungrouped contacts each carry the full μ·r") }
            if manifold { XCTAssertGreaterThan(grouped, 0, "grouped manifold points carry a 1/n share") }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 3. Opt-in per body: a patch on one body leaves every other body bit-identical.
    // ══════════════════════════════════════════════════════════════════════════

    /// The torsion flag switches on for the whole solver when any body has a patch, but a
    /// contact with no patch on either side and no carried torsion returns before touching
    /// anything: a pile far from the one patched worker steps bit for bit as without it.
    func testPatchOnOneBodyLeavesTheRestBitIdentical() throws {
        func run(patch: Bool) throws -> [SIMD4<Float>] {
            let (s, q) = try makeSolver(maxCoins: 32)
            s.warmStart = true
            var pile: [Int] = []
            for i in 0..<12 {
                let p = SIMD3<Float>(Float(i % 3) * 0.009 - 0.009, 0.006 + Float(i / 3) * 0.009, Float(i % 2) * 0.004)
                let b = try XCTUnwrap(i % 2 == 0
                    ? s.spawnBox(at: p, halfExtents: SIMD3(0.004, 0.004, 0.004), mass: 0.004, friction: 0.5, restitution: 0.1)
                    : s.spawnSphere(at: p, radius: 0.004, mass: 0.004, friction: 0.5, restitution: 0.1))
                s.setAngularVelocity(ofSlot: b, to: SIMD3(1, 3, -2))
                pile.append(b)
            }
            let w = try XCTUnwrap(spawnWorker(s, x: 0.2, z: 0.1))
            if patch { s.setPatchRadius(w, 0.003) }
            s.setAngularVelocity(ofSlot: w, to: SIMD3(0, 4, 0))
            step(s, q, frames: 60)
            XCTAssertEqual(s.torsionEnabled, patch)
            return pile.flatMap { b -> [SIMD4<Float>] in
                let p = s.position(of: b)!, o = s.orientation(of: b)!
                return [SIMD4(p, 0), SIMD4(o.imag, o.real), SIMD4(s.velocity(of: b)!, 0), SIMD4(s.angularVelocity(of: b)!, 0)]
            }
        }
        let a = try run(patch: false), b = try run(patch: true)
        let worst = zip(a, b).map { simd_length($0 - $1) }.max()!
        print("TORSION_3 pile beside a patched worker vs no patch anywhere: worst |Δ| over 12 bodies' states = \(worst)")
        XCTAssertEqual(worst, 0, "bit-identical")
    }

    /// Slot bookkeeping: a despawned (or culled, or cleared) body's patch goes with it, a dead
    /// slot refuses one, and the body spawned into a reused slot is a point contact — so the
    /// torsion flag (and its per-substep cost) never outlives the bodies that asked for it.
    func testPatchNeverOutlivesItsBody() throws {
        let (s, _) = try makeSolver()
        let w = try XCTUnwrap(spawnWorker(s))
        s.setPatchRadius(w, 0.003)
        XCTAssertTrue(s.torsionEnabled)
        s.despawn(w)
        XCTAssertEqual(s.patchRadius(of: w), 0, "despawn clears the patch")
        XCTAssertFalse(s.torsionEnabled, "…and the flag with it")
        s.setPatchRadius(w, 0.003)
        XCTAssertEqual(s.patchRadius(of: w), 0, "a dead slot takes no patch")
        XCTAssertFalse(s.torsionEnabled)
        let b = try XCTUnwrap(s.spawnSphere(at: SIMD3(0, 0.01, 0), radius: 0.004, mass: 0.004))
        XCTAssertEqual(b, w, "precondition: the slot was reused")
        XCTAssertEqual(s.patchRadius(of: b), 0, "the new body is a point contact")
        s.setPatchRadius(b, 0.002)
        XCTAssertEqual(s.cull(belowY: 1), 1)
        XCTAssertFalse(s.torsionEnabled, "cull clears it")
        let c = try XCTUnwrap(spawnWorker(s))
        s.setPatchRadius(c, 0.002)
        s.clearAll()
        XCTAssertFalse(s.torsionEnabled, "clearAll clears it")
        XCTAssertEqual(s.patchRadius(of: c), 0)
    }

    /// Clearing the LAST patch mid-slide, warm start on (stage-B2 verifier): the contact is a point
    /// contact again, so the sphere keeps the spin it had. The torsion flag drops with the patch,
    /// so the row that clamps / hands back a carried torsion impulse no longer runs — the warm
    /// start must stop carrying it too. Before the fix the carried μ·r·Λn (0.373 rad/s per substep
    /// on this 10 mm sphere) was re-applied every substep and never re-solved: ω 3.50 → −7.68 rad/s
    /// in 30 substeps, spinning up without end. Controls: another body keeping its patch (flag
    /// stays set, the row hands the impulse back), and warm start off.
    func testClearingTheLastPatchLeavesNoStaleTorsion() throws {
        let R: Float = 0.01, r: Float = 0.0005, omega0: Float = 5
        for (warm, keepOther) in [(true, false), (true, true), (false, false)] {
            let (s, q) = try makeSolver()
            s.warmStart = warm
            let b = try XCTUnwrap(s.spawnSphere(at: SIMD3(0, R, 0), radius: R, mass: 0.01, friction: 0.6, restitution: 0.1))
            let other = try XCTUnwrap(s.spawnSphere(at: SIMD3(0.15, R, 0.1), radius: R, mass: 0.01, friction: 0.6, restitution: 0.1))
            step(s, q, frames: 40)
            s.setPatchRadius(b, r)
            if keepOther { s.setPatchRadius(other, r) }
            s.setAngularVelocity(ofSlot: b, to: SIMD3(0, omega0, 0))
            step(s, q, frames: 4, wallDt: s.fixedDt)
            let wClear = s.angularVelocity(of: b)!.y
            XCTAssertLessThan(wClear, 0.8 * omega0, "precondition: the patch was braking the spin")
            s.setPatchRadius(b, 0)
            XCTAssertEqual(s.torsionEnabled, keepOther)
            step(s, q, frames: 30, wallDt: s.fixedDt)
            let wEnd = s.angularVelocity(of: b)!.y
            print(String(format: "TORSION_CLEAR warm %d keepOther %d: ω at clear %.3f → 30 substeps later %.3f rad/s",
                         warm ? 1 : 0, keepOther ? 1 : 0, wClear, wEnd))
            XCTAssertEqual(wEnd, wClear, accuracy: 0.02 * wClear, "a point contact keeps its spin once the patch is gone")
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 4. With the spring foot: each patch counted once, and the yaw motor vs shell friction.
    // ══════════════════════════════════════════════════════════════════════════

    /// A worker standing on its shell AND a preloaded pad (the pad carrying ~⅓ of its weight),
    /// both patches 3 mm with the same μ, the pad's yaw motor braking (target 0, torque
    /// unlimited): spun at 5 rad/s it stops in exactly the single-contact time I·ω/(μ·r·m·g) —
    /// the pad's row bounded by the leg's normal impulse, the shell's by the shell's, their sum
    /// the weight. Counting the pad's share twice (or the shell's contact at the full weight)
    /// would stop it ≈ ⅓ sooner.
    func testFootAndShellTorsionCountEachPatchOnce() throws {
        let omega0: Float = 5, r: Float = 0.003
        let (s, q) = try makeSolver(feet: true)
        let w = try XCTUnwrap(spawnWorker(s))
        var spec = CoinFootSpec.weebleWorker()
        spec.spinFriction = Self.mu          // the pad's μ_spin = the shell contact's μ
        spec.patchRadius = r
        spec.yawMaxTorque = 1                // the motor never limits the brake: friction does
        XCTAssertTrue(s.addFoot(body: w, spec: spec))
        s.setPatchRadius(w, r)
        step(s, q, frames: 40)
        let hip = s.position(of: w)!.y       // hip at the COM, table top at 0
        let preload: Float = 0.0002          // k·δ ≈ 0.046 N ≈ 32 % of the weight (stable stance: < 42 %)
        s.setFoot(body: w, rest: .extended, extendedLength: hip + preload, cant: 0, yawRate: 0)
        step(s, q, frames: 30)
        let legForce = try XCTUnwrap(s.footReading(body: w)).force
        let weight = Self.props.mass * Self.g
        let run = spinDown(s, q, worker: w, omega0: omega0)
        let reading = try XCTUnwrap(s.footReading(body: w))
        let tOnce = Self.iAxis * omega0 / (Self.mu * r * weight)
        let tDouble = Self.iAxis * omega0 / (Self.mu * r * (weight + legForce))
        print(String(format: "TORSION_4 pad carries %.4f N of %.4f N (%.0f %%): stop %.3f ms vs once-counted %.3f ms (%+.2f %%), pad counted twice %.3f ms | pad torsion total %.3e N·m·s | ω %@",
                     legForce, weight, legForce / weight * 100, run.stop * 1000, tOnce * 1000, (run.stop / tOnce - 1) * 100,
                     tDouble * 1000, reading.torsionTotal,
                     run.samples.prefix(5).map { String(format: "%.3f", $0) }.joined(separator: " ")))
        XCTAssertTrue(reading.planted, "the pad stayed planted")
        XCTAssertGreaterThan(legForce, 0.25 * weight, "the pad carries a real share (the control is meaningful)")
        XCTAssertLessThan(legForce, 0.42 * weight, "…inside the stable stance")
        XCTAssertEqual(run.stop, tOnce, accuracy: 0.05 * tOnce, "shell + pad torsion = μ·r·(weight), counted once")
        XCTAssertGreaterThan(abs(run.stop - tDouble), 0.15 * tOnce, "clearly not the double-counted time")
        XCTAssertLessThan(reading.torsionTotal, 0, "the pad's own row took part in the brake")
    }

    /// The scene's planted turn (0.15 mm preload, pad r 2 mm / μ_spin 0.5, 6e-5 N·m yaw motor,
    /// π/2 rad/s) against the shell's own patch. The pad grips μ_spin·r·N_leg ≈ 3.5e-5 N·m:
    /// with a 3 mm shell patch (μ·r·N_shell ≈ 1.8e-4 N·m) the motor must STALL — the rows
    /// converge together; the pad solved once after the contacts (the pre-B2 schedule) turned
    /// the worker anyway at τ_pad·dt/I per substep. With a 0.1 mm shell patch (a hard ABS shell's
    /// Hertz contact is ≈ 0.08 mm) the motor wins and the worker turns at its commanded rate.
    func testYawMotorAgainstShellPatchStallsOrTurnsByTheLaw() throws {
        func turn(shellPatch: Float) throws -> (deg: Float, rateEnd: Float, legForce: Float) {
            let (s, q) = try makeSolver(feet: true)
            let w = try XCTUnwrap(spawnWorker(s))
            var spec = CoinFootSpec.weebleWorker()
            spec.yawMaxTorque = 6e-5
            XCTAssertTrue(s.addFoot(body: w, spec: spec))
            s.setPatchRadius(w, shellPatch)
            step(s, q, frames: 40)
            let hip = s.position(of: w)!.y
            s.setFoot(body: w, rest: .extended, extendedLength: hip + 0.00015, cant: 0, yawRate: 0)
            step(s, q, frames: 20)
            let h0 = Self.heading(s.orientation(of: w)!)
            s.setFoot(body: w, yawRate: .pi / 2)
            var legForce: Float = 0
            step(s, q, frames: 60) { _ in legForce = max(legForce, s.footReading(body: w)!.force) }
            let turned = Self.wrap(Self.heading(s.orientation(of: w)!) - h0)
            return (turned * 180 / .pi, s.angularVelocity(of: w)!.y, legForce)
        }
        let stall = try turn(shellPatch: 0.003)
        let hertz = try turn(shellPatch: 0.0001)
        let point = try turn(shellPatch: 0)
        print(String(format: "TORSION_5 yaw motor π/2 rad/s for 1 s: shell patch 3 mm → turned %.3f° (ω %.4f); 0.1 mm → %.2f° (ω %.3f); no patch → %.2f° (ω %.3f) | leg force %.4f N",
                     stall.deg, stall.rateEnd, hertz.deg, hertz.rateEnd, point.deg, point.rateEnd, stall.legForce))
        XCTAssertLessThan(abs(stall.deg), 0.5, "a motor weaker than the shell's patch friction stalls")
        XCTAssertEqual(hertz.rateEnd, .pi / 2, accuracy: 0.05 * .pi / 2, "a hard shell's tiny patch: the motor turns it")
        XCTAssertGreaterThan(hertz.deg, 80, "…through ≈ 90° in the second")
        XCTAssertEqual(point.rateEnd, .pi / 2, accuracy: 0.05 * .pi / 2, "no patch: the scene's turn as before")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 6. Cost: the torsion rows, and planted feet solved every iteration.
    // ══════════════════════════════════════════════════════════════════════════

    /// Four workers standing on the table, sleep OFF so nothing skips, in four otherwise identical
    /// worlds stepped in interleaved blocks: (a) point contacts, (b) a 3 mm patch on each worker
    /// (a torsion row per contact), (c) each worker also planted on a stance foot with no patch
    /// (the pad rows solved once after the contacts), (d) feet AND patches (the pad rows solved
    /// every velocity iteration, CoinDEMSolver+Foot.swift). The per-substep deltas are the price
    /// of stage B2's rows; the bound is loose (a shared GPU) — the numbers are the report.
    func testTorsionAndFootRowCost() throws {
        func world(patch: Bool, feet: Bool) throws -> (CoinDEMSolver, MTLCommandQueue) {
            let (s, q) = try makeSolver(feet: feet)
            s.sleepEnabled = false
            var ws: [Int] = []
            for i in 0..<4 {
                let w = try XCTUnwrap(spawnWorker(s, x: Float(i) * 0.05 - 0.075))
                if feet { XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker())) }
                if patch { s.setPatchRadius(w, 0.003) }
                ws.append(w)
            }
            step(s, q, frames: 30)
            if feet {
                for w in ws {
                    s.setFoot(body: w, rest: .extended, extendedLength: s.position(of: w)!.y + 0.00015, cant: 0, yawRate: 0)
                }
            }
            step(s, q, frames: 30)
            if feet { XCTAssertEqual(ws.filter { s.footReading(body: $0)!.planted }.count, 4) }
            return (s, q)
        }
        let worlds = try [(false, false), (true, false), (false, true), (true, true)].map { try world(patch: $0.0, feet: $0.1) }
        var ms = [[Double]](repeating: [], count: 4)
        for _ in 0..<6 {                                                   // interleaved blocks
            for (i, w) in worlds.enumerated() { ms[i] += step(w.0, w.1, frames: 40) }
        }
        func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
        let substeps = Double(worlds[0].0.lastStepCount)
        let m = ms.map(median)
        let perSub = { (a: Int, b: Int) in (m[a] - m[b]) / substeps * 1000 }
        print(String(format: "TORSION_6 4 workers, frame p50: point %.3f ms, patch %.3f, feet %.3f, feet+patch %.3f → torsion rows %+.1f µs/substep, feet (once after) %+.1f, feet every iteration + torsion %+.1f over feet alone (%d substeps/frame)",
                     m[0], m[1], m[2], m[3], perSub(1, 0), perSub(2, 0), perSub(3, 2), Int(substeps)))
        XCTAssertLessThan(perSub(1, 0), 250, "torsion rows on 4 workers cost well under a quarter millisecond per substep")
        XCTAssertLessThan(perSub(3, 2), 250, "…and so does solving planted pads every iteration")
    }
}
