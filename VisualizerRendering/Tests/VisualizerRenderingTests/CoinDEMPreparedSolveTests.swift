import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Stage B3: the OPT-IN prepared contact rows (`CoinDEMSolver.preparedContactSolve`,
/// Shaders/CoinDEMPreparedSolve.h). The prepare pass computes each contact's pose-constant factors
/// (effective masses, I⁻¹(r×d)) once per substep; the solve reads them. The rows, bounds, targets
/// and order are the default solve's, so the physics must be the same — only Metal's fast-math
/// rounding of the regrouped arithmetic differs. Each test steps one scenario both ways and holds
/// the two trajectories together frame by frame (non-chaotic scenarios, so a rounding difference
/// stays a rounding difference), and checks the physical law where there is one.
/// (Bit-parity of the two PATHS with the rows prepared: CoinDEMSmallWorldTests.)
@MainActor
final class CoinDEMPreparedSolveTests: XCTestCase {

    private static var cached: (SimEngine, MTLLibrary, MTLCommandQueue)?
    private static func shared() throws -> (SimEngine, MTLLibrary, MTLCommandQueue) {
        if let c = cached { return c }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness queue
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        let c = (SimEngine(device: device), try MetalSourceLoader.makeLibrary(device: device, contentsOf: shader), queue)
        cached = c
        return c
    }

    private func makeSolver(prepared: Bool, manifold: Bool = false, maxCoins: Int = 16) throws -> CoinDEMSolver {
        let (engine, lib, _) = try Self.shared()
        let s = try XCTUnwrap(CoinDEMSolver(engine: engine, library: lib, maxCoins: maxCoins, coinRadius: 0.03, halfThickness: 0.03,
                                            boundsMin: SIMD3(-0.4, -0.1, -0.4), boundsMax: SIMD3(0.4, 0.4, 0.4)))
        s.solverMode = .constraint
        s.gravity = 9.81; s.fixedDt = 1.0 / 180; s.maxSubsteps = 10; s.velocityIterations = 6
        s.colorRounds = 16; s.frictionCoeff = 0.5; s.restitution = 0.0; s.restThreshold = 0.14
        s.contactSlop = 1e-4; s.baumgarteBeta = 0.2; s.speculativeMargin = 2e-4
        s.linDamping = 1; s.angDamping = 1; s.maxSpeed = 5; s.maxHSpeed = 5; s.maxOmega = 60; s.floorY = -0.5
        s.sleepEnabled = false
        s.manifoldSolve = manifold; s.warmStart = manifold
        s.preparedContactSolve = prepared
        s.smallWorldPath = false
        s.setColliders([.box(center: SIMD3(0, -0.01, 0), halfExtents: SIMD3(0.35, 0.01, 0.35), friction: 0.5, restitution: 0.0)])
        return s
    }

    private func frame(_ s: CoinDEMSolver) throws {
        let (_, _, q) = try Self.shared()
        let cb = try XCTUnwrap(q.makeCommandBuffer())
        s.encode(to: cb, wallDt: 1.0 / 30)
        cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
        XCTAssertNotEqual(cb.status, .error, "command buffer error: \(String(describing: cb.error))")
    }

    /// Step the same scene with the rows prepared and not, `frames` frames; `body` is read after
    /// each. Returns the largest position / velocity / spin gap between the two runs.
    private func twin(frames: Int, manifold: Bool = false, setup: (CoinDEMSolver) throws -> [Int],
                      each: ((Int, CoinBodyState, CoinBodyState) -> Void)? = nil) throws -> (dx: Float, dv: Float, dw: Float) {
        let a = try makeSolver(prepared: false, manifold: manifold), b = try makeSolver(prepared: true, manifold: manifold)
        let sa = try setup(a), sb = try setup(b)
        var dx: Float = 0, dv: Float = 0, dw: Float = 0
        var ra: [CoinBodyState] = [], rb: [CoinBodyState] = []
        for f in 0..<frames {
            try frame(a); try frame(b)
            a.readStates(sa, into: &ra); b.readStates(sb, into: &rb)
            for (x, y) in zip(ra, rb) {
                dx = max(dx, simd_length(x.position - y.position))
                dv = max(dv, simd_length(x.velocity - y.velocity))
                dw = max(dw, simd_length(x.angularVelocity - y.angularVelocity))
            }
            each?(f, ra[0], rb[0])
        }
        XCTAssertTrue(b.preparedContactsLive, "the prepared rows ran")
        XCTAssertFalse(a.preparedContactsLive)
        return (dx, dv, dw)
    }

    /// Coulomb sliding: a box launched at 0.6 m/s stops after v²/(2μg) − v·dt/2 (μ = 0.5 combined;
    /// the second term is the semi-implicit Euler step's: each substep moves at the velocity AFTER
    /// that substep's friction impulse).
    /// The prepared rows' buffer: the host allocates `contactPrepStride` bytes per contact and the
    /// kernels index CD_PREP_F4 float4 per contact (Shaders/CoinDEMPreparedSolve.h). A mismatch would
    /// have every prepare write past its row (and the last ones past the buffer), silently.
    func testPreparedRowStrideMatchesTheKernel() throws {
        let header = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEMPreparedSolve.h")
        let text = try String(contentsOf: header, encoding: .utf8)
        let m = try XCTUnwrap(text.firstMatch(of: /constant uint CD_PREP_F4 = (\d+)u;/), "CD_PREP_F4 not found")
        let f4 = try XCTUnwrap(Int(m.1))
        XCTAssertEqual(f4 * 16, CoinDEMSolver.contactPrepStride, "CD_PREP_F4 float4 per contact == CoinDEMSolver.contactPrepStride bytes")
    }

    func testSlidingBoxStopsWhereCoulombSays() throws {
        var stopA: Float = 0, stopB: Float = 0
        let (dx, dv, _) = try twin(frames: 30, setup: { s in
            let b = try XCTUnwrap(s.spawnBox(at: SIMD3(-0.2, 0.004, 0), halfExtents: SIMD3(0.006, 0.004, 0.006), mass: 0.01,
                                             friction: 0.5, restitution: 0.0))
            s.setRigidVelocity(slots: [b], linear: SIMD3(0.6, 0, 0), angular: .zero, about: s.position(of: b)!)
            return [b]
        }, each: { _, x, y in stopA = x.position.x; stopB = y.position.x })
        let expected: Float = 0.6 * 0.6 / (2 * 0.5 * 9.81) - 0.6 * (1.0 / 180) / 2
        print("PREP_slide stop default \((stopA + 0.2) * 1000) mm, prepared \((stopB + 0.2) * 1000) mm (discrete Coulomb \(expected * 1000) mm); max gap x \(dx * 1e6) µm v \(dv * 1e3) mm/s")
        XCTAssertEqual(stopB + 0.2, expected, accuracy: 0.01 * expected, "prepared: stops where discrete Coulomb friction says")
        XCTAssertEqual(stopA + 0.2, expected, accuracy: 0.01 * expected, "default: the same")
        XCTAssertLessThan(dx, 2e-6, "the two runs agree to rounding")
        XCTAssertLessThan(dv, 1e-4)
    }

    /// Rolling resistance, the legacy and the accumulated row: a rolling sphere decelerates the same.
    func testRollingSphereDeceleratesTheSame() throws {
        for accumulated in [false, true] {
            let (dx, dv, dw) = try twin(frames: 45, setup: { s in
                s.rollingResistance = 0.02
                s.accumulatedRollingResistance = accumulated
                let b = try XCTUnwrap(s.spawnSphere(at: SIMD3(-0.25, 0.01, 0), radius: 0.01, mass: 0.01))
                s.setRigidVelocity(slots: [b], linear: SIMD3(0.5, 0, 0), angular: SIMD3(0, 0, -50), about: s.position(of: b)!)
                return [b]
            })
            print("PREP_roll accumulated=\(accumulated) max gap x \(dx * 1e6) µm v \(dv * 1e3) mm/s ω \(dw) rad/s")
            XCTAssertLessThan(dx, 5e-6, "accumulated=\(accumulated): positions agree to rounding")
            XCTAssertLessThan(dv, 2e-4)
            XCTAssertLessThan(dw, 2e-2)
        }
    }

    /// Torsion (CD_FLAG_TORSION): a patched sphere spun at 5 rad/s about the vertical stops in
    /// I·ω/(μ·r·m·g), the same both ways.
    func testTorsionSpinDownIsTheSame() throws {
        var stopFrameA = -1, stopFrameB = -1
        let (_, _, dw) = try twin(frames: 40, setup: { s in
            let b = try XCTUnwrap(s.spawnSphere(at: SIMD3(0, 0.01, 0), radius: 0.01, mass: 0.01))
            s.setPatchRadius(b, 0.0005)
            s.setRigidVelocity(slots: [b], linear: .zero, angular: SIMD3(0, 5, 0), about: s.position(of: b)!)
            return [b]
        }, each: { f, x, y in
            if stopFrameA < 0 && abs(x.angularVelocity.y) < 1e-3 { stopFrameA = f }
            if stopFrameB < 0 && abs(y.angularVelocity.y) < 1e-3 { stopFrameB = f }
        })
        let I: Float = 0.4 * 0.01 * 0.01 * 0.01, tStop = I * 5 / (0.5 * 0.0005 * 0.01 * 9.81)
        print("PREP_torsion stop frame default \(stopFrameA) prepared \(stopFrameB) (predicted \(tStop * 30) frames); max ω gap \(dw) rad/s")
        XCTAssertEqual(stopFrameA, stopFrameB, "the spin stops on the same frame")
        XCTAssertEqual(Float(stopFrameB), tStop * 30, accuracy: 2, "at I·ω/(μ·r·m·g)")
        XCTAssertLessThan(dw, 1e-3)
    }

    /// Restitution: a sphere dropped from 5 cm with e = 0.5 bounces to the same apex.
    func testBounceApexIsTheSame() throws {
        var apexA: Float = 0, apexB: Float = 0
        let (dx, _, _) = try twin(frames: 30, setup: { s in
            s.restitution = 0.5
            let b = try XCTUnwrap(s.spawnSphere(at: SIMD3(0, 0.06, 0), radius: 0.01, mass: 0.01, friction: 0.5, restitution: 0.5))
            return [b]
        }, each: { f, x, y in
            if f > 4 { apexA = max(apexA, x.position.y); apexB = max(apexB, y.position.y) }
        })
        print("PREP_bounce apex default \(apexA * 1000) mm prepared \(apexB * 1000) mm; max gap \(dx * 1e6) µm")
        XCTAssertEqual(apexA, apexB, accuracy: 1e-5)
        XCTAssertLessThan(dx, 1e-5)
    }

    /// Grouped manifolds (manifold solve + warm start): a 3-box stack stands 3 s without creep
    /// either way, and the two stacks agree.
    func testManifoldStackRestsTheSame() throws {
        var driftB: Float = 0
        var start: [SIMD3<Float>] = []
        let (dx, _, _) = try twin(frames: 90, manifold: true, setup: { s in
            var slots: [Int] = []
            for i in 0..<3 {
                slots.append(try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.005 + Float(i) * 0.0101, 0), halfExtents: SIMD3(0.006, 0.005, 0.006),
                                                      mass: 0.005, friction: 0.5, restitution: 0.0)))
            }
            if start.isEmpty { start = slots.map { s.position(of: $0)! } }
            return slots
        }, each: { f, _, y in if f == 0 { driftB = 0 }; driftB = max(driftB, simd_length(y.position - start[0])) })
        print("PREP_stack bottom-box drift prepared \(driftB * 1e6) µm; max gap between runs \(dx * 1e6) µm")
        XCTAssertLessThan(driftB, 2e-4, "the stack stands (bottom box within the settle of its first frames)")
        XCTAssertLessThan(dx, 2e-6, "and matches the default solve")
    }

    /// The rows between two DYNAMIC bodies — where the prepare's B-side factors are read (I_B⁻¹(r_B×d)
    /// for the normal / friction rows, I_B⁻¹n for torsion, I_B⁻¹t₁ / I_B⁻¹t₂ for the accumulated
    /// rolling row). Every other test here presses a body on the static floor, where those factors
    /// are 0 and a wrong one would pass unseen (verifier, stage B3: pointing the torsion row's B
    /// response at A's factor, the accumulated rolling row's at A's, or the second friction row's at
    /// the first's, each passed every test above). A patched sphere spinning and rolling on a dynamic
    /// box that rests on a frictionless floor: the box takes the reaction of each row (it spins up to
    /// ≈ 0.26 rad/s, as I_ball·ω₀ / I_box predicts), and both bodies must step the same either way.
    func testDynamicPairRowsAreTheSame() throws {
        for accumulated in [false, true] {
            var boxYawA: Float = 0, boxYawB: Float = 0
            let (dx, dv, dw) = try twin(frames: 40, setup: { s in
                s.rollingResistance = 0.05
                s.accumulatedRollingResistance = accumulated
                s.sleepLinVel = 0     // no dead-stop: the box's slow reaction spin must survive finalize
                // A frictionless floor, so the box turns freely under the ball's rows (on a gripping one its
                // static friction holds it: the reaction spin stays ≈ 1e-3 rad/s).
                s.setColliders([.box(center: SIMD3(0, -0.01, 0), halfExtents: SIMD3(0.35, 0.01, 0.35), friction: 0.0, restitution: 0.0)])
                let box = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.01, 0), halfExtents: SIMD3(0.03, 0.01, 0.03), mass: 0.05,
                                                   friction: 0.5, restitution: 0.0))
                let ball = try XCTUnwrap(s.spawnSphere(at: SIMD3(0, 0.03, 0), radius: 0.01, mass: 0.01, friction: 0.5, restitution: 0.0))
                s.setPatchRadius(ball, 0.004)
                s.setRigidVelocity(slots: [ball], linear: SIMD3(0.05, 0, 0), angular: SIMD3(0, 20, -5), about: s.position(of: ball)!)
                return [box, ball]
            }, each: { _, x, y in boxYawA = max(boxYawA, abs(x.angularVelocity.y)); boxYawB = max(boxYawB, abs(y.angularVelocity.y)) })
            print("PREP_pair accumulated=\(accumulated) box peak yaw rate default \(boxYawA) prepared \(boxYawB) rad/s; max gap x \(dx * 1e6) µm v \(dv * 1e3) mm/s ω \(dw) rad/s")
            XCTAssertGreaterThan(boxYawA, 1e-2, "precondition: the ball's torsion row turns the box (a B-side response is exercised)")
            // Rounding-level gaps (measured ≈ 1e-8 m, 2e-7 m/s, 2e-5 rad/s); a wrong B-side factor
            // moves the box by orders of magnitude more.
            XCTAssertLessThan(dx, 5e-7, "accumulated=\(accumulated): positions agree to rounding")
            XCTAssertLessThan(dv, 2e-5)
            XCTAssertLessThan(dw, 1e-3)
        }
    }
}
