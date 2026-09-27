import Metal
import XCTest
import simd
@testable import VisualizerRendering

/// `MLSMPMSolver.vessel` — liquid simulated against a container of revolution (the grid's and
/// the particles' separating wall/floor condition + the G2P projection), in the vessel's own
/// frame, with the long-run volume options a carried vessel needs (`volumeCorrection`,
/// `noTension`). The physics is checked against the linear sloshing theory it must reproduce:
/// the first antisymmetric mode of an upright cylinder of radius R filled to depth h rings at
/// ω² = (g·ξ/R)·tanh(ξ·h/R), ξ = 1.8412 (the first zero of J₁′) — Ibrahim, *Liquid Sloshing
/// Dynamics* (2005), §2.
@MainActor
final class MLSMPMVesselTests: XCTestCase {

    private let R: Float = 0.55
    private let depth: Float = 0.6

    /// The coffee-pot tuning (Vintage Diner Ultra): 6 cm nodes, 8 particles a node, an EOS stiff
    /// enough that sound (≈ 7.5 m/s) outruns the slosh waves (≈ 1.7 m/s) four times over, steps of
    /// 1/300 s (CFL ≈ 0.45), no viscosity, water's EOS (no tension), J held to the particles'
    /// real crowding.
    private func makeSolver() throws -> (MLSMPMSolver, Int) {
        let solver = try XCTUnwrap(MLSMPMSolver(engine: .shared, boundsMin: SIMD3(-0.72, -0.1, -0.72),
                                                boundsMax: SIMD3(0.72, 1.3, 0.72), cellSize: 0.06,
                                                maxParticles: 40_000))
        solver.vessel = MLSMPMSolver.Vessel(floorY: 0, profile: [SIMD2(0, R), SIMD2(1.2, R)], friction: 0)
        let n = solver.seedVessel(level: depth)
        solver.bulkModulus = 150
        solver.gamma = 3
        solver.viscosity = 0
        solver.volumeCorrection = 0.1
        solver.noTension = true
        solver.settleDamping = 0.97
        return (solver, n)
    }

    /// One frame (1/60 s in 5 steps) under effective gravity `g`; returns its GPU ms.
    @discardableResult
    private func frame(_ solver: MLSMPMSolver, _ g: SIMD3<Float> = SIMD3(0, -9.81, 0)) throws -> Double {
        let l = simd_length(g)
        solver.gravity = l
        solver.gravityDirection = g / l
        solver.substepsPerFrame = 5
        let cb = try XCTUnwrap(SimEngine.shared.commandQueue.makeCommandBuffer())
        solver.encode(to: cb, wallDt: 1.0 / 60)
        cb.commit()
        cb.waitUntilCompleted()   // gpu-ok: test-time readback
        XCTAssertNil(cb.error)
        return (cb.gpuEndTime - cb.gpuStartTime) * 1000
    }

    private func positions(_ solver: MLSMPMSolver) -> [SIMD3<Float>] {
        let p = solver.particleBuffer.contents
        return (0 ..< solver.particleBuffer.count).map {
            SIMD3(p[$0].positionMass.x, p[$0].positionMass.y, p[$0].positionMass.z)
        }
    }

    private func centreHeight(_ solver: MLSMPMSolver) -> Float {
        let ps = positions(solver)
        return ps.reduce(0) { $0 + $1.y } / Float(ps.count)
    }

    private func outside(_ solver: MLSMPMSolver) -> Int {
        positions(solver).filter { simd_length(SIMD2($0.x, $0.z)) > R + 2e-3 || $0.y < -2e-3 }.count
    }

    /// A walk's accelerations (m/s²): sway at the stride, surge and heel-strike bob at twice it.
    private func walk(_ t: Float) -> SIMD3<Float> {
        let w = 2 * Float.pi * 0.65
        return SIMD3(1.2 * sin(w * t), 1.5 * sin(2 * w * t + 0.4), 0.8 * sin(2 * w * t))
    }

    func testLiquidStaysInTheVesselAndRestsNearItsLevel() throws {
        let (solver, seeded) = try makeSolver()
        XCTAssertGreaterThan(seeded, 15_000)
        var gpu: [Double] = []
        for _ in 0 ..< 120 { gpu.append(try frame(solver)) }
        XCTAssertEqual(outside(solver), 0, "particles outside the vessel")
        let core = positions(solver).filter { simd_length(SIMD2($0.x, $0.z)) < 0.35 }.map(\.y).sorted()
        let top = core[Int(Float(core.count) * 0.98)]
        print("MLSMPMVessel: \(seeded) particles, rest top \(top) (seed \(depth)), GPU \(gpu.suffix(30).reduce(0, +) / 30) ms/frame (5 substeps)")
        XCTAssertEqual(top, depth * 0.93, accuracy: 0.04, "rest level")
    }

    /// The floor layer does not pack: a particle in the band one cell off the floor keeps no
    /// velocity into it, so the column does not silently shorten at rest (it lost 5 % in 12 s
    /// when only the grid nodes were constrained — the bottom 2 cm gained 46 % of its particles).
    func testRestingColumnHoldsItsHeight() throws {
        let (solver, _) = try makeSolver()
        for _ in 0 ..< 60 { try frame(solver) }
        solver.settleDamping = 1
        for _ in 0 ..< 60 { try frame(solver) }
        let h0 = centreHeight(solver)
        for _ in 0 ..< 600 { try frame(solver) }
        let h1 = centreHeight(solver)
        print("MLSMPMVessel: resting column centre \(h0) → \(h1) over 10 s")
        // No compaction; the column may still relax up (a few %) out of the settle's squeeze
        // toward the state the volume correction holds (≈ the seed level).
        XCTAssertGreaterThanOrEqual(h1, h0 * 0.99, "resting column compacts")
        XCTAssertLessThanOrEqual(h1, h0 * 1.06, "resting column swells")
    }

    /// Twenty seconds of a lumbering walk and ten of standing: the level holds (the integrated J
    /// alone lets the agitated liquid compact to 40 % of its height — volumeCorrection).
    func testLevelHoldsThroughAWalk() throws {
        let (solver, _) = try makeSolver()
        for _ in 0 ..< 60 { try frame(solver) }
        solver.settleDamping = 1
        var t: Float = 0
        var heights: [Float] = []
        for sec in 0 ..< 30 {
            for _ in 0 ..< 60 {
                try frame(solver, SIMD3(0, -9.81, 0) - (sec < 20 ? walk(t) : .zero))
                t += 1.0 / 60
            }
            if (sec + 1) % 5 == 0 { heights.append(centreHeight(solver)) }
        }
        print("MLSMPMVessel: centre height every 5 s (walk ×4, stand ×2): \(heights)")
        XCTAssertEqual(outside(solver), 0, "particles outside the vessel")
        for h in heights { XCTAssertEqual(h, heights[0], accuracy: heights[0] * 0.05, "level through the walk") }
    }

    func testFirstSloshModeRingsAtTheAnalyticPeriod() throws {
        let (solver, n) = try makeSolver()
        for _ in 0 ..< 90 { try frame(solver) }          // settle (damped)
        solver.settleDamping = 1
        // Kick: effective gravity tilted 6° for 0.1 s (a lateral acceleration of g·tan 6°).
        let tilt: Float = 6 * .pi / 180
        for _ in 0 ..< 6 { try frame(solver, SIMD3(sin(tilt), -cos(tilt), 0) * 9.81) }
        // Free sloshing: the liquid's centre of mass swings along x at the first mode.
        var xs: [Float] = []
        let p = solver.particleBuffer.contents
        for _ in 0 ..< 240 {
            try frame(solver)
            var sx: Float = 0
            for i in 0 ..< n { sx += p[i].positionMass.x }
            xs.append(sx / Float(n))
        }
        let mean = xs.reduce(0, +) / Float(xs.count)
        let c = xs.map { $0 - mean }
        var crossings: [Float] = []
        for i in 1 ..< c.count where (c[i - 1] < 0) != (c[i] < 0) {
            crossings.append((Float(i - 1) + c[i - 1] / (c[i - 1] - c[i])) / 60)
        }
        XCTAssertGreaterThanOrEqual(crossings.count, 4, "no oscillation")
        let halfPeriods = zip(crossings.dropFirst(), crossings).map { $0 - $1 }
        let period = 2 * halfPeriods.reduce(0, +) / Float(max(1, halfPeriods.count))
        let xi: Float = 1.8412
        let analytic = 2 * Float.pi / sqrt(9.81 * xi / R * tanh(xi * depth / R))
        let amps = stride(from: 0, to: c.count - 60, by: 60).map { i in c[i ..< i + 60].map(abs).max() ?? 0 }
        print("MLSMPMVessel: slosh period \(period) s vs analytic \(analytic) s; CoM amplitude per second \(amps)")
        XCTAssertEqual(period, analytic, accuracy: analytic * 0.04, "first-mode period")
    }
}
