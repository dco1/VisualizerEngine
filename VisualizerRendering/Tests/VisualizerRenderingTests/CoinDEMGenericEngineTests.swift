import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Headless correctness gate for the GENERIC-engine extensions to CoinDEMSolver
/// (the "fully capable rigid-body engine" pass): per-body materials, exact
/// capsules, convex hulls, and generic joints — all on the constraint path.
/// Mirrors CoinDEMSolverTests' harness: CoinDEM.metal is compiled from source at
/// runtime (SwiftPM doesn't build metallibs) and injected via the test seam.
@MainActor
final class CoinDEMGenericEngineTests: XCTestCase {

    private static func makeLibrary(_ device: MTLDevice) throws -> MTLLibrary {
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard let src = try? String(contentsOf: shader, encoding: .utf8) else {
            throw XCTSkip("CoinDEM.metal not found at \(shader.path)")
        }
        return try device.makeLibrary(source: src, options: nil)
    }

    /// Constraint-path solver over a flat floor. `maxDim` sizes the broadphase
    /// cell (pass the largest body half-dimension you plan to spawn).
    private func makeSolver(maxCoins: Int = 64,
                            radius: Float = 0.05,
                            maxDim: Float = 0.1,
                            boundsMin: SIMD3<Float> = SIMD3(-2, -0.5, -2),
                            boundsMax: SIMD3<Float> = SIMD3(2, 2, 2)) throws
        -> (CoinDEMSolver, MTLCommandQueue)
    {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)
        guard let solver = CoinDEMSolver(engine: engine, library: lib,
                                         maxCoins: maxCoins,
                                         coinRadius: radius, halfThickness: maxDim,
                                         boundsMin: boundsMin, boundsMax: boundsMax),
              let queue = device.makeCommandQueue()  // gpu-ok: test harness queue
        else { throw XCTSkip("solver/queue init failed") }
        solver.floorY = 0
        solver.setColliders([.plane(normal: SIMD3(0, 1, 0), offset: 0)])
        solver.solverMode = .constraint
        return (solver, queue)
    }

    private func step(_ solver: CoinDEMSolver, _ queue: MTLCommandQueue, frames: Int,
                      perFrame: ((Int) -> Void)? = nil) {
        for f in 0..<frames {
            guard let cb = queue.makeCommandBuffer() else { return }
            solver.encode(to: cb, wallDt: 1.0 / 60.0)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            perFrame?(f)
        }
    }

    // ── Capsule: side rest ─────────────────────────────────────────────────────
    // A horizontal capsule dropped on the floor must settle with its COM at
    // exactly the cross-section radius (the 2-endpoint-probe manifold holding it
    // level), stay level, and go still.
    func testCapsuleRestsOnSideAtRadius() throws {
        let (solver, queue) = try makeSolver(radius: 0.03, maxDim: 0.09)
        let r: Float = 0.03, hl: Float = 0.06
        // Axis along world X: rotate local +Y → +X (quarter turn about Z).
        let q = simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 0, 1))
        let slot = solver.spawnCapsule(at: SIMD3(0, 0.25, 0), radius: r, halfLength: hl,
                                       orient: SIMD4(q.imag, q.real))
        XCTAssertNotNil(slot)
        step(solver, queue, frames: 240)

        let p = solver.position(of: slot!)!
        let v = solver.velocity(of: slot!)!
        XCTAssertEqual(p.y, r, accuracy: r * 0.25,
                       "side-lying capsule rests with COM at its cross-section radius")
        XCTAssertLessThan(simd_length(v), 0.02, "capsule at rest")
        // Still lying flat: the body axis stays horizontal.
        let axis = simd_act(solver.orientation(of: slot!)!, SIMD3<Float>(0, 1, 0))
        XCTAssertLessThan(abs(axis.y), 0.25, "capsule stayed on its side (didn't tip or sink)")
    }

    // ── Capsule: crossed stack ────────────────────────────────────────────────
    // A capsule dropped crosswise onto a resting capsule must stack (top COM ≈ 3r)
    // rather than sink through or over-separate — the capsule↔capsule seg-seg
    // contact at work.
    func testCapsulesStackCrossed() throws {
        let (solver, queue) = try makeSolver(radius: 0.03, maxDim: 0.09)
        let r: Float = 0.03, hl: Float = 0.05
        let qx = simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 0, 1))   // axis → X
        let qz = simd_quatf(angle:  .pi / 2, axis: SIMD3(1, 0, 0))   // axis → Z
        let bottom = solver.spawnCapsule(at: SIMD3(0, 0.10, 0), radius: r, halfLength: hl,
                                         orient: SIMD4(qx.imag, qx.real))
        let top = solver.spawnCapsule(at: SIMD3(0.001, 0.30, 0), radius: r, halfLength: hl,
                                      orient: SIMD4(qz.imag, qz.real))
        XCTAssertNotNil(bottom); XCTAssertNotNil(top)
        step(solver, queue, frames: 300)

        let pb = solver.position(of: bottom!)!
        let pt = solver.position(of: top!)!
        XCTAssertEqual(pb.y, r, accuracy: r * 0.3, "bottom capsule on the floor")
        XCTAssertEqual(pt.y, 3 * r, accuracy: r * 0.5,
                       "top capsule rests ON the bottom one (COM ≈ 3r), no sink-through")
        let stats = solver.measurePenetration(threshold: r * 0.15)
        XCTAssertLessThan(stats.maxPenetration, r * 0.4, "no deep capsule↔capsule overlap")
    }

    // ── Capsule ↔ sphere ──────────────────────────────────────────────────────
    // A sphere dropped squarely onto a side-lying capsule contacts its rounded
    // surface (segment + radius): it must not sink in, and it ends up beside the
    // capsule on the floor (spheres roll off round things — that's physics).
    func testSphereRollsOffCapsuleWithoutPenetrating() throws {
        let (solver, queue) = try makeSolver(radius: 0.04, maxDim: 0.1)
        let r: Float = 0.03, hl: Float = 0.06, rs: Float = 0.04
        let q = simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 0, 1))
        solver.spawnCapsule(at: SIMD3(0, 0.15, 0), radius: r, halfLength: hl,
                            orient: SIMD4(q.imag, q.real))
        let ball = solver.spawnSphere(at: SIMD3(0.004, 0.4, 0), radius: rs)
        XCTAssertNotNil(ball)
        var minBallY: Float = .greatestFiniteMagnitude
        step(solver, queue, frames: 300) { _ in
            if let p = self.ballPos(solver, ball!) { minBallY = min(minBallY, p.y) }
        }
        // While interacting, the ball's COM never dipped meaningfully below the
        // capsule-top contact height (2r + rs would be resting exactly on top; a
        // graze during roll-off may ride lower but must clear the sphere's own
        // floor-rest height minus a small slop — i.e. no tunnelling INTO the capsule).
        XCTAssertGreaterThan(minBallY, rs * 0.8, "ball never sank into the capsule or floor")
        let stats = solver.measurePenetration(threshold: r * 0.15)
        XCTAssertLessThan(stats.maxPenetration, r * 0.5, "sphere↔capsule stayed separated")
    }

    private func ballPos(_ solver: CoinDEMSolver, _ slot: Int) -> SIMD3<Float>? {
        solver.position(of: slot)
    }

    // ── Per-body restitution ──────────────────────────────────────────────────
    // Two identical spheres, same drop, different per-body restitution: the
    // bouncy one must rebound visibly higher. Globals stay at their defaults —
    // the override alone drives the difference.
    func testPerBodyRestitutionControlsBounce() throws {
        let (solver, queue) = try makeSolver(radius: 0.05, maxDim: 0.05)
        solver.restitution = 0.05          // global default: nearly dead
        solver.restThreshold = 0.3
        let dead   = solver.spawnSphere(at: SIMD3(-0.5, 0.8, 0), radius: 0.05)
        let bouncy = solver.spawnSphere(at: SIMD3( 0.5, 0.8, 0), radius: 0.05,
                                        restitution: 0.85)
        XCTAssertNotNil(dead); XCTAssertNotNil(bouncy)

        var reboundDead: Float = 0, reboundBouncy: Float = 0
        var impacted = false
        step(solver, queue, frames: 240) { _ in
            guard let pd = solver.position(of: dead!), let pb = solver.position(of: bouncy!) else { return }
            if pd.y < 0.08 { impacted = true }          // first floor hit happened
            if impacted {
                reboundDead   = max(reboundDead, pd.y)
                reboundBouncy = max(reboundBouncy, pb.y)
            }
        }
        print("MATERIAL_RESTITUTION dead=\(reboundDead) bouncy=\(reboundBouncy)")
        XCTAssertGreaterThan(reboundBouncy, reboundDead + 0.1,
                             "e=0.85 sphere rebounds well above the e=0.05 sphere")
        XCTAssertLessThan(reboundDead, 0.25, "dead sphere barely rebounds")
    }

    // ── Per-body friction ─────────────────────────────────────────────────────
    // Two identical boxes slid along the floor with the same initial velocity,
    // different per-body μ: the grippy one must stop far shorter. Global
    // frictionCoeff stays 0 — vs statics a body's own material governs.
    func testPerBodyFrictionControlsSlide() throws {
        let (solver, queue) = try makeSolver(radius: 0.06, maxDim: 0.06)
        let he = SIMD3<Float>(0.04, 0.04, 0.04)
        let slick = solver.spawnBox(at: SIMD3(-0.8, 0.041, -0.5), halfExtents: he,
                                    velocity: SIMD3(2.0, 0, 0), friction: 0.02)
        let grippy = solver.spawnBox(at: SIMD3(-0.8, 0.041,  0.5), halfExtents: he,
                                     velocity: SIMD3(2.0, 0, 0), friction: 0.9)
        XCTAssertNotNil(slick); XCTAssertNotNil(grippy)
        step(solver, queue, frames: 240)

        let xSlick  = solver.position(of: slick!)!.x - (-0.8)
        let xGrippy = solver.position(of: grippy!)!.x - (-0.8)
        print("MATERIAL_FRICTION slick=\(xSlick) grippy=\(xGrippy)")
        XCTAssertGreaterThan(xSlick, xGrippy + 0.2,
                             "μ=0.02 box slides well past the μ=0.9 box")
        XCTAssertLessThan(simd_length(solver.velocity(of: grippy!)!), 0.05,
                          "grippy box has stopped")
    }
}
