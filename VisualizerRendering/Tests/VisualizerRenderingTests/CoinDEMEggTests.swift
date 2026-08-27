import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Tests for the OVOID (egg) body — tag 5, the sphere-swept cone. CPU cases
/// check the spawn-time solid integration; GPU cases drive the real solver
/// through the same runtime-compiled-library seam as the other CoinDEM tests
/// and assert an egg falls, collides, and RESTS like an egg (on its flank,
/// COM between its two end radii — never standing on a tip, never sunk).
@MainActor
final class CoinDEMEggTests: XCTestCase {

    // ── CPU: spawn-time solid integration ────────────────────────────────────

    /// Degenerate ovoid with equal radii and zero centre distance IS a sphere:
    /// COM at the centre, isotropic inverse inertia 1/(0.4 R²).
    func testEggPropertiesSphereDegenerate() {
        let r: Float = 0.05
        let p = CoinDEMSolver.eggProperties(fatRadius: r, tipRadius: r, centerDistance: 0)
        XCTAssertEqual(p.yFat, 0, accuracy: 1e-3 * r)
        XCTAssertEqual(p.yTip, 0, accuracy: 1e-3 * r)
        let expected = 1.0 / (0.4 * r * r)
        for lane in [p.invInertiaK.x, p.invInertiaK.y, p.invInertiaK.z] {
            XCTAssertEqual(lane, expected, accuracy: expected * 0.02,
                           "sphere-degenerate ovoid must integrate to solid-sphere inertia")
        }
    }

    /// Equal radii with separation is a capsule: the COM sits at the segment
    /// midpoint and the inertia matches the analytic capsule aggregation.
    func testEggPropertiesCapsuleDegenerate() {
        let r: Float = 0.03, d: Float = 0.08
        let p = CoinDEMSolver.eggProperties(fatRadius: r, tipRadius: r, centerDistance: d)
        XCTAssertEqual(p.yFat, -d / 2, accuracy: 1e-3 * d)
        XCTAssertEqual(p.yTip,  d / 2, accuracy: 1e-3 * d)
        // Analytic solid capsule (cylinder + hemispheres), hl = d/2 — the same
        // formula cdCapsuleInvInertia uses in CoinDEM.metal.
        let hl = d / 2, L = d
        let vc = Float.pi * r * r * L
        let vs = (4.0 / 3.0) * Float.pi * r * r * r
        let v = vc + vs
        let mc = vc / v, ms = vs / v
        let iyyK = mc * 0.5 * r * r + ms * 0.4 * r * r
        let ixxK = mc * (L * L / 12 + 0.25 * r * r)
                 + ms * (0.4 * r * r + 0.25 * L * L + 0.375 * L * r)
        _ = hl
        XCTAssertEqual(p.invInertiaK.y, 1 / iyyK, accuracy: (1 / iyyK) * 0.03)
        XCTAssertEqual(p.invInertiaK.x, 1 / ixxK, accuracy: (1 / ixxK) * 0.03)
    }

    /// A real egg's COM is biased toward the FAT end: the fat-sphere centre
    /// sits closer to the COM than the tip-sphere centre does.
    func testEggComBiasesTowardFatEnd() {
        let p = CoinDEMSolver.eggProperties(fatRadius: 0.055, tipRadius: 0.040,
                                            centerDistance: 0.050)
        XCTAssertLessThan(abs(p.yFat), abs(p.yTip),
                          "COM must sit nearer the fat end (|yFat| < |yTip|)")
        // And the diameter moment exceeds the axis moment (prolate solid).
        XCTAssertGreaterThan(p.invInertiaK.y, p.invInertiaK.x,
                             "spinning about the long axis must be easier than tumbling")
    }

    // ── GPU harness (mirrors RigidPileFieldTests) ─────────────────────────────

    private static func makeLibrary(_ device: MTLDevice) throws -> MTLLibrary {
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard FileManager.default.fileExists(atPath: shader.path) else {
            throw XCTSkip("CoinDEM.metal not found at \(shader.path)")
        }
        return try MetalSourceLoader.makeLibrary(device: device, contentsOf: shader)
    }

    private func makeField(maxBodies: Int, bodyScale: Float,
                           colliders: [CoinStaticCollider]) throws
        -> (RigidPileField, MTLCommandQueue)
    {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)
        var cfg = RigidPileField.Config(maxBodies: maxBodies, bodyScale: bodyScale,
                                        bounds: (SIMD3(-1, -0.5, -1), SIMD3(1, 2, 1)))
        cfg.floorY = 0
        cfg.useConstraintSolver = true       // ovoid is a constraint-path shape
        cfg.frictionCoeff = 0.4              // real Coulomb grip so eggs roll + settle
        guard let field = RigidPileField(engine: engine, library: lib, config: cfg,
                                         colliders: colliders),
              let queue = device.makeCommandQueue()  // gpu-ok: test harness queue
        else { throw XCTSkip("field/queue init failed") }
        return (field, queue)
    }

    private func step(_ field: RigidPileField, _ queue: MTLCommandQueue, frames: Int) {
        for _ in 0..<frames {
            guard let cb = queue.makeCommandBuffer() else { return }
            field.encode(to: cb, dt: 1.0 / 60.0)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
        }
    }

    // Rain-egg reference dimensions (fat end down to tip, metres).
    private static let rFat: Float = 0.055
    private static let rTip: Float = 0.040
    private static let dCtr: Float = 0.050

    /// One tumbled egg dropped on the floor comes to REST LYING ON ITS FLANK:
    /// KE dies, the COM height lands between the tip and fat radii (the exact
    /// band a side-lying swept cone rests in), and the long axis ends up near
    /// horizontal — not balanced on an end.
    func testEggSettlesLyingOnItsSide() throws {
        let (field, queue) = try makeField(maxBodies: 4, bodyScale: 0.11,
                                           colliders: [RigidPileField.floor(y: 0)])
        guard let id = field.dropEgg(at: SIMD3(0, 0.5, 0),
                                     fatRadius: Self.rFat, tipRadius: Self.rTip,
                                     centerDistance: Self.dCtr,
                                     tumble: SIMD3(3, 1, 2)) else {
            return XCTFail("dropEgg returned nil")
        }
        step(field, queue, frames: 600)

        guard let p = field.position(of: id), let q = field.orientation(of: id),
              let v = field.velocity(of: id) else { return XCTFail("egg vanished") }
        XCTAssertFalse(p.y.isNaN, "no NaN blow-up")
        XCTAssertLessThan(simd_length(v), 0.05, "the egg comes to rest")
        // Side-lying rest band: both end spheres tangent to the floor puts the
        // COM between rTip and rFat; allow the solver's contact slop each way.
        XCTAssertGreaterThan(p.y, Self.rTip - 0.008, "not sunk into the floor")
        XCTAssertLessThan(p.y, Self.rFat + 0.008, "not propped or balanced upright")
        let axisUp = abs(simd_act(q, SIMD3<Float>(0, 1, 0)).y)
        XCTAssertLessThan(axisUp, 0.45, "long axis near horizontal (lying, not standing)")
    }

    /// A dozen eggs rained into a bin settle into a quiet heap — egg↔egg and
    /// egg↔plane contacts hold, nothing tunnels, explodes, or NaNs. The
    /// pile-of-mess behaviour, with eggs.
    func testEggPileSettlesInBin() throws {
        let n = 12
        let bin = RigidPileField.bin(innerHalf: SIMD2(0.25, 0.25), floorY: 0)
        let (field, queue) = try makeField(maxBodies: n + 2, bodyScale: 0.11, colliders: bin)
        var seed: UInt64 = 0xE66
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for _ in 0..<n {
            field.dropEgg(at: SIMD3((rnd() - 0.5) * 0.3, 0.3 + rnd() * 0.9, (rnd() - 0.5) * 0.3),
                          fatRadius: Self.rFat, tipRadius: Self.rTip, centerDistance: Self.dCtr,
                          tumble: SIMD3((rnd() - 0.5) * 5, (rnd() - 0.5) * 5, (rnd() - 0.5) * 5))
        }
        step(field, queue, frames: 600)

        let r = CoinDiagnostics.measure(field.solver)
        print("EGG_PILE n=\(r.activeCount) KE=\(r.kineticEnergy) belowFloor=\(r.belowFloorCount) maxY=\(r.maxY)")
        XCTAssertEqual(r.activeCount, n)
        XCTAssertFalse(r.kineticEnergy.isNaN, "no NaN blow-up")
        XCTAssertEqual(r.belowFloorCount, 0, "nothing tunnelled the floor")
        XCTAssertLessThan(r.kineticEnergy, 0.8, "the egg heap settles")
        XCTAssertLessThan(r.maxY, 0.4, "the heap stays in the bin, not launched")
    }

    /// Eggs coexist with the other shapes in ONE mixed pile — the swept-pair
    /// generalization must not regress capsules, and egg↔sphere / egg↔box /
    /// egg↔capsule contacts all resolve.
    func testEggsMixIntoTheMess() throws {
        let n = 16
        let bin = RigidPileField.bin(innerHalf: SIMD2(0.3, 0.3), floorY: 0)
        let (field, queue) = try makeField(maxBodies: n + 2, bodyScale: 0.11, colliders: bin)
        var seed: UInt64 = 0xBEEF
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for i in 0..<n {
            let p = SIMD3<Float>((rnd() - 0.5) * 0.4, 0.25 + rnd() * 0.8, (rnd() - 0.5) * 0.4)
            let tumble = SIMD3<Float>((rnd() - 0.5) * 4, (rnd() - 0.5) * 4, (rnd() - 0.5) * 4)
            switch i % 4 {
            case 0: field.dropEgg(at: p, fatRadius: Self.rFat, tipRadius: Self.rTip,
                                  centerDistance: Self.dCtr, tumble: tumble, type: 0)
            case 1: field.dropSphere(at: p, radius: 0.04, tumble: tumble, type: 1)
            case 2: field.dropRod(at: p, radius: 0.015, halfLength: 0.06, tumble: tumble, type: 2)
            default: field.dropBox(at: p, halfExtents: SIMD3(0.04, 0.03, 0.04), tumble: tumble, type: 3)
            }
        }
        step(field, queue, frames: 600)

        let r = CoinDiagnostics.measure(field.solver)
        print("EGG_MIX n=\(r.activeCount) KE=\(r.kineticEnergy) belowFloor=\(r.belowFloorCount) maxY=\(r.maxY)")
        XCTAssertEqual(r.activeCount, n)
        XCTAssertFalse(r.kineticEnergy.isNaN, "no NaN blow-up")
        XCTAssertEqual(r.belowFloorCount, 0, "nothing tunnelled the floor")
        XCTAssertLessThan(r.kineticEnergy, 1.2, "the mixed heap settles")
        XCTAssertLessThan(r.maxY, 0.45, "the heap stays in the bin")
    }
}
