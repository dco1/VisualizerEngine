import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Tests for the `RigidDiscField` facade — the disc-pile sibling of `RigidBallField`.
/// The GPU cases use the same runtime-compiled-library seam as `CoinDEMSolverTests`
/// (the SwiftPM CLI doesn't build the package metallib), so the REAL field +
/// solver orchestration runs; only the pipeline source changes.
@MainActor
final class RigidDiscFieldTests: XCTestCase {

    private static let R: Float = 0.05
    private static let h: Float = 0.0037

    private static func makeLibrary(_ device: MTLDevice) throws -> MTLLibrary {
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard let src = try? String(contentsOf: shader, encoding: .utf8) else {
            throw XCTSkip("CoinDEM.metal not found at \(shader.path)")
        }
        return try device.makeLibrary(source: src, options: nil)
    }

    private func makeField(maxDiscs: Int,
                           colliders: [CoinStaticCollider],
                           buoyant: Bool = false) throws -> (RigidDiscField, MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)
        var cfg = RigidDiscField.Config(maxDiscs: maxDiscs, discRadius: Self.R,
                                        discHalfThickness: Self.h,
                                        bounds: (SIMD3(-1, -0.5, -1), SIMD3(1, 2, 1)))
        cfg.floorY = 0
        guard let field = RigidDiscField(engine: engine, library: lib, config: cfg,
                                         colliders: colliders),
              let queue = device.makeCommandQueue()  // gpu-ok: test harness queue
        else { throw XCTSkip("field/queue init failed") }
        return (field, queue)
    }

    private func step(_ field: RigidDiscField, _ queue: MTLCommandQueue, frames: Int) {
        for _ in 0..<frames {
            guard let cb = queue.makeCommandBuffer() else { return }
            field.encode(to: cb, dt: 1.0 / 60.0)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
        }
    }

    // ── Pure: the bin helper reproduces the floor + 4-wall half-spaces a scene
    //         would hand-write (matched against the Tavern layout). No GPU. ──────
    func testBinHelperHalfSpaces() {
        let center = SIMD3<Float>(0, 0, 6.5)
        let inner = SIMD2<Float>(4.03, 1.94)
        let cols = RigidDiscField.bin(center: center, innerHalf: inner, floorY: 0)
        XCTAssertEqual(cols.count, 5, "floor + 4 walls")

        func tag(_ c: CoinStaticCollider) -> UInt32 { c.a.w.bitPattern }
        // All five are planes (tag 0).
        for c in cols { XCTAssertEqual(tag(c), 0, "bin colliders are plane half-spaces") }

        // Floor.
        XCTAssertEqual(cols[0].a.x, 0); XCTAssertEqual(cols[0].a.y, 1, accuracy: 1e-6)
        XCTAssertEqual(cols[0].b.w, 0, accuracy: 1e-6, "floor offset = floorY")
        // Left wall (+x normal): d = center.x - innerHalf.x = -4.03.
        XCTAssertEqual(cols[1].a.x, 1, accuracy: 1e-6)
        XCTAssertEqual(cols[1].b.w, -4.03, accuracy: 1e-4)
        // Right wall (-x normal): d = -(center.x + innerHalf.x) = -4.03.
        XCTAssertEqual(cols[2].a.x, -1, accuracy: 1e-6)
        XCTAssertEqual(cols[2].b.w, -4.03, accuracy: 1e-4)
        // Near wall (+z normal): d = center.z - innerHalf.y = 4.56.
        XCTAssertEqual(cols[3].a.z, 1, accuracy: 1e-6)
        XCTAssertEqual(cols[3].b.w, 4.56, accuracy: 1e-4)
        // Far wall (-z normal): d = -(center.z + innerHalf.y) = -8.44.
        XCTAssertEqual(cols[4].a.z, -1, accuracy: 1e-6)
        XCTAssertEqual(cols[4].b.w, -8.44, accuracy: 1e-4)
    }

    // ── A dropped disc pile settles flat and de-penetrated inside a bin. ───────
    func testDiscPileSettlesFlatInBin() throws {
        let n = 100
        let bin = RigidDiscField.bin(innerHalf: SIMD2(0.35, 0.35), floorY: 0)
        let (field, queue) = try makeField(maxDiscs: n + 2, colliders: bin)
        var seed: UInt64 = 0xC0FFEE
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for _ in 0..<n {
            field.drop(at: SIMD3((rnd()-0.5)*0.6, 0.15 + rnd()*0.8, (rnd()-0.5)*0.6),
                       tumble: SIMD3((rnd()-0.5)*4, (rnd()-0.5)*4, (rnd()-0.5)*4))
        }
        step(field, queue, frames: 300)

        let r = CoinDiagnostics.measure(field.solver)
        print("DISC_PILE n=\(r.activeCount) KE=\(r.kineticEnergy) maxPen=\(r.maxPenetration) maxY=\(r.maxY)")
        XCTAssertEqual(r.activeCount, n)
        XCTAssertEqual(r.belowFloorCount, 0, "no disc tunnelled the floor")
        XCTAssertLessThan(r.maxPenetration, 2.0 * Self.h, "discs de-penetrate (oriented SAT)")
        XCTAssertLessThan(r.kineticEnergy, 1.0, "pile settles, no jitter/explosion")
        XCTAssertFalse(r.kineticEnergy.isNaN)
    }

    // ── Buoyancy lifts the pile: with negative-gravity float on, the discs rise. ─
    func testBuoyancyLiftsDiscs() throws {
        let n = 40
        let bin = RigidDiscField.bin(innerHalf: SIMD2(0.35, 0.35), floorY: 0)
        let (field, queue) = try makeField(maxDiscs: n + 2, colliders: bin)
        var seed: UInt64 = 0x1234
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for _ in 0..<n {
            field.drop(at: SIMD3((rnd()-0.5)*0.6, 0.1 + rnd()*0.4, (rnd()-0.5)*0.6))
        }
        step(field, queue, frames: 200)                 // settle into a low pile
        let settledMaxY = CoinDiagnostics.measure(field.solver).maxY

        field.setBuoyant(true)                           // float up
        field.kick()                                     // break the clump
        step(field, queue, frames: 200)
        let floatedMaxY = CoinDiagnostics.measure(field.solver).maxY

        print("BUOYANCY settledMaxY=\(settledMaxY) floatedMaxY=\(floatedMaxY)")
        XCTAssertGreaterThan(floatedMaxY, settledMaxY + 0.2, "buoyancy should lift the discs")
    }

    // ── The reciprocating pusher shoves the pile forward at the plate's speed (not
    // faster — the forward-only anti-launch path), via the setPusher convenience. ──
    func testPusherShovesDiscsForward() throws {
        let (field, queue) = try makeField(maxDiscs: 16,
                                           colliders: [RigidDiscField.floor(y: 0)])
        // A row of discs resting on the floor in the plate's forward path.
        for z: Float in stride(from: 0.0, through: 0.42, by: 0.06) {
            field.drop(at: SIMD3(0, 0.05, z))
        }
        step(field, queue, frames: 90)             // settle flat

        let plateSpeed: Float = 1.0
        let dt: Float = 1.0 / 60.0
        var plateZ: Float = -0.4
        let ptr = field.solver.coinBuffer.buffer.contents()
            .bindMemory(to: CoinBody.self, capacity: field.solver.highWater)
        var maxForwardVZ: Float = 0
        for _ in 0..<90 {
            plateZ += plateSpeed * dt
            field.setPusher(center: SIMD3(0, 0.15, plateZ),
                            halfExtents: SIMD3(0.5, 0.2, 0.05),
                            velocity: SIMD3(0, 0, plateSpeed))
            guard let cb = queue.makeCommandBuffer() else { break }
            field.encode(to: cb, dt: dt); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test
            for i in 0..<field.solver.highWater where ptr[i].posInvMass.w != 0 {
                maxForwardVZ = max(maxForwardVZ, ptr[i].vel.z)
            }
        }
        print("DISC_PUSHER maxForwardVZ=\(maxForwardVZ) plateSpeed=\(plateSpeed)")
        XCTAssertGreaterThan(maxForwardVZ, 0.2, "the pusher must actually shove the discs")
        XCTAssertLessThan(maxForwardVZ, 1.8, "discs must not be launched faster than the plate")
    }
}
