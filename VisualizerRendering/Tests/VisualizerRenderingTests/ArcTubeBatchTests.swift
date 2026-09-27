import Metal
import XCTest
import simd
@testable import VisualizerRendering

/// `ArcTubeBatch` — the GPU sweep of constant-length arc tubes (ArcTube.metal).
@MainActor
final class ArcTubeBatchTests: XCTestCase {

    /// A quarter-circle arc and a straight one: every ring vertex sits exactly one tube
    /// radius from its centreline point, its normal is that offset's direction, free slots
    /// collapse far below the world, and every triangle winds with its normals (outward).
    func testSweepMatchesTheClosedFormAndWindsOutward() throws {
        let engine = SimEngine.shared
        let rings = 9, radial = 8
        let batch = try XCTUnwrap(ArcTubeBatch(engine: engine, maxTubes: 3, rings: rings, radial: radial))

        // Quarter circle of radius 0.5 about the origin, from (0.5, 0, 0) to (0, 0.5, 0).
        let R: Float = 0.5, theta: Float = .pi / 4           // half-turn: the arc spans 90°
        let a = SIMD3<Float>(R, 0, 0), b = SIMD3<Float>(0, R, 0)
        let e = simd_normalize(b - a)
        let mid = (a + b) * 0.5
        let u = simd_normalize(mid)                           // bows away from the centre (origin)
        let arcLen = R * 2 * theta
        let curved = ArcTubeBatch.Arc(start: a, chordDirection: e, bowDirection: u, length: arcLen,
                                      halfTurn: theta, centre: .zero, arcRadius: R, tubeRadius: 0.05)
        let straight = ArcTubeBatch.Arc(start: SIMD3(0, 0, 1), chordDirection: SIMD3(0, 0, 1),
                                        bowDirection: SIMD3(0, 1, 0), length: 2, halfTurn: 0,
                                        centre: .zero, arcRadius: 0, tubeRadius: 0.1)
        let cb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
        batch.encode([curved, straight], into: cb)
        cb.commit()
        cb.waitUntilCompleted()   // gpu-ok: test-time readback
        XCTAssertNil(cb.error)

        let n = batch.vertexCount
        let pp = batch.positionBuffer.contents().bindMemory(to: Float.self, capacity: n * 3)
        let np = batch.normalBuffer.contents().bindMemory(to: Float.self, capacity: n * 3)
        func P(_ i: Int) -> SIMD3<Float> { SIMD3(pp[i * 3], pp[i * 3 + 1], pp[i * 3 + 2]) }
        func N(_ i: Int) -> SIMD3<Float> { SIMD3(np[i * 3], np[i * 3 + 1], np[i * 3 + 2]) }

        // Curved tube: ring i's centre is the arc point at t = i/(rings−1).
        for i in 0 ..< rings {
            let t = Float(i) / Float(rings - 1)
            let phi = -theta + 2 * theta * t
            let c = (e * sin(phi) + u * cos(phi)) * R
            for j in 0 ..< radial {
                let v = i * radial + j
                XCTAssertEqual(simd_distance(P(v), c), 0.05, accuracy: 1e-4)
                XCTAssertEqual(simd_dot(simd_normalize(P(v) - c), N(v)), 1, accuracy: 1e-4)
            }
        }
        XCTAssertEqual(simd_distance(P(0), a), 0.05, accuracy: 1e-4, "starts at the root")
        XCTAssertEqual(simd_distance(P((rings - 1) * radial), b), 0.05, accuracy: 1e-4, "ends at the tip")
        // Straight tube: rings evenly along +z from z = 1 to z = 3.
        let base = rings * radial
        XCTAssertEqual(P(base).z, 1, accuracy: 1e-4)
        XCTAssertEqual(P(base + (rings - 1) * radial).z, 3, accuracy: 1e-4)
        // The free third slot collapsed.
        XCTAssertEqual(P(2 * base).y, -100_000, accuracy: 1)

        // Winding: each live triangle's face normal agrees with its vertices' normals.
        let ip = batch.indexBuffer.contents().bindMemory(to: UInt32.self, capacity: batch.indexCount)
        var disagree = 0, checked = 0
        for t in stride(from: 0, to: batch.indexCount, by: 3) {
            let i0 = Int(ip[t]), i1 = Int(ip[t + 1]), i2 = Int(ip[t + 2])
            guard i0 < 2 * base else { continue }
            let f = simd_cross(P(i1) - P(i0), P(i2) - P(i0))
            guard simd_length(f) > 1e-10 else { continue }
            checked += 1
            if simd_dot(f, N(i0) + N(i1) + N(i2)) < 0 { disagree += 1 }
        }
        XCTAssertGreaterThan(checked, 0)
        XCTAssertEqual(disagree, 0, "inside-out triangles: \(disagree)/\(checked)")
    }
}
