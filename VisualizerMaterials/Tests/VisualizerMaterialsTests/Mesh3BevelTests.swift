import XCTest
@testable import VisualizerMaterials

/// DH-1437 — `Mesh3.beveled` keeps a closed solid closed, shrinks it only at its edges, and
/// leaves no face wound against its own normals.
final class Mesh3BevelTests: XCTestCase {

    private func box(_ w: Double, _ h: Double, _ d: Double) -> Mesh3 {
        var m = Mesh3()
        let (x, y, z) = (w, h, d)
        m.addQuad(Vec3(x,0,0), Vec3(x,y,0), Vec3(x,y,z), Vec3(x,0,z), outward: Vec3( 1,0,0))
        m.addQuad(Vec3(0,0,0), Vec3(0,0,z), Vec3(0,y,z), Vec3(0,y,0), outward: Vec3(-1,0,0))
        m.addQuad(Vec3(0,y,0), Vec3(0,y,z), Vec3(x,y,z), Vec3(x,y,0), outward: Vec3(0, 1,0))
        m.addQuad(Vec3(0,0,0), Vec3(x,0,0), Vec3(x,0,z), Vec3(0,0,z), outward: Vec3(0,-1,0))
        m.addQuad(Vec3(0,0,z), Vec3(x,0,z), Vec3(x,y,z), Vec3(0,y,z), outward: Vec3(0,0, 1))
        m.addQuad(Vec3(0,0,0), Vec3(0,y,0), Vec3(x,y,0), Vec3(x,0,0), outward: Vec3(0,0,-1))
        return m
    }

    /// Undirected edges (welded by position) used by other than two triangles.
    private func openEdges(_ m: Mesh3) -> Int {
        func key(_ p: Vec3) -> SIMD3<Int64> {
            SIMD3(Int64((p.x * 1e6).rounded()), Int64((p.y * 1e6).rounded()), Int64((p.z * 1e6).rounded()))
        }
        struct E: Hashable { let a: SIMD3<Int64>, b: SIMD3<Int64> }
        var count: [E: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            let k = (0..<3).map { key(m.positions[Int(m.indices[t + $0])]) }
            for i in 0..<3 {
                let a = k[i], b = k[(i + 1) % 3]
                let lessThan = (a.x, a.y, a.z) < (b.x, b.y, b.z)
                count[lessThan ? E(a: a, b: b) : E(a: b, b: a), default: 0] += 1
            }
            t += 3
        }
        return count.values.filter { $0 != 2 }.count
    }

    private func volume(_ m: Mesh3) -> Double {
        var v = 0.0, t = 0
        while t + 2 < m.indices.count {
            let a = m.positions[Int(m.indices[t])], b = m.positions[Int(m.indices[t + 1])],
                c = m.positions[Int(m.indices[t + 2])]
            v += dot3(a, cross3(b, c)) / 6; t += 3
        }
        return v
    }

    /// Triangles whose winding disagrees with their stored normals (inside-out).
    private func invertedFaces(_ m: Mesh3) -> Int {
        var bad = 0, t = 0
        while t + 2 < m.indices.count {
            let i = (0..<3).map { Int(m.indices[t + $0]) }
            let n = cross3(m.positions[i[1]] - m.positions[i[0]], m.positions[i[2]] - m.positions[i[0]])
            let s = m.normals[i[0]] + m.normals[i[1]] + m.normals[i[2]]
            if dot3(n, s) <= 0 { bad += 1 }
            t += 3
        }
        return bad
    }

    func testBeveledBoxStaysClosedAndKeepsItsExtents() {
        for seg in [1, 3] {
            let b = box(0.1, 0.06, 0.04).beveled(radius: 0.006, segments: seg)
            XCTAssertEqual(openEdges(b), 0, "segments \(seg): bevelled box must stay watertight")
            XCTAssertEqual(invertedFaces(b), 0, "segments \(seg): no inside-out face")
            let bb = b.bounds!
            XCTAssertEqual(bb.max.x - bb.min.x, 0.1, accuracy: 1e-6)
            XCTAssertEqual(bb.max.y - bb.min.y, 0.06, accuracy: 1e-6)
            XCTAssertEqual(bb.max.z - bb.min.z, 0.04, accuracy: 1e-6)
        }
    }

    func testBevelRemovesVolumeOnlyAtTheEdges() {
        let plain = box(0.1, 0.06, 0.04), r = 0.006
        let b = plain.beveled(radius: r, segments: 4)
        let lost = volume(plain) - volume(b)
        // 12 edges × (1 − π/4)·r² ≈ the rounded-away sliver; allow a wide band for the Bézier
        // and the dome corners, but it must be a SHAVING, not a collapse.
        XCTAssertGreaterThan(lost, 0)
        XCTAssertLessThan(lost, 0.05 * volume(plain))
    }

    func testThinPlateTakesAnEaseWithoutCrossing() {
        let plate = box(0.02, 0.0024, 0.15)                 // a tine-like 2.4 mm section
        let b = plate.beveled(radius: 0.0005, segments: 3)
        XCTAssertEqual(openEdges(b), 0)
        XCTAssertEqual(invertedFaces(b), 0)
        XCTAssertGreaterThan(volume(b), 0.9 * volume(plate))
    }

    func testSmoothMeshIsLeftAlone() {
        let m = box(1, 1, 1)
        let same = m.beveled(radius: 0.1, creaseDegrees: 120)   // nothing is sharper than 120°
        XCTAssertEqual(same, m)
    }

    func testBevelAddsRoundedEdgeNormals() {
        let b = box(0.1, 0.06, 0.04).beveled(radius: 0.006, segments: 3)
        // some normal must sit strictly between the face axes: the swing that makes it a round
        XCTAssertTrue(b.normals.contains { n in
            let m = [abs(n.x), abs(n.y), abs(n.z)].sorted(); return m[2] < 0.95 && m[1] > 0.3
        })
    }
}
