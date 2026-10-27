import XCTest
import simd
@testable import Signack

/// `SignackEdgeCollapse.collapseOpenBorders` (Daydream DH-1432): a sheet whose every vertex is near
/// an open edge — a leaf's strip between two slits — simplifies only when its border may slide, and
/// sliding keeps the outline: the corners stay put and the border stays on the sheet's own edge.
final class SignackOpenBorderTests: XCTestCase {

    /// A long, narrow open strip one quad deep, so every vertex is on the border.
    private func strip() -> SignackTestMeshes.Mesh {
        let nx = 480, nz = 1
        var p: [SIMD3<Float>] = []
        for j in 0...nz { for i in 0...nx {
            let x = Float(i) / Float(nx), z = Float(j) / 48
            p.append(SIMD3(x, 0.02 * sin(x * 6), z))
        } }
        var idx: [UInt32] = []
        for j in 0..<nz { for i in 0..<nx {
            let a = UInt32(j * (nx + 1) + i), b = a + 1, c = a + UInt32(nx + 1), d = c + 1
            idx += [a, c, b, b, c, d]
        } }
        return SignackTestMeshes.Mesh(positions: p, indices: idx)
    }

    private func coarsest(_ m: SignackTestMeshes.Mesh, open: Bool) -> (SignackHierarchy, [UInt32]) {
        let h = SignackHierarchyBuilder().build(positions: m.positions, indices: m.indices,
                                                simplifier: SignackEdgeCollapse(collapseOpenBorders: open))
        return (h, h.flatten(SignackSelection.cut(h, tolerance: 10)))
    }

    func testAnAllBorderStripSimplifiesOnlyWithOpenBorders() {
        let m = strip()
        let locked = coarsest(m, open: false).1.count / 3
        let open = coarsest(m, open: true).1.count / 3
        XCTAssertEqual(locked, m.indices.count / 3, "locked borders: an all-border strip cannot lose a triangle")
        XCTAssertLessThan(open, locked / 2, "open borders: the strip must simplify (\(open) of \(locked))")
    }

    func testSlidingKeepsTheOutline() {
        let m = strip()
        let (h, cut) = coarsest(m, open: true)
        let pos = h.vertices(host: m.positions, position: \.self)
        let used = Set(cut).map { pos[Int($0)] }
        // The four corners survive.
        let corners: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(1, 0), SIMD2(0, 1.0 / 48), SIMD2(1, 1.0 / 48)]
        for c in corners {
            XCTAssertTrue(used.contains { abs($0.x - c.x) < 1e-6 && abs($0.z - c.y) < 1e-6 }, "corner \(c) moved")
        }
        // Every surviving vertex is still a source vertex (half-edge collapse) inside the strip.
        for v in used {
            XCTAssertTrue(v.x >= -1e-6 && v.x <= 1 + 1e-6 && v.z >= -1e-6 && v.z <= 1.0 / 48 + 1e-6)
        }
    }

    func testOffByDefaultLeavesClosedMeshesUnchanged() {
        let m = SignackTestMeshes.icosphere(subdivisions: 4)
        let a = SignackHierarchyBuilder().build(positions: m.positions, indices: m.indices, simplifier: SignackEdgeCollapse())
        let b = SignackHierarchyBuilder().build(positions: m.positions, indices: m.indices,
                                                simplifier: SignackEdgeCollapse(collapseOpenBorders: true))
        XCTAssertEqual(a.indices, b.indices, "a closed mesh has no border: the option must change nothing")
    }
}
