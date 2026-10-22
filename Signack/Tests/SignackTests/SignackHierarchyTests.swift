import XCTest
import simd
@testable import Signack

/// S1 invariants, GPU-free: clusters partition the source, error rises up the DAG, every cut is
/// crack-free, and every level shares one vertex buffer.
final class SignackHierarchyTests: XCTestCase {

    private func build(_ m: SignackTestMeshes.Mesh) -> SignackHierarchy {
        SignackHierarchyBuilder().build(positions: m.positions, indices: m.indices,
                                        simplifier: SignackEdgeCollapse())
    }

    private func cutTriangles(_ h: SignackHierarchy, _ cut: [UInt32]) -> [SIMD3<UInt32>] {
        let f = h.flatten(cut)
        return stride(from: 0, to: f.count, by: 3).map { SIMD3(f[$0], f[$0 + 1], f[$0 + 2]) }
    }

    /// Views from far to inside, at several thresholds — enough to produce mixed-level cuts.
    private func views() -> [SignackView] {
        var v: [SignackView] = []
        for d: Float in [0.3, 1.2, 1.6, 2.5, 4, 8, 20, 60, 300] {
            for px: Float in [0.5, 1, 4, 16] {
                v.append(.perspective(position: SIMD3(d, 0.3 * d, 0.2), fovY: 0.8, viewportHeight: 1200, threshold: px))
            }
        }
        return v
    }

    func testLevelZeroClustersPartitionTheSource() {
        let m = SignackTestMeshes.icosphere(subdivisions: 5)
        let h = build(m)
        let level0 = h.clusters.enumerated().filter { $0.element.level == 0 }.map { UInt32($0.offset) }
        XCTAssertTrue(h.clusters.allSatisfy { $0.triangleCount <= 128 && $0.triangleCount > 0 })
        let source = Set(stride(from: 0, to: m.indices.count, by: 3).map { [m.indices[$0], m.indices[$0 + 1], m.indices[$0 + 2]].sorted() })
        let got = cutTriangles(h, level0).map { [$0.x, $0.y, $0.z].sorted() }
        XCTAssertEqual(got.count, source.count, "every source triangle exactly once")
        XCTAssertEqual(Set(got), source)
        XCTAssertGreaterThan(h.levelCount, 4)
    }

    func testErrorIsMonotoneUpTheDAG() {
        for m in [SignackTestMeshes.icosphere(subdivisions: 5), SignackTestMeshes.grid(80)] {
            let h = build(m)
            var byGroup: [Int32: (SignackSphere, Float)] = [:]
            for c in h.clusters {
                if c.level == 0 { XCTAssertEqual(c.lodError, 0) }
                guard !c.isRoot else { continue }
                XCTAssertGreaterThanOrEqual(c.parentError, c.lodError)
                XCTAssertTrue(SignackBounds.contains(c.parentSphere, c.lodSphere))
                if let g = byGroup[c.group] {
                    XCTAssertEqual(g.0, c.parentSphere); XCTAssertEqual(g.1, c.parentError)
                } else { byGroup[c.group] = (c.parentSphere, c.parentError) }
            }
        }
    }

    func testEveryCutIsCrackFree() {
        let meshes = [SignackTestMeshes.icosphere(subdivisions: 5),
                      SignackTestMeshes.uvSphere(meridians: 96, parallels: 48),
                      SignackTestMeshes.grid(90)]
        for m in meshes {
            let h = build(m)
            let src = cutTriangles(h, h.clusters.indices.filter { h.clusters[$0].level == 0 }.map(UInt32.init))
            let srcOpen = SignackEdges.openEdges(src, h.positionID)
            var mixed = 0
            var sizes = Set<Int>()
            for v in views() {
                let cut = SignackSelection.select(h, view: v)
                let tris = cutTriangles(h, cut)
                XCTAssertEqual(SignackEdges.openEdges(tris, h.positionID), srcOpen,
                               "cut at \(v.position) / \(v.threshold) px opens or closes an edge")
                let levels = Set(cut.map { h.clusters[Int($0)].level })
                if levels.count > 1 { mixed += 1 }
                sizes.insert(tris.count)
            }
            XCTAssertGreaterThan(mixed, 3, "the views must exercise mixed-level cuts")
            XCTAssertGreaterThan(sizes.count, 5)
        }
    }

    func testEdgeCollapseSharesTheSourceVertexBuffer() {
        let m = SignackTestMeshes.uvSphere(meridians: 96, parallels: 48)
        let h = build(m)
        XCTAssertTrue(h.derivedVertices.isEmpty)
        XCTAssertEqual(h.vertexCount, m.positions.count)
        XCTAssertTrue(h.indices.allSatisfy { Int($0) < m.positions.count })
        let coarse = SignackSelection.cut(h, tolerance: 0.05)
        XCTAssertLessThan(h.triangleCount(coarse), h.sourceTriangleCount / 4)
        // The whole DAG costs well under 3× the source's indices.
        XCTAssertLessThan(h.indices.count, 3 * m.indices.count)
    }

    func testFoliageCutsNeverDoubleACardAndSmallerTolerancesKeepMore() {
        let (m, atoms, cards) = SignackTestMeshes.cards(6000)
        let h = SignackHierarchyBuilder().build(
            positions: m.positions, indices: m.indices, atoms: atoms,
            simplifier: SignackCardSimplifier(cards: cards, seed: 7))
        XCTAssertGreaterThan(h.levelCount, 2)
        // Every derived vertex is an enlarged copy of a host vertex.
        for d in h.derivedVertices {
            XCTAssertLessThan(Int(d.source), m.positions.count)
            XCTAssertGreaterThan(d.scale, 1); XCTAssertLessThanOrEqual(d.scale, 1.4 + 1e-5)
        }
        let cardOf: (UInt32) -> Int = { v in
            let src = Int(v) < m.positions.count ? Int(v) : Int(h.derivedVertices[Int(v) - m.positions.count].source)
            return src / 4
        }
        var last = Int.max
        for tol: Float in [0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.5] {
            let cut = SignackSelection.cut(h, tolerance: tol)
            let tris = cutTriangles(h, cut)
            var seen: [Int: Int] = [:]
            for t in tris { seen[cardOf(t.x), default: 0] += 1 }
            XCTAssertTrue(seen.values.allSatisfy { $0 == 2 }, "a card drawn twice (or torn) at tol \(tol)")
            XCTAssertLessThanOrEqual(tris.count, last)
            last = tris.count
        }
        XCTAssertLessThan(last, 2 * 6000 / 4)
    }

    func testCutTrackerHoldsUntilTheCameraReallyMoves() {
        var t = SignackCutTracker()
        let c = SIMD3<Float>(0, 0, 0)
        func v(_ p: SIMD3<Float>) -> SignackView { .perspective(position: p, fovY: 0.8, viewportHeight: 1000) }
        XCTAssertTrue(t.shouldReselect(v(SIMD3(10, 0, 0)), center: c, radius: 1))
        XCTAssertFalse(t.shouldReselect(v(SIMD3(10.5, 0, 0.5)), center: c, radius: 1))
        XCTAssertTrue(t.shouldReselect(v(SIMD3(7, 0, 0)), center: c, radius: 1))
    }
}
