import XCTest
@testable import VisualizerMaterials

final class MeshRemeshTests: XCTestCase {

    private func stats(_ m: Mesh3, _ L: Double) -> (mean: Double, inBand: Double) {
        let e = MeshKit.edgeLengths(m)
        return (e.reduce(0, +) / Double(e.count), Double(e.filter { $0 > 0.7 * L && $0 < 1.3 * L }.count) / Double(e.count))
    }

    func testCoarsenSphereToTargetEdgeLength() {
        let sphere = MeshKit.icosphere(levels: 4)           // edge ~0.066
        let L = 0.15
        let t0 = Date()
        let r = sphere.isotropicRemeshed(targetEdgeLength: L, iterations: 6)
        print("remesh coarsen \(sphere.triangleCount) -> \(r.triangleCount) in \(Date().timeIntervalSince(t0))s")
        let s = stats(r, L); print("coarsen mean \(s.mean) inBand \(s.inBand)")
        XCTAssertEqual(s.mean / L, 1, accuracy: 0.2)
        XCTAssertGreaterThan(s.inBand, 0.85)
        XCTAssertEqual(MeshKit.openEdges(r), 0)
        XCTAssertEqual(MeshKit.volume(r) / MeshKit.volume(sphere), 1, accuracy: 0.03)
        XCTAssertLessThan(r.positions.map { abs(len3($0) - 1) }.max()!, 0.02, "stays on the original surface")
    }

    func testRefineSphereToTargetEdgeLength() {
        let sphere = MeshKit.icosphere(levels: 2)           // edge ~0.26
        let L = 0.1
        let t0 = Date()
        let r = sphere.isotropicRemeshed(targetEdgeLength: L, iterations: 6)
        print("remesh refine \(sphere.triangleCount) -> \(r.triangleCount) in \(Date().timeIntervalSince(t0))s")
        let s = stats(r, L); print("refine mean \(s.mean) inBand \(s.inBand)")
        XCTAssertEqual(s.mean / L, 1, accuracy: 0.2)
        XCTAssertGreaterThan(s.inBand, 0.85)
        XCTAssertEqual(MeshKit.openEdges(r), 0)
        // projected onto the ORIGINAL (faceted) surface, so inside the unit sphere by at most its sagitta
        XCTAssertGreaterThan(r.positions.map { len3($0) }.min()!, 0.9)
    }

    func testRemeshKeepsOpenBoundary() {
        let plane = MeshKit.plane(n: 10)
        let r = plane.isotropicRemeshed(targetEdgeLength: 0.2, iterations: 4)
        XCTAssertEqual(MeshKit.area(r), 1, accuracy: 1e-9)
        for i in MeshKit.boundaryVertices(r) {
            let p = r.positions[i]
            XCTAssertTrue(p.x < 1e-12 || p.x > 1 - 1e-12 || p.z < 1e-12 || p.z > 1 - 1e-12)
        }
        let s = stats(r, 0.2); print("plane mean \(s.mean) inBand \(s.inBand)")
        XCTAssertGreaterThan(s.inBand, 0.7)
    }
}
