import simd
import XCTest
import VisualizerMaterials
@testable import VisualizerVegetation

/// The painted-foliage building blocks: a colour that stays with ITS vertex, and a sheet leaf that
/// is two sound faces.
final class PaintedMeshTests: XCTestCase {

    private let red = Vec3(1, 0, 0), green = Vec3(0, 1, 0), blue = Vec3(0, 0, 1)

    /// `Mesh3.addTriangle` swaps the last two corners when the given order faces away from
    /// `outward`. The paint must swap with them — a colour array appended in the GIVEN order would
    /// put b's colour on c from the first rewound triangle on.
    func testPaintFollowsItsVertexThroughTheWindingSwap() {
        let a = Vec3(0, 0, 0), b = Vec3(1, 0, 0), c = Vec3(0, 1, 0)   // CCW about +z
        for outward in [Vec3(0, 0, 1), Vec3(0, 0, -1)] {           // as given, then rewound
            var p = PaintedMesh()
            p.addTriangle(a, b, c, colors: red, green, blue, outward: outward)
            XCTAssertEqual(p.colors.count, p.mesh.vertexCount)
            for (i, pos) in p.mesh.positions.enumerated() {
                let want = pos == a ? red : (pos == b ? green : blue)
                XCTAssertEqual(p.colors[i], want, "vertex \(pos) wears \(p.colors[i]) (outward \(outward))")
            }
        }
    }

    /// A degenerate triangle adds nothing — no geometry and no paint — so the arrays stay parallel.
    func testADegenerateTriangleAddsNoPaint() {
        var p = PaintedMesh()
        p.addTriangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0), colors: red, green, blue,
                      outward: Vec3(0, 0, 1))
        XCTAssertTrue(p.mesh.isEmpty)
        XCTAssertTrue(p.colors.isEmpty)
    }

    /// Smoothing and appending keep vertex ORDER, so the paint stays aligned through both.
    func testSmoothingAndAppendingKeepThePaintAligned() {
        var p = PaintedMesh()
        p.addQuad(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0.2), Vec3(0, 1, 0.2),
                  colors: red, green, blue, red, outward: Vec3(0, 0, 1))
        var q = p
        q.append(p.smoothed())
        XCTAssertEqual(q.colors.count, q.mesh.vertexCount)
        XCTAssertEqual(Array(q.colors[p.colors.count...]), p.colors)
    }

    /// A sheet blade is two faces over ONE surface: every upper-face triangle faces the upper side
    /// of the surface, every lower-face one the other, and it is painted per face.
    func testSheetBladeIsTwoFacedAndPaintedPerFace() {
        let s = LeafSheet.Surface(base: Vec3(0, 0, 0), xAxis: Vec3(1, 0, 0), yAxis: Vec3(0, 0, -1),
                                  normal: Vec3(0, 1, 0), length: 0.3, halfWidth: 0.38)
        var p = PaintedMesh()
        LeafSheet.emitBlade(into: &p, surface: s,
                            margin: LeafSilhouette.fiddleLeafFigBlade.margin(tier: .hero, subdivisions: 2),
                            aspect: 0.76) { _, face in face == .upper ? self.green : self.red }
        XCTAssertGreaterThan(p.triangleCount, 100)
        var up = 0, down = 0
        for t in 0 ..< p.triangleCount {
            let i = (0 ..< 3).map { Int(p.mesh.indices[t * 3 + $0]) }
            let n = cross(p.mesh.positions[i[1]] - p.mesh.positions[i[0]], p.mesh.positions[i[2]] - p.mesh.positions[i[0]])
            let paint = p.colors[i[0]]
            XCTAssertTrue(i.allSatisfy { p.colors[$0] == paint }, "one face, one paint")
            if paint == green { XCTAssertGreaterThan(n.y, 0, "upper face faces up"); up += 1 }
            else { XCTAssertLessThan(n.y, 0, "lower face faces down"); down += 1 }
        }
        XCTAssertEqual(up, down, "both faces are emitted, triangle for triangle")
    }

    /// The monstera's cuts are cells left out: a mature half-blade has less area than the whole
    /// heart, and a juvenile blade's cut list is empty.
    func testMonsteraCutsRemoveAreaAndAJuvenileLeafIsEntire() {
        let s = LeafSheet.Surface(base: Vec3(0, 0, 0), xAxis: Vec3(1, 0, 0), yAxis: Vec3(0, 0, -1),
                                  normal: Vec3(0, 1, 0), length: 1, halfWidth: 0.48)
        func area(_ cuts: LeafSheet.MonsteraCuts) -> Double {
            var p = PaintedMesh()
            LeafSheet.emitMonsteraHalf(into: &p, surface: s, cuts: cuts, side: 1) { _, _ in self.green }
            var a = 0.0
            for t in 0 ..< p.triangleCount {
                let i = (0 ..< 3).map { Int(p.mesh.indices[t * 3 + $0]) }
                a += simd_length(cross(p.mesh.positions[i[1]] - p.mesh.positions[i[0]],
                                       p.mesh.positions[i[2]] - p.mesh.positions[i[0]])) / 2
            }
            return a
        }
        var rng = PottedPlantMesh.SplitMix(3)
        let mature = PottedPlantMesh.monsteraCuts(maturity: 1, rng: &rng)
        XCTAssertFalse(mature.splits.isEmpty)
        XCTAssertLessThan(area(mature), area(.init(splits: [], holes: [])) * 0.95)
        var young = PottedPlantMesh.SplitMix(3)
        XCTAssertTrue(PottedPlantMesh.monsteraCuts(maturity: 0.1, rng: &young).splits.isEmpty)
    }
}
