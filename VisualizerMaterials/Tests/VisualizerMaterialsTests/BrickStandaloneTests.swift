import XCTest
@testable import VisualizerMaterials

/// DH-1497 — clay facing brick on the shared laid-unit lattice. GPU-free.
final class BrickStandaloneTests: XCTestCase {

    /// The standard, as numbers: a 215 × 65 mm face with 10 mm joints (225 × 75 mm module), laid
    /// EXACTLY on an adjustable wall, in a running bond whose course count wraps.
    func testStandardBrickLaysAtTrueScaleOnAWall() {
        let p = BrickParams()
        let layout = p.layout(on: .baked(2.0))
        XCTAssertEqual(layout.realisedSizeMeters, 0.225, accuracy: 1e-9, "module = 215 mm face + 10 mm joint")
        XCTAssertEqual(layout.realisedSizeMeters / Double(p.pattern.shape.aspect), 0.075, accuracy: 1e-9,
                       "course = 65 mm face + 10 mm joint")
        XCTAssertEqual(layout.jointWidthMeters(p.pattern.jointFraction), 0.010, accuracy: 0.0005,
                       "the bake draws the 10 mm joint it was asked for (not the floor)")
        XCTAssertEqual(layout.rows % 2, 0, "a running bond needs an even course count to wrap")
        XCTAssertEqual(layout.rows, layout.cols * 3)
        XCTAssertEqual(p.pattern.bond, .running)
    }

    func testEveryPresetIsPlausible() {
        for (name, params) in BrickParams.presets {
            let ch = MaterialGenerator.brick(size: 256, params: params, tiling: .baked(2.0))
            let a = TextureAudit.audit(ch)
            XCTAssertTrue(a.isPlausible, "\(name): \(a)")
            XCTAssertEqual(ch.category, .brick)
            XCTAssertEqual(ch.diffuseRoughness, 1.0, "fired clay + mortar: OpenPBR's brick, 1.0")
            XCTAssertEqual(ch.clearcoat, 0)
        }
    }

    /// The running-bond cell declaration is only right if the engine's per-cell index
    /// (`MaterialChannels.patternCellIndex`, the shader's mirror) is CONSTANT over every brick the
    /// bake draws and different between neighbours — otherwise the per-brick tone step would cut a
    /// brick mid-face, the defect a bonded axis used to ship 0 to avoid.
    func testRunningBondCellIndexFollowsEveryBrick() {
        let size = 256
        let p = BrickParams()
        let layout = p.layout(on: .baked(2.0))
        let ch = MaterialGenerator.brick(size: size, params: p, tiling: .baked(2.0))
        XCTAssertEqual(ch.patternCells.x, -Double(layout.cols), "running bond declares a negative u count")
        XCTAssertEqual(ch.patternCells.y, Double(layout.rows))
        XCTAssertGreaterThan(ch.patternJitter, 0)
        let lattice = UnitLattice(pattern: p.pattern, layout: layout, size: size)
        var cellOf: [Int: (Double, Double)] = [:]
        var bricksSeen = Set<Int>()
        // Sample the surface over TWO tiles each way (uv keeps counting across the surface; only the
        // atlas lookup wraps), mapping each physical sample back to the bake's brick identity.
        for ty in 0..<2 { for tx in 0..<2 { for y in stride(from: 0, to: size, by: 3) { for x in stride(from: 0, to: size, by: 3) {
            let (u, v) = lattice.uv(x: x, y: y)
            let c = lattice.cell(u: u, v: v)
            guard c.joint < 0.01 else { continue }
            let su = u + Double(tx), sv = v + Double(ty)
            let idx = MaterialChannels.patternCellIndex(u: su, v: sv, cells: ch.patternCells)
            // The bake's own brick identity, UNWRAPPED across tiles: a brick on an offset course
            // straddles the u seam, so its two halves sit in neighbouring tiles under one key.
            let rawI = Int(floor(u * Double(layout.cols) + p.pattern.bond.offset(row: c.keyJ)))
            let key = (ty * layout.rows + c.keyJ) * 10_000 + (tx * layout.cols + rawI)
            bricksSeen.insert(key)
            if let prev = cellOf[key] {
                XCTAssertEqual(prev.0, idx.x, "brick \(c.keyI),\(c.keyJ): the hashed cell changes mid-brick")
                XCTAssertEqual(prev.1, idx.y)
            } else { cellOf[key] = (idx.x, idx.y) }
        } } } }
        let distinct = Set(cellOf.values.map { "\($0.0),\($0.1)" })
        XCTAssertEqual(distinct.count, bricksSeen.count, "every physical brick hashes a cell of its own")
    }

    /// The joint is a GROOVE: the mortar bed sits behind the brick face in the height field, and
    /// the relief is metric (a ~4 mm struck joint), so a raking light reads it as one.
    func testMortarIsRecessedBehindTheFace() {
        let size = 256
        let p = BrickParams()
        let layout = p.layout(on: .baked(2.0))
        let ch = MaterialGenerator.brick(size: size, params: p, tiling: .baked(2.0))
        let lattice = UnitLattice(pattern: p.pattern, layout: layout, size: size)
        var face = 0.0, nf = 0, joint = 0.0, nj = 0
        var mortarL = 0.0, faceL = 0.0
        for y in 0..<size { for x in 0..<size {
            let (u, v) = lattice.uv(x: x, y: y)
            let c = lattice.cell(u: u, v: v)
            let i = ch.idx(x, y)
            if c.rimMeters < lattice.jointHalfMeters * 0.5 { joint += ch.height[i]; nj += 1; mortarL += TextureAudit.luma(ch.albedo[i]) }
            else if c.rimMeters > lattice.jointHalfMeters + 0.008 { face += ch.height[i]; nf += 1; faceL += TextureAudit.luma(ch.albedo[i]) }
        } }
        let recess = (face / Double(nf) - joint / Double(nj)) * MaterialGenerator.brickReliefMeters
        XCTAssertGreaterThan(recess, 0.003, "the mortar bed sits ≥3 mm behind the face")
        XCTAssertLessThan(recess, 0.008)
        let mortarY = mortarL / Double(nj), faceY = faceL / Double(nf)
        XCTAssertLessThan(mortarY, 0.5, "mortar is a grey-tan, not tile's near-white grout")
        XCTAssertGreaterThan(mortarY, faceY, "on a red brick the mortar reads lighter than the clay")
    }
}
