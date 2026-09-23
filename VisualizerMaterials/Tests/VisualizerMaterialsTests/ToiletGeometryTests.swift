import XCTest
@testable import VisualizerMaterials

/// The shared toilet's structural promises — each one a defect a render once showed.
final class ToiletGeometryTests: XCTestCase {
    let parts = ToiletGeometry.build(.roundFront)

    /// Every part exists and has the fixture's real extents (ANSI round front: 0.40 × 0.679 m).
    func testPartsAndExtents() {
        for (name, m) in [("pan", parts.pan), ("water", parts.water), ("seat", parts.seat),
                          ("lid", parts.lid), ("cistern", parts.cistern), ("flush", parts.flush)] {
            XCTAssertFalse(m.isEmpty, "\(name) is empty")
            XCTAssertEqual(m.normals.count, m.positions.count, "\(name) normals")
        }
        var all = parts.pan
        for m in [parts.seat, parts.lid, parts.cistern, parts.flush] { all.append(m) }
        let b = all.bounds!
        XCTAssertEqual(b.max.x - b.min.x, 0.40, accuracy: 0.02)
        XCTAssertEqual(b.max.z - b.min.z, ToiletGeometry.Dimensions.roundFront.depth, accuracy: 0.02)
        XCTAssertEqual(b.max.y, 0.784, accuracy: 0.012)
    }

    /// The water is a pool INSIDE the pan, facing up — warped at the pan's own scale (warped at
    /// 1, `ovoidPlan`'s own-radius normalisation made it rim-wide: a flange through the walls).
    func testWaterIsInsideThePanAndFacesUp() {
        let w = parts.water.bounds!, p = parts.pan.bounds!
        XCTAssertLessThan(w.max.x - w.min.x, 0.6 * (p.max.x - p.min.x), "water wider than the pan's cavity")
        XCTAssertTrue(parts.water.normals.allSatisfy { $0.y > 0.99 }, "water must face up")
    }

    /// The pan is HOLLOW: nothing of the fixture's porcelain other than the pan's own inner wall
    /// rises above the water inside the cavity (the trapway shelf once stood up in it as a box).
    func testCavityIsClear() {
        let water = parts.water.bounds!
        let cx = (water.min.x + water.max.x) / 2, cz = (water.min.z + water.max.z) / 2
        let rx = (water.max.x - water.min.x) / 2, rz = (water.max.z - water.min.z) / 2
        let waterY = water.max.y
        let intruders = parts.cistern.positions.filter { q in
            let u = (q.x - cx) / rx, v = (q.z - cz) / rz
            return u * u + v * v < 0.9 && q.y > waterY + 0.001
        }
        XCTAssertTrue(intruders.isEmpty, "\(intruders.count) cistern vertices stand inside the bowl above the water")
    }

    /// The pan's inner wall faces the axis (out of the porcelain, into the cavity) — the hollow
    /// lathe is one sheet, so an inside-out inner wall would render invisible.
    func testInnerWallFacesIntoTheCavity() {
        let water = parts.water.bounds!
        let cx = (water.min.x + water.max.x) / 2, cz = (water.min.z + water.max.z) / 2
        let waterY = water.max.y, rimY = parts.hinge.y - 0.028
        var inner = 0, wrong = 0
        for (i, q) in parts.pan.positions.enumerated() where q.y > waterY + 0.01 && q.y < rimY - 0.02 {
            let radial = Vec3(q.x - cx, 0, q.z - cz)
            let n = parts.pan.normals[i]
            // Inner-wall vertices: close to the axis for their height.
            guard (radial.x * radial.x + radial.z * radial.z).squareRoot() < 0.12 else { continue }
            inner += 1
            if n.x * radial.x + n.z * radial.z > 0 { wrong += 1 }
        }
        XCTAssertGreaterThan(inner, 20)
        XCTAssertEqual(wrong, 0, "inner-wall normals pointing away from the cavity")
    }

    /// Opened, the lid comes to rest against the cistern just past upright — touching, not in it.
    func testOpenLidRestsAgainstTheCistern() {
        let a = ToiletGeometry.openAngle(parts)
        XCTAssertGreaterThan(a, 90 * .pi / 180)
        XCTAssertLessThan(a, 125 * .pi / 180, "the solver fell through to its fallback")
        let back = parts.lid.positions.map { ToiletGeometry.hinged($0, angle: a - 0.01, in: parts) }
        XCTAssertFalse(back.contains { $0.y < parts.tankLidBottomY && $0.z > parts.tankFrontZ },
                       "a hair before rest the lid is already inside the cistern")
    }

    /// Up-facing area of the triangles lying wholly in the plane y = `y`.
    private func planeArea(_ m: Mesh3, y: Double) -> (area: Double, allUp: Bool) {
        var area = 0.0, allUp = true
        for t in stride(from: 0, to: m.indices.count, by: 3) {
            let a = m.positions[Int(m.indices[t])], b = m.positions[Int(m.indices[t + 1])], c = m.positions[Int(m.indices[t + 2])]
            guard abs(a.y - y) < 1e-9, abs(b.y - y) < 1e-9, abs(c.y - y) < 1e-9 else { continue }
            let n = cross3(b - a, c - a)
            area += len3(n) / 2
            if n.y <= 0 { allUp = false }
            for k in 0 ..< 3 where m.normals[Int(m.indices[t + k])].y < 0.9999 { allUp = false }
        }
        return (area, allUp)
    }

    /// The cistern lid's deck is truly flat (every deck triangle faces straight up), and the tank
    /// under it stops at the lid's underside: the cistern's top plane holds the lid's deck ALONE.
    /// (Two coplanar tops, triangulated differently, z-fought as an X across the lid.)
    func testCisternLidDeckIsFlatAndNotCoplanarWithTheTank() {
        let slab = ToiletGeometry.roundedSlab(halfW: 0.206, halfD: 0.11, y0: 0, y1: 0.025,
                                              topEdge: 0.010, planCorner: 0.014)
        let deck = planeArea(slab, y: 0.025)
        XCTAssertGreaterThan(deck.area, 0.05)
        XCTAssertTrue(deck.allUp, "deck triangles must all face straight up")
        XCTAssertLessThan(slab.indices.count / 3, 220, "facets sized by the corner radius, not the half-diagonal")
        let topY = parts.cistern.positions.map(\.y).max()!
        let top = planeArea(parts.cistern, y: topY)
        // The lid deck alone: (2·(0.206−0.01)) × (2·(0.11−0.01)) ≈ 0.078 m², less its corners.
        XCTAssertLessThan(top.area, 0.085, "a second surface shares the cistern's top plane (z-fight)")
    }

    /// The hinge sits at the back of the seat stack, in front of the cistern.
    func testHingeLine() {
        XCTAssertLessThan(parts.hinge.z, parts.tankFrontZ)
        let seatTop = parts.seat.positions.map(\.y).max()!
        XCTAssertEqual(parts.hinge.y, seatTop, accuracy: 0.002)
    }
}
