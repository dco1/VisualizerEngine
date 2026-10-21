import XCTest
@testable import VisualizerMaterials

/// The modelling kit: mirror/array, path sweep, booleans, subdivision. Every op is judged by the VOLUME it must
/// have (a number a shape cannot fake), by winding (no inside-out face) and, where it promises it, by closure.
final class Mesh3ModellingTests: XCTestCase {

    // MARK: helpers

    private func box(_ lo: Vec3, _ hi: Vec3) -> Mesh3 {
        var m = Mesh3()
        let (x0, y0, z0, x1, y1, z1) = (lo.x, lo.y, lo.z, hi.x, hi.y, hi.z)
        m.addQuad(Vec3(x1,y0,z0), Vec3(x1,y1,z0), Vec3(x1,y1,z1), Vec3(x1,y0,z1), outward: Vec3( 1,0,0))
        m.addQuad(Vec3(x0,y0,z0), Vec3(x0,y0,z1), Vec3(x0,y1,z1), Vec3(x0,y1,z0), outward: Vec3(-1,0,0))
        m.addQuad(Vec3(x0,y1,z0), Vec3(x0,y1,z1), Vec3(x1,y1,z1), Vec3(x1,y1,z0), outward: Vec3(0, 1,0))
        m.addQuad(Vec3(x0,y0,z0), Vec3(x1,y0,z0), Vec3(x1,y0,z1), Vec3(x0,y0,z1), outward: Vec3(0,-1,0))
        m.addQuad(Vec3(x0,y0,z1), Vec3(x1,y0,z1), Vec3(x1,y1,z1), Vec3(x0,y1,z1), outward: Vec3(0,0, 1))
        m.addQuad(Vec3(x0,y0,z0), Vec3(x0,y1,z0), Vec3(x1,y1,z0), Vec3(x1,y0,z0), outward: Vec3(0,0,-1))
        return m
    }

    private func volume(_ m: Mesh3) -> Double {
        var v = 0.0, t = 0
        while t + 2 < m.indices.count {
            let a = m.positions[Int(m.indices[t])], b = m.positions[Int(m.indices[t + 1])], c = m.positions[Int(m.indices[t + 2])]
            v += dot3(a, cross3(b, c)) / 6; t += 3
        }
        return v
    }

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
                let lt = (a.x, a.y, a.z) < (b.x, b.y, b.z)
                count[lt ? E(a: a, b: b) : E(a: b, b: a), default: 0] += 1
            }
            t += 3
        }
        return count.values.filter { $0 != 2 }.count
    }

    /// Triangles wound against their own stored normals (inside-out).
    private func inverted(_ m: Mesh3) -> Int {
        var bad = 0, t = 0
        while t + 2 < m.indices.count {
            let i = (0..<3).map { Int(m.indices[t + $0]) }
            let n = cross3(m.positions[i[1]] - m.positions[i[0]], m.positions[i[2]] - m.positions[i[0]])
            if dot3(n, m.normals[i[0]] + m.normals[i[1]] + m.normals[i[2]]) <= 0 { bad += 1 }
            t += 3
        }
        return bad
    }

    private func polygonArea(_ n: Int, _ r: Double) -> Double { Double(n) / 2 * r * r * sin(2 * .pi / Double(n)) }

    // MARK: mirror / array

    func testMirrorKeepsTheSolidOutwardAndLandsOnTheOtherSide() {
        let m = box(Vec3(1, 0, 0), Vec3(2, 1, 3))
        let r = m.mirrored(.x)
        XCTAssertEqual(r.bounds!.min.x, -2, accuracy: 1e-9)
        XCTAssertEqual(r.bounds!.max.x, -1, accuracy: 1e-9)
        XCTAssertEqual(volume(r), volume(m), accuracy: 1e-9, "a mirrored solid must keep a POSITIVE volume")
        XCTAssertEqual(inverted(r), 0)
        // about a plane that is not the origin
        XCTAssertEqual(m.mirrored(.x, at: 3).bounds!.min.x, 4, accuracy: 1e-9)
    }

    func testArrays() {
        let m = box(Vec3(0, 0, 0), Vec3(1, 1, 1))
        let row = m.arrayed(count: 4, offset: Vec3(2, 0, 0))
        XCTAssertEqual(row.triangleCount, 4 * m.triangleCount)
        XCTAssertEqual(row.bounds!.max.x, 7, accuracy: 1e-9)
        let ring = box(Vec3(1, 0, -0.1), Vec3(1.2, 0.3, 0.1)).radialArray(count: 6)
        XCTAssertEqual(ring.triangleCount, 6 * 12)
        XCTAssertEqual(volume(ring), 6 * 0.2 * 0.3 * 0.2, accuracy: 1e-9)
        XCTAssertEqual(ring.bounds!.max.x, 1.2, accuracy: 0.02)
        XCTAssertEqual(ring.bounds!.min.x, -1.2, accuracy: 0.02)
    }

    // MARK: sweep

    func testStraightTubeHasTheProfilesVolume() {
        let n = 32, r = 0.01, L = 0.2
        let t = Mesh3.sweep(profile: Mesh3.circleProfile(radius: r, segments: n),
                            along: (0...10).map { Vec3(0, 0, L * Double($0) / 10) })
        XCTAssertEqual(volume(t), polygonArea(n, r) * L, accuracy: 1e-9)
        XCTAssertEqual(openEdges(t), 0, "a capped tube is closed")
        XCTAssertEqual(inverted(t), 0)
    }

    func testBentTubeFollowsThePathAndStaysClosed() {
        let n = 24, r = 0.01
        let path = Curve3.arc(centre: .zero, from: Vec3(0.1, 0, 0), toward: Vec3(0, 0.1, 0), radius: 0.1,
                              angle: .pi / 2, samples: 40)
        let t = Mesh3.sweep(profile: Mesh3.circleProfile(radius: r, segments: n), along: path)
        // Pappus: area × length of the centre line (a rotation-minimising frame does not shear the section)
        XCTAssertEqual(volume(t), polygonArea(n, r) * Curve3.length(path), accuracy: 0.01 * polygonArea(n, r) * Curve3.length(path))
        XCTAssertEqual(openEdges(t), 0)
        XCTAssertEqual(inverted(t), 0)
    }

    func testAClockwiseProfileIsTurnedRoundNotInsideOut() {
        let prof = Mesh3.circleProfile(radius: 0.01, segments: 16).reversed() as [Vec2]
        let t = Mesh3.sweep(profile: prof, along: [Vec3(0, 0, 0), Vec3(0, 0, 0.1), Vec3(0, 0, 0.2)])
        XCTAssertGreaterThan(volume(t), 0)
        XCTAssertEqual(inverted(t), 0)
    }

    func testTwistAndTaper() {
        // A twist turns each side of a square into a SKEW quad, and splitting a skew quad into two triangles shaves a
        // thin wedge off it: the volume error grows with (twist per station)². 4.5°/station lost 2.7 %; 1.1°/station
        // (81 stations over 90°) loses ~0.2 %. A twisted section needs stations to match the twist it asks for.
        let nT = 81
        let pathT = (0..<nT).map { Vec3(0, 0, 0.2 * Double($0) / Double(nT - 1)) }
        let square = Mesh3.roundedRectProfile(halfWidth: 0.01, halfHeight: 0.01, cornerRadius: 0.0001, corner: 1)
        let area = Mesh3.sweep(profile: square, along: pathT)
        let twisted = Mesh3.sweep(profile: square, along: pathT, twist: (0..<nT).map { Double($0) / Double(nT - 1) * .pi / 2 })
        XCTAssertEqual(volume(twisted), volume(area), accuracy: 0.01 * volume(area), "twisting a section keeps its area")
        XCTAssertEqual(openEdges(twisted), 0)
        let n = 21
        let path = (0..<n).map { Vec3(0, 0, 0.2 * Double($0) / Double(n - 1)) }
        // a cone: radius 0.02 → 0.01 over 0.2
        let cone = Mesh3.sweep(profile: Mesh3.circleProfile(radius: 0.02, segments: 48), along: path,
                               scale: (0..<n).map { 1 - 0.5 * Double($0) / Double(n - 1) })
        let frustum = Double.pi * 0.2 / 3 * (0.02 * 0.02 + 0.02 * 0.01 + 0.01 * 0.01)
        XCTAssertEqual(volume(cone), frustum, accuracy: 0.03 * frustum)
    }

    func testLoftOfIdenticalSectionsIsASweepAndOfShrinkingOnesIsAFrustum() {
        let n = 21, r = 0.02
        let path = (0..<n).map { Vec3(0, 0, 0.2 * Double($0) / Double(n - 1)) }
        let circle = Mesh3.circleProfile(radius: r, segments: 48)
        let same = Mesh3.loft(sections: Array(repeating: circle, count: n), along: path)
        XCTAssertEqual(volume(same), volume(Mesh3.sweep(profile: circle, along: path)), accuracy: 1e-12)
        // sections that shrink 0.02 → 0.01, each its OWN array
        let cone = Mesh3.loft(sections: (0..<n).map { i in
            Mesh3.circleProfile(radius: r * (1 - 0.5 * Double(i) / Double(n - 1)), segments: 48)
        }, along: path)
        let frustum = Double.pi * 0.2 / 3 * (0.02 * 0.02 + 0.02 * 0.01 + 0.01 * 0.01)
        XCTAssertEqual(volume(cone), frustum, accuracy: 0.03 * frustum)
        XCTAssertEqual(openEdges(cone), 0)
        XCTAssertEqual(inverted(cone), 0)
        // mismatched section sizes are refused, not mangled
        var bad = Array(repeating: circle, count: n); bad[3] = Mesh3.circleProfile(radius: r, segments: 12)
        XCTAssertTrue(Mesh3.loft(sections: bad, along: path).isEmpty)
    }

    func testInvertedIsACavity() {
        let m = box(Vec3(0, 0, 0), Vec3(1, 1, 1))
        let cav = m.inverted()
        XCTAssertEqual(volume(cav), -volume(m), accuracy: 1e-12, "an inside-out solid has negative volume")
        XCTAssertEqual(cav.triangleCount, m.triangleCount)
        XCTAssertEqual(openEdges(cav), 0)
        XCTAssertEqual(inverted(cav), 0, "its stored normals still agree with its (reversed) winding")
    }

    func testCatmullRomPassesThroughItsPoints() {
        let pts = [Vec3(0, 0, 0), Vec3(0.1, 0.05, 0), Vec3(0.2, 0, 0.05), Vec3(0.3, -0.02, 0)]
        let c = Curve3.catmullRom(pts, samples: 10)
        for p in pts { XCTAssertLessThan(c.map { len3($0 - p) }.min()!, 1e-9) }
    }

    func testCylinderIsSmoothWithinItsChordErrorAndClosed() {
        let r = 0.0055
        let n = Mesh3.circleSegments(radius: r)
        XCTAssertEqual(n % 4, 0)
        XCTAssertLessThanOrEqual(r * (1 - cos(.pi / Double(n))), 0.00005 + 1e-12, "within the chord error")
        let c = Mesh3.cylinder(from: Vec3(0, 0, 0), to: Vec3(0, 0.03, 0), radius: r)
        XCTAssertEqual(openEdges(c), 0)
        XCTAssertEqual(inverted(c), 0)
        // …and a bore cut with it keeps SMOOTH wall normals (the booleans interpolate normals across their cuts)
        let block = box(Vec3(-0.02, 0, -0.02), Vec3(0.02, 0.03, 0.02))
        let bored = block.subtracting(Mesh3.cylinder(from: Vec3(0, -0.005, 0), to: Vec3(0, 0.035, 0), radius: r))
        // The wall = triangles whose corners all sit in the band between the facets' midpoint radius and the true radius
        // (a clipped wall quad also has points ON its diagonal chord, slightly inside the circle; the flat faces'
        // triangles all reach out to the block's own edges, so they never qualify). On a SMOOTH cylinder a ring vertex's
        // normal is exactly anti-radial (dot −1); on a flat-shaded one it is the facet's normal, half a step off (dot cos 7.5°).
        let lo = r * cos(.pi / Double(n)) - 1e-9
        var checked = 0, smooth = 0
        var t = 0
        while t + 2 < bored.indices.count {
            let i = (0..<3).map { Int(bored.indices[t + $0]) }
            t += 3
            guard i.allSatisfy({ let h = hypot(bored.positions[$0].x, bored.positions[$0].z); return h >= lo && h <= r + 1e-9 }) else { continue }
            for k in i {
                let p = bored.positions[k], h = hypot(p.x, p.z)
                guard abs(h - r) < 1e-9 else { continue }                  // a ring vertex
                checked += 1
                let radial = Vec3(p.x / h, 0, p.z / h)
                // the wall faces OUT of the metal, i.e. into the hole: exactly anti-radial when smooth
                if dot3(bored.normals[k], radial) < -0.9999 { smooth += 1 }
            }
        }
        XCTAssertGreaterThan(checked, 20, "the bore's wall must be in the result")
        XCTAssertEqual(smooth, checked, "bore walls face into the hole, smoothly")
    }

    // MARK: booleans

    func testBooleanVolumes() {
        let a = box(Vec3(0, 0, 0), Vec3(1, 1, 1)), b = box(Vec3(0.5, 0.5, 0.5), Vec3(1.5, 1.5, 1.5))
        XCTAssertEqual(volume(a.intersection(b)), 0.125, accuracy: 1e-6)
        XCTAssertEqual(volume(a.union(b)), 1.875, accuracy: 1e-6)
        XCTAssertEqual(volume(a.subtracting(b)), 0.875, accuracy: 1e-6)
        for r in [a.intersection(b), a.union(b), a.subtracting(b)] {
            XCTAssertEqual(inverted(r), 0, "every boolean face must face outward")
        }
    }

    func testADrilledHole() {
        let block = box(Vec3(-0.05, -0.05, 0), Vec3(0.05, 0.05, 0.04))
        let n = 24, r = 0.01
        let drill = Mesh3.sweep(profile: Mesh3.circleProfile(radius: r, segments: n),
                                along: [Vec3(0, 0, -0.01), Vec3(0, 0, 0.05)])     // runs clear through, no coplanar faces
        let drilled = block.subtracting(drill)
        XCTAssertEqual(volume(drilled), 0.1 * 0.1 * 0.04 - polygonArea(n, r) * 0.04, accuracy: 1e-8)
        XCTAssertEqual(inverted(drilled), 0)
    }

    /// Chained cuts on an already-fragmented mesh (the demo block: bolt circle + bore + slot through two of the holes).
    /// Ground truth is a Monte-Carlo count against the analytic solid, so the test does not grade the BSP with itself.
    func testChainedCutsMatchTheAnalyticSolid() {
        func cutter(_ r: Double, at x: Double, n: Int = 20) -> Mesh3 {
            Mesh3.sweep(profile: Mesh3.circleProfile(radius: r, segments: n), along: [Vec3(x, -0.005, 0), Vec3(x, 0.035, 0)])
        }
        let block = box(Vec3(-0.045, 0, -0.035), Vec3(0.045, 0.03, 0.035))
        let bolts = cutter(0.0065, at: 0.027).radialArray(count: 4, startAngle: .pi / 4)
        let bore = cutter(0.011, at: 0)
        let slot = Mesh3.sweep(profile: [Vec2(-0.055, -0.004), Vec2(0.055, -0.004), Vec2(0.055, 0.004), Vec2(-0.055, 0.004)],
                               along: [Vec3(0, 0.021, 0), Vec3(0, 0.036, 0)]).rotatedY(.pi / 4)

        // the analytic solid, with equal-area circles standing in for the 20-gons
        let req = { (r: Double) in r * (20.0 / 2 * sin(2 * .pi / 20) / .pi).squareRoot() }
        func inside(_ p: Vec3) -> Bool {
            guard abs(p.x) < 0.045, abs(p.z) < 0.035, p.y > 0, p.y < 0.03 else { return false }
            if hypot(p.x, p.z) < req(0.011) { return false }
            for k in 0..<4 {
                let a = .pi / 4 + Double(k) * .pi / 2
                if hypot(p.x - 0.027 * cos(a), p.z + 0.027 * sin(a)) < req(0.0065) { return false }
            }
            // the slot: 8 mm wide, 110 mm long along the 45° diagonal (the rotatedY maps its long axis to z→x/z mix)
            let c = cos(Double.pi / 4), s = sin(Double.pi / 4)
            let lx = c * p.x - s * p.z, lz = s * p.x + c * p.z          // into the slot's own frame
            if p.y > 0.021, abs(lx) < 0.004, abs(lz) < 0.055 { return false }
            return true
        }
        var rng = SystemRandomNumberGenerator()
        var hits = 0
        let N = 400_000
        for _ in 0..<N {
            let p = Vec3(Double.random(in: -0.045...0.045, using: &rng), Double.random(in: 0...0.03, using: &rng),
                         Double.random(in: -0.035...0.035, using: &rng))
            if inside(p) { hits += 1 }
        }
        let truth = 0.09 * 0.03 * 0.07 * Double(hits) / Double(N)

        let a = block.subtracting(bolts).subtracting(bore).subtracting(slot)
        let b = block.subtracting(slot).subtracting(bolts).subtracting(bore)
        let c = block.subtracting(bolts.union(bore).union(slot))
        print("CHAIN truth \(truth)  holes-then-slot \(volume(a))  slot-first \(volume(b))  union-cutters \(volume(c))")
        XCTAssertEqual(volume(b), truth, accuracy: 0.02 * truth, "slot first, then holes")
        XCTAssertEqual(volume(a), truth, accuracy: 0.02 * truth, "holes first, then the slot")
        XCTAssertEqual(volume(c), truth, accuracy: 0.02 * truth, "cutters unioned, one subtraction")
    }

    /// DH-1448: a small cutter must not slice the rest of a big, finely tessellated body into slivers.
    func testLocalisedCutDoesNotFragmentTheRestOfTheBody() {
        let body = Mesh3.sweep(profile: Mesh3.circleProfile(radius: 0.05, segments: 48),
                               along: (0...40).map { Vec3(0, 0, 0.2 * Double($0) / 40) })
        let cutter = box(Vec3(-0.01, 0.03, 0.18), Vec3(0.01, 0.06, 0.21))             // pokes through the top near one end
        let cut = body.subtracting(cutter)
        XCTAssertLessThan(cut.triangleCount, body.triangleCount + 120,
                          "body \(body.triangleCount) → \(cut.triangleCount): the far polygons must pass through untouched")
        XCTAssertGreaterThan(cut.triangleCount, body.triangleCount - 20)
        // …and the volume is still the volume of the cut: body minus the wedge of the cutter that lies inside it
        XCTAssertLessThan(volume(cut), volume(body))
        XCTAssertEqual(inverted(cut), 0)
        // union and intersection stay correct under localisation
        XCTAssertEqual(volume(body.intersection(cutter)), volume(body) - volume(cut), accuracy: 2e-7)
        XCTAssertEqual(volume(body.union(cutter)), volume(body) + volume(cutter) - volume(body.intersection(cutter)), accuracy: 2e-7)
    }

    func testBooleanAgainstNothing() {
        let a = box(Vec3(0, 0, 0), Vec3(1, 1, 1))
        XCTAssertEqual(a.union(Mesh3()), a)
        XCTAssertEqual(a.subtracting(Mesh3()), a)
        XCTAssertTrue(a.intersection(Mesh3()).isEmpty)
        XCTAssertEqual(volume(a.subtracting(box(Vec3(5, 5, 5), Vec3(6, 6, 6)))), 1, accuracy: 1e-9, "a disjoint cutter changes nothing")
    }

    // MARK: subdivision

    func testSmoothSubdivisionRoundsAClosedCage() {
        let cage = box(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        let s = cage.subdivided(levels: 2, creaseDegrees: 120)           // nothing is creased: a cube becomes a blob
        XCTAssertEqual(s.triangleCount, cage.triangleCount * 16)
        XCTAssertEqual(openEdges(s), 0, "subdividing a closed cage keeps it closed")
        XCTAssertEqual(inverted(s), 0)
        XCTAssertLessThan(volume(s), 0.8 * 8, "a smooth surface shrinks inside its cage")
        XCTAssertGreaterThan(volume(s), 0.4 * 8)
        XCTAssertLessThan(s.bounds!.max.x, 1.0 - 0.01)
    }

    func testCreasedEdgesStayAndFlatFacesStayFlat() {
        let cage = box(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        let s = cage.subdivided(levels: 2, creaseDegrees: 35)            // every 90° edge is a crease
        XCTAssertEqual(volume(s), 8, accuracy: 1e-6, "a fully creased flat-faced cage is unchanged")
        XCTAssertEqual(s.bounds!.max.x, 1, accuracy: 1e-9)
        XCTAssertEqual(openEdges(s), 0)
    }

    func testPartlyCreasedCageRoundsOnlyWhereItIsAllowedTo() {
        // a slab whose two long edges stay sharp (crease), ends rounded: build as a box then subdivide with a tolerance
        // that only the 90° edges exceed — every edge of a box is 90°, so instead assert the principle on a tube:
        // a tube's caps meet its wall at 90° (creased) while the wall's own facets (11.25°) round away.
        let tube = Mesh3.sweep(profile: Mesh3.circleProfile(radius: 0.02, segments: 32),
                               along: [Vec3(0, 0, 0), Vec3(0, 0, 0.1), Vec3(0, 0, 0.2)])
        let s = tube.subdivided(levels: 1, creaseDegrees: 35)
        XCTAssertEqual(s.bounds!.min.z, 0, accuracy: 1e-9, "the cap ring is a crease and does not creep inward")
        XCTAssertEqual(s.bounds!.max.z, 0.2, accuracy: 1e-9)
        XCTAssertEqual(openEdges(s), 0)
        XCTAssertEqual(inverted(s), 0)
    }
}
