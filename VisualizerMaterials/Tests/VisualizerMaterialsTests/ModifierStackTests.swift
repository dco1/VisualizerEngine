import XCTest
@testable import VisualizerMaterials

/// Modelling-kit item E: the modifier protocol, the stack and each modifier's headline guarantee.
final class ModifierStackTests: XCTestCase {

    // MARK: builders / measures

    /// A w×d sheet in the XZ plane at y = 0 facing +Y, divided nx × nz.
    private func plane(_ w: Double, _ d: Double, nx: Int = 8, nz: Int = 4) -> Mesh3 {
        var m = Mesh3()
        for i in 0..<nx { for k in 0..<nz {
            let x0 = w * Double(i) / Double(nx) - w / 2, x1 = w * Double(i + 1) / Double(nx) - w / 2
            let z0 = d * Double(k) / Double(nz) - d / 2, z1 = d * Double(k + 1) / Double(nz) - d / 2
            m.addQuad(Vec3(x0, 0, z0), Vec3(x0, 0, z1), Vec3(x1, 0, z1), Vec3(x1, 0, z0), outward: Vec3(0, 1, 0))
        } }
        return m
    }

    /// A sheet in the XY plane (x −0.5…0.5, y 0…1) facing +Z, n × n.
    private func upright(n: Int) -> Mesh3 {
        var m = Mesh3()
        for i in 0..<n { for j in 0..<n {
            let x0 = Double(i) / Double(n) - 0.5, x1 = Double(i + 1) / Double(n) - 0.5
            let y0 = Double(j) / Double(n), y1 = Double(j + 1) / Double(n)
            m.addQuad(Vec3(x0, y0, 0), Vec3(x1, y0, 0), Vec3(x1, y1, 0), Vec3(x0, y1, 0), outward: Vec3(0, 0, 1))
        } }
        return m
    }

    /// Unit cube centred on the origin in x/z, y 0…1, each face n × n.
    private func box(n: Int) -> Mesh3 {
        var m = Mesh3()
        func face(_ o: Vec3, _ u: Vec3, _ v: Vec3, _ out: Vec3) {
            for i in 0..<n { for j in 0..<n {
                let a = Double(i) / Double(n), b = Double(i + 1) / Double(n), c = Double(j) / Double(n), d = Double(j + 1) / Double(n)
                m.addQuad(o + u * a + v * c, o + u * b + v * c, o + u * b + v * d, o + u * a + v * d, outward: out)
            } }
        }
        let o = Vec3(-0.5, 0, -0.5), x = Vec3(1, 0, 0), y = Vec3(0, 1, 0), z = Vec3(0, 0, 1)
        face(o, x, y, Vec3(0, 0, -1)); face(o + z, x, y, Vec3(0, 0, 1))
        face(o, z, y, Vec3(-1, 0, 0)); face(o + x, z, y, Vec3(1, 0, 0))
        face(o, x, z, Vec3(0, -1, 0)); face(o + y, x, z, Vec3(0, 1, 0))
        return m
    }

    private func sphere(r: Double, rings: Int = 20, segs: Int = 32) -> Mesh3 {
        func p(_ i: Int, _ j: Int) -> Vec3 {
            let th = Double.pi * Double(i) / Double(rings), ph = 2 * Double.pi * Double(j) / Double(segs)
            return Vec3(r * sin(th) * cos(ph), r * cos(th), r * sin(th) * sin(ph))
        }
        var m = Mesh3()
        for i in 0..<rings { for j in 0..<segs {
            let a = p(i, j), b = p(i + 1, j), c = p(i + 1, j + 1), d = p(i, j + 1)
            let out = normalize3(a + b + c + d)
            m.addTriangle(a, b, c, outward: out); m.addTriangle(a, c, d, outward: out)
        } }
        return m
    }

    private func volume(_ m: Mesh3) -> Double {
        var v = 0.0, t = 0
        while t + 2 < m.indices.count {
            v += dot3(m.positions[Int(m.indices[t])], cross3(m.positions[Int(m.indices[t + 1])], m.positions[Int(m.indices[t + 2])])) / 6
            t += 3
        }
        return v
    }

    /// Undirected welded edges not shared by exactly two triangles, and directed edges used twice the same way.
    private func openEdges(_ m: Mesh3) -> (open: Int, misoriented: Int) {
        func key(_ p: Vec3) -> SIMD3<Int64> {
            SIMD3(Int64((p.x * 1e5).rounded()), Int64((p.y * 1e5).rounded()), Int64((p.z * 1e5).rounded()))
        }
        struct E: Hashable { let a: SIMD3<Int64>, b: SIMD3<Int64> }
        var und: [E: Int] = [:], dir: [E: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            let k = (0..<3).map { key(m.positions[Int(m.indices[t + $0])]) }
            for i in 0..<3 {
                let a = k[i], b = k[(i + 1) % 3]
                dir[E(a: a, b: b), default: 0] += 1
                und[(a.x, a.y, a.z) < (b.x, b.y, b.z) ? E(a: a, b: b) : E(a: b, b: a), default: 0] += 1
            }
            t += 3
        }
        return (und.values.filter { $0 != 2 }.count, dir.values.filter { $0 > 1 }.count)
    }

    /// Triangles whose winding disagrees with their stored normals.
    private func invertedFaces(_ m: Mesh3) -> Int {
        var bad = 0, t = 0
        while t + 2 < m.indices.count {
            let i = (0..<3).map { Int(m.indices[t + $0]) }
            let n = cross3(m.positions[i[1]] - m.positions[i[0]], m.positions[i[2]] - m.positions[i[0]])
            if dot3(n, m.normals[i[0]] + m.normals[i[1]] + m.normals[i[2]]) <= 0 { bad += 1 }
            t += 3
        }
        return bad
    }

    private func edgeLengths(_ m: Mesh3) -> [Double] {
        var out: [Double] = [], t = 0
        while t + 2 < m.indices.count {
            let p = (0..<3).map { m.positions[Int(m.indices[t + $0])] }
            out += [dist3(p[0], p[1]), dist3(p[1], p[2]), dist3(p[2], p[0])]
            t += 3
        }
        return out
    }

    // MARK: Solidify

    func testSolidifyPlaneIsClosedVolumeAreaTimesThickness() {
        let sheet = plane(2, 1)
        for offset in [-1.0, 0, 1] {
            let s = SolidifyModifier(thickness: 0.1, offset: offset).apply(sheet)
            let e = openEdges(s)
            XCTAssertEqual(e.open, 0, "offset \(offset): closed")
            XCTAssertEqual(e.misoriented, 0, "offset \(offset): consistently wound")
            XCTAssertEqual(volume(s), 2 * 1 * 0.1, accuracy: 1e-9, "offset \(offset): volume = area × thickness")
            XCTAssertEqual(invertedFaces(s), 0)
        }
        // offset -1 keeps the original as the top skin: everything between y = -t and 0.
        let down = SolidifyModifier(thickness: 0.1, offset: -1).apply(sheet).bounds!
        XCTAssertEqual(down.max.y, 0, accuracy: 1e-12); XCTAssertEqual(down.min.y, -0.1, accuracy: 1e-12)
        let mid = SolidifyModifier(thickness: 0.1, offset: 0).apply(sheet).bounds!
        XCTAssertEqual(mid.max.y, 0.05, accuracy: 1e-12)
    }

    func testSolidifyWithoutRimIsOpenAndEvenThicknessKeepsWallsThick() {
        XCTAssertGreaterThan(openEdges(SolidifyModifier(thickness: 0.1, rim: false).apply(plane(2, 1))).open, 0)
        // A 90° folded sheet: even thickness keeps the inner offset ≈ t at the fold, plain offset thins it by cos(45°).
        var fold = Mesh3()
        for i in 0..<4 { // two panels: floor z∈[0,1] y=0, wall y∈[0,1] z=0 ; welded along x-axis line
            let x0 = Double(i) / 4, x1 = Double(i + 1) / 4
            fold.addQuad(Vec3(x0, 0, 1), Vec3(x1, 0, 1), Vec3(x1, 0, 0), Vec3(x0, 0, 0), outward: Vec3(0, 1, 0))
            fold.addQuad(Vec3(x0, 0, 0), Vec3(x1, 0, 0), Vec3(x1, 1, 0), Vec3(x0, 1, 0), outward: Vec3(0, 0, 1))
        }
        func fold_t(_ even: Bool) -> Double {
            let s = SolidifyModifier(thickness: 0.1, offset: -1, evenThickness: even).apply(fold)
            // inner corner vertex: the lowest-y point of the shell's inner fold, distance to the fold line (y=0,z=0)
            let inner = s.positions.filter { abs($0.x) < 1e-9 && $0.y < -1e-6 || abs($0.x) < 1e-9 && $0.z < -1e-6 }
            return inner.map { len3(Vec3(0, $0.y, $0.z)) }.min() ?? 0
        }
        // Corner of the inner layer is displaced √2·t/… more with even thickness.
        XCTAssertEqual(fold_t(false), 0.1, accuracy: 0.005)
        XCTAssertGreaterThan(fold_t(true), 0.12)
    }

    // MARK: SimpleDeform

    func testBendKeepsEdgeLengths() {
        let sheet = upright(n: 20)
        let bent = SimpleDeformModifier(.bend(angle: .pi / 2), axis: Vec3(0, 1, 0), lower: 0, upper: 1).apply(sheet)
        let a = edgeLengths(sheet), b = edgeLengths(bent)
        XCTAssertEqual(a.count, b.count)
        for (x, y) in zip(a, b) { XCTAssertEqual(y / x, 1, accuracy: 0.02) }
        XCTAssertEqual(invertedFaces(bent), 0)
        // It really bent: top edge is a quarter turn round.
        let top = bent.bounds!
        XCTAssertEqual(top.max.y - top.min.y, 2 / Double.pi, accuracy: 0.02)     // sin(90°)·R, R = 2/π
        // …toward +v, which for a +Y axis is +Z.
        XCTAssertGreaterThan(top.max.z, 0.3)
    }

    func testBendBeyondLimitsContinuesStraightAndZeroAngleIsIdentity() {
        let sheet = upright(n: 6)
        XCTAssertEqual(SimpleDeformModifier(.bend(angle: 0)).apply(sheet).positions, sheet.positions)
        let b = SimpleDeformModifier(.bend(angle: .pi / 2), lower: 0, upper: 0.5).apply(sheet)
        // The part above the limit is rigid: same edge lengths and heights are 0.5·(2/π)… the tip points sideways.
        XCTAssertGreaterThan(b.bounds!.max.z, 0.3)
        for (x, y) in zip(edgeLengths(sheet), edgeLengths(b)) { XCTAssertEqual(y / x, 1, accuracy: 0.03) }
    }

    func testTwistKeepsVolume() {
        let cube = box(n: 16)
        let v0 = volume(cube)
        let twisted = SimpleDeformModifier(.twist(angle: .pi / 2), lower: 0, upper: 1).apply(cube)
        XCTAssertEqual(volume(twisted) / v0, 1, accuracy: 0.03)
        XCTAssertEqual(openEdges(twisted).open, 0)
        XCTAssertEqual(invertedFaces(twisted), 0)
        // top face turned a quarter: the point (0.5, 1, -0.5) lands at (0.5,1,0.5)-ish (rotation about +Y).
        let top = twisted.positions.max { $0.y < $1.y }!
        XCTAssertEqual(top.y, 1, accuracy: 1e-9)
    }

    func testTaperAndStretch() {
        let cube = box(n: 4)
        let t = SimpleDeformModifier(.taper(factor: -0.5), lower: 0, upper: 1).apply(cube).bounds!
        XCTAssertEqual(t.max.y, 1, accuracy: 1e-12)
        let top = SimpleDeformModifier(.taper(factor: -0.5), lower: 0, upper: 1).apply(cube)
            .positions.filter { $0.y > 0.999 }.map { abs($0.x) }.max()!
        XCTAssertEqual(top, 0.25, accuracy: 1e-9)
        let s = SimpleDeformModifier(.stretch(factor: 1), lower: 0, upper: 1).apply(cube)
        XCTAssertEqual(s.bounds!.max.y, 2, accuracy: 1e-9)
        XCTAssertEqual(volume(s) / volume(cube), 1, accuracy: 0.01)    // squeezed to keep it
    }

    // MARK: Lattice

    func testLatticeIdentityAndMovedPoint() {
        let cube = box(n: 4)
        for interp in [LatticeDeformModifier.Interpolation.trilinear, .cubic] {
            var lat = LatticeDeformModifier(boxMin: Vec3(-0.5, 0, -0.5), boxMax: Vec3(0.5, 1, 0.5), divisions: SIMD3(3, 3, 3), interpolation: interp)
            let same = lat.apply(cube)
            for (a, b) in zip(cube.positions, same.positions) { XCTAssertLessThan(dist3(a, b), 1e-9, "\(interp) identity") }
            lat.displace(1, 2, 1, by: Vec3(0, 0.3, 0))          // pull the top-middle point up
            let moved = lat.apply(cube)
            XCTAssertEqual(moved.bounds!.max.y, 1.3, accuracy: 0.05)
            XCTAssertEqual(openEdges(moved).open, 0)
            // corners (not influenced at a distance for trilinear) stay put
            XCTAssertEqual(moved.positions.filter { abs($0.x + 0.5) < 1e-9 && abs($0.z + 0.5) < 1e-9 && $0.y > 0.999 }.first!.y,
                           1, accuracy: interp == .trilinear ? 1e-9 : 0.2)
        }
    }

    // MARK: Displace

    func testDisplaceAlongNormalsKeepsMeshClosedAndMovesIt() {
        let cube = box(n: 6)
        let flat = DisplaceModifier(label: "const", strength: 0.1, midlevel: 0) { _, _ in 1 }.apply(cube)
        XCTAssertEqual(openEdges(flat).open, 0, "welded normals: no tearing at hard edges")
        XCTAssertGreaterThan(volume(flat), volume(cube) * 1.2)
        let noisy = DisplaceModifier.noise(strength: 0.2, frequency: 4, seed: 7).apply(cube)
        XCTAssertEqual(openEdges(noisy).open, 0)
        XCTAssertNotEqual(noisy.positions, cube.positions)
        XCTAssertEqual(noisy.positions, DisplaceModifier.noise(strength: 0.2, frequency: 4, seed: 7).apply(cube).positions, "deterministic")
        XCTAssertNotEqual(noisy.positions, DisplaceModifier.noise(strength: 0.2, frequency: 4, seed: 8).apply(cube).positions)
        let fixed = DisplaceModifier(strength: 1, midlevel: 0, direction: .fixed(Vec3(1, 0, 0))) { _, _ in 0.25 }.apply(cube)
        XCTAssertEqual(fixed.bounds!.min.x, -0.25, accuracy: 1e-12)
    }

    // MARK: Shrinkwrap

    func testShrinkwrapNearestLandsOnTarget() {
        let target = sphere(r: 1)
        let bvh = TriangleBVH(target)
        let src = sphere(r: 1.6, rings: 10, segs: 14)
        let wrapped = ShrinkwrapModifier(target: target).apply(src)
        for p in wrapped.positions { XCTAssertLessThan(bvh.closest(to: p)!.distance, 1e-9) }
        XCTAssertEqual(openEdges(wrapped).open, 0)
        // offset lifts along the target's normal
        let off = ShrinkwrapModifier(target: target, offset: 0.2).apply(src)
        for p in off.positions { XCTAssertEqual(bvh.closest(to: p)!.distance, 0.2, accuracy: 0.03) }
        // factor 0 = unchanged
        XCTAssertEqual(ShrinkwrapModifier(target: target, factor: 0).apply(src).positions, src.positions)
    }

    func testShrinkwrapProjectAlongNormalAndDirection() {
        let target = sphere(r: 1)
        let bvh = TriangleBVH(target)
        let src = sphere(r: 1.6, rings: 10, segs: 14)
        let p = ShrinkwrapModifier(target: target, method: .projectAlongNormal(.negative)).apply(src)
        for q in p.positions { XCTAssertLessThan(bvh.closest(to: q)!.distance, 1e-9) }
        // Radial: a vertex projected inward along its (radial) normal keeps its direction from the centre.
        for (a, b) in zip(src.positions, p.positions) where len3(a) > 0.1 { XCTAssertGreaterThan(dot3(normalize3(a), normalize3(b)), 0.98) }
        // Wrong sign misses a target that lies behind: positive from outside hits nothing -> unchanged.
        let miss = ShrinkwrapModifier(target: target, method: .projectAlongNormal(.positive)).apply(src)
        XCTAssertEqual(miss.positions, src.positions)
        // fixed direction: a flat sheet above the sphere dropped along -Y lands on the sphere's top half
        let sheet = plane(1, 1, nx: 6, nz: 6).translated(by: Vec3(0, 3, 0))
        let dropped = ShrinkwrapModifier(target: target, method: .projectAlong(Vec3(0, -1, 0))).apply(sheet)
        for q in dropped.positions { XCTAssertEqual(len3(q), 1, accuracy: 0.02); XCTAssertGreaterThan(q.y, 0.5) }
    }

    // MARK: Stack

    func testStackOrderMattersAndDisabledIsSkipped() {
        let sheet = plane(2, 1)
        func stack(_ order: [any Modifier]) -> ModifierStack { var s = ModifierStack(); order.forEach { s.add($0) }; return s }
        let solid = SolidifyModifier(thickness: 0.1)
        let arr = ArrayModifier(count: 3, offset: Vec3(0, 1, 0))
        let bend = SimpleDeformModifier(.twist(angle: 1), lower: 0, upper: 1)
        // Array then bend vs bend then array: different result.
        let a = stack([arr, bend]).evaluate(sheet), b = stack([bend, arr]).evaluate(sheet)
        XCTAssertNotEqual(a.positions, b.positions)
        // Solidify before mirror is closed twice over; mirror before solidify too, but via a different path: counts differ.
        let sm = stack([solid, MirrorModifier(.y, at: 0.5)]).evaluate(sheet)
        let ms = stack([MirrorModifier(.y, at: 0.5), solid]).evaluate(sheet)
        XCTAssertNotEqual(sm.positions, ms.positions)

        var s = stack([solid, arr])
        let id = s.entries[0].id
        let full = s.evaluate(sheet)
        s.setEnabled(false, id: id)
        let skipped = s.evaluate(sheet)
        XCTAssertEqual(skipped, arr.apply(sheet))
        XCTAssertNotEqual(skipped, full)
        XCTAssertEqual(s.evaluate(sheet, upTo: 1), sheet, "upTo 1 = only the (disabled) first entry")
        s.setEnabled(true, id: id)
        XCTAssertEqual(s.evaluate(sheet), full, "re-evaluation after re-enable gives the same mesh")
    }

    func testStackDescriptionEqualityAndEditing() {
        var s = ModifierStack()
        let a = s.add(SolidifyModifier(thickness: 0.1), name: "Shell")
        s.add(BevelModifier(radius: 0.01), name: "Round")
        var t = s
        XCTAssertEqual(s, t)
        t.setEnabled(false, id: a)
        XCTAssertNotEqual(s, t)
        XCTAssertTrue(t.description.hasPrefix("- Shell: Solidify"))
        t = s; t.replace(id: a, with: SolidifyModifier(thickness: 0.2))
        XCTAssertNotEqual(s, t)
        t = s; t.move(from: 0, to: 1)
        XCTAssertEqual(t.entries.map(\.name), ["Round", "Shell"])
        t.remove(id: a)
        XCTAssertEqual(t.entries.count, 1)
        // a stack is itself a modifier
        let inner = ModifierStack()
        XCTAssertEqual(inner.apply(plane(1, 1)), plane(1, 1))
    }

    func testWrappersAndSwappableBoolean() {
        let cube = box(n: 2)
        XCTAssertEqual(MirrorModifier(.x, at: 1).apply(cube).triangleCount, cube.triangleCount * 2)
        XCTAssertEqual(ArrayModifier(count: 3, offset: Vec3(2, 0, 0)).apply(cube).triangleCount, cube.triangleCount * 3)
        XCTAssertEqual(ArrayModifier(radialCount: 4).apply(cube).triangleCount, cube.triangleCount * 4)
        XCTAssertGreaterThan(SubdivideModifier(levels: 1).apply(cube).triangleCount, cube.triangleCount)
        XCTAssertGreaterThan(BevelModifier(radius: 0.02).apply(box(n: 1)).triangleCount, 12)
        // The engine closure is what runs: a stand-in robust boolean replaces the BSP.
        let marker = plane(1, 1, nx: 1, nz: 1)
        let swapped = BooleanModifier(.difference, with: cube) { _, _, op in op == .difference ? marker : Mesh3() }
        XCTAssertEqual(swapped.apply(cube), marker)
        // The default engine is the existing BSP.
        let other = box(n: 1).translated(by: Vec3(0.5, 0.25, 0))
        XCTAssertEqual(BooleanModifier(.union, with: other).apply(box(n: 1)), box(n: 1).union(other))
    }
}
