//  Split from Mesh3.swift (DH-0523): transforms & partitioning (translate / rotate / scale / smooth / partition / volume).
//  Pure move — see Mesh3.swift for the type declaration and MeshAudit for the auditor.

import simd

extension Mesh3 {
    /// Split the mesh's triangles into `(match, rest)` by a predicate on each
    /// triangle's stored vertex normal (they agree with the geometric normal by
    /// construction — `addTriangle(outward:)`). Vertex attributes are copied
    /// verbatim, winding untouched. Used to give one solid two materials along a
    /// facing split — e.g. an exterior wall's OUTSIDE faces vs its room faces.
    public func partitioned(by isMatch: (Vec3) -> Bool) -> (match: Mesh3, rest: Mesh3) {
        partitioned { n, _ in isMatch(n) }
    }

    /// `partitioned(by:)` with the triangle's CENTROID as well as its normal — for a claim that
    /// covers part of a surface rather than a whole facing. A wall merged through a junction
    /// (`WallEdge.passThrough`) is one mesh with a different room against each stretch of it, so
    /// "which triangles are this room's" is a question about WHERE as well as which way.
    public func partitioned(by isMatch: (Vec3, Vec3) -> Bool) -> (match: Mesh3, rest: Mesh3) {
        var match = Mesh3(), rest = Mesh3()
        func append(_ ia: Int, _ ib: Int, _ ic: Int, to m: inout Mesh3) {
            let base = UInt32(m.positions.count)
            for vi in [ia, ib, ic] {
                m.positions.append(positions[vi])
                m.normals.append(normals[vi])
                m.uvs.append(uvs[vi])
            }
            m.indices.append(contentsOf: [base, base + 1, base + 2])
        }
        var i = 0
        while i + 2 < indices.count {
            let ia = Int(indices[i]), ib = Int(indices[i + 1]), ic = Int(indices[i + 2])
            let centroid = (positions[ia] + positions[ib] + positions[ic]) / 3
            if isMatch(normals[ia], centroid) { append(ia, ib, ic, to: &match) }
            else { append(ia, ib, ic, to: &rest) }
            i += 3
        }
        return (match, rest)
    }

    /// Signed volume of the mesh by the divergence theorem: `(1/6)·Σ v0·(v1×v2)`
    /// over every triangle, in winding order. For a closed mesh wound **outward**
    /// (the `addTriangle(outward:)` convention) this is the geometric volume and is
    /// strictly **positive**; an inside-out mesh yields a negative value. Because
    /// the sum is linear over triangles it stays positive for an assembly of several
    /// outward-wound closed solids (the union's volume), even with internal overlaps —
    /// which makes it a GPU-free winding gate for the multi-piece procedural meshes
    /// that `MeshAudit`'s (tautological) winding check cannot police. Only meaningful
    /// for meshes that are closed (or unions of closed solids); open shells give a
    /// value with no clean sign guarantee.
    public var signedVolume: Double {
        var v = 0.0
        var i = 0
        while i < indices.count {
            let a = positions[Int(indices[i])]
            let b = positions[Int(indices[i + 1])]
            let c = positions[Int(indices[i + 2])]
            v += dot3(a, cross3(b, c))
            i += 3
        }
        return v / 6.0
    }

    /// Return a copy of this mesh translated by `delta`. Normals are unchanged.
    public func translated(by delta: Vec3) -> Mesh3 {
        Mesh3(positions: positions.map { $0 + delta }, normals: normals, uvs: uvs, indices: indices)
    }

    /// Return a copy rotated about the local **Y axis** by `radians` (positions AND normals),
    /// the plan-heading rotation a placeable applies to a sub-part before the parts are appended
    /// (e.g. a book laid across a stack at a casual angle). Winding is preserved — the rotation is
    /// a rigid isometry, so no face is reversed and `MeshAudit` stays satisfied.
    public func rotatedY(_ radians: Double) -> Mesh3 {
        let c = cos(radians), s = sin(radians)
        func rot(_ v: Vec3) -> Vec3 { Vec3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z) }
        return Mesh3(positions: positions.map(rot), normals: normals.map(rot), uvs: uvs, indices: indices)
    }

    /// Return a copy rotated about the local **X axis** by `radians` (positions AND normals) — the
    /// tilt a sub-part takes before it is appended (a leaf lying over a fruit, a lid ajar). A rigid
    /// isometry: winding is preserved and `MeshAudit` stays satisfied, like `rotatedY`.
    public func rotatedX(_ radians: Double) -> Mesh3 {
        let c = cos(radians), s = sin(radians)
        func rot(_ v: Vec3) -> Vec3 { Vec3(v.x, c * v.y - s * v.z, s * v.y + c * v.z) }
        return Mesh3(positions: positions.map(rot), normals: normals.map(rot), uvs: uvs, indices: indices)
    }

    /// Return a copy rotated about the local **Z axis** by `radians` (positions AND normals) — the
    /// sibling of `rotatedX` for a roll about the depth axis. Rigid; winding preserved.
    public func rotatedZ(_ radians: Double) -> Mesh3 {
        let c = cos(radians), s = sin(radians)
        func rot(_ v: Vec3) -> Vec3 { Vec3(c * v.x - s * v.y, s * v.x + c * v.y, v.z) }
        return Mesh3(positions: positions.map(rot), normals: normals.map(rot), uvs: uvs, indices: indices)
    }

    /// Return a copy whose **plan is warped from a circle into an OVOID** — full width at the
    /// rear (+Z), tapering to a narrower nose at the front (−Z) — about the local Y axis.
    ///
    /// For a body of revolution this is the difference between a barrel and an egg, and it is the
    /// shape most real sanitaryware and seating actually has. A `scaled(x:y:z:)` squash cannot do
    /// it: an ellipse is symmetric front-to-back, so it makes a barrel wider than it is deep,
    /// never a pear (Danny, 2026-08-23, on the toilet: *"toilet bowls aren't really perfect
    /// circles, are they?"* — and the pan was in fact a perfect circle, because its squash factor
    /// cancelled to exactly 1.0).
    ///
    /// The warp acts on the ANGLE around the axis, not on absolute Z, so the plan stays
    /// self-similar at every height and the vertical profile is untouched. `noseWidthFraction` is
    /// the nose's width as a fraction of the rear's (0.6 ⇒ the front is 60 % as wide). The result
    /// is renormalised so the **maximum half-width is exactly `halfWidth`** — otherwise tapering
    /// would silently shrink the piece below its spec — and the widest station sits behind the
    /// midpoint, which is where an egg's is.
    ///
    /// Positions only: the deformation is non-linear, so its correct normal transform is a
    /// per-vertex inverse-transpose Jacobian. Rather than approximate that, this returns a mesh
    /// whose stored normals are **re-derived flat from the warped winding**, which is the exact
    /// answer for a flat-shaded mesh and the correct input for `smoothed(creaseDegrees:)`
    /// (which recomputes face normals from positions anyway). Winding is preserved: both scale
    /// factors are strictly positive, so orientation cannot flip.
    public func ovoidPlan(halfWidth: Double, halfDepth: Double,
                          noseWidthFraction: Double) -> Mesh3 {
        let k = 1 - max(0.05, min(1, noseWidthFraction))     // how much width the nose gives up
        // Width profile as a function of c = cos(angle from the rear): 1 at the rear, (1 − k) at
        // the nose, smooth in between — no slope break anywhere, or the flanks would carry a
        // visible crease down the piece.
        func widthAt(_ c: Double) -> Double { 1 - k * (1 - c) / 2 }
        // Normalise so the widest point is exactly `halfWidth`. Sampled rather than solved: the
        // extremum of |sin θ| · widthAt(cos θ) moves with k, and 720 samples fix it to well under
        // a micron at furniture scale.
        var peak = 1e-9
        for i in 0 ... 720 {
            let a = Double.pi * Double(i) / 720
            peak = Swift.max(peak, abs(sin(a)) * widthAt(cos(a)))
        }
        // Normalised by the SOURCE's own plan radius, so the contract is "make this body of
        // revolution an ovoid of exactly these half-extents" rather than "multiply by these" —
        // one fewer way for a caller to end up at an accidental 1.0, which is precisely how the
        // pan became a circle.
        var maxRho = 1e-12
        for q in positions { maxRho = Swift.max(maxRho, (q.x * q.x + q.z * q.z).squareRoot()) }
        let xGain = halfWidth / (peak * maxRho)
        let zGain = halfDepth / maxRho

        var out = Mesh3()
        func warp(_ p: Vec3) -> Vec3 {
            let rho = (p.x * p.x + p.z * p.z).squareRoot()
            guard rho > 1e-12 else { return Vec3(p.x, p.y, p.z * zGain) }
            let c = p.z / rho                                 // +1 rear, −1 nose
            return Vec3(p.x * widthAt(c) * xGain, p.y, p.z * zGain)
        }
        var i = 0
        while i + 2 < indices.count {
            let a = warp(positions[Int(indices[i])])
            let b = warp(positions[Int(indices[i + 1])])
            let c = warp(positions[Int(indices[i + 2])])
            i += 3
            let n = cross3(b - a, c - a)
            guard len3(n) > 1e-12 else { continue }
            _ = out.addTriangle(a, b, c, outward: normalize3(n))
        }
        return out
    }

    /// Return a copy scaled (about the origin) by per-axis factors. Positions scale
    /// directly; normals transform by the **inverse-transpose** (`1/sx, 1/sy, 1/sz`) then
    /// renormalise, so they stay perpendicular to the deformed surface — keeping the
    /// `addTriangle(outward:)` winding/normal agreement that `MeshAudit` checks intact
    /// under a non-uniform squash (e.g. circular revolve → elongated oval). Factors must
    /// be positive (a negative factor mirrors and would invert winding).
    public func scaled(x sx: Double, y sy: Double, z sz: Double) -> Mesh3 {
        let s = Vec3(sx, sy, sz)
        let inv = Vec3(1.0 / sx, 1.0 / sy, 1.0 / sz)
        return Mesh3(positions: positions.map { $0 * s },
                     normals: normals.map { normalize3($0 * inv) },
                     uvs: uvs, indices: indices)
    }

    /// Return a copy with **smooth (averaged) vertex normals** — the fix for a curved surface
    /// that shades as a set of flat panels.
    ///
    /// Every builder here declares winding through `addTriangle(outward:)`, which writes ONE
    /// face normal to all three of a triangle's vertices. That is correct for a box and wrong
    /// for anything curved: the renderer is handed a faceted normal field and dutifully shades
    /// facets, so a rolled arm or a domed cushion reads as polygons however finely it is
    /// tessellated (Danny, 2026-08-20: *"I can see all the different faces making up the couch
    /// arm"*). Extra segments only make the panels smaller; they never make the surface curve.
    ///
    /// This averages the face normals meeting at each position, so the shading normal varies
    /// continuously across a curved band. Faces meeting at more than `creaseDegrees` are treated
    /// as a genuine ARRIS and excluded from each other's average — so a cushion's seam, a cap,
    /// and the join between a roll and a flat stay crisp instead of smearing into a soft blur.
    ///
    /// **Apply it per PART, before parts are appended together.** Two overlapping solids (the
    /// sofa's arm and its skirt) can share a coincident vertex position, and averaging across
    /// that would blend two unrelated surfaces' normals.
    ///
    /// Winding is untouched: only the stored normals change, so the engine's single-sided cull
    /// and the GPU winding census are unaffected. `MeshAudit.isSound` compares the stored normal
    /// against the face normal within ~8°, which a well-tessellated smooth surface stays inside;
    /// where a surface turns harder than that per facet, the answer is more segments there, not
    /// a looser audit.
    public func smoothed(creaseDegrees: Double = 40) -> Mesh3 {
        guard !indices.isEmpty else { return self }
        let cosCrease = cos(creaseDegrees * .pi / 180)
        let triCount = indices.count / 3

        /// Interior angle of triangle `(a,b,c)` at corner `a` — the weight each incident face
        /// gets in a vertex's average.
        ///
        /// Weighting by AREA (the cross product's own length) is the obvious choice and it is
        /// wrong here. Around an arm's section outline a long straight run meets a bull-nose
        /// whose facets are a quarter the length; area weighting lets the big facets dominate
        /// the shared vertex and drags the small facets' normals right off their own surface —
        /// far enough that `MeshAudit` counted them as winding mismatches. The angle a face
        /// subtends at the vertex is the geometrically meaningful share and is independent of
        /// how finely its neighbours happen to be cut.
        func cornerAngle(_ a: Vec3, _ b: Vec3, _ c: Vec3) -> Double {
            let u = b - a, v = c - a
            let lu = len3(u), lv = len3(v)
            guard lu > 1e-12, lv > 1e-12 else { return 0 }
            return acos(max(-1, min(1, dot3(u, v) / (lu * lv))))
        }

        // Per position: every incident face as (unit normal, angular weight).
        var byPos: [SIMD3<Int64>: [(n: Vec3, w: Double)]] = [:]
        var faceNormal = [Vec3](repeating: Vec3(0, 0, 0), count: triCount)
        for t in 0 ..< triCount {
            let ia = Int(indices[t * 3]), ib = Int(indices[t * 3 + 1]), ic = Int(indices[t * 3 + 2])
            guard ia < positions.count, ib < positions.count, ic < positions.count else { continue }
            let a = positions[ia], b = positions[ib], c = positions[ic]
            let raw = cross3(b - a, c - a)
            guard len3(raw) > 1e-12 else { continue }
            let fn = normalize3(raw)
            faceNormal[t] = fn
            for (i, ang) in [(ia, cornerAngle(a, b, c)), (ib, cornerAngle(b, c, a)), (ic, cornerAngle(c, a, b))]
            where ang > 1e-9 {
                byPos[Mesh3.qpos(positions[i]), default: []].append((fn, ang))
            }
        }

        var out = normals
        for t in 0 ..< triCount {
            let fn = faceNormal[t]
            guard len3(fn) > 1e-12 else { continue }
            for k in 0 ..< 3 {
                let i = Int(indices[t * 3 + k])
                guard i < positions.count, i < out.count else { continue }
                var acc = Vec3(0, 0, 0)
                for c in byPos[Mesh3.qpos(positions[i])] ?? [] where dot3(fn, c.n) >= cosCrease {
                    acc = acc + c.n * c.w
                }
                out[i] = len3(acc) > 1e-12 ? normalize3(acc) : fn
            }
        }
        return Mesh3(positions: positions, normals: out, uvs: uvs, indices: indices)
    }

    /// Position quantised to 0.1 mm — the same grid `MeshAudit`'s edge keys use, so "the same
    /// point" means the same thing to the auditor and to the smoother.
    public static func qpos(_ v: Vec3) -> SIMD3<Int64> {
        SIMD3<Int64>(Int64((v.x * 1e4).rounded()),
                     Int64((v.y * 1e4).rounded()),
                     Int64((v.z * 1e4).rounded()))
    }

    /// Return a copy rigidly re-based: a local point `(x,y,z)` maps to
    /// `origin + x·xAxis + y·yAxis + z·zAxis`. The three axes must be **orthonormal**
    /// (a pure rotation, det = +1) — then normals transform by the very same map (no
    /// inverse-transpose needed) and the `addTriangle(outward:)` winding/normal agreement
    /// that `MeshAudit` checks stays intact. This is the rigid analogue of `scaled` /
    /// `translated`: it places a mesh authored in a convenient local frame (e.g. a slab
    /// built axis-aligned about the origin) into an arbitrary orientation + position
    /// without hand-winding anything. A left-handed/mirrored basis would flip winding and
    /// is not supported.
    public func placed(xAxis: Vec3, yAxis: Vec3, zAxis: Vec3, origin: Vec3) -> Mesh3 {
        func map(_ p: Vec3) -> Vec3 { xAxis * p.x + yAxis * p.y + zAxis * p.z }
        return Mesh3(positions: positions.map { origin + map($0) },
                     normals: normals.map { normalize3(map($0)) },
                     uvs: uvs, indices: indices)
    }
}
