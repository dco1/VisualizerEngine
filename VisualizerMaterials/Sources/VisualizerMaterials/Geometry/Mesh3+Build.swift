//  Split from Mesh3.swift (DH-0523): construction primitives (addTriangle / addQuad / append).
//  Pure move — see Mesh3.swift for the type declaration and MeshAudit for the auditor.

import simd

extension Mesh3 {
    /// Append a triangle, **ordering the winding so its geometric normal aligns with
    /// `outward`** (right-hand rule). The aligned normal is stored on all three
    /// vertices. Degenerate (collinear) triangles are dropped. This is the single
    /// choke point that makes winding/normal agreement true *by construction* — the
    /// builders declare which way is "out" and never hand-wind a triangle.
    @discardableResult
    public mutating func addTriangle(_ a: Vec3, _ b: Vec3, _ c: Vec3,
                                     outward: Vec3,
                                     uv: (Vec2, Vec2, Vec2) = (.zero, .zero, .zero)) -> Bool {
        var p = (a, b, c)
        var t = (uv.0, uv.1, uv.2)
        let raw = cross3(p.1 - p.0, p.2 - p.0)
        guard len3(raw) > 1e-12 else { return false }           // degenerate → skip
        if dot3(raw, outward) < 0 {                              // wind to face `outward`
            swap(&p.1, &p.2); swap(&t.1, &t.2)
        }
        let n = normalize3(cross3(p.1 - p.0, p.2 - p.0))
        let base = UInt32(positions.count)
        positions.append(p.0); positions.append(p.1); positions.append(p.2)
        normals.append(n); normals.append(n); normals.append(n)
        uvs.append(t.0); uvs.append(t.1); uvs.append(t.2)
        indices.append(base); indices.append(base + 1); indices.append(base + 2)
        return true
    }

    /// Append a planar quad `a→b→c→d` (in order around the face) as two triangles,
    /// each wound to face `outward`.
    public mutating func addQuad(_ a: Vec3, _ b: Vec3, _ c: Vec3, _ d: Vec3,
                                 outward: Vec3,
                                 uv: (Vec2, Vec2, Vec2, Vec2) = (.zero, .zero, .zero, .zero)) {
        addTriangle(a, b, c, outward: outward, uv: (uv.0, uv.1, uv.2))
        addTriangle(a, c, d, outward: outward, uv: (uv.0, uv.2, uv.3))
    }

    /// Append a triangle with **explicit per-vertex normals** (smooth / Gouraud shading).
    /// The *winding* is still declared by `outward` (so the triangle faces the right way and
    /// the engine's single-sided cull stays correct — verified by the GPU winding census),
    /// but the stored vertex normals are the caller-supplied `normals` instead of the flat
    /// face normal. This is for **smooth heightfields** (terrain), where area/gradient-averaged
    /// vertex normals are the physically-correct shading normals and *intentionally* diverge
    /// from each triangle's face normal — so `MeshAudit.windingMismatches` fires by design.
    /// Use `MeshAudit.isSoundHeightfield` (winding proven up-facing) rather than `isSound`.
    @discardableResult
    public mutating func addTriangleSmooth(_ a: Vec3, _ b: Vec3, _ c: Vec3,
                                           outward: Vec3,
                                           normals nm: (Vec3, Vec3, Vec3),
                                           uv: (Vec2, Vec2, Vec2) = (.zero, .zero, .zero)) -> Bool {
        var p = (a, b, c)
        var t = (uv.0, uv.1, uv.2)
        var n = (nm.0, nm.1, nm.2)
        let raw = cross3(p.1 - p.0, p.2 - p.0)
        guard len3(raw) > 1e-12 else { return false }           // degenerate → skip
        if dot3(raw, outward) < 0 {                              // wind to face `outward`
            swap(&p.1, &p.2); swap(&t.1, &t.2); swap(&n.1, &n.2)
        }
        let base = UInt32(positions.count)
        positions.append(p.0); positions.append(p.1); positions.append(p.2)
        normals.append(normalize3(n.0)); normals.append(normalize3(n.1)); normals.append(normalize3(n.2))
        uvs.append(t.0); uvs.append(t.1); uvs.append(t.2)
        indices.append(base); indices.append(base + 1); indices.append(base + 2)
        return true
    }

    /// Append a quad `a→b→c→d` with explicit per-vertex smooth normals. The winding is
    /// declared by `outward`; normals are stored verbatim. See `addTriangleSmooth`.
    public mutating func addQuadSmooth(_ a: Vec3, _ b: Vec3, _ c: Vec3, _ d: Vec3,
                                       outward: Vec3,
                                       normals nm: (Vec3, Vec3, Vec3, Vec3),
                                       uv: (Vec2, Vec2, Vec2, Vec2) = (.zero, .zero, .zero, .zero)) {
        addTriangleSmooth(a, b, c, outward: outward, normals: (nm.0, nm.1, nm.2), uv: (uv.0, uv.1, uv.2))
        addTriangleSmooth(a, c, d, outward: outward, normals: (nm.0, nm.2, nm.3), uv: (uv.0, uv.2, uv.3))
    }

    /// Merge another mesh in, offsetting its indices.
    public mutating func append(_ other: Mesh3) {
        let base = UInt32(positions.count)
        positions.append(contentsOf: other.positions)
        normals.append(contentsOf: other.normals)
        uvs.append(contentsOf: other.uvs)
        indices.append(contentsOf: other.indices.map { $0 + base })
    }
}
