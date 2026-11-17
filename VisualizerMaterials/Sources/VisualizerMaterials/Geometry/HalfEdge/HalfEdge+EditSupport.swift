//  HalfEdge+EditSupport — what the topology edit tools (`HalfEdge+Extrude`, `+Inset`, `+LoopCut`, ...) share.
//
//  Every tool is a `mutating` extension on HalfEdgeMesh that takes an `HESelection` and returns the new one
//  (`nil` = refused: the mesh is left exactly as it was, so a tool is all-or-nothing). Tools that change
//  faces do it through `rebuild(removing:adding:)`: remove the old faces, add the new polygons (a polygon may
//  name a `slot`, so a face keeps its id), and let `addFace` prove every directed edge is used once. A tool
//  that returned a selection leaves `validate()` empty (the tests sweep that).
//  Guarantee: topology edits are CPU; the bulk per-vertex maths (push/pull offsets, Laplacian smoothing) has a
//  Metal path above `HEEdit.gpuThreshold` selected vertices (`HalfEdge+EditGPU`) with these CPU twins.
//  Not guaranteed: geometric validity (a deep extrude can self-intersect), planarity of n-gons.

import simd

/// Tunables shared by the edit tools.
public enum HEEdit {
    /// Selections with more vertices than this run their per-vertex arithmetic on the GPU (when there is one).
    nonisolated(unsafe) public static var gpuThreshold = 5000
    /// A mitred offset (per-face normals) never grows past this many times the asked distance.
    public static let miterLimit = 3.0
}

/// How a vertex shared by several selected faces picks its direction.
public enum HENormalMode: Sendable {
    /// The area-weighted average of the faces' normals; the vertex moves `distance` along it.
    case averaged
    /// Each face keeps moving `distance` along its own normal: the vertex takes the mitre that does that for
    /// all its faces at once (exact for two, least-squares for more; capped by `HEEdit.miterLimit`).
    case perFace
}

struct EditPoly {
    var v: [Int]
    var uv: [Vec2]
    var n: [Vec3]? = nil
    var slot: Int? = nil
    init(v: [Int], uv: [Vec2]? = nil, n: [Vec3]? = nil, slot: Int? = nil) {
        self.v = v; self.uv = uv ?? [Vec2](repeating: .zero, count: v.count); self.n = n; self.slot = slot
    }
}

extension HalfEdgeMesh {

    /// The face as a polygon (corner data included); `keepSlot` makes it re-add under the same face id.
    func editPoly(of f: Int, keepSlot: Bool = true) -> EditPoly {
        let hs = halfedges(of: f)
        return EditPoly(v: hs.map { heOrigin[$0] }, uv: hs.map { heUV[$0] }, n: hs.map { heNormal[$0] }, slot: keepSlot ? f : nil)
    }

    /// Remove `removing`, add `adding`. Returns the new face ids in order, or nil with the mesh as it was
    /// (vertices added before the call stay: callers that add vertices keep their own backup).
    mutating func rebuild(removing: [Int], adding: [EditPoly]) -> [Int]? {
        let backup = self
        removeFaces(removing, scan: false)
        var ids: [Int] = []
        for p in adding {
            guard let id = addFace(p.v, uvs: p.uv, normals: p.n, slot: p.slot) else { self = backup; return nil }
            ids.append(id)
        }
        repairVertexLinks()
        return ids
    }

    /// Point every vertex at a live outgoing halfedge (or -1 when it has none).
    mutating func repairVertexLinks() {
        for h in 0..<heOrigin.count where heFace[h] >= 0 {
            let v = heOrigin[h]
            if vertexHE[v] < 0 || heFace[vertexHE[v]] < 0 { vertexHE[v] = h }
        }
        for v in 0..<positions.count where vertexHE[v] >= 0 && heFace[vertexHE[v]] < 0 { vertexHE[v] = -1 }
    }

    /// Newell vector (2 × area × unit normal) of a face.
    func areaVector(_ f: Int) -> Vec3 { HalfEdgeMesh.newell(corners(of: f)) }

    /// Live faces of a selection, whatever its mode (the selection is flushed, so `faces` is always filled in).
    func selectedFaces(_ sel: HESelection) -> [Int] { sel.faces.filter { isAlive(face: $0) }.sorted() }

    // MARK: flat shading follows the geometry

    /// Faces around `vs` whose corner normals are all the face normal (flat shaded), for `reshade` afterwards.
    func flatFaces(around vs: some Sequence<Int>) -> [Int] {
        var seen = Set<Int>(), out: [Int] = []
        for v in vs { for f in faces(around: v) where seen.insert(f).inserted {
            let n = faceNormal(f)
            if halfedges(of: f).allSatisfy({ len3(heNormal[$0] - n) < 1e-4 }) { out.append(f) }
        } }
        return out
    }
    /// Set the corner normals of `fs` to their (new) face normals.
    mutating func reshade(_ fs: [Int]) {
        for f in fs where isAlive(face: f) {
            let n = faceNormal(f)
            if len3(n) > 0 { for h in halfedges(of: f) { heNormal[h] = n } }
        }
    }

    // MARK: offsets along normals (CPU twin of the `he_offsets` kernel)

    /// One vertex's displacement from the (area-weighted) normals of the faces it is displaced by.
    static func normalOffset(_ areaVectors: [Vec3], distance d: Double, mode: HENormalMode) -> Vec3 {
        switch mode {
        case .averaged:
            let s = areaVectors.reduce(Vec3.zero, +), l = len3(s)
            return l > 1e-20 ? s / l * d : .zero
        case .perFace:
            var m = Vec3.zero
            for a in areaVectors { let l = len3(a); if l > 1e-20 { m += a / l } }
            m /= Double(Swift.max(areaVectors.count, 1))
            let mm = dot3(m, m)
            guard mm > 1e-8 else { return .zero }
            var r = m * (d / mm)
            let cap = HEEdit.miterLimit * abs(d)
            if len3(r) > cap { r *= cap / len3(r) }
            return r
        }
    }

    /// Displacement per vertex for `entries` (vertex -> area vectors of the faces moving it), CPU or GPU.
    func normalOffsets(_ entries: [(vertex: Int, vectors: [Vec3])], distance: Double, mode: HENormalMode, useGPU: Bool?) -> [Vec3] {
        let gpu = useGPU ?? (entries.count > HEEdit.gpuThreshold)
        if gpu, let r = HalfEdgeEditGPU.shared?.offsets(entries.map { $0.vectors }, distance: distance, perFace: mode == .perFace) { return r }
        return entries.map { HalfEdgeMesh.normalOffset($0.vectors, distance: distance, mode: mode) }
    }

    /// vertex -> area vectors of those of `fs` around it.
    func areaVectorsByVertex(_ fs: some Sequence<Int>) -> [(vertex: Int, vectors: [Vec3])] {
        var acc: [Int: [Vec3]] = [:]
        for f in fs { let av = areaVector(f); for v in vertices(of: f) { acc[v, default: []].append(av) } }
        return acc.keys.sorted().map { ($0, acc[$0]!) }
    }

    // MARK: shared filling helpers

    /// Basis for a polygon with Newell normal `n`.
    static func planeBasis(_ n: Vec3) -> (Vec3, Vec3) {
        let nn = normalize3(n)
        let u = normalize3(abs(nn.x) < 0.9 ? cross3(nn, Vec3(1, 0, 0)) : cross3(nn, Vec3(0, 1, 0)))
        return (u, cross3(nn, u))
    }

    /// Index triples that triangulate polygon `p` (wound counter-clockwise about its Newell normal).
    static func triangulatePolygon(_ p: [Vec3]) -> [(Int, Int, Int)] {
        if p.count == 3 { return [(0, 1, 2)] }
        let (u, w) = planeBasis(newell(p))
        let tris = Triangulation.earClip(p.map { Vec2(dot3($0, u), dot3($0, w)) })
        return tris.count == p.count - 2 ? tris : (1..<(p.count - 1)).map { (0, $0, $0 + 1) }
    }
}
