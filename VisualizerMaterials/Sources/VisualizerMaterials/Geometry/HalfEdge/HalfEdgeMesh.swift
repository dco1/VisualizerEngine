//  HalfEdgeMesh — an editable polygon mesh (n-gon faces) with full adjacency.
//
//  Guarantees: every alive face is a closed loop of >= 3 halfedges with distinct vertices; a directed
//  edge (a -> b) exists at most once, so the mesh is consistently oriented where faces are shared; twins
//  and edges are kept in step by every mutation; `validate()` proves it. Non-manifold input is not
//  represented: faces whose directed edge is already taken are rejected (`Mesh3` import tries the
//  reversed winding first). Bow-tie vertices (two fans on one vertex) can exist and are reported by
//  `nonManifoldVertices()`, but vertex traversal only visits one fan.
//
//  Representation: halfedges are NOT paired up; a boundary halfedge simply has `twin == -1`. An edge is a
//  first-class id (`heEdge`) shared by a halfedge and its twin. Corner data (uv, normal) lives on the
//  halfedge (the corner at its origin in its face), so seams and hard edges are just corners that differ
//  across an edge (`isSeam`, `isHard`). Deleted things are tombstones (face `faceHE == -1`, halfedge
//  `heFace == -1`, edge `edgeHE == -1`); ids stay stable across mutations until `compacted()`.
//
//  Mutation primitives (extend the mesh in `HalfEdgeMesh+<Tool>.swift`): `addVertex`, `addFace`,
//  `deleteFace(s)`, `splitEdge`, `splitFace`, `collapseEdge`, `dissolveEdge`, `flipEdge`, `flip(faces:)`.
//  Not guaranteed: geometric validity (self-intersection, planarity). CPU only: topology edits are
//  pointer-chasing on small neighbourhoods, nothing per-vertex-parallel to put on the GPU.

import simd

public struct HalfEdgeMesh: Sendable {
    /// Vertex positions. Free to edit; topology is changed only through the mutation primitives.
    public var positions: [Vec3] = []

    // topology (internal: tests and extensions in this module may poke it)
    var vertexHE: [Int] = []          // an outgoing halfedge, -1 = isolated
    var heOrigin: [Int] = []
    var heNext: [Int] = []
    var hePrev: [Int] = []
    var heTwin: [Int] = []            // -1 on a boundary
    var heFace: [Int] = []            // -1 = deleted
    var heEdge: [Int] = []
    var heUV: [Vec2] = []             // corner data: at the halfedge's origin, in its face
    var heNormal: [Vec3] = []
    var faceHE: [Int] = []            // -1 = deleted
    var edgeHE: [Int] = []            // -1 = deleted
    var directed: [UInt64: Int] = [:] // (origin << 32 | dest) -> halfedge

    public init() {}

    @inline(__always) static func key(_ a: Int, _ b: Int) -> UInt64 { UInt64(a) << 32 | UInt64(b) }

    // MARK: counts and ids

    public var vertexSlots: Int { positions.count }
    public var faceSlots: Int { faceHE.count }
    public var edgeSlots: Int { edgeHE.count }
    public var halfedgeSlots: Int { heOrigin.count }
    public var faceCount: Int { faceHE.reduce(0) { $0 + ($1 >= 0 ? 1 : 0) } }
    public var edgeCount: Int { edgeHE.reduce(0) { $0 + ($1 >= 0 ? 1 : 0) } }
    /// Vertices used by at least one face.
    public var vertexCount: Int { vertexHE.reduce(0) { $0 + ($1 >= 0 ? 1 : 0) } }
    public var faces: [Int] { faceHE.indices.filter { faceHE[$0] >= 0 } }
    public var edges: [Int] { edgeHE.indices.filter { edgeHE[$0] >= 0 } }
    public var vertices: [Int] { vertexHE.indices.filter { vertexHE[$0] >= 0 } }
    public func isAlive(face f: Int) -> Bool { f >= 0 && f < faceHE.count && faceHE[f] >= 0 }
    public func isAlive(edge e: Int) -> Bool { e >= 0 && e < edgeHE.count && edgeHE[e] >= 0 }
    public func isIsolated(vertex v: Int) -> Bool { vertexHE[v] < 0 }

    // MARK: halfedge accessors

    public func origin(of h: Int) -> Int { heOrigin[h] }
    public func destination(of h: Int) -> Int { heOrigin[heNext[h]] }
    public func next(of h: Int) -> Int { heNext[h] }
    public func prev(of h: Int) -> Int { hePrev[h] }
    /// The opposite halfedge, or -1 on a boundary.
    public func twin(of h: Int) -> Int { heTwin[h] }
    public func face(of h: Int) -> Int { heFace[h] }
    public func edge(of h: Int) -> Int { heEdge[h] }
    public func uv(of h: Int) -> Vec2 { heUV[h] }
    public func normal(of h: Int) -> Vec3 { heNormal[h] }
    public mutating func setUV(_ uv: Vec2, of h: Int) { heUV[h] = uv }
    public mutating func setNormal(_ n: Vec3, of h: Int) { heNormal[h] = n }
    /// One halfedge of a face (its loop is `halfedges(of face:)`).
    public func halfedge(of face: Int) -> Int { faceHE[face] }
    /// One halfedge of an edge (the second, if any, is its twin).
    public func halfedge(ofEdge e: Int) -> Int { edgeHE[e] }

    // MARK: building

    /// New isolated vertex.
    @discardableResult
    public mutating func addVertex(_ p: Vec3) -> Int {
        positions.append(p); vertexHE.append(-1)
        return positions.count - 1
    }

    /// Add a polygon over existing vertices (wound as given; outward = counter-clockwise). Returns the face id,
    /// or nil when it would break the mesh: < 3 corners, repeated or unknown vertex, or a directed edge already
    /// in use (flip the winding). `uvs` / `normals` are per corner; normals default to the flat face normal.
    @discardableResult
    public mutating func addFace(_ verts: [Int], uvs: [Vec2]? = nil, normals: [Vec3]? = nil) -> Int? {
        addFace(verts, uvs: uvs, normals: normals, slot: nil)
    }

    @discardableResult
    mutating func addFace(_ verts: [Int], uvs: [Vec2]?, normals: [Vec3]?, slot: Int?) -> Int? {
        let n = verts.count
        guard n >= 3, uvs == nil || uvs!.count == n, normals == nil || normals!.count == n else { return nil }
        guard verts.allSatisfy({ $0 >= 0 && $0 < positions.count }), Set(verts).count == n else { return nil }
        for i in 0..<n where directed[Self.key(verts[i], verts[(i + 1) % n])] != nil { return nil }
        if let s = slot, s < faceHE.count, faceHE[s] >= 0 { return nil }

        let flat = normals ?? {
            let nn = normalize3(Self.newell(verts.map { positions[$0] }))
            return [Vec3](repeating: nn, count: n)
        }()
        let base = heOrigin.count
        let f: Int
        if let s = slot, s < faceHE.count { f = s; faceHE[s] = base } else { f = faceHE.count; faceHE.append(base) }
        for i in 0..<n {
            let h = base + i
            heOrigin.append(verts[i]); heNext.append(base + (i + 1) % n); hePrev.append(base + (i + n - 1) % n)
            heTwin.append(-1); heFace.append(f); heEdge.append(-1)
            heUV.append(uvs?[i] ?? .zero); heNormal.append(flat[i])
            if vertexHE[verts[i]] < 0 || heFace[vertexHE[verts[i]]] < 0 { vertexHE[verts[i]] = h }
        }
        for i in 0..<n {
            let h = base + i
            directed[Self.key(verts[i], verts[(i + 1) % n])] = h
            if let t = directed[Self.key(verts[(i + 1) % n], verts[i])] {
                heTwin[h] = t; heTwin[t] = h; heEdge[h] = heEdge[t]
            } else {
                heEdge[h] = edgeHE.count; edgeHE.append(h)
            }
        }
        return f
    }

    /// Delete a face; its edges lose a side (an edge with no side left disappears). Vertices that lose all
    /// faces become isolated (they stay in `positions`; `compacted()` drops them).
    public mutating func deleteFace(_ f: Int) { removeFaces([f], scan: true) }

    public mutating func deleteFaces(_ fs: some Sequence<Int>) { removeFaces(Array(fs), scan: true) }

    /// `scan`: when a vertex's remembered halfedge dies and no neighbour supplies another, search every
    /// halfedge (O(H)); needed only for bow-tie vertices, so internal rebuilds pass false and re-add faces.
    mutating func removeFaces(_ fs: [Int], scan: Bool) {
        var dying: [Int] = []
        for f in Set(fs) where isAlive(face: f) { dying += halfedges(of: f) }
        guard !dying.isEmpty else { return }
        var fix: [(v: Int, c1: Int, c2: Int)] = []
        for h in dying {
            let t = heTwin[h]
            fix.append((heOrigin[h], heTwin[hePrev[h]], t >= 0 ? heNext[t] : -1))
        }
        for h in dying {
            directed[Self.key(heOrigin[h], heOrigin[heNext[h]])] = nil
            let t = heTwin[h], e = heEdge[h]
            if t >= 0 { heTwin[t] = -1; if edgeHE[e] == h { edgeHE[e] = t } } else { edgeHE[e] = -1 }
            faceHE[heFace[h]] = -1
        }
        for h in dying { heFace[h] = -1; heNext[h] = -1; hePrev[h] = -1; heTwin[h] = -1 }
        for x in fix where vertexHE[x.v] >= 0 && heFace[vertexHE[x.v]] < 0 {
            if x.c1 >= 0, heFace[x.c1] >= 0 { vertexHE[x.v] = x.c1 }
            else if x.c2 >= 0, heFace[x.c2] >= 0 { vertexHE[x.v] = x.c2 }
            else if scan, let h = (0..<heOrigin.count).first(where: { heFace[$0] >= 0 && heOrigin[$0] == x.v }) {
                vertexHE[x.v] = h
            } else { vertexHE[x.v] = -1 }
        }
    }

    static func newell(_ p: [Vec3]) -> Vec3 {
        var n = Vec3.zero
        for i in 0..<p.count {
            let a = p[i], b = p[(i + 1) % p.count]
            n.x += (a.y - b.y) * (a.z + b.z); n.y += (a.z - b.z) * (a.x + b.x); n.z += (a.x - b.x) * (a.y + b.y)
        }
        return n
    }

    // MARK: compaction

    /// Old id -> new id (-1 = gone) for each element kind, from `compacted()`.
    public struct Remap: Sendable { public var vertex: [Int]; public var face: [Int] }

    /// A tight copy: dead faces, edges and halfedges removed and ids renumbered in order (isolated vertices
    /// dropped unless `keepIsolated`). Corner data is kept. Edge ids are not mapped (they are renumbered).
    public func compacted(keepIsolated: Bool = false) -> (mesh: HalfEdgeMesh, remap: Remap) {
        var out = HalfEdgeMesh()
        var vmap = [Int](repeating: -1, count: positions.count)
        for v in 0..<positions.count where keepIsolated || vertexHE[v] >= 0 { vmap[v] = out.addVertex(positions[v]) }
        var fmap = [Int](repeating: -1, count: faceHE.count)
        for f in faces {
            let hs = halfedges(of: f)
            fmap[f] = out.addFace(hs.map { vmap[heOrigin[$0]] }, uvs: hs.map { heUV[$0] },
                                  normals: hs.map { heNormal[$0] }) ?? -1
        }
        return (out, Remap(vertex: vmap, face: fmap))
    }
}
