import Foundation
import simd

// UVTopology — a lightweight welded triangle adjacency for the UV tools (item F of docs/ModellingKit.md).
//
// Guarantees: vertices are welded by position (so a triangle-soup mesh and an indexed one give the same
// topology); edges are keyed by welded vertex pair; seams are sets of those edges; islands are the
// triangle groups left by cutting at the seams, with *corner groups* so a vertex a seam passes through
// gets one UV per side of the cut (a cylinder cut by one seam line gets two copies of every seam vertex).
// Doesn't: repair non-manifold input (an edge shared by other than two triangles is treated as a
// boundary: it never joins triangles), nor invent seams (`seams(path:)` is the only helper that cuts).
// CPU only: this is index bookkeeping with data-dependent branches, not per-vertex number crunching.

/// An undirected edge between two welded vertices (`a < b`).
public struct UVEdge: Hashable, Sendable {
    public let a: Int
    public let b: Int
    public init(_ x: Int, _ y: Int) { a = min(x, y); b = max(x, y) }
}

/// A set of edges the surface is cut along.
public struct UVSeams: Sendable, Equatable {
    public var edges: Set<UVEdge>
    public init(edges: Set<UVEdge> = []) { self.edges = edges }
    public func contains(_ e: UVEdge) -> Bool { edges.contains(e) }
    public func union(_ other: UVSeams) -> UVSeams { UVSeams(edges: edges.union(other.edges)) }
    public var count: Int { edges.count }
}

/// One piece of the surface after cutting: its triangles and, per triangle corner, a local vertex id.
public struct UVIsland: Sendable {
    /// Triangle indices into the mesh (triangle `t` uses `indices[3t..<3t+3]`).
    public var triangles: [Int]
    /// Local vertex id for each corner, 3 per entry of `triangles`.
    public var corners: [Int]
    public var vertexCount: Int
}

public struct UVTopology: Sendable {
    public let weldedID: [Int]
    public let weldedCount: Int
    public let triangleCount: Int
    public let edgeTriangles: [UVEdge: [Int]]
    public let triangleNormals: [Vec3]
    public let triangleAreas: [Double]
    let indices: [UInt32]

    public init(_ mesh: Mesh3, tolerance: Double = 1e-6) {
        struct Key: Hashable { let x: Int64, y: Int64, z: Int64 }
        var ids: [Key: Int] = [:]
        var weld = [Int](repeating: 0, count: mesh.positions.count)
        for (i, p) in mesh.positions.enumerated() {
            let k = Key(x: Int64((p.x / tolerance).rounded()), y: Int64((p.y / tolerance).rounded()),
                        z: Int64((p.z / tolerance).rounded()))
            if let id = ids[k] { weld[i] = id } else { weld[i] = ids.count; ids[k] = ids.count }
        }
        weldedID = weld; weldedCount = ids.count; indices = mesh.indices
        let tc = mesh.indices.count / 3
        triangleCount = tc
        var et: [UVEdge: [Int]] = [:]
        var nrm = [Vec3](), area = [Double]()
        nrm.reserveCapacity(tc); area.reserveCapacity(tc)
        for t in 0..<tc {
            let i = (0..<3).map { Int(mesh.indices[3 * t + $0]) }
            let c = cross3(mesh.positions[i[1]] - mesh.positions[i[0]], mesh.positions[i[2]] - mesh.positions[i[0]])
            let l = len3(c)
            area.append(l / 2); nrm.append(l > 1e-300 ? c / l : .zero)
            for k in 0..<3 {
                let a = weld[i[k]], b = weld[i[(k + 1) % 3]]
                if a != b { et[UVEdge(a, b), default: []].append(t) }
            }
        }
        edgeTriangles = et; triangleNormals = nrm; triangleAreas = area
    }

    /// Welded vertex id of corner `c` of triangle `t`.
    public func welded(_ t: Int, _ c: Int) -> Int { weldedID[Int(indices[3 * t + c])] }
    /// Mesh vertex index of corner `c` of triangle `t`.
    public func vertex(_ t: Int, _ c: Int) -> Int { Int(indices[3 * t + c]) }
    func corner(of w: Int, in t: Int) -> Int? { (0..<3).first { welded(t, $0) == w } }

    /// Edges used by exactly one triangle.
    public var boundaryEdges: Set<UVEdge> { Set(edgeTriangles.filter { $0.value.count == 1 }.keys) }

    // MARK: Seam selection

    /// Seams from explicit pairs of *mesh vertex indices* (welded internally, order irrelevant).
    public func seams(edges pairs: [(Int, Int)]) -> UVSeams {
        UVSeams(edges: Set(pairs.map { UVEdge(weldedID[$0.0], weldedID[$0.1]) }.filter { $0.a != $0.b }))
    }

    /// Edges whose two triangles' normals differ by more than `degrees` (and, optionally, open edges).
    public func seams(angleDegrees degrees: Double, includeBoundary: Bool = false) -> UVSeams {
        let limit = cos(degrees * .pi / 180)
        var s = Set<UVEdge>()
        for (e, ts) in edgeTriangles {
            if ts.count == 2 {
                if dot3(triangleNormals[ts[0]], triangleNormals[ts[1]]) < limit { s.insert(e) }
            } else if includeBoundary { s.insert(e) }
        }
        return UVSeams(edges: s)
    }

    /// Edges where the mesh's existing UVs break: the two triangles disagree about a shared endpoint's UV.
    public func seams(uvBoundaryOf mesh: Mesh3, tolerance: Double = 1e-6) -> UVSeams {
        guard mesh.uvs.count == mesh.positions.count else { return UVSeams() }
        func uv(_ t: Int, _ w: Int) -> Vec2 { mesh.uvs[vertex(t, corner(of: w, in: t) ?? 0)] }
        var s = Set<UVEdge>()
        for (e, ts) in edgeTriangles where ts.count == 2 {
            for w in [e.a, e.b] where len(uv(ts[0], w) - uv(ts[1], w)) > tolerance { s.insert(e); break }
        }
        return UVSeams(edges: s)
    }

    /// The shortest edge path (by 3D length) between two mesh vertices, as seams. Cutting a sphere pole to
    /// pole, or a closed tube end to end, makes it a disc. Empty when the vertices aren't connected.
    public func seams(pathFrom from: Int, to: Int, in mesh: Mesh3) -> UVSeams {
        var adj = [[Int]](repeating: [], count: weldedCount)
        var pos = [Vec3](repeating: .zero, count: weldedCount)
        for (i, p) in mesh.positions.enumerated() { pos[weldedID[i]] = p }
        for e in edgeTriangles.keys { adj[e.a].append(e.b); adj[e.b].append(e.a) }
        let s = weldedID[from], g = weldedID[to]
        var dist = [Double](repeating: .infinity, count: weldedCount)
        var prev = [Int](repeating: -1, count: weldedCount)
        var done = [Bool](repeating: false, count: weldedCount)
        dist[s] = 0
        // O(V²) Dijkstra: seam picking is interactive-scale and this avoids a heap type.
        while true {
            var u = -1, best = Double.infinity
            for i in 0..<weldedCount where !done[i] && dist[i] < best { best = dist[i]; u = i }
            if u < 0 || u == g { break }
            done[u] = true
            for v in adj[u] {
                let d = dist[u] + len3(pos[u] - pos[v])
                if d < dist[v] { dist[v] = d; prev[v] = u }
            }
        }
        guard dist[g].isFinite else { return UVSeams() }
        var out = Set<UVEdge>(), cur = g
        while cur != s { out.insert(UVEdge(cur, prev[cur])); cur = prev[cur] }
        return UVSeams(edges: out)
    }

    // MARK: Islands

    /// Cut the surface at `seams` and return its pieces, largest-index-stable (ordered by first triangle).
    public func islands(_ seams: UVSeams) -> [UVIsland] {
        var cornerUF = UnionFind(3 * triangleCount), triUF = UnionFind(triangleCount)
        for (e, ts) in edgeTriangles where ts.count == 2 && !seams.contains(e) {
            triUF.union(ts[0], ts[1])
            for w in [e.a, e.b] {
                if let c0 = corner(of: w, in: ts[0]), let c1 = corner(of: w, in: ts[1]) {
                    cornerUF.union(3 * ts[0] + c0, 3 * ts[1] + c1)
                }
            }
        }
        var order: [Int: Int] = [:]
        var islands: [UVIsland] = []
        var local: [[Int: Int]] = []
        for t in 0..<triangleCount {
            let r = triUF.find(t)
            let k: Int
            if let found = order[r] { k = found } else {
                k = islands.count; order[r] = k
                islands.append(UVIsland(triangles: [], corners: [], vertexCount: 0)); local.append([:])
            }
            islands[k].triangles.append(t)
            for c in 0..<3 {
                let root = cornerUF.find(3 * t + c)
                let id: Int
                if let f = local[k][root] { id = f } else { id = local[k].count; local[k][root] = id }
                islands[k].corners.append(id)
            }
        }
        for k in 0..<islands.count { islands[k].vertexCount = local[k].count }
        return islands
    }
}

struct UnionFind {
    var parent: [Int]
    init(_ n: Int) { parent = Array(0..<n) }
    mutating func find(_ x: Int) -> Int {
        var r = x
        while parent[r] != r { r = parent[r] }
        var c = x
        while parent[c] != r { let n = parent[c]; parent[c] = r; c = n }
        return r
    }
    mutating func union(_ a: Int, _ b: Int) {
        let ra = find(a), rb = find(b)
        if ra != rb { parent[max(ra, rb)] = min(ra, rb) }
    }
}
