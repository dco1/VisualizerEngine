import Foundation
import simd

// ── CoinHullTopology ─────────────────────────────────────────────────────────
//
// Face / edge topology of a registered convex hull, for the EXACT polytope
// narrowphase in CoinDEMNarrowphase.h (separating-axis test over face normals and
// Gauss-map-pruned edge pairs, then reference/incident face clipping). Computed ONCE
// at `registerHull`, never per frame.
//
// WHY. The GPU used to know a hull only as a vertex list. Against a static box it
// probed the VERTICES alone (trap T5): a bar edge crossing a box edge, or a box edge
// pressing into a bar FACE, has no vertex inside anything, so the pair passed through
// each other (a bar resting across a pocket rim sank into the rim). Hull–hull pairs
// went through GJK/EPA + an "eps band" support set whose band was an ABSOLUTE 1 mm +
// 2 % R — at the Digital Clock's millimetre scale that band swallowed the bar's
// chamfer ring, and flat face–face GJK simplices fed EPA a bogus 0.26 mm depth for two
// bars 0.01 mm apart. Real faces fix both: SAT is exact for polytopes, and the
// contact polygon is the true face overlap.
//
// OUTPUT (all in the hull's principal frame, indices into `Prepared.vertices`):
//   • faces — coplanar quickhull triangles merged into convex polygons: outward unit
//     normal, plane offset, and the vertex loop CCW about the normal;
//   • edges — every edge once, with the two faces it separates (their normals are
//     the edge's arc on the Gauss map).
// A hull whose merged face exceeds `maxFaceVertices` (the GPU clip buffer), or whose
// counts exceed the GPU caps, gets no topology and keeps the vertex-probe paths.

struct CoinHullTopology: Sendable {
    struct Face: Sendable {
        var normal: SIMD3<Double>      // unit, outward
        var offset: Double             // normal · x on the face
        var loop: [Int]                // CCW about `normal`
    }
    struct Edge: Sendable {
        var v0: Int, v1: Int           // endpoints (v0 → v1 runs CCW on face f0)
        var f0: Int, f1: Int           // the two faces the edge separates
    }
    var faces: [Face]
    var edges: [Edge]

    /// GPU limits (CoinDEMNarrowphase.h: CD_PMAXV, CD_PMAXF, CD_PMAXE).
    static let maxFaceVertices = 16
    static let maxFaces = 64
    static let maxEdges = 192

    /// Merge the outward-wound triangles `tris` (indices into `pts`) into polygon
    /// faces. `tol` is the plane-distance tolerance quickhull used (scale-relative).
    static func build(points pts: [SIMD3<Double>], triangles tris: [(Int, Int, Int)], tol: Double) -> CoinHullTopology? {
        let nT = tris.count
        guard nT >= 4 else { return nil }
        let nV = pts.count
        func key(_ u: Int, _ v: Int) -> Int { u &* nV &+ v }
        var owner: [Int: Int] = [:]                              // directed edge → triangle
        var tn: [SIMD3<Double>] = [], td: [Double] = [], ta: [Double] = []
        for (k, t) in tris.enumerated() {
            let c = simd_cross(pts[t.1] - pts[t.0], pts[t.2] - pts[t.0])
            let a = simd_length(c)
            let n = a > 0 ? c / a : SIMD3<Double>.zero
            tn.append(n); td.append(simd_dot(n, pts[t.0])); ta.append(a)
            owner[key(t.0, t.1)] = k; owner[key(t.1, t.2)] = k; owner[key(t.2, t.0)] = k
        }
        // Union-find over edge-adjacent coplanar triangles: a neighbour is coplanar when
        // its far vertex lies on this triangle's plane (and vice versa) within `tol`,
        // with the same orientation.
        var parent = Array(0..<nT)
        func find(_ x: Int) -> Int { var r = x; while parent[r] != r { r = parent[r] }; var y = x
            while parent[y] != r { let nx = parent[y]; parent[y] = r; y = nx }; return r }
        func third(_ t: (Int, Int, Int), _ u: Int, _ v: Int) -> Int {
            [t.0, t.1, t.2].first { $0 != u && $0 != v }!
        }
        for (k, t) in tris.enumerated() {
            for (u, v) in [(t.0, t.1), (t.1, t.2), (t.2, t.0)] {
                guard let g = owner[key(v, u)], g != k else { return nil }   // not closed
                let w = third(tris[g], u, v), wk = third(t, u, v)
                let onK = abs(simd_dot(tn[k], pts[w]) - td[k]) <= tol
                let onG = abs(simd_dot(tn[g], pts[wk]) - td[g]) <= tol
                if onK && onG && simd_dot(tn[k], tn[g]) > 0 {
                    let a = find(k), b = find(g)
                    if a != b { parent[max(a, b)] = min(a, b) }
                }
            }
        }
        // Faces in order of their lowest triangle (quickhull's order is deterministic).
        var faceOf = [Int](repeating: -1, count: nT)
        var groups: [[Int]] = []
        var rootFace: [Int: Int] = [:]
        for k in 0..<nT {
            let r = find(k)
            if let f = rootFace[r] { faceOf[k] = f; groups[f].append(k) }
            else { rootFace[r] = groups.count; faceOf[k] = groups.count; groups.append([k]) }
        }
        var faces: [Face] = []
        for (f, g) in groups.enumerated() {
            // Boundary: directed edges whose twin triangle lies in another face.
            var next: [Int: Int] = [:]
            var areaN = SIMD3<Double>.zero
            for k in g {
                let t = tris[k]
                areaN += tn[k] * ta[k]
                for (u, v) in [(t.0, t.1), (t.1, t.2), (t.2, t.0)] {
                    guard let tw = owner[key(v, u)] else { return nil }
                    if faceOf[tw] != f {
                        if next[u] != nil { return nil }                     // non-manifold
                        next[u] = v
                    }
                }
            }
            guard let start = next.keys.min() else { return nil }
            var loop = [start]
            var cur = next[start]!
            while cur != start {
                guard loop.count <= next.count, let nx = next[cur] else { return nil }
                loop.append(cur); cur = nx
            }
            guard loop.count == next.count, loop.count >= 3 else { return nil }
            let ln = simd_length(areaN)
            guard ln > 0 else { return nil }
            let n = areaN / ln
            var off = 0.0
            for v in loop { off += simd_dot(n, pts[v]) }
            off /= Double(loop.count)
            for v in loop where abs(simd_dot(n, pts[v]) - off) > 4 * tol { return nil }
            faces.append(Face(normal: n, offset: off, loop: loop))
        }
        // Edges: each boundary edge once (from the face with the smaller index).
        var edgeFace: [Int: Int] = [:]
        for (f, face) in faces.enumerated() {
            for i in 0..<face.loop.count { edgeFace[key(face.loop[i], face.loop[(i + 1) % face.loop.count])] = f }
        }
        var edges: [Edge] = []
        for (f, face) in faces.enumerated() {
            for i in 0..<face.loop.count {
                let u = face.loop[i], v = face.loop[(i + 1) % face.loop.count]
                guard let g = edgeFace[key(v, u)] else { return nil }
                if f < g { edges.append(Edge(v0: u, v1: v, f0: f, f1: g)) }
            }
        }
        var used = Set<Int>()
        for face in faces { used.formUnion(face.loop) }
        guard used.count - edges.count + faces.count == 2 else { return nil }   // Euler: V − E + F = 2
        return CoinHullTopology(faces: faces, edges: edges)
    }

    /// Whether the GPU narrowphase can use this topology (else: vertex-probe fallback).
    var fitsGPU: Bool {
        faces.count <= Self.maxFaces && edges.count <= Self.maxEdges
            && faces.allSatisfy { $0.loop.count <= Self.maxFaceVertices }
    }

    /// Number of float4 slots `pack` writes after a hull's vertices (the header included).
    var packedCount: Int {
        let loopTotal = faces.reduce(0) { $0 + $1.loop.count }
        return 1 + 2 * faces.count + loopTotal + edges.count
    }

    /// The GPU layout, appended after the hull's vertices in `hullVertexBuffer`
    /// (CoinDEMNarrowphase.h `cdHullTopoOf`):
    ///   [0]                header  uint4(faceCount, edgeCount, loopTotal, flags = 1)
    ///   [1 ..< 1+2F]       face f: (n.xyz, d) then uint4(loopStart, loopCount, 0, 0)
    ///   [.. + L]           loop entries: (vertex.xyz, vertex index as bits) — the position
    ///                      inline, so a face clip reads each corner with ONE load
    ///   [.. + E]           edge e: uint4(v0, v1, f0, f1)
    func pack(vertices: [SIMD3<Float>]) -> [SIMD4<Float>] {
        func u(_ a: Int, _ b: Int, _ c: Int, _ d: Int) -> SIMD4<Float> {
            SIMD4(Float(bitPattern: UInt32(a)), Float(bitPattern: UInt32(b)),
                  Float(bitPattern: UInt32(c)), Float(bitPattern: UInt32(d)))
        }
        let loopTotal = faces.reduce(0) { $0 + $1.loop.count }
        var out: [SIMD4<Float>] = [u(faces.count, edges.count, loopTotal, 1)]
        var cursor = 0
        for f in faces {
            out.append(SIMD4(Float(f.normal.x), Float(f.normal.y), Float(f.normal.z), Float(f.offset)))
            out.append(u(cursor, f.loop.count, 0, 0))
            cursor += f.loop.count
        }
        for f in faces {
            for v in f.loop { out.append(SIMD4(vertices[v], Float(bitPattern: UInt32(v)))) }
        }
        for e in edges { out.append(u(e.v0, e.v1, e.f0, e.f1)) }
        return out
    }

    /// The header a hull WITHOUT usable topology gets (flags 0): the kernels read it
    /// and fall back to the vertex-probe / GJK paths.
    static let emptyHeader = SIMD4<Float>(0, 0, 0, 0)
}
