//  GraphCanonical.swift — a canonical vertex / triangle order, so a graph's result is byte-stable.
//
//  Several engine operators (subdivide, bevel, decimate, booleans) build their output by walking Swift
//  dictionaries, whose iteration order is seeded per instance: the SURFACE is identical run to run, the vertex and
//  triangle ORDER is not. A graph promises "same graph, same mesh", so the nodes that wrap those operators sort
//  their output: each triangle is rotated to start at its smallest corner (winding kept), triangles are sorted by
//  their corners, and vertices are numbered by first use (identical vertices are merged).
//
//  Those sums also differ in the last bits, so coordinates are snapped to 1 nm (normals, UVs to 1e-9) first.
//  Applied only to what LEAVES the graph (`evaluate`'s target), never between nodes: snapping a boolean's output
//  before the next boolean read it left sub-tolerance slivers and opened edges (ModelGraphBooleanChainTests).
//  Guarantees: same surface in, same arrays out, whatever order the operator produced; winding and the per-corner
//  normal / UV are untouched. Not guaranteed: two different surfaces never compare equal, but a surface with exactly
//  coincident duplicate triangles keeps both. O(n log n), CPU; a mesh pass at authoring size.

import Foundation

extension Mesh3 {
    func canonicalized() -> Mesh3 {
        guard !indices.isEmpty, normals.count == positions.count, uvs.count == positions.count else { return self }
        typealias Key = [Double]
        func snap(_ v: Double) -> Double { (v * 1e9).rounded() / 1e9 }
        func key(_ i: Int) -> Key {
            let p = positions[i], n = normals[i], t = uvs[i]
            return [p.x, p.y, p.z, n.x, n.y, n.z, t.x, t.y].map(snap)
        }
        func less(_ a: Key, _ b: Key) -> Bool {
            for i in 0 ..< a.count where a[i] != b[i] { return a[i] < b[i] }
            return false
        }
        var tris: [(k: [Key], v: [Int])] = []
        tris.reserveCapacity(indices.count / 3)
        var t = 0
        while t + 2 < indices.count {
            var v = [Int(indices[t]), Int(indices[t + 1]), Int(indices[t + 2])]
            t += 3
            var k = v.map(key)
            var lo = 0
            for j in 1 ..< 3 where less(k[j], k[lo]) { lo = j }
            if lo != 0 { v = Array(v[lo...] + v[..<lo]); k = Array(k[lo...] + k[..<lo]) }
            tris.append((k, v))
        }
        tris.sort { a, b in
            for j in 0 ..< 3 { if less(a.k[j], b.k[j]) { return true }; if less(b.k[j], a.k[j]) { return false } }
            return false
        }
        struct Bits: Hashable { let a: [UInt64] }
        var ids: [Bits: UInt32] = [:]
        var out = Mesh3()
        for tri in tris {
            for j in 0 ..< 3 {
                let b = Bits(a: tri.k[j].map(\.bitPattern))
                if let id = ids[b] { out.indices.append(id) } else {
                    let id = UInt32(out.positions.count)
                    ids[b] = id
                    let k = tri.k[j]
                    out.positions.append(Vec3(k[0], k[1], k[2])); out.normals.append(Vec3(k[3], k[4], k[5])); out.uvs.append(Vec2(k[6], k[7]))
                    out.indices.append(id)
                }
            }
        }
        return out
    }
}
