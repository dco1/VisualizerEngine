//  Consistency checks for HalfEdgeMesh. `validate()` returns every problem it finds (empty = sound).
//  Errors mean the structure is broken (a tool must not continue); warnings are legal-but-suspect
//  (bow-tie vertices, zero-area faces, non-finite positions). Read-only, O(H).

import simd

public struct HalfEdgeIssue: Sendable, CustomStringConvertible, Equatable {
    public enum Severity: Sendable { case error, warning }
    public var severity: Severity
    public var message: String
    public var description: String { "\(severity == .error ? "error" : "warning"): \(message)" }
}

public extension HalfEdgeMesh {

    /// Every inconsistency in the mesh. A sound mesh returns `[]`.
    func validate() -> [HalfEdgeIssue] {
        var out: [HalfEdgeIssue] = []
        func err(_ s: String) { out.append(HalfEdgeIssue(severity: .error, message: s)) }
        func warn(_ s: String) { out.append(HalfEdgeIssue(severity: .warning, message: s)) }
        let H = heOrigin.count
        guard [heNext.count, hePrev.count, heTwin.count, heFace.count, heEdge.count, heUV.count, heNormal.count]
            .allSatisfy({ $0 == H }), vertexHE.count == positions.count else { err("array lengths disagree"); return out }
        func ok(_ h: Int) -> Bool { h >= 0 && h < H && heFace[h] >= 0 }

        var edgeUse = [Int](repeating: 0, count: edgeHE.count)
        var alive = 0
        for h in 0..<H where heFace[h] >= 0 {
            alive += 1
            let f = heFace[h]
            if f >= faceHE.count || faceHE[f] < 0 { err("halfedge \(h) belongs to dead face \(f)"); continue }
            guard ok(heNext[h]), ok(hePrev[h]) else { err("halfedge \(h) has a dead next/prev"); continue }
            if hePrev[heNext[h]] != h { err("halfedge \(h): prev(next) != self") }
            if heNext[hePrev[h]] != h { err("halfedge \(h): next(prev) != self") }
            if heFace[heNext[h]] != f { err("halfedge \(h): next is in another face") }
            if heOrigin[h] < 0 || heOrigin[h] >= positions.count { err("halfedge \(h): origin out of range"); continue }
            let e = heEdge[h]
            if e < 0 || e >= edgeHE.count || edgeHE[e] < 0 { err("halfedge \(h) has dead edge \(e)") } else { edgeUse[e] += 1 }
            let dest = heOrigin[heNext[h]]
            if heOrigin[h] == dest { err("halfedge \(h) is a loop at vertex \(dest)") }
            if directed[Self.key(heOrigin[h], dest)] != h { err("directed table disagrees at halfedge \(h)") }
            let t = heTwin[h]
            if t >= 0 {
                if !ok(t) { err("halfedge \(h) twin \(t) is dead") }
                else {
                    if heTwin[t] != h { err("halfedge \(h): twin(twin) != self") }
                    if heOrigin[t] != dest || heOrigin[heNext[t]] != heOrigin[h] { err("halfedge \(h): twin runs the wrong way") }
                    if heEdge[t] != e { err("halfedge \(h): twin has another edge") }
                    if heFace[t] == f { err("halfedge \(h): twin in the same face") }
                }
            } else if directed[Self.key(dest, heOrigin[h])] != nil { err("halfedge \(h) lacks the twin the directed table has") }
        }
        if directed.count != alive { err("directed table has \(directed.count) entries for \(alive) halfedges") }
        for e in 0..<edgeHE.count where edgeHE[e] >= 0 {
            let h = edgeHE[e]
            if !ok(h) || heEdge[h] != e { err("edge \(e) points at a wrong halfedge"); continue }
            let want = heTwin[h] >= 0 ? 2 : 1
            if edgeUse[e] != want { err("edge \(e) has \(edgeUse[e]) halfedges, expected \(want)") }
        }
        for f in 0..<faceHE.count where faceHE[f] >= 0 {
            guard ok(faceHE[f]), heFace[faceHE[f]] == f else { err("face \(f) points at a wrong halfedge"); continue }
            var h = faceHE[f], n = 0
            repeat { h = heNext[h]; n += 1 } while h != faceHE[f] && n <= H && ok(h)
            if h != faceHE[f] { err("face \(f) loop does not close"); continue }
            if n < 3 { err("face \(f) has \(n) corners") }
            let vs = vertices(of: f)
            if Set(vs).count != vs.count { err("face \(f) repeats a vertex") }
            if faceArea(f) < 1e-18 { warn("face \(f) has zero area") }
        }
        var used = Set<Int>()
        for h in 0..<H where heFace[h] >= 0 && heOrigin[h] >= 0 && heOrigin[h] < positions.count { used.insert(heOrigin[h]) }
        for v in 0..<positions.count {
            let h = vertexHE[v]
            if h >= 0 { if !ok(h) || heOrigin[h] != v { err("vertex \(v) points at a wrong halfedge") } }
            else if used.contains(v) { err("vertex \(v) is used but marked isolated") }
            let p = positions[v]
            if !(p.x.isFinite && p.y.isFinite && p.z.isFinite) { warn("vertex \(v) is not finite") }
        }
        if out.allSatisfy({ $0.severity == .warning }) {
            for v in nonManifoldVertices() { warn("vertex \(v) is a bow-tie (non-manifold)") }
        }
        return out
    }

    /// No errors (warnings allowed).
    var isValid: Bool { !validate().contains { $0.severity == .error } }
}
