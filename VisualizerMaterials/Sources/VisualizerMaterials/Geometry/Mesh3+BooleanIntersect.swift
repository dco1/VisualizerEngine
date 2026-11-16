//  Triangle–triangle intersection for the mesh-intersection boolean.
//
//  Which side of a triangle's plane a vertex is on is decided with the EXACT `Robust.orient3d` signs, so every
//  triangle that shares that vertex reaches the same verdict. Only the intersection POINTS are computed in plain
//  doubles (a lerp along an edge); the pipeline welds them by tolerance.
//
//  Result: nothing, a segment (transversal crossing), or a convex polygon (the two triangles are coplanar to
//  within `eps`: the overlap of their areas, whose edges become constraints on both).
//  A single touching point yields nothing. Two triangles that merely share a boundary edge yield nothing
//  (the shared edge is already an edge of both).

import Foundation
import simd

enum PairHit {
    case none
    case segment(Vec3, Vec3)
    case polygon([Vec3])
}

enum TriTri {
    static func intersect(_ T: (Vec3, Vec3, Vec3), _ S: (Vec3, Vec3, Vec3), eps: Double) -> PairHit {
        let nT = cross3(T.1 - T.0, T.2 - T.0)
        let nTl = len3(nT)
        guard nTl > 0 else { return .none }
        let nTu = nT / nTl
        let sv = [S.0, S.1, S.2]
        let d = sv.map { dot3(nTu, $0 - T.0) }

        // coplanar to within eps: both ways
        if d.allSatisfy({ abs($0) <= eps }) {
            let nS = cross3(S.1 - S.0, S.2 - S.0)
            let nSl = len3(nS)
            if nSl > 0 {
                let nSu = nS / nSl
                if [T.0, T.1, T.2].allSatisfy({ abs(dot3(nSu, $0 - S.0)) <= eps }) {
                    return coplanar(T, nTu, sv, eps: eps)
                }
            }
        }

        // exact signs
        let s = sv.map { Robust.orient3d(T.0, T.1, T.2, $0) }
        if s[0] > 0 && s[1] > 0 && s[2] > 0 { return .none }
        if s[0] < 0 && s[1] < 0 && s[2] < 0 { return .none }
        let tv = [T.0, T.1, T.2]
        let st = tv.map { Robust.orient3d(S.0, S.1, S.2, $0) }
        if st[0] > 0 && st[1] > 0 && st[2] > 0 { return .none }
        if st[0] < 0 && st[1] < 0 && st[2] < 0 { return .none }

        // S ∩ plane(T): points on S's vertices and edges
        var pts: [Vec3] = []
        for i in 0..<3 where s[i] == 0 { pts.append(sv[i]) }
        for i in 0..<3 {
            let j = (i + 1) % 3
            if s[i] * s[j] < 0 {
                let den = d[i] - d[j]
                let t = den == 0 ? 0.5 : min(1, max(0, d[i] / den))
                pts.append(sv[i] + (sv[j] - sv[i]) * t)
            }
        }
        guard pts.count >= 2 else { return .none }
        var P = pts[0], Q = pts[1]
        if pts.count > 2 {                                   // keep the farthest pair
            var best = -1.0
            for i in 0..<pts.count { for j in (i + 1)..<pts.count {
                let l = len3(pts[i] - pts[j])
                if l > best { best = l; P = pts[i]; Q = pts[j] }
            } }
        }
        // clip to T (Cyrus–Beck against its three inward half-planes)
        let tol = eps * 0.5
        for e in 0..<3 {
            let a = tv[e], b = tv[(e + 1) % 3]
            let m = normalize3(cross3(nTu, b - a))
            let dP = dot3(m, P - a), dQ = dot3(m, Q - a)
            let inP = dP >= -tol, inQ = dQ >= -tol
            if inP && inQ { continue }
            if !inP && !inQ { return .none }
            let t = dP / (dP - dQ)
            let D = Q - P
            if !inP { P = P + D * t } else { Q = P + D * t }
        }
        if len3(P - Q) <= eps { return .none }
        return .segment(P, Q)
    }

    private static func coplanar(_ T: (Vec3, Vec3, Vec3), _ n: Vec3, _ S: [Vec3], eps: Double) -> PairHit {
        var poly = S
        let tv = [T.0, T.1, T.2]
        for e in 0..<3 {
            let a = tv[e], b = tv[(e + 1) % 3]
            let m = normalize3(cross3(n, b - a))
            var out: [Vec3] = []
            for i in 0..<poly.count {
                let c = poly[i], nx = poly[(i + 1) % poly.count]
                let dc = dot3(m, c - a), dn = dot3(m, nx - a)
                if dc >= 0 { out.append(c) }
                if (dc >= 0) != (dn >= 0) {
                    let t = dc / (dc - dn)
                    out.append(c + (nx - c) * t)
                }
            }
            poly = out
            if poly.count < 3 { return .none }
        }
        var clean: [Vec3] = []
        for p in poly where clean.last.map({ len3($0 - p) > eps }) ?? true { clean.append(p) }
        while clean.count > 1, len3(clean[0] - clean[clean.count - 1]) <= eps { clean.removeLast() }
        guard clean.count >= 3 else { return .none }
        var area = Vec3.zero
        for i in 1..<(clean.count - 1) { area += cross3(clean[i] - clean[0], clean[i + 1] - clean[0]) }
        if len3(area) * 0.5 <= eps * eps * 100 { return .none }
        return .polygon(clean)
    }
}
