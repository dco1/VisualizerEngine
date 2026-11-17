//  Isotropic remeshing (Botsch & Kobbelt 2004) — Blender's Remesh (Quad-free "Voxel/Smooth" cousin) for `Mesh3`.
//
//  Guarantees: after `iterations` rounds of split-long (> 4/3 L) → collapse-short (< 4/5 L, never creating an edge
//  longer than 4/3 L) → flip toward valence 6 (4 on boundaries) → tangential smoothing projected back onto the
//  ORIGINAL surface, the edge lengths cluster around `targetEdgeLength`, the shape stays within a fraction of L
//  of the input, open boundaries and creases (≥ `creaseDegrees`) keep their vertices fixed (they can still be
//  split and shortened along themselves), winding is preserved and a closed mesh stays closed.
//  Does not guarantee: UVs (the result has none — remeshing invents a new tessellation; re-project afterwards),
//  or perfect uniformity on very thin features (< L).
//
//  CPU: topology edits (split / collapse / flip) are graph surgery. The smoothing + projection step is per-vertex
//  and is the part a Metal port would take; it is a one-time modelling operation, not per-frame.

import simd

public extension Mesh3 {

    func isotropicRemeshed(targetEdgeLength L: Double, iterations: Int = 5, creaseDegrees: Double = 40) -> Mesh3 {
        guard L > 0, !indices.isEmpty else { return self }
        var m = self; m.uvs = []              // UVs are dropped: the tessellation is new
        var t = TriTopology(m, creaseDegrees: creaseDegrees)
        let projector = SurfaceProjector(t, cell: L * 2)
        let hi = L * 4 / 3, lo = L * 4 / 5
        let cosTurn = cos(creaseDegrees * .pi / 180)
        func edges(_ t: TriTopology) -> [(Int, Int)] {
            var out: [(Int, Int)] = []
            for v in 0 ..< t.pos.count { for w in t.neighbors(v) where v < w { out.append((v, w)) } }
            return out
        }
        for _ in 0 ..< iterations {
            // 1. split long edges (longest first, a few sweeps)
            for _ in 0 ..< 4 {
                var did = false
                for (a, b) in edges(t).filter({ t.edgeLength($0.0, $0.1) > hi }).sorted(by: { t.edgeLength($0.0, $0.1) > t.edgeLength($1.0, $1.1) }) {
                    guard !t.facesOn(a, b).isEmpty, t.edgeLength(a, b) > hi else { continue }
                    var mid = (t.pos[a] + t.pos[b]) * 0.5
                    if !t.isFeatureEdge(a, b) { mid = projector.closest(to: mid) }
                    t.split(a, b, at: mid); did = true
                }
                if !did { break }
            }
            // 2. collapse short edges
            for (a, b) in edges(t).filter({ t.edgeLength($0.0, $0.1) < lo }).sorted(by: { t.edgeLength($0.0, $0.1) < t.edgeLength($1.0, $1.1) }) {
                guard !t.vf[a].isEmpty, !t.vf[b].isEmpty, !t.facesOn(a, b).isEmpty, t.edgeLength(a, b) < lo else { continue }
                let rule = t.collapseRule(a, b, cosTurn: cosTurn)
                var cands: [(keep: Int, remove: Int, p: Vec3)] = []
                if rule.free { cands = [(b, a, (t.pos[a] + t.pos[b]) * 0.5)] }
                else {
                    if rule.removeU { cands.append((b, a, t.pos[b])) }
                    if rule.removeV { cands.append((a, b, t.pos[a])) }
                }
                for c in cands {
                    let ring = Set(t.neighbors(a) + t.neighbors(b)).subtracting([a, b])
                    if ring.contains(where: { len3(c.p - t.pos[$0]) > hi }) { continue }
                    guard t.collapseIsValid(keep: c.keep, remove: c.remove, to: c.p) else { continue }
                    t.collapse(keep: c.keep, remove: c.remove, to: c.p, lerpT: 0.5)
                    break
                }
            }
            // 3. flip toward the ideal valence
            func target(_ v: Int) -> Int { t.vf[v].count < t.neighbors(v).count ? 4 : 6 }
            for (a, b) in edges(t) {
                let sh = t.facesOn(a, b)
                guard sh.count == 2, !t.isFeatureEdge(a, b) else { continue }
                let x1 = t.third(sh[0], a, b), x2 = t.third(sh[1], a, b)
                let vs = [a, b, x1, x2]
                let val = vs.map { t.neighbors($0).count }, tg = vs.map(target)
                let before = (0 ..< 4).reduce(0) { $0 + (val[$1] - tg[$1]) * (val[$1] - tg[$1]) }
                let after = (val[0] - 1 - tg[0]) * (val[0] - 1 - tg[0]) + (val[1] - 1 - tg[1]) * (val[1] - 1 - tg[1])
                    + (val[2] + 1 - tg[2]) * (val[2] + 1 - tg[2]) + (val[3] + 1 - tg[3]) * (val[3] + 1 - tg[3])
                if after < before { t.flip(a, b) }
            }
            // 4. tangential relaxation, projected back onto the input surface
            for v in 0 ..< t.pos.count where !t.vf[v].isEmpty && t.featureNeighbors(v).isEmpty {
                let nb = t.neighbors(v)
                let c = nb.reduce(Vec3.zero) { $0 + t.pos[$1] } / Double(nb.count)
                let n = normalize3(t.vf[v].reduce(Vec3.zero) { $0 + t.rawNormal($1) })
                var d = c - t.pos[v]
                d -= n * dot3(d, n)
                t.pos[v] = projector.closest(to: t.pos[v] + d * 0.8)
            }
        }
        return t.toMesh3(creaseDegrees: creaseDegrees)
    }
}
