//  Quadric-error edge-collapse decimation (Garland–Heckbert) — Blender's Decimate modifier, "Collapse" mode.
//
//  Guarantees:
//   · Reaches `targetTriangles` (or stops early when no legal collapse is left — see `Report.triangles`).
//   · Open boundaries, crease edges (dihedral ≥ `creaseDegrees`) and UV seams are features: a feature vertex only
//     collapses along a feature edge into a neighbour, so it never leaves its feature; corners (three or more
//     feature edges, or a feature curve bending by ≥ `creaseDegrees`) are never removed. Constraint planes along
//     boundary and crease edges (weighted) also make collapsing along them cost something when they bend.
//   · No collapse is taken that breaks the link condition (so manifold stays manifold; a closed mesh stays
//     closed) or folds / squashes a neighbouring triangle.
//   · Winding is preserved; UVs are carried per corner (seams stay split, interior UVs interpolate).
//   · `Report.maxError` is the largest quadric error paid, as a distance in metres: the worst sqrt(Σ distance² to
//     the planes the collapsed vertex stood for). It bounds how far a collapse moved the surface from the
//     planes it had accumulated; it is a good ordering/size guide, not a true Hausdorff distance (tests measure
//     that separately).
//  Does not guarantee: normals of the input are kept (they are rebuilt from the result, smooth within
//  `creaseDegrees`), or that disconnected tiny shells survive a deep target.
//
//  CPU: a priority queue over a mutable mesh (lazy-invalidated binary heap); graph surgery, not data-parallel.

import simd

public extension Mesh3 {

    /// What a decimation did.
    struct DecimationReport: Sendable {
        public var triangles = 0          // triangles left
        public var collapses = 0
        public var maxError = 0.0         // metres, see the file header
    }

    /// Decimate to about `targetTriangles` (never below what legal collapses allow).
    func decimated(targetTriangles: Int, creaseDegrees: Double = 40) -> Mesh3 {
        decimate(targetTriangles: targetTriangles, creaseDegrees: creaseDegrees).mesh
    }

    /// Decimate to `ratio` (0…1) of the triangles.
    func decimated(ratio: Double, creaseDegrees: Double = 40) -> Mesh3 {
        decimated(targetTriangles: Int((Double(triangleCount) * Swift.max(0, Swift.min(1, ratio))).rounded()),
                  creaseDegrees: creaseDegrees)
    }

    /// Decimate and report the worst error paid.
    func decimate(targetTriangles: Int, creaseDegrees: Double = 40) -> (mesh: Mesh3, report: DecimationReport) {
        var d = Decimator(TriTopology(self, creaseDegrees: creaseDegrees), creaseDegrees: creaseDegrees)
        let report = d.run(target: Swift.max(4, targetTriangles))
        return (d.t.toMesh3(creaseDegrees: creaseDegrees), report)
    }
}

/// Symmetric 4×4 error quadric of planes: error(p) = Σ w·(n·p + d)².
struct Quadric {
    var m = [Double](repeating: 0, count: 10)   // a b c d e f g h i j of [a b c d; b e f g; c f h i; d g i j]
    init() {}
    init(normal n: Vec3, offset d: Double, weight w: Double) {
        m = [n.x * n.x, n.x * n.y, n.x * n.z, n.x * d, n.y * n.y, n.y * n.z, n.y * d, n.z * n.z, n.z * d, d * d].map { $0 * w }
    }
    static func + (l: Quadric, r: Quadric) -> Quadric { var o = l; for i in 0 ..< 10 { o.m[i] += r.m[i] }; return o }
    static func += (l: inout Quadric, r: Quadric) { for i in 0 ..< 10 { l.m[i] += r.m[i] } }
    func error(_ p: Vec3) -> Double {
        let x = p.x, y = p.y, z = p.z
        let v = m[0] * x * x + 2 * m[1] * x * y + 2 * m[2] * x * z + 2 * m[3] * x
            + m[4] * y * y + 2 * m[5] * y * z + 2 * m[6] * y + m[7] * z * z + 2 * m[8] * z + m[9]
        return Swift.max(0, v)
    }
    /// The point minimising the error, nil when the system is (near) singular.
    func optimum() -> Vec3? {
        let a = m[0], b = m[1], c = m[2], e = m[4], f = m[5], h = m[7]
        let det = a * (e * h - f * f) - b * (b * h - f * c) + c * (b * f - e * c)
        let scale = Swift.max(abs(a), abs(e), abs(h), 1e-30)
        guard abs(det) > 1e-9 * scale * scale * scale else { return nil }
        let r = Vec3(-m[3], -m[6], -m[8])
        let x = (r.x * (e * h - f * f) - b * (r.y * h - f * r.z) + c * (r.y * f - e * r.z)) / det
        let y = (a * (r.y * h - f * r.z) - r.x * (b * h - f * c) + c * (b * r.z - r.y * c)) / det
        let z = (a * (e * r.z - r.y * f) - b * (b * r.z - r.y * c) + r.x * (b * f - e * c)) / det
        return Vec3(x, y, z)
    }
}

struct Decimator {
    var t: TriTopology
    var q: [Quadric]
    var ver: [Int]
    var heap: [Entry] = []
    let cosTurn: Double

    struct Entry {
        var cost: Double
        var keep: Int, remove: Int
        var p: Vec3
        var lerpT: Double?
        var stamp: Int
    }

    init(_ topo: TriTopology, creaseDegrees: Double) {
        t = topo
        cosTurn = cos(creaseDegrees * .pi / 180)
        q = Array(repeating: Quadric(), count: topo.pos.count)
        ver = Array(repeating: 0, count: topo.pos.count)
        for f in 0 ..< t.tri.count where t.alive[f] {
            let n = t.rawNormal(f), area2 = len3(n)
            guard area2 > 1e-18 else { continue }
            let u = n / area2, tr = t.tri[f]
            let qf = Quadric(normal: u, offset: -dot3(u, t.pos[tr.x]), weight: area2 * 0.5)
            q[tr.x] += qf; q[tr.y] += qf; q[tr.z] += qf
        }
        // Constraint planes along boundary / crease edges: perpendicular to the adjoining face, through the edge.
        for f in 0 ..< t.tri.count where t.alive[f] {
            for k in 0 ..< 3 {
                let a = t.tri[f][k], b = t.tri[f][(k + 1) % 3]
                let sh = t.facesOn(a, b)
                guard sh.count == 1 || t.creases.contains(TriTopology.key(a, b)) || sh.count > 2 else { continue }
                let e = t.pos[b] - t.pos[a], len = len3(e)
                let fn = normalize3(t.rawNormal(f))
                let m = cross3(e, fn)
                guard len > 1e-12, len3(m) > 1e-12 else { continue }
                let qc = Quadric(normal: normalize3(m), offset: -dot3(normalize3(m), t.pos[a]), weight: len * len)
                q[a] += qc; q[b] += qc
            }
        }
    }

    // MARK: heap

    mutating func push(_ e: Entry) {
        heap.append(e)
        var i = heap.count - 1
        while i > 0 {
            let p = (i - 1) / 2
            if heap[p].cost <= heap[i].cost { break }
            heap.swapAt(p, i); i = p
        }
    }
    mutating func pop() -> Entry? {
        guard let top = heap.first else { return nil }
        let last = heap.removeLast()
        if !heap.isEmpty {
            heap[0] = last
            var i = 0
            while true {
                let l = 2 * i + 1, r = l + 1
                var s = i
                if l < heap.count, heap[l].cost < heap[s].cost { s = l }
                if r < heap.count, heap[r].cost < heap[s].cost { s = r }
                if s == i { break }
                heap.swapAt(s, i); i = s
            }
        }
        return top
    }

    // MARK: edges

    /// The cheapest legal collapse of edge (u, v), or nil.
    func best(_ u: Int, _ v: Int) -> Entry? {
        let rule = t.collapseRule(u, v, cosTurn: cosTurn)
        guard rule.removeU || rule.removeV else { return nil }
        let qe = q[u] + q[v]
        var cands: [(keep: Int, remove: Int, p: Vec3, lerp: Double?)] = []
        if rule.free {
            let a = t.pos[u], b = t.pos[v], ab = b - a
            func param(_ p: Vec3) -> Double { Swift.max(0, Swift.min(1, dot3(p - a, ab) / Swift.max(dot3(ab, ab), 1e-30))) }
            var ps = [(a + b) * 0.5, a, b]
            if let o = qe.optimum(), len3(o - (a + b) * 0.5) < 2 * len3(ab) { ps.append(o) }
            for p in ps { cands.append((v, u, p, param(p))) }
        } else {
            if rule.removeU { cands.append((v, u, t.pos[v], nil)) }
            if rule.removeV { cands.append((u, v, t.pos[u], nil)) }
        }
        let scored = cands.map { ($0, qe.error($0.p)) }.sorted { $0.1 < $1.1 }
        for (c, cost) in scored where t.collapseIsValid(keep: c.keep, remove: c.remove, to: c.p) {
            return Entry(cost: cost, keep: c.keep, remove: c.remove, p: c.p, lerpT: c.lerp, stamp: ver[u] + ver[v])
        }
        return nil
    }

    mutating func enqueue(_ u: Int, _ v: Int) { if let e = best(u, v) { push(e) } }

    mutating func run(target: Int) -> Mesh3.DecimationReport {
        var report = Mesh3.DecimationReport()
        for v in 0 ..< t.pos.count { for w in t.neighbors(v) where v < w { enqueue(v, w) } }
        var maxCost = 0.0
        while t.aliveCount > target, let e = pop() {
            guard !t.vf[e.keep].isEmpty, !t.vf[e.remove].isEmpty,
                  e.stamp == ver[e.keep] + ver[e.remove],
                  t.collapseIsValid(keep: e.keep, remove: e.remove, to: e.p) else { continue }
            let qm = q[e.keep] + q[e.remove]
            t.collapse(keep: e.keep, remove: e.remove, to: e.p, lerpT: e.lerpT)
            q[e.keep] = qm
            maxCost = Swift.max(maxCost, e.cost)
            report.collapses += 1
            ver[e.keep] += 1; ver[e.remove] += 1
            let ring = t.neighbors(e.keep)
            for w in ring { ver[w] += 1 }
            for w in ring {
                enqueue(e.keep, w)
                for x in t.neighbors(w) where x != e.keep && !(ring.contains(x) && x < w) { enqueue(w, x) }
            }
        }
        report.triangles = t.aliveCount
        report.maxError = maxCost.squareRoot()
        return report
    }
}
