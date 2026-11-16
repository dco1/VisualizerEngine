//  Constrained triangulation of one cut triangle, for the mesh-intersection boolean.
//
//  Input is a triangle's boundary ring (corners plus every vertex that sits on an edge, so neighbours that were
//  cut agree), interior points, and constraint segments (the intersection curve). Output is a triangulation of
//  the triangle that contains every constraint as a chain of edges and uses only the given vertices plus one
//  private centre vertex (the start of the triangulation is a star round the centroid).
//
//  Method: star from the centroid → Delaunay insertion of the interior points (exact `orient2d` locates) →
//  constraint recovery by edge flips (Sloan), a constraint through a vertex split at it → Delaunay flips on the
//  unconstrained edges for shape. No Steiner points on constraints or the boundary: the cut curve and the
//  neighbours' edges are exactly the pooled vertices, which is what lets both meshes share the curve.
//
//  Guarantees: output triangles are CCW and tile the triangle exactly. A constraint that cannot be recovered
//  (flip budget exhausted: pathological near-collinear input) is skipped and counted in `failures`.

import Foundation

struct CDT2 {
    var p: [Vec2]
    private(set) var tv: [SIMD3<Int32>] = []
    private var tn: [SIMD3<Int32>] = []
    private var vtri: [Int32] = []
    private(set) var constrained = Set<UInt64>()
    private(set) var failures = 0
    private var last: Int32 = 0
    private var tol = 0.0
    private var lineTol = 0.0

    static func key(_ a: Int, _ b: Int) -> UInt64 { UInt64(min(a, b)) << 32 | UInt64(max(a, b)) }

    /// `ring`: CCW polygon points (the triangle's boundary with its edge vertices). The centroid is vertex
    /// `ring.count` and is private to this triangulation.
    init(ring: [Vec2]) {
        p = ring
        let m = ring.count
        var c = Vec2.zero
        for q in ring { c += q }
        p.append(c / Double(m))
        vtri = Array(repeating: 0, count: m + 1)
        for i in 0..<m {
            tv.append(SIMD3(Int32(i), Int32((i + 1) % m), Int32(m)))
            tn.append(SIMD3(Int32((i + 1) % m), Int32((i + m - 1) % m), -1))
            vtri[i] = Int32(i)
        }
        vtri[m] = 0
        var ext = 0.0
        for q in ring { ext = max(ext, abs(q.x - c.x), abs(q.y - c.y)) }
        tol = ext * ext * ext * ext * 1e-13
        lineTol = ext * 1e-9
        legalizeAll()
    }

    // MARK: topology helpers

    private func slot(_ t: Int32, of v: Int32) -> Int {
        let x = tv[Int(t)]
        return x.x == v ? 0 : (x.y == v ? 1 : 2)
    }
    private mutating func replaceNeighbour(_ n: Int32, _ old: Int32, _ new: Int32) {
        guard n >= 0 else { return }
        let x = tn[Int(n)]
        if x.x == old { tn[Int(n)].x = new } else if x.y == old { tn[Int(n)].y = new } else { tn[Int(n)].z = new }
    }
    private func inCircle(_ a: Int32, _ b: Int32, _ c: Int32, _ d: Int32) -> Double {
        let A = p[Int(a)] - p[Int(d)], B = p[Int(b)] - p[Int(d)], C = p[Int(c)] - p[Int(d)]
        let al = A.x * A.x + A.y * A.y, bl = B.x * B.x + B.y * B.y, cl = C.x * C.x + C.y * C.y
        return A.x * (B.y * cl - bl * C.y) - A.y * (B.x * cl - bl * C.x) + al * (B.x * C.y - B.y * C.x)
    }

    /// Triangles around `u`, as (triangle, slot of u in it).
    private func fan(_ u: Int32) -> [(Int32, Int)] {
        var out: [(Int32, Int)] = []
        let start = vtri[Int(u)]
        var t = start
        var guardN = 0
        repeat {
            let i = slot(t, of: u)
            out.append((t, i))
            t = tn[Int(t)][(i + 1) % 3]            // across edge (q,u): next counter-clockwise
            guardN += 1
        } while t >= 0 && t != start && guardN < 100_000
        if t < 0 {                                   // hit the hull: walk the other way from start
            t = start
            while true {
                let i = slot(t, of: u)
                let nxt = tn[Int(t)][(i + 2) % 3]    // across edge (u,p)
                if nxt < 0 || nxt == start { break }
                t = nxt
                out.append((t, slot(t, of: u)))
                guardN += 1
                if guardN > 200_000 { break }
            }
        }
        return out
    }

    private func edgeExists(_ a: Int32, _ b: Int32) -> Bool {
        for (t, i) in fan(a) where tv[Int(t)][(i + 1) % 3] == b || tv[Int(t)][(i + 2) % 3] == b { return true }
        return false
    }

    // MARK: flips

    private mutating func flip(_ t: Int32, _ k: Int) {
        let s = tn[Int(t)][k]
        let a = tv[Int(t)][k], b = tv[Int(t)][(k + 1) % 3], c = tv[Int(t)][(k + 2) % 3]
        let nCA = tn[Int(t)][(k + 1) % 3], nAB = tn[Int(t)][(k + 2) % 3]
        var j = 0
        for q in 0..<3 where tn[Int(s)][q] == t { j = q }
        let d = tv[Int(s)][j]
        let nBD = tn[Int(s)][(j + 1) % 3], nDC = tn[Int(s)][(j + 2) % 3]
        tv[Int(t)] = SIMD3(a, b, d); tn[Int(t)] = SIMD3(nBD, s, nAB)
        tv[Int(s)] = SIMD3(a, d, c); tn[Int(s)] = SIMD3(nDC, nCA, t)
        replaceNeighbour(nBD, s, t)
        replaceNeighbour(nCA, t, s)
        vtri[Int(a)] = t; vtri[Int(b)] = t; vtri[Int(d)] = t; vtri[Int(c)] = s
    }

    private mutating func legalize(_ start: [(Int32, Int)]) {
        var stack = start
        var budget = 200_000
        while let (t, k) = stack.popLast(), budget > 0 {
            budget -= 1
            let s = tn[Int(t)][k]
            guard s >= 0 else { continue }
            let a = tv[Int(t)][k], b = tv[Int(t)][(k + 1) % 3], c = tv[Int(t)][(k + 2) % 3]
            if constrained.contains(Self.key(Int(b), Int(c))) { continue }
            var j = 0
            for q in 0..<3 where tn[Int(s)][q] == t { j = q }
            let d = tv[Int(s)][j]
            if inCircle(a, b, c, d) > tol {
                flip(t, k)
                // after flip, t = (a,b,d), s = (a,d,c): their edges opposite a are the new outer edges
                stack.append((t, 0)); stack.append((s, 0))
            }
        }
    }

    private mutating func legalizeAll() {
        var work: [(Int32, Int)] = []
        for t in 0..<tv.count { for k in 0..<3 where tn[t][k] >= 0 && Int32(t) < tn[t][k] { work.append((Int32(t), k)) } }
        var rounds = 0
        while !work.isEmpty && rounds < 50 {
            rounds += 1
            legalize(work)
            work = []
            // one more sweep catches edges whose flip enabled another
            for t in 0..<tv.count { for k in 0..<3 {
                let s = tn[t][k]
                guard s >= 0, Int32(t) < s else { continue }
                let b = tv[t][(k + 1) % 3], c = tv[t][(k + 2) % 3]
                if constrained.contains(Self.key(Int(b), Int(c))) { continue }
                var j = 0
                for q in 0..<3 where tn[Int(s)][q] == Int32(t) { j = q }
                if inCircle(tv[t][k], b, c, tv[Int(s)][j]) > tol { work.append((Int32(t), k)) }
            } }
        }
    }

    // MARK: point insertion

    private enum Where { case inside(Int32), onEdge(Int32, Int), atVertex(Int32), outside }

    private func locate(_ q: Vec2) -> Where {
        var t = last
        var steps = 0
        while steps < tv.count * 3 + 16 {
            steps += 1
            var moved = false
            var zeros: [Int] = []
            for kk in 0..<3 {
                let k = (kk + steps) % 3
                let a = p[Int(tv[Int(t)][(k + 1) % 3])], b = p[Int(tv[Int(t)][(k + 2) % 3])]
                let o = Robust.orient2d(a, b, q)
                if o < 0 {
                    let n = tn[Int(t)][k]
                    if n < 0 { return .outside }
                    t = n; moved = true; break
                }
                if o == 0 { zeros.append(k) }
            }
            if moved { continue }
            if zeros.isEmpty { return .inside(t) }
            if zeros.count == 1 { return .onEdge(t, zeros[0]) }
            // on two edges' lines: the vertex they share
            let k = zeros[0], k2 = zeros[1]
            let v = tv[Int(t)][3 - k - k2]
            return .atVertex(v)
        }
        // fallback: exhaustive
        for t in 0..<tv.count {
            var neg = false, zeros: [Int] = []
            for k in 0..<3 {
                let o = Robust.orient2d(p[Int(tv[t][(k + 1) % 3])], p[Int(tv[t][(k + 2) % 3])], q)
                if o < 0 { neg = true; break }
                if o == 0 { zeros.append(k) }
            }
            if neg { continue }
            if zeros.isEmpty { return .inside(Int32(t)) }
            if zeros.count == 1 { return .onEdge(Int32(t), zeros[0]) }
            return .atVertex(tv[t][3 - zeros[0] - zeros[1]])
        }
        return .outside
    }

    /// Inserts a point; returns its vertex index, an existing vertex it coincides with, or nil when it falls
    /// outside the triangle.
    mutating func insert(_ q: Vec2) -> Int? {
        switch locate(q) {
        case .outside: return nil
        case .atVertex(let v): return Int(v)
        case .inside(let t):
            let v = Int32(p.count); p.append(q); vtri.append(t)
            let (a, b, c) = (tv[Int(t)].x, tv[Int(t)].y, tv[Int(t)].z)
            let nA = tn[Int(t)].x, nB = tn[Int(t)].y, nC = tn[Int(t)].z
            let t1 = Int32(tv.count), t2 = t1 + 1
            tv[Int(t)] = SIMD3(a, b, v); tn[Int(t)] = SIMD3(t1, t2, nC)
            tv.append(SIMD3(b, c, v)); tn.append(SIMD3(t2, t, nA))
            tv.append(SIMD3(c, a, v)); tn.append(SIMD3(t, t1, nB))
            replaceNeighbour(nA, t, t1); replaceNeighbour(nB, t, t2)
            vtri[Int(a)] = t; vtri[Int(b)] = t; vtri[Int(c)] = t1; vtri[Int(v)] = t
            last = t
            legalize([(t, 2), (t1, 2), (t2, 2)])
            return Int(v)
        case .onEdge(let t, let k):
            let v = Int32(p.count); p.append(q); vtri.append(t)
            let a = tv[Int(t)][k], b = tv[Int(t)][(k + 1) % 3], c = tv[Int(t)][(k + 2) % 3]
            let nCA = tn[Int(t)][(k + 1) % 3], nAB = tn[Int(t)][(k + 2) % 3]
            let s = tn[Int(t)][k]
            let tNew = Int32(tv.count)
            tv[Int(t)] = SIMD3(a, b, v)
            tv.append(SIMD3(a, v, c))
            tn.append(SIMD3(-1, nCA, t))
            tn[Int(t)] = SIMD3(-1, tNew, nAB)
            replaceNeighbour(nCA, t, tNew)
            vtri[Int(a)] = t; vtri[Int(b)] = t; vtri[Int(c)] = tNew; vtri[Int(v)] = t
            var work: [(Int32, Int)] = [(t, 2), (tNew, 1)]
            if s >= 0 {
                var j = 0
                for q in 0..<3 where tn[Int(s)][q] == t { j = q }
                let d = tv[Int(s)][j]
                let nBD = tn[Int(s)][(j + 1) % 3], nDC = tn[Int(s)][(j + 2) % 3]
                let sB = Int32(tv.count)
                tv[Int(s)] = SIMD3(d, c, v);  tn[Int(s)] = SIMD3(tNew, sB, nDC)
                tv.append(SIMD3(d, v, b));    tn.append(SIMD3(t, nBD, s))
                tn[Int(t)].x = sB; tn[Int(tNew)].x = s
                replaceNeighbour(nBD, s, sB)
                vtri[Int(d)] = s; vtri[Int(c)] = s; vtri[Int(b)] = sB
                work.append((s, 2)); work.append((sB, 1))
            }
            last = t
            legalize(work)
            return Int(v)
        }
    }

    // MARK: constraints

    /// Forces the segment `u`–`v` (vertex indices) into the triangulation. Returns the vertices actually
    /// chained (the segment may pass through other vertices, which split it).
    mutating func insertConstraint(_ u0: Int, _ v0: Int) {
        var u = Int32(u0)
        let v = Int32(v0)
        var rounds = 0
        while u != v && rounds < 10_000 {
            rounds += 1
            guard let w = recoverOne(u, v) else { failures += 1; return }
            constrained.insert(Self.key(Int(u), Int(w)))
            u = w
        }
    }

    private func side(_ a: Int32, _ b: Int32, _ c: Int32) -> Int { Robust.orient2d(p[Int(a)], p[Int(b)], p[Int(c)]) }

    /// Makes `u`→(first vertex on the way to `v`) an edge; returns that vertex.
    private mutating func recoverOne(_ u: Int32, _ v: Int32) -> Int32? {
        if edgeExists(u, v) { return v }
        // wedge around u containing the direction to v
        var cross: [(Int32, Int32)] = []
        var found = false
        var firstTri: Int32 = -1, firstSlot = 0
        for (t, i) in fan(u) {
            let pp = tv[Int(t)][(i + 1) % 3], qq = tv[Int(t)][(i + 2) % 3]
            let sp = side(u, pp, v), sq = side(u, qq, v)
            if onLine(pp, u, v) && onSegmentBefore(pp, u, v) { return pp }
            if onLine(qq, u, v) && onSegmentBefore(qq, u, v) { return qq }
            if sp > 0 && sq < 0 { firstTri = t; firstSlot = i; found = true; break }
        }
        guard found else { return nil }
        var t = firstTri
        var a = tv[Int(t)][(firstSlot + 1) % 3], b = tv[Int(t)][(firstSlot + 2) % 3]   // a right of u→v, b left
        var target = v
        var steps = 0
        while true {
            steps += 1
            if steps > 100_000 { return nil }
            cross.append((a, b))
            // neighbour across edge (a,b): the slot opposite the apex of t
            let apexSlot = (0..<3).first { tv[Int(t)][$0] != a && tv[Int(t)][$0] != b }!
            let n = tn[Int(t)][apexSlot]
            if n < 0 { return nil }
            var j = 0
            for q in 0..<3 where tn[Int(n)][q] == t { j = q }
            let w = tv[Int(n)][j]
            if w == v { break }
            let sw = side(u, v, w)
            if sw == 0 || onLine(w, u, v) { target = w; break }
            if constrained.contains(Self.key(Int(a), Int(b))) { return nil }
            if sw > 0 { b = w } else { a = w }
            t = n
        }
        // a crossing edge that is constrained blocks the segment
        for (x, y) in cross where constrained.contains(Self.key(Int(x), Int(y))) { return nil }

        var queue = cross
        var budget = cross.count * 40 + 200
        while !queue.isEmpty && budget > 0 {
            budget -= 1
            let (x, y) = queue.removeFirst()
            guard let (t0, k0) = findEdge(x, y) else { continue }
            let s0 = tn[Int(t0)][k0]
            guard s0 >= 0 else { return nil }
            var j = 0
            for q in 0..<3 where tn[Int(s0)][q] == t0 { j = q }
            let c = tv[Int(t0)][k0], d = tv[Int(s0)][j]
            // convex quad (x, c, y, d) iff c and d are on opposite sides of x–y (true) and x, y either side of c–d
            if side(c, d, x) * side(c, d, y) < 0 {
                flip(t0, k0)
                let straddles = side(u, target, c) * side(u, target, d) < 0
                if straddles { queue.append((c, d)) }
            } else {
                queue.append((x, y))
            }
        }
        if !queue.isEmpty { return nil }
        return target
    }

    /// `w` within `lineTol` of the line through `u`,`v` (a split point is only collinear to rounding).
    private func onLine(_ w: Int32, _ u: Int32, _ v: Int32) -> Bool {
        let d = p[Int(v)] - p[Int(u)], e = p[Int(w)] - p[Int(u)]
        let l = (d.x * d.x + d.y * d.y).squareRoot()
        return l > 0 && abs(d.x * e.y - d.y * e.x) <= lineTol * l
    }

    private func onSegmentBefore(_ w: Int32, _ u: Int32, _ v: Int32) -> Bool {
        let d = p[Int(v)] - p[Int(u)], e = p[Int(w)] - p[Int(u)]
        let dd = d.x * d.x + d.y * d.y
        let t = (d.x * e.x + d.y * e.y) / dd
        return t > 0 && t < 1
    }

    private func findEdge(_ x: Int32, _ y: Int32) -> (Int32, Int)? {
        for (t, i) in fan(x) {
            if tv[Int(t)][(i + 1) % 3] == y { return (t, (i + 2) % 3) }      // edge (x,y): opposite slot
            if tv[Int(t)][(i + 2) % 3] == y { return (t, (i + 1) % 3) }
        }
        return nil
    }

    /// Delaunay-improve everything that isn't a constraint.
    mutating func finish() { legalizeAll() }
}
