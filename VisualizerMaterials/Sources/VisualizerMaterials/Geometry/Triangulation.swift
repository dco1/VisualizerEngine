import simd

/// Ear-clipping triangulation of a **simple** polygon (no self-intersection, no
/// holes). Used to cap extruded wall solids and to floor a detected room face,
/// which can be concave (an L- or U-shaped room). O(n²) — fine at house scale
/// (rooms are tens of vertices, not thousands).
public enum Triangulation {

    /// Triangulate `poly` (vertices in order, no repeated closing point). Returns
    /// index triples into `poly`, each wound **CCW** in plan (matching the input's
    /// CCW orientation; a CW input is handled and still yields CCW triples).
    /// Returns `[]` for degenerate input (< 3 vertices or zero area).
    public static func earClip(_ poly: [Vec2]) -> [(Int, Int, Int)] {
        let n = poly.count
        guard n >= 3 else { return [] }
        guard abs(signedArea(poly)) > 1e-12 else { return [] }

        // Work on a doubly-implied ring of original indices, oriented CCW so the
        // convex/ear tests below have a single sign convention.
        var idx = Array(0..<n)
        if signedArea(poly) < 0 { idx.reverse() }

        var tris: [(Int, Int, Int)] = []
        tris.reserveCapacity(n - 2)
        var guardCount = 0
        let guardLimit = n * n + 8           // hard stop against a pathological loop

        while idx.count > 3 {
            guardCount += 1
            if guardCount > guardLimit { break }
            var clipped = false
            let m = idx.count
            for i in 0..<m {
                let i0 = idx[(i + m - 1) % m]
                let i1 = idx[i]
                let i2 = idx[(i + 1) % m]
                if isEar(poly, ring: idx, prev: i0, cur: i1, next: i2) {
                    tris.append((i0, i1, i2))
                    idx.remove(at: i)
                    clipped = true
                    break
                }
            }
            // No ear found (numerically tricky polygon) → clip the first corner to
            // make progress rather than hang. Keeps output a valid fan-ish cover.
            if !clipped {
                let i0 = idx[m - 1], i1 = idx[0], i2 = idx[1]
                tris.append((i0, i1, i2))
                idx.remove(at: 0)
            }
        }
        if idx.count == 3 { tris.append((idx[0], idx[1], idx[2])) }
        return tris
    }

    /// Triangulate a polygon **with holes**, returning point triples. Each hole is
    /// bridged into the outer ring (Eberly's "Triangulation by Ear Clipping"): the
    /// hole's rightmost vertex is joined to a visible outer vertex by a zero-width
    /// two-way bridge, turning outer-with-holes into one weakly-simple polygon that
    /// ear-clips normally. Outer is taken CCW, holes CW. Used for floor plates with a
    /// stairwell, and for rooms with a courtyard/column (PROJECT_SCOPE §6.2).
    public static func triangulate(_ outer: [Vec2], holes: [[Vec2]]) -> [(Vec2, Vec2, Vec2)] {
        guard outer.count >= 3 else { return [] }
        let liveHoles = holes.filter { $0.count >= 3 && abs(signedArea($0)) > 1e-12 }
        if liveHoles.isEmpty {
            return earClip(outer).map { (outer[$0.0], outer[$0.1], outer[$0.2]) }
        }
        var poly = signedArea(outer) < 0 ? Array(outer.reversed()) : outer       // CCW
        // Bridge rightmost-first so an earlier bridge can't block a later hole.
        let cw = liveHoles
            .map { signedArea($0) > 0 ? Array($0.reversed()) : $0 }               // CW
            .sorted { ($0.max { $0.x < $1.x }!.x) > ($1.max { $0.x < $1.x }!.x) }
        for hole in cw { poly = bridge(outer: poly, hole: hole) }
        return earClip(poly).map { (poly[$0.0], poly[$0.1], poly[$0.2]) }
    }

    // MARK: subtraction (cutouts that may cross the ring)

    /// Triangulate `outer` with the **convex** regions `cutouts` removed.
    ///
    /// This exists because `triangulate(_:holes:)` cannot express a cutout that crosses
    /// `outer`'s boundary: it bridges each hole into the ring, and `bridge` bails out
    /// (`guard bestEdge >= 0`) — silently keeping the polygon SOLID — whenever the
    /// hole's rightmost vertex casts no ray onto an outer edge. A stairwell is exactly
    /// that case: a real flight runs along/through a wall, so its footprint spans two
    /// or more detected rooms of the storey above and lies inside none of them.
    ///
    /// Here the cutout is subtracted from the **triangle soup** instead: each triangle
    /// that meets the cutout is split into convex cells and the covered ones dropped, so
    /// any overlap — interior, straddling, or entirely outside — removes exactly the area
    /// it covers and nothing else.
    ///
    /// `cutouts` must be convex and simple (a stair footprint is an `OBB2` rectangle).
    public static func subtract(_ outer: [Vec2], cutouts: [[Vec2]]) -> [(Vec2, Vec2, Vec2)] {
        guard outer.count >= 3 else { return [] }
        var tris = earClip(outer).map { (outer[$0.0], outer[$0.1], outer[$0.2]) }
        let live = cutouts
            .filter { $0.count >= 3 && abs(signedArea($0)) > 1e-12 }
            .map { signedArea($0) < 0 ? Array($0.reversed()) : $0 }        // CCW
        guard !live.isEmpty else { return tris }

        for cut in live {
            let box = bounds(cut)
            var kept: [(Vec2, Vec2, Vec2)] = []
            kept.reserveCapacity(tris.count)
            for t in tris {
                let tb = bounds([t.0, t.1, t.2])
                // Cheap reject: a triangle clear of the cutout's bounding box survives whole.
                if tb.maxX <= box.minX || tb.minX >= box.maxX
                    || tb.maxY <= box.minY || tb.minY >= box.maxY {
                    kept.append(t); continue
                }
                kept += subtractConvex(triangle: t, cutout: cut)
            }
            tris = kept
        }
        return tris
    }

    /// The outline of a plan-space triangle soup: the directed edges used by exactly one
    /// triangle, each traversed with the covered area on its LEFT (the soup's triangles
    /// are CCW). This is how a slab gets its side walls — reading the outline back off
    /// the cap keeps the wall subdivided exactly as the cap is, so cutting a stairwell
    /// can never leave a T-junction between them, and the wall follows a cutout that bit
    /// into the ring instead of a ring that no longer exists.
    public static func outline(of triangles: [(Vec2, Vec2, Vec2)]) -> [(Vec2, Vec2)] {
        struct Key: Hashable { let a: SIMD2<Int64>; let b: SIMD2<Int64> }
        func q(_ v: Vec2) -> SIMD2<Int64> {
            SIMD2<Int64>(Int64((v.x * 1e7).rounded()), Int64((v.y * 1e7).rounded()))
        }
        var edges: [Key: (Vec2, Vec2)] = [:]
        for t in triangles {
            var p = [t.0, t.1, t.2]
            if signedArea(p) < 0 { p.reverse() }
            for i in 0..<3 {
                let a = p[i], b = p[(i + 1) % 3]
                guard dist(a, b) > 1e-9 else { continue }
                let opposite = Key(a: q(b), b: q(a))
                if edges.removeValue(forKey: opposite) != nil { continue }   // interior edge — cancels
                edges[Key(a: q(a), b: q(b))] = (a, b)
            }
        }
        // Chain the surviving edges into loops, head to tail.
        //
        // This used to `return Array(edges.values)`, which was wrong twice over. A
        // dictionary's iteration order is unspecified, and it observably varies even
        // WITHIN one process — measured at 3 distinct orders over 200 000 identical calls
        // — so the slab's side faces came out permuted and re-exporting an unchanged
        // document was a spurious diff about one run in five. It was also geometrically
        // arbitrary, so the `u` that `floorSlab` accumulates along this sequence hopped
        // between opposite sides of the perimeter instead of running around it, which is
        // exactly what the side-wall UV is supposed to do.
        //
        // Walking each loop from its smallest vertex fixes both, and reproduces what the
        // dictionary happened to return in the common case, so slab edges keep the
        // texturing they have today.
        var outgoing: [SIMD2<Int64>: [Key]] = [:]
        for k in edges.keys { outgoing[k.a, default: []].append(k) }
        // A well-formed outline gives each vertex one outgoing edge; sort anyway so a
        // degenerate soup (a pinch point) still resolves the same way every time.
        outgoing = outgoing.mapValues { $0.sorted { ($0.b.x, $0.b.y) < ($1.b.x, $1.b.y) } }

        var remaining = edges
        var chained: [(Vec2, Vec2)] = []
        chained.reserveCapacity(edges.count)
        while let start = remaining.keys.min(by: {
            ($0.a.x, $0.a.y, $0.b.x, $0.b.y) < ($1.a.x, $1.a.y, $1.b.x, $1.b.y)
        }) {
            var cur = start
            while let segment = remaining.removeValue(forKey: cur) {
                chained.append(segment)
                guard let next = outgoing[cur.b]?.first(where: { remaining[$0] != nil }) else { break }
                cur = next
            }
        }
        return chained
    }

    /// One triangle minus one CCW convex `cutout`.
    ///
    /// The triangle is split by **every** cutout edge line into the full arrangement of
    /// convex cells, then each cell is kept or dropped by which side of the cutout its
    /// centroid falls on. Splitting by all lines (rather than peeling off one "outside"
    /// piece per edge) is what keeps the result T-junction-free: two cells that meet
    /// along a line were cut from the same parent by the same clip, so they share the
    /// crossing vertex exactly. Peeling gives one long edge facing two shorter ones,
    /// which leaves a hairline crack the slab's side wall can't close.
    ///
    /// A triangle that only grazes the cutout's bounding box is returned whole — the
    /// cutout's edge LINES are infinite, so splitting unconditionally would subdivide
    /// (and re-outline) triangles nowhere near the opening.
    private static func subtractConvex(triangle t: (Vec2, Vec2, Vec2),
                                       cutout cut: [Vec2]) -> [(Vec2, Vec2, Vec2)] {
        var tri = [t.0, t.1, t.2]
        if signedArea(tri) < 0 { tri.reverse() }
        var overlap = tri
        for i in 0..<cut.count where overlap.count >= 3 {
            overlap = clip(overlap, cut[i], cut[(i + 1) % cut.count], keepLeft: true)
        }
        guard abs(signedArea(overlap)) > 1e-12 else { return [t] }   // no real overlap

        var cells = [tri]
        for i in 0..<cut.count {
            let a = cut[i], b = cut[(i + 1) % cut.count]
            var next: [[Vec2]] = []
            next.reserveCapacity(cells.count * 2)
            for c in cells {
                for side in [true, false] {
                    let part = clip(c, a, b, keepLeft: side)
                    if abs(signedArea(part)) > 1e-12 { next.append(part) }
                }
            }
            cells = next
            if cells.isEmpty { break }
        }
        var out: [(Vec2, Vec2, Vec2)] = []
        for cell in cells {
            let mid = centroid(cell)
            let removed = (0..<cut.count).allSatisfy { j in
                sideDistance(mid, cut[j], cut[(j + 1) % cut.count]) >= 0     // left of every CCW edge
            }
            if !removed { out += fan(cell) }
        }
        return out
    }

    /// Sutherland–Hodgman clip of a **convex** polygon by the half-plane of line a→b.
    /// `keepLeft` keeps the side the line's left normal points to (the interior of a CCW
    /// ring); `false` keeps the right side.
    private static func clip(_ poly: [Vec2], _ a: Vec2, _ b: Vec2, keepLeft: Bool) -> [Vec2] {
        guard poly.count >= 3 else { return [] }
        let sign = keepLeft ? 1.0 : -1.0
        func d(_ p: Vec2) -> Double { sideDistance(p, a, b) * sign }
        var out: [Vec2] = []
        out.reserveCapacity(poly.count + 2)
        for i in 0..<poly.count {
            let p = poly[i], q = poly[(i + 1) % poly.count]
            let dp = d(p), dq = d(q)
            if dp >= 0 { out.append(p) }
            if (dp > 0 && dq < 0) || (dp < 0 && dq > 0) {
                out.append(p + (q - p) * (dp / (dp - dq)))
            }
        }
        return out.count >= 3 ? out : []
    }

    /// Signed distance of `p` from the line a→b — positive on the line's LEFT.
    private static func sideDistance(_ p: Vec2, _ a: Vec2, _ b: Vec2) -> Double {
        let e = b - a
        let L = len(e)
        guard L > 1e-12 else { return 0 }
        return cross2(e, p - a) / L
    }

    /// Fan-triangulate a convex polygon.
    private static func fan(_ poly: [Vec2]) -> [(Vec2, Vec2, Vec2)] {
        guard poly.count >= 3 else { return [] }
        return (1..<(poly.count - 1)).map { (poly[0], poly[$0], poly[$0 + 1]) }
    }

    private static func bounds(_ pts: [Vec2]) -> (minX: Double, maxX: Double, minY: Double, maxY: Double) {
        var lo = pts[0], hi = pts[0]
        for p in pts { lo = Vec2(Swift.min(lo.x, p.x), Swift.min(lo.y, p.y))
                       hi = Vec2(Swift.max(hi.x, p.x), Swift.max(hi.y, p.y)) }
        return (lo.x, hi.x, lo.y, hi.y)
    }

    /// Splice one CW `hole` into the CCW `outer` ring via a visible-vertex bridge.
    private static func bridge(outer: [Vec2], hole: [Vec2]) -> [Vec2] {
        let mi = hole.indices.max { hole[$0].x < hole[$1].x }!     // M: rightmost hole vertex
        let M = hole[mi]

        // Cast a +x ray from M; find the nearest outer edge it crosses.
        var bestX = Double.greatestFiniteMagnitude, bestEdge = -1
        var I = M
        for i in outer.indices {
            let a = outer[i], b = outer[(i + 1) % outer.count]
            guard (a.y <= M.y && b.y >= M.y) || (b.y <= M.y && a.y >= M.y),
                  abs(b.y - a.y) > 1e-12 else { continue }
            let t = (M.y - a.y) / (b.y - a.y)
            let x = a.x + t * (b.x - a.x)
            if x >= M.x - 1e-9 && x < bestX { bestX = x; bestEdge = i; I = Vec2(x, M.y) }
        }
        guard bestEdge >= 0 else { return outer }                  // hole not inside — give up

        // Candidate P = the hit edge's vertex with larger x; refine to the reflex
        // vertex inside triangle (M, I, P) that's most "in line" with the ray.
        let a = outer[bestEdge], b = outer[(bestEdge + 1) % outer.count]
        var pIdx = a.x > b.x ? bestEdge : (bestEdge + 1) % outer.count
        var P = outer[pIdx]
        var bestAngle = abs(atan2(P.y - M.y, P.x - M.x))
        for i in outer.indices where outer[i] != P {
            let v = outer[i]
            guard pointInTriangle(v, M, I, P), isReflex(outer, i) else { continue }
            let ang = abs(atan2(v.y - M.y, v.x - M.x))
            if ang < bestAngle { bestAngle = ang; pIdx = i; P = v }
        }

        // P → (M, hole…, M) → P, then resume the outer ring.
        var out = Array(outer[0...pIdx])
        for k in 0..<hole.count { out.append(hole[(mi + k) % hole.count]) }
        out.append(M)
        out.append(P)
        if pIdx + 1 < outer.count { out.append(contentsOf: outer[(pIdx + 1)...]) }
        return out
    }

    private static func coincident(_ a: Vec2, _ b: Vec2) -> Bool { dist(a, b) < 1e-9 }

    /// A vertex is reflex in a CCW polygon when its turn is a right turn.
    private static func isReflex(_ poly: [Vec2], _ i: Int) -> Bool {
        let n = poly.count
        return orient(poly[(i + n - 1) % n], poly[i], poly[(i + 1) % n]) < 0
    }

    /// A corner is an ear iff it's convex (left turn for a CCW ring) and no other
    /// ring vertex lies inside the triangle (prev, cur, next).
    private static func isEar(_ poly: [Vec2], ring: [Int],
                              prev: Int, cur: Int, next: Int) -> Bool {
        let a = poly[prev], b = poly[cur], c = poly[next]
        if orient(a, b, c) <= 0 { return false }          // reflex or collinear
        for j in ring where j != prev && j != cur && j != next {
            let p = poly[j]
            // Hole-bridging duplicates a vertex (P, M appear twice); a duplicate that
            // coincides with an ear corner isn't really "inside" — skip it, else no
            // ear is ever found near a bridge.
            if coincident(p, a) || coincident(p, b) || coincident(p, c) { continue }
            if pointInTriangle(p, a, b, c) { return false }
        }
        return true
    }

    /// Barycentric inside test (inclusive of edges, so a vertex sitting exactly on
    /// the ear's edge still blocks the clip — the conservative choice).
    private static func pointInTriangle(_ p: Vec2, _ a: Vec2, _ b: Vec2, _ c: Vec2) -> Bool {
        let d1 = orient(a, b, p)
        let d2 = orient(b, c, p)
        let d3 = orient(c, a, p)
        let hasNeg = d1 < 0 || d2 < 0 || d3 < 0
        let hasPos = d1 > 0 || d2 > 0 || d3 > 0
        return !(hasNeg && hasPos)
    }
}
