//  Sweep a 2D profile along a 3D path — handles, tubes, pipes, faucet necks, bent wire, rails, trim.
//
//  The profile is a closed polygon in (u, v); at every path point it is stood up in the plane
//  perpendicular to the path and carried along on a ROTATION-MINIMISING frame (Wang et al., "double
//  reflection"), so the section does not spin about the path the way a Frenet frame does at an inflection.
//  Per-station `scale` and `twist` vary the section along the length (a tapering handle, a twisted rail).
//
//  Winding is declared through `addQuad(outward:)` from the profile edge and the path tangent, so it is right
//  for ANY profile orientation (a clockwise profile is turned round first) and for concave profiles — it is
//  not guessed from a centre point. The ends are closed with a fan from the profile's centroid, which is
//  exact for a star-shaped profile (a circle, a rounded rectangle, a D) and wrong for a deep C.
//
//  ⚠️ A TWIST needs stations to match: each side of a square becomes a skew quad, and the triangulated skew quad
//  shaves a thin wedge — volume error ≈ (twist per station)². 4.5° per station loses ~2.7 %, 1° about 0.2 %.
//
//  `Curve3` gives the paths: a Catmull-Rom spline through points, an arc, a cubic Bézier.

import simd

public enum Curve3 {

    /// A smooth curve THROUGH `points` (Catmull-Rom, end points doubled), `samples` points per span.
    public static func catmullRom(_ p: [Vec3], samples: Int = 12) -> [Vec3] {
        guard p.count >= 2 else { return p }
        let n = Swift.max(1, samples)
        var q = p
        q.insert(p[0] * 2 - p[1], at: 0)
        q.append(p[p.count - 1] * 2 - p[p.count - 2])
        var out: [Vec3] = []
        for i in 1..<(q.count - 2) {
            let (p0, p1, p2, p3) = (q[i - 1], q[i], q[i + 1], q[i + 2])
            for s in 0..<n {
                let t = Double(s) / Double(n), t2 = t * t, t3 = t2 * t
                out.append(0.5 * ((2 * p1) + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                                  + (3 * p1 - p0 - 3 * p2 + p3) * t3))
            }
        }
        out.append(p[p.count - 1])
        return out
    }

    /// A circular arc about `centre` in the plane spanned by `from` (the start direction) and `to` (a direction
    /// the arc turns toward), `radius` m, sweeping `angle` radians.
    public static func arc(centre: Vec3, from: Vec3, toward: Vec3, radius: Double, angle: Double,
                           samples: Int = 24) -> [Vec3] {
        let u = normalize3(from)
        let w = normalize3(toward - u * dot3(toward, u))
        return (0...Swift.max(1, samples)).map { i in
            let a = angle * Double(i) / Double(Swift.max(1, samples))
            return centre + (u * cos(a) + w * sin(a)) * radius
        }
    }

    /// A cubic Bézier through control points `a b c d`.
    public static func bezier(_ a: Vec3, _ b: Vec3, _ c: Vec3, _ d: Vec3, samples: Int = 24) -> [Vec3] {
        (0...Swift.max(1, samples)).map { i in
            let t = Double(i) / Double(Swift.max(1, samples)), s = 1 - t
            return a * (s * s * s) + b * (3 * s * s * t) + c * (3 * s * t * t) + d * (t * t * t)
        }
    }

    public static func length(_ p: [Vec3]) -> Double {
        zip(p, p.dropFirst()).reduce(0) { $0 + len3($1.1 - $1.0) }
    }
}

public extension Mesh3 {

    /// A regular `segments`-gon of `radius`, counter-clockwise — the profile of a round tube.
    static func circleProfile(radius: Double, segments: Int = 16) -> [Vec2] {
        (0..<Swift.max(3, segments)).map { i in
            let a = 2 * Double.pi * Double(i) / Double(Swift.max(3, segments))
            return Vec2(radius * cos(a), radius * sin(a))
        }
    }

    /// Facets needed so a circle of `radius` is within `chordError` m of true (default 0.05 mm — invisible at any
    /// render distance this app shoots), rounded up to a multiple of 4 so a round hole lines up with the axes.
    static func circleSegments(radius: Double, chordError: Double = 0.00005) -> Int {
        let inner = Swift.max(-1.0, Swift.min(1.0, 1 - chordError / Swift.max(radius, 1e-9)))
        let n = Int((Double.pi / acos(inner)).rounded(.up))
        return Swift.max(12, (n + 3) / 4 * 4)
    }

    /// A round bar / drill / pin from `a` to `b`, SMOOTH-shaded with its end caps left creased, so a bore cut with it
    /// reads as a true cylinder (the booleans interpolate normals across their cuts). The facet count is chosen from
    /// `chordError`, not guessed.
    static func cylinder(from a: Vec3, to b: Vec3, radius: Double, chordError: Double = 0.00005) -> Mesh3 {
        sweep(profile: circleProfile(radius: radius, segments: circleSegments(radius: radius, chordError: chordError)),
              along: [a, b]).smoothed(creaseDegrees: 50)
    }

    /// A rounded rectangle of half-sizes `hw × hh`, corner radius `r`, `corner` facets per corner.
    static func roundedRectProfile(halfWidth hw: Double, halfHeight hh: Double, cornerRadius r: Double,
                                   corner: Int = 4) -> [Vec2] {
        let rr = Swift.min(r, Swift.min(hw, hh) * 0.999)
        var out: [Vec2] = []
        for (cx, cy, a0) in [(hw - rr, hh - rr, 0.0), (-(hw - rr), hh - rr, 90.0),
                             (-(hw - rr), -(hh - rr), 180.0), (hw - rr, -(hh - rr), 270.0)] {
            for k in 0...Swift.max(1, corner) {
                let a = (a0 + 90 * Double(k) / Double(Swift.max(1, corner))) * .pi / 180
                out.append(Vec2(cx + rr * cos(a), cy + rr * sin(a)))
            }
        }
        return out
    }

    /// Sweep `profile` along `path`. `scale[i]` multiplies the section at path point i (default 1) and
    /// `twist[i]` turns it about the path (radians, default 0); each, if given, has one entry per path point.
    static func sweep(profile: [Vec2], along path: [Vec3], scale: [Double]? = nil, twist: [Double]? = nil,
                      capStart: Bool = true, capEnd: Bool = true) -> Mesh3 {
        guard profile.count >= 3, path.count >= 2 else { return Mesh3() }
        let sections: [[Vec2]] = (0..<path.count).map { i in
            let k = scale?[i] ?? 1, th = twist?[i] ?? 0
            let c = cos(th), s = sin(th)
            return profile.map { p in Vec2((p.x * c - p.y * s) * k, (p.x * s + p.y * c) * k) }
        }
        return loft(sections: sections, along: path, capStart: capStart, capEnd: capEnd)
    }

    /// A LOFT of 2D sections along a 3D path: every path point has its OWN section (all the same point count). This is
    /// what a form whose outline changes along its length needs — a body whose plan pulls in as its top edge rounds over
    /// (sections = the plan offset inward along a fillet arc, so the rounding is exact), a handle flaring into a head.
    ///
    /// The sections are stood up in rotation-minimising frames (Wang et al.'s double reflection, so they do not spin
    /// about the path at an inflection) and the resulting 3D rings are stitched by the existing `loft(rings:)` — whose
    /// outward-from-winding rule and ear-clipped concave caps are the audited ones (DH-0916).
    static func loft(sections: [[Vec2]], along path: [Vec3], capStart: Bool = true, capEnd: Bool = true) -> Mesh3 {
        guard path.count >= 2, sections.count == path.count, let m = sections.first?.count, m >= 3,
              sections.allSatisfy({ $0.count == m }) else { return Mesh3() }
        let n = path.count
        var tan: [Vec3] = (0..<n).map { i in normalize3(path[Swift.min(n - 1, i + 1)] - path[Swift.max(0, i - 1)]) }
        for i in 0..<n where len3(tan[i]) < 1e-12 { tan[i] = i > 0 ? tan[i - 1] : Vec3(0, 0, 1) }
        let ref = abs(tan[0].y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)
        var r = normalize3(cross3(tan[0], ref))
        var frames: [(r: Vec3, s: Vec3)] = [(r, cross3(tan[0], r))]
        for i in 0..<(n - 1) {
            let v1 = path[i + 1] - path[i]
            let c1 = dot3(v1, v1)
            guard c1 > 1e-18 else { frames.append(frames[i]); continue }
            let rL = r - v1 * (2 / c1 * dot3(v1, r))
            let tL = tan[i] - v1 * (2 / c1 * dot3(v1, tan[i]))
            let v2 = tan[i + 1] - tL
            let c2 = dot3(v2, v2)
            r = c2 > 1e-18 ? rL - v2 * (2 / c2 * dot3(v2, rL)) : rL
            r = normalize3(r - tan[i + 1] * dot3(r, tan[i + 1]))
            frames.append((r, cross3(tan[i + 1], r)))
        }
        let rings: [[Vec3]] = (0..<n).map { i in
            sections[i].map { p in path[i] + frames[i].r * p.x + frames[i].s * p.y }
        }
        var mesh = Mesh3()
        _ = mesh.loft(rings: rings, capStart: capStart, capEnd: capEnd)
        return mesh
    }

    /// A round tube along `path`, `radius` m (optionally tapering via `radiusScale`), smooth-shaded.
    static func tube(along path: [Vec3], radius: Double, segments: Int = 16, radiusScale: [Double]? = nil) -> Mesh3 {
        sweep(profile: circleProfile(radius: radius, segments: segments), along: path, scale: radiusScale)
            .smoothed(creaseDegrees: 50)
    }
}
