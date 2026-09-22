//  Split from Mesh3.swift (DH-0523): lathe & sweep (revolve / sweep / loft).
//  Pure move — see Mesh3.swift for the type declaration and MeshAudit for the auditor.

import simd

extension Mesh3 {
    /// A 2D profile point for `revolve` / lathe geometry: radial distance `r` (≥ 0)
    /// from the +Y axis and height `y`. The profile polyline is read bottom-to-top
    /// (increasing `y` is the convention, but not required).
    public struct ProfilePoint: Equatable, Sendable {
        public var r: Double
        public var y: Double
        public init(r: Double, y: Double) { self.r = max(0, r); self.y = y }
    }

    /// Turn a SPARSE, hand-authored profile polyline into a smoothly-CURVED one by fitting a
    /// Catmull-Rom spline (averaged-slope cubic Hermite — the same construction
    /// `ForestTreeGeometry.hermite` uses for tree trunk/branch radius profiles, generalized here
    /// to both channels of a `ProfilePoint`) through the control points and resampling
    /// `perSegment` steps between each pair.
    ///
    /// **Why this exists:** `revolve`'s adaptive segment count (`segmentsFor`) only controls the
    /// AZIMUTHAL facet count (how round the silhouette looks from above) — it has no notion of
    /// meridional curvature, so a silhouette with only a handful of `(r, y)` waypoints stays
    /// exactly that many straight chords no matter how many longitude segments it gets. Two or
    /// three straight chords through a real bend (a vase's belly→neck→lip) still shade smoothly
    /// after `Mesh3.smoothed()` (normals blend continuously across them), but a glossy material's
    /// SPECULAR highlight visibly kinks at each chord's edge, because the geometry underneath is
    /// still just a few flat bands — smoothing fixes the shading gradient, not the missing
    /// curvature. Densifying the CONTROL POINTS with a real spline (not just adding more collinear
    /// points along the same straight lines, which would add zero new curvature) is what actually
    /// rounds the bend.
    ///
    /// Every original control point is preserved as an exact resample point (never smoothed away),
    /// so a genuinely sharp, deliberate corner authored in the profile survives — only the space
    /// BETWEEN points gains curvature. Passing `perSegment: 1` is a no-op (returns `points`
    /// unchanged), so a caller that wants to keep a segment perfectly straight (e.g. the rim's
    /// inward lip fold, a deliberate crease) can leave it alone.
    public static func smoothedProfile(_ points: [ProfilePoint], perSegment: Int = 4) -> [ProfilePoint] {
        guard points.count >= 3, perSegment > 1 else { return points }
        let n = points.count
        func hermite(_ p0: ProfilePoint, _ p1: ProfilePoint, _ p2: ProfilePoint, _ p3: ProfilePoint,
                    _ t: Double) -> ProfilePoint {
            let t2 = t * t, t3 = t2 * t
            func c(_ a: Double, _ b: Double, _ cc: Double, _ d: Double) -> Double {
                0.5 * ((2 * b) + (-a + cc) * t + (2 * a - 5 * b + 4 * cc - d) * t2
                       + (-a + 3 * b - 3 * cc + d) * t3)
            }
            return ProfilePoint(r: c(p0.r, p1.r, p2.r, p3.r), y: c(p0.y, p1.y, p2.y, p3.y))
        }
        var out: [ProfilePoint] = []
        out.reserveCapacity((n - 1) * perSegment + 1)
        for i in 0 ..< (n - 1) {
            let p0 = points[max(0, i - 1)]
            let p1 = points[i]
            let p2 = points[i + 1]
            let p3 = points[min(n - 1, i + 2)]
            for s in 0 ..< perSegment {
                out.append(hermite(p0, p1, p2, p3, Double(s) / Double(perSegment)))
            }
        }
        out.append(points[n - 1])
        return out
    }

    /// **Revolve** an arbitrary 2D profile polyline around the +Y axis into a closed
    /// solid of revolution — the generalisation of the lamp lathe (`LampMesh`) into a
    /// reusable `Mesh3` primitive (docs/PLACEABLE_FIDELITY.md §3). Each adjacent pair of
    /// profile points becomes a ring of side quads; the longitude segment count is chosen
    /// **once** for the whole revolve (sized to the profile's MAX radius via
    /// `segmentsFor(radius:)`) so every ring shares the same vertex columns — no
    /// T-junction seams between a wide ring and a narrow one (the documented footgun if
    /// each ring picked its own count). If an end of the profile does not reach the axis
    /// (`r > 0`) the corresponding cap is filled with a fan so the result is watertight
    /// and outward-wound (`signedVolume > 0`, `MeshAudit.isClosedManifold`). All output
    /// flows through `addTriangle(outward:)`, so winding stays declared-once and
    /// `MeshAudit` keeps policing it.
    ///
    /// - Parameters:
    ///   - profile: the lathe outline, `(r, y)` points; needs ≥ 2 points. Zero-radius
    ///     endpoints collapse to the axis (a pointed/closed tip — no cap fan there).
    ///   - segments: longitude count override; when `nil`, adaptive from the max radius.
    ///   - maxChordError: chord-error target (m) for the adaptive count. Default 1 mm.
    ///   - capBottom / capTop: fill the first/last ring with an axis fan when its radius
    ///     is > 0 (default true). Set false to leave an open aperture (e.g. a basin you
    ///     pour into) — then the mesh is an open shell, not a closed solid.
    @discardableResult
    public mutating func revolve(profile: [ProfilePoint],
                                 segments: Int? = nil,
                                 maxChordError: Double = 0.001,
                                 capBottom: Bool = true,
                                 capTop: Bool = true) -> Bool {
        guard profile.count >= 2 else { return false }
        let maxR = profile.map(\.r).max() ?? 0
        guard maxR > 1e-9 else { return false }            // a pure axis line has no surface
        // ONE segment count for the whole revolve, sized to the widest ring, so every
        // ring shares the same longitude columns (vertex-aligned → no T-junctions).
        let n = segments ?? Mesh3.segmentsFor(radius: maxR, maxChordError: maxChordError)
        guard n >= 3 else { return false }
        let twoPi = Double.pi * 2

        // Precompute the n ring positions for a given radius/height.
        func ring(_ r: Double, _ y: Double) -> [Vec3] {
            (0 ..< n).map { i in
                let a = twoPi * Double(i) / Double(n)
                return Vec3(r * cos(a), y, r * sin(a))
            }
        }

        // Side wall: one band of quads per adjacent profile segment. The outward reference is
        // the segment's TRUE in-profile outward normal (radial component +dY, axial component
        // −dR), revolved to each quad's mid-longitude — NOT a pure-radial vector. A radial-only
        // reference is perpendicular to the real normal wherever a segment folds inward and
        // near-horizontally (a pot/vase rim lip, a soil/water dome, a cone shoulder): it can't
        // say which way is "up", so `addTriangle` locked in the DOWN-facing winding and those
        // rings rendered inside-out (back-face culled → invisible under a single-sided material).
        // For a vertical wall (dR=0) this reduces to a positive multiple of the old radial vector,
        // so cylinders/posts/stems are byte-for-byte unchanged; only sloped bands are corrected.
        // Same idiom as `roundedTopSlab` (search "in-plane perpendicular pushed AWAY from the axis").
        for s in 0 ..< (profile.count - 1) {
            let lo = profile[s], hi = profile[s + 1]
            // Both ends on the axis → degenerate band, skip.
            if lo.r <= 1e-9 && hi.r <= 1e-9 { continue }
            let ringLo = ring(lo.r, lo.y)
            let ringHi = ring(hi.r, hi.y)
            let dR = hi.r - lo.r, dY = hi.y - lo.y
            for i in 0 ..< n {
                let j = (i + 1) % n
                let a = twoPi * (Double(i) + 0.5) / Double(n)
                let outward = Vec3(cos(a) * dY, -dR, sin(a) * dY)
                if lo.r <= 1e-9 {
                    // Bottom collapses to the axis → triangle fan to the apex point.
                    let apex = Vec3(0, lo.y, 0)
                    addTriangle(apex, ringHi[i], ringHi[j], outward: outward)
                } else if hi.r <= 1e-9 {
                    // Top collapses to the axis → triangle fan to the apex point.
                    let apex = Vec3(0, hi.y, 0)
                    addTriangle(ringLo[i], ringLo[j], apex, outward: outward)
                } else {
                    addQuad(ringLo[i], ringLo[j], ringHi[j], ringHi[i], outward: outward)
                }
            }
        }

        // End caps — only when the terminal ring has area and the caller wants closure.
        let first = profile.first!, last = profile.last!
        if capBottom, first.r > 1e-9 {
            let center = Vec3(0, first.y, 0)
            let r = ring(first.r, first.y)
            for i in 0 ..< n {
                addTriangle(center, r[(i + 1) % n], r[i], outward: Vec3(0, -1, 0))
            }
        }
        if capTop, last.r > 1e-9 {
            let center = Vec3(0, last.y, 0)
            let r = ring(last.r, last.y)
            for i in 0 ..< n {
                addTriangle(center, r[i], r[(i + 1) % n], outward: Vec3(0, 1, 0))
            }
        }
        return true
    }

    // ── Sweep: drag a cross-section along a 3D path (taper falls out) ──────────
    //
    // The second profile-kit primitive (docs/PLACEABLE_FIDELITY.md §3): drag a closed
    // 2D cross-section along a path of 3D points, optionally SCALING the section per
    // station so it shrinks toward one end — that scale IS a taper (a tapered chair
    // leg, a turned post, later a rolled sofa arm / handrail / molding). "The bevel
    // falls out for free" — a chamfer is just a profile with a clipped corner.
    //
    // A rounded-square section gives the vertical corners a curved-band dihedral, so
    // `PlaceableAudit` reads them as softened curved rings (not crude 90° arrises) and
    // `FacetingAudit` can measure their chord error — the whole point of swapping the
    // chair's extruded boxes for swept tapered legs.

    /// A closed 2D **cross-section** for `sweep`, given as a polyline of `(u, v)` points in
    /// the section's local plane (the sweep frame's two perpendicular axes). Read CCW so the
    /// outward normal of each emitted side band points away from the path. The loop closes
    /// implicitly (last point → first); do NOT repeat the first point.
    public struct SectionProfile: Equatable, Sendable {
        public var points: [Vec2]
        public init(points: [Vec2]) { self.points = points }

        /// A **rounded square** of half-extent `half` with corner radius `radius`, sampled at
        /// `cornerSegs` facets per quarter-corner (so each corner becomes a curved-band ring,
        /// not a hard 90° arris). `radius` is clamped to `< half`; `radius ≤ 0` yields a plain
        /// square. CCW order.
        public static func roundedSquare(half: Double, radius: Double, cornerSegs: Int = 4) -> SectionProfile {
            let r = Swift.min(Swift.max(0, radius), half * 0.999)
            guard r > 1e-6 else {
                // Plain square, CCW: (+,+) (-,+) (-,-) (+,-)
                return SectionProfile(points: [
                    Vec2( half,  half), Vec2(-half,  half),
                    Vec2(-half, -half), Vec2( half, -half),
                ])
            }
            let inner = half - r
            let segs = Swift.max(1, cornerSegs)
            // Four corner centres, each sweeping a 90° quarter-arc CCW. Start at +u +v.
            let corners: [(cu: Double, cv: Double, a0: Double)] = [
                ( inner,  inner, 0),               // +u +v, arc 0 → 90°
                (-inner,  inner, .pi / 2),         // -u +v, arc 90 → 180°
                (-inner, -inner, .pi),             // -u -v, arc 180 → 270°
                ( inner, -inner, 3 * .pi / 2),     // +u -v, arc 270 → 360°
            ]
            var pts: [Vec2] = []
            for c in corners {
                for s in 0 ... segs {
                    let ang = c.a0 + (.pi / 2) * Double(s) / Double(segs)
                    pts.append(Vec2(c.cu + r * cos(ang), c.cv + r * sin(ang)))
                }
            }
            return SectionProfile(points: pts)
        }

        /// A **rounded rectangle** — half-extent `halfU` on the section's first axis and `halfV`
        /// on its second, corners filleted by `radius`, `cornerSegs` facets per quarter.
        ///
        /// The kit's third section after `roundedSquare` and `circle`, and the one every SLAB
        /// wants: a chair seat, a shelf, a rail whose section is wider than it is thick. Built by
        /// the same corner-arc construction as `roundedSquare` (which is this with
        /// `halfU == halfV`), so the fillets read as curved-band rings to `PlaceableAudit` and
        /// their chord error is measurable by `FacetingAudit`. CCW.
        public static func roundedRect(halfU: Double, halfV: Double,
                                       radius: Double, cornerSegs: Int = 4) -> SectionProfile {
            let hu = Swift.max(1e-6, halfU), hv = Swift.max(1e-6, halfV)
            let r = Swift.min(Swift.max(0, radius), Swift.min(hu, hv) * 0.999)
            guard r > 1e-6 else {
                return SectionProfile(points: [
                    Vec2( hu,  hv), Vec2(-hu,  hv), Vec2(-hu, -hv), Vec2( hu, -hv),
                ])
            }
            let iu = hu - r, iv = hv - r
            let segs = Swift.max(1, cornerSegs)
            let corners: [(cu: Double, cv: Double, a0: Double)] = [
                ( iu,  iv, 0), (-iu,  iv, .pi / 2), (-iu, -iv, .pi), ( iu, -iv, 3 * .pi / 2),
            ]
            var pts: [Vec2] = []
            for c in corners {
                for k in 0 ... segs {
                    let ang = c.a0 + (.pi / 2) * Double(k) / Double(segs)
                    pts.append(Vec2(c.cu + r * cos(ang), c.cv + r * sin(ang)))
                }
            }
            return SectionProfile(points: pts)
        }

        /// **Fillet an arbitrary convex CCW polygon** — every corner replaced by a tangent
        /// circular arc of `radius`, `cornerSegs` facets per corner.
        ///
        /// `roundedRect`/`roundedSquare` are the axis-aligned special cases; this is the one to
        /// reach for when the plan outline is NOT a rectangle — a chair seat's trapezoid being
        /// the motivating case, since a seat is wider at the front than the back and its front
        /// corners are rounded on every real chair. Filleting by the angle bisector (rather than
        /// insetting on the axes) is what keeps the arc tangent to BOTH edges when they are not
        /// perpendicular, which is exactly the case a trapezoid presents.
        ///
        /// The tangent run-back at each corner is clamped to half of each adjacent edge, so a
        /// radius larger than a short edge can accommodate degrades to a smaller fillet instead
        /// of producing a self-intersecting outline.
        public static func rounded(polygon: [Vec2], radius: Double,
                                   cornerSegs: Int = 3) -> SectionProfile {
            let n = polygon.count
            guard n >= 3 else { return SectionProfile(points: polygon) }
            let r = Swift.max(0, radius)
            guard r > 1e-6 else { return SectionProfile(points: polygon) }
            let segs = Swift.max(1, cornerSegs)
            var out: [Vec2] = []
            for i in 0 ..< n {
                let v = polygon[i]
                let p = polygon[(i + n - 1) % n]
                let q = polygon[(i + 1) % n]
                let e1 = p - v, e2 = q - v
                let l1 = (e1.x * e1.x + e1.y * e1.y).squareRoot()
                let l2 = (e2.x * e2.x + e2.y * e2.y).squareRoot()
                guard l1 > 1e-9, l2 > 1e-9 else { out.append(v); continue }
                let d1 = Vec2(e1.x / l1, e1.y / l1)
                let d2 = Vec2(e2.x / l2, e2.y / l2)
                let cosT = Swift.min(Swift.max(d1.x * d2.x + d1.y * d2.y, -1), 1)
                let theta = acos(cosT)
                // A straight (180°) or folded-back (0°) joint has no fillet to build.
                guard theta > 1e-4, theta < .pi - 1e-4 else { out.append(v); continue }
                var t = r / tan(theta / 2)
                t = Swift.min(t, Swift.min(l1, l2) * 0.5)
                let rr = t * tan(theta / 2)
                let a = Vec2(v.x + d1.x * t, v.y + d1.y * t)     // arc start (toward p)
                let b = Vec2(v.x + d2.x * t, v.y + d2.y * t)     // arc end   (toward q)
                // Centre sits along the inward bisector at r / sin(θ/2).
                let bis = Vec2(d1.x + d2.x, d1.y + d2.y)
                let bl = (bis.x * bis.x + bis.y * bis.y).squareRoot()
                guard bl > 1e-9 else { out.append(v); continue }
                let c = Vec2(v.x + bis.x / bl * (rr / sin(theta / 2)),
                             v.y + bis.y / bl * (rr / sin(theta / 2)))
                var a0 = atan2(a.y - c.y, a.x - c.x)
                let a1 = atan2(b.y - c.y, b.x - c.x)
                // Walk the SHORT way round, which is the tangent arc.
                var sweep = a1 - a0
                while sweep >  .pi { sweep -= 2 * .pi }
                while sweep < -.pi { sweep += 2 * .pi }
                for k in 0 ... segs {
                    let ang = a0 + sweep * Double(k) / Double(segs)
                    out.append(Vec2(c.x + rr * cos(ang), c.y + rr * sin(ang)))
                }
                a0 = a1
            }
            return SectionProfile(points: out)
        }

        /// A **circle** of `radius`, sampled at an adaptive facet count (or `segments` if given).
        /// The third and commonest sweep section after the rounded square — pipe, rail, spindle,
        /// turned leg — so it lives in the kit rather than being re-derived per generator.
        ///
        /// Deliberately NOT `roundedSquare(half: r, radius: r)`: that degenerate form emits FOUR
        /// pairs of vertices separated by `2·(half − radius)`, which for a near-circle is microns
        /// apart. `MeshAudit` welds by position, so those pairs merge and the swept tube reads as a
        /// NON-MANIFOLD solid — a real sweep that fails the soundness gate for a purely notational
        /// reason. A true circle has no coincident points and audits clean. CCW.
        public static func circle(radius: Double, segments: Int? = nil) -> SectionProfile {
            let r = Swift.max(1e-6, radius)
            let n = Swift.max(3, segments ?? Mesh3.segmentsFor(radius: r, maxChordError: 0.0004))
            let twoPi = Double.pi * 2
            return SectionProfile(points: (0 ..< n).map { i in
                let a = twoPi * Double(i) / Double(n)
                return Vec2(r * cos(a), r * sin(a))
            })
        }
    }

    /// **Sweep** a closed cross-section `profile` along the 3D `path`, optionally scaling the
    /// section per station (`scales`, parallel to `path`) so it tapers. Returns a watertight,
    /// outward-wound closed solid (`signedVolume > 0`, `MeshAudit.isClosedManifold`) when
    /// `capStart`/`capEnd` are true. The cross-section keeps ONE vertex count along the whole
    /// sweep (so adjacent stations share vertex columns — no T-junctions), and every triangle
    /// flows through `addTriangle(outward:)`, so winding stays declared-once and `MeshAudit`
    /// keeps policing it.
    ///
    /// At each path point a local frame is built: the tangent runs point→point (averaged at
    /// interior joints), and the section's two in-plane axes are carried along the path by
    /// **parallel transport** (each station rotates the previous frame's axes by the minimal
    /// rotation that aligns the previous tangent to the new one), so a curved path twists as
    /// little as possible and a straight path keeps a fixed orientation. The section point
    /// `(u, v)` lands at `pathPoint + (u·scale)·axisU + (v·scale)·axisV`.
    ///
    /// - Parameters:
    ///   - profile: the closed cross-section (≥ 3 points, CCW in the `(u, v)` plane).
    ///   - path: the 3D centre-line (≥ 2 points). Consecutive duplicates are tolerated.
    ///   - scales: per-station uniform scale of the section (default all 1). Count must equal
    ///     `path.count` when provided; a value < 1 shrinks the section there (taper).
    ///   - up: a reference axis used only to seed the first frame's section plane (the axis
    ///     least parallel to the first tangent is chosen if `up` is too aligned). Default +Y.
    ///   - capStart / capEnd: fan-fill the first/last section into a flat cap (default true) so
    ///     the result is a closed solid. Set false for an open tube.
    @discardableResult
    public mutating func sweep(profile: SectionProfile,
                               along path: [Vec3],
                               scales: [Double]? = nil,
                               up: Vec3 = Vec3(0, 1, 0),
                               capStart: Bool = true,
                               capEnd: Bool = true) -> Bool {
        let loop = profile.points
        guard loop.count >= 3, path.count >= 2 else { return false }
        if let s = scales, s.count != path.count { return false }
        let scale = scales ?? Array(repeating: 1.0, count: path.count)

        // ── Tangents at each station: forward difference at the ends, averaged at interior
        //    joints (so a corner station faces the mean of its two legs). Skip zero-length
        //    legs when forming the tangent. ──
        func dir(_ from: Vec3, _ to: Vec3) -> Vec3? {
            let d = to - from
            return len3(d) > 1e-9 ? normalize3(d) : nil
        }
        var tangents = [Vec3](repeating: Vec3(0, 0, 1), count: path.count)
        // Seed each station's tangent.
        for i in 0 ..< path.count {
            let prev = i > 0 ? dir(path[i - 1], path[i]) : nil
            let next = i < path.count - 1 ? dir(path[i], path[i + 1]) : nil
            switch (prev, next) {
            case let (p?, n?): tangents[i] = normalize3(p + n)
            case let (p?, nil): tangents[i] = p
            case let (nil, n?): tangents[i] = n
            case (nil, nil): tangents[i] = Vec3(0, 0, 1)   // degenerate path; arbitrary
            }
        }

        // ── Parallel-transport the section frame along the path. Seed axisU from the
        //    reference `up` projected perpendicular to the first tangent; if `up` is nearly
        //    parallel to the tangent, fall back to the X axis. ──
        func perpendicularSeed(to t: Vec3) -> Vec3 {
            var ref = up
            if abs(dot3(normalize3(ref), t)) > 0.95 { ref = Vec3(1, 0, 0) }
            if abs(dot3(normalize3(ref), t)) > 0.95 { ref = Vec3(0, 0, 1) }
            let u = ref - t * dot3(ref, t)
            return normalize3(u)
        }
        /// Rotate `v` by the minimal rotation taking unit `from` → unit `to` (Rodrigues).
        func rotateMinimal(_ v: Vec3, from: Vec3, to: Vec3) -> Vec3 {
            let axis = cross3(from, to)
            let s = len3(axis)
            let c = dot3(from, to)
            guard s > 1e-9 else { return c > 0 ? v : v }   // parallel (or anti-, rare) → leave
            let k = axis / s
            let ang = atan2(s, c)
            // Rodrigues: v·cos + (k×v)·sin + k·(k·v)·(1−cos)
            return v * cos(ang) + cross3(k, v) * sin(ang) + k * (dot3(k, v) * (1 - cos(ang)))
        }

        var axisU = [Vec3](repeating: Vec3(1, 0, 0), count: path.count)
        var axisV = [Vec3](repeating: Vec3(0, 0, 1), count: path.count)
        axisU[0] = perpendicularSeed(to: tangents[0])
        axisV[0] = normalize3(cross3(tangents[0], axisU[0]))
        for i in 1 ..< path.count {
            let u = rotateMinimal(axisU[i - 1], from: tangents[i - 1], to: tangents[i])
            // Re-orthonormalise against the new tangent to kill drift.
            let uPerp = normalize3(u - tangents[i] * dot3(u, tangents[i]))
            axisU[i] = uPerp
            axisV[i] = normalize3(cross3(tangents[i], uPerp))
        }

        // ── Station rings: P points each (P = loop.count), one ring per path station. ──
        let P = loop.count
        func ring(_ i: Int) -> [Vec3] {
            let c = path[i], su = axisU[i], sv = axisV[i], k = scale[i]
            return loop.map { p in c + su * (p.x * k) + sv * (p.y * k) }
        }
        let rings = (0 ..< path.count).map { ring($0) }

        // ── Side bands: one quad strip per path segment. Outward normal = the section
        //    edge's mid-point normal pushed out of the path (computed via the local frame).
        //    Build it geometrically per quad through addTriangle(outward:) using the quad's
        //    own cross product, seeded by an outward reference (away from the path centre). ──
        for s in 0 ..< (path.count - 1) {
            let r0 = rings[s], r1 = rings[s + 1]
            let c0 = path[s], c1 = path[s + 1]
            for e in 0 ..< P {
                let f = (e + 1) % P
                // Reference "outward" = midpoint of the quad minus the path centre-line at
                // its mid-station (push the band away from the axis). addTriangle re-winds to
                // match, so a CCW section yields outward-facing side walls.
                let mid = (r0[e] + r0[f] + r1[e] + r1[f]) * 0.25
                let axis = (c0 + c1) * 0.5
                var outward = mid - axis
                if len3(outward) < 1e-9 { outward = axisU[s] }   // degenerate (point on axis)
                addQuad(r0[e], r0[f], r1[f], r1[e], outward: normalize3(outward))
            }
        }

        // ── End caps: fan the terminal ring to its centroid, facing along ∓tangent. ──
        if capStart {
            let r = rings[0]
            let centre = r.reduce(Vec3(0, 0, 0), +) / Double(P)
            let outward = -tangents[0]                       // start cap faces back down the path
            for e in 0 ..< P {
                addTriangle(centre, r[(e + 1) % P], r[e], outward: outward)
            }
        }
        if capEnd {
            let r = rings[path.count - 1]
            let centre = r.reduce(Vec3(0, 0, 0), +) / Double(P)
            let outward = tangents[path.count - 1]           // end cap faces forward
            for e in 0 ..< P {
                addTriangle(centre, r[e], r[(e + 1) % P], outward: outward)
            }
        }
        return true
    }

    /// Knit a **loft** through pre-built closed rings — the per-station-section sibling of
    /// `sweep`.
    ///
    /// `sweep` carries ONE section along a path, which is the right tool for a bar of constant
    /// profile. It cannot express a form whose *outline changes* along its run — and an
    /// upholstered arm is exactly that: its top line rises toward the back to meet the back
    /// rail, and its front end rolls closed instead of being chopped off by a flat cap. Lofting
    /// per-station rings is how that shape is stated directly.
    ///
    /// Every ring must carry the same point count, in the same order and orientation (rings are
    /// knitted index-to-index). Side bands are wound outward from the local centre-line and the
    /// terminal rings are fanned to their centroids as caps, so the result is a closed
    /// outward-wound solid — `MeshAudit.isSound`, `signedVolume > 0` — exactly as `sweep`'s is.
    ///
    /// Rings must stay **finite**: a ring collapsed to a single point makes every quad into it
    /// degenerate, `addTriangle` skips degenerate triangles, and the skipped triangles leave
    /// open edges that fail `MeshAudit`. Close a rolled end with a small finite ring and let the
    /// cap fan cover it, rather than collapsing it to a point.
    @discardableResult
    /// Ear-clipped cap indices for a ring that is **concave** in the plane ⊥ `axis`, or `nil`
    /// when the ring is convex (in which case the caller's centroid fan is correct, cheaper,
    /// and — for every existing caller — the triangle count they already budget for).
    public static func concaveCapTriangles(ring: [Vec3], axis: Vec3) -> [(Int, Int, Int)]? {
        guard ring.count >= 4, len3(axis) > 1e-12 else { return nil }
        let n = normalize3(axis)
        // Any basis in the plane ⊥ n.
        let seed = abs(n.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)
        let u = normalize3(cross3(seed, n))
        let v = cross3(n, u)
        let flat = ring.map { Vec2(dot3($0, u), dot3($0, v)) }
        // Convex? Every turn the same sign.
        var pos = false, neg = false
        for i in flat.indices {
            let a = flat[(i - 1 + flat.count) % flat.count], b = flat[i], c = flat[(i + 1) % flat.count]
            let cr = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            if cr > 1e-12 { pos = true } else if cr < -1e-12 { neg = true }
        }
        guard pos && neg else { return nil }                 // convex — fan is fine
        let tris = Triangulation.earClip(flat)
        return tris.isEmpty ? nil : tris
    }

    public mutating func loft(rings: [[Vec3]], capStart: Bool = true, capEnd: Bool = true) -> Bool {
        guard rings.count >= 2, let P = rings.first?.count, P >= 3,
              rings.allSatisfy({ $0.count == P }) else { return false }
        let centres = rings.map { $0.reduce(Vec3(0, 0, 0), +) / Double(P) }

        // Side bands: one quad strip per station gap.
        //
        // **Outward comes from the ring's WINDING about the loft axis, not from its centroid.**
        // This used to reference `mid − centreline`, which is only sound when the ring is
        // evenly sampled: `centres` is the mean of the ring's POINTS, so a ring that spends
        // most of its samples on one side drags that mean off the true centre, and the
        // sparsely-sampled edges come out INSIDE-OUT. Measured on a grand piano's rim outline
        // (~120 of 137 points on one curved side, a handful on the two straight ones):
        // back-face culling dropped its covered pixels from 180 375 to 123 163 — a third of
        // the silhouette rendering as holes you could see through. `MeshAudit` reported it
        // perfectly sound throughout, because topology and self-consistency were fine; it is
        // which SIDE is outside that was wrong (DH-0916).
        //
        // The winding is an AREA integral over the whole ring, so how the ring is sampled
        // cannot reach it, and it stays correct for a concave ring where a per-point radial
        // test does not.
        for s in 0 ..< (rings.count - 1) {
            let r0 = rings[s], r1 = rings[s + 1]
            let axis = (centres[s] + centres[s + 1]) * 0.5
            var axisDir = centres[s + 1] - centres[s]
            if len3(axisDir) < 1e-12 { axisDir = Vec3(0, 1, 0) }
            axisDir = normalize3(axisDir)

            // Signed area the ring sweeps about its own centre, projected on the axis.
            var swept = 0.0
            for e in 0 ..< P {
                let f = (e + 1) % P
                swept += dot3(cross3(r0[e] - centres[s], r0[f] - centres[s]), axisDir)
            }
            let sign: Double = swept < 0 ? 1 : -1

            for e in 0 ..< P {
                let f = (e + 1) % P
                // `outward` only has to land within 90° of the face's true normal — it declares
                // the WINDING; the normal is derived from it (see `addQuad`).
                var outward = cross3(axisDir, r0[f] - r0[e]) * sign
                if len3(outward) < 1e-12 || abs(swept) < 1e-15 {
                    // Degenerate ring, or an edge running along the axis: fall back to the
                    // radial reference, which is right for the evenly-sampled rings that
                    // reach this branch.
                    let mid = (r0[e] + r0[f] + r1[e] + r1[f]) * 0.25
                    outward = mid - axis
                    if len3(outward) < 1e-9 { outward = axisDir }
                }
                addQuad(r0[e], r0[f], r1[f], r1[e], outward: normalize3(outward))
            }
        }

        // End caps face along the terminal station gap's direction (∓ for start / end).
        //
        // A **convex** ring caps as a fan from its centroid — which is what every caller has
        // always got, so their triangle counts are unchanged. A **concave** ring cannot: a fan
        // from the centroid lays triangles outside the polygon and overlapping each other, and
        // the result is a solid only by topology. It renders with holes. Those ear-clip
        // instead (DH-0916); the piano's bentside is the case that found this.
        func addCap(_ r: [Vec3], centre: Vec3, outward: Vec3, reversed: Bool) {
            if let tris = Self.concaveCapTriangles(ring: r, axis: outward) {
                for (i, j, k) in tris {
                    if reversed { addTriangle(r[i], r[k], r[j], outward: outward) }
                    else { addTriangle(r[i], r[j], r[k], outward: outward) }
                }
            } else {
                for e in 0 ..< P {
                    let a = r[e], b = r[(e + 1) % P]
                    if reversed { addTriangle(centre, b, a, outward: outward) }
                    else { addTriangle(centre, a, b, outward: outward) }
                }
            }
        }
        if capStart {
            addCap(rings[0], centre: centres[0],
                   outward: normalize3(centres[0] - centres[1]), reversed: true)
        }
        if capEnd {
            let n = rings.count - 1
            addCap(rings[n], centre: centres[n],
                   outward: normalize3(centres[n] - centres[n - 1]), reversed: false)
        }
        return true
    }

    // ── Rounded-rect recessed basin well ──────────────────────────────────────
    //
    // The rounded-box analogue of `revolve` (docs/PLACEABLE_FIDELITY.md §5 Tier 2):
    // a basin you look DOWN into that is a *rounded box*, not a revolved bowl — a
    // kitchen sink, a vanity lavatory drop-in. Shared by `MillworkMesh.sink` and
    // `FurnitureMesh.vanity` so the proven recessed-basin form lives in ONE place
    // (single source of truth), not duplicated per generator.
}
