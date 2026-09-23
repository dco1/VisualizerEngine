import Foundation

/// **A close-coupled floor toilet, as separable parts** — the one shared builder for every app
/// that needs a WC: Superbloom Hills' Toilets meadow today; Daydream Home's bathroom fixture is
/// MIGRATING to it (its `FurnitureMesh.toiletParts` still builds the older solid-pan original
/// until then — two sources of truth, pending).
///
/// Ported from Daydream Home's `FurnitureMesh.toiletParts` (DaydreamCore) — same ANSI/ASME
/// proportions, same revolved ovoid pan, seat ring, domed lid, corner-broken cistern and pill
/// flush — with two things that fixture never needed because a closed lid always hid them:
///
/// - **The pan is HOLLOW.** Daydream's lathe closed flat across the top at the rim (a porcelain
///   plug under the lid). Here the same outside runs over the same lip and down an inner wall to
///   the water line, and the standing water is its own part (its own material).
/// - **The seat and lid are separate parts on a real hinge line** (`Parts.hinge`, along +X at the
///   back of the seat stack), so a host can open them, or let the wind lift them.
///
/// Frame: +X width, +Y up (the foot 4 mm above y = 0), the front of the bowl toward −Z; the
/// cistern at +Z. Metres. Every part is closed or deliberately open-shelled, outward-wound
/// through `addTriangle(outward:)` / `revolve` / `loft`.
public enum ToiletGeometry {

    /// Real WC dimensions (ANSI/ASME A112.19.2 nominals). The two bowls the trade ships differ in
    /// PROJECTION, not width — so only `bowlLength` varies; the fixture's depth is derived.
    public struct Dimensions: Sendable, Equatable {
        /// Seat-bolt centres to the front of the rim: 0.419 m round front (16½″), 0.470 elongated.
        public var bowlLength: Double
        /// The fixture's width (the cistern is the widest part; the pan tucks inside it).
        public var width: Double = 0.40
        /// Top of the cistern.
        public var height: Double = 0.78
        /// Widest point of the bowl (the same for both variants).
        public var bowlWidth: Double = 0.362
        /// Nose width as a fraction of the widest station — what makes the plan a PEAR (ovoid).
        public var noseWidthFraction: Double = 0.62
        public var tankDepth: Double = 0.20
        /// Porcelain behind the seat-bolt line, under the cistern.
        public var rearSetback: Double = 0.06
        public init(bowlLength: Double) { self.bowlLength = bowlLength }
        public static let roundFront = Dimensions(bowlLength: 0.419)
        public static let elongated = Dimensions(bowlLength: 0.470)
        /// Back of the cistern to the front of the rim — derived, never typed.
        public var depth: Double { tankDepth + bowlLength + rearSetback }
    }

    public struct Parts: Sendable {
        /// Pedestal + hollow bowl (glazed porcelain).
        public var pan: Mesh3
        /// The standing water in the bowl: a flat pool facing up.
        public var water: Mesh3
        /// The seat ring and the domed lid, CLOSED, in the fixture frame.
        public var seat: Mesh3
        public var lid: Mesh3
        /// Trapway shelf + cistern + cistern lid (glazed porcelain).
        public var cistern: Mesh3
        /// The flush button (chrome).
        public var flush: Mesh3
        /// A point on the seat/lid hinge line (which runs along +X): the back of the seat stack,
        /// level with the seat's top face.
        public var hinge: Vec3
        /// The cistern's front face; its lid's overhanging front edge, underside and top — what
        /// an opened lid comes to rest against.
        public var tankFrontZ: Double
        public var tankLidFrontZ: Double
        public var tankLidBottomY: Double
        public var tankTopY: Double
    }

    /// Plan-corner radius of the cistern and shelf; the cistern lid's rolled top edge.
    public static let cornerRadius = 0.014
    public static let lidEdgeRadius = 0.010
    /// Crease angle for the smooth-normal pass (Daydream's furniture value).
    public static let smoothCrease = 40.0

    public static func build(_ dims: Dimensions = .roundFront) -> Parts {
        let w = dims.width, d = dims.depth, h = dims.height
        let hd = d / 2
        let lift = 0.004
        let tankD = dims.tankDepth

        // ── Pan: a revolved outline warped to an ovoid (pear) plan ─────────────────
        // Sized so the seat's overhang — the fixture's front-most geometry — lands on the
        // footprint edge.
        let seatOverhang = 1.07
        let bowlLength = max(0.20, d - tankD - dims.rearSetback)
        let bowlHalfD = bowlLength / 2 / seatOverhang
        let bowlHalfW = w * (dims.bowlWidth / dims.width) / 2 / seatOverhang
        let bowlCenterZ = -hd + bowlLength / 2
        let seatR = bowlHalfD
        let seatY = h * 0.52                               // the rim (~0.40 m)
        let lipY = seatY + h * 0.045                       // the rim's flat top
        let waterY = seatY - 0.095
        func ovoid(_ m: Mesh3, overhang: Double = 1) -> Mesh3 {
            m.ovoidPlan(halfWidth: bowlHalfW * overhang, halfDepth: bowlHalfD * overhang,
                        noseWidthFraction: dims.noseWidthFraction)
        }
        // Up the outside (flared foot, full pedestal waist, the bowl ballooning to the rim), over
        // the lip, and DOWN the inside to the water line — one sheet, so both walls face out of
        // the porcelain. Sampled at 0.5 mm chord error (Daydream's measured FacetingAudit figure).
        var pan = Mesh3()
        pan.revolve(profile: [
            .init(r: seatR * 0.62, y: lift), .init(r: seatR * 0.66, y: h * 0.06),
            .init(r: seatR * 0.54, y: h * 0.20), .init(r: seatR * 0.58, y: h * 0.30),
            .init(r: seatR * 0.84, y: seatY * 0.78), .init(r: seatR * 1.00, y: seatY),
            .init(r: seatR * 0.92, y: lipY), .init(r: seatR * 0.80, y: lipY),
            .init(r: seatR * 0.76, y: seatY - 0.012), .init(r: seatR * 0.64, y: seatY - 0.055),
            .init(r: seatR * 0.46, y: waterY),
        ], maxChordError: 0.0005, capTop: false)
        pan = ovoid(pan).translated(by: Vec3(0, 0, bowlCenterZ)).smoothed(creaseDegrees: smoothCrease)

        // The standing water, meeting the inner wall at the water line; faces UP. `ovoidPlan`
        // normalises by the source's OWN widest radius, so the pool is warped at its own fraction
        // of the pan (0.47 R, a hair past the wall's 0.46 R) — warped at 1 it came out as wide as
        // the rim and stuck through the walls as a flange.
        let poolR = 0.47
        var water = Mesh3()
        water.revolve(profile: [.init(r: seatR * poolR, y: waterY), .init(r: 0, y: waterY)],
                      capBottom: false, capTop: false)
        water = ovoid(water, overhang: poolR).translated(by: Vec3(0, 0, bowlCenterZ))
        if water.normals.reduce(0.0, { $0 + $1.y }) < 0 {
            water.normals = water.normals.map { Vec3(-$0.x, -$0.y, -$0.z) }
            for t in stride(from: 0, to: water.indices.count, by: 3) { water.indices.swapAt(t + 1, t + 2) }
        }

        // ── Seat ring + domed lid, closed on the rim, warped to the same ovoid one overhang up.
        let seatThk = 0.028
        let holeR = seatR * 0.52, outR = seatR * 1.07
        var seat = Mesh3()
        seat.revolve(profile: [
            .init(r: holeR, y: lipY), .init(r: outR * 0.97, y: lipY), .init(r: outR, y: lipY + seatThk * 0.5),
            .init(r: outR * 0.97, y: lipY + seatThk), .init(r: holeR, y: lipY + seatThk), .init(r: holeR, y: lipY),
        ], capBottom: false, capTop: false)
        seat = ovoid(seat, overhang: seatOverhang).translated(by: Vec3(0, 0, bowlCenterZ))
            .smoothed(creaseDegrees: smoothCrease)
        var lid = Mesh3()
        lid.revolve(profile: [
            .init(r: outR * 0.99, y: lipY + seatThk), .init(r: outR, y: lipY + seatThk + 0.010),
            .init(r: outR * 0.88, y: lipY + seatThk + 0.020), .init(r: 0, y: lipY + seatThk + 0.024),
        ])
        lid = ovoid(lid, overhang: seatOverhang).translated(by: Vec3(0, 0, bowlCenterZ))
            .smoothed(creaseDegrees: smoothCrease)

        // ── Cistern: landscape, sitting ON the pan's rear shelf; the trapway shelf carries it
        // down to the floor so it never hangs in mid-air.
        let thw = w * 0.98 / 2
        let tankBaseY = seatY + lift
        let tankTopY = h + lift
        var cistern = Mesh3()
        // Its front stops inside the rear porcelain wall (between the rim's inner lip at 0.80 R and
        // the outside at 1.0 R) — from the bowl's centre, as in the solid pan, it stood up inside
        // the hollow bowl above the water.
        let shelfFrontZ = bowlCenterZ + bowlHalfD * 0.82
        let shelfHalfD = (hd - shelfFrontZ) / 2
        cistern.append(roundedPrism(halfW: bowlHalfW * 0.45, halfD: shelfHalfD, y0: lift, y1: tankBaseY,
                                    corner: cornerRadius)
                        .translated(by: Vec3(0, 0, shelfFrontZ + shelfHalfD)))
        cistern.append(roundedPrism(halfW: thw, halfD: tankD / 2, y0: tankBaseY, y1: tankTopY,
                                    corner: cornerRadius)
                        .translated(by: Vec3(0, 0, hd - tankD / 2)))
        // Its lid: overhanging on all four edges, with a rolled top perimeter.
        let lidHalfD = tankD / 2 + 0.01
        let tankLidBottomY = h - 0.025 + lift
        cistern.append(roundedSlab(halfW: thw + 0.01, halfD: lidHalfD, y0: tankLidBottomY, y1: tankTopY,
                                   topEdge: lidEdgeRadius, planCorner: cornerRadius)
                        .translated(by: Vec3(0, 0, hd + 0.01 - lidHalfD)))

        // ── Flush pill on the cistern lid (the only brightwork: its own part).
        let flushHalfD = tankD * 0.125
        let flush = roundedSlab(halfW: 0.03, halfD: flushHalfD, y0: tankTopY, y1: tankTopY + 0.008,
                                topEdge: 0.004, planCorner: flushHalfD)
            .translated(by: Vec3(0, 0, hd - tankD * 0.425))

        let hingeZ = (seat.positions + lid.positions).map(\.z).max() ?? bowlCenterZ + bowlHalfD
        return Parts(pan: pan, water: water, seat: seat, lid: lid, cistern: cistern, flush: flush,
                     hinge: Vec3(0, lipY + seatThk, hingeZ),
                     tankFrontZ: hd - tankD, tankLidFrontZ: hd + 0.01 - 2 * lidHalfD,
                     tankLidBottomY: tankLidBottomY, tankTopY: tankTopY)
    }

    // ── The hinge ────────────────────────────────────────────────────────────────────────────

    /// `p` turned by `angle` about the hinge line (+X through `parts.hinge`); a positive angle
    /// lifts the front of the seat/lid (right-handed about +X: −Z swings toward +Y).
    public static func hinged(_ p: Vec3, angle: Double, in parts: Parts) -> Vec3 {
        let r = p - parts.hinge, c = cos(angle), s = sin(angle)
        return parts.hinge + Vec3(r.x, r.y * c - r.z * s, r.y * s + r.z * c)
    }

    /// The open pose (radians): the lid turned back past upright until it meets the cistern — its
    /// front face, or the overhanging edge of the cistern lid — found by turning the real lid,
    /// not guessed. The seat can open to the same angle: the two are a stack on one hinge, so the
    /// same turn cannot pass one through the other.
    public static func openAngle(_ parts: Parts) -> Double {
        var a = 90.0 * .pi / 180
        while a < 130 * .pi / 180 {
            let touches = parts.lid.positions.contains { v in
                let q = hinged(v, angle: a, in: parts)
                guard q.y < parts.tankTopY else { return false }
                return q.z >= (q.y >= parts.tankLidBottomY ? parts.tankLidFrontZ : parts.tankFrontZ) - 0.001
            }
            if touches { return a }
            a += 0.25 * .pi / 180
        }
        return 105 * .pi / 180
    }

    // ── Rounded-rectangle solids (lofted rings, so winding is right by construction) ──────────

    /// One rounded-rectangle ring at height `y`: four quarter-arcs of radius `corner`, `segs`
    /// points each, counter-clockwise seen from above.
    static func ring(halfW: Double, halfD: Double, corner: Double, y: Double, segs: Int) -> [Vec3] {
        let r = max(0.0002, min(corner, min(halfW, halfD)))
        let ix = halfW - r, iz = halfD - r
        var out: [Vec3] = []
        out.reserveCapacity(segs * 4)
        for c in [(ix, iz, 0.0), (-ix, iz, Double.pi / 2), (-ix, -iz, Double.pi), (ix, -iz, 3 * Double.pi / 2)] {
            for s in 0 ..< segs {
                let a = c.2 + (.pi / 2) * Double(s) / Double(segs)
                out.append(Vec3(c.0 + r * cos(a), y, c.1 + r * sin(a)))
            }
        }
        return out
    }

    /// Facets per quarter-corner, sized from the half-diagonal (Daydream's FacetingAudit basis).
    static func cornerSegments(halfW: Double, halfD: Double) -> Int {
        max(4, (Mesh3.segmentsFor(radius: (halfW * halfW + halfD * halfD).squareRoot()) + 3) / 4)
    }

    /// A prism with rounded vertical corners, capped top and bottom, smooth round the corners.
    public static func roundedPrism(halfW: Double, halfD: Double, y0: Double, y1: Double, corner: Double) -> Mesh3 {
        guard halfW > 1e-4, halfD > 1e-4, y1 > y0 else { return Mesh3() }
        let segs = cornerSegments(halfW: halfW, halfD: halfD)
        var m = Mesh3()
        _ = m.loft(rings: [ring(halfW: halfW, halfD: halfD, corner: corner, y: y0, segs: segs),
                       ring(halfW: halfW, halfD: halfD, corner: corner, y: y1, segs: segs)])
        return m.smoothed(creaseDegrees: smoothCrease)
    }

    /// A slab whose TOP perimeter edge rolls over with radius `topEdge` (a quarter-round), on a
    /// rounded-rectangle plan of corner `planCorner` — a cistern lid, a flush pill.
    public static func roundedSlab(halfW: Double, halfD: Double, y0: Double, y1: Double,
                                   topEdge: Double, planCorner: Double) -> Mesh3 {
        guard halfW > 1e-4, halfD > 1e-4, y1 > y0 else { return Mesh3() }
        let rr = max(0, min(topEdge, (y1 - y0) * 0.95, min(halfW, halfD) * 0.5))
        let segs = cornerSegments(halfW: halfW, halfD: halfD)
        var rings = [ring(halfW: halfW, halfD: halfD, corner: planCorner, y: y0, segs: segs)]
        let steps = rr > 0 ? 5 : 1
        for k in 0 ... steps {
            let t = rr > 0 ? Double(k) / Double(steps) * .pi / 2 : .pi / 2
            let inset = rr * (1 - cos(t)), y = y1 - rr + rr * sin(t)
            rings.append(ring(halfW: halfW - inset, halfD: halfD - inset,
                              corner: max(0.0002, planCorner - inset), y: y, segs: segs))
        }
        // A flat ring a few mm inside the roll: the cap then fans from HERE, so the roll's tilted
        // normals (averaged into the rim within the smoothing crease) stay in a thin rim strip
        // instead of being interpolated across whole fan triangles to the centre — which drew an
        // X of shading corner-to-corner across the lid.
        let flat = rr + min(0.006, min(halfW, halfD) * 0.25)
        if halfW - flat > 1e-4, halfD - flat > 1e-4 {
            rings.append(ring(halfW: halfW - flat, halfD: halfD - flat,
                              corner: max(0.0002, planCorner - flat), y: y1, segs: segs))
        }
        var m = Mesh3()
        _ = m.loft(rings: rings)
        return m.smoothed(creaseDegrees: smoothCrease)
    }
}
