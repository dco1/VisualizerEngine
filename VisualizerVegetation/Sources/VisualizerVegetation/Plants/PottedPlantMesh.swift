import Foundation
import simd
import VisualizerMaterials

/// Procedural potted-plant mesh — a revolved pot or vase with its soil/water disc, and the plant
/// growing out of it.
///
/// The five FOLIAGE houseplants are each grown by their own habit and painted leaf by leaf —
/// `PottedPlantMesh+Houseplants` (fig, monstera, snake plant, fern, succulent), built on
/// `LeafSheet` blades and `FleshyLeaf` solids; the bouquet styles and the Christmas tree build their
/// own parts below and in `+ChristmasTree`.
///
/// The parts are returned **separately** because they render with different treatment: the
/// vessel takes the user's finish, the substrate (soil, trunk) fixed natural materials, the
/// foliage its per-vertex paint (`Parts.foliageColors`) and the style's leaf roughness, and each
/// bloom colour group its own albedo.
///
/// **Single source of truth:** every dimension is read from `PottedPlantParams` — the pot from
/// `potRadius`/`potHeight`, the foliage from `plantSize`/`plantStyle`/`foliageDensity`/`seed`.
/// This file never re-derives a size.
///
/// **Local space:** X/Z plan (centered on the axis), Y up from the BOTTOM of the pot (Y=0 is
/// the pot base). The render bridge rotates by the plant's `angle` and lifts to its rest height.
///
/// **Avoids the four "tells"** (PROCEDURAL_GENERATORS §0): the pot is a real revolved solid
/// with a soil disc (not a sealed hollow box), the stem is a real tapered revolve (positive
/// volume), leaves are curved two-faced sheets or closed solids with a genuine spread (not a
/// coplanar decal), and the pot's rounded profile softens the rim arris so
/// `PlaceableAudit`/`FacetingAudit` read curved rings, not a hard CAD cylinder.
public enum PottedPlantMesh {

    /// A coloured sub-mesh with its own draw group — the bouquet's blooms, a Christmas tree's
    /// baubles, a dried spray's stems and sprigs. Either ONE albedo (`color`) stamped over the
    /// group, or — for a PAINTED group (the bouquet's petals, `PottedPlantMesh+Bouquet`) — one
    /// albedo per vertex in `colors`, which the bridge then stamps instead.
    public struct ColoredGroup: Sendable {
        public var color: Vec3     // linear-ish albedo for this group (the flat stamp)
        public var mesh: Mesh3
        /// One albedo per `mesh` vertex for a painted group; EMPTY for a flat `color` group.
        public var colors: [Vec3]
        public init(color: Vec3, mesh: Mesh3, colors: [Vec3] = []) {
            precondition(colors.isEmpty || colors.count == mesh.vertexCount,
                         "a painted group needs one colour per vertex")
            self.color = color; self.mesh = mesh; self.colors = colors
        }
    }

    /// The plant's mesh parts, split so a picked finish paints ONLY the vessel — never the plant.
    /// `vessel` = the pot/vase SHELL (outer wall + rim), the surface the user's `potMaterial` finish
    /// applies to. `substrate` = the soil / water disc + the opaque stem — the growing medium and the
    /// plant's own trunk, which get FIXED natural materials (dark earth / woody), so choosing a
    /// marble pot doesn't turn the stem and soil to marble. Both are opaque (vertex alpha 1).
    /// `foliage` = the leaves (and a bouquet's green stems), one draw group for the whole plant.
    /// `blooms` = per-colour petal/center groups (the `.flowers` vivid bloom colours); empty for the
    /// foliage-only styles.
    public struct Parts: Sendable {
        public var vessel: Mesh3
        public var substrate: Mesh3
        public var foliage: Mesh3
        /// One albedo per `foliage` vertex, painted by the generator (`FoliagePaint`): leaf to leaf,
        /// top to underside, rib to margin. EMPTY for a style that is not painted (the bouquet greens,
        /// the fir canopy) — the bridge then stamps the style's one `leafColor`, as it always did.
        public var foliageColors: [Vec3]
        public var blooms: [ColoredGroup]
        /// The string-light BULBS — the one self-lit part of any plant (`PlantStyle.hasStringLights`),
        /// so it is its own draw group: the bridge gives it emission when the string is on and a
        /// plain glass-bulb look when it is off. Empty for every other style.
        public var bulbs: Mesh3

        public init(vessel: Mesh3, substrate: Mesh3, foliage: Mesh3, foliageColors: [Vec3] = [],
                    blooms: [ColoredGroup] = [], bulbs: Mesh3 = Mesh3()) {
            precondition(foliageColors.isEmpty || foliageColors.count == foliage.vertexCount,
                         "painted foliage needs one colour per vertex")
            self.vessel = vessel; self.substrate = substrate
            self.foliage = foliage; self.foliageColors = foliageColors
            self.blooms = blooms; self.bulbs = bulbs
        }

        /// The opaque solid the winding/volume auditors check: the vessel shell closed by the
        /// soil/water disc, plus the stem — i.e. the old single `body` (pot + soil + stem). Split for
        /// *materials* only; geometrically it's still one watertight positive-volume solid.
        public var opaqueBody: Mesh3 { var m = vessel; m.append(substrate); return m }
    }

    // MARK: - Assembly

    /// `localBendDir` — a unit direction in the plant's OWN local X/Z (before the render bridge's
    /// placement rotation) that a `.flowers` bouquet's stems bow toward, e.g. "the nearest window"
    /// resolved by the caller (the mesh itself has no notion of the room it sits in — see
    /// `HouseRenderBridge+PottedPlant.swift`). `nil` (the default, and every existing call site) is
    /// a plain seeded bend with no directional bias.
    public static func parts(params: some PottedPlantGeometry, localBendDir: Vec3? = nil) -> Parts {
        // The vessel SHELL takes the user's finish; the soil/water disc is substrate (fixed natural).
        let vessel = vesselShell(params: params)
        var substrate = vesselFill(params: params)
        var rng = SplitMix(params.seed &* 0x2545F4914F6CDD1D &+ 0x9E3779B97F4A7C15)

        // The soil / water line sits a bit below the rim; stems + foliage rise from it. Single-
        // source from the vessel (pot soil disc vs vase water disc).
        let soilY = vesselMouthY(params)

        if params.plantStyle.isBouquet {
            // A cut-flower bouquet: painted green stems + leaves (the foliage group) and painted
            // flowers (ONE bloom group) — `PottedPlantMesh+Bouquet`.
            let bouquet = bouquetMesh(params: params, soilY: soilY, rng: &rng, localBendDir: localBendDir)
            return Parts(vessel: vessel, substrate: substrate, foliage: bouquet.greens,
                         foliageColors: bouquet.greenColors, blooms: bouquet.blooms)
        }

        // Foliage plant: a single trunk/stub stem (opaque — the plant's own woody trunk, so it's
        // substrate, NOT the vessel finish) + the painted leaves.
        let stem = stemMesh(params: params, baseY: soilY)
        substrate.append(stem)
        if params.plantStyle == .christmasTree {
            // A fir is a solid tiered canopy, not leaf cards — see PottedPlantMesh+ChristmasTree.
            // Baubles ride the per-colour bloom groups; the bulbs are their own self-lit group.
            let tree = christmasTree(params: params, soilY: soilY, rng: &rng)
            return Parts(vessel: vessel, substrate: substrate, foliage: tree.canopy,
                         blooms: tree.baubles, bulbs: tree.bulbs)
        }
        let grown = foliage(params: params, soilY: soilY, rng: &rng)
        substrate.append(grown.woody)
        return Parts(vessel: vessel, substrate: substrate, foliage: grown.mesh, foliageColors: grown.colors)
    }

    /// Convenience: the whole plant as ONE mesh (vessel + substrate + leaves/blooms) — for the
    /// auditors, which check the merged solid's winding/holes/volume. (The bridge uses `parts` for
    /// the per-group materials + SSS flag + per-bloom colour.)
    public static func mesh(params: some PottedPlantGeometry, localBendDir: Vec3? = nil) -> Mesh3 {
        let p = parts(params: params, localBendDir: localBendDir)
        var m = p.opaqueBody
        m.append(p.foliage)
        for g in p.blooms { m.append(g.mesh) }
        m.append(p.bulbs)
        return m
    }

    /// A non-bouquet style's leaves, grown the way both the pot and the garden grow them: the five
    /// houseplants painted leaf by leaf (`PottedPlantMesh+Houseplants`), the fir its tiered canopy
    /// (unpainted — `colors` empty, the bridge stamps its one green).
    public static func foliage(params: some PottedPlantGeometry, soilY: Double, rng: inout SplitMix)
        -> (mesh: Mesh3, colors: [Vec3], woody: Mesh3) {
        if params.plantStyle == .christmasTree {
            let tiers = coniferTiers(params: params, soilY: soilY, rng: &rng)
            return (coniferCanopy(tiers), [], Mesh3())
        }
        guard let grown = houseplantFoliage(params: params, soilY: soilY, rng: &rng) else {
            return (Mesh3(), [], Mesh3())
        }
        return (grown.leaves.mesh, grown.leaves.colors, grown.woody)
    }

    // MARK: - Vessel: pot or vase (dispatch on the resolved vessel kind)

    /// The vessel body — a tapered planter (`.pot`) or a bellied flower vase (`.vase`), chosen by
    /// `params.resolvedVessel`. Both are closed positive-volume revolves with a soil/water disc so
    /// they never read as a hollow tube, and both keep their rim arris soft (curved rings).
    public static func vesselMesh(params: some PottedPlantGeometry) -> Mesh3 {
        switch params.resolvedVessel {
        case .pot:  return potMesh(params: params)
        case .vase: return vaseMesh(params: params)
        }
    }

    /// The vessel SHELL only (outer wall + rim, no soil/water disc) — the surface the user's finish
    /// paints. Open at the top (the substrate's soil/water disc closes it in the merged solid).
    ///
    /// `.smoothed()` here (not inside `potShellMesh`/`vaseShellMesh`) is the ONE site every vessel
    /// passes through, pot or vase: the revolved wall was flat-shaded (each longitude facet its own
    /// stored normal), which reads as a faceted/diamond-cut surface up close even though
    /// `FacetingAudit`'s chord error already passed — chord error is a pure geometry metric and
    /// doesn't see per-facet shading. The rim's inward lip fold is a genuine crease (the wall
    /// normal and the fold's normal differ by far more than 40°) and stays sharp; the belly/neck/
    /// foot curve is a gently-changing profile and smooths cleanly. Smoothing a sub-part flips its
    /// contract from `MeshAudit.isSound`'s flat-shading `windingMismatches` check to
    /// `isSoundSmooth`'s `invertedWindingFaces` — see `PottedPlantMeshTests`.
    public static func vesselShell(params: some PottedPlantGeometry) -> Mesh3 {
        let raw: Mesh3
        switch params.resolvedVessel {
        case .pot:  raw = potShellMesh(params: params)
        case .vase: raw = vaseShellMesh(params: params)
        }
        return raw.smoothed()
    }

    /// The vessel FILL only (the soil disc for a pot, the water disc for a vase) — substrate, given a
    /// fixed natural material by the bridge, never the user's vessel finish.
    public static func vesselFill(params: some PottedPlantGeometry) -> Mesh3 {
        switch params.resolvedVessel {
        case .pot:  return potSoilMesh(params: params)
        case .vase: return vaseWaterMesh(params: params)
        }
    }

    // MARK: - Pot: a tapered revolved vessel + a soil disc near the rim

    /// A tapered planter — base narrower than the rim, `potHeight` tall — with a thin inner soil
    /// disc just below the rim so it doesn't read as a hollow tube. All revolved, so the rim
    /// arris is a curved ring (soft), and the whole thing is a closed positive-volume solid.
    public static func potMesh(params: some PottedPlantGeometry) -> Mesh3 {
        var m = potShellMesh(params: params).smoothed()
        m.append(potSoilMesh(params: params))
        return m
    }

    /// ONE longitude segment count for the WHOLE pot — wall, lip fold, AND the (separately-built)
    /// soil disc all share it. `revolve()` only guarantees matching vertex columns WITHIN one call
    /// across its own profile bands; it says nothing about two SEPARATE `revolve()` calls whose
    /// rings are meant to be the same physical circle (the wall's open top ring and the lip fold's
    /// first ring both sit at `(rRim, h)`; the fold's last ring and the soil disc's first both sit
    /// at `(rSoil, h − 0.006)`). Each call used to size itself from its OWN profile's widest radius,
    /// so the wall (max `rRim`) and the fold (max `rRim` too, but narrower on average) could still
    /// round to different segment counts than the soil disc (max `rSoil`, smaller) — two differently
    /// -sided polygons inscribed in the SAME circle don't share edges, which leaves a thin sliver
    /// gap all the way around the seam ("looks like some gaps", clipping/see-through at the rim).
    /// `segmentsFor` is a pure function of radius, so every one of `potShellMesh`/`potSoilMesh` —
    /// called independently, never passing this value to each other — derives the IDENTICAL count
    /// from the SAME `rRim`, closing every seam without any shared state.
    private static func potSegments(_ params: some PottedPlantGeometry) -> Int {
        Mesh3.segmentsFor(radius: params.potRadius)
    }

    /// **A thrown pot's lip is ROLLED, and this is the arc that rolls it.**
    ///
    /// A wall that runs straight up to the rim and then turns inward is a cut TUBE — the join is a
    /// ~90° arris, `Mesh3.smoothed(creaseDegrees: 40)` correctly keeps it as a crease, and it
    /// renders as the hard machined edge Danny caught on the first glazed pot (2026-09-09: *"look
    /// at the hard edges on that rim"*). No material can fix that: it is the shape.
    ///
    /// So the top of the wall turns over a **half-round bull-nose spanning the full wall
    /// thickness** — the shape a potter's fingers actually leave, and the shape a glaze then pools
    /// on and breaks over. The arc is sampled at `stations` points from θ = 0 (the outer wall's
    /// top, vertical tangent, so the join is tangent-continuous and smooths away) over the crown
    /// to θ = 180° (the inner face, vertical tangent again). Radius is HALF the wall thickness, so
    /// the far end lands exactly on the inner radius with nothing left to bridge.
    ///
    /// `stations: 8` is 22.5° per facet — under `smoothed`'s 40° crease angle, so the roll shades
    /// as one continuous surface rather than as a bevel of flats.
    public static func rolledLip(outerR: Double, innerR: Double, y: Double, stations: Int = 8)
        -> [Mesh3.ProfilePoint] {
        let roll = (outerR - innerR) / 2
        let cR = outerR - roll
        let n = Swift.max(3, stations)
        return (0...n).map { i in
            // **The two ENDPOINTS are stated, not computed**, and that is not tidiness. This arc's
            // last point is the ring a separately-built fill disc starts on, and those two calls
            // have to agree to the last bit or `revolve` emits two rings that don't weld:
            // `cR + roll·cos(π)` is `outerR − 2·roll`, which is `innerR` in algebra and `innerR`
            // ± an ULP in Double — and `sin(π)` is 1.2e-16, not 0. Measured: the vase came out
            // with 4 open edges (`PottedPlantMeshTests.testVesselHasNoSeamGaps`) — only 4, because
            // `MeshAudit` welds by a quantised position hash and just four of the ring's vertices
            // straddled a bucket boundary. That is the failure this file's `potSegments` note is
            // about, one decimal place further down.
            if i == 0 { return .init(r: outerR, y: y) }
            if i == n { return .init(r: innerR, y: y) }
            let t = Double.pi * Double(i) / Double(n)     // 0 → π, outer over the crown to inner
            return .init(r: cR + roll * cos(t), y: y + roll * sin(t))
        }
    }

    /// **The one seam ring where a pot's shell meets its soil disc** — the physical circle two
    /// independent `revolve()` calls have to agree on to the last bit.
    ///
    /// It exists for the same reason `potSegments` does, one level down: the two calls used to
    /// each type `(rRim * 0.92, h - 0.006)`, and a literal typed twice is a fact typed twice. The
    /// rim roll moves this ring (the lip now turns over instead of folding flat), which is exactly
    /// the edit that would have left one of the two behind.
    public static func potRimSeam(_ params: some PottedPlantGeometry) -> Mesh3.ProfilePoint {
        .init(r: params.potRadius * 0.92, y: params.potHeight - potRimRoll(params))
    }

    /// The pot's wall thickness — and so, halved, its lip's roll radius. 8 % of the rim radius: a
    /// planter is a chunky thing, and a wall thinner than this reads as a plastic cup.
    public static func potRimRoll(_ params: some PottedPlantGeometry) -> Double { params.potRadius * 0.04 }

    /// The pot SHELL — outer wall + inward rim lip, open at the soil radius (the soil disc closes
    /// it). This is what the user's picked finish paints.
    public static func potShellMesh(params: some PottedPlantGeometry) -> Mesh3 {
        let rRim = params.potRadius
        let rBase = rRim * 0.78               // classic tapered planter (base narrower than rim)
        let h = params.potHeight
        let rSoil = rRim * 0.92
        let segments = potSegments(params)

        var m = Mesh3()
        // Outer wall: base ring → rim ring. capBottom seals the base; leave the top OPEN so the
        // inner lip + soil disc can close it (a real reveal, not a coplanar cap). `smoothedProfile`
        // turns the 3 hand-picked waypoints into a real curve (not just more collinear points on
        // the same straight chords) so the foot flare reads as a genuine bend under specular
        // lighting, not a kink at 2 flat panels.
        // The wall stops one roll radius SHORT of the full height — the rolled lip below carries
        // it the rest of the way, and its crown is what sits at `h`. Total height is unchanged.
        let roll = potRimRoll(params)
        m.revolve(profile: Mesh3.smoothedProfile([
            .init(r: rBase, y: 0),
            .init(r: rBase * 1.02, y: h * 0.06),   // a slight foot flare
            .init(r: rRim, y: h - roll),
        ]), segments: segments, capBottom: true, capTop: false)

        // Rim lip: a ROLLED half-round over the wall thickness, outer face up over the crown and
        // back down to the soil radius — a thrown lip, not a square-cut tube (see `rolledLip`).
        // Its last point IS `potRimSeam`, so the soil disc meets it with nothing to bridge.
        m.revolve(profile: rolledLip(outerR: rRim, innerR: rSoil, y: h - roll),
                  segments: segments, capBottom: false, capTop: false)
        return m
    }

    /// The pot's SOIL disc — a low domed cap capping the interior. Substrate (dark matte earth),
    /// NOT the vessel finish.
    public static func potSoilMesh(params: some PottedPlantGeometry) -> Mesh3 {
        let rRim = params.potRadius
        let h = params.potHeight
        let soilY = h * 0.86                  // soil disc sits a little below the rim
        let rSoil = rRim * 0.92

        var m = Mesh3()
        m.revolve(profile: [
            potRimSeam(params),                   // the shared ring — never re-derived here
            .init(r: rSoil * 0.7, y: soilY + 0.004),
            .init(r: 0, y: soilY + 0.010),
        ], segments: potSegments(params), capBottom: false, capTop: false)
        return m
    }

    // MARK: - Vase: a taller, narrower bellied flower vase

    /// A classic flower vase — a revolved silhouette that is **taller and narrower** than the
    /// planter pot: a foot, a bulging BELLY, a pinched NECK, then a slightly FLARED lip. All
    /// single-source from `potRadius`/`potHeight`: the vase is ~1.8× the pot height and its widest
    /// belly is the `potRadius` (so the footprint OBB still fits), the neck ~0.55× that. A thin
    /// water disc caps the interior a little below the rim (never a hollow pipe). Closed positive
    /// volume; all revolved so `FacetingAudit`/`PlaceableAudit` read smooth curved rings.
    public static func vaseMesh(params: some PottedPlantGeometry) -> Mesh3 {
        var m = vaseShellMesh(params: params).smoothed()
        m.append(vaseWaterMesh(params: params))
        return m
    }

    /// Single-source vase silhouette radii/heights — the shell, the water disc, `vesselMouthY`, and
    /// the bouquet's stem-vs-neck clearance clamp (`bouquetMesh`) all read from here, so they can
    /// never drift out of step the way `rNeck`/`waterY` etc. used to (each re-derived separately in
    /// three different functions). Every radius is a fraction of `potRadius`, the belly/footprint.
    public struct VaseProfile {
        public var h: Double         // overall vase height
        public var rBelly: Double
        public var rFoot: Double
        public var rNeck: Double     // the pinched waist a bouquet's stems must clear on the way up
        public var rLip: Double
        public var rWater: Double
        public var waterY: Double    // where stems/blooms emerge (mirrors `vesselMouthY`)
        public var neckY: Double     // height of the neck pinch
    }

    /// The vase's lip roll radius — half its wall thickness, exactly as the pot's is. Derived from
    /// the lip and water radii the profile already declares, so the two can never disagree.
    public static func vaseRimRoll(_ vp: VaseProfile) -> Double { (vp.rLip - vp.rWater) / 2 }

    /// **The one seam ring where a vase's shell meets its water disc** — the pot's `potRimSeam`,
    /// one vessel over, and there for the same reason: `(rWater, h - 0.006)` used to be typed in
    /// both `vaseShellMesh` and `vaseWaterMesh`, and rolling the lip moves it.
    public static func vaseRimSeam(_ vp: VaseProfile) -> Mesh3.ProfilePoint {
        .init(r: vp.rWater, y: vp.h - vaseRimRoll(vp))
    }

    public static func vaseProfile(_ params: some PottedPlantGeometry) -> VaseProfile {
        let rBelly = params.potRadius
        let h = params.potHeight * 1.8
        let rNeck = rBelly * 0.55
        return VaseProfile(h: h, rBelly: rBelly, rFoot: rBelly * 0.52, rNeck: rNeck,
                           rLip: rBelly * 0.64, rWater: rNeck * 0.9,
                           waterY: h * 0.72, neckY: h * 0.82)
    }

    /// ONE longitude segment count for the WHOLE vase — wall, lip fold, AND the (separately-built)
    /// water disc all share it, for the same reason `potSegments` exists: three independent
    /// `revolve()` calls whose rings are meant to be the SAME physical circle (wall↔fold at
    /// `rLip`, fold↔water disc at `rWater`) each used to size itself off its own profile's widest
    /// radius, so they could land on different segment counts — two differently-sided polygons
    /// inscribed in one circle don't share edges, leaving a sliver gap at the seam. A pure function
    /// of `rBelly`, so `vaseShellMesh`/`vaseWaterMesh` (never passing this to each other) derive
    /// the identical count independently.
    private static func vaseSegments(_ params: some PottedPlantGeometry) -> Int {
        Mesh3.segmentsFor(radius: params.potRadius)
    }

    /// The vase SHELL — outer wall (foot → belly → neck → flared lip) + inward lip fold, open at the
    /// water radius (the water disc closes it). This is what the user's picked finish paints.
    public static func vaseShellMesh(params: some PottedPlantGeometry) -> Mesh3 {
        let vp = vaseProfile(params)
        let segments = vaseSegments(params)
        let roll = vaseRimRoll(vp)
        var m = Mesh3()
        // Outer wall, base → belly → neck → flared lip. capBottom seals the foot; top OPEN so the
        // lip fold + water disc close it (a real reveal, not a coplanar cap). `smoothedProfile`
        // fits a real curve through these 6 hand-picked waypoints instead of connecting them with
        // straight chords: `revolve`'s adaptive segment count only rounds the AZIMUTHAL facet
        // count, not the meridional one, so the belly→neck→lip bend — which turns noticeably over
        // just 2–3 of the original chords — smoothed continuously in SHADING (`Mesh3.smoothed()`)
        // but still kinked a glossy material's specular highlight at each flat chord's edge. More
        // real curvature here is what actually rounds it, not just more collinear samples on the
        // same straight lines.
        m.revolve(profile: Mesh3.smoothedProfile([
            .init(r: vp.rFoot, y: 0),
            .init(r: vp.rFoot * 1.04, y: vp.h * 0.04),   // a slight foot flare (stability read)
            .init(r: vp.rBelly, y: vp.h * 0.42),         // the belly bulge
            .init(r: vp.rBelly * 0.86, y: vp.h * 0.62),
            .init(r: vp.rNeck, y: vp.neckY),             // pinched neck
            .init(r: vp.rLip, y: vp.h - roll),           // flared lip, one roll short of the top
        ]), segments: segments, capBottom: true, capTop: false)

        // Lip: a ROLLED half-round over the wall thickness — a thrown lip, not a square-cut tube
        // (see `rolledLip`). Its last point IS `vaseRimSeam`, which the water disc reads.
        m.revolve(profile: rolledLip(outerR: vp.rLip, innerR: vp.rWater, y: vp.h - roll),
                  segments: segments, capBottom: false, capTop: false)
        return m
    }

    /// The vase's WATER disc — a low domed cap inside the neck (a dark still fill). Substrate, NOT
    /// the vessel finish.
    public static func vaseWaterMesh(params: some PottedPlantGeometry) -> Mesh3 {
        let vp = vaseProfile(params)
        var m = Mesh3()
        m.revolve(profile: [
            vaseRimSeam(vp),                             // the shared ring — never re-derived here
            .init(r: vp.rWater * 0.7, y: vp.waterY + 0.004),
            .init(r: 0, y: vp.waterY + 0.010),
        ], segments: vaseSegments(params), capBottom: false, capTop: false)
        return m
    }

    /// Where stems/blooms emerge from the vase interior (the water line). Single-source: mirrors
    /// `vaseProfile`'s `waterY`. For a pot, this is just below the soil disc (`potHeight * 0.86`).
    public static func vesselMouthY(_ params: some PottedPlantGeometry) -> Double {
        switch params.resolvedVessel {
        case .pot:  return params.potHeight * 0.86
        case .vase: return vaseProfile(params).waterY
        }
    }

    // MARK: - Stem: a tapered revolved post from the soil into the foliage

    /// The stem's own base radius (metres) — SINGLE SOURCE for `stemMesh`'s revolve profile AND
    /// `emitBlade`'s petiole attach point (`stemRadius(_:at:)` below). A petiole used to attach at
    /// a fraction of the POT radius, which has no relationship to how thick the trunk actually is
    /// at that height — the trunk tapers to under half its base radius by the top of a fiddle-leaf
    /// fig, so upper petioles started 3-4× further out than the real trunk surface: a visible
    /// floating gap, worse the higher up the stem a leaf sits (Danny, 2026-09-11: "wrong position").
    private static func stemBaseRadius(_ style: PlantStyle) -> Double {
        switch style {
        case .fiddleLeafFig: return 0.018
        case .monstera:      return 0.016
        case .snakePlant:    return 0.012
        case .fern:          return 0.010
        case .succulent:     return 0.014
        case .flowers, .driedSpray: return 0.006   // unused (bouquetMesh builds its own stems)
        case .christmasTree: return 0.026          // a real trunk, visible below the lowest tier
        }
    }

    /// The stem's radius at `t` (0 = base, 1 = top) — the SAME taper `stemMesh` revolves (base →
    /// 0.7× at mid-height → 0.45× at the top), evaluated as a function so a petiole's attach point
    /// can read the trunk's REAL surface at its own height instead of guessing.
    public static func stemRadius(_ style: PlantStyle, at t: Double) -> Double {
        let baseR = stemBaseRadius(style)
        let tc = max(0, min(1, t))
        return tc <= 0.5
            ? baseR + (baseR * 0.7 - baseR) * (tc / 0.5)
            : baseR * 0.7 + (baseR * 0.45 - baseR * 0.7) * ((tc - 0.5) / 0.5)
    }

    public static func stemMesh(params: some PottedPlantGeometry, baseY: Double) -> Mesh3 {
        let style = params.plantStyle
        // Stem height as a fraction of the foliage size — tall trunk for the tree-form plants,
        // a short stub for the rosette plants (their blades emerge near the soil).
        let frac: Double
        switch style {
        case .fiddleLeafFig: frac = 0.75
        case .monstera:      frac = 0.30
        case .snakePlant:    frac = 0.06
        case .fern:          frac = 0.10
        case .succulent:     frac = 0.04
        case .flowers, .driedSpray: frac = 0.80   // unused (bouquetMesh builds its own stems)
        case .christmasTree: frac = 0.85          // the trunk runs up the canopy's core
        }
        let baseR = stemBaseRadius(style)
        let stemH = max(0.02, params.plantSize * frac)
        var m = Mesh3()
        m.revolve(profile: [
            .init(r: baseR, y: baseY),
            .init(r: baseR * 0.7, y: baseY + stemH * 0.5),
            .init(r: baseR * 0.45, y: baseY + stemH),
        ], capBottom: false, capTop: true)
        return m
    }

    static func stemFraction(_ style: PlantStyle) -> Double {
        switch style {
        case .fiddleLeafFig: return 0.75
        case .monstera:      return 0.30
        case .snakePlant:    return 0.06
        case .fern:          return 0.10
        case .succulent:     return 0.04
        case .flowers:       return 0.80   // stems reach most of plantSize (used by bouquetMesh)
        case .driedSpray:    return 0.90
        case .christmasTree: return 0.85   // mirrors `stemMesh` (the canopy builds its own tiers)
        }
    }

    // MARK: - Bouquet (.flowers): painted stems + flowers (`PottedPlantMesh+Bouquet`)

    /// The planter bed's flower palette (`PlanterMesh`'s seasonal indices point into it) — white /
    /// pink / yellow / red / purple / coral, each `(petal, center)`. The bouquet's own palette,
    /// `bouquetPalette`, runs in the same order. Taste: neutral defaults, flagged for Danny.
    public static let bloomPalette: [(petal: Vec3, center: Vec3)] = [
        (Vec3(0.96, 0.96, 0.94), Vec3(0.92, 0.78, 0.16)),   // white daisy, yellow eye
        (Vec3(0.95, 0.55, 0.70), Vec3(0.90, 0.72, 0.20)),   // pink, gold center
        (Vec3(0.97, 0.82, 0.20), Vec3(0.55, 0.34, 0.08)),   // yellow, brown center
        (Vec3(0.85, 0.16, 0.18), Vec3(0.30, 0.06, 0.05)),   // red, dark center
        (Vec3(0.62, 0.34, 0.72), Vec3(0.94, 0.86, 0.28)),   // purple, yellow center
        (Vec3(0.96, 0.48, 0.28), Vec3(0.70, 0.28, 0.10)),   // coral, rust center
    ]

    /// A bouquet's parts: the green stems and leaves (`greenColors` their paint, or empty for a
    /// flat stamp) and the bloom groups.
    public struct BouquetParts {
        public var greens: Mesh3
        public var greenColors: [Vec3] = []
        public var blooms: [ColoredGroup]
    }

    /// Build a cut-flower bouquet rising from the vessel mouth at `soilY`: several thin green stems
    /// fanning out and bowing (toward `localBendDir`, the nearest window, when there is one), a
    /// leaf or two on the lower stems, and a FLOWER on each — a rose, tulip, daisy or anemone,
    /// its form and colour drawn together from `bouquetPalette`, now and then a closed bud
    /// (`PottedPlantMesh+Bouquet`). Stems and leaves are the painted `greens`; every flower is one
    /// painted bloom group. Dimensions from `params` (stem height from `plantSize`, count from
    /// `foliageDensity`).
    public static func bouquetMesh(params: some PottedPlantGeometry, soilY: Double, rng: inout SplitMix,
                            localBendDir: Vec3? = nil) -> BouquetParts {
        if params.plantStyle == .driedSpray { return driedSprayMesh(params: params, soilY: soilY, rng: &rng) }
        var greens = PaintedMesh()
        var flowers = PaintedMesh()
        let palette = PlantStyle.flowers.foliagePalette

        let size = params.plantSize
        let density = Double(params.foliageDensity)
        // Stem count scales with density (a fuller bouquet). 5 stems at 1.0×, clamped 3…12.
        let stemCount = max(3, min(12, Int((5.0 * density).rounded())))
        let stemTopFrac = 0.75                          // stems reach ~0.75 of plantSize on average

        // A real cut stem is 3–6 mm across whatever the bunch; scaled a little with the size, and
        // clamped — a small bouquet keeping "garden hose" stems read as a zucchini stalk.
        let stemRBase = min(0.0032, max(0.0014, size * 0.009))
        let stemRTip = stemRBase * 0.7

        // The vase's neck/lip pinch the opening well inside the belly — a stem that starts near
        // the water's edge and leans outward can otherwise punch through that pinch on its way up
        // ("stems intersecting the pot"). `vp` (nil for a pot, whose mouth is wide open) gives the
        // single-sourced radii the per-stem lean clamp below checks against.
        let vp = params.resolvedVessel == .vase ? vaseProfile(params) : nil
        let mouthR = vp.map { $0.rWater * 0.82 } ?? params.potRadius * 0.6
        // Leaves grow only ABOVE the vessel's lip: below it they would run through the vase's neck
        // (and a florist strips a cut stem's lower leaves before it goes in the water anyway).
        let lipY = vp?.h ?? params.potHeight

        // A stable per-arrangement palette rotation so seeds vary the colour mix.
        let paletteStart = Int(rng.next() % UInt64(bouquetPalette.count))
        let golden = Phyllotaxis.goldenAngle
        let startAngle = rng.unit() * 2 * Double.pi

        for i in 0 ..< stemCount {
            // Fan the stems out from the mouth: azimuth by golden angle, a small outward lean.
            let azimuth = startAngle + Double(i) * golden + (rng.unit() - 0.5) * 0.4
            let outX = cos(azimuth), outZ = sin(azimuth)
            // Outer stems lean out more; a little length variance so the bouquet isn't a flat top.
            var lean = 0.06 + rng.unit() * 0.22
            let stemH = size * (stemTopFrac + (rng.unit() - 0.5) * 0.28)
            let stemH2 = max(0.06, stemH)
            let bendAmount = 0.05 + rng.unit() * 0.05   // needed below for the clearance margin too

            // Stem base at the mouth (slightly off-axis so stems don't all stack on the axis).
            let baseR = mouthR * (0.2 + rng.unit() * 0.8)

            // Clamp the outward lean so this stem clears the vase's pinched neck AND flared lip —
            // whichever is tighter given how far up each sits. Without this, a wide lean on a
            // narrow-necked vase pushes the stem's radial position past the neck wall partway up.
            // `bendMargin` reserves room for the stem's own BOW too (below): the bow is purely
            // horizontal and could, worst case, land fully radially outward on top of the lean, so
            // the lean budget must shrink to leave space for it rather than being solved alone.
            if let vp {
                let riseNeck = max(0.001, vp.neckY - soilY)
                let riseLip  = max(0.001, vp.h - soilY)
                func bendMargin(atRise rise: Double) -> Double {
                    let t = min(1, rise / stemH2)   // ≈ the bow's own path parameter at that height
                    return bendAmount * stemH2 * t * t
                }
                let maxLeanNeck = atan2(max(0, vp.rNeck * 0.90 - baseR - bendMargin(atRise: riseNeck)), riseNeck)
                let maxLeanLip  = atan2(max(0, vp.rLip * 0.93 - baseR - bendMargin(atRise: riseLip)), riseLip)
                lean = min(lean, maxLeanNeck, maxLeanLip)
            }

            let base = Vec3(outX * baseR, soilY + 0.004, outZ * baseR)
            // Growth direction: mostly up, leaning `lean` outward.
            let up = cos(lean), out = sin(lean)
            let dir = normalize3(Vec3(outX * out, up, outZ * out))
            let straightTip = base + dir * stemH2

            // A SLIGHT bend, mostly in the top half — a real cut stem stays fairly stiff near its
            // cut base and curves over near the bloom. Bows toward `localBendDir` (the nearest
            // window, resolved by the caller) blended with the stem's own outward fan direction so
            // the whole bouquet reads as leaning one way without every stem looking identical; with
            // no window direction it just continues curving along its own outward lean.
            let ownOutward = Vec3(outX, 0, outZ)
            let bendDir = localBendDir.map { normalize3($0 * 0.75 + ownOutward * 0.25) } ?? ownOutward

            // A real round, smoothly-shaded tapered tube swept along the bent centerline, painted
            // one green per stem.
            var stem = Mesh3()
            let (curvedTip, curvedTangent) = appendBentStemTube(
                &stem, from: base, to: straightTip,
                rBase: stemRBase, rTip: stemRTip, bendDir: bendDir, bendAmount: bendAmount)
            let stemTone = FoliagePaint.tone(youth: 0.3, palette: palette, rng: &rng)
            greens.append(stem, color: FoliagePaint.base(palette, stemTone, face: .upper))

            // A leaf or two partway up (painted: a pale rib, a paler back), clear of the lip by a
            // quarter of its own length, and seated ON the bent stem — the same centreline
            // `appendBentStemTube` swept, not the straight line it bowed away from.
            let leaves = rng.unit() < 0.75 ? 1 + Int(rng.next() % 2) : 0
            for k in 0 ..< leaves {
                let length = size * (0.15 + rng.unit() * 0.06)
                let lowest = (lipY + length * 0.25 - base.y) / max(1e-6, dir.y * stemH2)
                let along = max(0.30, lowest) + 0.26 * Double(k) + rng.unit() * 0.10
                guard along < 0.80 else { break }
                let lp = base + dir * (stemH2 * along) + normalize3(bendDir) * (stemH2 * bendAmount * along * along)
                let lAz = azimuth + Double.pi * (k % 2 == 0 ? 0.5 : -0.5) + (rng.unit() - 0.5) * 0.6
                emitStemLeaf(into: &greens, at: lp, azimuth: lAz, tilt: 0.95 + rng.unit() * 0.4,
                             length: length, palette: palette, rng: &rng)
            }

            // The flower sits at the CURVED tip (not the straight-line one) and faces along the bent
            // stem's own tip tangent, so it aims the way its stem actually curved. Its form and colour
            // are drawn together (a white daisy, a red anemone…), and now and then it is a closed bud.
            let entry = bouquetPalette[(paletteStart + i) % bouquetPalette.count]
            let form = entry.forms[Int(rng.next() % UInt64(entry.forms.count))]
            let bud = rng.unit() < 0.18
            let scale = size * (form == .daisy ? 0.13 : 0.12) * (0.85 + rng.unit() * 0.3)
            emitBloom(into: &flowers, form: form, bud: bud, at: curvedTip, axis: curvedTangent,
                      scale: scale, petal: entry.petal, center: entry.center, rng: &rng)
        }

        let blooms = flowers.isEmpty ? [] : [ColoredGroup(color: Vec3(1, 1, 1), mesh: flowers.mesh, colors: flowers.colors)]
        return BouquetParts(greens: greens.mesh, greenColors: greens.colors, blooms: blooms)
    }

    /// One painted stem leaf at `pos`: lanceolate, arching out, a pale rib, a paler back.
    static func emitStemLeaf(into p: inout PaintedMesh, at pos: Vec3, azimuth: Double, tilt: Double,
                             length: Double, palette: FoliagePalette, rng: inout SplitMix) {
        let f = leafFrame(azimuth: azimuth, tilt: tilt, roll: (rng.unit() - 0.5) * 0.5)
        var s = LeafSheet.Surface(base: pos, xAxis: f.x, yAxis: f.y, normal: f.n, length: length, halfWidth: 0.12)
        s.cup = 0.04
        s.droop = 0.12 + rng.unit() * 0.10
        s.twist = (rng.unit() - 0.5) * 0.4
        let tone = FoliagePaint.tone(youth: rng.unit() * 0.4, palette: palette, rng: &rng)
        var leaf = PaintedMesh()
        LeafSheet.emitBlade(into: &leaf, surface: s,
                            margin: LeafSilhouette.stemLeaf.margin(atRows: [0, 0.08, 0.2, 0.34, 0.5, 0.66, 0.8, 0.92, 1]),
                            aspect: 0.24, columns: [0.55], midribBand: 0.006) { site, face in
            FoliagePaint.sheet(palette, tone, site: site, face: face, ribHalfWidth: 0.006)
        }
        p.append(leaf.smoothed())
    }

    /// The PLANTER's bloom (`PlanterMesh`) — a ring (or two) of petal cards round a small centre
    /// disc, all facing `faceDir`, one flat colour per group. Cheap enough for a bed of twenty; the
    /// bouquet grows real flower forms instead (`emitBloom(into:form:…)`, `PottedPlantMesh+Bouquet`).
    /// `scale` sets the bloom radius (petal length). Petals cup toward `faceDir` (an open flower).
    public static func emitBloom(petals: inout Mesh3, center: inout Mesh3,
                          at pos: Vec3, faceDir: Vec3, scale: Double, rng: inout SplitMix) {
        let n = normalize3(faceDir)
        // In-plane basis for arranging petals around the bloom axis.
        let ref = abs(n.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)
        let uAxis = normalize3(cross3(ref, n))
        let vAxis = normalize3(cross3(n, uAxis))

        let petalLen = scale
        let petalW = scale * 0.62
        // DH-0488: always TWO layered rings so every bloom reads full (was sometimes a single flat
        // ring), and each petal cups in 3D via `curl` — inner petals stand up and curl in around the
        // center, outer petals open flatter. A fuller, dimensional flower beside the furniture.
        let rings = 2
        let twoPi = Double.pi * 2
        for ring in 0 ..< rings {
            let count = ring == 0 ? 5 : 4
            let ringScale = ring == 0 ? 1.0 : 0.72
            let phase = Double(ring) * 0.4 + rng.unit() * 0.3
            // Petals in the outer ring lie flatter; inner ring stands more upright (cupped center).
            let pitch = 0.42 + Double(ring) * 0.30      // radians off the bloom axis
            let curl = ring == 0 ? 0.38 : 0.22          // inner petals cup harder; outer open flatter
            for k in 0 ..< count {
                let a = phase + twoPi * Double(k) / Double(count)
                let radial = normalize3(uAxis * cos(a) + vAxis * sin(a))
                // Petal grows outward-and-up: from the bloom axis toward `radial`, pitched by `pitch`.
                let yAxis = normalize3(n * cos(pitch) + radial * sin(pitch))
                let xAxis = normalize3(cross3(yAxis, n))
                // Petal center sits a little out from the bloom center along `radial`.
                let petalPos = pos + radial * (petalLen * 0.22 * ringScale) + n * (scale * 0.04)
                // Cup toward the bloom axis (petals curl up around the center).
                let bentN = normalize3(n * 0.7 + radial * 0.3)
                // Build each petal into its OWN mesh and smooth it BEFORE merging (same rule as
                // every leaf card, `emitBlade` above): a petal's fold+curl is genuine curvature,
                // and left flat-shaded it reads as a faceted paper/gem facet even once the margin
                // itself is rounded — smoothing the whole `petals` accumulator at once would blend
                // normals across neighbouring petals that share near-coincident base positions.
                var petal = Mesh3()
                PlantPetalCard.emitPetal(&petal, pos: petalPos, xAxis: xAxis, yAxis: yAxis,
                                         normal: bentN, w: petalW * ringScale, h: petalLen * ringScale,
                                         curl: curl)
                petals.append(petal.smoothed())
            }
        }
        // Contrasting center button — a small disc facing the bloom axis, lifted slightly. Built
        // and smoothed on its own for the same reason as the petals above.
        var centerDisc = Mesh3()
        PlantPetalCard.emitCenter(&centerDisc, center: pos + n * (scale * 0.08), normal: n,
                                  r: scale * 0.28)
        center.append(centerDisc.smoothed())
    }

    // MARK: - Dried spray (.driedSpray): wiry brown stems + sprays of tiny mauve blooms

    /// The dried-flower palette — the dusty mauve / rose / plum of waxflower, statice and
    /// heather once they have dried. Stems are a dead twig brown, NOT the bouquet's green.
    public static let driedPalette: [Vec3] = [
        Vec3(0.70, 0.40, 0.52),   // dusty mauve
        Vec3(0.58, 0.28, 0.40),   // plum
        Vec3(0.84, 0.60, 0.68),   // faded rose
    ]
    public static let driedStemColor = Vec3(0.34, 0.22, 0.13)

    /// A dried arrangement: MANY wiry stems (2 mm, brown — emitted as a bloom-coloured group so
    /// the bridge does not paint them foliage-green) fanning wide from the vessel mouth, each
    /// carrying a few short side branchlets whose ends hold sprays of tiny four-petal blooms.
    /// The read is a cloud of small colour on a haze of twigs, the commonest vase dressing in
    /// interior photographs. A handful of small olive leaves low on the stems keeps the
    /// arrangement's `foliage` group real (every style yields foliage). Dims from `params`:
    /// stem height from `plantSize`, stem count from `foliageDensity`, all seeded.
    public static func driedSprayMesh(params: some PottedPlantGeometry, soilY: Double, rng: inout SplitMix)
        -> BouquetParts {
        var greens = Mesh3()
        var stems = Mesh3()
        var petalMeshes = [Int: Mesh3]()
        let size = params.plantSize
        let density = Double(params.foliageDensity)
        let stemCount = max(10, min(30, Int((18.0 * density).rounded())))
        let mouthR = params.resolvedVessel == .vase ? params.potRadius * 0.5 : params.potRadius * 0.6
        let golden = Phyllotaxis.goldenAngle
        let startAngle = rng.unit() * 2 * Double.pi
        let paletteStart = Int(rng.next() % UInt64(driedPalette.count))

        for i in 0 ..< stemCount {
            let azimuth = startAngle + Double(i) * golden + (rng.unit() - 0.5) * 0.5
            let outX = cos(azimuth), outZ = sin(azimuth)
            // Dried stems splay wider than a fresh bouquet and vary more in length.
            let lean = 0.12 + rng.unit() * 0.42
            let stemH = max(0.06, size * (0.55 + rng.unit() * 0.45))
            let baseR = mouthR * (0.15 + rng.unit() * 0.85)
            let base = Vec3(outX * baseR, soilY + 0.004, outZ * baseR)
            let dir = normalize3(Vec3(outX * sin(lean), cos(lean), outZ * sin(lean)))
            let tip = base + dir * stemH
            appendStemTube(&stems, from: base, to: tip, rBase: 0.0022, rTip: 0.0012, sides: 4)

            // A few dried leaves low on some stems — the arrangement's foliage group.
            if i % 5 == 0 {
                let lp = base + dir * (stemH * (0.2 + rng.unit() * 0.2))
                emitBladeAt(&greens, pos: lp, azimuth: azimuth + (rng.unit() < 0.5 ? 1.4 : -1.4),
                            tilt: 1.1 + rng.unit() * 0.3, length: size * 0.09, width: size * 0.025,
                            silhouette: .blade)
            }

            // Side branchlets in the top 45 % of the stem, each ending in a spray of tiny blooms;
            // the stem tip carries one too.
            let pIdx = (paletteStart + i) % driedPalette.count
            var pm = petalMeshes[pIdx] ?? Mesh3()
            let branchlets = 1 + Int(rng.next() % 2)
            var sprayEnds: [(Vec3, Vec3)] = [(tip, dir)]
            for _ in 0 ..< branchlets {
                let along = 0.55 + rng.unit() * 0.4
                let origin = base + dir * (stemH * along)
                let bAz = rng.unit() * 2 * Double.pi
                let side = normalize3(Vec3(cos(bAz), 0, sin(bAz)))
                let bDir = normalize3(dir * 0.7 + side * 0.5)
                let bLen = size * (0.06 + rng.unit() * 0.06)
                let end = origin + bDir * bLen
                appendStemTube(&stems, from: origin, to: end, rBase: 0.0012, rTip: 0.0008, sides: 3)
                sprayEnds.append((end, bDir))
            }
            for (at, face) in sprayEnds {
                let blooms = 2 + Int(rng.next() % 2)
                for b in 0 ..< blooms {
                    let jitter = Vec3(rng.unit() - 0.5, rng.unit() - 0.5, rng.unit() - 0.5) * (size * 0.02)
                    let faceDir = b == 0 ? face : normalize3(face + jitter * 40)
                    emitSprig(petals: &pm, at: at + jitter, faceDir: faceDir,
                              scale: size * (0.018 + rng.unit() * 0.012), rng: &rng)
                }
            }
            petalMeshes[pIdx] = pm
        }
        var blooms = [ColoredGroup]()
        if !stems.isEmpty { blooms.append(ColoredGroup(color: driedStemColor, mesh: stems)) }
        for idx in petalMeshes.keys.sorted() {
            if let pm = petalMeshes[idx], !pm.isEmpty { blooms.append(ColoredGroup(color: driedPalette[idx], mesh: pm)) }
        }
        return BouquetParts(greens: greens, blooms: blooms)
    }

    /// One tiny dried bloom: two crossed double-sided cards standing off the sprig — eight
    /// triangles. A 5 mm waxflower is a dot of colour at any distance a camera sees a vase from;
    /// a cupped hero-tier petal here costs 30× the triangles for nothing the eye can read.
    public static func emitSprig(petals: inout Mesh3, at pos: Vec3, faceDir: Vec3, scale: Double, rng: inout SplitMix) {
        let n = normalize3(faceDir)
        let ref = abs(n.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)
        let u = normalize3(cross3(ref, n)), v = normalize3(cross3(n, u))
        let phase = rng.unit() * Double.pi
        for k in 0 ..< 2 {
            let a = phase + Double.pi * 0.5 * Double(k)
            let w = normalize3(u * cos(a) + v * sin(a))
            let half = w * (scale * 0.5), top = n * scale
            let p0 = pos - half, p1 = pos + half, p2 = pos + half + top, p3 = pos - half + top
            let outward = normalize3(cross3(w, n))
            petals.addQuad(p0, p1, p2, p3, outward: outward)
            petals.addQuad(p3, p2, p1, p0, outward: Vec3(-outward.x, -outward.y, -outward.z))
        }
    }

    /// A thin tapered, gently BENT green stem from `a` toward `b` — a real round swept tube (the
    /// `.flowers` bouquet's replacement for `appendStemTube`'s hard-edged straight prism, which
    /// flat-shaded into visible ridges, the "zucchini stalk" read). Bows sideways toward `bendDir`
    /// (projected flat if it isn't already horizontal), with the curvature concentrated in the TOP
    /// half — a real cut stem stays fairly stiff near its cut base and leans over near the bloom.
    /// `bendAmount` is the peak sideways offset as a fraction of the straight `a`→`b` length (0 =
    /// ruler-straight). Built on its OWN `Mesh3` via `Mesh3.sweep`, then `.smoothed()` BEFORE
    /// appending into `mesh` — smoothing after the append would blend normals across whatever
    /// unrelated geometry is already there (CLAUDE.md "smooth per PART, before parts are
    /// appended"). Returns the actual (curved) tip position + the tip's tangent direction, so the
    /// caller can seat a bloom exactly where the bent stem ends and facing the way it curved.
    @discardableResult
    public static func appendBentStemTube(_ mesh: inout Mesh3, from a: Vec3, to b: Vec3,
                                   rBase: Double, rTip: Double,
                                   bendDir: Vec3, bendAmount: Double,
                                   sides: Int = 6) -> (tip: Vec3, tangent: Vec3) {
        let axis = b - a
        let len = len3(axis)
        guard len > 1e-5 else { return (b, Vec3(0, 1, 0)) }
        let bow = len3(bendDir) > 1e-6 ? normalize3(bendDir) : Vec3(1, 0, 0)

        let stations = 3
        var path: [Vec3] = []
        var scales: [Double] = []
        for i in 0 ... stations {
            let t = Double(i) / Double(stations)
            // t² concentrates the bow near the tip; zero at the base by construction.
            path.append(a + axis * t + bow * (len * bendAmount * t * t))
            scales.append(1.0 + (rTip / rBase - 1.0) * t)
        }

        var part = Mesh3()
        guard part.sweep(profile: .circle(radius: rBase, segments: sides), along: path, scales: scales)
        else { return (b, normalize3(axis)) }
        mesh.append(part.smoothed())

        let tip = path[path.count - 1]
        let tangent = normalize3(tip - path[path.count - 2])
        return (tip, tangent)
    }

    /// A thin tapered GREEN stem tube from `a` to `b` (a swept few-sided prism). Built directly
    /// (not `revolve`, which is axis-aligned) so a leaning stem stays cheap + winding-safe. Closed
    /// enough to read solid; a handful of sides keeps the tri count tiny.
    public static func appendStemTube(_ mesh: inout Mesh3, from a: Vec3, to b: Vec3,
                               rBase: Double, rTip: Double, sides: Int = 5) {
        let axis = b - a
        let len = len3(axis)
        guard len > 1e-5 else { return }
        let dir = axis / len
        // Perpendicular basis.
        let ref = abs(dir.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)
        let u = normalize3(cross3(ref, dir))
        let v = normalize3(cross3(dir, u))
        let twoPi = Double.pi * 2
        func ring(_ center: Vec3, _ r: Double) -> [Vec3] {
            (0 ..< sides).map { i in
                let ang = twoPi * Double(i) / Double(sides)
                return center + (u * cos(ang) + v * sin(ang)) * r
            }
        }
        let r0 = ring(a, rBase), r1 = ring(b, rTip)
        for i in 0 ..< sides {
            let j = (i + 1) % sides
            let ang = twoPi * (Double(i) + 0.5) / Double(sides)
            let radial = normalize3(u * cos(ang) + v * sin(ang))
            mesh.addQuad(r0[i], r0[j], r1[j], r1[i], outward: radial)
        }
        // Caps so the stem tube is a closed positive-volume solid (base + tip fans).
        let cA = a, cB = b
        for i in 0 ..< sides {
            let j = (i + 1) % sides
            mesh.addTriangle(cA, r0[j], r0[i], outward: Vec3(-dir.x, -dir.y, -dir.z))
            mesh.addTriangle(cB, r1[i], r1[j], outward: dir)
        }
    }

    /// Place one green blade/leaf directly at `pos` (used for the bouquet stem-leaves) — a thin
    /// version of `emitBlade` that takes an explicit position (not a stem-relative petiole).
    public static func emitBladeAt(_ mesh: inout Mesh3, pos: Vec3, azimuth: Double, tilt: Double,
                            length: Double, width: Double, silhouette: PlantLeafCard.Silhouette) {
        let outX = cos(azimuth), outZ = sin(azimuth)
        let up = cos(tilt), out = sin(tilt)
        let yAxis = normalize3(Vec3(outX * out, up, outZ * out))
        let xAxis = normalize3(Vec3(-outZ, 0, outX))
        let cardN = normalize3(cross3(yAxis, xAxis))
        let bentN = normalize3(Vec3(outX * out, up * 0.5 + 0.45, outZ * out) * 0.7 + cardN * 0.3)
        var card = Mesh3()   // smooth PER leaf before merging — see `emitBlade`'s doc
        PlantLeafCard.emit(&card, pos: pos, xAxis: xAxis, yAxis: yAxis, normal: bentN,
                           w: width, h: length, silhouette: silhouette)
        mesh.append(card.smoothed())
    }

    // MARK: - Seeded RNG (SplitMix64) — deterministic per (seed, params)

    /// A tiny, deterministic PRNG. Same construction as `TreeScatter.SplitMix`; kept local so the
    /// plant's variation is reproducible and never reaches for an unseeded `.random`.
    public struct SplitMix {
        public var state: UInt64
        public init(_ seed: UInt64) { state = seed == 0 ? 0x9E3779B97F4A7C15 : seed }
        public mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        /// A double in [0, 1).
        public mutating func unit() -> Double { Double(next() >> 11) * (1.0 / 9007199254740992.0) }
    }
}
