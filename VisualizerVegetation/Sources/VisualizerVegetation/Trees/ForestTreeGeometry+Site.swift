import Foundation
import simd

// Per-tree site sampling (TreeSite/makeSite) and the addTree orchestrator.
extension ForestTreeGeometry {
    public struct TreeSite {
        public var pos: SIMD3<Float>
        public var species: Species
        public var height: Float
        public var trunkR: Float
        public var canopyR: Float     // crown radius estimate (ground litter / darkening)
        public var baseYaw: Float
        public var age: Float
        public var leafDensity: Float
        public var detail: Float      // 1 = hero (full tessellation), <1 = background LOD
        public var lean: Float        // trunk lean from vertical, radians (per-tree variation)
        public var leanAz: Float      // azimuth the lean tilts toward
        public var seed: UInt64
        // True when this tree's crown reaches within ~13 m of a review camera. The
        // crown-VOLUME fill always emits the cheap 2-tri card (detail 0 → far tier),
        // which is ENLARGED 1.85× — fine at 25 m but reads as a regular grid of
        // plate-sized triangles when a flank/cluster tree sits in the hero/low
        // foreground. For camera-near trees the fill keeps the cheap 2-tri card but
        // SKIPS the enlarge (true single-leaf scale) so it reads as foliage texture,
        // not plates — the 2.7M-leaf vertex budget is untouched (still 2 tris/card).
        public var nearCamera: Bool = false
        /// **PHOTOREALISM #9 / D3 — THE leaf-density knob, one meaning everywhere.**
        ///
        /// Multiplies EVERY leaf population this tree emits: the crown-volume fill's cluster
        /// count (`fillCrownVolume`) and the per-twig phyllotaxis node count (`emitLeaves`).
        /// 1.0 = the generator's full authored density; the app ships
        /// `HouseRenderBridge.treeFoliageScale`.
        ///
        /// It exists as SITE state rather than as a static because the static was the bug: the
        /// caller's value reached the generator only through `leafDensity`, which feeds a count
        /// that is immediately clamped, while the crown fill read `foliageDensityScale` directly
        /// and never saw the caller at all — so `treeFoliageScale` 0.30 → 0.55 changed the bake
        /// by 0 triangles on 8 of 9 measured trees. Gate:
        /// `ForestTreeValidityTests.testFoliageScaleIsALiveLeafDensityKnob`.
        ///
        /// Defaulted to the global so a call site that does not care keeps today's behaviour.
        public var foliageScale: Float = ForestTreeGeometry.foliageDensityScale
    }

    // ===== makeSite (lines 1485-1565) =====
    // Adapted for Daydream: `groundY` is supplied by the caller (the yard baker
    // resolves ground height off Daydream's terrain heightfield or flat plane),
    // replacing the Forest scene's `groundHeightApprox`. Made `internal static`
    // (was `private`) so `ForestTreeBaker` can drive per-tree placement directly.
    public static func makeSite(p: SIMD3<Float>, sp: Species, h: Float,
                                 detail: Float, lean: Float = 0, groundY: Float = 0,
                                 rng: inout ForestRNG) -> TreeSite {
        // trunkR — DBH/height ratio is STRONGLY height-correlated so the stand
        // reads as a range of trunk gauges (thick boles on emergents, whips on
        // suppressed saplings), not one repeated diameter. Real broadleaf DBH
        // scales super-linearly with height: a 15 m emergent ≈ 0.018–0.022·h
        // radius (~0.28–0.34 m), a 10 m dominant ≈ 0.16–0.20 m, a 6 m understory
        // ≈ 0.07–0.10 m, a 4 m sapling ≈ a 0.04–0.06 m whip. The OLD formula's
        // weak slenderness slope (0.024→0.032) let a ±20 % jitter swamp the
        // height signal → adjacent trees of different height landed at near-equal
        // radius → "uniform trunk thickness" tell. Steepen the slope (≈2× tier
        // spread), shrink age's radius pull (it was over-fattening emergents to
        // telephone-pole gauge), and split the per-tree variation into a vigor
        // term (so SAME-height trees still differ) + a modest random jitter.
        let heightT = smoothstepf(4.0, 14.0, h)          // 0 sapling .. 1 emergent
        let age: Float = 0.25 + heightT * 0.55 + rng.unit() * 0.20
        // Dominance: tall trees are the canopy winners; bias their gauge up a
        // touch and saplings' down, on top of the height-correlated slope.
        let dominance = 0.88 + heightT * 0.20            // 0.88 sapling .. 1.08 emergent
        // Per-tree vigor so two equal-height trees differ in bole gauge.
        let vigor = 0.90 + rng.unit() * 0.22
        // Slope ≈ 0.011·h (whip) .. 0.020·h (emergent) — ~2× tier spread.
        let slenderness = 0.011 + 0.009 * heightT
        // Mild age contribution kept (gnarled old boles read slightly thicker)
        // but capped so it can't double the radius like the old ageRadiusFactor.
        let ageRadiusFactor = 0.90 + age * 0.18
        let trunkR = h * slenderness * dominance * ageRadiusFactor * vigor
        // Structural #1: crown radius scales with height AND with the size tier
        // so emergents carry broad crowns and suppressed saplings carry small
        // tight ones — the canopy silhouette staircases rather than reading as
        // one repeated globe. Per-tree jitter so no two crowns match.
        // OVERLAP FIX: boosted coefficients so adjacent crowns at typical PNW
        // spacing (~7–12 m) interpenetrate rather than leaving an air gap. The
        // old 0.28+0.12 gave 2.9–3.9 m for a 10 m tree — smaller than the
        // inter-trunk gap → lollipop sapling look. At 0.40+0.16 a 10 m tree
        // carries a 4.0–5.3 m crown → crowns touch/overlap at realistic spacing.
        let canopyR = h * (0.40 + 0.16 * heightT) * (0.88 + rng.unit() * 0.24)
        // Leaf density scales super-linearly with the size tier: a big mature
        // crown is far denser than a sapling's sparse spray. Drives the
        // staircased canopy MASS, not just the silhouette outline.
        let leafDensity = 0.45 + 1.05 * heightT + rng.unit() * 0.25
        var ground = p
        ground.y = groundY   // Daydream: caller-resolved ground height (was groundHeightApprox)

        // ── CAMERA-NEAR TIER PROMOTION (Round 3 structural #1) ────────────────
        // The far tier (detail < 0.45) emits a CHEAP 2-tri leaf card ENLARGED
        // 1.85× (farEnlarge) — fine for 25–46 m backdrop crowns whose silhouette
        // is sub-pixel, but a disaster when a flank/cluster tree assigned that
        // tier sits in a REVIEW CAMERA's foreground: the enlarged cards read as a
        // regular grid of plate-sized solid-green triangles (the dominant defect
        // in low_trianglecrown.png). The flank/side ranks land detail 0.38 and the
        // closest of them sit 8–15 m off the hero/low cameras with crowns reaching
        // to within a few metres — squarely in frame. Fix the TIER ASSIGNMENT, not
        // the far card: if this tree's CROWN SURFACE comes within `nearPromoteR` of
        // any review camera, floor its detail at the MID tier so it emits the real
        // lobed outline instead of enlarged triangles. Only a few dozen trees
        // qualify (the near flank/cluster ones), so the 2.7M-leaf vertex budget is
        // untouched: mid cards are ~16 tris but only the camera-near trees promote.
        // Crown-surface distance = horizontal distance(camera, crown centre) minus
        // canopyR; promote when that clears < nearPromoteR for any camera.
        let crownCentre2 = SIMD2<Float>(ground.x, ground.z)
        var promotedDetail = detail
        var nearCam = false
        do {
            let nearPromoteR: Float = 13.0
            for camXZ in Self.reviewCameraXZ {
                let surfDist = simd_distance(crownCentre2, camXZ) - canopyR
                if surfDist < nearPromoteR { nearCam = true; break }
            }
            // Floor the TWIG-TIP leaf tier to MID so the per-twig leaves get a real
            // lobed outline (not enlarged 2-tri cards) when in a camera's foreground.
            if nearCam && detail < 0.45 { promotedDetail = max(detail, 0.50) }
        }

        return TreeSite(pos: ground, species: sp, height: h, trunkR: trunkR,
                        canopyR: canopyR, baseYaw: rng.unit() * 2 * .pi,
                        age: age, leafDensity: leafDensity, detail: promotedDetail,
                        lean: lean, leanAz: rng.unit() * 2 * .pi, seed: rng.next(),
                        nearCamera: nearCam)
    }

    // ===== addTree+CrownLobe (lines 1623-2136) =====
    public static func addTree(_ s: inout Soup, _ site: TreeSite) {
        // NEON diagnostic isolation: FOREST_ONLY_HERO renders ONLY the hero tree
        // (others culled) so its trunk→branch→fill structure can be measured in
        // screen space with zero overlap from neighbours.
        if Soup.neonOn, ProcessInfo.processInfo.environment["FOREST_ONLY_HERO"] != nil,
           site.seed != Self.heroSeed { return }
        var rng = ForestRNG(seed: site.seed)
        let look = Self.look(for: site.species)
        let h = site.height

        // NEON probe: bin ONLY the nearest hero tree (everything emitted in this
        // call is this tree's geometry; ground/understory are separate passes).
        if Soup.neonOn && site.seed == Self.heroSeed {
            s.heroBegin(baseY: site.pos.y, topY: site.pos.y + h)
        }
        defer { if s.heroProbeActive { s.heroReport() } }

        // ── Aerial perspective (polish #58): fade the far treeline into the warm
        // horizon glow. Distance drives a haze blend that desaturates + lifts the
        // crown toward the golden-hour haze colour, so the back ranks read as
        // receding depth rather than a flat saturated scrim.
        // ROUND-7 ROOT CAUSE of the recurring "side canopy bleaches to khaki":
        // this bake was keyed to the HERO anchor only, so the side camera (45 m
        // from hero) stared point-blank at a wall of trees baked at the full
        // 0.55 lerp toward khaki — which is why albedo/exposure pulls never moved
        // it (the lerp target dominates). Key the bake to the NEAREST review
        // camera instead (same multi-camera pattern as the leaf-tier promotion
        // above): no camera sees baked haze in its own foreground, while trees
        // far from EVERY camera still recede.
        var camDist = simd_length(site.pos - cameraAnchor)
        for camXZ in Self.reviewCameraXZ {
            camDist = min(camDist, simd_distance(SIMD2(site.pos.x, site.pos.z), camXZ))
        }
        let haze = smoothstepf(16.0, 40.0, camDist) * 0.55
        let prevHazeMix = s.hazeMix
        s.hazeMix = haze
        s.hazeColor = SIMD3<Float>(0.60, 0.46, 0.30)   // warm golden-hour haze
        defer { s.hazeMix = prevHazeMix }
        // ── Per-vertex tree-wind context (#58 #1) ───────────────────────────────
        // Stamp this tree's base height, crown height, and a per-tree phase so
        // `tri()` packs hierarchical-wind weights into every vertex's tangent
        // (stiff trunk → swaying canopy + leaf flutter). The vertex shaders read
        // it and displace by the frame wind. Reset on return so ground/props sway
        // not at all.
        let prevWindActive = s.windActive
        s.windActive = true
        s.windBaseY  = site.pos.y
        s.windCrownH = max(1.0, h)
        s.windPhase  = Float(site.seed % 1024) / 1024.0 * 6.2831853
        defer { s.windActive = prevWindActive }
        // Species-coded bark marker (round 3): 2 oak / 3 birch / 4 maple. The
        // shader keeps `w > 1.5` as the wood test (all three ≥ 2), then sub-selects
        // the species-correct per-pixel bark field. Leaves override woodMark→1 in
        // tri(), so this only stamps the trunk/branch wood; foliage is unaffected.
        let prevSpecies = s.speciesMark
        switch site.species {
        case .oak:            s.speciesMark = 2.0
        case .birch:          s.speciesMark = 3.0
        case .maple:          s.speciesMark = 4.0
        // Citrus have no bespoke bark field — route them to the generic-oak wood marker (2), which
        // matches the oak-blocky plate geometry `barkPlateField` emits for any non-birch/non-maple
        // species, so the trunk shader and the trunk mesh agree. (The dense low crown hides most of
        // the bole anyway.)
        case .orange, .lemon: s.speciesMark = 2.0
        case .elderberry:     s.speciesMark = 2.0     // furrowed grey-brown: the generic plate field
        }
        defer { s.speciesMark = prevSpecies }
        // Per-tree lean: tilt the whole tree about its base toward `leanAz`.
        let leanTilt = site.lean != 0
            ? rotY(site.leanAz) * rotX(site.lean) * rotY(-site.leanAz)
            : matrix_identity_float4x4
        let world = translate(site.pos) * leanTilt * rotY(site.baseYaw)

        let ageBoost = 0.85 + site.age * 0.30
        let trunkLen = h * look.trunkHeightFrac * ageBoost
        // ── TRUNK LEAN — single source of truth ───────────────────────────────
        // The trunk mesh sways laterally with height (`ringLean`), so its TOP sits
        // at local offset (leanX·y, leanZ·y). Compute it HERE (dedicated RNG, so the
        // value is independent of trunk tessellation) and pass it into BOTH the
        // trunk mesh AND the crown/leader attachment below — otherwise the crown
        // forks on the un-leaned axis and reads disconnected from the trunk ("off
        // to the side", measured lat≈0.24 m on birch). The crown now rides the
        // leaned trunk top. (See known-issue: forest trunk/branch junction gap.)
        var leanRng = ForestRNG(seed: site.seed &+ 0x1EAF_1EAF)
        let leanMag = (leanRng.unit() - 0.5) * 2 * site.age * 0.08
        let leanDir = leanRng.unit() * 2 * .pi
        let trunkLeanX = leanMag * cos(leanDir)
        let trunkLeanZ = leanMag * sin(leanDir)
        @inline(__always) func leanOffset(_ y: Float) -> SIMD3<Float> {
            SIMD3<Float>(trunkLeanX * y, y, trunkLeanZ * y)
        }
        s.debugClass = 1
        emitTrunkMesh(&s, world: world, height: trunkLen, trunkR: site.trunkR,
                      topMult: look.trunkTopMult, age: site.age, look: look,
                      detail: site.detail, seed: site.seed,
                      leanX: trunkLeanX, leanZ: trunkLeanZ)
        s.debugClass = 0
        // Trunk-base MOSS is intentionally NOT emitted for Daydream yard trees.
        //
        // WHY: `emitTrunkMoss` piles a dense, dark, damp-green cushion+skirt mass on
        // the lower ~half of the trunk. In the Forest SCENE (its origin) that mass is
        // half-buried by a knee-high grass carpet and reads as damp bark. A Daydream
        // yard tree stands on short lawn with the whole lower trunk exposed, so the
        // moss instead reads as a fake, ugly little dark-green conifer/shrub sitting at
        // the foot of every tree (Danny, 2026-07-06 — verified by a moss-off GPU
        // capture: base cluster gone, clean bare trunk). Disabling the emission is the
        // whole fix — the base cluster was ENTIRELY this moss, not crown-fill droop.
        //
        // This is an APP-TARGET decision, not a submodule/engine one: the moss lives
        // only in this ported `ForestTreeGeometry` (and, upstream, in the sibling
        // Visualizer app's `Scenes/Forest/ForestGeometry.swift` — NOT the shared
        // VisualizerEngine submodule, which has no moss). Nothing to bump; the sibling
        // Forest scene keeps its moss because there the grass hides it. `emitTrunkMoss`
        // + `mossCoverageAt` remain defined below but are deliberately dead in this
        // port — kept so a future grass-height-aware re-enable is a one-line call.
        let trunkTopR = site.trunkR * look.trunkTopMult

        // (Central leader is emitted AFTER the crown geometry is computed, so its
        //  length can be clamped to terminate INSIDE the leaf-fill shell — see the
        //  CENTRAL-LEADER CLAMP block below. Emitting it here, before crownRY/
        //  fillRY exist, was the root cause of the bare twig poking above the
        //  canopy on the skyline.)

        // Per-tree sun-side azimuth (tree-local).
        let sunAz = rng.unit() * 2 * .pi
        let sunDir = SIMD3<Float>(cos(sunAz), 0, sin(sunAz))

        // Crown geometry — computed here so the branch loop can clamp to the
        // crown boundary (BRANCH OVERSHOOT FIX, #58).
        let crownH = max(1.5, h - trunkLen)
        // Crown centre rides the leaned trunk TOP (X/Z = trunk-top lean), so the
        // fill mass + branch clamp sit over the trunk, not the un-leaned axis.
        let crownLocalCenter = SIMD3<Float>(trunkLeanX * trunkLen,
                                            trunkLen + crownH * 0.42,
                                            trunkLeanZ * trunkLen)
        let crownWC = xf(world, crownLocalCenter)   // world-space crown centre
        // Independent ±18 % per-axis jitter breaks the cloned-lollipop silhouette
        // that the reviewer flagged: every tree was a near-identical sphere because
        // crownRX and crownRY tracked each other (both from the same site.canopyR / crownH).
        // With independent jitter, some crowns are wider than tall, some taller than
        // wide — the natural spread seen in a real broadleaf stand.
        let rxJit = 0.82 + rng.unit() * 0.36   // 0.82–1.18
        let ryJit = 0.82 + rng.unit() * 0.36   // independent — not correlated with rxJit
        let crownRX = site.canopyR * (look.isBirch ? 0.62 : 1.0) * look.crownWidthMult * rxJit
        let crownRY = crownH * (look.isBirch ? 0.52 : 0.46) * ryJit

        // Crown pivot, slightly embedded so primary collars overlap the trunk top.
        let crownEmbed = look.isBirch ? trunkTopR * 1.0 : trunkTopR * 0.35
        let crownWorld = world * translate(leanOffset(trunkLen - crownEmbed))

        // ── Leaf-fill ellipsoid (single source of truth) ─────────────────────
        // The volumetric crown fill (fillCrownVolume, below) scatters leaf cards
        // inside an ellipsoid centred at fillCY with vertical radius fillRY, out to
        // fillLimit·fillRY. Both the central leader and the birch fork tips clamp
        // against the TOP of this shell so no bare branchwork can poke above the
        // foliage on the skyline. Keep these three in lockstep with the values used
        // at the fillCrownVolume call site.
        let fillRY: Float = crownRY * 1.30
        let fillCY: Float = crownWC.y + crownRY * 0.13
        let fillLimit: Float = Self.fillLimitConst
        // ── CENTRAL LEADER (oak + maple) — clamped INSIDE the leaf MASS ───────
        // Root-cause fix: the leader was previously `h * 0.40` emitted straight up
        // from trunkLen, reaching ≈ trunkLen + 0.40h ≈ 1.05h while the leaf mass
        // topped out ≈ 0.95h — the top ~0.10h was a leafless twig on the skyline.
        //
        // IMPORTANT: the leader must terminate inside the DENSE leaf mass, not the
        // mathematical fill-ellipsoid top. The fill scatters cards out to
        // fillLimit·fillRY, but on the vertical crown axis the leaf density thins
        // well before that radius (cards there need hz≈1 AND a peak-warp), so the
        // visible foliage top sits ~crownRY (not fillRY) above the crown centre.
        // The reviewer's literal formula (crownRY·fillLimit, ~0.91h) STILL poked on
        // the sparse far/backdrop crowns in the neon debug pass — on the vertical
        // axis the scattered fill thins to near-nothing above ~0.6·crownRY, so the
        // leader has to terminate inside the DENSE core, not at the nominal shell.
        // crownRY·0.55 lands the tip ~0.74h — well buried under even the thin far
        // canopy — while the leader still carries the trunk axis up into the crown.
        //
        // BARE-ARMATURE SPIKE FIX (#21 item 2 — three-quarter upper-left tree):
        // the leader is emitted as a PLAIN emitBranchMesh and does NOT pass through
        // growBranchSoup's lobe-gate clamp or its primary-tip rosette — its only
        // protection is this precomputed length. Root cause of the regression: the
        // 0.55·crownRY tip is buried only WHEN the per-tree vertical lobe warp puts
        // a PEAK at the top; on a tree whose lobePhase2 puts a GAP on the +Y axis,
        // the fill (warp<1 there) pulls inward and the dense leaf top drops below
        // 0.55·crownRY → the leader's bare top pokes the skyline. This is the same
        // leader-terminates-above-shell class fixed before, just exposed on the one
        // tree whose vertical warp rolled a top-gap. Two-part robust fix matching the
        // primary path: (a) pull the leader tip deeper into the GUARANTEED-dense core
        // (0.55→0.40·crownRY — well below the worst-case vertical thinning), and
        // (b) cap the leader tip with the SAME emitTipRosette tuft the primaries use,
        // so even a residual top-gap tree has its wood curtained on the silhouette.
        let leaderTipLocalY = crownLocalCenter.y + crownRY * 0.40
        if !look.isBirch {
            let leaderLen = max(0, leaderTipLocalY - trunkLen)
            if leaderLen > 0.05 {
                let leaderWorld = world * translate(leanOffset(trunkLen))
                s.debugClass = 2
                emitBranchMesh(&s, world: leaderWorld, length: leaderLen,
                               bottomR: trunkTopR, topR: trunkTopR * 0.18,
                               collarMult: 1.0,
                               radialSegments: lod(14, site.detail),
                               curveAmount: 0, curveAxis: SIMD3(1, 0, 0),
                               look: look, baseY: trunkLen, seed: site.seed)
                s.debugClass = 0
                // Curtain the leader tip (+Y straight up) with a leaf tuft so no
                // bare leader wood can read on the skyline even when the vertical
                // fill thins at the top — same mechanism as the primary-tip rosette.
                let leaderTipWorld = leaderWorld * translate(SIMD3(0, leaderLen, 0))
                s.debugClass = 2
                emitTipRosette(&s, tipWorld: leaderTipWorld,
                               leafSize: look.leafCardSize,
                               look: look, site: site, rng: &rng)
                s.debugClass = 0
            }
        } else {
            // ROUND-5 polish #6: several hero-mid BIRCHES read as truncated stubs —
            // the clean upper trunk pokes BARE through the top of its forked crown
            // ("crown floating / absent"). Birch (forking, no central leader) was
            // exempted from the leader-tip rosette, so when the fork crown thins at
            // the top the trunk wood shows on the skyline. We do NOT add a central
            // leader (birch forks) and we do NOT touch the crown fill or growBranchSoup
            // — we only CURTAIN the bare trunk top with a leaf tuft (the same rosette
            // the primaries use) at the trunk apex, so no bare birch wood reads above
            // the crown. A few tufts up the last stretch of trunk so the curtain spans
            // the gap between trunk top and the fork mass, not just a dot at the apex.
            let apexWorld = world * translate(leanOffset(trunkLen))
            s.debugClass = 2
            emitTipRosette(&s, tipWorld: apexWorld,
                           leafSize: look.leafCardSize,
                           look: look, site: site, rng: &rng)
            // One more tuft a touch below the apex so the curtain has vertical extent
            // and reliably hides the bare upper-trunk stretch where the fork thins.
            let subApex = world * translate(leanOffset(trunkLen) - SIMD3(0, crownRY * 0.18, 0))
            emitTipRosette(&s, tipWorld: subApex,
                           leafSize: look.leafCardSize,
                           look: look, site: site, rng: &rng)
            s.debugClass = 0
        }

        let primaryCount = look.primaryCountMin
            + Int(rng.unit() * Float(look.primaryCountMax - look.primaryCountMin + 1))
        for i in 0..<primaryCount {
            let baseYaw = Float(i) / Float(primaryCount) * 2 * .pi
                + (rng.unit() - 0.5) * 0.45
            var pitch = look.primaryPitchMin + rng.unit() * look.primaryPitchRange
            if look.lobePitchBoost > 0 && rng.unit() < 0.30 {
                pitch = min(pitch + look.lobePitchBoost, .pi * 0.88)
            }
            let primaryAxis = SIMD3<Float>(sin(baseYaw), 0, cos(baseYaw))
            let sunDot = simd_dot(primaryAxis, sunDir)
            let lengthSunBias = 1.0 + sunDot * 0.20
            pitch = max(0.20, pitch - sunDot * 0.18)
            let limbLen = trunkLen * (0.58 + rng.unit() * 0.26) * lengthSunBias
            let limbBaseR: Float
            if look.isBirch {
                let n = Float(max(2, primaryCount))
                limbBaseR = trunkTopR / sqrt(n) * 1.10
            } else {
                limbBaseR = trunkTopR * 0.78
            }
            let limbTipR = limbBaseR * 0.50
            let primaryCollar: Float = look.isBirch ? 1.55 : 1.40
            // BUDGET FIX: LOD-based crownDepth to enforce per-tier triangle budgets.
            // Root-cause: 250 trees × 336 branches/tree × 80-192 tris/branch =
            // 6.7M-16M branch tris — exhausting the triCap before any backdrop/
            // cap trees get geometry, leaving the treeline invisible.
            // Fix: reduce recursion depth by LOD tier:
            //   detail ≥ 0.90 → full species depth (3 for oak, 4 for birch)
            //   detail 0.45-0.89 → depth 2 (primary + secondary + tip clusters)
            //   detail < 0.45  → depth 1 (primary arms + tip clusters only)
            //
            // Occlusion math for far trees (depth=1, 35m, 5m crown, 1.85× cards):
            // 30 twig clusters × 20 nodes × 0.027m² = 16.2m² leaf area in a
            // 60m³ crown → 0.27m²/m³. Over 5m depth: 1 - (1-0.27)^5 = 78%.
            // Enough for a solid backdrop silhouette without consuming the budget.
            let crownDepth: Int
            if site.detail >= 0.9 {
                crownDepth = look.maxDepth     // 3 for oak, 4 for birch (full)
            } else if site.detail >= 0.45 {
                crownDepth = 2                 // mid-tier: primary + secondary
            } else {
                crownDepth = 1                 // far LOD: primary arms + tip tufts
            }
            growBranchSoup(&s, parentWorld: crownWorld, length: limbLen,
                           bottomR: limbBaseR, topR: limbTipR, pitch: pitch,
                           yaw: baseYaw, roll: (rng.unit() - 0.5) * 0.3,
                           collarMult: primaryCollar, depth: 1,
                           maxDepth: crownDepth,
                           site: site, look: look, sunDir: sunDir,
                           crownCenter: crownWC, crownRX: crownRX, crownRY: crownRY,
                           rng: &rng)
        }

        // ── VOLUMETRIC CROWN FILL (rebuild verdict #58) ───────────────────────
        // The reviewer cited: "the foliage approach — sparse small leaf cards hung
        // on a recursive branch armature — does not produce leaf-mass canopy."
        // Root cause confirmed: emitLeaves places cards along short (~0.2 m) twig
        // segments and leaves the crown INTERIOR entirely empty. No orientation
        // or node-count tuning can fill what the mechanism structurally cannot.
        //
        // Fix: fill the crown ELLIPSOID directly with Halton-distributed cluster
        // cards, independent of the branch structure. The branch skeleton (trunks
        // + primary arms + twig clusters) stays for structural filigree; the
        // OPACITY now comes from this volume fill.
        //
        // Occlusion math (hero oak, h=10m, boosted canopyR≈4.8m):
        //   Projected area: π×4.8² ≈ 72 m². 5000 clusters × 5 leaves × (0.085m)²
        //   × 1.3 aspect ≈ 234 m² leaf area → LAI 3.3.
        //   Horizontal opacity: 1 − e^(−0.5×3.3) ≈ 81 %. ✓ Single-tree.
        //
        // DENSITY FIX (rebuild verdict): the "few large cards" bet fought itself:
        //   big cards → sparse sky gaps AND cut-paper edge-on simultaneously.
        //   Correct approach: MANY SMALL cards (≤0.10 m, single-leaf scale)
        //   so opacity comes from accumulated overlap and no single card reads
        //   as a flat plane from any direction. fillScale dropped to 0.85
        //   (from 2.5–3.5), cluster counts raised 6–17×. Shell bias removed.
        // crownH/crownWC/crownRX/crownRY already computed above before the branch loop.
        // Cluster counts calibrated for ≥77 % single-viewpoint opacity.
        // fillCrownVolume always emits flat 2-tri cards (detail 0.0) so the
        // budget is ~10× cheaper than the 24-tri silhouette path — which lets
        // cluster counts be raised to fill opacity correctly.
        //
        // Effective LAI derivation (single fixed viewpoint):
        //   projected area per 0.085 m card, avg cos(θ)≈0.5 → 0.00722×0.5=0.00361 m²
        //   hero crown area π×4.8²≈72 m²; near wall π×4.1²≈53 m²
        //   hero:  12000×5×0.00361/72  = LAI 3.0 → opacity 78 % ✓
        //   near:  6000×4×0.00361/53   = LAI 1.6 → opacity 55 %; 2-rank→80 % ✓
        //   mid:   4000×3×0.00361/crown = ~LAI 1.3 → 3-rank wall ≥85 % ✓
        //   far:   3000×2×0.00361/crown = ~LAI 0.8 → 4-rank wall ≥80 % ✓
        //
        // Budget: hero 8×12000×5×2=960K, near 16×6000×4×2=768K,
        //         mid 19×4000×3×2=456K, far 60×3000×2×2=720K,
        //         side ranks ~44×3000×3×2=792K → ~3.7 M fill tris total (safe)
        // SHADOW-SIDE FIX (priority-2): the unlit three-quarter view read the
        // crown SILHOUETTE EDGES as branchy because the shell density there was
        // just thin enough to let bare twig poke through against the bright sky.
        // Bump the near-wall + mid tiers ~15% so the edge mass thickens and the
        // twig filigree is occluded — the trees the three-quarter angle frames are
        // exactly the detail 0.35–0.55 tiers. Hero stays at 12000 (already opaque);
        // backdrop/cap stay at 3000 (sub-pixel, no edge filigree visible). Adds
        // ~16×900 + 19×600 = ~26K clusters → well within the 7 M tri cap.
        // SHADOW-SIDE STRUCTURAL FIX (#20 items 1+2): the three-quarter / side /
        // TOP review angles showed the bare twig armature silhouetted THROUGH and
        // ABOVE the mid/far/backdrop crowns. Root cause: those tiers ran an
        // effective LAI of only 0.8–1.3 (vs hero 3.0), so the volumetric fill
        // never reached the Beer-Lambert opacity needed to curtain the tertiary
        // twig skeleton — the cheap flat cards were simply too sparse. Per the
        // reviewer's directive, GROW THE CHEAP FILL CARDS rather than deepen
        // recursion: bump the mid-far + backdrop/cap cluster counts (and the
        // backdrop leavesPerCluster 2→3, below) so effective LAI climbs toward
        // ~2.0–2.4. The fill ellipsoid (fillRY ×1.16, fillLimit 0.92) already
        // reaches the crown shell, so the extra cards land exactly on the exposed
        // twig hemisphere and occlude it from every angle, including straight down.
        // BUDGET: the backdrop/far/cap/side-deep tiers are ~110 trees. At
        // 4200×3=12600 leaves/tree ×2 tris = 25.2K tris/tree → ~2.8M tris for the
        // whole far mass — added to the ~3.7M near/hero fill + branch tris this
        // sits near but under the 7M triCap. If the cap is hit (trees emitted late
        // get dropped), the next lever is the shadow-hemisphere interior shell, not
        // a blanket count raise. Measured each render against the printed tri count.
        let baseClusters: Int
        if site.detail >= 0.85 {
            baseClusters = 12000  // hero: 77% single-viewpoint opacity
        } else if site.detail >= 0.50 {
            baseClusters = 6900   // near wall: thicker edge mass (was 6000)
        } else if site.detail >= 0.35 {
            baseClusters = 5800   // mid-far: LAI ~1.65 to curtain twigs (was 4600)
        } else {
            baseClusters = 4200   // backdrop/cap (25–46 m): LAI ~1.7 w/ 3 leaves (was 3000)
        }
        // Scale by projected crown area so LAI is consistent across tree sizes.
        // Without this, small-crown understory saplings (crownRX ≈ 1.5 m) get
        // 3–4× the leaf density of hero oaks (crownRX ≈ 4.8 m) and read as
        // opaque topiary spheres instead of foliated small trees.
        // Formula: hero reference is 12000 clusters @ canopyR 4.8 m (LAI 3.0).
        //   scaled = 12000 × (crownRX/4.8)² keeps LAI constant across sizes.
        //   cap at baseClusters so large/near trees don't exceed their tier budget.
        //   hero 4.8 m → 12000; near wall 4.1 m → 8760, capped at 6000;
        //   mid 2.5 m → 3246; understory 1.5 m → 1172; sapling 0.8 m → 333.
        // Cluster count must NOT track the per-axis crownRX jitter (rxJit): a tree
        // that happened to roll a wide rxJit would get proportionally more leaf
        // mass and read denser than its same-size neighbour. Use the UNJITTERED
        // crown radius so density is a stable function of tree size; rxJit only
        // reshapes the silhouette (via rxEff/ryEff in the fill), not its opacity.
        let heroRef: Float = 4.8 * 4.8
        let heroBase: Float = 12000
        let crownRBase = site.canopyR * (look.isBirch ? 0.62 : 1.0) * look.crownWidthMult   // crownRX sans rxJit
        let clusters: Int
        if site.detail >= 0.5 {
            // Near tiers (hero + near wall): scale by projected crown area so LAI
            // stays constant across tree sizes; cap at the per-tier budget.
            // ROUND 2 SPIRE FIX — RAISE THE FLOOR. The old `max(80, …)` let a SMALL
            // mid-detail tree (h≈5–6 m, det 0.50–0.55) drop to ~80–1200 clusters.
            // At that density the volumetric fill could not bury the tree's own
            // leader+primary whorl armature, so the discrete tip rosettes read as a
            // tiered CONIFER SPIKE in the clearing gap (the "faceted conifer" tell).
            // A small broadleaf whip still carries a dense leafy crown in life; floor
            // the fill at 2000 clusters so even the smallest resolvable tree presents
            // a continuous leaf mass, not a bare tiered armature. (The conifer-ring
            // tell is fixed primarily by damping the rim-seal shell band for small
            // crowns; this floor just guarantees enough core mass to merge.) The
            // area-scale + baseClusters cap still govern larger trees.
            clusters = max(2000, min(baseClusters,
                                     Int(heroBase * crownRBase * crownRBase / heroRef)))
        } else {
            // Wall / backdrop tiers: a solid mass of canopy whose only job is to
            // close the sky. Area-scaling them down (small understory crowns) makes
            // the treeline sparse and lets sky leak through — keep full baseClusters.
            clusters = baseClusters
        }
        // CAMERA-NEAR FILL DENSITY (round 3 structural #1): a camera-near tree's
        // fill cards are no longer enlarged 1.85× (they read as plate triangles in
        // the foreground), so each card now covers ≈1/3 the projected area. Multiply
        // the cluster count ≈2.6× to RESTORE crown opacity with the smaller cards,
        // otherwise the de-plated crown develops sky gaps. Only a few dozen trees
        // qualify (the near flank/cluster ones) so the tri budget holds: ≈3 dozen ×
        // ~6000 clusters × 3 leaves × 2 tris ≈ 1.3 M extra tris, well under the cap.
        let clustersRaw = site.nearCamera ? Int(Float(clusters) * 2.6) : clusters
        // Apply the density knob. **This read `Self.foliageDensityScale` — the STATIC — until
        // 2026-08-09 (PHOTOREALISM #9 / D3), which is why `HouseRenderBridge.treeFoliageScale`
        // was a dead control: the biggest leaf population in the tree never saw the caller's
        // value.** It is now `site.foliageScale`, whose default IS that static, so a call site
        // that passes nothing is unchanged. Floored at 1 so a tree never drops to zero leaves
        // and reads as a bare armature.
        let clustersN = max(1, Int((Float(clustersRaw) * site.foliageScale).rounded()))
        // SHADOW-SIDE STRUCTURAL FIX (#20 items 1+2, round 2): the side + top +
        // three-quarter angles STILL showed BARE upper limbs poking through the top
        // of the leaf mass. Closed-form root cause: the branch crown-gate clamps
        // primaries to lobeLimit = max(0.65, 1.12·lWarp) — i.e. branch TIPS reach
        // ~1.12·crownRY (×warp) — but the fill shell only reached 0.92·fillRY =
        // 0.92·1.16 = 1.07·crownRY. So branch tips sat ~5% OUTSIDE the leaf fill in
        // the up-hemisphere → exposed twig fringe against the bright sky. Fix: make
        // the fill envelope reach the BRANCH envelope. fillRY 1.16→1.30 so
        // 0.95·1.30 = 1.235·crownRY ≥ the 1.12 branch limit (with margin for warp),
        // and the fill-shell limit (fillLimit in fillCrownVolume) raised 0.92→0.95.
        // Lift the centre a touch more (0.10→0.13) so the extra height biases UP,
        // where the bare leader + primary tips live, not down into the trunk gap.
        // Cost is redistributing the same `clusters` — no extra triangles.
        // fillRY / fillCY / fillLimit are the single-source values computed up by
        // the crown geometry (also used to clamp the central leader + birch fork
        // tips). Keep them there; this call just consumes them.
        fillCrownVolume(&s, crownCX: crownWC.x, crownCY: fillCY,
                        crownCZ: crownWC.z,
                        rx: crownRX, ry: fillRY, rz: crownRX,
                        fillLimit: fillLimit,
                        clusters: clustersN, look: look, site: site, rng: &rng)

        // ── RIPE FRUIT (DH-0570) — the citrus signature ───────────────────────
        // An orange bears round orange fruit, a lemon a crown of yellow ovoids; the broadleaves
        // bear none (`look.fruit == nil`). This is what tells the two apart, and either from an
        // oak. Emitted here, after the crown exists, so the fruit can be placed against the real
        // crown centre + radii. It merges into the ONE tree soup (no new draw instance) and rides
        // the leaf shading class (no bark shader on the fruit).
        if let fruit = look.fruit {
            emitCrownFruit(&s, crownCenter: crownWC, crownRX: crownRX, crownRY: crownRY,
                           fruit: fruit, height: h, seed: site.seed)
        }
    }
}
