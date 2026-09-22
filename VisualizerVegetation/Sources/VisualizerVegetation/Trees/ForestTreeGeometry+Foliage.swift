import Foundation
import simd

// ─────────────────────────────────────────────────────────────────────────────
// DH-0651 — the yard tree's leaf blade is stitched by THE shared constructor.
//
// This sink is the whole adapter. `LeafConstructor.emitBlade` owns the spine /
// margin / taco-fold / cantilever-droop math (one copy, shared with the headless
// plant meshes in DaydreamCore); everything below is the part that was only ever
// the TREE's — a shading normal blended 0.62 bent / 0.38 geometric, and the
// midrib→margin→tip albedo gradient. Both are pure functions of where a vertex
// sits ON the blade, which is exactly what `LeafStripVertex.v` / `.u` carry, so
// nothing had to move into the shared module to make the stitch shareable.
//
// It holds the Soup by value and is `swap`ped in/out around the emit rather than
// copying it: the soup's arrays are millions of elements by the end of a bake and
// a second live reference would COW-copy all of them on the first triangle of
// every one of ~22k leaf cards.
// ─────────────────────────────────────────────────────────────────────────────
private struct TreeLeafStitch: LeafCardSink {
    public typealias Scalar = Float

    public var soup = ForestTreeGeometry.Soup()
    public var bentNormal = SIMD3<Float>(0, 1, 0)
    public var midribCol = SIMD3<Float>(0, 0, 0)
    public var marginCol = SIMD3<Float>(0, 0, 0)
    public var tipCol = SIMD3<Float>(0, 0, 0)

    /// Midrib → margin → tip albedo, by the vertex's own place on the blade.
    private func colour(_ p: LeafStripVertex<Float>) -> SIMD3<Float> {
        let edge = ForestTreeGeometry.mixv(midribCol, marginCol, min(1, p.u * 1.4))
        // Tip whitens.
        return ForestTreeGeometry.mixv(edge, tipCol,
                                       ForestTreeGeometry.smoothstepf(0.82, 1.0, p.v))
    }

    /// Per-vertex shading normal blends the outward `bentNormal` with the strip's local
    /// geometric normal so the cup shades as a soft volume, not a flat facet. BOTH halves
    /// get the same normal: Illuminatorama shades from the supplied normals, not the
    /// cross product, so the mirrored winding must not flip the shading (see `Soup.tri`).
    public mutating func addLeafTriangle(_ a: LeafStripVertex<Float>,
                                  _ b: LeafStripVertex<Float>,
                                  _ c: LeafStripVertex<Float>,
                                  outward: SIMD3<Float>,
                                  geometricNormal: SIMD3<Float>) {
        let n = ForestTreeGeometry.safeNormalize(bentNormal * 0.62 + geometricNormal * 0.38,
                                                 fallback: bentNormal)
        soup.tri(a.position, b.position, c.position,
                 n0: n, n1: n, n2: n,
                 c0: colour(a), c1: colour(b), c2: colour(c))
    }
}

// Phyllotaxis leaf cards, crown fruit, tip curvature.
extension ForestTreeGeometry {
    /// Phyllotaxis leaf placement (port of `attachLeaves`) → diamond leaf cards.
    /// Takes the whole `site` rather than a bare `detail`: it needs BOTH the LOD tier and the
    /// tree's `foliageScale`, and passing the two separately is how they drifted apart (D3).
    public static func emitLeaves(_ s: inout Soup, world: simd_float4x4,
                                   twigLength: Float, leafSize: Float, density: Float,
                                   look: TreeLook, site: TreeSite,
                                   rng: inout ForestRNG) {
        let detail = site.detail
        let prevDbg = s.debugClass; s.debugClass = 6; defer { s.debugClass = prevDbg }
        let isOpposite = look.opposite
        // DH-0662: `Phyllotaxis.goldenAngleF`, not the literal 2.39996 this used to carry —
        // which was a truncation 20x coarser than Float itself, and 3.2e-06 rad off the value
        // `+Crown.swift` used for the SAME spiral. One derived constant, no third spelling.
        let spiralAngle: Float = isOpposite ? (.pi / 2) : Phyllotaxis.goldenAngleF
        let leavesPerNode = isOpposite ? 2 : 1
        // Polish (#58): tighter internode spacing → leaves pack densely ALONG
        // the twig arc-length (leaf-dense branchlets) rather than relying on
        // bigger cards or a fat radial spray. Halved vs the prior frac so the
        // smaller cards recover crown density by count, not size.
        let internodeLen = max(0.018, leafSize * look.internodeFrac * 0.5)
        let basalSkip: Float = 0.08
        let usableLen = twigLength * (1.0 - basalSkip)
        // D3 — the twig leaf count, at FULL density, then scaled by the ONE knob.
        //
        // Clamping first and scaling second is the whole point. The old form scaled `density`
        // and clamped afterwards, so on every twig whose raw count already reached the cap the
        // multiply was thrown away — and on a real oak `rawNodes` sits at ~20, exactly the cap,
        // which is how a 1.83× density change produced a 0-triangle diff. Both terms now live in
        // "full density" units, so `min(cap, raw) × scale` is linear in the knob everywhere the
        // 2-node floor is not binding, and reproduces the historical count at the 0.30 default
        // (min(66, raw/0.30) × 0.30 ≡ min(19.8, raw)).
        let rawNodesFull = usableLen / internodeLen * density * leafDensityScaleAtFullDensity
        let nodesFull = min(Float(leafNodeCapAtFullDensity), rawNodesFull)
        let nodeCount = max(2, Int((nodesFull * site.foliageScale).rounded()))
        let petBase = leafSize * look.petioleFrac
        let startAngle = rng.unit() * 2 * .pi
        let planeW = leafSize * look.leafAspectW
        let planeH = leafSize * look.leafAspectH

        // ── RACHIS-OCCLUSION (top-down fern fix) ─────────────────────────────
        // The top-down angle combed into pinnate fern fronds because leaves were
        // placed at an EVENLY-spaced phyllotaxis row down each twig: a bare twig
        // tube (rachis) showed between every adjacent leaf, and that radiating
        // line is the dead "botanist sees ferns" tell. Real broadleaf shoots do
        // NOT space their leaves evenly: they bunch into a dense TERMINAL ROSETTE
        // at the shoot tip with tighter sub-clusters along the shoot, and the
        // overlapping blades hide the twig from above. So instead of mapping
        // `nodeIdx` linearly onto the twig, we (1) bias the along-twig position
        // toward the TIP (pow curve), and (2) SNAP groups of consecutive leaves
        // onto shared stations with only tiny intra-station along-jitter, so each
        // station is a tight overlapping rosette that occludes the rachis between
        // stations rather than a single isolated leaf with bare twig on both
        // sides. `rosettePack` is how many leaves share one along-station; the
        // tip station gets the densest pack. Target ~3–4 stations on a short
        // twiglet and up to ~7 down a long shoot, with several overlapping blades
        // each, so the bare rachis between leaves is occluded but the shoot still
        // reads as a leafy branchlet (not one fat pom-pom). Derived from nodeCount
        // so a sparse far twig and a dense hero twig both rosette.
        let targetStations = max(2, min(7, Int((Float(nodeCount) / 5.0).rounded())))
        let rosettePack = max(2, (nodeCount + targetStations - 1) / targetStations)
        let stationCount = max(2, (nodeCount + rosettePack - 1) / rosettePack)

        // Everything emitted in this call is foliage — tag it so the deferred
        // pass can apply the leaf-transmission term. Reset on the way out.
        let prevFoliageMark = s.foliageMark
        s.foliageMark = 1
        defer { s.foliageMark = prevFoliageMark }

        for nodeIdx in 0..<nodeCount {
            // Which along-station this leaf belongs to, and tip-biased placement.
            let station = nodeIdx / rosettePack
            let stationFrac = Float(station) / Float(max(1, stationCount - 1))
            // Tip bias: pow>1 pushes stations toward the tip so the shoot END
            // carries the densest rosette (real apical leaf cluster). Base of the
            // twig stays sparse, tip packs tight.
            let tipBiased = pow(stationFrac, 1.45)
            // Small per-leaf along-jitter WITHIN a station so the rosette has a
            // little depth without smearing back into an even row. ±0.4 internode.
            let withinStation = (rng.unit() - 0.5) * 0.8
            let nodeFrac = min(1.0, max(0.0,
                tipBiased + withinStation * (internodeLen / max(usableLen, 1e-4))))
            let along = twigLength * basalSkip + usableLen * nodeFrac
            let nodeAngle = startAngle + Float(nodeIdx) * spiralAngle
            for leafIdx in 0..<leavesPerNode {
                // ── Per-cluster gaps (light-permeable canopy, issue #58 fix) ──
                // A real crown is not a packed shell: there are sky-holes and
                // thin lacy patches all over the silhouette. Stochastically
                // SKIP a fraction of leaves using a low-discrepancy (Halton-2)
                // sequence keyed to the node index, so the gaps are evenly
                // scattered (blue-noise-ish) rather than clumped. This both
                // breaks the opaque-blob read AND lets the low sun pour through
                // interior gaps to backlight the leaves that remain.
                let h2 = halton(UInt32(nodeIdx) &* 2 &+ UInt32(leafIdx), 2)
                // Combine an even (Halton) gap field with a coarse clustered
                // gap (value-noise on the node index) so the crown has BOTH
                // fine lace and larger sky-holes — a broken, light-permeable
                // silhouette rather than a uniformly-perforated shell.
                let clump = valueNoise(Float(nodeIdx) * 0.7 + startAngle * 3.0,
                                       Float(leafIdx) * 1.3)
                if h2 < 0.18 || clump > 0.80 { continue }   // ~18% even + clustered holes
                let myAngle = nodeAngle + (leafIdx == 1 ? .pi : 0)
                // Polish (#58): cards ~25% smaller so the nearest crowns read at
                // life-size (oak blade ≤0.20 m, birch ≤0.10 m per the species
                // reference) instead of the 0.3–0.4 m "dinner-plate" leaves.
                // Density is recovered by the tighter internode spacing above,
                // not by card area. Mean sizeJit 1.05 → 0.79.
                let sizeJit = 0.55 + rng.unit() * 0.48
                let wLeaf = planeW * sizeJit, hLeaf = planeH * sizeJit
                let forwardRake: Float = isOpposite ? 0.26 : 0.44
                let myPetLen = petBase * (0.85 + rng.unit() * 0.30)
                let cosA = cos(myAngle), sinA = sin(myAngle)
                let cosR = cos(forwardRake), sinR = sin(forwardRake)
                let petDir = simd_normalize(SIMD3<Float>(cosA * cosR, sinR, sinA * cosR))
                // ── Leaves HANG OFF THE TWIG, they don't orbit it (issue #58) ──
                // The "broccoli-puff" read came from flinging leaves into a big
                // spherical cloud (reach up to ~5.4× leafSize) AROUND each twig,
                // so the leaf mass floated free of any visible branchlet and the
                // backlight passed through a hollow ball. A real leaf sits at the
                // end of a short petiole right off the twig surface. Pull the
                // reach IN so each leaf clings to the branchlet: the leaf area is
                // now driven by leaf DENSITY ALONG the twig arc-length (more
                // nodes via tighter internodes), not by a fat radial spray. The
                // result is leaf-dense branchlets — dark twig filigree threading
                // THROUGH the leaves — instead of a free-floating floret.
                let rOuter = rng.unit()
                // Small, tight reach: just clear the petiole + half-leaf so the
                // blade hangs at the twig surface with a little jitter. No more
                // multi-leaf-length radial throw.
                let reach = (0.10 + rOuter * 0.55) * leafSize
                let centerOffset = myPetLen + hLeaf * 0.50 + reach
                let leafPos = SIMD3<Float>(petDir.x * centerOffset,
                                           along + petDir.y * centerOffset,
                                           petDir.z * centerOffset)
                let yAxis = petDir
                let upHint = SIMD3<Float>(0, 1, 0)
                let dotYU = simd_dot(yAxis, upHint)
                let zRaw = abs(dotYU) > 0.97
                    ? simd_normalize(simd_cross(yAxis, SIMD3<Float>(1, 0, 0)))
                    : simd_normalize(upHint - dotYU * yAxis)
                let xRaw = simd_normalize(simd_cross(yAxis, zRaw))
                // Roll the blade about the petiole — FULL RANDOM for all tiers.
                // Root-cause: leaf cards are built in the xAxis-yAxis plane, which
                // is HORIZONTAL for horizontal branches (zAxis starts at (0,1,0)).
                // With ±22° roll, all cards stayed near-horizontal → edge-on to the
                // hero camera (which looks roughly horizontally) → ~17% visible
                // projected area per card → "hollow crown" appearance.
                // With ±π (full circle), leaf normals are uniformly distributed on
                // the cylinder around the branch axis: ~1/3 face each camera
                // direction, giving 3× the perceived density for the same card count.
                // Real broadleaf leaves ARE randomly oriented — this is physically
                // correct, not a hack. Also eliminates the rachis/comb pattern.
                let rollSpread: Float = 2.0 * .pi
                let rollR = (rng.unit() - 0.5) * rollSpread
                let cr = cos(rollR), sr = sin(rollR)
                let zAxis = zRaw * cr - xRaw * sr
                let xAxis = zRaw * sr + xRaw * cr
                // World-space leaf basis (segWorld is rigid).
                let wPos = xf(world, leafPos)
                let wX = xfDir(world, xAxis)
                let wY = xfDir(world, yAxis)
                let wZ = xfDir(world, zAxis)
                // Bent foliage normal (#3): instead of the flat card normal,
                // point the leaf's shading normal up-and-outward from the twig
                // axis so the canopy shades as a soft spherical volume — sunny
                // outer leaves catch the light, interior leaves fall into shade,
                // rather than every card sharing one hard plane normal. `petDir`
                // already points outward (radially from the twig, fwd-raked);
                // bias it toward the sky and blend a little of the real card
                // normal back in for per-leaf variation. Flows to BOTH paths:
                // the deferred per-vertex normal and the RT triNormal (which is
                // the average of the supplied shading normals).
                let outwardLocal = safeNormalize(petDir + SIMD3<Float>(0, 0.45, 0), fallback: petDir)
                let wOut = xfDir(world, outwardLocal)
                let bentN = safeNormalize(wOut * 0.72 + wZ * 0.28, fallback: wOut)
                let col = leafColorVary(look, frac: nodeFrac, rng: &rng)
                // Per-leaf droop/curl (polish): pitch the blade ±10–15° off the
                // petiole so a cluster isn't a uniform fan of identically-oriented
                // blades. Rotate the midrib (yAxis) and the cup normal about the
                // leaf's xAxis (the across-blade hinge) by a small per-leaf angle.
                let droop = (rng.unit() - 0.45) * 0.52     // ≈ −13°..+16°
                let cd = cos(droop), sd2 = sin(droop)
                let wYd = safeNormalize(wY * cd + bentN * sd2, fallback: wY)
                let bentD = safeNormalize(bentN * cd - wY * sd2, fallback: bentN)
                emitLeafCard(&s, pos: wPos, xAxis: wX, yAxis: wYd, normal: bentD,
                             w: wLeaf, h: hLeaf, color: col,
                             species: look.species, detail: detail,
                             curl: look.leafCurl, rng: &rng)
            }
        }
    }

    /// A broadleaf blade with a REAL species silhouette — not a folded diamond.
    /// `replace the weak primitive, don't decorate it`: the diamond card combed
    /// into fern/conifer pinnae from the side, so a botanist read ferns. The
    /// blade is now built from a species-keyed margin OUTLINE traced as a
    /// triangle strip from a midrib spine to the lobed/serrated margin:
    ///   • oak   — pinnately lobed obovate (rounded lobes, widest past middle)
    ///   • birch — doubly-serrated triangular-ovate (saw-tooth edge, pointed)
    ///   • maple — palmate 5-lobed (pointed lobes, deep sinuses)
    /// The half-margin is a list of `(v, u)` samples (v = base→tip along midrib,
    /// u = half-width fraction); it's mirrored to both sides and stitched to the
    /// spine. The shallow taco `fold` lifts the margin toward `bentNormal` so the
    /// blade still presents area from any view and reads as a soft cup, but the
    /// LOBES — not the fold — now carry the silhouette. Vert count is kept modest
    /// (≈6 margin samples at hero, ~3 at LOD) and gated off `detail` so ~22k
    /// leaves stay in budget. A midrib→margin albedo gradient + per-leaf hue
    /// jitter + a near-white thin-edge tip break the single-paint-chip read.
    /// `// winding-ok:` native Illuminatorama soup (no SCNGeometry winding).

    // ===== emitLeaves+emitLeafCard (lines 3717-3932) =====
    public static func emitLeafCard(_ s: inout Soup, pos: SIMD3<Float>,
                                     xAxis: SIMD3<Float>, yAxis: SIMD3<Float>,
                                     normal bentNormal: SIMD3<Float>, w: Float, h: Float,
                                     color: SIMD3<Float>, species: Species,
                                     detail: Float, noEnlarge: Bool = false,
                                     curl: Float = 0,
                                     rng: inout ForestRNG) {
        // ── ONE CARD = ONE LEAF (PHOTOREALISM #9 / D7) ────────────────────────
        // Every triangle emitted below — 2 (far), 16 (mid) or 24 (hero) — is part
        // of the SAME leaf and is only a leaf together with its siblings. Stamp
        // them with a shared card id so a downstream LOD can remove the leaf
        // rather than chew it. Scoped exactly like `foliageMark`: raised here,
        // restored on every exit path (the far tier `return`s early).
        let prevCard = s.cardMark
        s.cardMark = s.nextCardID
        s.nextCardID &+= 1
        defer { s.cardMark = prevCard }
        // SUN-FACING GATE (#20 polish item 3): the bright margin/tip albedo boost
        // (the "backlit-rim" cue) was baked into EVERY leaf card regardless of
        // orientation. On the three-quarter / side review angles the rim leaves
        // whose face pointed at the camera (NOT the sun) still carried the +18%
        // tip boost and, once the sun's directional term hit them, clipped to
        // near-white chartreuse. Gate the brightening to leaves whose normal
        // actually faces the world sun: `sunFace` ∈ [0,1] is dot(N, sunDir)
        // ramped, so a shadow-side leaf keeps the cooler base green while a true
        // sun-facing leaf gets the warm-translucent rim — and we CLAMP the peak so
        // even a sun-facing tip reads warm green, not white. This is an albedo gate
        // only; the deferred thin-sheet TRANSMISSION term stays physically
        // back-face-gated in-shader (we do NOT crush exposure — that would kill the
        // god-ray shafts; we gate the term itself per memory illuminatorama_shaft).
        let sunFace = smoothstepf(0.0, 0.55, simd_dot(bentNormal, worldSunDir))
        // ROUND-5 polish #4: the side-view SUNLIT leaf mass still bleached to pale
        // khaki — the lit-hemisphere albedo compounded with the full directional sun
        // and clipped to cream. Pull the lit-hemisphere base albedo down ~15 % (a
        // geometry-side bake, gated by sunFace so only sun-facing cards darken) AND
        // bias the lit cards slightly GREENER (trim red/blue a touch more than green)
        // so the lit canopy lands a saturated warm green instead of cream khaki. This
        // does NOT touch the in-shader backlit TRANSMISSION term nor the held
        // exposure — it is an albedo-only trim on the lit cards.
        let litTrim = 1.0 - 0.15 * sunFace
        let color = color * SIMD3<Float>(litTrim - 0.04 * sunFace,
                                         litTrim,
                                         litTrim - 0.05 * sunFace)
        // Albedo gradient (shared by all tiers): deeper/cooler along the midrib
        // (thicker blade), lighter + warmer toward the margins (thinner, more
        // translucent), warm-translucent green at the sun-facing tip.
        let midribCol = color * 0.86
        // Margin boost scales with sun-facing: shadow-side margins stay near the
        // base green; sun-facing margins lighten/warm a touch.
        // ROUND-3 EXPOSURE-SPLIT FIX (target 3): on the FRONT-LIT side angle the
        // sun-facing margins/tips compounded with the full directional sun and
        // BLEACHED the canopy to cream. Per the reviewer, warm-DARKEN the foliage
        // (pull the front-lit bleach back) WITHOUT crushing global exposure: trim the
        // sun-facing boosts (margin 0.10→0.06, tip 0.18→0.11, +0.08→+0.04) and warm
        // (less blue) the tip-lift tint so the lit canopy lands a saturated warm green
        // instead of cream. The backlit rim is carried by the in-shader transmission
        // term (untouched), so this does NOT dim the SSS glow.
        let marginBoost = 1.0 + 0.06 * sunFace
        let marginCol = mixv(color, color + SIMD3<Float>(0.05, 0.08, 0.025), 0.60) * marginBoost
        // Thin-edge tip colour: only sun-facing leaves get the warm rim lift.
        // Clamp so the peak stays warm-translucent GREEN — never near-white cream.
        let tipLift = sunFace
        let thinEdge = color * (1.0 + 0.11 * tipLift)
                     + SIMD3<Float>(0.030, 0.044, 0.008) * tipLift
        let tipCol = mixv(color, thinEdge, 0.50) * (1.0 + 0.04 * tipLift)

        let tierHero = detail >= 0.85
        let tierMid  = detail >= 0.45

        // ── FAR mass (detail < 0.45) — the backdrop/cap/far-wall crowns at
        // 25–46 m, which are the VAST majority of the ~2.7M leaves and never
        // resolve as individual blades. A full margin-fan here blows the Metal
        // vertex buffer (>20 GB → makeBuffer(nil) assert), so far leaves keep the
        // cheap 2-triangle folded card — but now with the midrib→margin albedo
        // gradient so even the distant mass reads as dappled foliage, not flat
        // paint. Silhouette at that distance is sub-pixel; cost is what matters.
        if !tierMid {
            // Round the far card up toward w≈h (widen X, shorten Y a touch): a
            // tall narrow diamond reads as a NEEDLE edge-on and that's what combs
            // into fern pinnae in the top-down mass. A rounder card stays a foliage
            // fleck from any roll. ALSO enlarge the far card ~1.5×: distant crowns
            // (25–46 m) never read as "dinner-plate" leaves (the near-tree life-
            // size concern), but their TWIG filigree shows through the gaps from
            // directly overhead and radiates as a fern rachis. Bigger far cards
            // OVERLAP into a continuous textured canopy that hides the radiating
            // twigs — which is exactly how a real oak/maple reads from a drone.
            // Combined with the wide ±60° roll scatter at the call site.
            // RACHIS-OCCLUSION (top-down): the far mass is now CLUSTERED into
            // tip rosettes (emitLeaves), so the far cards bunch and overlap at the
            // shoot tips instead of stringing evenly down a visible twig. Enlarge
            // + round them further (1.5→1.85, hw 0.62→0.74) so adjacent rosette
            // cards overlap into a CONTINUOUS lobed crown surface from directly
            // above — burying the radiating twig entirely — rather than separable
            // flecks with bare twig between. Adds NO vertices (still 2 tris/card);
            // the 2.7M-leaf Metal vertex budget is untouched.
            // CAMERA-NEAR FILL (round 3 structural #1): when this card belongs to a
            // crown-volume fill of a tree in a review camera's foreground, DROP the
            // 1.85× enlarge — at 25 m the enlarge overlaps rosettes into a continuous
            // canopy, but at 3–10 m it reads as a regular grid of plate-sized solid
            // triangles (low_trianglecrown). Keep the cheap 2-tri card but at true
            // single-leaf scale (slight 1.05× so adjacent cards still overlap into
            // mass) so it reads as fine foliage texture, not plates. No vertex cost.
            let farEnlarge: Float = noEnlarge ? 1.05 : 1.85
            let hw0 = w * 0.74 * farEnlarge
            let hh0 = h * 0.80 * farEnlarge
            // TACO-FOLD on the BACKDROP/FAR card (#20 polish item 5): the far card
            // was a near-flat 2-tri quad — the two margin verts lifted along
            // bentNormal but the midrib spine (bottom/top) stayed in-plane, so
            // viewed EDGE-ON down the midrib it collapsed to a flat sliver and the
            // sparse backdrop showed cut-paper slivers on the side angle. Deepen the
            // V-cup: push the SPINE ends BACK along −bentNormal while the margins
            // lift FORWARD, so the card folds around the midrib like the hero blade.
            // Now the card presents cross-sectional area (and a real shading
            // gradient) from any roll, killing the edge-on sliver. Adds no tris.
            let spineDrop = bentNormal * (hw0 * 0.16)        // midrib sits in a valley
            let lift = bentNormal * (hw0 * 0.46)             // margins curl up (was 0.30)
            let bottom = pos - yAxis * (hh0 * 0.5) - spineDrop
            let top = pos + yAxis * (hh0 * 0.5) - spineDrop
            let left = pos - xAxis * hw0 + yAxis * (hh0 * 0.06) + lift
            let right = pos + xAxis * hw0 + yAxis * (hh0 * 0.06) + lift
            // A collapsed (collinear) card gives a zero cross → NaN; fall back to
            // the card's outward bentNormal so the half still shades outward.
            let gnR = faceToward(safeNormalize(simd_cross(right - bottom, top - bottom),
                                               fallback: bentNormal), bentNormal)
            let gnL = faceToward(safeNormalize(simd_cross(top - bottom, left - bottom),
                                               fallback: bentNormal), bentNormal)
            // Per-half normals weighted toward the geometric (cupped) normal so the
            // two halves shade as a folded blade, not a flat card.
            let nR = safeNormalize(bentNormal * 0.45 + gnR * 0.55, fallback: bentNormal)
            let nL = safeNormalize(bentNormal * 0.45 + gnL * 0.55, fallback: bentNormal)
            s.tri(bottom, right, top, n0: nR, n1: nR, n2: nR,
                  c0: midribCol, c1: marginCol, c2: tipCol)
            s.tri(bottom, top, left, n0: nL, n1: nL, n2: nL,
                  c0: midribCol, c1: tipCol, c2: marginCol)
            return
        }

        // ── HERO / MID — resolvable near trees get a REAL species outline.
        // Half-margin profile: (v, u). v∈[0,1] along midrib (0 base, 1 tip);
        // u∈[0,1] half-width (×w*0.5). Sinuses (u→small) between lobes give the
        // silhouette its broadleaf bite. Hero = full lobed/serrated outline
        // (~7 pts → 24 tris); mid = a reduced but still LOBED outline (~5 pts →
        // 16 tris) so the near wall + side flanks still read broadleaf, not
        // pinnae. Only ~hundreds of trees hit these tiers, so the tri budget holds.
        //
        // SINGLE SOURCE (DH-0651): the outline AND the strip-stitch that traces it
        // both live in `VisualizerVegetation` — `LeafSilhouette` is the margin
        // catalog, `LeafConstructor.emitBlade` is THE blade constructor, shared with
        // the headless plant/petal meshes in DaydreamCore. This bake used to read the
        // margin table from there and then re-implement the spine/margin/fold/droop
        // loop itself in Float; that second copy is what let a margin fix land in one
        // system and not the other (DH-0626 / DH-0628 / DH-0629 / DH-0647 each
        // rediscovered the same shard-edged leaf). Only the tree's OWN work is left
        // here: the sun-face albedo gate + midrib→margin→tip gradient above, and the
        // blended shading normal + per-vertex colour in `TreeLeafStitch`.
        //
        // `subdivisions: 1` keeps the authored controls untouched — the byte-identity
        // path the pinned bake counts in `ForestTreeValidityTests` depend on. Do NOT
        // raise it here without re-deriving those numbers.
        let silhouette: LeafSilhouette
        switch species {
        case .oak:            silhouette = .oak
        case .birch:          silhouette = .birch
        case .maple:          silhouette = .maple
        case .orange, .lemon: silhouette = .citrus
        case .elderberry:     silhouette = .citrus     // an entire lanceolate leaflet; the look narrows it
        }

        // fold 0.30 — the taco cup: the margin lifts toward bentNormal ∝ u so the
        // midrib sits in a valley. Tuned down from 0.34 pre-silhouette and up from a
        // too-flat 0.22: enough cup that the blade catches a real cross-surface
        // shading gradient — killing the "green cut-paper" flat-facet tell on the
        // nearest leaves — without pinching the lobes back into a fern needle. It is
        // `LeafConstructor.defaultFold`, passed explicitly so this call site still
        // states the value it was tuned to.
        //
        // `curl` is the longitudinal gravity-droop (DH-0570, the DH-0488 leaf
        // technique for trees): the blade bows out of its flat plane along
        // −bentNormal as v², a 3D arch at zero extra triangles. curl == 0 is the old
        // flat blade byte-for-byte, so the broadleaves that pass 0 are unchanged.
        //
        // `.singleSided` — one face per strip quad. The leaf draw group is registered
        // double-sided at the material level, so the second face would be pure cost:
        // at canopy scale it is the difference between a ~50k and a ~100k tri tree.
        var stitch = TreeLeafStitch(bentNormal: bentNormal,
                                    midribCol: midribCol,
                                    marginCol: marginCol,
                                    tipCol: tipCol)
        swap(&stitch.soup, &s)
        LeafConstructor.emitBlade(
            into: &stitch,
            // `pos` is the leaf centroid from emitLeaves; the constructor seats the
            // blade base 0.46·h back along −yAxis so the petiole attaches at the base.
            placement: LeafConstructor.Placement(position: pos,
                                                 xAxis: xAxis,
                                                 yAxis: yAxis,
                                                 bentNormal: bentNormal,
                                                 width: w,
                                                 height: h),
            silhouette: silhouette,
            tier: tierHero ? .hero : .mid,
            subdivisions: 1,
            fold: 0.30,
            curl: curl,
            winding: .singleSided)
        swap(&stitch.soup, &s)

        _ = rng   // reserved for future per-leaf margin jitter
    }

    /// Scatter ripe citrus fruit through the crown (DH-0570 — the orange/lemon signature).
    /// `crownCenter` is world-space (the tree's `crownWC`); `crownRX`/`crownRY` are its horizontal
    /// / vertical radii. Each fruit nestles in the OUTER-LOWER crown so it reads as hanging among
    /// the leaves, hung just below its attachment. Emitted as FOLIAGE (so it rides the leaf draw
    /// group + leaf census and NEVER the bark shader), each fruit carrying its own card id so the
    /// leaf-decimation stride removes a whole fruit rather than tearing one. No new draw instance —
    /// it merges into the single tree soup.
    public static func emitCrownFruit(_ s: inout Soup,
                                       crownCenter: SIMD3<Float>, crownRX: Float, crownRY: Float,
                                       fruit: FruitLook, height: Float, seed: UInt64) {
        var rng = ForestRNG(seed: seed &+ 0x00F7_017F_7017_F701)   // dedicated fruit stream
        // Keep fruit ~life-size: scale only mildly with tree height so a sapling isn't hung with
        // boulders and an old-growth tree isn't dotted with peas.
        let fr = fruit.radius * min(Float(1.25), max(Float(0.7), height / 6.5))
        let prevFoliage = s.foliageMark
        s.foliageMark = 1                       // ride the leaf shading class (no bark shader)
        defer { s.foliageMark = prevFoliage }
        s.debugClass = 6                        // a foliage class (cosmetic — for the hero probe)
        defer { s.debugClass = 0 }
        for _ in 0..<fruit.count {
            let az = rng.unit() * 2 * Float.pi
            let crowning = fruit.placement == .crowning
            // Hanging fruit: downward-biased polar cos, −0.68 (low skirt) .. +0.18 (just above
            // centre), a little inside the surface. A crowning flower cluster: the upper dome,
            // +0.05 .. +0.95, ON the surface where the sky (and the street) can see it.
            let cosT = crowning ? 0.05 + rng.unit() * 0.90 : -0.68 + rng.unit() * 0.86
            let sinT = (1 - cosT * cosT).squareRoot()
            let rad = crowning ? 0.90 + rng.unit() * 0.12 : 0.58 + rng.unit() * 0.34
            var c = crownCenter + SIMD3<Float>(cos(az) * sinT * crownRX * rad,
                                               cosT * crownRY * rad,
                                               sin(az) * sinT * crownRX * rad)
            if !crowning { c.y -= fr }           // fruit hangs just below its attachment point
            s.nextCardID &+= 1
            let prevCard = s.cardMark
            s.cardMark = s.nextCardID
            emitFruitBody(&s, center: c, radius: fr, elongate: fruit.elongate, color: fruit.color)
            s.cardMark = prevCard
        }
    }

    /// One fruit: a small UV sphere stretched vertically by `elongate` (1 = round orange, >1 =
    /// ovoid lemon), with correct ellipsoid smooth normals and a flat per-vertex fruit colour.
    /// The leaf group is registered `doubleSided`, so the sphere's winding is free — the supplied
    /// radial normals carry the shading. ~140 tris/fruit.
    public static func emitFruitBody(_ s: inout Soup, center: SIMD3<Float>, radius: Float,
                                      elongate: Float, color: SIMD3<Float>) {
        let lon = 10, lat = 7
        func vert(_ i: Int, _ j: Int) -> (p: SIMD3<Float>, n: SIMD3<Float>) {
            let u = Float(i) / Float(lon) * 2 * Float.pi
            let v = Float(j) / Float(lat) * Float.pi
            let sinV = sin(v), cosV = cos(v)
            let nx = cos(u) * sinV, ny = cosV, nz = sin(u) * sinV
            let p = center + SIMD3<Float>(nx * radius, ny * radius * elongate, nz * radius)
            // Ellipsoid normal: gradient of (x/a)²+(y/b)²+(z/c)², i.e. divide each axis by its
            // squared semi-axis, then normalise.
            let en = safeNormalize(SIMD3<Float>(nx / radius,
                                                ny / (radius * elongate),
                                                nz / radius),
                                   fallback: SIMD3<Float>(nx, ny, nz))
            return (p, en)
        }
        for j in 0..<lat {
            for i in 0..<lon {
                let a = vert(i, j), b = vert(i + 1, j)
                let cc = vert(i, j + 1), d = vert(i + 1, j + 1)
                s.tri(a.p, cc.p, b.p, n0: a.n, n1: cc.n, n2: b.n, c0: color, c1: color, c2: color)
                s.tri(b.p, cc.p, d.p, n0: b.n, n1: cc.n, n2: d.n, c0: color, c1: color, c2: color)
            }
        }
    }

    /// Flip `n` to the same hemisphere as `ref` (the leaf's two halves wind in
    /// opposite directions, so one geometric normal points away from the canopy
    /// outward direction — flip it so both halves shade as the same outward cup).

    // ===== helpers faceToward..hermite (lines 3933-4211) =====
    public static func faceToward(_ n: SIMD3<Float>, _ ref: SIMD3<Float>) -> SIMD3<Float> {
        simd_dot(n, ref) < 0 ? -n : n
    }

    // ── Tip-offset / tangent (port of curveTipOffset / curveTipTangent) ───

    public static func curveTipOffset(length: Float, curveAmount: Float,
                                       curveAxis: SIMD3<Float>) -> SIMD3<Float> {
        let cAxRaw = SIMD3<Float>(curveAxis.x, 0, curveAxis.z)
        let l = simd_length(cAxRaw)
        let ax = l > 1e-4 ? cAxRaw / l : SIMD3<Float>(1, 0, 0)
        let side = curveAmount * length
        return SIMD3<Float>(ax.x * side, length, ax.z * side)
    }

    public static func curveTipTangent(length: Float, curveAmount: Float,
                                        curveAxis: SIMD3<Float>) -> SIMD3<Float> {
        let cAxRaw = SIMD3<Float>(curveAxis.x, 0, curveAxis.z)
        let l = simd_length(cAxRaw)
        let ax = l > 1e-4 ? cAxRaw / l : SIMD3<Float>(1, 0, 0)
        let tanSide = 2 * curveAmount * length
        let tanY = length
        let len = max(1e-4, sqrt(tanSide * tanSide + tanY * tanY))
        return SIMD3<Float>(ax.x * tanSide / len, tanY / len, ax.z * tanSide / len)
    }
}
