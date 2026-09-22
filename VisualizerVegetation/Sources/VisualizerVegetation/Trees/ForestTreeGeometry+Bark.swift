import Foundation
import simd

// Bark plate field + bark/leaf colour variation.
extension ForestTreeGeometry {

    // ── Bark plate field (front 2: discrete shedding plates) ──────────────────
    //
    // A Worley/cellular tessellation of the (angle × height) trunk surface into
    // discrete bark PLATES with REAL radial depth: each plate centre bulges
    // OUTWARD and the seams between plates recess INWARD, so the golden-hour rake
    // light reads cast shadow in the furrows — not a painted streak on a smooth
    // tube. Species-correct cell ASPECT: oak = blocky deep furrowed plates,
    // maple = vertically-stretched shaggy lifting plates, birch handled in
    // `barkColorAt` as papery peeling lenticel bands (it stays near-smooth).
    //
    // Returns:
    //   disp     — radial displacement as a FRACTION of trunk radius
    //              (+ outward plate crown, − recessed seam).
    //   liftN    — extra outward NORMAL tilt at the plate's leading (shed) edge,
    //              so a lifting scale catches a bright top rim + dark undercut.
    //   plateId  — stable per-plate hash (0..1) → per-plate albedo/roughness var.
    //   seam     — 0 at a plate centre → 1 deep in a seam (drives furrow darkening).
    public struct BarkPlate { var disp: Float; var liftN: Float; var plateId: Float; var seam: Float }

    /// Worley F1/F2 over a jittered lattice in (angle-cells × height-cells).
    /// `ax` wraps around the circumference (period = cellsU).
    public static func barkPlateField(look: TreeLook, u: Float, t: Float,
                                       seed: UInt64) -> BarkPlate {
        // Birch: papery, near-smooth — only faint horizontal peel relief.
        if look.isBirch {
            // Thin horizontal peel curls: gentle outward lift in bands, no deep
            // furrow. Keeps the paper-bark read without faking blocky plates.
            let band = valueNoise(t * 7.0 + Float(seed % 31), u * 1.3)
            let peel = smoothstepf(0.62, 0.92, band)
            return BarkPlate(disp: peel * 0.05, liftN: peel * 0.5,
                             plateId: band, seam: 0)
        }
        // Cell grid: oak = blocky (similar U/T cell size), maple = tall cells
        // (shaggy vertical strips). cellsU must be an INTEGER so the Worley
        // lattice wraps seamlessly around the trunk.
        let isMaple = (look.species == .maple)
        let cellsU: Float = isMaple ? 9 : 11          // around circumference
        let cellsT: Float = isMaple ? 5.5 : 8.0       // up the visible trunk
        let su = Float(seed % 101) * 0.123
        let st = Float(seed % 89) * 0.211
        let pu = (u + su) * cellsU                     // wrapped lattice coord
        let pt = (t + st) * cellsT
        // F1/F2 nearest-feature search over the 3×3 neighbourhood, U wrapped.
        var f1: Float = 10, f2: Float = 10
        var bestId: Float = 0
        let cu0 = Int(floor(pu)), ct0 = Int(floor(pt))
        for du in -1...1 {
            for dt in -1...1 {
                let cuW = cu0 + du
                let ct = ct0 + dt
                // Wrap the U cell index into [0, cellsU) so the seam is invisible.
                let cuWrapped = ((cuW % Int(cellsU)) + Int(cellsU)) % Int(cellsU)
                let jx = hash2(cuWrapped &* 7 &+ Int(seed % 13), ct)
                let jy = hash2(cuWrapped, ct &* 5 &+ Int(seed % 17))
                let fx = Float(cuW) + jx          // feature point (unwrapped U for distance)
                let ft = Float(ct) + jy
                var ddu = pu - fx
                // Shorten the U distance across the wrap so plates don't stretch.
                if ddu > cellsU * 0.5 { ddu -= cellsU }
                if ddu < -cellsU * 0.5 { ddu += cellsU }
                let ddt = pt - ft
                let d = sqrt(ddu * ddu + ddt * ddt)
                if d < f1 { f2 = f1; f1 = d; bestId = hash2(cuWrapped &+ 3, ct &+ 7) }
                else if d < f2 { f2 = d }
            }
        }
        // Seam strength: F2−F1 is small at a cell border (two features equidistant)
        // → that's the furrow floor. Ramp it.
        let border = f2 - f1                                   // ~0 at seam, large mid-plate
        let seam = 1.0 - smoothstepf(0.0, 0.22, border)        // 1 in seam, 0 mid-plate
        // Plate crown: bulge mid-plate, recess in the seam. Oak deeper than maple.
        let depth: Float = isMaple ? 0.10 : 0.14
        let crown = (1.0 - smoothstepf(0.0, 0.85, f1))          // 1 at centre → 0 at edge
        let disp = crown * depth - seam * depth * 1.15
        // Lifting/shedding edge: maple plates LIFT at their lower edge (shaggy);
        // tilt the normal outward there. Keyed to per-plate id so only some lift.
        let liftN = (isMaple ? 0.6 : 0.28) * smoothstepf(0.45, 0.85, border) * (0.4 + bestId)
        return BarkPlate(disp: disp, liftN: liftN, plateId: bestId, seam: seam)
    }

    // ── Bark / leaf colour ────────────────────────────────────────────────

    public static func barkColorAt(_ look: TreeLook, u: Float, t: Float,
                                    seed: UInt64) -> SIMD3<Float> {
        let sd = Float(seed % 97)
        // Two octaves of vertical streaking + fine ring-grain → fissured bark
        // tonal break-up so trunks don't read as flat plastic cylinders.
        let streak = valueNoise(sd + t * 2.5, u * 6)
        let coarse = valueNoise(sd * 0.3 + t * 0.7, u * 2.0)
        let fine = valueNoise(sd * 1.7 + t * 11.0, u * 22.0)
        if look.isBirch {
            // Birch: pale paper bark with dark lenticel dashes + occasional grey
            // patches where the bark has peeled.
            // Stronger, more frequent dark lenticel dashes + grey peel patches so
            // the birch reads as real paper bark, not a white plastic pillar.
            let dark = SIMD3<Float>(0.030, 0.030, 0.034)
            let grey = SIMD3<Float>(0.28, 0.27, 0.26)
            // Polish: dark HORIZONTAL lenticel dashes. Birch lenticels run
            // across the trunk (low freq in u = around the circumference, high
            // freq in t = stacked up the height) → short dark dashes in
            // horizontal bands. Two scales (bold + fine) for paper-bark realism.
            let lent = smoothstepf(0.66, 0.86, valueNoise(t * 9.0, u * 1.1)) * 1.0
            let lentFine = smoothstepf(0.80, 0.95, valueNoise(t * 22, u * 2.4)) * 0.8
            var c = look.bark * (0.74 + 0.30 * streak)
            c = mixv(c, grey, smoothstepf(0.55, 0.82, coarse) * 0.55)
            c = mixv(c, dark, max(lent, lentFine))
            c *= (0.88 + 0.20 * fine)
            return c
        }
        // Deciduous: deep fissures (dark) between lighter ridges.
        var c = look.bark * (0.55 + 0.6 * streak)
        c = mixv(c, look.bark * 0.34, smoothstepf(0.55, 0.9, coarse) * 0.6)  // fissures
        c *= (0.88 + 0.24 * fine)                                            // fine grain
        return c
    }

    /// Plate-aware bark albedo: starts from the base fissure colour, then adds
    /// the discrete-plate cues the brief asks for — weathered GREY-LICHEN ridge
    /// tops on the raised plate crowns, DARK DAMP furrow floors in the seams, and
    /// an OCCASIONAL pale-green LICHEN blotch on a minority of plates. Returns the
    /// modulated colour; `seam` and `plate.plateId` come from `barkPlateField`.
    public static func barkColorPlated(_ look: TreeLook, u: Float, t: Float,
                                        seed: UInt64, plate: BarkPlate) -> SIMD3<Float> {
        var c = barkColorAt(look, u: u, t: t, seed: seed)
        if look.isBirch {
            // Birch peel: lift the curled band a touch warmer/paler, no furrow.
            c = mixv(c, c * 1.18 + SIMD3<Float>(0.02, 0.018, 0.012), plate.plateId * 0.0)
            return c
        }
        // Weathered ridge top: plate crowns (low seam) read drier, greyer, lighter
        // (sun-bleached). Damp furrow floor (high seam) reads darker, cooler.
        let crown = 1.0 - plate.seam
        let weathered = SIMD3<Float>(0.30, 0.28, 0.25)         // dry grey ridge cap
        c = mixv(c, weathered, crown * (0.18 + 0.22 * plate.plateId))   // per-plate grey amount
        c = mixv(c, c * 0.42, plate.seam * 0.85)               // dark damp furrow floor
        // Per-plate albedo jitter so adjacent plates don't read identical.
        c *= 0.86 + 0.30 * plate.plateId
        // Occasional lichen: ~22% of plates carry a pale sage-green blotch, denser
        // on the (cooler/north) shaded side handled by the caller's u-phase.
        if plate.plateId > 0.78 {
            let lichen = SIMD3<Float>(0.115, 0.135, 0.085)     // pale sage-green, linear
            let blot = smoothstepf(0.78, 0.94, plate.plateId) * (0.55 + 0.45 * crown)
            c = mixv(c, lichen, blot)
        }
        return c
    }

    public static func leafColorVary(_ look: TreeLook, frac: Float,
                                      rng: inout ForestRNG) -> SIMD3<Float> {
        // ── Structural #4: TRANSMISSION VARIETY ──────────────────────────────
        // Real sun-through-leaf is not one flat emissive green: it varies with
        // blade thickness and how many leaves stack in front of the sun — bright
        // LIME where a single thin leaf, deep OLIVE where stacked, near-WHITE at
        // a thin edge. The deferred/RT leaf-transmission term scatters through
        // the leaf's ALBEDO, so varying the per-leaf base colour (warm↔lime,
        // light↔dark) is what makes the backlit canopy read as dappled
        // stained-glass instead of a uniform glow. (Per-leaf in geometry — NOT
        // a shader edit; HOLD constraint respected.)
        //
        // `frac` is the node position along the twig (0 = base, 1 = tip); tip
        // leaves are younger/thinner → push them toward bright lime, base leaves
        // toward deep olive (proxy for cluster depth / overlap).
        // COLOR FIX: prior lime/golden were authored as if sRGB display values but
        // used directly as linear RGB — net green channel up to 0.46 linear, which
        // the Illuminatorama HDR pipeline renders as neon highlighter (0.7+ display
        // luminance). Real broadleaf foliage in direct golden-hour sun: green
        // channel ≈ 0.10–0.22 linear (sRGB ≈ 0.35–0.50). Scale lime and golden
        // to realistic linear values by applying the sRGB gamma (≈×0.35); keep
        // olive which was already in a plausible linear range.
        // olive at 0.075/0.135/0.040 ≈ sRGB (0.30/0.38/0.22) — realistic ✓
        let golden = SIMD3<Float>(0.095, 0.075, 0.018)  // warm autumn leaf, linear (≈sRGB 0.33/0.29/0.14)
        let lime   = SIMD3<Float>(0.090, 0.165, 0.035)  // bright thin single leaf, linear (≈sRGB 0.33/0.44/0.20)
        let olive  = SIMD3<Float>(0.055, 0.100, 0.028)  // deep stacked-leaf, linear (slightly darkened)
        let j = rng.unit()
        // Per-tree/per-leaf autumn turn (some warm leaves to glow through).
        var c = mixv(look.leaf, golden, j * 0.30)
        // Warm↔lime hue jitter, ±~15%: half the leaves lean lime, half toward
        // the olive/warm base. Keyed to a second random so it's independent of
        // the autumn turn.
        let hueJit = rng.unit()                          // 0..1
        if hueJit > 0.55 {
            c = mixv(c, lime, (hueJit - 0.55) / 0.45 * 0.45)   // toward lime
        } else {
            c = mixv(c, olive, (0.55 - hueJit) / 0.55 * 0.40)  // toward deep olive
        }
        // Cluster-depth modulation: tip leaves slightly brighter, interior slightly
        // darker. Reduced multipliers to stay in a plausible range (old 0.80+0.45=1.25
        // at tip was blowing the budget even after color fix).
        let depthBoost = 0.88 + 0.17 * frac              // tip → up to 1.05×
        c = mixv(c, lime, frac * 0.18) * depthBoost
        // Brightness jitter (per-leaf thickness variance). Tighter range to avoid
        // bright-outlier cards reading as a different material.
        // ROUND-5 polish: the side-view sun-facing canopy bleached to CREAM (the
        // prior ~10–15 % pull was insufficient). Roll the leaf albedo ceiling down
        // ~18 % (0.80+0.28 → 0.66+0.24, top 1.08→0.90) so direct golden-hour sun on
        // the sun-facing leaf mass no longer clips to white. The backlit RIM reads
        // via the leafTransmission term (a SEPARATE path that scatters through the
        // albedo), so this roll-off tames the bleach without crushing the SSS glow.
        c *= 0.66 + 0.24 * rng.unit()
        return c
    }
}
