import Foundation
import simd
import VisualizerMaterials

extension LeafSilhouette {
    /// A cut-flower stem leaf: lanceolate, widest a third of the way up, a clean point.
    public static let stemLeaf = LeafSilhouette(
        name: "stemLeaf",
        hero: [(0.00, 0.00), (0.08, 0.34), (0.30, 0.78), (0.52, 0.80), (0.76, 0.52), (0.93, 0.18), (1.00, 0.00)])
    /// A tulip tepal: a long ellipse, broadest past the middle, closing to a soft point.
    public static let tulipTepal = LeafSilhouette(
        name: "tulipTepal",
        hero: [(0.00, 0.00), (0.08, 0.46), (0.30, 0.84), (0.60, 1.00), (0.84, 0.82), (0.96, 0.44), (1.00, 0.00)])
}

/// The `.flowers` bouquet, grown flower by flower and PAINTED — the cut-flower counterpart of the
/// foliage houseplants (`PottedPlantMesh+Houseplants`).
///
/// What it replaced: every stem topped with the same bloom — two rings of strip-stitched petal
/// cards round a flat disc, one flat colour per palette entry — which read as crumpled paper cups
/// and cost a draw instance per colour (up to twelve per bouquet). Each stem now carries one of
/// four real flower FORMS, drawn together with its colour:
///
///  - **Rose** — thirteen petals on a golden-angle spiral, from a tight incurved heart to an opened
///    outer ring whose lips turn back.
///  - **Tulip** — six tepals in two whorls of three, a closed goblet, tips curled in.
///  - **Daisy** — a ring of narrow rays (two layers, offset) round a domed disc.
///  - **Anemone** — six broad sepals in two layers of three, an open bowl round a dark domed eye.
///
/// Petals SHINGLE: every one sits just outside, and a little more open than, the petal it
/// overlaps — the spiral does it for the rose, a second layer for the others. Petals sharing one
/// pitch and one radius while overlapping two-fold cut straight through each other, and that read
/// as crumpled paper. Their cups are shallow (a petal's margins lift a fraction of its WIDTH) and
/// their lips round.
///
/// …and now and then a closed BUD instead. Every petal is a `LeafSheet` blade (curved, smooth, both
/// faces) painted from claw to lip — deeper at the base, the flower's colour through the blade, a
/// touch lighter at the lip, a white flower greening at its heart — the gradient every real petal
/// has. Stems and leaves are painted too. The bouquet is ONE painted foliage group plus ONE painted
/// bloom group: two draw instances, whatever its size.
///
/// The planter's flowers (`PlanterMesh`) still use the older card `emitBloom(petals:center:…)` and
/// `bloomPalette`: a bed of twenty sheet flowers is past that placeable's triangle budget.
extension PottedPlantMesh {

    public enum BloomForm: String, Sendable, CaseIterable { case rose, tulip, daisy, anemone }

    /// The mixed cut-flower palette — each entry a petal colour, its centre, and the forms that
    /// colour comes in (a white daisy, a red rose or anemone…). Same order as `bloomPalette`.
    /// Taste: neutral defaults, flagged.
    public static let bouquetPalette: [(petal: Vec3, center: Vec3, forms: [BloomForm])] = [
        (Vec3(0.92, 0.91, 0.86), Vec3(0.90, 0.70, 0.10), [.daisy, .rose]),        // white
        (Vec3(0.90, 0.42, 0.58), Vec3(0.86, 0.66, 0.18), [.rose, .tulip]),        // pink
        (Vec3(0.95, 0.74, 0.12), Vec3(0.40, 0.22, 0.06), [.daisy, .tulip]),       // yellow
        (Vec3(0.72, 0.06, 0.08), Vec3(0.14, 0.04, 0.05), [.rose, .anemone]),      // red
        (Vec3(0.46, 0.22, 0.62), Vec3(0.10, 0.06, 0.14), [.anemone, .tulip]),     // purple
        (Vec3(0.94, 0.40, 0.22), Vec3(0.55, 0.20, 0.08), [.rose, .tulip]),        // coral
    ]

    /// Paint one petal vertex: the claw deeper (and, for a pale flower, greened), the blade the
    /// flower's colour, the lip a touch lighter; the back of a petal slightly paler.
    static func petalPaint(_ petal: Vec3, site: LeafSheet.Site, face: LeafFace) -> Vec3 {
        let v = max(0, min(1, site.y))
        let pale = (petal.x + petal.y + petal.z) / 3 > 0.8
        let claw = pale ? Vec3(0.62, 0.74, 0.34) : petal * 0.62
        var c = FoliagePaint.mix3(claw, petal, smoothstep(0.02, 0.42, v))
        c = FoliagePaint.mix3(c, petal * 1.06 + Vec3(0.03, 0.03, 0.03), smoothstep(0.62, 1.0, v) * 0.6)
        if face == .lower { c = FoliagePaint.mix3(c, Vec3(0.9, 0.9, 0.88), 0.12) }
        return FoliagePaint.clampColor(c)
    }

    /// The rows a petal is cut into — close at the rounded lip, where the outline turns fastest: with
    /// the last rows at 0.94 and 1 every petal ended in a point and a flower's rim was a paper crown.
    static let petalRows: [Double] = [0, 0.12, 0.30, 0.50, 0.70, 0.85, 0.94, 0.98, 1]

    /// One petal as a painted sheet: its claw at `base`, running along `yAxis`, `inner` the face
    /// toward the flower's heart. `cup` rolls its margins in toward the heart; a positive `flare`
    /// throws its lip OUT, a negative one curls it IN over the heart. A narrow ray needs no interior
    /// column (`columns: []`); a broad petal takes one so its cup is a curve, not a fold.
    ///
    /// Smoothed with an 85° crease, not a leaf's 40°: a petal is ONE smooth sheet — it has no midrib
    /// channel to keep — and across a grid this coarse a well-cupped petal's two middle cells meet
    /// at up to ~60° down its midline. Held as an arris, that fold (and the ones beside it) shaded
    /// every petal as creased paper. The two faces stay apart: they meet at 180°.
    static func emitPetal(into p: inout PaintedMesh, base: Vec3, yAxis: Vec3, inner: Vec3,
                          length: Double, aspect: Double, cup: Double, flare: Double,
                          silhouette: LeafSilhouette, columns: [Double], color: Vec3) {
        let x = normalize3(cross3(yAxis, inner))
        var s = LeafSheet.Surface(base: base, xAxis: x, yAxis: yAxis, normal: inner,
                                  length: length, halfWidth: aspect / 2)
        s.cup = cup
        s.cupPower = 2       // a petal cups as a round U, flat along its midline
        s.droop = flare
        var petal = PaintedMesh()
        LeafSheet.emitBlade(into: &petal, surface: s, margin: silhouette.margin(atRows: petalRows),
                            aspect: aspect, columns: columns, midribBand: 0) { site, face in
            petalPaint(color, site: site, face: face)
        }
        p.append(petal.smoothed(creaseDegrees: 85))
    }

    /// A whorl of `count` petals round `axis`, each pitched `pitch` off it, its claw `baseRadius`
    /// petal-lengths out from the axis (the receptacle it stands on).
    static func emitWhorl(into p: inout PaintedMesh, at centre: Vec3, axis: Vec3, count: Int, phase: Double,
                          pitch: Double, length: Double, aspect: Double, cup: Double, flare: Double,
                          lift: Double, baseRadius: Double = 0.06, silhouette: LeafSilhouette, color: Vec3,
                          rng: inout SplitMix) {
        let f = GardenPlantMesh.frame(yAxis: axis)
        let columns: [Double] = aspect < 0.4 ? [] : [0.5]
        for k in 0 ..< count {
            // A little irregularity, kept small: petals of one whorl overlap their neighbours, and a
            // larger scatter drives them through each other.
            let a = phase + 2 * Double.pi * Double(k) / Double(count) + (rng.unit() - 0.5) * 0.16
            let radial = normalize3(f.x * cos(a) + f.z * sin(a))
            let pk = pitch + (rng.unit() - 0.5) * 0.10
            let yAxis = normalize3(axis * cos(pk) + radial * sin(pk))
            let inner = normalize3(axis * sin(pk) - radial * cos(pk))
            emitPetal(into: &p, base: centre + axis * lift + radial * (length * baseRadius), yAxis: yAxis,
                      inner: inner, length: length * (0.9 + rng.unit() * 0.2), aspect: aspect,
                      cup: cup, flare: flare, silhouette: silhouette, columns: columns, color: color)
        }
    }

    /// A domed centre (a daisy's disc, an anemone's eye) facing along `axis`, `rim` at its edge
    /// shading to `color` at its crown.
    static func emitDisc(into p: inout PaintedMesh, at centre: Vec3, axis: Vec3, radius: Double,
                         dome: Double, color: Vec3, rim: Vec3) {
        let prof: [(r: Double, y: Double)] = [(radius, 0), (radius * 0.86, dome * 0.55), (radius * 0.5, dome * 0.92), (0, dome)]
        let disc = GardenPlantMesh.revolvedPart(prof, segments: 12, origin: centre, axis: axis)
        var painted = PaintedMesh()
        painted.mesh = disc
        painted.colors = disc.positions.map { pos in
            let h = dot3(pos - centre, normalize3(axis)) / max(1e-9, dome)
            return FoliagePaint.mix3(rim, color, smoothstep(0.1, 0.7, h))
        }
        p.append(painted)
    }

    /// A rose: `count` petals on a golden-angle spiral round `axis`, from the heart (short, upright,
    /// cupped, tips curled in) to the outer ring (long, open, flatter, lips turned back). Each petal
    /// is a little longer and more open than the one before, so neighbours SHINGLE instead of
    /// cutting through each other.
    static func emitRose(into p: inout PaintedMesh, at pos: Vec3, axis: Vec3, count: Int, scale: Double,
                         color: Vec3, rng: inout SplitMix) {
        let f = GardenPlantMesh.frame(yAxis: axis)
        let phase = rng.unit() * 2 * Double.pi
        for i in 0 ..< count {
            let t = Double(i) / Double(max(1, count - 1))       // 0 the heart → 1 the outermost
            let a = phase + Double(i) * Phyllotaxis.goldenAngle
            let radial = normalize3(f.x * cos(a) + f.z * sin(a))
            let pitch = 0.10 + 0.78 * pow(t, 1.4) + (rng.unit() - 0.5) * 0.05
            let yAxis = normalize3(axis * cos(pitch) + radial * sin(pitch))
            let inner = normalize3(axis * sin(pitch) - radial * cos(pitch))
            emitPetal(into: &p, base: pos + radial * (scale * (0.03 + 0.06 * t)) + axis * (scale * 0.05 * (1 - t)),
                      yAxis: yAxis, inner: inner, length: scale * (0.46 + 0.54 * pow(t, 0.8)),
                      aspect: 0.70 + 0.22 * t, cup: 0.14 - 0.08 * t, flare: -0.10 + 0.22 * t * t,
                      silhouette: WildflowerPlantingMesh.poppyPetalSilhouette, columns: [0.5], color: color)
        }
    }

    /// One flower of `form`, its heart at `pos`, facing `faceDir`, `scale` its petal length.
    public static func emitBloom(into p: inout PaintedMesh, form: BloomForm, bud: Bool, at pos: Vec3,
                                 axis faceDir: Vec3, scale: Double, petal: Vec3, center: Vec3,
                                 rng: inout SplitMix) {
        let axis = normalize3(faceDir)
        let phase = rng.unit() * 2 * Double.pi
        if bud {
            // A closed bud: one whorl, petals nearly along the axis, wrapped round each other.
            emitWhorl(into: &p, at: pos, axis: axis, count: 5, phase: phase, pitch: 0.10, length: scale * 0.55,
                      aspect: 0.62, cup: 0.18, flare: -0.10, lift: 0, baseRadius: 0.05, silhouette: .petal,
                      color: petal, rng: &rng)
            return
        }
        switch form {
        case .rose:
            emitRose(into: &p, at: pos, axis: axis, count: 13, scale: scale, color: petal, rng: &rng)
        case .tulip:
            // Two whorls of three: the outer tepals broad and standing off a wide receptacle, the
            // inner three inside their gaps; every tip curled back in over the heart.
            emitWhorl(into: &p, at: pos, axis: axis, count: 3, phase: phase, pitch: 0.30, length: scale * 1.15,
                      aspect: 0.64, cup: 0.12, flare: -0.14, lift: 0, baseRadius: 0.12,
                      silhouette: .tulipTepal, color: petal, rng: &rng)
            emitWhorl(into: &p, at: pos, axis: axis, count: 3, phase: phase + Double.pi / 3, pitch: 0.20,
                      length: scale * 1.08, aspect: 0.60, cup: 0.12, flare: -0.12, lift: scale * 0.02,
                      baseRadius: 0.07, silhouette: .tulipTepal, color: petal, rng: &rng)
        case .daisy:
            let rays = 16
            let ray = WildflowerPlantingMesh.rayPetalSilhouette
            emitWhorl(into: &p, at: pos, axis: axis, count: rays, phase: phase, pitch: 1.38, length: scale * 0.95,
                      aspect: 0.26, cup: 0.03, flare: 0.04, lift: 0, silhouette: ray, color: petal, rng: &rng)
            emitWhorl(into: &p, at: pos, axis: axis, count: rays / 2, phase: phase + Double.pi / Double(rays),
                      pitch: 1.24, length: scale * 0.78, aspect: 0.26, cup: 0.03, flare: 0.03, lift: scale * 0.02,
                      silhouette: ray, color: petal, rng: &rng)
            emitDisc(into: &p, at: pos + axis * (scale * 0.01), axis: axis, radius: scale * 0.30,
                     dome: scale * 0.16, color: center, rim: center * 0.7)
        case .anemone:
            // Six sepals in two layers of three: the lower layer a little more open, so the bowl's
            // petals overlap without meeting.
            let broad = WildflowerPlantingMesh.poppyPetalSilhouette
            emitWhorl(into: &p, at: pos, axis: axis, count: 3, phase: phase, pitch: 0.80, length: scale,
                      aspect: 0.80, cup: 0.12, flare: 0.02, lift: 0, baseRadius: 0.05,
                      silhouette: broad, color: petal, rng: &rng)
            emitWhorl(into: &p, at: pos, axis: axis, count: 3, phase: phase + Double.pi / 3, pitch: 0.95,
                      length: scale * 0.96, aspect: 0.80, cup: 0.12, flare: 0.03, lift: -scale * 0.012,
                      baseRadius: 0.07, silhouette: broad, color: petal, rng: &rng)
            emitDisc(into: &p, at: pos + axis * (scale * 0.02), axis: axis, radius: scale * 0.24,
                     dome: scale * 0.14, color: center * 0.6, rim: center)
        }
    }
}
