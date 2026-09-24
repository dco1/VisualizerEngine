import Foundation
import simd
import VisualizerMaterials

/// How a houseplant's foliage is COLOURED — the palette each style wears, and the rules that paint
/// a vertex from where it sits on which leaf.
///
/// A flat stamp — one green over every vertex of the plant — was the loudest copy-paste tell: real
/// foliage is a different shade leaf to leaf (the new leaf lime, the old one deep green), paler
/// underneath than on top, paler along the midrib, banded on a snake plant, blushed at an
/// echeveria's tips. Each palette's `mature` is `PlantStyle.leafColor` (see there for how its level
/// was calibrated against the rendered room); the rest are set against it.
///
/// Everything here is a pure function of (style, leaf, site, face), so a seeded plant paints the
/// same every build.
public struct FoliagePalette: Sendable, Equatable {
    /// The mature upper face — `PlantStyle.leafColor`, the one source for "this plant's green".
    public var mature: Vec3
    /// The newest leaf's upper face: lighter and yellower, the flush of new growth.
    public var young: Vec3
    /// The underside: paler, greyer, matter.
    public var underside: Vec3
    /// The midrib (and lateral veins, at a fraction of the strength).
    public var rib: Vec3
    /// How strongly the lateral veins take the rib colour on the upper face (0…1).
    public var veinStrength: Double
    /// A stalk: petiole, rachis, or the stems of a bouquet.
    public var stalk: Vec3
    /// Per-leaf value scatter (± fraction) — no two leaves the same shade.
    public var leafValueJitter: Double
    /// Per-leaf hue scatter: how far a leaf leans toward yellow (+) or blue (−), as a fraction.
    public var leafHueJitter: Double

    public init(mature: Vec3, young: Vec3, underside: Vec3, rib: Vec3, veinStrength: Double,
                stalk: Vec3, leafValueJitter: Double, leafHueJitter: Double) {
        self.mature = mature; self.young = young; self.underside = underside; self.rib = rib
        self.veinStrength = veinStrength; self.stalk = stalk
        self.leafValueJitter = leafValueJitter; self.leafHueJitter = leafHueJitter
    }
}

/// One leaf's own colour identity, drawn once per leaf from the plant's seeded stream.
public struct LeafTone: Sendable, Equatable {
    /// 0 a mature leaf … 1 the newest: blends `mature` → `young`.
    public var youth: Double
    /// Value multiplier (1 ± `leafValueJitter`).
    public var value: Double
    /// −1 … 1: lean toward blue-green (−) or yellow-green (+).
    public var hue: Double
    public init(youth: Double, value: Double = 1, hue: Double = 0) {
        self.youth = max(0, min(1, youth)); self.value = value; self.hue = max(-1, min(1, hue))
    }
}

public extension PlantStyle {
    /// This style's palette. `mature` IS `leafColor` — the one source, so the render bridge, the
    /// garden plant and the photometric gates that read `leafColor` all agree.
    var foliagePalette: FoliagePalette {
        let m = leafColor
        switch self {
        case .fiddleLeafFig:
            // Deep, glossy, near-black green on top; the flush of new leaves a clear yellow-lime;
            // a pale yellow-green midrib; a dull olive underside.
            return FoliagePalette(mature: m, young: Vec3(0.340, 0.560, 0.140),
                                  underside: Vec3(0.320, 0.420, 0.240), rib: Vec3(0.520, 0.580, 0.230),
                                  veinStrength: 0, stalk: Vec3(0.200, 0.210, 0.120),
                                  leafValueJitter: 0.10, leafHueJitter: 0.06)
        case .monstera:
            return FoliagePalette(mature: m, young: Vec3(0.300, 0.540, 0.160),
                                  underside: Vec3(0.280, 0.400, 0.230), rib: Vec3(0.340, 0.500, 0.200),
                                  veinStrength: 0.30, stalk: Vec3(0.200, 0.370, 0.130),
                                  leafValueJitter: 0.09, leafHueJitter: 0.05)
        case .snakePlant:
            // The painter's banding (`snakePlantBand`) carries this style's pattern; `young` is the
            // grey-green of the light bands, `rib` the gold of a 'Laurentii' margin.
            return FoliagePalette(mature: m, young: Vec3(0.280, 0.360, 0.230),
                                  underside: m, rib: Vec3(0.620, 0.520, 0.100),
                                  veinStrength: 0, stalk: m,
                                  leafValueJitter: 0.06, leafHueJitter: 0.04)
        case .fern:
            return FoliagePalette(mature: m, young: Vec3(0.360, 0.620, 0.160),
                                  underside: Vec3(0.320, 0.480, 0.200), rib: Vec3(0.260, 0.380, 0.120),
                                  veinStrength: 0.15, stalk: Vec3(0.200, 0.250, 0.100),
                                  leafValueJitter: 0.08, leafHueJitter: 0.05)
        case .succulent:
            // A farina-dusted sage rosette; `rib` is the blush its tips take in good light.
            return FoliagePalette(mature: m, young: Vec3(0.500, 0.620, 0.440),
                                  underside: Vec3(0.480, 0.560, 0.450), rib: Vec3(0.760, 0.420, 0.400),
                                  veinStrength: 0, stalk: m,
                                  leafValueJitter: 0.06, leafHueJitter: 0.04)
        case .flowers, .driedSpray, .christmasTree:
            return FoliagePalette(mature: m, young: m * 1.25, underside: m * 1.2, rib: m * 1.3,
                                  veinStrength: 0, stalk: m,
                                  leafValueJitter: 0.07, leafHueJitter: 0.04)
        }
    }
}

public enum FoliagePaint {

    /// Draw one leaf's tone from the plant's stream. `youth` is decided by the caller (it is a fact
    /// about WHERE the leaf is — the newest leaves are at the growing tip); only the scatter is random.
    public static func tone(youth: Double, palette: FoliagePalette, rng: inout PottedPlantMesh.SplitMix) -> LeafTone {
        let value = 1 + (rng.unit() * 2 - 1) * palette.leafValueJitter
        let hue = rng.unit() * 2 - 1
        return LeafTone(youth: youth, value: value, hue: hue)
    }

    /// The leaf's base colour for a face, before rib/vein detail.
    public static func base(_ palette: FoliagePalette, _ tone: LeafTone, face: LeafFace) -> Vec3 {
        let top = mix3(palette.mature, palette.young, tone.youth)
        var c: Vec3
        switch face {
        case .upper, .rim: c = top
        // The underside is paler and greyer than the top at EVERY age: a mature leaf's is the
        // palette's own dull olive; the flush's is its lime top lifted and greyed (a thin young
        // leaf's underside is paler still, never darker than its face).
        case .lower: c = mix3(palette.underside, top * 1.15 + Vec3(0.03, 0.03, 0.03), tone.youth)
        }
        c = c * tone.value
        // Hue: lean the leaf toward yellow-green (+red, −blue) or blue-green (−red, +blue).
        let h = tone.hue * palette.leafHueJitter
        c = Vec3(c.x * (1 + h), c.y, c.z * (1 - h))
        return clampColor(c)
    }

    /// Paint a sheet leaf's vertex: its face's base colour, lightened along the midrib band and the
    /// lateral veins. `ribHalfWidth` is the rib band's half-width in blade units.
    public static func sheet(_ palette: FoliagePalette, _ tone: LeafTone, site: LeafSheet.Site,
                             face: LeafFace, ribHalfWidth: Double) -> Vec3 {
        var c = base(palette, tone, face: face)
        // The rib: full strength on the midrib, fading off the band's edge, and thinning toward the
        // tip (a real midrib tapers with the blade).
        let r = max(0, 1 - abs(site.x) / max(1e-6, ribHalfWidth * 1.15))
        let taper = max(0, 1 - 0.75 * max(0, site.y))
        let rib = r * taper * (face == .lower ? 0.85 : 1.0)
        c = mix3(c, palette.rib * (face == .lower ? 1.10 : 1.0), rib)
        // Lateral veins: on the upper face a soft pale line; below, raised and paler still.
        let vs = palette.veinStrength * (face == .lower ? 1.2 : 1.0)
        c = mix3(c, mix3(c, palette.rib, 0.8), min(1, site.vein * vs))
        return clampColor(c)
    }

    /// A snake-plant ('Laurentii'-less trifasciata) blade: dark green crossed by pale grey-green
    /// wavy bands. `v` runs up the blade, `u` across it (0 midline → 1 margin), `phase` offsets one
    /// leaf's bands from the next so no two blades stripe in step.
    public static func snakePlantBand(_ palette: FoliagePalette, _ tone: LeafTone,
                                      v: Double, u: Double, phase: Double, margin: Bool) -> Vec3 {
        let dark = palette.mature * tone.value
        let light = palette.young * tone.value
        // Irregular transverse bands — eight up a blade, their spacing wandered by a slower wave so
        // no two are the same width — bent into a shallow chevron across it (the "tiger" zig-zag).
        // Eight, not the dozen-plus a close photograph shows: a blade carries 33 evenly spaced
        // vertex rows (`snakePlantSwordRows`), four to a band, and a finer pattern than that aliases
        // into blotches rather than stripes.
        let w = v * 8.0 + 0.6 * sin(v * 4.3 + phase) + 0.30 * abs(u) + phase
        let band = smoothstep(0.55, 0.95, 0.5 + 0.5 * sin(w * 2 * Double.pi))
        var c = mix3(dark, light, band * 0.85)
        if margin { c = mix3(c, palette.rib, 0.8) }
        return clampColor(c)
    }

    // MARK: helpers

    public static func mix3(_ a: Vec3, _ b: Vec3, _ t: Double) -> Vec3 { a + (b - a) * max(0, min(1, t)) }
    public static func clampColor(_ c: Vec3) -> Vec3 {
        Vec3(max(0, min(1, c.x)), max(0, min(1, c.y)), max(0, min(1, c.z)))
    }
}
