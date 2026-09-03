import simd
import Foundation

/// **Carpet — a pile field, not a weave grid.**
///
/// The defect this exists to fix (DH-0457): a rug had no material of its own, so it wore
/// `velvet` — an *upholstery* cloth authored for a ~0.30 m cushion. On a 3 m rug that weave
/// prints a regular two-tone grid, a razor-cut edge, and no nap. A carpet's tell is the
/// opposite of a weave's: it is a dense STOCHASTIC field of tufts, each with its own colour,
/// height and a laid direction, and its whole character is that it CHANGES VALUE with view
/// angle because the fibres lie one way (the "nap").
///
/// Three things carry that look, and none of them is a periodic pattern:
///
///   1. **A pile field.** Tufts are Voronoi cells at ~2.5 mm gauge (real cut-pile is knotted
///      on a ~1/10″ needle pitch), so the dominant period sits BELOW `MaterialScaleAudit`'s
///      3 mm touch band and never resolves as a lattice — the opposite of velvet's ~15 mm
///      weave. Each tuft crowns at its centre and darkens into the gap between tufts (the
///      pile's self-shadow), which is what makes carpet read as depth rather than a printed
///      plane. Stochastic by construction, so it opts INTO the engine's hex de-repeat
///      (`.carpet ∈ stochasticPatternCategories`) and the tile repeat dissolves.
///   2. **A laid nap.** `grainTangent` is a dominant direction with only a whisper of
///      per-tuft jitter — coherent enough to clear `TextureAudit.grainFloor`, so the engine's
///      anisotropic-GGX branch streaks the highlight along the lay — plus a real cloth
///      **sheen** lobe for the grazing glow. This is the "changes with view angle" tell.
///   3. **Roughness + tone life.** Matte and high-roughness (a carpet is never glossy),
///      broken tuft-by-tuft and with a gentle broad drift so it isn't one dead tone.
///
/// Parameterised by `CarpetPile` so cut-pile / loop (berber) / flatweave can differ in tuft
/// shape, gloss and jitter without a second generator. The library ships cut-pile as the
/// default "Carpet"; the others are reachable through the same entry.
public enum CarpetPile: Equatable, Sendable {
    /// Dense upright cut yarn — the soft, matte broadloom a bedroom rug is. The default.
    case cutPile
    /// Uncut looped yarn (berber) — rounder crowns, a touch glossier, tighter gauge.
    case loop
    /// Low flat woven (dhurrie / kilim) — almost no pile, a shallow tighter field.
    case flatweave
}

extension MaterialGenerator {

    /// A carpet / rug pile. `base` is the yarn colour (a rug's per-instance tint multiplies
    /// this, exactly as it does for the fabric slices). See `CarpetPile` for how the pile
    /// style shifts the crown shape, gloss and nap jitter.
    ///
    /// **Gauge is physical, and it assumes a ~0.30 m surface run** (`carpetMeshUVScale` — the
    /// tile the rug mesh bakes its UVs at). At 512² that is ~0.59 mm per texel, so a
    /// ~2.5 mm tuft is ~4 texels across: comfortably above Nyquist (no tuft moiré) and below
    /// the 3 mm touch band the scale audit requires. A carpet applied to a *structural* floor
    /// tiles at the floor's coarser period and reads softer — acceptable; the rug is the case
    /// this is built for.
    public static func carpet(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 251,
                              base: Vec3 = Vec3(0.42, 0.40, 0.36),
                              pile: CarpetPile = .cutPile) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .carpet)
        let sh = seed

        // Tuft gauge, in cells across the ~0.30 m tile. Loop pilies are tighter-set; a
        // flatweave's "tufts" are the tight woven knots, tighter still. All land the dominant
        // feature below the 3 mm touch band at the assumed run.
        let gauge: Int
        let crownPower: Double        // how sharply a tuft crowns (loops are rounder)
        let gloss: Double             // sheen strength — cloth grazing lobe
        let napJitter: Double         // per-tuft deviation from the laid direction (radians)
        let reliefStrength: Double    // detail-band pile fuzz
        switch pile {
        case .cutPile:   gauge = 120; crownPower = 0.55; gloss = 0.50; napJitter = 0.22; reliefStrength = 0.85
        case .loop:      gauge = 150; crownPower = 0.85; gloss = 0.62; napJitter = 0.16; reliefStrength = 0.70
        case .flatweave: gauge = 180; crownPower = 1.10; gloss = 0.30; napJitter = 0.10; reliefStrength = 0.55
        }

        // The nap: a dominant lay direction plus a small per-tuft wobble. Coherent (the whole
        // pile leans one way) so it clears TextureAudit.grainFloor and the engine streaks the
        // grazing highlight along it — the view-angle value shift that reads as carpet.
        let napAngle = 0.35                                   // ~20° off +u, an arbitrary lay
        var grain = [Vec2](repeating: Vec2(1, 0), count: size * size)

        // Height field first (authoring is height-first): tuft crowns over a self-shadowing gap.
        var height = [Double](repeating: 0, count: size * size)
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                let tuft = Noise.voronoiTiled(u, v, cells: gauge, jitter: 0.95, seed: sh)
                // Crown: 1 at the tuft centre (f1→0), falling into the gap. `crownPower` sets
                // how domed each tuft is; loops are rounder, a flatweave nearly flat.
                let crown = pow(clamp01(1.0 - tuft.f1 * 1.7), 1.0 / crownPower)
                // Per-tuft height variation — a real pile is never perfectly level.
                let tid = Double(tuft.cellId & 0xFFFF) / 65535.0
                let pileTop = 0.34 + 0.52 * crown * (0.7 + 0.3 * tid)
                // A whisper of sub-tuft fibre so a single tuft isn't a smooth dome.
                let fibre = (Noise.fbmTiled(u * 40, v * 40, baseCells: 8, octaves: 2, seed: sh ^ 0x2B) - 0.5) * 0.05
                height[ch.idx(x, y)] = clamp01(pileTop + fibre)

                // Nap tangent: dominant lay, wobbled per tuft (deterministic in the tuft id).
                let a = napAngle + (tid - 0.5) * 2.0 * napJitter
                grain[ch.idx(x, y)] = Vec2(cos(a), sin(a))
            }
        }
        ch.height = height
        ch.grainTangent = grain

        // Albedo + roughness read off the pile. A tuft crown catches light (lighter, a touch
        // less rough); the gap between tufts is the pile's shadow (darker, rougher). Per-tuft
        // yarn-colour jitter keeps it from a single tone; a broad low-frequency drift gives it
        // metre-ish life without a landmark that could print at the tile period.
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                let tuft = Noise.voronoiTiled(u, v, cells: gauge, jitter: 0.95, seed: sh)
                let crown = pow(clamp01(1.0 - tuft.f1 * 1.7), 1.0 / crownPower)
                let gap = smoothstep(0.0, 0.30, tuft.f2 - tuft.f1)   // ~1 mid-tuft, →0 at the seam
                let tid = Double(tuft.cellId & 0xFFFF) / 65535.0
                // Per-yarn colour jitter: ±6% value, plus a faint hue wander so a wool pile
                // reads as many dyed strands, not one flat fill.
                let yarn = 0.90 + 0.16 * tid
                let hue = Vec3(1.0 + (tid - 0.5) * 0.06,
                               1.0 + (Double((tuft.cellId >> 16) & 0xFF) / 255.0 - 0.5) * 0.05,
                               1.0 + (Double((tuft.cellId >> 24) & 0xFF) / 255.0 - 0.5) * 0.06)
                // Broad drift for large-scale life (subtle — the hex de-repeat scrambles it,
                // and it must not print a 0.30 m lattice on the rug).
                let drift = 0.94 + 0.10 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: sh ^ 0x77)
                // Pile self-shadow: darker in the gap, where light doesn't reach the backing.
                let shade = 0.62 + 0.38 * (crown * 0.6 + gap * 0.4)
                let a = base * yarn * drift * shade
                ch.albedo[ch.idx(x, y)] = clampBand(Vec3(a.x * hue.x, a.y * hue.y, a.z * hue.z))
                // Matte, high, broken tuft-by-tuft. Crowns are slightly less rough (the fibre
                // tips catch a soft specular); gaps are rougher, matte shadow.
                ch.roughness[ch.idx(x, y)] = clamp01(0.90 - crown * 0.10
                    + (tid - 0.5) * 0.08
                    + (Noise.fbmTiled(u * 3, v * 3, baseCells: 2, octaves: 2, seed: sh ^ 0x3D) - 0.5) * 0.10)
            }
        }

        ch.sheen = gloss
        ch.sheenRoughness = 0.45                     // upright pile → a broad, soft grazing sheen
                                                     // (DH-0081; broader than a flat weave)
        // Strong relief from the pile field — the tufts stand up; the normal should read them.
        ch.deriveNormals(strength: 6)
        // Fine sub-fibre fuzz (the pile at grazing range) as the detail band. A carpet is matte,
        // so the diffuse-visible occlusion companion is what actually reads (detail normals are a
        // specular-band effect on a matte dielectric — see setDetailRelief).
        addMicroDetail(&ch, seed: sh ^ 0xC7, baseCells: 96, octaves: 2, strength: reliefStrength)
        return ch
    }
}
