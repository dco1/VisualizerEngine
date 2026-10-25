// Microcement — a thin, hand-troweled cementitious skim coat in any colour (DH-1518).
import Foundation

extension MaterialGenerator {
    /// **Troweled microcement**: the smooth, sealed, mottled grey wall of a modern bath. Everything
    /// stucco is NOT — there is no sand grit and almost no relief; the whole read is TONE: broad
    /// cloudy patches 15–30 cm across where a coat was laid thicker or burnished harder, each
    /// carrying the faint direction of the trowel pass, over a satin (never matte, never glossy)
    /// sealer. Albedo mottle here is the material's identity — it is baked into the colour, so it
    /// does not move with the light and is not the lighting mottle the export gates police.
    ///
    /// Bands on the paint tile (0.6 m): wash 2 cells (30 cm, ±5.5 %), patch 5 cells (12 cm, ±3.5 %),
    /// stroke — Worley arcs from two cell sizes, soft, ±2 % in tone — and a 5 mm micro tooth.
    public static func microcement(size: Int = MaterialGenerator.bakeSize, color: Vec3 = Vec3(0.42, 0.41, 0.38),
                                   seed: UInt64 = 71) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .paint)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                let wash  = Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: seed) - 0.5
                let patch = Noise.fbmTiled(u, v, baseCells: 5, octaves: 3, seed: seed ^ 0x33) - 0.5
                // The trowel's pass: stretch the patch field along one diagonal by sampling it on a
                // 45° skewed axis, so the cloud has a grain rather than reading as isotropic noise.
                let su = (u + v).truncatingRemainder(dividingBy: 1)   // integer skew: tiles in both axes
                let streak = Noise.fbmTiled(su, v, baseCells: 3, octaves: 2, seed: seed ^ 0x77) - 0.5
                let tooth = Noise.fbmTiled(u, v, baseCells: 120, octaves: 2, seed: seed ^ 0x9D) - 0.5
                let tone = 1.0 + 0.11 * wash + 0.07 * patch + 0.05 * streak + 0.015 * tooth
                // A faint warm/cool drift between patches — cement never cures one neutral.
                let drift = 0.025 * (Noise.fbmTiled(u, v, baseCells: 3, octaves: 2, seed: seed ^ 0x5A) - 0.5)
                ch.albedo[ch.idx(x, y)] = clampBand(Vec3(color.x * tone * (1 + drift),
                                                         color.y * tone,
                                                         color.z * tone * (1 - drift)))
                ch.height[ch.idx(x, y)] = clamp01(0.5 + 0.10 * streak + 0.06 * tooth)
                // Satin sealer, burnished harder where the coat is thicker/lighter.
                ch.roughness[ch.idx(x, y)] = clamp01(0.52 - 0.30 * wash + 0.12 * streak + 0.06 * tooth)
            }
        }
        ch.clearcoat = 0
        ch.diffuseRoughness = 0.45
        ch.deriveNormals(strength: 0.9)
        addMicroDetail(&ch, seed: seed ^ 0xC1, baseCells: 96, strength: 0.22)
        return ch
    }
}
