import simd
import Foundation

/// **Paper towel — a quilted diamond EMBOSS, not a printed pattern** (DH-1007).
///
/// What makes a paper-towel roll read as one is relief catching light: a lattice of pillowy
/// diamonds separated by pressed grooves, so the lit side of the roll shows crisp diagonal ridges
/// and the shade side goes soft. A pattern painted into the albedo is the failure this exists to
/// avoid — it prints at the same contrast in every light.
///
/// So the material is height-first. The tile is `paperTowelEmbossCellsPerTile` diamonds square, the
/// lattice is exactly periodic (two diagonal line families whose spacing divides the tile), and the
/// normal map is derived from the height, so the grooves, the pillow domes and the micro-occlusion
/// in the grooves can never disagree ([[feedback-single-source-of-truth]]).
///
/// **The tile is physical but not typed here.** A real quilted towel's diamond pitch is ~3.4 mm; the
/// mesh that wears this (`BowlMesh.towelPaper`) bakes its UVs so one tile spans exactly
/// `paperTowelEmbossCellsPerTile` cells around the roll — this file owns the COUNT, the mesh owns
/// the metres, and the cell count is the only thing the two share. At 512² that is 32 texels per
/// diamond, ~0.1 mm per texel: comfortably above Nyquist and well inside the touch band.
///
/// **Coherent, not stochastic.** A regular lattice double-prints under the engine's hex de-repeat
/// (`.fabric` is a coherent category, as velvet's weave is), and needs none: a periodic lattice
/// has no landmark to repeat. A pillow's apex (u, v) = (1/32, 0) is flat — the mesh maps a roll's
/// plain end faces to that one texel. The sheet's own PERFORATION seams are therefore NOT in this tile
/// (they would recur every ~54 mm); they are geometry on the roll.
extension MaterialGenerator {

    /// Diamonds per tile edge. Even, so the two diagonal families share one period.
    public static let paperTowelEmbossCellsPerTile = 16

    /// A quilted-diamond paper towel. Off-white, never pure white — a towel's optical brightener
    /// leaves it a hair warm, and a 1.0 albedo clips under any sun.
    public static func paperTowel(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 1001,
                                  base: Vec3 = Vec3(0.84, 0.83, 0.79)) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .fabric)
        let k = Double(paperTowelEmbossCellsPerTile)

        // Fraction (0 on a line … 0.5 mid-pillow … back to 1 on the next line) across one diagonal
        // family. The lattice repeats every 1/k of u±v.
        func phase(_ a: Double) -> Double { a - floor(a) }

        var height = [Double](repeating: 0, count: size * size)
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                // Two diagonal families, phase in cells. u+v and u−v both repeat every 1/k.
                let a = k * (u + v) * 0.5 + 0.25
                let b = k * (u - v) * 0.5 + 0.25
                let fa = phase(a), fb = phase(b)
                let d = Swift.min(Swift.min(fa, 1 - fa), Swift.min(fb, 1 - fb))   // 0 on a line … 0.5
                // The pressed groove: narrow and crisp, then a domed pillow rising out of it.
                let groove = 1 - smoothstep(0.0, 0.085, d)             // 1 on the line, 0 off it
                let dome = pow(clamp01(sin(Double.pi * fa) * sin(Double.pi * fb)), 0.55)
                // Tissue formation: a whisper of cloud, below the pillow's own relief.
                let formation = Noise.fbmTiled(u * 6, v * 6, baseCells: 6, octaves: 3, seed: seed) - 0.5
                height[ch.idx(x, y)] = clamp01(0.20 + 0.72 * dome * (1 - 0.85 * groove) + 0.05 * formation)
            }
        }
        ch.height = height

        for y in 0..<size {
            for x in 0..<size {
                let i = ch.idx(x, y)
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                let h = height[i]
                // The grooves are compressed fibre: a touch denser, so a touch darker in reflected
                // light; the pillow tops are the loftiest, so the brightest.
                let tone = 0.925 + 0.075 * smoothstep(0.15, 0.85, h)
                let mottle = 1 + 0.02 * (Noise.fbmTiled(u * 5, v * 5, baseCells: 5, octaves: 3,
                                                        seed: seed ^ 0x7A) - 0.5) * 2
                ch.albedo[i] = clampBand(base * tone * mottle)
                // Matte tissue, and the relief drives the finish: the pillow tops are loose, fuzzy
                // fibre (rougher), the pressed grooves are compacted and a touch smoother.
                ch.roughness[i] = clamp01(0.80 + 0.15 * h + 0.2 * (mottle - 1))
            }
        }
        // Loose fibres give paper its faint cloth-like grazing glow, never a gloss.
        ch.sheen = 0.18
        ch.sheenRoughness = 0.55
        ch.deriveNormals(strength: 4.5)
        // The grooves' own micro-occlusion + a fine detail normal, from the SAME height.
        setDetailRelief(&ch, height: height, strength: 2.5, occlusionStrength: 3.0)
        return ch
    }
}
