import Foundation

/// **Daydream DH-1316 — what made velvet and linen read as velvet and linen in the reference photos,
/// and was missing from the bakes.**
///
/// Read against close-ups of real upholstery (velvet armchairs and a crushed-velvet macro; a linen
/// macro — SOURCES in the ticket):
///
/// • **Velvet has a NAP.** The pile leans one way, and where it was brushed or sat on it leans
///   another, in soft patches a few centimetres across. Face-on that is invisible; at grazing the
///   patches answer the light differently — the light/dark "pile marks" on every velvet seat and the
///   reason two panels of one chair, cut with the nap running different ways, read as two tones.
///   The bake had one uniform pile direction, so the sheen rim was one even band.
///   → `applyNapLean`: a smooth lean field tilts the pile normal a few degrees (one dominant
///   direction, swirled in patches), and a crushed patch is a touch lighter face-on.
/// • **Linen is slubby.** Flax yarn swells and thins along its length, so a linen cloth is crossed by
///   short, irregular lighter streaks along the weft (and a few along the warp), and each yarn is its
///   own shade of flax — the cloth is heathered, not one flat oatmeal. The bake had an isotropic fbm
///   "slub" (round blobs) and a perfectly uniform yarn colour.
///   → `linenSlub`: per-yarn tone + elongated thick/thin streaks, thread-aligned.
///
/// `refinedFabricLook` is the A/B lever (a GPU A/B must also drop the bridge's slice cache).
extension MaterialGenerator {

    /// DH-1316 lever: the nap lean (velvet) and the slub / heathered yarn (linen). `false` (the default,
    /// so every host's bake is byte-identical until it opts in) bakes the pre-DH-1316 channels exactly.
    nonisolated(unsafe) public static var refinedFabricLook: Bool = false

    /// Tilt a pile's tangent-space normals by a nap-lean field (see the type note). `lean` is the
    /// typical tilt (tangent of the angle), `angle` the dominant direction (radians from +U),
    /// `patchCells` the patch lattice over the tile, `swirl` how far a patch turns away from the
    /// dominant direction (radians), `crushLift` how much lighter a fully crushed patch reads.
    static func applyNapLean(_ ch: inout MaterialChannels, seed: UInt64, angle: Double, lean: Double,
                             patchCells: Int, swirl: Double, crushLift: Double) {
        let n = ch.size
        for y in 0..<n {
            for x in 0..<n {
                let u = (Double(x) + 0.5) / Double(n), v = (Double(y) + 0.5) / Double(n)
                let turn = (Noise.fbmTiled(u, v, baseCells: patchCells, octaves: 2, seed: seed) - 0.5) * 2
                let crush = Noise.fbmTiled(u, v, baseCells: patchCells, octaves: 2, seed: seed ^ 0x5A17)
                let phi = angle + swirl * turn
                let mag = lean * (0.45 + 1.1 * crush)
                let i = ch.idx(x, y)
                let nn = ch.normal[i]
                ch.normal[i] = normalize3(Vec3(nn.x + mag * cos(phi), nn.y + mag * sin(phi), nn.z))
                ch.albedo[i] = clampBand(ch.albedo[i] * (1 + crushLift * (crush - 0.5) * 2))
            }
        }
    }

    /// Linen's yarn character at texel (x, y): `tone` (multiplies albedo — per-yarn heather and the
    /// thick / thin streaks), `lift` (added to height — a slub stands proud) and `gloss` (subtracted
    /// from roughness — flax is a lustrous fibre, so a fat slub catches a little light). Yarns are
    /// ~2 texels (≈ 1.2 mm on the 0.30 m upholstery tile); a slub runs ~1–3 cm along its yarn.
    static func linenSlub(x: Int, y: Int, size: Int, seed: UInt64) -> (tone: Double, lift: Double, gloss: Double) {
        let u = (Double(x) + 0.5) / Double(size), v = (Double(y) + 0.5) / Double(size)
        let yarns = max(2, size / 2)
        // Weft (along +U): long cells along the yarn, one cell row per yarn.
        let w = Noise.fbmTiledAniso(u, v, cellsX: 24, cellsY: yarns, octaves: 2, seed: seed)
        let thickW = smoothstep(0.60, 0.80, w), thinW = smoothstep(0.62, 0.82, 1 - w)
        // Warp (along +V): sparser and weaker — upholstery linen's slub shows mostly in the weft.
        let p = Noise.fbmTiledAniso(u, v, cellsX: yarns, cellsY: 24, octaves: 2, seed: seed ^ 0x3C3C)
        let thickP = smoothstep(0.66, 0.84, p)
        // Each yarn its own shade of flax (heather), weft stronger than warp.
        let rowTone = Noise.unit(Noise.mix(Noise.hash2(0, y / 2, seed ^ 0x77)))
        let colTone = Noise.unit(Noise.mix(Noise.hash2(x / 2, 0, seed ^ 0x99)))
        let tone = (1 + 0.045 * (rowTone - 0.5) * 2) * (1 + 0.025 * (colTone - 0.5) * 2)
                 * (1 + 0.08 * thickW - 0.06 * thinW + 0.05 * thickP)
        return (tone, 0.14 * thickW + 0.08 * thickP - 0.05 * thinW, 0.05 * thickW + 0.03 * thickP)
    }
}
