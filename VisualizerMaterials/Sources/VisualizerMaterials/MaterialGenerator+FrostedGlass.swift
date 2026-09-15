import Foundation

extension MaterialGenerator {
    /// **Frosted (acid-etched / sandblasted) glass** — the milky diffusing pane of a bathroom
    /// window, and the shade of an opal-globe light. Same float-glass base as `clearGlass`; the
    /// surface is then etched into a dense micro-relief, so the tile's ROUGHNESS sits in the
    /// diffusing band (≈ 0.5–0.7, a fine grain rather than a wave) and its albedo lifts toward
    /// milk — light scatters at the surface instead of passing through it.
    ///
    /// The pane's REAL optics (IOR, transmission roughness, density) are `GlassMaterial.optics`
    /// on the app side; like `clearGlass`, this map feeds the swatch and the raster lane. The
    /// clearcoat is low: an etched face has no polished top layer to reflect the room.
    public static func frostedGlass(size: Int = MaterialGenerator.bakeSize,
                                    seed: UInt64 = 331) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .glass)
        let milk = Vec3(0.86, 0.88, 0.89)
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size), v = (Double(y) + 0.5) / Double(size)
                // Etch grain: a fine, high-frequency field (the blast pattern) over a faint
                // long-wave undulation (the float glass underneath).
                let grain = Noise.fbmTiled(u, v, baseCells: 96, octaves: 3, seed: seed)
                let wave = Noise.fbmTiled(u, v, baseCells: 4, octaves: 3, seed: seed &+ 7)
                ch.albedo[ch.idx(x, y)] = clampBand(milk * (0.96 + 0.06 * grain))
                ch.roughness[ch.idx(x, y)] = clamp01(0.52 + 0.16 * grain + 0.04 * (wave - 0.5))
                ch.height[ch.idx(x, y)] = clamp01(0.5 + 0.06 * (grain - 0.5) + 0.02 * (wave - 0.5))
            }
        }
        ch.clearcoat = 0.15                // etched: no polished top layer
        ch.deriveNormals(strength: 0.6)
        // The etch grain as DETAIL relief too (normal + occlusion, through the one helper every
        // matte material uses) — a diffusing pane without it reads as flat plastic up close.
        addMicroDetail(&ch, seed: seed ^ 0xE3, baseCells: 110, strength: 0.35)
        return ch
    }
}
