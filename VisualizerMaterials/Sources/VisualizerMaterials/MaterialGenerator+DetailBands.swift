import simd
import Foundation

extension MaterialGenerator {
    /// **How much of a GLAZED or POLISHED surface's micro-relief may reach the occlusion
    /// channel**, as a fraction of its detail-normal strength.
    ///
    /// Same shape of argument as the metal case in `HouseRenderBridge.instanceData`, and the
    /// same lesson: the detail band's occlusion is a DIFFUSE self-shadowing term, so it is only
    /// physical where the micro-relief forms cavities that can trap light. A ceramic glaze is a
    /// vitreous film that has flowed and levelled — its "orange peel" is a smooth undulation,
    /// not a field of pits. It tilts the surface (which the detail NORMAL carries, and should)
    /// while occluding essentially nothing.
    ///
    /// Ignoring that is visible, not theoretical. With the S2.4 band first switched on at full
    /// strength, the `debugTester-materialsWalk` frame showed dark pepper scattered across the
    /// glazed subway tile, the square tile and the terrazzo chips: on a bright glossy white
    /// surface a 1 mm occlusion minimum does not read as tooth, it reads as **grime**. Those are
    /// the two materials Danny had just called good, so that is a regression, not a lift.
    ///
    /// Matte and porous surfaces — plaster, cork, leather, cloth, concrete, unglazed stone — keep
    /// the full authored strength, because their relief genuinely is cavities and the diffuse
    /// occlusion is the whole reason S2.4 exists.
    public static let glazeDetailOcclusionShare = 0.25

    /// Derive a tangent-space normal map from an arbitrary height array.
    /// Used for detail normals that are independent of the main `MaterialChannels.height`.
    /// Add a high-frequency micro-relief **detail normal** (pores / grit / weave) to a material
    /// that doesn't already author one. The engine samples `detailNormal` at `detailNormalUVScale ×
    /// uv` and blends it over the macro normal, so a 256² tile of fine noise gives close-range
    /// surface life on every wall/floor/counter — the cheap half of the "looks real" lift (audit
    /// worklist #1). No-op if the material already set its own detail normal.
    ///
    /// `grainAspect` stretches every feature by that factor — the anisotropic micro-relief a
    /// directional material needs — ALONG V by default (wood's pore channels run down the board;
    /// a round pit is what made oak read as corrugated), or ALONG U with `grainAlongU` (a brushed
    /// metal's hairline runs with its +U grain tangent). The cross-grain axis gets `aspect ×` the
    /// cells, so the coarsest octave keeps its along-grain count. It is a real `cellsX × cellsY`
    /// lattice (`Noise.fbmTiledAniso`): this used to multiply the coordinate instead, which tiles
    /// one strip `aspect` times — an exact repeat inside the tile (DH-0965).
    static func addMicroDetail(_ ch: inout MaterialChannels, seed: UInt64,
                               baseCells: Int = 96, octaves: Int = 2, strength: Double = 0.9,
                               occlusionStrength: Double? = nil, grainAspect: Int = 1,
                               grainAlongU: Bool = false) {
        guard ch.detailNormal == nil else { return }
        let n = ch.size
        let cross = baseCells * max(1, grainAspect)
        let cellsX = grainAlongU ? baseCells : cross
        let cellsY = grainAlongU ? cross : baseCells
        var h = [Double](repeating: 0, count: n * n)
        for y in 0..<n {
            for x in 0..<n {
                let u = Double(x) / Double(n), v = Double(y) / Double(n)
                h[ch.idx(x, y)] = Noise.fbmTiledAniso(u, v, cellsX: cellsX, cellsY: cellsY,
                                                      octaves: octaves, seed: seed)
            }
        }
        setDetailRelief(&ch, height: h, strength: strength, occlusionStrength: occlusionStrength)
    }

    /// **The ONE place a detail band is written.** Takes the high-frequency height field and
    /// derives BOTH halves of it — the tangent-space `detailNormal` (specular-band micro-tilt)
    /// and S2.4's `detailOcclusion` (diffuse-visible micro-shadowing). One relief, one height
    /// field, one writer, so the two can never describe different surfaces
    /// ([[feedback-single-source-of-truth]]).
    ///
    /// `occlusionStrength` defaults to `strength` — the shared relief amount — and exists for
    /// the one case where those genuinely differ: a surface whose *normal* has to be held to a
    /// whisper for a reason that is specific to the normal. Linen is the worked example; its
    /// `strength: 0.03` is documented in-source as "a stronger detail normal sparkles into a
    /// dirty grain on a dimly-lit grazing panel", which is a statement about a specular-band
    /// artefact, not about how deep the weave is. The occlusion has no such failure mode (it
    /// multiplies the diffuse and mip-filters correctly), so inheriting that number would let
    /// a defect in one consumer silently suppress the other. Set it only with that kind of
    /// reason written down.
    static func setDetailRelief(_ ch: inout MaterialChannels, height: [Double], strength: Double,
                                occlusionStrength: Double? = nil) {
        ch.detailNormal = deriveNormalMap(height: height, size: ch.size, strength: strength)
        ch.detailOcclusion = deriveDetailOcclusion(height: height, size: ch.size,
                                                   strength: occlusionStrength ?? strength)
    }

    /// Micro-occlusion from a height field: how much of the hemisphere a texel loses to the
    /// micro-relief immediately around it.
    ///
    /// **The measure** is depth below the LOCAL surface level — `blur_R(h) − h` — not height
    /// against the tile mean. A pit is dark because its own rim occludes it; a low-lying but
    /// locally flat region is not occluded at all, and an absolute-height reading would wrongly
    /// darken it (and would double-print the macro tone the albedo already carries). The blur
    /// radius is one detail feature wide, so the term is band-limited to the detail frequency.
    ///
    /// **Amount** is driven by the same `strength` the normal uses — one relief authored once,
    /// read two ways — and clamped to `maxOcclusion` so no generator can crush a texel to black.
    /// The result is ≤ 1 everywhere with a mean a little under 1, which is exactly what makes it
    /// mip-safe: coarse levels converge to that mean instead of to "no effect".
    static func deriveDetailOcclusion(height: [Double], size: Int, strength: Double) -> [Double] {
        let maxOcclusion = 0.5
        // One feature wide. `addMicroDetail`'s coarsest octave is ~60–130 cells across the
        // tile, so a feature is size/130 … size/60 texels; radius = size/128 sits at the fine
        // end of that and stays ≥ 1 for the 64² bakes the tests use.
        let r = max(1, size / 128)
        let n = size
        let wrap: (Int) -> Int = { i in ((i % n) + n) % n }
        // Separable box blur — toroidal, matching how the tile is sampled.
        var rowBlur = [Double](repeating: 0, count: n * n)
        let inv = 1.0 / Double(2 * r + 1)
        for y in 0..<n {
            for x in 0..<n {
                var s = 0.0
                for d in -r...r { s += height[y * n + wrap(x + d)] }
                rowBlur[y * n + x] = s * inv
            }
        }
        var depth = [Double](repeating: 0, count: n * n)
        for y in 0..<n {
            for x in 0..<n {
                var s = 0.0
                for d in -r...r { s += rowBlur[wrap(y + d) * n + x] }
                depth[y * n + x] = Swift.max(0, s * inv - height[y * n + x])
            }
        }
        // Normalise by the field's OWN depth RMS, then scale to a target contrast.
        //
        // This is deliberate and it is what the detail NORMAL already does implicitly:
        // `deriveNormalMap` normalises every gradient to a unit vector, so `strength` sets
        // the micro-TILT and the height field's absolute amplitude drops out. Reading raw
        // height differences here put the occlusion on a completely different footing — the
        // library's baked detail fields differ 16× in depth RMS, so a fixed gain gave linen
        // σ 0.004 (invisible) and asphalt σ 0.093 (dirt). Measured on the real Metal path,
        // the added micro-contrast tracks σ almost exactly, so an un-normalised bake means
        // "does this material show micro-relief?" is decided by an incidental authoring
        // amplitude rather than by the surface.
        //
        // `strength` still modulates — a generator that asks for gentle relief gets gentler
        // occlusion — but within a band, so no material lands at invisible or at grime.
        // …but a strength of ZERO means OFF, not "the floor". The band below floors at 0.30 so
        // that a generator asking for gentle relief still gets *visible* relief — but that
        // floor made `occlusionStrength: 0` yield contrast 0.048 instead of nothing, so a
        // material could not opt out at all. That is a real trap: it silently gave interior
        // PAINT a micro-occlusion and regressed the wall-splotch blotch 0.07 → 0.25, undoing
        // a third of the SSAO hemisphere fix. Every other opt-in term in this engine treats 0
        // as an exact no-op; so does this one now.
        guard strength > 0 else { return [Double](repeating: 1, count: n * n) }
        let rms = (depth.reduce(0) { $0 + $1 * $1 } / Double(n * n)).squareRoot()
        let contrast = 0.16 * Swift.min(1.4, Swift.max(0.30, max(0, strength) / 0.7))
        let gain = rms > 1e-9 ? contrast / rms : 0
        var occ = [Double](repeating: 1, count: n * n)
        for i in 0..<(n * n) {
            occ[i] = 1 - Swift.min(maxOcclusion, depth[i] * gain)
        }
        // MEAN-NEUTRALISE. Micro-occlusion must modulate a surface AROUND its value, not
        // lower it: the material's overall lightness is the author's decision (and, for a
        // metal, F0 itself), while this band's job is purely the structure. Un-normalised, the
        // field's mean sits below 1 and every material silently darkens — measured, that
        // pushed matte-black's specular IBL *below its single-scatter level* (0.01043 vs the
        // 0.010908 → 0.012121 the multi-scatter fix had just won, i.e. it read as if S1.3b had
        // been reverted) and shifted a whole test frame by 20 levels. Both were real gate
        // failures, and both were this. Re-centring costs nothing: the spatial structure —
        // which is the entire point — is untouched, only the DC term is.
        let occMean = occ.reduce(0, +) / Double(n * n)
        if occMean > 1e-6 {
            for i in 0..<(n * n) { occ[i] = Swift.min(1.0, occ[i] / occMean) }
        }
        return occ
    }

    static func deriveNormalMap(height: [Double], size: Int, strength: Double) -> [Vec3] {
        var normals = [Vec3](repeating: Vec3(0, 0, 1), count: size * size)
        let wrap: (Int) -> Int = { i in ((i % size) + size) % size }
        for y in 0..<size {
            for x in 0..<size {
                let hL = height[wrap(y) * size + wrap(x - 1)]
                let hR = height[wrap(y) * size + wrap(x + 1)]
                let hD = height[wrap(y - 1) * size + wrap(x)]
                let hU = height[wrap(y + 1) * size + wrap(x)]
                normals[y * size + x] = normalize3(Vec3((hL - hR) * strength, (hD - hU) * strength, 1))
            }
        }
        return normals
    }
}
