import simd
import Foundation

/// The §8 material auditors made numeric and headless — a baked `MaterialChannels`
/// gets the same treatment as a mesh (`MeshAudit`) or the shipped palette
/// (`MaterialAudit`). Each field measures one of the §2 plausibility tells so it can't
/// silently regress: albedo out of physical range (#4), flat uniform roughness (#3),
/// a visible tile seam (#5), a dead flat single color (#5 macro), and broken normals.
public struct TextureAudit: Equatable, Sendable {
    public var size: Int

    /// Pixels whose albedo leaves the dielectric band `[0.04, 0.94]` (pure black/white
    /// read CG). Skipped for metals/glass, whose look lives in spec/transmission.
    public var albedoOutOfRange: Int
    /// Std-dev of the roughness map. Real surfaces vary; a flat scalar is tell #3.
    public var roughnessStdDev: Double
    /// Std-dev of albedo luma — the macro tonal variation that stops a big surface
    /// reading as one dead flat color (tell #5 / §4 macro layer).
    public var macroContrast: Double
    /// Seam score per axis: edge-wrap pixel difference ÷ mean interior-neighbor
    /// difference. ≈1 means the wrap is as smooth as the interior (seamless); a large
    /// value is a visible repeat boundary (tell #5).
    public var seamScoreX: Double
    public var seamScoreY: Double
    /// Worst normal-length error from unit (a broken normal map shades wrong).
    public var normalUnitError: Double
    /// Whether this material family should carry a grain tangent (wood, brushed metal).
    public var expectsGrain: Bool
    /// Axial order parameter of the grain tangent (0 random … 1 perfectly aligned).
    public var grainCoherence: Double
    /// An anisotropic family missing a coherent grain tangent (tell #7).
    public var missingGrain: Bool
    /// Whether the second-lobe scalars (clearcoat, sheen) are in `[0, 1]`.
    public var lobesInRange: Bool
    /// Mean albedo luma — pairs with `macroContrast` to judge flatness *relative* to
    /// brightness, so a legitimately dark material (velvet) isn't penalized for the
    /// small absolute luma spread that darkness alone imposes.
    public var lumaMean: Double
    /// **Albedo↔roughness structural correlation (DH-0460).** Signed Pearson correlation
    /// of the HIGH-PASSED albedo-luma and roughness fields — the two channels' *structure*,
    /// with the broad independent drifts (board tone, `linen`'s low-frequency roughness
    /// patches) removed by `structuralHighPassLevels` of box low-pass first.
    ///
    /// **Why this exists.** `roughnessStdDev` already fails a FLAT roughness map, but a map that
    /// varies by its own low-frequency noise field passes it while still reading as "the same
    /// varnished plastic everywhere": in a real material the same physical structure that draws
    /// the albedo (oak's earlywood pore band, paint's roller texture) also draws the specular
    /// response, so the highlight has the surface's grain in it. `|corr|` near 0 means the two
    /// channels share no structure; a value toward ±1 means the grain that is visible in the
    /// colour is also visible in the reflection. The SIGN is physical and not judged — oak's
    /// dark latewood is also its rougher band (negative), a pore both darkens and roughens.
    ///
    /// 0 when either high-passed channel is effectively flat (no structure to correlate).
    public var albedoRoughnessCorr: Double = 0
    /// The material family being audited. Carried because two of the tells are DIELECTRIC
    /// SURFACE tells, not universal ones — see `flatnessIsCorrect`. Defaulted so every existing
    /// construction site keeps compiling.
    public var category: MaterialCategory = .paint

    // Thresholds — the bar each tell must clear. Tuned against the shipped library.
    public static let roughnessFloor = 0.012   // below → effectively flat
    public static let macroFloor     = 0.008   // absolute floor (kept for reference/tests)
    public static let macroCVFloor   = 0.012   // relative floor: SD ÷ mean luma
    public static let seamCeiling     = 3.0     // above → the wrap reads as a seam
    public static let normalTolerance = 1e-3
    public static let grainFloor      = 0.5     // below → grain direction is incoherent
    /// Box low-pass octaves removed before the albedo↔roughness correlation is taken, so the
    /// metric sees only the shared STRUCTURE band (grain/weave/tooth) and not the broad,
    /// legitimately-independent drifts (board tone, a cushion's low-frequency roughness patches).
    /// 3 removes everything coarser than 2³ = 8 texels; the grain and weave bands survive.
    public static let structuralHighPassLevels = 3
    /// **|corr| a category that MUST share structure has to clear.** Below → the roughness map
    /// is drawn from its own noise field, decoupled from the grain that draws the colour
    /// (DH-0460). Deliberately modest: even a real wood only puts a fraction of its grain into
    /// the finish, so this catches "roughness is a totally separate field", not "roughness could
    /// track the grain harder". Tuned against the shipped library (see `MaterialCorrelationTests`).
    public static let correlationFloor = 0.15

    /// **The one family for which "dead flat, single colour" is CORRECT.** A mirror is silvered
    /// plate: uniform roughness and uniform albedo are what make it a mirror, and the only way to
    /// satisfy the flatness tells below would be to frost it. These two tells exist to catch a
    /// wall or a floor baked as a flat scalar (which reads as plastic) — they are surface-texture
    /// tells, so they are scoped to surfaces that have texture. Same exemption shape as
    /// `albedoOutOfRange`, which metals and glass already skip.
    public var flatnessIsCorrect: Bool { category == .mirror }

    /// **Near-uniform TONE is correct for paint, and forcing tone variation on it is what made
    /// walls read as sponge-painting** (Danny, 2026-08-05: "someone has painted the entire thing
    /// with a sponge… a lot of the parts look like black mold"). Measured on the real Metal path
    /// (`HouseRenderBridgeGPUTests_WallSplotch`): the mottle was the paint generator's albedo
    /// `tooth`, which existed only to clear `macroCVFloor` — and the 2026-07-13 pass at the same
    /// complaint MOVED that variance rather than removing it ("Same total albedo SD as before
    /// (gate stays green)"), trading a metre-scale blotch for a 9 cm stipple. The gate was the
    /// thing driving the defect, so the gate is what changes.
    ///
    /// Rolled paint on drywall genuinely IS near-uniform in colour: what stops it reading as
    /// plastic is its ROUGHNESS breakup and relief, not tonal blotch. So the albedo tell is
    /// lifted for `.paint` while the roughness tell below deliberately still binds — that is the
    /// whole point of splitting these two, and it is why this is NOT folded into
    /// `flatnessIsCorrect` (a paint bake with flat roughness is still a bug).
    /// Extended to `.metal` 2026-08-05, for the same physical reason and on the same
    /// evidence. **A metal's albedo IS its spectral reflectance, which is uniform by
    /// nature** — what makes brushed steel read as brushed steel is anisotropic ROUGHNESS
    /// and grain direction, not tonal blotch. The S2.1 audit found `matte-black` carrying
    /// 8.51 % coarse albedo CV with 80 % of it low-frequency and unstructured (a 3× luma
    /// swing at decimetre scale on a near-black surface — the "black mold" report), and
    /// the patina metals `copper`/`brass`/`bronze` at 94–96 % low-frequency wash. That is
    /// the same mandated-defect signature paint had: tone added to clear `macroCVFloor`.
    ///
    /// `roughnessIsFlat` deliberately still binds on metals — a metal baked with flat
    /// roughness is exactly the injection-moulded-plastic tell, and that guard is the
    /// whole reason this is a separate property from `flatnessIsCorrect`.
    public var flatColorIsCorrect: Bool {
        flatnessIsCorrect || category == .paint || category == .metal
    }

    /// **The families for which the colour's structure MUST drive the finish's structure**
    /// (DH-0460). A grained solid — wood and its laminate print — has one physical structure
    /// (the growth ring, the pore band) that produces both channels; roughness drawn from an
    /// independent field is the defect this catches. Everything else is left exempt for now, not
    /// because its roughness *should* be decoupled but because whether it does is a per-generator
    /// judgement the ranking is meant to settle first (paint's roller tooth, plaster's trowel,
    /// fabric's weave are the filed follow-ups) — the gate binds only where the physics is
    /// unambiguous, exactly as `flatColorIsCorrect` scopes the tone tell. Flat/near-flat
    /// materials (mirror, glass, a legitimately smooth metal) have no structure to correlate and
    /// are covered by `correlationIsMeasurable`.
    public var expectsCorrelatedRoughness: Bool {
        category == .wood || category == .laminate
    }
    /// True only when BOTH channels actually carry high-passed structure — otherwise |corr| is
    /// undefined (0) and must not be read as a decoupling failure. `audit` reports 0 in that case.
    public var correlationIsMeasurable: Bool { abs(albedoRoughnessCorr) > 1e-9 }
    /// A grained material whose finish structure is decoupled from its colour structure (tell —
    /// DH-0460). Scoped to `expectsCorrelatedRoughness`; measured, not assumed.
    public var roughnessDecoupled: Bool {
        expectsCorrelatedRoughness && correlationIsMeasurable
            && abs(albedoRoughnessCorr) < Self.correlationFloor
    }

    public var roughnessIsFlat: Bool { !flatnessIsCorrect && roughnessStdDev < Self.roughnessFloor }
    /// Dead flat single color, judged on the coefficient of variation (brightness-fair).
    public var isFlatColor: Bool {
        !flatColorIsCorrect && macroContrast / max(lumaMean, 0.05) < Self.macroCVFloor
    }
    public var isSeamy: Bool         { max(seamScoreX, seamScoreY) > Self.seamCeiling }
    public var normalsAreUnit: Bool  { normalUnitError < Self.normalTolerance }

    public var isPlausible: Bool {
        albedoOutOfRange == 0 && !roughnessIsFlat && !isFlatColor && !isSeamy
            && normalsAreUnit && !missingGrain && lobesInRange && !roughnessDecoupled
    }

    @inlinable static func luma(_ c: Vec3) -> Double { 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }

    public static func audit(_ ch: MaterialChannels) -> TextureAudit {
        let n = ch.size
        let metalOrGlass = ch.category == .metal || ch.category == .mirror || ch.category == .glass

        // albedo range (dielectric band only)
        var outOfRange = 0
        if !metalOrGlass {
            for c in ch.albedo {
                for v in [c.x, c.y, c.z] where v < 0.04 || v > 0.94 { outOfRange += 1 }
            }
        }

        func stdDev(_ xs: [Double]) -> Double {
            guard !xs.isEmpty else { return 0 }
            let m = xs.reduce(0, +) / Double(xs.count)
            let varc = xs.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(xs.count)
            return varc.squareRoot()
        }
        let roughSD = stdDev(ch.roughness)
        let lumaField = ch.albedo.map(luma)
        let macroSD = stdDev(lumaField)
        let lumaMean = lumaField.isEmpty ? 0 : lumaField.reduce(0, +) / Double(lumaField.count)

        // seam score: the edge-wrap difference vs the *mean* interior-neighbor difference
        // along each line. Averaging over every interior pair (not one sample) means a
        // legitimate hard edge inside the tile — a board joint, a grout line — counts in
        // the baseline, so only a wrap that's worse than the interior reads as a seam.
        func seam(horizontal: Bool) -> Double {
            var edge = 0.0, interior = 0.0
            for a in 0..<n {
                let p0 = horizontal ? ch.idx(n - 1, a) : ch.idx(a, n - 1)
                let p1 = horizontal ? ch.idx(0, a)     : ch.idx(a, 0)
                edge += abs(lumaField[p0] - lumaField[p1])
                var line = 0.0
                for k in 0..<(n - 1) {
                    let q0 = horizontal ? ch.idx(k, a) : ch.idx(a, k)
                    let q1 = horizontal ? ch.idx(k + 1, a) : ch.idx(a, k + 1)
                    line += abs(lumaField[q0] - lumaField[q1])
                }
                interior += line / Double(max(n - 1, 1))
            }
            return edge / max(interior, 1e-6)
        }

        var normErr = 0.0
        for nrm in ch.normal { normErr = max(normErr, abs(len3(nrm) - 1)) }

        // grain: wood is always anisotropic, so it must carry a grain tangent. And *any*
        // material that provides one must keep it coherent so the engine streaks the
        // highlight along it (tell #7) — an incoherent grain is a bug. Coherence is the
        // axial order parameter |mean exp(i·2θ)| — 1 = perfectly aligned, 0 = random.
        // (Polished metals like brass are isotropic and legitimately carry no grain.)
        let expectsGrain = ch.category == .wood || ch.category == .laminate
        var coherence = 0.0
        if let g = ch.grainTangent, !g.isEmpty {
            var sx = 0.0, sy = 0.0
            for t in g { let a = 2 * atan2(t.y, t.x); sx += cos(a); sy += sin(a) }
            coherence = (sx * sx + sy * sy).squareRoot() / Double(g.count)
        }
        let hasGrain = ch.grainTangent != nil
        let missingGrain = (expectsGrain && !hasGrain) || (hasGrain && coherence < Self.grainFloor)
        let lobesInRange = (0...1).contains(ch.clearcoat) && (0...1).contains(ch.sheen)

        // albedo↔roughness structural correlation (DH-0460): high-pass both channels to the
        // shared structure band, then take their signed Pearson correlation.
        let hpLuma = highPass(lumaField, size: n, levels: structuralHighPassLevels)
        let hpRough = highPass(ch.roughness, size: n, levels: structuralHighPassLevels)
        let corr = pearson(hpLuma, hpRough)

        return TextureAudit(size: n, albedoOutOfRange: outOfRange, roughnessStdDev: roughSD,
                            macroContrast: macroSD, seamScoreX: seam(horizontal: true),
                            seamScoreY: seam(horizontal: false), normalUnitError: normErr,
                            expectsGrain: expectsGrain, grainCoherence: coherence,
                            missingGrain: missingGrain, lobesInRange: lobesInRange,
                            lumaMean: lumaMean, albedoRoughnessCorr: corr,
                            category: ch.category)
    }

    /// The high-frequency residual of a square toroidal field: `field − upsample(box-decimate
    /// `levels` times)`, i.e. everything FINER than `2^levels` texels. Same 2×2 box-decimation
    /// as `MaterialScaleAudit.octaveVariances`, so the two auditors band the field identically.
    static func highPass(_ field: [Double], size: Int, levels: Int) -> [Double] {
        guard size > 1, field.count == size * size, levels > 0 else { return field }
        var cur = field
        var n = size
        var lv = 0
        while n > 1 && lv < levels {
            let half = n / 2
            var down = [Double](repeating: 0, count: half * half)
            for y in 0..<half {
                for x in 0..<half {
                    let a = cur[(2 * y) * n + 2 * x], b = cur[(2 * y) * n + 2 * x + 1]
                    let c = cur[(2 * y + 1) * n + 2 * x], d = cur[(2 * y + 1) * n + 2 * x + 1]
                    down[y * half + x] = (a + b + c + d) * 0.25
                }
            }
            cur = down
            n = half
            lv += 1
        }
        // Nearest-upsample the low-pass back to full resolution (block replicate) and subtract.
        let scale = size / n
        var out = [Double](repeating: 0, count: size * size)
        for y in 0..<size {
            for x in 0..<size {
                out[y * size + x] = field[y * size + x] - cur[(y / scale) * n + (x / scale)]
            }
        }
        return out
    }

    /// Signed Pearson correlation of two equal-length fields; 0 when either is flat.
    static func pearson(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, a.count > 1 else { return 0 }
        let cnt = Double(a.count)
        let ma = a.reduce(0, +) / cnt, mb = b.reduce(0, +) / cnt
        var cov = 0.0, va = 0.0, vb = 0.0
        for i in a.indices {
            let da = a[i] - ma, db = b[i] - mb
            cov += da * db; va += da * da; vb += db * db
        }
        guard va > 1e-18, vb > 1e-18 else { return 0 }
        return cov / (va * vb).squareRoot()
    }
}
