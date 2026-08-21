import simd
import Foundation

/// **The SCALE auditor — "is this material's detail drawn at a size a person can see?"**
///
/// `TextureAudit` is a *plausibility floor*: it catches a bake that is flat, seamy, or
/// out of physical range. It is deliberately scale-blind — it measures variance without
/// asking how big, in millimetres, the varying features are. That blindness is what let
/// the whole library pass "plausible" while reading as flat, and it is the same defect
/// class the paint generator was diagnosed with on 2026-08-15: **the root cause was
/// FREQUENCY, not amplitude.** Paint baked 512² over a 3 m span, so every noise band
/// landed at decimetre scale, where any amplitude reads as blotch — and two successive
/// fix attempts cut amplitude, removing the blotch and every trace of life with it.
///
/// The two materials Danny has called good — **tile and terrazzo** — are the two whose
/// generators reason in real millimetres and then take the surface's UV period FROM the
/// size they chose (`TileLayout.targetRunMeters`, `terrazzoRunMeters` = 0.6 m ⇒ 1.17 mm
/// per texel). Every other library material inherited the *surface's* period instead —
/// a wall/floor's nominal 2.0 m, i.e. **3.9 mm per texel** — which makes touch-scale
/// detail not merely absent but unrepresentable: leather pores, linen weave, wool loops,
/// concrete fines and marble micro-veining all live below 3 mm and there is no texel to
/// draw them with.
///
/// So this instrument measures the thing that actually differs, per material:
///
///  1. **`mmPerTexel`** — the bake's physical resolution at the run it renders at.
///  2. **A residual pyramid per channel** (albedo luma, roughness, height): successive
///     2× box decimations, each level's variance taken on the residual it adds over the
///     level above. Octave `L` carries features of wavelength ≈ `2^(L+1)` texels, so its
///     physical size is `mmPerTexel · 2^(L+1)` — the ×2 is not a fudge, it is the
///     Nyquist pair the paint band table already reasons with ("orange peel needs two
///     texels per feature").
///  3. **`touchScaleFraction`** — the share of a material's total detail energy that
///     lands at or below `touchScaleMM`. This is the headline number and the worklist
///     rank: it is what separates a surface that reads as a material from one that reads
///     as a tinted plane with a pattern painted on it.
///
/// The metric is **fail-closed by construction**: a material baking at 3.9 mm per texel
/// has a finest possible octave of 7.8 mm, so its `touchScaleFraction` is exactly 0 no
/// matter how good its generator is. A zero here is never a tuning opinion — it is proof
/// that the material's resolution budget was spent somewhere the eye cannot use it.
///
/// GPU-free, deterministic, and runs in `swift test`, like `MeshAudit` / `FacetingAudit` /
/// `PlaceableAudit`. Ranking every material worst-first by `touchScaleFraction` IS the
/// material-quality worklist, the same way `FacetingAudit`'s chord-error table is the
/// smoothness worklist.
public struct MaterialScaleAudit: Equatable, Sendable {
    /// Bake resolution, pixels per side.
    public var size: Int
    /// World metres one bake spans on the surface it renders on.
    public var runMeters: Double
    /// Millimetres of real surface per baked texel — `runMeters · 1000 ÷ size`.
    public var mmPerTexel: Double

    /// Residual variance per octave, index 0 = finest. One entry per pyramid level.
    public var albedoOctaves: [Double]
    public var roughnessOctaves: [Double]
    public var heightOctaves: [Double]

    /// The physical feature size, in millimetres, of each octave index.
    public var octaveMM: [Double]

    /// **Detail at or below `touchScaleMM`, as a fraction of all detail energy.** The
    /// headline number. Computed over the three channels jointly, each normalised by its
    /// own total first so a channel with large absolute swing (albedo on a dark material,
    /// height on a deep relief) cannot drown the others out.
    public var touchScaleFraction: Double
    /// The same share for each channel separately — a material can carry fine ROUGHNESS
    /// breakup with no fine albedo (correct for paint and for metals, whose albedo is
    /// spectral reflectance and uniform by nature), so the split is what tells a correct
    /// near-uniform-tone material apart from an under-resolved one.
    public var albedoTouchFraction: Double
    public var roughnessTouchFraction: Double
    public var heightTouchFraction: Double

    // MARK: - the DETAIL band (the second, finer sampling)

    /// **The fine band that does not live in the macro channels.** `detailNormal` /
    /// `detailOcclusion` are sampled shader-side at `detailNormalUVScale × uv` (8 by default),
    /// so they tile at `runMeters ÷ detailUVScale` and DO reach the touch band even when the
    /// macro channels cannot. Measuring the macro pyramid alone would therefore libel every
    /// material that carries one — which is nearly all of them (`addMicroDetail`).
    ///
    /// What it does NOT carry is the point: the detail band is relief only. There is no fine
    /// **albedo** and no fine **roughness** in it, so a surface whose macro texel is 3.9 mm has
    /// no touch-scale TONE and no touch-scale specular breakup by any route — only a micro-shadow
    /// multiply. That asymmetry, not the band's absence, is the real finding.
    public var hasDetailBand: Bool
    /// Physical feature size of the detail band, millimetres.
    public var detailFeatureMM: Double
    /// SD of `detailOcclusion` — the band's DIFFUSE-visible amplitude (S2.4). The detail
    /// normal is a specular-band effect measured at ~1.00× on every dielectric
    /// ([[daydream-detail-normals-metal-only]]), so on a matte surface this is the only part
    /// of the band that reads at all.
    public var detailOcclusionSD: Double
    /// Mean tangent-plane tilt of `detailNormal` (0 = flat).
    public var detailNormalTilt: Double

    /// Fine TONE is absent: nothing in any channel varies the surface's colour at touch scale.
    public var fineAlbedoIsAbsent: Bool { albedoTouchFraction <= 0 }
    /// Fine ROUGHNESS is absent: the specular lobe is uniform across the touch band.
    public var fineRoughnessIsAbsent: Bool { roughnessTouchFraction <= 0 }

    /// The finest octave the bake can represent at all, in millimetres. Equals
    /// `mmPerTexel · 2` — below this the material is not under-authored, it is
    /// under-RESOLVED, and no generator change can reach the band.
    public var finestRepresentableMM: Double

    public var category: MaterialCategory

    /// The band a viewer resolves as *texture* (tooth, weave, pore, grain) rather than as
    /// *pattern* (a plank, a tile, a vein). Detail coarser than this still matters — it is
    /// what stops a surface reading as one dead tone — but it is not what makes a surface
    /// read as a real material up close, and it is the band this library systematically
    /// lacked. 3 mm at a ~1.5 m viewing distance is a few arc-minutes: comfortably visible.
    public static let touchScaleMM = 3.0

    /// The floor a textured dielectric surface must clear. Set from the measured reference
    /// pair rather than invented: terrazzo and tile — the two Danny called good — are the
    /// materials this is calibrated against, and it is set with margin BELOW them so the
    /// gate proves the band EXISTS without dictating how much of it a generator uses.
    public static let touchScaleFloor = 0.10

    /// Families for which a near-zero `touchScaleFraction` is physically CORRECT, not a
    /// defect — the same exemption shape `TextureAudit.flatnessIsCorrect` uses, and for the
    /// same reason: these tells are SURFACE-TEXTURE tells, so they scope to surfaces that
    /// have texture. A mirror is silvered plate and a sheet of glass is optically smooth;
    /// authoring millimetre tooth onto either is how you get frosted glass, which is a
    /// different material rather than a better one.
    ///
    /// ⚠️ **This category test is NOT the whole exemption list, and reading it as one will make
    /// you break things.** Whether a given material may carry a fine band at all is decided by
    /// `MaterialMicroReliefTests.classification` — a fail-closed, per-id table. `chrome`,
    /// `brass`, `copper`, `bronze` and `quartz-counter` are `.intentionallySmooth` polished
    /// finishes whose pitting and patina live in the MACRO channels, and `grass` / `lawn` are
    /// `.reliefViaGeometry` (real blades). All of those read ~0 here and all of them are
    /// correct. This audit measures resolution; that table holds the intent.
    public var flatIsCorrect: Bool { category == .mirror || category == .glass }

    /// Under-resolved: the bake physically cannot draw the touch-scale band. Distinct from
    /// "authored flat" and the distinction is the whole point — this one is not fixable in
    /// the generator, only in the run.
    public var isUnderResolved: Bool {
        !flatIsCorrect && finestRepresentableMM > Self.touchScaleMM
    }

    /// Clears the bar: it can represent the band AND actually puts detail there.
    public var hasTouchScaleDetail: Bool {
        flatIsCorrect || (!isUnderResolved && touchScaleFraction >= Self.touchScaleFloor)
    }

    // MARK: - measurement

    /// Audit a baked material at the world run it will actually render at.
    ///
    /// `runMeters` is not optional and has no default on purpose: a scale audit taken at an
    /// assumed run measures nothing, and defaulting it is how the wrong number gets quoted
    /// with confidence. Callers pass `MaterialRegistry.runMeters(for:)`, the single source.
    /// The engine's default `IlluminatoramaMaterial.detailNormalUVScale`. Mirrored here (not
    /// imported — `DaydreamCore` does not depend on the renderer) so the audit reports the band
    /// at the frequency the shader actually samples it.
    public static let detailUVScaleDefault = 8.0

    public static func audit(_ ch: MaterialChannels, runMeters: Double,
                             detailUVScale: Double = MaterialScaleAudit.detailUVScaleDefault) -> MaterialScaleAudit {
        let n = ch.size
        let run = Swift.max(runMeters, 1e-4)
        let mmPerTexel = run * 1000 / Double(n)

        let luma = ch.albedo.map(TextureAudit.luma)
        let albedoOct = octaveVariances(luma, size: n)
        let roughOct = octaveVariances(ch.roughness, size: n)
        let heightOct = octaveVariances(ch.height, size: n)

        // Octave L is the residual the pyramid adds at level L; its wavelength is ~2^(L+1)
        // texels (a residual is a difference between two scales, so its finest content sits
        // at the Nyquist pair of the level below it).
        let levels = albedoOct.count
        let octaveMM = (0..<levels).map { mmPerTexel * pow(2, Double($0 + 1)) }

        /// Share of one channel's residual energy at or below the touch-scale band.
        func fineShare(_ oct: [Double]) -> Double {
            let total = oct.reduce(0, +)
            guard total > 1e-18 else { return 0 }
            var fine = 0.0
            for (i, v) in oct.enumerated() where octaveMM[i] <= touchScaleMM { fine += v }
            return fine / total
        }
        let aFine = fineShare(albedoOct)
        let rFine = fineShare(roughOct)
        let hFine = fineShare(heightOct)

        // Joint figure: mean of the per-channel shares over the channels that carry ANY
        // energy. Normalising per channel first is what stops a single loud channel from
        // deciding the verdict — an albedo swing of 0.3 and a roughness swing of 0.02 are
        // equally legible on screen, because roughness moves the specular lobe, not the tone.
        var shares: [Double] = []
        if albedoOct.reduce(0, +) > 1e-18 { shares.append(aFine) }
        if roughOct.reduce(0, +) > 1e-18 { shares.append(rFine) }
        if heightOct.reduce(0, +) > 1e-18 { shares.append(hFine) }
        let joint = shares.isEmpty ? 0 : shares.reduce(0, +) / Double(shares.count)

        // The detail band. Its tile is `run ÷ detailUVScale` wide, so one of ITS texels is that
        // much smaller; a feature is the same two-texel Nyquist pair used above.
        let detailMMPerTexel = mmPerTexel / detailUVScale
        var occSD = 0.0
        if let occ = ch.detailOcclusion, !occ.isEmpty {
            let m = occ.reduce(0, +) / Double(occ.count)
            occSD = (occ.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(occ.count)).squareRoot()
        }
        var tilt = 0.0
        if let dn = ch.detailNormal, !dn.isEmpty {
            tilt = dn.reduce(0) { $0 + (($1.x * $1.x + $1.y * $1.y).squareRoot()) } / Double(dn.count)
        }

        return MaterialScaleAudit(
            size: n, runMeters: run, mmPerTexel: mmPerTexel,
            albedoOctaves: albedoOct, roughnessOctaves: roughOct, heightOctaves: heightOct,
            octaveMM: octaveMM, touchScaleFraction: joint,
            albedoTouchFraction: aFine, roughnessTouchFraction: rFine,
            heightTouchFraction: hFine,
            hasDetailBand: ch.detailNormal != nil,
            detailFeatureMM: detailMMPerTexel * 2,
            detailOcclusionSD: occSD, detailNormalTilt: tilt,
            finestRepresentableMM: mmPerTexel * 2, category: ch.category)
    }

    /// Residual variance per octave of a square toroidal field, finest first.
    ///
    /// Public because the same decomposition is the right instrument on a FRAME as well as on
    /// a bake, and the two questions are different: a bake-side number says detail was
    /// AUTHORED, a frame-side one says it was DELIVERED — through the mip chain, the hex
    /// anti-tiling blend (three offset samples averaged, which costs contrast by construction),
    /// the lighting and the tone curve. `HouseRenderBridgeGPUTests_ExteriorLight` measures the
    /// yard's grain with it.
    ///
    /// Each level is a 2×2 box decimation of the one above (a Haar low-pass, which is exact
    /// and allocation-cheap); the octave's energy is the variance of what that decimation
    /// THREW AWAY — `level_L − upsample(level_{L+1})`. Summing the residual variances
    /// recovers the field's total variance to rounding, so the shares below are a real
    /// decomposition rather than a set of overlapping band-pass readings.
    public static func octaveVariances(_ field: [Double], size: Int) -> [Double] {
        guard size > 1, field.count == size * size else { return [] }
        var out: [Double] = []
        var cur = field
        var n = size
        while n > 1 {
            let half = n / 2
            var down = [Double](repeating: 0, count: half * half)
            for y in 0..<half {
                for x in 0..<half {
                    let a = cur[(2 * y) * n + 2 * x]
                    let b = cur[(2 * y) * n + 2 * x + 1]
                    let c = cur[(2 * y + 1) * n + 2 * x]
                    let d = cur[(2 * y + 1) * n + 2 * x + 1]
                    down[y * half + x] = (a + b + c + d) * 0.25
                }
            }
            // Residual variance. The residual's mean is 0 by construction (each 2×2 block's
            // deviations from its own mean sum to zero), so the variance is the mean square.
            var sumSq = 0.0
            for y in 0..<half {
                for x in 0..<half {
                    let m = down[y * half + x]
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let v = cur[(2 * y + dy) * n + 2 * x + dx] - m
                        sumSq += v * v
                    }
                }
            }
            out.append(sumSq / Double(n * n))
            cur = down
            n = half
        }
        return out
    }
}
