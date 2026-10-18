import simd
import Foundation

/// **The REPEAT census — "does this material's tile stamp itself across a wide surface?"**
///
/// `MaterialScaleAudit` answers a WITHIN-slice question (is the detail drawn at a size a person
/// can see). This answers the orthogonal, ACROSS-slice one that `MaterialScaleAudit` is blind to:
/// a coherent material — one whose repeat is a recognisable LANDMARK (a plank, a tile course, a
/// cathedral-arch veneer figure) — tiled several times across a metres-wide plane reads as a
/// printed sheet, because the eye recognises the same landmark at the tile pitch. That is the
/// DH-0456 defect: an oak `panel` wall finish bakes one 512² slice at the ~1 m panel pitch, a 4 m
/// wall samples it ~4×, and with nothing to break the period the four panels are byte-identical.
///
/// Three things can break the period, and this census scores whether ANY is present:
///
///  1. **Per-cell VALUE de-repeat** — `patternCells`/`patternJitter`. The shader hashes
///     `floor(uv · patternCells)` (unique per physical unit across the whole surface) into a ± tone
///     multiplier, so each panel / plank / course renders a slightly different cut. This census
///     REPRODUCES that hash (`patternCellHash`, mirrored below) so `periodRepeatResidual` is the
///     break the render will actually show, not a proxy.
///  2. **Geometry** — real panel divisions with a reveal joint. Out of scope for DH-0456 (Danny's
///     call: no geometry).
///  3. **The hex-stochastic blend** — INVALID for a coherent material (it superimposes misaligned
///     copies), which is exactly why the coherent categories need mechanism 1.
///
/// GPU-free, deterministic, `swift test`-native, like `MaterialScaleAudit` / `MeshAudit`.
public struct MaterialRepeatCensus: Equatable, Sendable {
    /// World metres one texture repeat spans on the surface (the material's chosen period).
    public var runMeters: Double
    /// The surface the material is judged on — a wall's width, a floor's run.
    public var surfaceWidthMeters: Double
    /// How many times the slice tiles across the surface — `surfaceWidth ÷ run`.
    public var repeatsAcross: Double

    public var category: MaterialCategory
    /// A repeat of THIS material reads as a landmark (a plank, a course, a veneer figure), so an
    /// unbroken tile is visible as a stamp. The stochastic materials (soil, plaster, stone scatter)
    /// are exempt: their repeat is a texture, not a landmark, and the hex blend already serves them.
    public var isCoherent: Bool

    public var patternCells: Vec2
    public var patternJitter: Double
    /// The per-repeat tone spread the shader applies ALONG the tiling (U) axis. `patternJitter`
    /// when the U axis is subdivided (`patternCells.x ≥ 1`), else 0 — a de-repeat on the bonded
    /// (V) axis alone does not break a horizontally-tiling wall.
    public var perRepeatToneVariation: Double

    /// **Mean relative luma difference between the surface and itself shifted one period**, over the
    /// modelled span. 0 ⇒ a byte-exact stamp (the defect); grows as the per-cell de-repeat cuts each
    /// repeat from a different tone. Computed by reproducing the shader's `patternCellHash`, so it is
    /// the literal break the render shows. (A per-panel achromatic tone step is scale-invariant, so a
    /// *normalised* autocorrelation would not credit it — this SAD-style residual is what moves.)
    public var periodRepeatResidual: Double

    // MARK: - thresholds

    /// Above this many repeats a coherent stamp is plainly visible. Below ~1.5 the surface shows at
    /// most one full period plus a sliver — there is no side-by-side repeat for the eye to catch.
    public static let visibleRepeatCount = 1.5

    /// The coherent categories — mirrors `HouseScene.coherentPatternCategories` (asserted equal in
    /// the app's `RenderBridgeMaterialTests`) and `PatternCellsTests`. A repeat of one of these is a
    /// landmark.
    public static let coherentCategories: Set<MaterialCategory> =
        [.tile, .wallpaper, .brick, .wood, .laminate, .fabric, .ceramic]

    /// The surface tiles often enough that an unbroken repeat is visible.
    public var tilesVisibly: Bool { isCoherent && repeatsAcross > Self.visibleRepeatCount }

    /// **The DH-0456 defect, as a boolean**: a coherent material tiling visibly with nothing to
    /// break the period. Fail-closed — a new coherent material on a wide surface trips this until it
    /// declares a de-repeat.
    public var readsAsStampedRepeat: Bool { tilesVisibly && perRepeatToneVariation <= 0 }

    // MARK: - measurement

    public static func audit(_ ch: MaterialChannels, runMeters: Double,
                             surfaceWidthMeters: Double) -> MaterialRepeatCensus {
        let run = Swift.max(runMeters, 1e-4)
        let repeats = surfaceWidthMeters / run
        let isCoherent = coherentCategories.contains(ch.category)
        // The tone de-repeat only breaks the horizontal (U) tiling if the U axis carries cells.
        let breaksU = ch.patternCells.x >= 1 && ch.patternJitter > 0
        let perRepeat = breaksU ? ch.patternJitter : 0

        // Faithful residual: model K side-by-side copies, each toned by the shader's own hash, and
        // measure the mean relative luma step between adjacent periods. The slice luma factors out
        // of the difference (`|L·mₖ − L·mₖ₋₁| = |L|·|Δm|`), so the number is
        // `(mean|L| ÷ meanL) · mean|Δm|` — cheap and exact for the tone mechanism, and it would also
        // pick up a future per-cell FIGURE de-repeat because it is defined on the modelled surface.
        var residual = 0.0
        let k = Int(repeats.rounded(.down))
        if k >= 2, ch.patternJitter > 0 {
            let luma = ch.albedo.map(TextureAudit.luma)
            let meanL = luma.reduce(0, +) / Double(Swift.max(1, luma.count))
            let meanAbsL = luma.reduce(0) { $0 + Swift.abs($1) } / Double(Swift.max(1, luma.count))
            // `patternCellHash(floor(uv·cells))`: for the horizontal repeat the cell along U is the
            // copy index (when patternCells.x == 1) — the general `floor(k · cells.x)` covers a
            // plank run (cells.x > 1) too, indexing the FIRST cell of copy k.
            var stepSum = 0.0
            var prev = toneMultiplier(copy: 0, cells: ch.patternCells, jitter: ch.patternJitter)
            for c in 1..<k {
                let m = toneMultiplier(copy: c, cells: ch.patternCells, jitter: ch.patternJitter)
                stepSum += Swift.abs(m - prev)
                prev = m
            }
            let meanStep = stepSum / Double(k - 1)
            residual = meanL > 1e-9 ? (meanAbsL / meanL) * meanStep : 0
        }

        return MaterialRepeatCensus(
            runMeters: run, surfaceWidthMeters: surfaceWidthMeters, repeatsAcross: repeats,
            category: ch.category, isCoherent: isCoherent,
            patternCells: ch.patternCells, patternJitter: ch.patternJitter,
            perRepeatToneVariation: perRepeat, periodRepeatResidual: residual)
    }

    // MARK: - period self-correlation (DH-1409)

    /// **How strongly a unit grid's pattern of tones repeats one period over**, as a Pearson
    /// correlation of the per-unit tone with itself shifted by one bake (`cols` units). The surface
    /// is modelled exactly as the render builds it: the baked tone of unit `(i mod cols, j mod rows)`
    /// times the shader's `patternCellHash` multiplier of the physical unit `(I, J)`.
    /// 1 = the same stamp every period (the defect), ~0 = every period is a fresh draw.
    /// `bake` must be a unit-grid bake (`MaterialGenerator.tile`) of `cols x rows` units.
    public static func unitTonePeriodCorrelation(_ bake: MaterialChannels, cols: Int, rows: Int,
                                                 periods: Int = 4) -> Double {
        let n = bake.size
        var tone = [Double](repeating: 1, count: cols * rows)
        var mean = 0.0
        for j in 0..<rows { for i in 0..<cols {
            let x = Int((Double(i) + 0.5) / Double(cols) * Double(n))
            let y = Int((Double(j) + 0.5) / Double(rows) * Double(n))
            tone[j * cols + i] = TextureAudit.luma(bake.albedo[bake.idx(x, y)])
            mean += tone[j * cols + i]
        } }
        mean /= Double(cols * rows)
        guard mean > 1e-9 else { return 0 }
        for k in tone.indices { tone[k] /= mean }
        let w = cols * Swift.max(2, periods), h = rows * 2
        let hashed = bake.patternJitter > 0 && bake.patternCells.x >= 1
        func surface(_ i: Int, _ j: Int) -> Double {
            let m = hashed
                ? 1 + bake.patternJitter * patternCellHash(Double(i) / Double(cols) * bake.patternCells.x,
                                                           Double(j) / Double(rows) * bake.patternCells.y)
                : 1
            return tone[(j % rows) * cols + i % cols] * m
        }
        var a = [Double](), b = [Double]()
        for j in 0..<h { for i in 0..<(w - cols) { a.append(surface(i, j)); b.append(surface(i + cols, j)) } }
        let ma = a.reduce(0, +) / Double(a.count), mb = b.reduce(0, +) / Double(b.count)
        var sab = 0.0, saa = 0.0, sbb = 0.0
        for k in a.indices { let da = a[k] - ma, db = b[k] - mb; sab += da * db; saa += da * da; sbb += db * db }
        return saa > 1e-12 && sbb > 1e-12 ? sab / (saa * sbb).squareRoot() : 0
    }

    /// **Stochastic materials: correlation of the rendered surface with itself one tile over**, with
    /// the hex de-repeat blended in at `antiTilingStrength` the way the frame does (the product
    /// ships 0.85, `HouseRenderBridge.antiTilingStrength`). 1 = a visible clone every tile.
    public static func stochasticPeriodCorrelation(_ bake: MaterialChannels, antiTilingStrength: Double,
                                                   tiles: Int = 4, pixelsPerTile: Int = 32) -> Double {
        let res = pixelsPerTile, n = tiles * res
        let mean = bake.albedoMean
        let st = Swift.min(1, Swift.max(0, antiTilingStrength))
        var f = [Double](repeating: 0, count: n * n)
        for y in 0..<n { for x in 0..<n {
            let u = Double(x) / Double(res), v = Double(y) / Double(res)
            let c = bake.sampleAlbedo(u, v) * (1 - st) + bake.sampleAlbedoHex(u, v, mean: mean) * st
            f[y * n + x] = TextureAudit.luma(c)
        } }
        var a = [Double](), b = [Double]()
        for y in 0..<n { for x in 0..<(n - res) { a.append(f[y * n + x]); b.append(f[y * n + x + res]) } }
        let ma = a.reduce(0, +) / Double(a.count), mb = b.reduce(0, +) / Double(b.count)
        var sab = 0.0, saa = 0.0, sbb = 0.0
        for k in a.indices { let da = a[k] - ma, db = b[k] - mb; sab += da * db; saa += da * da; sbb += db * db }
        return saa > 1e-12 && sbb > 1e-12 ? sab / (saa * sbb).squareRoot() : 0
    }

    /// A unit grid whose tones correlate above this with themselves one period over reads as a stamp.
    public static let maxPeriodCorrelation = 0.35

    /// The tone multiplier the shader applies to the U-cell that starts copy `copy`.
    /// `m = 1 + jitter · patternCellHash(floor(copy · cells.x), 0)`.
    static func toneMultiplier(copy: Int, cells: Vec2, jitter: Double) -> Double {
        let cellX = Double(copy) * cells.x
        return 1.0 + jitter * patternCellHash(cellX, 0)
    }

    /// **A byte-faithful Swift mirror of `IlluminatoramaMaterial.h : patternCellHash`.** Same
    /// integer mix, same `& 0xFFFF ÷ 32768 − 1` mapping into `[-1, 1)`, so the residual this census
    /// reports is the break the shader will produce. `&*` / `&<<` keep the UInt32 wrapping the GPU
    /// relies on defined here too.
    static func patternCellHash(_ x: Double, _ y: Double) -> Double {
        let cx = Int32((x + 0.5).rounded(.down))
        let cy = Int32((y + 0.5).rounded(.down))
        var h = UInt32(bitPattern: cx) &* 73856093 ^ UInt32(bitPattern: cy) &* 19349663
        h ^= h >> 11; h = h &* 0x45d9f3b; h ^= h >> 16
        return Double(h & 0xFFFF) / 32768.0 - 1.0
    }
}
