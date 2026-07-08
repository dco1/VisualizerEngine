import Metal
import simd

// ── STATIC GRASS FILLER (sim-LOD) ────────────────────────────────────────────
//
// NON-simulated bent blade cards — the sim-LOD that lets a grass field read far
// denser than the live XPBD budget allows. The live field is capped by the
// super-linear PBD solve+expand cost (~135k blades → 22 ms, ~450k → 217 ms — a
// cliff); these cards cost only their G-buffer draw (+ RT BLAS triangles), not a
// per-frame dispatch. Each blade is an N-segment bent card with a baked rest arc
// + a fixed wind lean; per-vertex colour carries the same root→mid→tip gradient
// and the foliage-SSS alpha-0 flag as live `GrassRibbonRenderer` blades, so a
// live↔static seam is invisible when the palettes match.
//
// GENERALIZED from the Meadow scene's `makeStaticGrassFillerMesh` (the proven
// look — issue #58) so ANY host can drive it: Meadow's annulus + inner/outer
// feather + 1/d² distance-LOD all fold into the caller's `density` closure, and
// a yard-scale host (Daydream Home) passes its rect + ellipse feather + live-
// field deficit instead. RT-BUDGET NOTE (from Meadow's tuning): cards land in
// the RT BLAS and RT cost scales with TOTAL TRIANGLES — a naive uniform fill of
// a 38 m field is ~3.7 M tris and tanks RT to ~18 fps. Keep the caller's
// `density` honest (thin with distance) and respect `maxTriangles`.
public enum GrassFillerMesh {

    public struct Params {
        /// Clump-grid pitch (m). Meadow uses 2× its live spacing.
        public var clumpSpacing: Float
        /// Blade cards per clump at density 1. Thinned by the local density.
        public var bladesPerClump: Int
        /// Max segments per blade (near cards); far cards collapse toward 1
        /// as density falls (sub-pixel arcs don't need the tessellation).
        public var maxSegments: Int
        /// Clump base-height range (m) BEFORE the per-blade ±15/+30% jitter.
        public var heightRange: ClosedRange<Float>
        /// Per-blade half-width range (m). Meadow: 2.8–4.4 mm (≈6–9 mm full).
        public var halfWidthRange: ClosedRange<Float>
        /// Root→mid→tip gradient — match the live field's palette or the LOD
        /// seam shows.
        public var rootColor: SIMD3<Float>
        public var midColor: SIMD3<Float>
        public var tipColor: SIMD3<Float>
        /// Baked global wind lean heading (radians) — match the live gust/sun
        /// heading so the static band leans WITH the animated one.
        public var windHeadingRadians: Float
        /// Hard triangle cap — generation stops here (RT-budget backstop).
        public var maxTriangles: Int
        public var seed: UInt32

        public init(clumpSpacing: Float,
                    bladesPerClump: Int = 24,
                    maxSegments: Int = 3,
                    heightRange: ClosedRange<Float> = 0.36...0.58,
                    halfWidthRange: ClosedRange<Float> = 0.0028...0.0044,
                    rootColor: SIMD3<Float>,
                    midColor: SIMD3<Float> = SIMD3(0.055, 0.130, 0.032),
                    tipColor: SIMD3<Float> = SIMD3(0.190, 0.270, 0.070),
                    windHeadingRadians: Float = 200 * .pi / 180,
                    maxTriangles: Int = 500_000,
                    seed: UInt32 = 0x0BAD_F00D) {
            self.clumpSpacing = clumpSpacing
            self.bladesPerClump = bladesPerClump
            self.maxSegments = maxSegments
            self.heightRange = heightRange
            self.halfWidthRange = halfWidthRange
            self.rootColor = rootColor
            self.midColor = midColor
            self.tipColor = tipColor
            self.windHeadingRadians = windHeadingRadians
            self.maxTriangles = maxTriangles
            self.seed = seed
        }
    }

    /// Deterministic CPU vertex/index soup for the filler band — build once, wrap in an
    /// `IlluminatoramaMesh`, register, done. Exposed separately from `build(device:…)`
    /// so hosts can unit-test the geometry GPU-free.
    ///
    /// - Parameters:
    ///   - boundsMin/boundsMax: the XZ rect the clump grid scans.
    ///   - density: 0…1 at a clump centre — 0 skips, and the value thins blade count,
    ///     collapses segment count, and gates per-blade keep-probability (fold ALL
    ///     feathering/distance-LOD into this one closure).
    ///   - heightScale: extra per-clump height multiplier (e.g. taper with distance);
    ///     return 1 for none.
    ///   - groundHeight: XZ → surface Y the blade roots bed onto.
    @MainActor
    public static func soup(params: Params,
                            boundsMin: SIMD2<Float>, boundsMax: SIMD2<Float>,
                            density: (Float, Float) -> Float,
                            heightScale: (Float, Float) -> Float,
                            groundHeight: (Float, Float) -> Float)
        -> (vertices: [IlluminatoramaVertex], indices: [UInt32]) {
        var seed = params.seed
        func rnd() -> Float {
            seed = seed &* 1664525 &+ 1013904223
            return Float(seed >> 8) / Float(1 << 24)
        }
        let windLeanDir = SIMD2<Float>(cos(params.windHeadingRadians), sin(params.windHeadingRadians))
        var verts: [IlluminatoramaVertex] = []
        var indices: [UInt32] = []
        verts.reserveCapacity(40_000)
        indices.reserveCapacity(60_000)

        let spacing = max(params.clumpSpacing, 0.05)
        var cz = boundsMin.y
        outer: while cz < boundsMax.y {
            var cx = boundsMin.x
            while cx < boundsMax.x {
                defer { cx += spacing }
                let ccx = cx + (rnd() - 0.5) * spacing * 0.9
                let ccz = cz + (rnd() - 0.5) * spacing * 0.9
                let keep = min(1, max(0, density(ccx, ccz)))
                if keep <= 0.02 { continue }

                let hLo = params.heightRange.lowerBound, hHi = params.heightRange.upperBound
                let clumpHeight = (hLo + rnd() * (hHi - hLo)) * max(0, heightScale(ccx, ccz))
                let clumpRadius = spacing * (0.40 + rnd() * 0.40)
                let n = max(1, Int(Float(params.bladesPerClump) * keep))
                let bladeSegs = max(1, Int((Float(params.maxSegments) * (0.35 + 0.65 * keep)).rounded()))

                for _ in 0..<n {
                    if rnd() > keep { continue }
                    if indices.count / 3 >= params.maxTriangles { break outer }
                    let a = rnd() * 2 * .pi
                    let rr = clumpRadius * rnd().squareRoot()    // uniform tuft fill
                    let bx = ccx + cos(a) * rr
                    let bz = ccz + sin(a) * rr
                    let h = clumpHeight * (0.85 + rnd() * 0.30)
                    let gy = groundHeight(bx, bz)
                    let wLo = params.halfWidthRange.lowerBound, wHi = params.halfWidthRange.upperBound
                    let halfW = wLo + rnd() * (wHi - wLo)
                    // Each card faces a random azimuth so a clump reads volumetric,
                    // not like aligned billboards.
                    let faceAz = rnd() * 2 * .pi
                    let faceDir = SIMD2<Float>(cos(faceAz), sin(faceAz))
                    let sideN = SIMD3<Float>(-faceDir.y, 0, faceDir.x)
                    let lean = (0.05 + rnd() * 0.10)
                    // Per-blade colour jitter (mirrors the live field's).
                    var hsh = seed
                    func bj() -> Float { hsh = hsh &* 1664525 &+ 1013904223; return Float(hsh >> 8) / Float(1 << 24) }
                    let bright = 0.80 + bj() * 0.40
                    let warm = (bj() - 0.5) * 0.040
                    let dry: Float = bj() < 0.04 ? 1.15 : 1.0
                    func colorAt(_ f: Float) -> SIMD4<Float> {
                        let segC = f < 0.5
                            ? params.rootColor + (params.midColor - params.rootColor) * (f / 0.5)
                            : params.midColor + (params.tipColor - params.midColor) * ((f - 0.5) / 0.5)
                        var c = segC * bright * dry
                        c.x += warm; c.y += warm * 0.30
                        c = simd_max(c, SIMD3<Float>(0, 0, 0))
                        return SIMD4(c, 0)   // alpha 0 = the foliage-SSS flag (same as live blades)
                    }

                    let base = UInt32(verts.count)
                    for s in 0...bladeSegs {
                        let f = Float(s) / Float(bladeSegs)
                        let w = halfW * (1.0 - 0.92 * f)         // taper to a sharp tip
                        let leanXZ = lean * f * f                 // arc grows toward the tip
                        let px = bx + windLeanDir.x * leanXZ
                        let pz = bz + windLeanDir.y * leanXZ
                        let py = gy + h * f
                        let col = colorAt(f)
                        verts.append(IlluminatoramaVertex(position: SIMD3(px, py, pz) + sideN * w,
                                                          normal: sideN, uv: SIMD2(0, f), color: col))
                        verts.append(IlluminatoramaVertex(position: SIMD3(px, py, pz) - sideN * w,
                                                          normal: sideN, uv: SIMD2(1, f), color: col))
                    }
                    for s in 0..<bladeSegs {
                        let r0 = base + UInt32(s * 2)
                        indices.append(contentsOf: [r0, r0 + 2, r0 + 1, r0 + 1, r0 + 2, r0 + 3])
                    }
                }
            }
            cz += spacing
        }

        if verts.isEmpty {
            // Degenerate guard: one zero-area tri keeps the mesh valid.
            verts = [IlluminatoramaVertex(position: .zero, normal: SIMD3(0, 1, 0), uv: .zero),
                     IlluminatoramaVertex(position: .zero, normal: SIMD3(0, 1, 0), uv: .zero),
                     IlluminatoramaVertex(position: .zero, normal: SIMD3(0, 1, 0), uv: .zero)]
            indices = [0, 1, 2]
        }
        return (verts, indices)
    }

    /// Build the filler as a registered-ready mesh. Flat cards read from both sides —
    /// the mesh ships `doubleSided = true` (Meadow's setting).
    @MainActor
    public static func build(device: MTLDevice, params: Params,
                             boundsMin: SIMD2<Float>, boundsMax: SIMD2<Float>,
                             density: (Float, Float) -> Float,
                             heightScale: (Float, Float) -> Float = { _, _ in 1 },
                             groundHeight: (Float, Float) -> Float = { _, _ in 0 }) -> IlluminatoramaMesh {
        let s = soup(params: params, boundsMin: boundsMin, boundsMax: boundsMax,
                     density: density, heightScale: heightScale, groundHeight: groundHeight)
        let mesh = IlluminatoramaMesh(device: device, vertices: s.vertices, indices: s.indices)
        mesh.doubleSided = true
        return mesh
    }
}
