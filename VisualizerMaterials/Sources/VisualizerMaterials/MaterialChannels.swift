import simd
import Foundation

/// The per-pixel PBR channel set a procedural material bakes into — the CPU analogue
/// of the engine's texture atlas (MATERIALS_AND_TEXTURES §3/§7). A square, **tileable**
/// grid: `albedo` is linear RGB, `roughness`/`height` are scalars in `[0, 1]`, `normal`
/// is tangent-space (+Z out, unit). Authoring is **height-first** (§4): a generator
/// fills `height`, then `deriveNormals()`/`curvature()` produce the rest, so edge wear
/// and relief come from one coherent field instead of hand-painted maps.
public struct MaterialChannels: Equatable, Sendable {
    public let size: Int                    // pixels per side
    public let category: MaterialCategory
    public var albedo: [Vec3]               // size*size, linear 0…1
    public var roughness: [Double]          // size*size, 0…1
    public var height: [Double]             // size*size, 0…1
    public var normal: [Vec3]               // size*size, unit tangent-space
    /// Grain tangent for anisotropic materials (wood/brushed metal), tangent-space.
    /// `nil` for isotropic materials — the engine then uses an isotropic specular lobe.
    public var grainTangent: [Vec2]?
    /// Second-lobe strengths (MATERIALS_AND_TEXTURES §3), uniform per material:
    /// **clearcoat** = the polished/lacquered/glazed wet-spec lobe (terrazzo, marble,
    /// glazed tile, satin wood); **sheen** = the retroreflective grazing lobe every
    /// cloth has (velvet = dark base + strong sheen). Branch per material — never 0
    /// globally (§3). 0 = lobe absent.
    public var clearcoat: Double = 0
    public var sheen: Double = 0
    /// **Per-material cloth-sheen ROUGHNESS** (DH-0081) — the width of the sheen nap, distinct
    /// from `sheen` (its strength). Low = a crisp, tight grazing highlight (sateen, silk); high =
    /// a broad, soft glow (velvet, wool bouclé). `0.30` is the library-wide default the lobe
    /// shipped with as one constant, so a material that leaves it alone renders exactly as before.
    /// The engine snaps this to the nearest of a few curated bands and folds the band into the
    /// same sign-multiplexed emission.alpha channel that carries `sheen` (see
    /// `clothSheenRoughnessForBand` in IlluminatoramaCommon.h) — inert (band 0) unless a material
    /// picks a non-default nap. Only meaningful where `sheen > 0`.
    public var sheenRoughness: Double = 0.30
    /// High-frequency (close-range) detail normal map — pores, weave, grain.
    /// Sampled at `detailNormalUVScale × uv` (shader-side) and blended over the
    /// macro normal via partial-derivative add. `nil` = no detail layer.
    public var detailNormal: [Vec3]? = nil
    /// **S2.4 — the detail band's diffuse-visible companion.** Micro-occlusion in `[0, 1]`
    /// (1 = fully open, < 1 = a pit that self-shadows), derived from the SAME high-frequency
    /// height field as `detailNormal` — it is never authored independently, so the two can't
    /// disagree ([[feedback-single-source-of-truth]]); `MaterialGenerator.setDetailRelief`
    /// is the one place both are written.
    ///
    /// **Why this exists.** The 2026-08-02 whole-library A/B measured the detail band moving
    /// metals 1.2–3.4× and *every* dielectric ~1.00×: a fine normal perturbation is a
    /// specular-band effect, and a matte dielectric's broad diffuse lobe averages it away.
    /// Micro-occlusion multiplies the DIFFUSE, so cloth / cork / plaster / stone finally get a
    /// close-range response. It also filters correctly — a non-negative quantity's mip average
    /// is its mean, so the term fades in contrast with distance instead of vanishing or
    /// aliasing (which is what forced the detail normal's under-filtered exception in S1.1).
    ///
    /// Rides in the **blue channel of the detail-normal slice** (see `detailNormalRGBA()`):
    /// the shader only ever reads that map's xy, so B was baked and discarded. No new atlas
    /// slice, no extra fetch, no G-buffer channel. `nil` ⇒ B encodes 255 ⇒ exactly neutral.
    public var detailOcclusion: [Double]? = nil

    /// **S2.5 half 2 — the coherent categories' de-repeat metadata.** Pattern cells per
    /// macro UV tile, per axis — e.g. a stack-bond 4×4 tile grid is `(4, 4)`; a 4-plank
    /// bake whose planks run unbroken along v is `(4, 0)`. The engine hashes
    /// `floor(uv * patternCells)` — unique per PHYSICAL tile/plank/course across the whole
    /// surface, because uv keeps counting and only the atlas lookup wraps — into a small
    /// ± tone multiplier, breaking the macro repeat with no UV displacement and no extra
    /// texture tap (the hex blend is invalid for coherent patterns; this is their answer).
    ///
    /// **A bonded axis must ship 0.** `floor(uv·0) = 0` keeps that axis's index constant,
    /// so a running-bond COURSE varies as a whole — a non-zero count there would put a
    /// cell boundary mid-tile on every offset course (a tone step cutting through tiles).
    /// `(0, 0)` (the default) disables the effect. Set ONLY by the generator that drew the
    /// grid — deriving the count a second time downstream is how the two drift apart
    /// ([[feedback-single-source-of-truth]]).
    /// **Sparse world-space knots this material asks the RENDERER to draw** — nil for everything
    /// that is not knotted wood.
    ///
    /// Not baked, and it cannot be: a knot is a LANDMARK, and a landmark baked into a tiling
    /// texture reappears on a lattice at the tile period. The wood tile is ~1 m (the resolution
    /// its grain needs), so baked knots drew a 1 m grid across every floor. The field is
    /// evaluated per pixel on the unwrapped UV instead (`sampleWoodKnots`), which never repeats
    /// — and which can warp the material UV so the grain flows AROUND the knot, something no
    /// decal composited on top could do.
    ///
    /// Physical, in metres. The render layer converts to the surface's UV units, because that is
    /// the only layer that knows the surface's period.
    public var knots: WoodKnotField? = nil

    /// **A carpet's metre-scale pile-lay tone bands this material asks the RENDERER to draw**
    /// — nil for everything that is not a pile carpet (DH-0472).
    ///
    /// Same reason as `knots`, one scale up: a carpet's dominant large-scale tell is that the pile
    /// physically LIES one way and shifts value where the nap turns — broad soft tonal bands metres
    /// across, true even of a brand-new unworn rug. That field cannot live in the bake: the carpet
    /// tile is ~0.30 m (the gauge its tufts need), so any low-frequency term authored inside it
    /// repeats every 0.30 m and then dissolves under the hex de-repeat (`.carpet` is stochastic).
    /// The rug then reads as one even tone at room distance — the defect DH-0472 records.
    ///
    /// So it is declared, not baked: evaluated per pixel on the UNWRAPPED uv (`sampleCarpetMacro`),
    /// which counts up across the whole rug and never repeats at the tile period — the exact shape
    /// of the wood-knot path. Physical, in metres; the render layer converts to the surface's UV
    /// units, because that is the only layer that knows the surface's period.
    public var carpetMacro: CarpetMacroField? = nil

    public var patternCells: Vec2 = .zero
    /// Half-amplitude of the per-cell tone jitter (0.04 ⇒ ±4 %). 0 = exact shader no-op.
    /// Real tile/plank batch variation lives around 3–6 %; the generator owns the number.
    public var patternJitter: Double = 0

    public init(size: Int, category: MaterialCategory) {
        self.size = max(1, size)
        self.category = category
        let n = self.size * self.size
        albedo = Array(repeating: Vec3(0.5, 0.5, 0.5), count: n)
        roughness = Array(repeating: 0.5, count: n)
        height = Array(repeating: 0.5, count: n)
        normal = Array(repeating: Vec3(0, 0, 1), count: n)
        grainTangent = nil
    }

    @inlinable public func idx(_ x: Int, _ y: Int) -> Int {
        (Noise.wrap(y, size)) * size + Noise.wrap(x, size)      // wrap → toroidal sampling
    }

    /// A copy with the albedo multiplied by `tint` (clamped to [0, 1]) — texture detail
    /// (weave, grain, roughness variation) is preserved, only the colour shifts. This is
    /// how ONE baked fabric/wood serves many colourways (a blue-grey sofa, a warm-linen
    /// bed, walnut-toned legs) without a bespoke generator per colour.
    public func tinted(by tint: Vec3) -> MaterialChannels {
        var out = self
        out.albedo = albedo.map {
            Vec3(min(1, max(0, $0.x * tint.x)),
                 min(1, max(0, $0.y * tint.y)),
                 min(1, max(0, $0.z * tint.z)))
        }
        return out
    }

    /// Nearest-neighbor albedo sample at a (tiling) UV — `u`/`v` wrap, so any real-world
    /// coordinate maps in. Used to shade the 3D house with a baked material.
    @inlinable public func sampleAlbedo(_ u: Double, _ v: Double) -> Vec3 {
        let x = Int((u - floor(u)) * Double(size)), y = Int((v - floor(v)) * Double(size))
        return albedo[idx(x, y)]
    }

    /// Hex-stochastic albedo sample — breaks tiling artifacts on large surfaces
    /// (floors, walls) by blending three samples from randomly-offset UV patches
    /// whose borders align on a triangular lattice. This is the CPU reference
    /// implementation of the technique that mirrors the GPU shader (Phase 7, §7).
    ///
    /// The UV scale factor `scale` should match the material's real-world repeat
    /// (e.g. 0.5 = 2 m tile → pass world coordinate in metres, `scale`=0.5).
    ///
    /// **S2.5 — `mean` is required, not optional, and that is deliberate.** It is μ for the
    /// variance-preserving blend (`albedoMean`), an O(size²) reduction; making it a
    /// defaulted parameter would have hidden a full-tile scan inside a per-pixel call in
    /// `RenderCheck`. Hoist it once per material.
    @inlinable public func sampleAlbedoHex(_ u: Double, _ v: Double, mean: Vec3) -> Vec3 {
        // Triangular grid: skew so that equilateral triangles tile the plane.
        // Skew matrix: (1, 0.5; 0, sqrt(3)/2).
        let sq3over2 = 0.866025404   // sqrt(3)/2
        let su = u + v * 0.5, sv = v * sq3over2
        // Cell origin in skewed space
        let si = floor(su), sj = floor(sv)
        let fu = su - si, fv = sv - sj    // fractional within rhombus cell

        // The rhombus is divided into 2 triangles. Determine which triangle we're in and
        // take its 3 vertices in **skewed-lattice integer** coordinates.
        //
        // **This used to un-skew the vertices back into UV space before hashing, and the
        // shader never did.** The hash FUNCTION was byte-identical (see `hexHash`) while
        // its ARGUMENT was not: the shader hashes `(si, sj)`, this hashed
        // `(si − sj/√3, sj·2/√3)` rounded — a different cell id for every vertex off the
        // sj = 0 row, hence a different offset, hence a different blend. Same failure mode
        // as the 32-bit overflow this file already records: a "CPU reference" that is
        // quietly a different algorithm cannot mirror anything.
        let (v0, v1, v2): (Vec2, Vec2, Vec2)
        let (w0, w1, w2): (Double, Double, Double)

        if fu + fv < 1.0 {
            // Lower triangle: vertices (si,sj), (si+1,sj), (si,sj+1)
            v0 = Vec2(si,       sj)
            v1 = Vec2(si + 1.0, sj)
            v2 = Vec2(si,       sj + 1.0)
            w0 = 1.0 - fu - fv; w1 = fu; w2 = fv
        } else {
            // Upper triangle: vertices (si+1,sj+1), (si,sj+1), (si+1,sj)
            v0 = Vec2(si + 1.0, sj + 1.0)
            v1 = Vec2(si,       sj + 1.0)
            v2 = Vec2(si + 1.0, sj)
            w0 = fu + fv - 1.0; w1 = 1.0 - fu; w2 = 1.0 - fv
        }

        // Random UV offsets from each vertex (deterministic hash, no sin/cos).
        let h0 = hexHash(v0), h1 = hexHash(v1), h2 = hexHash(v2)

        // Blend weights: smooth power curve so transitions aren't linear.
        let p0 = w0 * w0 * w0, p1 = w1 * w1 * w1, p2 = w2 * w2 * w2
        let pSum = p0 + p1 + p2

        let c0 = sampleAlbedo(u + h0.x, v + h0.y)
        let c1 = sampleAlbedo(u + h1.x, v + h1.y)
        let c2 = sampleAlbedo(u + h2.x, v + h2.y)
        let blend = (c0 * p0 + c1 * p1 + c2 * p2) * (1.0 / pSum)

        // S2.5 — variance-preserving rescale, mirroring `sampleAtlasHex`. Three i.i.d. taps
        // averaged with weights summing to 1 carry only `Σwᵢ²` of one tap's variance (1 at a
        // cell centre, ⅓ at a triangle centroid), so the naive blend wrote away up to ⅔ of
        // the material's contrast. `out = μ + (blend − μ)/√(Σwᵢ²)` restores the second
        // moment and is the identity at a cell centre, so the lattice stays seam-free.
        let n0 = p0 / pSum, n1 = p1 / pSum, n2 = p2 / pSum
        let sumSq = n0 * n0 + n1 * n1 + n2 * n2
        let rescale = 1.0 / Double.maximum(sumSq, 1e-12).squareRoot()
        let out = mean + (blend - mean) * rescale
        return Vec3(Swift.min(1, Swift.max(0, out.x)),
                    Swift.min(1, Swift.max(0, out.y)),
                    Swift.min(1, Swift.max(0, out.z)))
    }

    /// Mean linear albedo over the whole tile — μ for the variance-preserving hex blend.
    /// The CPU analogue of `IlluminatoramaTextureAtlas.sliceMeanBuffer`. **O(size²)** —
    /// hoist it out of per-pixel loops (`sampleAlbedoHex` takes it as an argument for
    /// exactly that reason).
    @inlinable public var albedoMean: Vec3 {
        guard !albedo.isEmpty else { return Vec3(0.5, 0.5, 0.5) }
        var s = Vec3(0, 0, 0)
        for c in albedo { s = s + c }
        return s * (1.0 / Double(albedo.count))
    }

    /// Deterministic hash: maps a 2D grid vertex to a (−1,1) UV offset.
    /// Uses integer bit-mixing — no sin, safe in inner loops.
    ///
    /// **Byte-identical to `hexHash2D` in Illuminatorama.metal, and it was not.** This ran the
    /// multiplies in 64-bit `Int` where the shader ran them in 32-bit, so the two produced
    /// different offsets for the same cell — and because this one cannot overflow, it could never
    /// reproduce the shader's failure past ±26 lattice units. A "CPU reference" that is a
    /// different algorithm from the thing it references is worse than none: it makes the GPU bug
    /// invisible from the side that is cheap to test. Both now round to an integer cell first and
    /// mix in wrapping 32-bit unsigned arithmetic.
    @inlinable public static func hexHash(_ v: Vec2) -> Vec2 {
        let cx = Int32(truncatingIfNeeded: Int(v.x.rounded()))
        let cy = Int32(truncatingIfNeeded: Int(v.y.rounded()))
        var ix = UInt32(bitPattern: cx) &* 73856093 ^ UInt32(bitPattern: cy) &* 19349663
        var iy = UInt32(bitPattern: cx) &* 83492791 ^ UInt32(bitPattern: cy) &* 23994923
        ix ^= ix >> 11; ix &*= 0x45d9f3b; ix ^= ix >> 16
        iy ^= iy >> 11; iy &*= 0x45d9f3b; iy ^= iy >> 16
        return Vec2(Double(ix & 0xFF) / 128.0 - 1.0, Double(iy & 0xFF) / 128.0 - 1.0)
    }

    @inlinable internal func hexHash(_ v: Vec2) -> Vec2 { MaterialChannels.hexHash(v) }

    /// Central-difference normal from the height field, sampled **toroidally** so the
    /// relief is continuous across the tile seam. `strength` scales the bump (steeper →
    /// more pronounced); the height delta is taken in pixels of the tiled surface.
    public mutating func deriveNormals(strength: Double = 8) {
        var out = normal
        for y in 0..<size {
            for x in 0..<size {
                let hL = height[idx(x - 1, y)], hR = height[idx(x + 1, y)]
                let hD = height[idx(x, y - 1)], hU = height[idx(x, y + 1)]
                // gradient → tangent-space normal (Sobel-lite central difference)
                let n = normalize3(Vec3((hL - hR) * strength, (hD - hU) * strength, 1))
                out[idx(x, y)] = n
            }
        }
        normal = out
    }

    /// Discrete Laplacian of the height field (toroidal): **positive = convex** (peaks,
    /// edges — read polished/worn), **negative = concave** (pits, crevices — hold grime).
    /// This is the §4 curvature signal that drives roughness and edge-wear without any
    /// hand painting.
    public func curvature() -> [Double] {
        var c = [Double](repeating: 0, count: size * size)
        for y in 0..<size {
            for x in 0..<size {
                let h0 = height[idx(x, y)]
                let lap = height[idx(x - 1, y)] + height[idx(x + 1, y)]
                        + height[idx(x, y - 1)] + height[idx(x, y + 1)] - 4 * h0
                c[idx(x, y)] = -lap        // sign so convex peaks come out positive
            }
        }
        return c
    }
}

/// Small scalar helpers shared by the generators.
@inlinable public func clamp01(_ v: Double) -> Double { max(0, min(1, v)) }

/// Clamp a dielectric albedo into the physical reflectance band `[0.045, 0.93]`
/// (MATERIALS_AND_TEXTURES §2 tell #4) — pure black/white read CG, so generators pin
/// every opaque base color inside the band by construction.
@inlinable public func clampBand(_ v: Double) -> Double { max(0.045, min(0.93, v)) }
@inlinable public func clampBand(_ c: Vec3) -> Vec3 { Vec3(clampBand(c.x), clampBand(c.y), clampBand(c.z)) }

/// Hermite step from 0→1 as `t` crosses `[edge0, edge1]`.
@inlinable public func smoothstep(_ edge0: Double, _ edge1: Double, _ t: Double) -> Double {
    if edge1 == edge0 { return t < edge0 ? 0 : 1 }
    let x = clamp01((t - edge0) / (edge1 - edge0))
    return x * x * (3 - 2 * x)
}

@inlinable public func mix(_ a: Vec3, _ b: Vec3, _ t: Double) -> Vec3 { a + (b - a) * t }
@inlinable public func mix(_ a: Double, _ b: Double, _ t: Double) -> Double { a + (b - a) * t }


/// **A wood's knot population**, in real-world units. See `MaterialChannels.knots`.
public struct WoodKnotField: Equatable, Hashable, Sendable, Codable {
    /// Knots per square metre of surface.
    public var perSquareMeter: Double
    /// Mean knot radius, metres. Individual knots are drawn from a squared distribution around
    /// it (0.45×…1.7×) — a field of same-sized discs reads as a stamp.
    public var radiusMeters: Double
    /// How dark the core is against the wood it grew through, [0, 1].
    public var darkness: Double

    public init(perSquareMeter: Double, radiusMeters: Double, darkness: Double) {
        self.perSquareMeter = Swift.max(0, perSquareMeter)
        self.radiusMeters = Swift.max(0.001, radiusMeters)
        self.darkness = darkness.clamped(to: 0...1)
    }

    /// **Lattice cell size, metres.** One candidate site per cell, so the cell has to be
    /// comfortably larger than a knot's influence (~5 radii) for the shader's 3×3 neighbourhood
    /// to be enough, and small enough that `density` stays under 1.
    ///
    /// **This constant is what caps how knotty a wood can be**, and pine reaches the cap. One
    /// site per cell makes `1 / cellMeters²` = 4/m² a hard ceiling, and the usable range stops
    /// well short of it: as `cellDensity` approaches 1 nearly every cell is occupied and the
    /// scatter degenerates into a jittered GRID — the regular-looking failure that a Poisson
    /// draw exists to avoid. Real #2-grade pine throws ~16 knots/m², so `knottyPine` ships at
    /// 2.4 (`cellDensity` 0.60) because that is where the field still has empty cells, not
    /// because that is what the wood does.
    ///
    /// Shrinking this is the lever if a wood ever needs to be knottier, and it is safe as far
    /// as ~0.35 m on the shipped species — the binding constraint is that a knot's reach
    /// (`radiusMeters × 1.7 × 5`, matching the shader's draw and cut-off exactly) must stay
    /// inside one cell, and the largest of those is pine's 0.255 m. It is NOT free: the lattice
    /// is what positions every knot, so changing it re-scatters oak and cherry too.
    public static let cellMeters = 0.5

    /// Fraction of lattice cells that carry a knot — what the shader takes.
    public var cellDensity: Double {
        Swift.min(1, perSquareMeter * Self.cellMeters * Self.cellMeters)
    }
}

/// **A carpet's pile-lay tone bands** (DH-0472), in real-world units. See `MaterialChannels.carpetMacro`.
///
/// This is the metre-scale nap variation a 0.30 m carpet tile physically cannot carry. The renderer
/// draws it as a low-frequency achromatic value field on the rug's UNWRAPPED uv (`sampleCarpetMacro`),
/// so it never repeats at the tile period — the wood-knot mechanism, one scale up.
public struct CarpetMacroField: Equatable, Hashable, Sendable, Codable {
    /// Characteristic band size, metres — the spacing between one nap-toward and the next nap-away
    /// tonal swell. Real broadloom shows ~0.6–1.2 m "shading"; the generator owns the number.
    public var bandMeters: Double
    /// Half-amplitude of the multiplicative tone swing (0.10 ⇒ ±10 %). 0 = an exact shader no-op.
    public var strength: Double

    public init(bandMeters: Double, strength: Double) {
        self.bandMeters = Swift.max(0.05, bandMeters)
        self.strength = strength.clamped(to: 0...0.5)
    }
}
