import simd
import Foundation

/// Deterministic, **seamlessly tileable** procedural noise — the substrate of the
/// material engine (MATERIALS_AND_TEXTURES §5). Everything here is a pure function of
/// `(uv, seed)`: no global RNG, no `Date`/`random`, so a material renders bit-identical
/// in `swift test` and on device. All samplers take UV in `[0, 1)` and **wrap on the
/// unit torus**, which is what makes a tile repeat without a visible seam (defeats
/// tell #5). Frequencies are integer cell counts across the tile so each octave wraps.
public enum Noise {

    // MARK: integer hash (splitmix64 finalizer)

    /// Avalanche a 64-bit word — the splitmix64 finalizer. Used to derive every
    /// per-lattice / per-cell random value; tiny, fast, and well-distributed.
    @inlinable public static func mix(_ a: UInt64) -> UInt64 {
        var z = a &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Hash a 2D integer lattice point + seed → 64-bit word.
    @inlinable public static func hash2(_ x: Int, _ y: Int, _ seed: UInt64) -> UInt64 {
        let hx = mix(seed ^ UInt64(bitPattern: Int64(x)) &* 0x100_0000_01B3)
        return mix(hx ^ UInt64(bitPattern: Int64(y)) &* 0xC2B2_AE3D_27D4_EB4F)
    }

    /// A hashed word → uniform `Double` in `[0, 1)` (top 53 bits → mantissa).
    @inlinable public static func unit(_ h: UInt64) -> Double {
        Double(h >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Positive modulo (Swift `%` keeps the sign of the dividend).
    @inlinable public static func wrap(_ a: Int, _ n: Int) -> Int { ((a % n) + n) % n }

    // MARK: value noise + fBm (tileable)

    @inlinable static func smooth(_ t: Double) -> Double { t * t * (3 - 2 * t) }

    /// Bilinear value noise on a `cells × cells` lattice that wraps on the torus —
    /// returns `[0, 1]`. `cells` is how many lattice periods span the tile.
    public static func valueTiled(_ u: Double, _ v: Double, cells: Int, seed: UInt64) -> Double {
        let n = max(1, cells)
        let x = u * Double(n), y = v * Double(n)
        let xi = Int(floor(x)), yi = Int(floor(y))
        let fx = x - floor(x), fy = y - floor(y)
        @inline(__always) func corner(_ di: Int, _ dj: Int) -> Double {
            unit(hash2(wrap(xi + di, n), wrap(yi + dj, n), seed))
        }
        let sx = smooth(fx), sy = smooth(fy)
        let top = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * sx
        let bot = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * sx
        return top + (bot - top) * sy
    }

    /// Fractal Brownian motion — octaves of `valueTiled`, each doubling the cell count
    /// (so each octave still wraps) and halving amplitude. Returns `[0, 1]`. This is the
    /// "macro tonal drift under everything" layer that kills flat single colors (§4/§5).
    public static func fbmTiled(_ u: Double, _ v: Double, baseCells: Int = 4,
                                octaves: Int = 5, gain: Double = 0.5, seed: UInt64 = 0) -> Double {
        var sum = 0.0, amp = 1.0, total = 0.0, cells = max(1, baseCells)
        for o in 0..<max(1, octaves) {
            sum += amp * valueTiled(u, v, cells: cells, seed: seed &+ UInt64(o) &* 0x9E37_79B1)
            total += amp; amp *= gain; cells *= 2
        }
        return sum / total
    }

    /// **Smooth 1-D value noise on an unbounded axis** — a fresh value per integer step,
    /// Hermite-interpolated between them.
    ///
    /// The use it exists for is per-RING variation in wood: growth rings are indexed by how far
    /// out from the pith they are, so anything that should differ ring to ring (darkness, band
    /// width, pore load) is a function of that ordinal. Hashing `floor(ordinal)` directly would
    /// be simpler and wrong — the ordinal drifts slowly along the board as the grain bows, so a
    /// stepped value draws a hard line ACROSS the board wherever it crosses a whole number.
    @inlinable public static func value1D(_ t: Double, seed: UInt64) -> Double {
        let i = Int(floor(t)), f = t - floor(t)
        let a = unit(hash2(i, 0, seed)), b = unit(hash2(i + 1, 0, seed))
        return a + (b - a) * smooth(f)
    }

    /// Smooth bilinear value noise over a NON-wrapping integer lattice, sampled at
    /// arbitrary (already-scaled) lattice coordinates. Unlike `valueTiled` this does not
    /// wrap to a period — it's the building block for world-space (non-repeating) noise.
    @inline(__always)
    static func latticeValue(_ x: Double, _ z: Double, seed: UInt64) -> Double {
        let xi = Int(floor(x)), zi = Int(floor(z))
        let fx = x - floor(x), fz = z - floor(z)
        @inline(__always) func corner(_ di: Int, _ dj: Int) -> Double {
            unit(hash2(xi + di, zi + dj, seed))
        }
        let sx = smooth(fx), sz = smooth(fz)
        let top = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * sx
        let bot = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * sx
        return top + (bot - top) * sz
    }

    /// NON-TILING world-space fbm sampled at real-world coordinates (metres). Unlike
    /// `fbmTiled` (which wraps over a UV period, so its lowest frequency equals the texture
    /// tile and cannot carry variation larger than one tile), this keys directly off a world
    /// lattice at a chosen `featureMeters` scale, so it never repeats. It's the correct layer
    /// for LARGE-SCALE terrain macro variation: a 250 m+ ground patch tiles a 0.5 m material
    /// tile hundreds of times, so any tile-baked macro averages to a flat sheet — but a
    /// world-space tint at ~tens-of-metres feature size reads as broad natural sweeps across
    /// the whole patch. Returns `[0, 1]`, mean ≈ 0.5.
    public static func worldFbm(x: Double, z: Double, featureMeters: Double,
                                octaves: Int = 4, gain: Double = 0.5, seed: UInt64 = 0) -> Double {
        var sum = 0.0, amp = 1.0, total = 0.0
        var freq = 1.0 / max(featureMeters, 0.001)
        for o in 0..<max(1, octaves) {
            sum += amp * latticeValue(x * freq, z * freq, seed: seed &+ UInt64(o) &* 0x9E37_79B1)
            total += amp; amp *= gain; freq *= 2.0
        }
        return total > 0 ? sum / total : 0.5
    }

    // MARK: Voronoi / Worley (tileable) — the aggregate node

    /// One Worley sample: nearest (`f1`) and second-nearest (`f2`) feature-point
    /// distances in **cell units**, the winning cell's hash (`cellId`, a stable per-chip
    /// key), and that feature point in UV. `f2 - f1` is small at cell boundaries — the
    /// binder seam between terrazzo chips (§5). Toroidal: feature points one tile over
    /// reuse the wrapped cell's hash, so chips read continuously across the seam.
    public struct Cell: Equatable, Sendable {
        public var f1: Double, f2: Double, cellId: UInt64, center: Vec2
    }

    /// Scatter one jittered feature point per `cells × cells` grid cell and find the two
    /// nearest to `(u, v)`. `jitter` 0 → a regular grid, 1 → fully random within the cell.
    public static func voronoiTiled(_ u: Double, _ v: Double, cells: Int,
                                    jitter: Double = 0.9, seed: UInt64 = 0) -> Cell {
        voronoiTiledAniso(u, v, cellsX: cells, cellsY: cells, jitter: jitter, seed: seed)
    }

    /// `voronoiTiled` with an independent cell count per axis — a lattice of **elongated**
    /// cells, for material features that are pressed flat in one direction (cork's
    /// granule flakes, a shale bed, a stretched pebble). `cellsY > cellsX` gives cells
    /// stretched along `u`.
    ///
    /// Distances are measured in **cell-index units**, so `f1`/`f2` keep exactly the
    /// meaning they have in the isotropic call and the cells stay round in index space
    /// while reading as flattened in UV. Toroidal on both axes independently, so the
    /// wrap is seamless at any aspect. `u`/`v` may lie outside `[0, 1]` — the lattice is
    /// periodic with period 1, so a caller may offset the domain freely (cork gives each
    /// floor tile its own offset so no flake runs across a joint).
    public static func voronoiTiledAniso(_ u: Double, _ v: Double, cellsX: Int, cellsY: Int,
                                         jitter: Double = 0.9, seed: UInt64 = 0) -> Cell {
        let nx = max(1, cellsX), ny = max(1, cellsY)
        let x = u * Double(nx), y = v * Double(ny)
        let xi = Int(floor(x)), yi = Int(floor(y))
        var f1 = Double.greatestFiniteMagnitude, f2 = f1
        var id: UInt64 = 0, center = Vec2(0, 0)
        for dj in -1...1 {
            for di in -1...1 {
                let cx = xi + di, cy = yi + dj            // unwrapped — keeps distance toroidal
                let h = hash2(wrap(cx, nx), wrap(cy, ny), seed)
                let jx = 0.5 + (unit(h) - 0.5) * jitter
                let jy = 0.5 + (unit(mix(h)) - 0.5) * jitter
                let px = Double(cx) + jx, py = Double(cy) + jy
                let d = ((px - x) * (px - x) + (py - y) * (py - y)).squareRoot()
                if d < f1 {
                    f2 = f1; f1 = d; id = h
                    center = Vec2((Double(wrap(cx, nx)) + jx) / Double(nx),
                                  (Double(wrap(cy, ny)) + jy) / Double(ny))
                } else if d < f2 {
                    f2 = d
                }
            }
        }
        return Cell(f1: f1, f2: f2, cellId: id, center: center)
    }
}
