import simd

/// SplitMix64 — the small, fast, process-independent generator every scatter draw comes from.
///
/// The same generator (and the same per-cell seeding constants) as the engine's wildflower sowing
/// (`WildflowerPlantingMesh.plants`, which this module's cell scheme is lifted from), so a planting
/// is a pure function of (seed, world cell) — never of `hashValue`, which Swift re-seeds per process.
public struct ScatterRandom: Sendable {
    public private(set) var state: UInt64

    public init(_ seed: UInt64) { state = seed }

    /// The generator for one world-anchored cell `(ix, iy)` of stream `salt` under `seed`.
    public init(seed: UInt64, cell ix: Int, _ iy: Int, salt: UInt64 = 0) {
        state = seed &+ UInt64(bitPattern: Int64(ix)) &* 0x9E37_79B9_7F4A_7C15
                     &+ UInt64(bitPattern: Int64(iy)) &* 0xC2B2_AE3D_27D4_EB4F
                     &+ salt &* 0x1656_67B1_9E37_79F9
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    public mutating func unit() -> Double { Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0) }

    /// Uniform in `range`.
    public mutating func uniform(_ range: ClosedRange<Double>) -> Double {
        range.lowerBound + (range.upperBound - range.lowerBound) * unit()
    }
}

/// Smooth value noise in [0, 1) on an integer lattice — deterministic and process-independent.
/// The species DRIFT field (a mix gathers into patches instead of salt-and-pepper), lifted from
/// `WildflowerPlantingMesh.valueNoise`.
enum ScatterNoise {
    static func value(_ p: SIMD2<Double>, seed: UInt64) -> Double {
        func h(_ x: Int, _ y: Int) -> Double {
            var r = ScatterRandom(seed: seed, cell: x, y)
            return r.unit()
        }
        func smooth(_ t: Double) -> Double { t * t * (3 - 2 * t) }
        let xf = p.x.rounded(.down), yf = p.y.rounded(.down)
        let x = Int(xf), y = Int(yf)
        let tx = smooth(p.x - xf), ty = smooth(p.y - yf)
        let a = h(x, y) + (h(x + 1, y) - h(x, y)) * tx
        let b = h(x, y + 1) + (h(x + 1, y + 1) - h(x, y + 1)) * tx
        return a + (b - a) * ty
    }
}
