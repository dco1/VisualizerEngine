import Foundation

/// The integer hash behind every deterministic choice in a flora field (which plants a thin level
/// keeps, a plant's tint jitter, a leaflet's survival). Pure, seedable, stable across launches —
/// never `hashValue`, which Swift re-seeds every process.
public enum FloraHash {
    @inline(__always) public static func hash(_ x: Int, _ y: Int, _ s: UInt64) -> UInt64 {
        var h = (UInt64(bitPattern: Int64(x)) &* 0x9E37_79B9_7F4A_7C15)
            ^ (UInt64(bitPattern: Int64(y)) &* 0xC2B2_AE3D_27D4_EB4F) ^ (s &* 0xD1B5_4A32_D192_ED03)
        h ^= h >> 31; h &*= 0xBF58_476D_1CE4_E5B9
        h ^= h >> 29; h &*= 0x94D0_49BB_1331_11EB
        h ^= h >> 32
        return h
    }

    /// Deterministic 0…1 for a small integer tuple.
    @inline(__always) public static func unit(_ x: Int, _ y: Int, _ s: UInt64) -> Double {
        Double(hash(x, y, s) >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }
}
