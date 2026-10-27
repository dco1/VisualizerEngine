/// 64-bit FNV-1a — the one stable (cross-launch) hash for cache keys, filenames and seeds.
///
/// Never `hashValue`: Swift re-seeds `Hasher` every process, so a hash-derived key or seed
/// reshuffles on each launch. Byte-wise over UTF-8 / raw bytes, or word-wise (one xor-multiply
/// per 32/64-bit word) for hashing float bit patterns; the two are different hashes, so a call
/// site keeps whichever it always used.
public enum FNV1a {
    public static let offsetBasis: UInt64 = 0xcbf2_9ce4_8422_2325
    public static let prime: UInt64 = 0x0000_0100_0000_01b3

    /// Byte-wise over any byte sequence, continuing from `seed` (default: the offset basis).
    @inlinable
    public static func hash<S: Sequence>(bytes: S, seed: UInt64 = offsetBasis) -> UInt64 where S.Element == UInt8 {
        var h = seed
        for b in bytes { h = (h ^ UInt64(b)) &* prime }
        return h
    }

    /// Byte-wise over a string's UTF-8.
    @inlinable
    public static func hash(_ string: String, seed: UInt64 = offsetBasis) -> UInt64 {
        hash(bytes: string.utf8, seed: seed)
    }

    /// Word-wise: one xor-multiply per word (e.g. `Float.bitPattern` values).
    @inlinable
    public static func fold(_ word: UInt64, into h: UInt64) -> UInt64 { (h ^ word) &* prime }
}
