import Foundation

/// SplitMix64: a small, deterministic generator. A seed gives the same numbers every run and on every platform (never
/// `hashValue`, which Swift re-seeds per process). Also a `RandomNumberGenerator`, so it drives `shuffled(using:)` and
/// `random(in:using:)`. The seed is used as given (0 is a valid seed); the stream is the same as `Noise.mix`'s finalizer
/// walked along the golden-ratio increment.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    public var state: UInt64
    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in 0..<1, 53 bits.
    public mutating func nextUnit() -> Double { Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0) }
    /// Uniform in 0..<1, 24 bits.
    public mutating func nextUnitFloat() -> Float { Float(next() >> 40) / Float(1 << 24) }
}
