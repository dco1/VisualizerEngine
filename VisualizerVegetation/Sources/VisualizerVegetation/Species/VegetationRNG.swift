import Foundation

/// A tiny, deterministic PRNG (SplitMix64) shared by every procedural builder in this module —
/// `PottedPlantMesh` (as `SplitMix`) and `ConiferBuilder`. Reproducible per seed; never an unseeded
/// `.random`, and never seeded from `hashValue` (Swift re-seeds hashing every process).
public struct VegetationRNG: Sendable {
    public var state: UInt64
    public init(_ seed: UInt64) { state = seed == 0 ? 0x9E3779B97F4A7C15 : seed }
    public mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    /// A double in [0, 1).
    public mutating func unit() -> Double { Double(next() >> 11) * (1.0 / 9007199254740992.0) }
}
