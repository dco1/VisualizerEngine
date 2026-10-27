import simd

/// One plant of a planting — an INDIVIDUAL: where it stands and how it differs from every other
/// plant baked from the same shape. Everything a renderer needs to vary an instance without a
/// re-bake (scale, yaw, lean, tint, wind phase), drawn from the plant's own seeded stream.
public struct ScatterPoint: Sendable, Equatable {
    /// Plan position (m).
    public var position: SIMD2<Double>
    /// Ground height under it (m), from the rule's terrain.
    public var groundHeight: Double
    /// `ScatterSpecies.id` and the baked shape variant (0 ..< variantCount).
    public var species: Int
    public var variant: Int
    /// 0 (young) … 1 (mature); `scale` follows it through the species' range.
    public var age: Double
    public var scale: Double
    /// Canopy radius as planted (species radius × scale).
    public var radius: Double
    /// Rotation about vertical (rad).
    public var yaw: Double
    /// Lean off vertical (rad) toward the plan direction `tiltAzimuth` (rad).
    public var tilt: Double
    public var tiltAzimuth: Double
    /// Multiplicative albedo tint, ≈ 1 ± a few % per channel (a plant's own health and light).
    public var tint: SIMD3<Float>
    /// Wind phase offset (rad, 0 ..< 2π) — no two plants of one shape sway in step.
    public var windPhase: Float
    /// Stable identity within the planting (the world cell it was drawn in) — keys anything a
    /// host caches per plant.
    public var key: UInt64
    /// Canopy layer (`ScatterSpecies.stratum`) and the trunk clearance as planted (stem radius ×
    /// scale) — what another stratum's plants keep clear of. DH-1470.
    public var stratum: Int = 0
    public var stemRadius: Double = 0
}
