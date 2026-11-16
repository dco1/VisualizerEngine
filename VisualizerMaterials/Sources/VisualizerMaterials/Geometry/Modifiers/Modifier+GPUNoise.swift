//  Modifier+GPUNoise.swift — a Displace whose field is the procedural fBm as DATA, so it can run on the GPU.
//
//  `DisplaceModifier` takes a Swift closure, which a Metal kernel cannot call: closure-based fields stay on the CPU
//  (the stack evaluator runs them there and says so in its report). `NoiseDisplaceModifier` holds the same
//  `ProceduralNoise.fbm` parameters as plain numbers; its CPU `apply` is exactly `DisplaceModifier.noise`, its GPU
//  path is the `displace_noise` kernel (integer hash bit-for-bit, Float interpolation).

import simd

public struct NoiseDisplaceModifier: Modifier {
    public var strength: Double
    public var frequency: Double
    public var octaves: Int
    public var lacunarity: Double
    public var gain: Double
    public var seed: UInt64
    public var midlevel: Double
    public var direction: DisplaceModifier.Direction

    public init(strength: Double, frequency: Double, octaves: Int = 4, lacunarity: Double = 2, gain: Double = 0.5,
                seed: UInt64 = 0, midlevel: Double = 0.5, direction: DisplaceModifier.Direction = .normal) {
        self.strength = strength; self.frequency = frequency; self.octaves = octaves; self.lacunarity = lacunarity
        self.gain = gain; self.seed = seed; self.midlevel = midlevel; self.direction = direction
    }

    public var summary: String { "NoiseDisplace(f:\(frequency) o:\(octaves) lac:\(lacunarity) gain:\(gain) seed:\(seed) strength:\(strength) mid:\(midlevel))" }

    /// The CPU oracle: the closure-based modifier with the same field.
    public var cpuModifier: DisplaceModifier {
        let (f, o, l, g, s) = (frequency, octaves, lacunarity, gain, seed)
        return DisplaceModifier(label: "noise", strength: strength, midlevel: midlevel, direction: direction) { p, _ in
            ProceduralNoise.fbm(p, frequency: f, octaves: o, lacunarity: l, gain: g, seed: s)
        }
    }

    public func apply(_ mesh: Mesh3) -> Mesh3 { cpuModifier.apply(mesh) }
}
