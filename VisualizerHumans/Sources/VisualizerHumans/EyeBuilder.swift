import simd

// Procedural eyeballs: sclera + iris sphere pair per eye, placed at the eye
// bones' heads (MakeHuman ships eyeballs as separate assets, so the body mesh
// has empty sockets). The host renders each as a unit sphere scaled by
// `radius`, carried by the posed eye-bone world for head tracking.
public struct EyeSphere: Sendable {
    public let center: SIMD3<Float>      // bind space; transform by boneWorlds[boneIndex]
    public let radius: Float
    public let color: SIMD3<Float>
    public let roughness: Float
    public let boneIndex: Int
}

public enum EyeBuilder {

    // Roughly world-realistic iris distribution, deterministic per seed.
    public static func irisColor(seed: UInt64) -> SIMD3<Float> {
        var rng = seed &* 0x9E3779B97F4A7C15 | 1
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
        let roll = Float(rng >> 40) / Float(1 << 24)
        switch roll {
        case ..<0.55: return SIMD3(0.16, 0.09, 0.05)    // brown
        case ..<0.72: return SIMD3(0.08, 0.06, 0.04)    // dark brown
        case ..<0.82: return SIMD3(0.16, 0.25, 0.35)    // blue
        case ..<0.90: return SIMD3(0.18, 0.22, 0.12)    // hazel-green
        case ..<0.96: return SIMD3(0.13, 0.20, 0.16)    // green
        default: return SIMD3(0.22, 0.24, 0.26)         // gray
        }
    }

    public static func eyes(for human: GeneratedHuman) -> [EyeSphere] {
        // Eyes scale sublinearly with stature — children keep relatively big eyes.
        let radius = 0.0115 * pow(human.heightMeters / 1.75, 0.3)
        let iris = irisColor(seed: human.spec.seed &+ 7)
        var out: [EyeSphere] = []
        for name in ["eye.L", "eye.R"] {
            guard let bi = human.boneIndex[name] else { continue }
            // Inset so the lids cover the sclera's rim — a fully exposed ball stares.
            let center = human.bones[bi].head - SIMD3(0, 0, radius * 0.18)
            out.append(EyeSphere(center: center, radius: radius,
                                 color: SIMD3(0.93, 0.92, 0.90), roughness: 0.22, boneIndex: bi))
            out.append(EyeSphere(center: center + SIMD3(0, 0, radius * 0.58), radius: radius * 0.62,
                                 color: iris, roughness: 0.15, boneIndex: bi))
        }
        return out
    }
}
