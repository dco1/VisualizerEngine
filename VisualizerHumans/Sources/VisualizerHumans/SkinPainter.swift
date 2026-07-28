import simd

// Per-vertex skin paint. The renderer multiplies vertex color over instance
// albedo, so the instance carries the melanin base tone and this layer adds
// the life: lip/cheek/ear flush, brow shading, nail keratin, joint redness,
// age mottling. All features are painted in bind space off skeleton landmarks,
// so they land correctly on every body shape and age.
public enum SkinPainter {

    // The melanin base for the instance albedo.
    public static func baseTone(_ spec: HumanSpec) -> SIMD3<Float> {
        simd_mix(SIMD3(0.88, 0.69, 0.56), SIMD3(0.30, 0.19, 0.13), SIMD3(repeating: spec.skinTone))
    }

    public static func paint(human: GeneratedHuman) -> [SIMD4<Float>] {
        let spec = human.spec
        var colors = [SIMD4<Float>](repeating: SIMD4(1, 1, 1, 1), count: human.positions.count)

        func bone(_ name: String) -> GeneratedHuman.BindBone? {
            human.boneIndex[name].map { human.bones[$0] }
        }
        guard let head = bone("head") else { return colors }
        let up = simd_normalize(head.tail - head.head)
        let span = simd_distance(head.tail, head.head)
        var forward = SIMD3<Float>(0, 0, 1) - up * up.z
        forward = simd_length(forward) > 1e-5 ? simd_normalize(forward) : SIMD3(0, 0, 1)

        let eyeL = bone("eye.L")?.head ?? head.head
        let eyeR = bone("eye.R")?.head ?? head.head
        let eyeMid = (eyeL + eyeR) / 2
        // Mouth estimate: below the eye line, on the face front. Scales with the skull.
        let mouth = eyeMid - up * (span * 0.52) + forward * (span * 0.10)

        // Flush landmarks: elbows, knees, knuckle rows.
        var flushPoints: [SIMD3<Float>] = []
        for name in ["lowerarm01.L", "lowerarm01.R", "lowerleg01.L", "lowerleg01.R"] {
            if let b = bone(name) { flushPoints.append(b.head) }
        }
        for side in ["L", "R"] {
            for finger in 1...5 {
                if let b = bone("finger\(finger)-1.\(side)") { flushPoints.append(b.head) }
            }
        }
        // Nail tips: last finger/toe segments' tails.
        var nailPoints: [SIMD3<Float>] = []
        for side in ["L", "R"] {
            for finger in 1...5 {
                if let b = bone("finger\(finger)-3.\(side)") { nailPoints.append(b.tail) }
            }
            for toe in 1...5 {
                if let b = bone("toe\(toe)-2.\(side)") ?? bone("toe\(toe)-3.\(side)") {
                    nailPoints.append(b.tail)
                }
            }
        }

        let ageMottle = min(1, max(0, (spec.ageYears - 30) / 50))
        let lipTint = SIMD3<Float>(1.22, 0.82, 0.80)
        let flushTint = SIMD3<Float>(1.10, 0.92, 0.90)
        let browTint = SIMD3<Float>(0.86, 0.84, 0.84)
        let nailTint = SIMD3<Float>(1.14, 1.10, 1.05)

        func gauss(_ d: Float, _ radius: Float) -> Float {
            let x = d / radius
            return exp(-x * x * 3)
        }

        for v in 0..<human.positions.count {
            let p = human.positions[v]
            var tint = SIMD3<Float>(1, 1, 1)

            // Low-frequency mottling: hash noise on position, stronger with age.
            let h = sin(p.x * 61.3 + p.y * 47.7) * sin(p.y * 53.1 + p.z * 71.9)
            tint *= 1 + h * (0.025 + 0.045 * ageMottle)

            // Face features only near the skull.
            let toHead = p - head.head
            if simd_length(toHead) < span * 2.2 {
                let mouthT = gauss(simd_distance(p, mouth), span * 0.24)
                    * max(0, simd_dot(simd_normalize(p - eyeMid + SIMD3(0, 0.001, 0)), forward) + 0.35)
                tint = simd_mix(tint, tint * lipTint, SIMD3(repeating: min(0.85, mouthT)))

                // Brow band: just above the eye line, front-facing.
                let above = simd_dot(p - eyeMid, up)
                let front = simd_dot(p - eyeMid, forward)
                if above > span * 0.06, above < span * 0.22, front > 0 {
                    let browT = gauss(above - span * 0.13, span * 0.06) * 0.5
                    tint = simd_mix(tint, tint * browTint, SIMD3(repeating: browT))
                }

                // Cheek / ear / nose-tip warmth: side extremes and face front near eye height.
                let lateral = abs(simd_dot(p - eyeMid, simd_cross(up, forward)))
                let cheekT = gauss(abs(above + span * 0.28), span * 0.24) * min(1, lateral / (span * 0.4)) * 0.5
                let earT = lateral > span * 0.52 && abs(above) < span * 0.35 ? Float(0.55) : 0
                tint = simd_mix(tint, tint * flushTint, SIMD3(repeating: min(0.8, cheekT + earT)))
            }

            for fp in flushPoints {
                let t = gauss(simd_distance(p, fp), 0.05) * 0.35
                if t > 0.01 { tint = simd_mix(tint, tint * flushTint, SIMD3(repeating: t)) }
            }
            for np in nailPoints {
                let t = gauss(simd_distance(p, np), 0.014)
                if t > 0.05 { tint = simd_mix(tint, tint * nailTint, SIMD3(repeating: min(1, t * 1.4))) }
            }

            colors[v] = SIMD4(tint, 1)
        }
        return colors
    }
}
