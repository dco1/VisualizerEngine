import simd

// Procedural hair as a scalp-cap shell: head-dominated body triangles above a
// hairline plane (defined in the head bone's frame so it tracks every skull
// shape), smoothed and inflated. Not strand hair — but a colored, styled cap
// covers the "no hair" uncanny gap at mid-shot, and it skins with the head.
public struct HairStyle: Sendable {
    public var color: SIMD3<Float>
    public var hairlineHeight: Float     // 0…1 up the skull; higher = more forehead
    public var thickness: Float          // shell inflation in meters
    public var strandLength: Float       // strand-layer length in meters
    public var bald: Bool

    public init(color: SIMD3<Float>, hairlineHeight: Float = 0.42,
                thickness: Float = 0.012, strandLength: Float = 0.06, bald: Bool = false) {
        self.color = color
        self.hairlineHeight = hairlineHeight
        self.thickness = thickness
        self.strandLength = strandLength
        self.bald = bald
    }

    // Age-aware deterministic roll: seniors trend gray and recede; a slice of
    // everyone is bald on purpose.
    public static func random(seed: UInt64, ageYears: Float) -> HairStyle {
        var rng = seed &* 0x2545_F491_4F6C_DD1D | 1
        func next() -> Float {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
            return Float(rng >> 40) / Float(1 << 24)
        }
        let palette: [SIMD3<Float>] = [
            SIMD3(0.05, 0.04, 0.03),   // black
            SIMD3(0.16, 0.10, 0.05),   // dark brown
            SIMD3(0.35, 0.22, 0.10),   // brown
            SIMD3(0.55, 0.35, 0.12),   // auburn
            SIMD3(0.72, 0.58, 0.30),   // blonde
            SIMD3(0.45, 0.16, 0.08),   // red
        ]
        var color = palette[Int(next() * 0.999 * Float(palette.count))]
        let grayness = min(1, max(0, (ageYears - 45) / 30) * (0.4 + next() * 0.6))
        color = simd_mix(color, SIMD3(0.75, 0.74, 0.72), SIMD3(repeating: grayness))
        let recede = ageYears > 35 && next() < 0.3 ? next() * 0.15 : 0
        return HairStyle(
            color: color,
            hairlineHeight: 0.38 + next() * 0.12 + recede,
            thickness: 0.008 + next() * 0.012,
            strandLength: 0.035 + next() * 0.07,
            bald: next() < 0.08)
    }
}

public enum HairBuilder {

    public static func hair(for human: GeneratedHuman, style: HairStyle) -> GarmentMesh? {
        guard !style.bald, let headIdx = human.boneIndex["head"] else { return nil }
        let head = human.bones[headIdx]
        let up = simd_normalize(head.tail - head.head)          // skull base → crown
        let skullSpan = simd_distance(head.tail, head.head)
        // Model faces +Z: forward is world Z projected off the skull axis.
        var forward = SIMD3<Float>(0, 0, 1) - up * up.z
        forward = simd_length(forward) > 1e-5 ? simd_normalize(forward) : SIMD3(0, 0, 1)

        // Head-weight majority AND above the hairline surface. The hairline dips
        // at the back (nape hair grows lower) and rises over the face.
        var inCap = [Bool](repeating: false, count: human.positions.count)
        for v in 0..<human.positions.count {
            var headWeight: Float = 0
            let j = human.skinJoints[v]
            let w = human.skinWeights[v]
            for k in 0..<4 where Int(j[k]) == headIdx { headWeight += w[k] }
            guard headWeight >= 0.5 else { continue }

            let rel = human.positions[v] - head.head
            let h = simd_dot(rel, up) / max(0.001, skullSpan)   // 0 base … 1 crown
            let f = simd_dot(rel, forward)                       // meters forward of skull axis
            // Nape keeps hair down to h≈0.15; the face needs h above the forehead.
            let hairline = f > 0.01 ? style.hairlineHeight + f * 3.2 : 0.20
            inCap[v] = h >= hairline
        }

        var remap = [Int32](repeating: -1, count: human.positions.count)
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var uvs: [SIMD2<Float>] = []
        var joints: [SIMD4<UInt16>] = []
        var weights: [SIMD4<Float>] = []
        var triangles: [UInt32] = []
        let tris = human.triangles
        for t in stride(from: 0, to: tris.count, by: 3) {
            let a = Int(tris[t]), b = Int(tris[t + 1]), c = Int(tris[t + 2])
            let members = (inCap[a] ? 1 : 0) + (inCap[b] ? 1 : 0) + (inCap[c] ? 1 : 0)
            guard members >= 2 else { continue }
            for v in [a, b, c] {
                if remap[v] == -1 {
                    remap[v] = Int32(positions.count)
                    positions.append(human.positions[v] + human.normals[v] * style.thickness)
                    normals.append(human.normals[v])
                    uvs.append(human.uvs[v])
                    joints.append(human.skinJoints[v])
                    weights.append(human.skinWeights[v])
                }
                triangles.append(UInt32(remap[v]))
            }
        }
        guard !triangles.isEmpty else { return nil }
        return GarmentMesh(
            positions: positions, normals: normals, uvs: uvs, triangles: triangles,
            skinJoints: joints, skinWeights: weights, color: style.color, roughness: 0.72)
    }
}
