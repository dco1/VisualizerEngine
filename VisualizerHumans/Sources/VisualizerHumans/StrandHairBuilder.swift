import simd

// Strand hair as geometry: hundreds of tapered 3-sided tubes grown from scalp
// seeds, packed into ONE GarmentMesh whose vertices are all weighted to the
// head bone — so it rides the ordinary garment skinning path (CPU or GPU) and
// tracks the head everywhere, unlike engine RT curves (which live only in the
// acceleration structures and never appear in primary rasterization).
// Layered over the scalp cap, which provides the dark under-fill.
public enum StrandHairBuilder {

    public static func strands(for human: GeneratedHuman, style: HairStyle) -> GarmentMesh? {
        guard !style.bald, let headIdx = human.boneIndex["head"] else { return nil }
        let head = human.bones[headIdx]
        let up = simd_normalize(head.tail - head.head)
        let skullSpan = simd_distance(head.tail, head.head)
        var forward = SIMD3<Float>(0, 0, 1) - up * up.z
        forward = simd_length(forward) > 1e-5 ? simd_normalize(forward) : SIMD3(0, 0, 1)

        // Same scalp criterion as the cap so the layers agree.
        var seeds: [Int] = []
        for v in 0..<human.positions.count {
            var headWeight: Float = 0
            let j = human.skinJoints[v]
            let w = human.skinWeights[v]
            for k in 0..<4 where Int(j[k]) == headIdx { headWeight += w[k] }
            guard headWeight >= 0.5 else { continue }
            let rel = human.positions[v] - head.head
            let h = simd_dot(rel, up) / max(0.001, skullSpan)
            let f = simd_dot(rel, forward)
            let hairline = f > 0.01 ? style.hairlineHeight + f * 3.2 : 0.20
            if h >= hairline { seeds.append(v) }
        }
        guard !seeds.isEmpty else { return nil }

        var rng = human.spec.seed &* 0x853C_49E6_748F_EA9B | 1
        func rand() -> Float {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
            return Float(rng >> 40) / Float(1 << 24)
        }

        let segments = 5
        let baseRadius: Float = 0.0013
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var uvs: [SIMD2<Float>] = []
        var joints: [SIMD4<UInt16>] = []
        var weights: [SIMD4<Float>] = []
        var triangles: [UInt32] = []

        for (si, seed) in seeds.enumerated() where si % 2 == 0 {
            let root = human.positions[seed]
            let n = human.normals[seed]
            var dir = simd_normalize(n * 0.55 + up * 0.25
                + SIMD3(rand() - 0.5, rand() - 0.5, rand() - 0.5) * 0.25)
            let length = style.strandLength * (0.75 + rand() * 0.5)
            let segLen = length / Float(segments)

            var centers: [SIMD3<Float>] = [root + n * 0.002]
            for s in 1...segments {
                // Gravity takes over after the first segment; strands drape.
                let gravity: Float = s < 2 ? 0.12 : 0.55
                dir = simd_normalize(dir + SIMD3(0, -gravity, 0))
                centers.append(centers[s - 1] + dir * segLen)
            }

            let ringBase = UInt32(positions.count)
            for (ci, c) in centers.enumerated() {
                let t = Float(ci) / Float(segments)
                let radius = baseRadius * (1 - t * 0.8)
                let axis = ci < segments
                    ? simd_normalize(centers[ci + 1] - c)
                    : simd_normalize(c - centers[ci - 1])
                var side = simd_cross(axis, SIMD3(0, 1, 0))
                if simd_length(side) < 1e-4 { side = simd_cross(axis, SIMD3(1, 0, 0)) }
                side = simd_normalize(side)
                let side2 = simd_normalize(simd_cross(axis, side))
                for k in 0..<3 {
                    let a = Float(k) / 3 * 2 * .pi
                    let radial = side * cos(a) + side2 * sin(a)
                    positions.append(c + radial * radius)
                    normals.append(radial)
                    uvs.append(SIMD2(0, t))
                    joints.append(SIMD4(UInt16(headIdx), 0, 0, 0))
                    weights.append(SIMD4(1, 0, 0, 0))
                }
            }
            for s in 0..<segments {
                let r0 = ringBase + UInt32(s * 3)
                let r1 = r0 + 3
                for k in 0..<3 {
                    let k1 = UInt32((k + 1) % 3)
                    let a = r0 + UInt32(k), b = r0 + k1, c = r1 + UInt32(k), d = r1 + k1
                    triangles += [a, c, b, b, c, d]
                }
            }
        }
        guard !triangles.isEmpty else { return nil }
        return GarmentMesh(
            positions: positions, normals: normals, uvs: uvs, triangles: triangles,
            skinJoints: joints, skinWeights: weights,
            color: style.color * 1.15, roughness: 0.6)
    }
}
