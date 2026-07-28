import simd

// Basic clothing as inflated shells of the body itself: pick body triangles by
// the dominant skinning bone, push them out along the vertex normal, keep the
// body's skin weights. Garments therefore inherit every morph and every pose
// for free — one wardrobe fits every body type. Deliberately simple (no cloth
// sim, no seams); reads as fitted knitwear at mid-shot.
public struct Outfit: Sendable {
    public var shirtColor: SIMD3<Float>
    public var pantsColor: SIMD3<Float>
    public var shoeColor: SIMD3<Float>
    public var longSleeves: Bool
    public var longPants: Bool

    public init(
        shirtColor: SIMD3<Float> = SIMD3(0.75, 0.22, 0.18),
        pantsColor: SIMD3<Float> = SIMD3(0.18, 0.22, 0.35),
        shoeColor: SIMD3<Float> = SIMD3(0.12, 0.10, 0.09),
        longSleeves: Bool = false,
        longPants: Bool = true
    ) {
        self.shirtColor = shirtColor
        self.pantsColor = pantsColor
        self.shoeColor = shoeColor
        self.longSleeves = longSleeves
        self.longPants = longPants
    }

    // Deterministic wardrobe roll for crowds.
    public static func random(seed: UInt64) -> Outfit {
        var rng = seed &* 0x9E3779B97F4A7C15 | 1
        func next() -> Float {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
            return Float(rng >> 40) / Float(1 << 24)
        }
        func color() -> SIMD3<Float> { SIMD3(0.1 + next() * 0.8, 0.1 + next() * 0.8, 0.1 + next() * 0.8) }
        return Outfit(shirtColor: color(), pantsColor: color() * 0.6,
                      shoeColor: SIMD3(repeating: 0.08 + next() * 0.25),
                      longSleeves: next() < 0.35, longPants: next() < 0.75)
    }
}

public struct GarmentMesh: Sendable {
    public let positions: [SIMD3<Float>]
    public let normals: [SIMD3<Float>]
    public let uvs: [SIMD2<Float>]
    public let triangles: [UInt32]
    public let skinJoints: [SIMD4<UInt16>]
    public let skinWeights: [SIMD4<Float>]
    public let color: SIMD3<Float>
    public let roughness: Float
}

public enum ClothingBuilder {

    public static func garments(for human: GeneratedHuman, outfit: Outfit) -> [GarmentMesh] {
        // Spine numbering runs DOWNWARD (spine01 = upper chest, spine05 = lower
        // abdomen), so spine05 belongs to the pants' waistband, not the shirt.
        var shirtBones: Set<String> = [
            "spine01", "spine02", "spine03", "spine04",
            "breast.L", "breast.R", "clavicle.L", "clavicle.R",
            "shoulder01.L", "shoulder01.R",
            "upperarm01.L", "upperarm01.R", "upperarm02.L", "upperarm02.R",
        ]
        if outfit.longSleeves {
            shirtBones.formUnion(["lowerarm01.L", "lowerarm01.R", "lowerarm02.L", "lowerarm02.R"])
        }
        var pantsBones: Set<String> = [
            "root", "pelvis.L", "pelvis.R", "spine05",
            "upperleg01.L", "upperleg01.R", "upperleg02.L", "upperleg02.R",
        ]
        if outfit.longPants {
            pantsBones.formUnion(["lowerleg01.L", "lowerleg01.R", "lowerleg02.L", "lowerleg02.R"])
        }
        var shoeBones: Set<String> = ["foot.L", "foot.R"]
        for side in ["L", "R"] {
            for toe in 1...5 {
                for seg in 1...4 { shoeBones.insert("toe\(toe)-\(seg).\(side)") }
            }
        }

        func boneSetIndices(_ names: Set<String>) -> Set<UInt16> {
            Set(names.compactMap { human.boneIndex[$0].map { UInt16($0) } })
        }

        return [
            shell(human: human, bones: boneSetIndices(shirtBones), offset: 0.014,
                  color: outfit.shirtColor, roughness: 0.85),
            shell(human: human, bones: boneSetIndices(pantsBones), offset: 0.009,
                  color: outfit.pantsColor, roughness: 0.8),
            shell(human: human, bones: boneSetIndices(shoeBones), offset: 0.014,
                  color: outfit.shoeColor, roughness: 0.45),
        ].compactMap { $0 }
    }

    private static func shell(
        human: GeneratedHuman, bones: Set<UInt16>, offset: Float,
        color: SIMD3<Float>, roughness: Float
    ) -> GarmentMesh? {
        // A vertex belongs to the garment when the garment bones carry the
        // majority of its skin weight. Weight falloff is smooth along a limb, so
        // this yields far cleaner hemlines than dominant-bone membership.
        var inGarment = [Bool](repeating: false, count: human.positions.count)
        for v in 0..<human.positions.count {
            var score: Float = 0
            let j = human.skinJoints[v]
            let w = human.skinWeights[v]
            for k in 0..<4 where bones.contains(j[k]) { score += w[k] }
            inGarment[v] = score >= 0.5
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
            // Majority rule: requiring all three verts leaves a jagged one-ring
            // gap along every garment boundary; two of three closes the seam.
            let members = (inGarment[a] ? 1 : 0) + (inGarment[b] ? 1 : 0) + (inGarment[c] ? 1 : 0)
            guard members >= 2 else { continue }
            for v in [a, b, c] {
                if remap[v] == -1 {
                    remap[v] = Int32(positions.count)
                    positions.append(human.positions[v])   // offset applied after smoothing
                    normals.append(human.normals[v])
                    uvs.append(human.uvs[v])
                    joints.append(human.skinJoints[v])
                    weights.append(human.skinWeights[v])
                }
                triangles.append(UInt32(remap[v]))
            }
        }
        guard !triangles.isEmpty else { return nil }

        let bodyPositions = positions          // pre-smoothing body surface, 1:1 with shell verts
        let bodyNormals = normals
        refine(positions: &positions, normals: &normals, triangles: triangles, offset: offset)
        // Smoothing shrinks concave/high-curvature regions (thighs!) and can pull
        // the shell inside the skin. Clamp every vertex to a minimum radial
        // clearance from its body vertex.
        let clearance = max(0.004, offset * 0.45)
        for v in 0..<positions.count {
            let radial = simd_dot(positions[v] - bodyPositions[v], bodyNormals[v])
            if radial < clearance {
                positions[v] += bodyNormals[v] * (clearance - radial)
            }
        }
        return GarmentMesh(
            positions: positions, normals: normals, uvs: uvs, triangles: triangles,
            skinJoints: joints, skinWeights: weights, color: color, roughness: roughness)
    }

    // Make the shell read as FABRIC rather than paint: Laplacian-smooth so body
    // anatomy (navel, muscle, chest detail) stops telegraphing through, then
    // inflate along the smoothed normals, tapering to near-zero at the open
    // boundary so hems hug the body like cuffs instead of ending in a step.
    private static func refine(
        positions: inout [SIMD3<Float>], normals: inout [SIMD3<Float>],
        triangles: [UInt32], offset: Float
    ) {
        let n = positions.count
        var neighbors = [Set<Int>](repeating: [], count: n)
        var edgeUse: [UInt64: Int] = [:]
        for t in stride(from: 0, to: triangles.count, by: 3) {
            let v = [Int(triangles[t]), Int(triangles[t + 1]), Int(triangles[t + 2])]
            for k in 0..<3 {
                let a = v[k], b = v[(k + 1) % 3]
                neighbors[a].insert(b)
                neighbors[b].insert(a)
                let key = UInt64(min(a, b)) << 32 | UInt64(max(a, b))
                edgeUse[key, default: 0] += 1
            }
        }

        // Open-boundary verts (edges used once), then a few BFS rings inland.
        var hemRing = [Int](repeating: .max, count: n)
        var frontier: [Int] = []
        for (key, uses) in edgeUse where uses == 1 {
            let a = Int(key >> 32), b = Int(key & 0xFFFF_FFFF)
            for v in [a, b] where hemRing[v] != 0 {
                hemRing[v] = 0
                frontier.append(v)
            }
        }
        var ring = 0
        while !frontier.isEmpty && ring < 3 {
            ring += 1
            var next: [Int] = []
            for v in frontier {
                for nb in neighbors[v] where hemRing[nb] > ring {
                    hemRing[nb] = ring
                    next.append(nb)
                }
            }
            frontier = next
        }

        // Smooth the boundary LOOPS first: the hem polyline follows jagged
        // triangle edges; averaging each boundary vertex with its two loop
        // neighbors turns the sawtooth into a clean curve (the clearance clamp
        // downstream catches any resulting dip inside the body).
        var loopNeighbors = [[Int]](repeating: [], count: n)
        for (key, uses) in edgeUse where uses == 1 {
            let a = Int(key >> 32), b = Int(key & 0xFFFF_FFFF)
            loopNeighbors[a].append(b)
            loopNeighbors[b].append(a)
        }
        for _ in 0..<5 {
            var out = positions
            for v in 0..<n where hemRing[v] == 0 && loopNeighbors[v].count == 2 {
                let avg = (positions[loopNeighbors[v][0]] + positions[loopNeighbors[v][1]]) / 2
                out[v] = simd_mix(positions[v], avg, SIMD3(repeating: 0.6))
            }
            positions = out
        }

        // Smooth interior (hem ring 0 pinned so the opening keeps the body line).
        for _ in 0..<4 {
            var out = positions
            for v in 0..<n where hemRing[v] != 0 && !neighbors[v].isEmpty {
                var avg = SIMD3<Float>.zero
                for nb in neighbors[v] { avg += positions[nb] }
                avg /= Float(neighbors[v].count)
                out[v] = simd_mix(positions[v], avg, SIMD3(repeating: 0.5))
            }
            positions = out
        }

        // Area-weighted normals of the smoothed shell.
        var smoothNormals = [SIMD3<Float>](repeating: .zero, count: n)
        for t in stride(from: 0, to: triangles.count, by: 3) {
            let (a, b, c) = (Int(triangles[t]), Int(triangles[t + 1]), Int(triangles[t + 2]))
            let fn = simd_cross(positions[b] - positions[a], positions[c] - positions[a])
            smoothNormals[a] += fn
            smoothNormals[b] += fn
            smoothNormals[c] += fn
        }
        for v in 0..<n {
            let len = simd_length(smoothNormals[v])
            smoothNormals[v] = len > 1e-9 ? smoothNormals[v] / len : normals[v]
        }

        for v in 0..<n {
            let taper: Float = hemRing[v] >= 3 ? 1 : Float(hemRing[v]) / 3 * 0.8 + 0.15
            positions[v] += smoothNormals[v] * (offset * taper)
        }
        normals = smoothNormals
    }
}
