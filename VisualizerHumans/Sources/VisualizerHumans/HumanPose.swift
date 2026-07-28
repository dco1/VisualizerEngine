import simd

// A pose = rotation per bone, expressed in WORLD axes, pivoting about the
// bone's bind-pose head. World axes (not bone-local frames) keep gait math
// predictable: X is everyone's sagittal hinge, Y twist, Z lateral sway —
// regardless of how a bone's roll plane orients its local frame.
// `skinned(_:)` is the CPU reference implementation the Metal kernel must match.
public struct HumanPose: Sendable {
    public var rotations: [String: simd_quatf] = [:]
    public init() {}

    public mutating func set(_ bone: String, _ q: simd_quatf) { rotations[bone] = q }

    // Sagittal bend (world X): negative swings a limb forward, positive back;
    // positive flexes a knee (heel up-behind).
    public mutating func bendX(_ bone: String, _ radians: Float) {
        set(bone, simd_quatf(angle: radians, axis: SIMD3(1, 0, 0)))
    }
}

public struct PosedHuman {
    public let skinningMatrices: [float4x4]     // world * inverseBind, per bone
    public let boneWorlds: [float4x4]

    public init(human: GeneratedHuman, pose: HumanPose) {
        // Parent-carried world-axis FK, delta-from-bind form: each bone's rotation
        // is authored in WORLD axes as they stand at bind, pivoting on the bone's
        // bind head; ancestors premultiply, so a child's axis is carried along by
        // the posed parent (an adducted arm's elbow hinge tilts with the arm).
        // With verts in bind space the cumulative product IS the skinning matrix.
        // WalkCycle's aim-solver (simd shortest-arc) is built against exactly
        // these semantics — change one and you change both.
        var worlds = [float4x4](repeating: matrix_identity_float4x4, count: human.bones.count)
        for (i, bone) in human.bones.enumerated() {
            let q = pose.rotations[bone.name] ?? simd_quatf(angle: 0, axis: SIMD3(1, 0, 0))
            var pivot = matrix_identity_float4x4
            pivot.columns.3 = SIMD4(bone.head, 1)
            var unpivot = matrix_identity_float4x4
            unpivot.columns.3 = SIMD4(-bone.head, 1)
            let spin = pivot * float4x4(q) * unpivot
            worlds[i] = bone.parent >= 0 ? worlds[bone.parent] * spin : spin
        }
        boneWorlds = worlds
        skinningMatrices = worlds
    }


    // CPU linear-blend skinning of the bind-pose mesh. Reference for the GPU kernel.
    public func skinned(_ human: GeneratedHuman) -> (positions: [SIMD3<Float>], normals: [SIMD3<Float>]) {
        skin(positions: human.positions, normals: human.normals,
             joints: human.skinJoints, weights: human.skinWeights)
    }

    // Generic form — clothing shells and other body-derived meshes carry the same
    // skin data layout and deform through the identical path.
    public func skin(
        positions: [SIMD3<Float>], normals: [SIMD3<Float>],
        joints allJoints: [SIMD4<UInt16>], weights allWeights: [SIMD4<Float>]
    ) -> (positions: [SIMD3<Float>], normals: [SIMD3<Float>]) {
        var pos = [SIMD3<Float>](repeating: .zero, count: positions.count)
        var nrm = [SIMD3<Float>](repeating: .zero, count: positions.count)
        for i in 0..<positions.count {
            let joints = allJoints[i]
            let weights = allWeights[i]
            let p4 = SIMD4(positions[i], 1)
            let n4 = SIMD4(normals[i], 0)
            var p = SIMD4<Float>.zero
            var n = SIMD4<Float>.zero
            for k in 0..<4 where weights[k] > 0 {
                let m = skinningMatrices[Int(joints[k])]
                p += m * p4 * weights[k]
                n += m * n4 * weights[k]
            }
            pos[i] = SIMD3(p.x, p.y, p.z)
            let len = simd_length(SIMD3(n.x, n.y, n.z))
            nrm[i] = len > 1e-9 ? SIMD3(n.x, n.y, n.z) / len : SIMD3(0, 1, 0)
        }
        return (pos, nrm)
    }
}
