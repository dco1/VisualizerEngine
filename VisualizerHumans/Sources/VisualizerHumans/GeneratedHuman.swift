import Foundation
import simd

// The bake: HumanSpec + HumanDataset → render-ready mesh, bind-pose skeleton,
// and per-vertex skinning data. Pure CPU, deterministic, no Metal — renderable
// by anything and testable headlessly.
public struct GeneratedHuman: Sendable {

    public struct BindBone: Sendable {
        public let name: String
        public let parent: Int              // -1 = root
        public let head: SIMD3<Float>       // bind-pose world positions (meters, Y-up)
        public let tail: SIMD3<Float>
        public let restWorld: float4x4      // bind frame: Y along the bone, origin at head
        public let inverseBind: float4x4
    }

    public let spec: HumanSpec
    public let positions: [SIMD3<Float>]    // per render vertex, bind pose
    public let normals: [SIMD3<Float>]
    public let uvs: [SIMD2<Float>]
    public let triangles: [UInt32]
    public let skinJoints: [SIMD4<UInt16>]  // per render vertex, up to 4 bone influences
    public let skinWeights: [SIMD4<Float>]
    public let bones: [BindBone]
    public let boneIndex: [String: Int]
    public let heightMeters: Float

    public init(spec: HumanSpec, dataset: HumanDataset = .shared) {
        self.spec = spec

        // 1. Morph the orig vertex table.
        var orig = dataset.origPositions
        for (morphIdx, weight) in MacroBlend.weights(for: spec, dataset: dataset) {
            let morph = dataset.morphs[morphIdx]
            for i in 0..<morph.indices.count {
                orig[Int(morph.indices[i])] += morph.deltas[i] * weight
            }
        }

        // 2. Ground the figure: feet on y=0, centered on x/z. Everything downstream
        // (render verts, joints, bones) shares the shifted orig table.
        var minP = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxP = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for oi in dataset.renderOrigIndex {
            let p = orig[Int(oi)]
            minP = simd_min(minP, p)
            maxP = simd_max(maxP, p)
        }
        let offset = SIMD3<Float>(-(minP.x + maxP.x) / 2, -minP.y, -(minP.z + maxP.z) / 2)
        for i in orig.indices { orig[i] += offset }

        // 3. Render vertex arrays. Normals accumulate per ORIG vertex so UV-seam
        // duplicates stay smooth-shaded.
        let renderCount = dataset.renderOrigIndex.count
        var pos: [SIMD3<Float>] = []
        pos.reserveCapacity(renderCount)
        for oi in dataset.renderOrigIndex { pos.append(orig[Int(oi)]) }

        var origNormals = [SIMD3<Float>](repeating: .zero, count: dataset.origPositions.count)
        let tris = dataset.triangles
        for t in stride(from: 0, to: tris.count, by: 3) {
            let (a, b, c) = (Int(tris[t]), Int(tris[t + 1]), Int(tris[t + 2]))
            let n = simd_cross(pos[b] - pos[a], pos[c] - pos[a])   // area-weighted
            origNormals[Int(dataset.renderOrigIndex[a])] += n
            origNormals[Int(dataset.renderOrigIndex[b])] += n
            origNormals[Int(dataset.renderOrigIndex[c])] += n
        }
        var nrm: [SIMD3<Float>] = []
        nrm.reserveCapacity(renderCount)
        for oi in dataset.renderOrigIndex {
            let n = origNormals[Int(oi)]
            let len = simd_length(n)
            nrm.append(len > 1e-9 ? n / len : SIMD3(0, 1, 0))
        }

        // 3. Joints track the morphed body: each is the centroid of its vertex set.
        var jointPos: [SIMD3<Float>] = []
        jointPos.reserveCapacity(dataset.joints.count)
        for joint in dataset.joints {
            var c = SIMD3<Float>.zero
            for v in joint.vertices { c += orig[Int(v)] }
            jointPos.append(c / Float(max(1, joint.vertices.count)))
        }

        // 4. Bind skeleton with ANATOMICAL frames: Y along the bone, X = world X
        // projected perpendicular to Y (the sagittal hinge — same world sense for
        // every limb at bind), Z = X×Y. Under local-frame FK the hinge then
        // follows the posed parent, so an adducted arm still flexes its elbow
        // forward. (MakeHuman's roll planes gave per-bone twist frames whose X
        // wasn't a hinge — that produced the splayed-leg / crooked-elbow poses.)
        var bindBones: [BindBone] = []
        bindBones.reserveCapacity(dataset.bones.count)
        for bone in dataset.bones {
            let head = jointPos[bone.headJoint]
            let tail = jointPos[bone.tailJoint]
            let y = simd_normalize(tail - head)

            var x = SIMD3<Float>(1, 0, 0) - y * y.x     // project world X ⊥ y
            if simd_length(x) < 0.15 {
                // Bone runs nearly along world X (clavicle/shoulder): any stable
                // perpendicular works; these bones get little direct posing.
                x = simd_cross(y, SIMD3(0, 1, 0))
                if simd_length(x) < 1e-6 { x = simd_cross(y, SIMD3(0, 0, 1)) }
            }
            x = simd_normalize(x)
            let z = simd_normalize(simd_cross(x, y))

            let world = float4x4(
                SIMD4(x, 0), SIMD4(y, 0), SIMD4(z, 0), SIMD4(head, 1))
            bindBones.append(BindBone(
                name: bone.name, parent: bone.parent, head: head, tail: tail,
                restWorld: world, inverseBind: world.inverse))
        }
        // Topologically sort bones (parents strictly before children). The dataset
        // stores bones ALPHABETICALLY — "lowerarm" precedes "upperarm" — and
        // PosedHuman composes worlds in array order, so without this a child
        // reads a stale identity parent world and poses fall apart silently.
        var oldToNew = [Int](repeating: -1, count: bindBones.count)
        var ordered: [BindBone] = []
        ordered.reserveCapacity(bindBones.count)
        while ordered.count < bindBones.count {
            var progressed = false
            for (oldIdx, bone) in bindBones.enumerated() where oldToNew[oldIdx] == -1 {
                if bone.parent < 0 || oldToNew[bone.parent] != -1 {
                    oldToNew[oldIdx] = ordered.count
                    ordered.append(BindBone(
                        name: bone.name,
                        parent: bone.parent < 0 ? -1 : oldToNew[bone.parent],
                        head: bone.head, tail: bone.tail,
                        restWorld: bone.restWorld, inverseBind: bone.inverseBind))
                    progressed = true
                }
            }
            precondition(progressed, "bone parent cycle in dataset")
        }
        bones = ordered
        boneIndex = Dictionary(uniqueKeysWithValues: ordered.enumerated().map { ($0.element.name, $0.offset) })

        // 5. Per-orig-vertex influence gather → top-4, normalized, mapped to render verts.
        // Influence indices use the topo order via oldToNew.
        var influences = [[(UInt16, Float)]](repeating: [], count: dataset.origPositions.count)
        for (bi, bone) in dataset.bones.enumerated() {
            for k in 0..<bone.weightIndices.count {
                influences[Int(bone.weightIndices[k])].append((UInt16(oldToNew[bi]), bone.weightValues[k]))
            }
        }
        var joints4: [SIMD4<UInt16>] = []
        var weights4: [SIMD4<Float>] = []
        joints4.reserveCapacity(renderCount)
        weights4.reserveCapacity(renderCount)
        for oi in dataset.renderOrigIndex {
            var list = influences[Int(oi)]
            list.sort { $0.1 > $1.1 }
            var j = SIMD4<UInt16>(repeating: 0)
            var w = SIMD4<Float>(repeating: 0)
            for k in 0..<min(4, list.count) {
                j[k] = list[k].0
                w[k] = list[k].1
            }
            let total = w.sum()
            if total > 1e-6 { w /= total } else { w = SIMD4(1, 0, 0, 0) }   // unweighted vert rides bone 0
            joints4.append(j)
            weights4.append(w)
        }
        skinJoints = joints4
        skinWeights = weights4

        positions = pos
        normals = nrm
        uvs = dataset.renderUV
        triangles = dataset.triangles

        var minY = Float.greatestFiniteMagnitude
        var maxY = -Float.greatestFiniteMagnitude
        for p in pos {
            minY = min(minY, p.y)
            maxY = max(maxY, p.y)
        }
        heightMeters = maxY - minY
    }
}
