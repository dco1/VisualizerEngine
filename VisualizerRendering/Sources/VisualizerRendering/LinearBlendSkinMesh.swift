import Foundation
import Metal
import OSLog
import VisualizerCore
import simd

// ── LINEAR BLEND SKIN MESH ───────────────────────────────────────────────────
//
// A static-topology mesh whose vertices follow a small bone palette on the GPU
// (`lbs_skin` in LinearBlendSkin.metal). The rest pose, bone indices and weights
// are uploaded ONCE; each frame the host encodes one dispatch with the current
// bone matrices, and the kernel writes the packed position/normal streams that
// `IlluminatoramaRenderer.registerGPUMesh(descriptor)` repacks and refits — the
// skinned surface is never read back or rebuilt on the CPU.
//
// Use it for anything that bends or stretches as a whole rather than
// simulating: a jaw opening a lip (two bones, skull + mandible), a rubbery
// tooth stretching and bending (a short bone chain), a hinged flap.
//
// USAGE
//
//   let skin = LinearBlendSkinMesh(engine: .shared, positions: p, normals: n,
//                                  uvs: uv, colors: c, indices: idx,
//                                  boneIndices: bi, boneWeights: bw, boneCount: 2)
//   let handle = renderer.registerGPUMesh(skin.descriptor)
//   // per frame, on the renderer's queue ahead of render():
//   skin.encode(to: cb, bones: [identity, mandible])
//
// Bone matrices are `current · rest⁻¹` (they map rest-pose object space to the
// current pose). Bones must not mirror. Weights should sum to 1.

@MainActor
public final class LinearBlendSkinMesh {
    private static let log = Logger(subsystem: AppLog.subsystem, category: "LinearBlendSkinMesh")

    public let vertexCount: Int
    public let boneCount: Int
    public let descriptor: IlluminatoramaGPUMeshDescriptor

    private let pipeline: MTLComputePipelineState
    private let restPos: MTLBuffer
    private let restNrm: MTLBuffer
    private let boneIndex: MTLBuffer
    private let boneWeight: MTLBuffer
    private let outPos: MTLBuffer
    private let outNrm: MTLBuffer
    /// Palettes above the 4 KB `setBytes` limit go through a small ring so a
    /// frame still in flight never sees next frame's matrices.
    private var boneRing: [MTLBuffer] = []
    private var ringSlot = 0
    private static let setBytesLimit = 4096
    private static let ringDepth = 3

    public init?(engine: SimEngine,
                 positions: [SIMD3<Float>],
                 normals: [SIMD3<Float>],
                 uvs: [SIMD2<Float>]? = nil,
                 colors: [SIMD4<Float>]? = nil,
                 indices: [UInt32],
                 boneIndices: [SIMD4<UInt16>],
                 boneWeights: [SIMD4<Float>],
                 boneCount: Int,
                 doubleSided: Bool = false,
                 label: String = "LinearBlendSkinMesh") {
        let n = positions.count
        guard n > 0, normals.count == n, boneIndices.count == n, boneWeights.count == n,
              !indices.isEmpty, boneCount > 0,
              let pso = engine.pipeline("lbs_skin") else {
            Self.log.error("\(label): invalid input or missing lbs_skin pipeline")
            return nil
        }
        let device = engine.device
        func packed(_ v: [SIMD3<Float>]) -> [Float] {
            var out = [Float](); out.reserveCapacity(v.count * 3)
            for p in v { out.append(p.x); out.append(p.y); out.append(p.z) }
            return out
        }
        let pp = packed(positions), nn = packed(normals)
        guard let rp = device.makeBuffer(bytes: pp, length: pp.count * 4, options: .storageModeShared),
              let rn = device.makeBuffer(bytes: nn, length: nn.count * 4, options: .storageModeShared),
              let bi = device.makeBuffer(bytes: boneIndices, length: n * MemoryLayout<SIMD4<UInt16>>.stride,
                                         options: .storageModeShared),
              let bw = device.makeBuffer(bytes: boneWeights, length: n * MemoryLayout<SIMD4<Float>>.stride,
                                         options: .storageModeShared),
              // Seeded with the rest pose so a frame drawn before the first dispatch is sane.
              let op = device.makeBuffer(bytes: pp, length: pp.count * 4, options: .storageModeShared),
              let on = device.makeBuffer(bytes: nn, length: nn.count * 4, options: .storageModeShared),
              let ib = device.makeBuffer(bytes: indices, length: indices.count * 4, options: .storageModeShared)
        else { return nil }
        rp.label = "\(label).restPos"; op.label = "\(label).pos"; on.label = "\(label).nrm"
        var uvBuf: MTLBuffer? = nil
        if let uvs, uvs.count == n {
            var flat = [Float](); flat.reserveCapacity(n * 2)
            for u in uvs { flat.append(u.x); flat.append(u.y) }
            uvBuf = device.makeBuffer(bytes: flat, length: flat.count * 4, options: .storageModeShared)
        }
        var colorBuf: MTLBuffer? = nil
        if let colors, colors.count == n {
            colorBuf = device.makeBuffer(bytes: colors, length: n * MemoryLayout<SIMD4<Float>>.stride,
                                         options: .storageModeShared)
        }
        let paletteBytes = boneCount * MemoryLayout<simd_float4x4>.stride
        if paletteBytes > Self.setBytesLimit {
            for _ in 0 ..< Self.ringDepth {
                guard let b = device.makeBuffer(length: paletteBytes, options: .storageModeShared) else { return nil }
                boneRing.append(b)
            }
        }
        self.pipeline = pso
        self.vertexCount = n
        self.boneCount = boneCount
        self.restPos = rp; self.restNrm = rn
        self.boneIndex = bi; self.boneWeight = bw
        self.outPos = op; self.outNrm = on
        self.descriptor = IlluminatoramaGPUMeshDescriptor(
            positionBuffer: op, normalBuffer: on, vertexCount: n,
            bodyIndexBuffer: ib, bodyIndexCount: indices.count, bodyIndexType: .uint32,
            uvBuffer: uvBuf, colorBuffer: colorBuf,
            doubleSided: doubleSided, shadowCastsBothFaces: doubleSided)
    }

    /// Encode one skinning pass with `bones` (count must equal `boneCount`) in its own encoder.
    public func encode(to cb: MTLCommandBuffer, bones: [simd_float4x4]) {
        guard bones.count == boneCount, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "LinearBlendSkinMesh.skin"
        encode(into: enc, bones: bones)
        enc.endEncoding()
    }

    /// Encode one skinning dispatch into an OPEN compute encoder — lets a host skin many meshes
    /// (a jaw's lips + 32 teeth) under one encoder. The caller ends the encoder.
    ///
    /// Palettes over 4 KB rotate through a 3-deep buffer ring, so encode each mesh at most once
    /// per frame with ≤ 2 frames in flight (the renderer's limit), or a frame still on the GPU
    /// would see the next frame's matrices.
    public func encode(into enc: MTLComputeCommandEncoder, bones: [simd_float4x4], bindPipeline: Bool = true) {
        guard bones.count == boneCount else { return }
        // A host skinning many meshes back to back binds the (shared) pipeline once and passes
        // `bindPipeline: false` after — the validation layer rejects redundant re-binds.
        if bindPipeline { enc.setComputePipelineState(pipeline) }
        enc.setBuffer(restPos, offset: 0, index: 0)
        enc.setBuffer(restNrm, offset: 0, index: 1)
        enc.setBuffer(boneIndex, offset: 0, index: 2)
        enc.setBuffer(boneWeight, offset: 0, index: 3)
        let bytes = bones.count * MemoryLayout<simd_float4x4>.stride
        if boneRing.isEmpty {
            bones.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: bytes, index: 4) }
        } else {
            let b = boneRing[ringSlot]
            ringSlot = (ringSlot + 1) % boneRing.count
            bones.withUnsafeBytes { b.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes) }
            enc.setBuffer(b, offset: 0, index: 4)
        }
        var count = UInt32(vertexCount)
        enc.setBytes(&count, length: 4, index: 5)
        enc.setBuffer(outPos, offset: 0, index: 6)
        enc.setBuffer(outNrm, offset: 0, index: 7)
        let w = min(pipeline.maxTotalThreadsPerThreadgroup, 128)
        enc.dispatchThreads(MTLSize(width: vertexCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }
}
