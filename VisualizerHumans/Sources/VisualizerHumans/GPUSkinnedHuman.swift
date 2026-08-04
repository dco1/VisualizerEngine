import Metal
import simd
import VisualizerRendering

// GPU-resident skinned mesh: bind data uploads once, per-frame work is one
// matrix-palette write + one compute dispatch into position/normal buffers that
// plug straight into IlluminatoramaGPUMeshDescriptor / registerGPUMesh (which
// brings BLAS refit + TAA prev-vertex motion vectors for free). Works for the
// body and for clothing/hair shells alike — anything carrying the human's
// skin-weight layout.
@MainActor
public final class GPUSkinnedHuman {

    public enum SkinError: Error { case bufferAllocation, libraryCompile, pipeline }

    public let vertexCount: Int
    public let boneCount: Int
    public let positionBuffer: MTLBuffer     // packed_float3 output
    public let normalBuffer: MTLBuffer
    public let indexBuffer: MTLBuffer
    public let indexCount: Int
    public let uvBuffer: MTLBuffer?
    public let colorBuffer: MTLBuffer?
    public let doubleSided: Bool

    private let bindPositions: MTLBuffer
    private let bindNormals: MTLBuffer
    private let jointsBuffer: MTLBuffer
    private let weightsBuffer: MTLBuffer
    private let matricesBuffer: MTLBuffer
    private let pipeline: MTLComputePipelineState

    // One runtime compile per device; kept tiny on purpose (see .metalsource).
    private static var libraries: [ObjectIdentifier: MTLLibrary] = [:]

    private static func pipeline(device: MTLDevice) throws -> MTLComputePipelineState {
        let key = ObjectIdentifier(device)
        let library: MTLLibrary
        if let cached = libraries[key] {
            library = cached
        } else {
            guard let url = Bundle.module.url(forResource: "HumanSkinning", withExtension: "metalsource") else {
                throw SkinError.libraryCompile
            }
            // Through `MetalSourceLoader` like every other runtime source-string compile: a
            // source string carries no include path, so a local `#include "…"` would fail with
            // "file not found" the day this shader grows one. Today it has none — which is
            // exactly when the loader is free to adopt.
            guard let lib = try? MetalSourceLoader.makeLibrary(device: device, contentsOf: url) else {
                throw SkinError.libraryCompile
            }
            library = lib
            libraries[key] = library
        }
        guard let fn = library.makeFunction(name: "human_skin_lbs") else { throw SkinError.pipeline }
        return try device.makeComputePipelineState(function: fn)
    }

    public convenience init(device: MTLDevice, human: GeneratedHuman) throws {
        try self.init(
            device: device,
            positions: human.positions, normals: human.normals, uvs: human.uvs,
            colors: SkinPainter.paint(human: human),
            joints: human.skinJoints, weights: human.skinWeights,
            triangles: human.triangles, boneCount: human.bones.count, doubleSided: false)
    }

    public convenience init(device: MTLDevice, garment: GarmentMesh, boneCount: Int) throws {
        try self.init(
            device: device,
            positions: garment.positions, normals: garment.normals, uvs: garment.uvs,
            colors: nil,
            joints: garment.skinJoints, weights: garment.skinWeights,
            triangles: garment.triangles, boneCount: boneCount, doubleSided: true)
    }

    public init(
        device: MTLDevice,
        positions: [SIMD3<Float>], normals: [SIMD3<Float>], uvs: [SIMD2<Float>],
        colors: [SIMD4<Float>]?,
        joints: [SIMD4<UInt16>], weights: [SIMD4<Float>],
        triangles: [UInt32], boneCount: Int, doubleSided: Bool
    ) throws {
        vertexCount = positions.count
        self.boneCount = boneCount
        self.doubleSided = doubleSided
        indexCount = triangles.count
        pipeline = try Self.pipeline(device: device)

        func upload<T>(_ array: [T]) throws -> MTLBuffer {
            guard let buffer = array.withUnsafeBytes({ raw in
                device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
            }) else { throw SkinError.bufferAllocation }
            return buffer
        }

        bindPositions = try upload(positions)
        bindNormals = try upload(normals)
        jointsBuffer = try upload(joints)
        weightsBuffer = try upload(weights)
        indexBuffer = try upload(triangles)
        uvBuffer = try upload(uvs)
        colorBuffer = try colors.map { try upload($0) }

        guard let mats = device.makeBuffer(length: MemoryLayout<float4x4>.stride * boneCount,
                                           options: .storageModeShared),
              let outP = device.makeBuffer(length: 12 * vertexCount, options: .storageModeShared),
              let outN = device.makeBuffer(length: 12 * vertexCount, options: .storageModeShared) else {
            throw SkinError.bufferAllocation
        }
        matricesBuffer = mats
        positionBuffer = outP
        normalBuffer = outN
    }

    // Per frame: write the matrix palette and encode the dispatch. The caller
    // owns the command buffer (batch all characters into one).
    public func encode(pose: PosedHuman, into commandBuffer: MTLCommandBuffer) {
        precondition(pose.skinningMatrices.count == boneCount)
        pose.skinningMatrices.withUnsafeBytes { raw in
            matricesBuffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "human_skin_lbs"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bindPositions, offset: 0, index: 0)
        encoder.setBuffer(bindNormals, offset: 0, index: 1)
        encoder.setBuffer(jointsBuffer, offset: 0, index: 2)
        encoder.setBuffer(weightsBuffer, offset: 0, index: 3)
        encoder.setBuffer(matricesBuffer, offset: 0, index: 4)
        encoder.setBuffer(positionBuffer, offset: 0, index: 5)
        encoder.setBuffer(normalBuffer, offset: 0, index: 6)
        var count = UInt32(vertexCount)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 7)
        let w = pipeline.threadExecutionWidth
        encoder.dispatchThreadgroups(
            MTLSize(width: (vertexCount + w - 1) / w, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        encoder.endEncoding()
    }

    public func gpuMeshDescriptor() -> IlluminatoramaGPUMeshDescriptor {
        IlluminatoramaGPUMeshDescriptor(
            positionBuffer: positionBuffer,
            normalBuffer: normalBuffer,
            vertexCount: vertexCount,
            bodyIndexBuffer: indexBuffer,
            bodyIndexCount: indexCount,
            bodyIndexType: .uint32,
            uvBuffer: uvBuffer,
            colorBuffer: colorBuffer,
            doubleSided: doubleSided)
    }
}
