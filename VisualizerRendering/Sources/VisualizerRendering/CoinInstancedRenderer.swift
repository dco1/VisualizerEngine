import Foundation
import Metal
import SceneKit
import simd
import VisualizerCore

// ── CoinInstancedRenderer ─────────────────────────────────────────────────────
//
// All coins in one SCNGeometry, one draw call — the same architecture as
// EggInstancedRenderer. Per-coin transforms come from CoinDEMSolver
// (`transformBuffer`); the `coin_expand_instances` kernel transforms the shared
// unit-coin mesh by each coin's CoinTransform into the position/normal MTLBuffers
// SceneKit reads in place. Per-coin gold variation rides the vertex-colour stream
// (baked once at spawn). Inactive coins are parked off-screen by the solver's
// transform kernel, so `setActiveInstanceCount(maxCoins)` is safe.

@MainActor
public final class CoinInstancedRenderer {

    public let geometry: SCNGeometry
    public let vertsPerInstance: Int
    public let indicesPerInstance: Int
    public let maxInstances: Int

    private let positionBuffer: MTLBuffer
    private let normalBuffer: MTLBuffer
    private let uvBuffer: MTLBuffer
    private let colorBuffer: MTLBuffer
    private let indexBuffer: MTLBuffer
    private let unitPosBuffer: MTLBuffer
    private let unitNrmBuffer: MTLBuffer

    private let engine: SimEngine
    private let pipeline: MTLComputePipelineState
    private let threadgroupSize: Int

    private(set) public var activeInstanceCount: Int = 0

    /// Mixed-pile rendering: when `filterByType` is on, the expand kernel only draws
    /// bodies whose per-body type == `instanceType` (the rest are parked off-screen),
    /// so several renderers can share ONE solver's transform buffer, each drawing its
    /// own asset mesh. Off (default) draws every active instance.
    public var instanceType: UInt32 = 0
    public var filterByType: Bool = false

    public convenience init?(engine: SimEngine = .shared,
                             mesh: CoinMesh.Mesh,
                             maxInstances: Int) {
        guard let pipeline = engine.pipeline("coin_expand_instances") else { return nil }
        self.init(engine: engine, pipeline: pipeline, mesh: mesh, maxInstances: maxInstances)
    }

    /// Test seam: pipeline from a runtime-compiled library (the SwiftPM CLI ships
    /// no metallib, so the headless tests inject `coin_expand_instances` here).
    init?(engine: SimEngine, pipeline: MTLComputePipelineState,
          mesh: CoinMesh.Mesh, maxInstances: Int) {
        guard
            mesh.positions.count == mesh.normals.count,
            mesh.positions.count == mesh.uvs.count,
            mesh.positions.count > 0, mesh.indices.count > 0, maxInstances > 0
        else { return nil }

        self.engine = engine
        self.pipeline = pipeline
        self.threadgroupSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        self.vertsPerInstance = mesh.positions.count
        self.indicesPerInstance = mesh.indices.count
        self.maxInstances = maxInstances

        let device = engine.device
        let totalVerts = vertsPerInstance * maxInstances
        let totalIndices = indicesPerInstance * maxInstances

        // packed_float3 (stride 12) — matches the Illuminatorama GPU-mesh repack's
        // assumption (it ignores any stride field). SceneKit reads stride-12 too.
        let vec3Stride = 3 * MemoryLayout<Float>.stride   // 12
        let posSize = totalVerts * vec3Stride
        guard
            let pos = device.makeBuffer(length: posSize, options: .storageModeShared),
            let nrm = device.makeBuffer(length: posSize, options: .storageModeShared)
        else { return nil }
        pos.label = "CoinInstanced.positions"
        nrm.label = "CoinInstanced.normals"
        // Park unused slots far below until the kernel overwrites them.
        let posPtr = pos.contents().bindMemory(to: Float.self, capacity: totalVerts * 3)
        let nrmPtr = nrm.contents().bindMemory(to: Float.self, capacity: totalVerts * 3)
        for i in 0..<totalVerts {
            posPtr[i*3] = 0; posPtr[i*3+1] = -100_000; posPtr[i*3+2] = 0
            nrmPtr[i*3] = 0; nrmPtr[i*3+1] = 1; nrmPtr[i*3+2] = 0
        }
        self.positionBuffer = pos
        self.normalBuffer = nrm

        let uvSize = totalVerts * MemoryLayout<SIMD2<Float>>.stride
        guard let uv = device.makeBuffer(length: uvSize, options: .storageModeShared) else { return nil }
        uv.label = "CoinInstanced.uv"
        let uvPtr = uv.contents().bindMemory(to: SIMD2<Float>.self, capacity: totalVerts)
        for inst in 0..<maxInstances {
            let base = inst * vertsPerInstance
            for v in 0..<vertsPerInstance { uvPtr[base + v] = mesh.uvs[v] }
        }
        self.uvBuffer = uv

        let colorSize = totalVerts * MemoryLayout<SIMD4<Float>>.stride
        guard let col = device.makeBuffer(length: colorSize, options: .storageModeShared) else { return nil }
        col.label = "CoinInstanced.color"
        let colPtr = col.contents().bindMemory(to: SIMD4<Float>.self, capacity: totalVerts)
        for i in 0..<totalVerts { colPtr[i] = SIMD4<Float>(1, 1, 1, 1) }
        self.colorBuffer = col

        let indexSize = totalIndices * MemoryLayout<Int32>.stride
        guard let idxBuf = device.makeBuffer(length: indexSize, options: .storageModeShared) else { return nil }
        idxBuf.label = "CoinInstanced.index"
        let idxPtr = idxBuf.contents().bindMemory(to: Int32.self, capacity: totalIndices)
        for inst in 0..<maxInstances {
            let vertBase = Int32(inst * vertsPerInstance)
            let outBase = inst * indicesPerInstance
            for i in 0..<indicesPerInstance { idxPtr[outBase + i] = mesh.indices[i] + vertBase }
        }
        self.indexBuffer = idxBuf

        let unitSize = vertsPerInstance * MemoryLayout<SIMD4<Float>>.stride
        guard
            let unitPos = device.makeBuffer(length: unitSize, options: .storageModeShared),
            let unitNrm = device.makeBuffer(length: unitSize, options: .storageModeShared)
        else { return nil }
        unitPos.label = "CoinInstanced.unitPos"
        unitNrm.label = "CoinInstanced.unitNrm"
        let upPtr = unitPos.contents().bindMemory(to: SIMD4<Float>.self, capacity: vertsPerInstance)
        let unPtr = unitNrm.contents().bindMemory(to: SIMD4<Float>.self, capacity: vertsPerInstance)
        for v in 0..<vertsPerInstance {
            upPtr[v] = SIMD4<Float>(mesh.positions[v], 0)
            unPtr[v] = SIMD4<Float>(mesh.normals[v], 0)
        }
        self.unitPosBuffer = unitPos
        self.unitNrmBuffer = unitNrm

        // Audit the authored unit-coin winding once (logs the inside-out triangle
        // % on first run) — the per-instance geometry is GPU-buffer-backed, so we
        // check the CPU source arrays the kernel transforms, not the live buffers.
        _ = GeometryWinding.audit(positions: mesh.positions, normals: mesh.normals,
                                  indices: mesh.indices.map { UInt32(bitPattern: $0) },
                                  label: "CoinMesh")

        // SceneKit consumes the live MTLBuffers each frame (no copy). winding-ok:
        // GPU-expanded instanced mesh, audited above against the CPU source arrays.
        let posSource = SCNGeometrySource(
            buffer: pos, vertexFormat: .float3, semantic: .vertex,
            vertexCount: totalVerts, dataOffset: 0, dataStride: vec3Stride)
        let nrmSource = SCNGeometrySource(
            buffer: nrm, vertexFormat: .float3, semantic: .normal,
            vertexCount: totalVerts, dataOffset: 0, dataStride: vec3Stride)
        let uvData = Data(bytesNoCopy: uv.contents(), count: uvSize, deallocator: .none)
        let colData = Data(bytesNoCopy: col.contents(), count: colorSize, deallocator: .none)
        let uvSource = SCNGeometrySource(
            data: uvData, semantic: .texcoord, vectorCount: totalVerts,
            usesFloatComponents: true, componentsPerVector: 2,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: MemoryLayout<SIMD2<Float>>.stride)
        let colSource = SCNGeometrySource(
            data: colData, semantic: .color, vectorCount: totalVerts,
            usesFloatComponents: true, componentsPerVector: 4,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: MemoryLayout<SIMD4<Float>>.stride)
        let indexData = Data(bytesNoCopy: idxBuf.contents(), count: indexSize, deallocator: .none)
        let element = SCNGeometryElement(
            data: indexData, primitiveType: .triangles,
            primitiveCount: totalIndices / 3, bytesPerIndex: MemoryLayout<Int32>.size)

        self.geometry = SCNGeometry(sources: [posSource, nrmSource, uvSource, colSource],
                                    elements: [element])
        geometry.name = "CoinInstanced"
    }

    public func setActiveInstanceCount(_ n: Int) {
        activeInstanceCount = max(0, min(maxInstances, n))
    }

    /// Describe the expanded coin soup for native Illuminatorama: the same
    /// position/normal/uv/index buffers SceneKit would read, handed to
    /// `IlluminatoramaRenderer.registerGPUMesh` which repacks them into its vertex
    /// format every frame. One custom mesh = the whole pile; one InstanceRef draws
    /// it. Parked (inactive) coins sit off-screen, so drawing the full capacity is
    /// harmless.
    public func gpuMeshDescriptor() -> IlluminatoramaGPUMeshDescriptor {
        IlluminatoramaGPUMeshDescriptor(
            positionBuffer: positionBuffer,
            normalBuffer: normalBuffer,
            positionStride: 3 * MemoryLayout<Float>.stride,   // 12, packed_float3
            normalStride: 3 * MemoryLayout<Float>.stride,
            vertexCount: maxInstances * vertsPerInstance,
            bodyIndexBuffer: indexBuffer,
            bodyIndexCount: maxInstances * indicesPerInstance,
            bodyIndexType: .uint32,
            uvBuffer: uvBuffer,
            uvStride: MemoryLayout<SIMD2<Float>>.stride,
            colorBuffer: colorBuffer,
            colorStride: MemoryLayout<SIMD4<Float>>.stride)
    }

    /// Test/diagnostic: centroid of instance `i`'s expanded vertices, read from
    /// the live position buffer (call after the expand kernel's cb completes).
    func debugInstanceCentroid(_ i: Int) -> SIMD3<Float> {
        let ptr = positionBuffer.contents().bindMemory(to: Float.self,
                                                       capacity: maxInstances * vertsPerInstance * 3)
        var acc = SIMD3<Float>(0, 0, 0)
        let base = i * vertsPerInstance
        for v in 0..<vertsPerInstance {
            let o = (base + v) * 3
            acc += SIMD3(ptr[o], ptr[o+1], ptr[o+2])
        }
        return acc / Float(vertsPerInstance)
    }

    /// Bake a coin's tint (gold variation) into all its vertices. Call once at spawn.
    public func setInstanceColor(_ instance: Int, _ color: SIMD4<Float>) {
        guard instance >= 0, instance < maxInstances else { return }
        let base = instance * vertsPerInstance
        let ptr = colorBuffer.contents().bindMemory(to: SIMD4<Float>.self,
                                                    capacity: maxInstances * vertsPerInstance)
        for v in 0..<vertsPerInstance { ptr[base + v] = color }
    }

    /// Run the expand kernel: unit mesh × per-body transform → shared mesh buffers.
    /// Chain onto the same command buffer the solver is using. `bodyType` is the
    /// solver's per-body type buffer; with `filterByType` on, only matching bodies
    /// are drawn (mixed pile). Pass the solver's `bodyTypeBuffer`.
    public func dispatchExpand(transforms: MTLBuffer, bodyType: MTLBuffer,
                               in commandBuffer: MTLCommandBuffer) {
        guard activeInstanceCount > 0, let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "CoinInstancedRenderer.expand"
        enc.setComputePipelineState(pipeline)

        struct Uniforms { var coinCount: UInt32; var vertsPerInstance: UInt32; var targetType: UInt32; var filterEnabled: UInt32 }
        var u = Uniforms(coinCount: UInt32(activeInstanceCount),
                         vertsPerInstance: UInt32(vertsPerInstance),
                         targetType: instanceType,
                         filterEnabled: filterByType ? 1 : 0)
        enc.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.setBuffer(transforms, offset: 0, index: 1)
        enc.setBuffer(unitPosBuffer, offset: 0, index: 2)
        enc.setBuffer(unitNrmBuffer, offset: 0, index: 3)
        enc.setBuffer(positionBuffer, offset: 0, index: 4)
        enc.setBuffer(normalBuffer, offset: 0, index: 5)
        enc.setBuffer(bodyType, offset: 0, index: 6)

        let totalThreads = activeInstanceCount * vertsPerInstance
        enc.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        enc.endEncoding()
    }
}
