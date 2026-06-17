import Foundation
import Metal
import SceneKit
import simd

// ── EggInstancedRenderer ─────────────────────────────────────────────────────
//
// All eggs in one SCNGeometry, one draw call. Per-egg state (transform,
// colour) lives in GPU buffers; a compute kernel each tick takes the
// transformed-per-instance vertices and writes them into the shared mesh's
// position + normal buffers. SceneKit walks the unified mesh exactly once.
//
// This is the "kill the 500 SCNNode draw cost" architectural shift. The
// trade-off vs the per-node path is per-egg material variety: with one
// shared SCNMaterial, every egg samples the same diffuse texture. Per-egg
// colour is preserved via the vertex-colour stream (written once per egg
// at populate-time, not per tick).
//
// Public API:
//   • `init(engine, vertsPerInstance, indicesPerInstance, maxInstances)`
//     allocates the buffers + compiles the kernel pipeline.
//   • `setUnitMesh(...)` once per scene load — installs the unit-egg
//     position/normal/UV sources + per-instance index template.
//   • `geometry` — assign to one SCNNode in the scene graph.
//   • `setActiveInstanceCount(_:)` — the kernel only updates this many
//     instances; remainder render at whatever was there last (initialise
//     to off-screen so unused slots are invisible).
//   • `setInstanceColor(_:_:)` — one-shot per egg at population time.
//   • `dispatchExpand(transforms:in:)` — chains the expand kernel onto a
//     command buffer the caller is already submitting.

@MainActor
public final class EggInstancedRenderer {

    public let geometry: SCNGeometry
    public let vertsPerInstance: Int
    public let indicesPerInstance: Int
    public let maxInstances: Int

    // GPU-resident, shared with SceneKit.
    private let positionBuffer: MTLBuffer
    private let normalBuffer: MTLBuffer
    private let uvBuffer: MTLBuffer
    private let colorBuffer: MTLBuffer
    private let indexBuffer: MTLBuffer

    // Unit-egg sources (xyz packed in float4 for kernel friendliness).
    private let unitPosBuffer: MTLBuffer
    private let unitNrmBuffer: MTLBuffer

    private let engine: SimEngine
    private let pipeline: MTLComputePipelineState
    private let threadgroupSize: Int

    private(set) public var activeInstanceCount: Int = 0

    public init?(
        engine: SimEngine = .shared,
        unitPositions: [SIMD3<Float>],
        unitNormals: [SIMD3<Float>],
        unitUVs: [SIMD2<Float>],
        unitIndices: [Int32],
        maxInstances: Int
    ) {
        guard
            unitPositions.count == unitNormals.count,
            unitPositions.count == unitUVs.count,
            unitPositions.count > 0,
            unitIndices.count > 0,
            maxInstances > 0
        else { return nil }
        guard let pipeline = engine.pipeline("eggs_expand_instances") else {
            return nil
        }

        self.engine = engine
        self.pipeline = pipeline
        self.threadgroupSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        self.vertsPerInstance = unitPositions.count
        self.indicesPerInstance = unitIndices.count
        self.maxInstances = maxInstances

        let device = engine.device
        let totalVerts = vertsPerInstance * maxInstances
        let totalIndices = indicesPerInstance * maxInstances

        // ── Output position / normal — float4 stride 16 ───────────────
        // SceneKit reads SCNGeometrySource by data stride; we use stride
        // 16 with `componentsPerVector = 3` so SCN only consumes xyz.
        let posSize = totalVerts * MemoryLayout<SIMD4<Float>>.stride
        let nrmSize = posSize
        guard
            let pos = device.makeBuffer(length: posSize, options: .storageModeShared),
            let nrm = device.makeBuffer(length: nrmSize, options: .storageModeShared)
        else { return nil }
        pos.label = "EggInstanced.positions"
        nrm.label = "EggInstanced.normals"
        // Initialise positions far below the floor so unused slots are
        // invisible until the kernel overwrites them on first activation.
        let initVal = SIMD4<Float>(0, -10_000, 0, 0)
        let posPtr = pos.contents().bindMemory(to: SIMD4<Float>.self, capacity: totalVerts)
        let nrmPtr = nrm.contents().bindMemory(to: SIMD4<Float>.self, capacity: totalVerts)
        for i in 0..<totalVerts {
            posPtr[i] = initVal
            nrmPtr[i] = SIMD4<Float>(0, 1, 0, 0)
        }
        self.positionBuffer = pos
        self.normalBuffer = nrm

        // ── UVs — static across instances; just unit-egg UVs replicated ──
        let uvSize = totalVerts * MemoryLayout<SIMD2<Float>>.stride
        guard let uv = device.makeBuffer(length: uvSize, options: .storageModeShared)
        else { return nil }
        uv.label = "EggInstanced.uv"
        let uvPtr = uv.contents().bindMemory(to: SIMD2<Float>.self, capacity: totalVerts)
        for inst in 0..<maxInstances {
            let base = inst * vertsPerInstance
            for v in 0..<vertsPerInstance {
                uvPtr[base + v] = unitUVs[v]
            }
        }
        self.uvBuffer = uv

        // ── Per-vertex colour — written by `setInstanceColor`. ────────
        let colorSize = totalVerts * MemoryLayout<SIMD4<Float>>.stride
        guard let col = device.makeBuffer(length: colorSize, options: .storageModeShared)
        else { return nil }
        col.label = "EggInstanced.color"
        // Default to white so an un-coloured instance still renders.
        let colPtr = col.contents().bindMemory(to: SIMD4<Float>.self, capacity: totalVerts)
        for i in 0..<totalVerts {
            colPtr[i] = SIMD4<Float>(1, 1, 1, 1)
        }
        self.colorBuffer = col

        // ── Indices — concatenated per-instance copies, each offset by ──
        // the instance's vertex-base. Static; built once.
        let indexSize = totalIndices * MemoryLayout<Int32>.stride
        guard let idx = device.makeBuffer(length: indexSize, options: .storageModeShared)
        else { return nil }
        idx.label = "EggInstanced.index"
        let idxPtr = idx.contents().bindMemory(to: Int32.self, capacity: totalIndices)
        for inst in 0..<maxInstances {
            let vertBase = Int32(inst * vertsPerInstance)
            let outBase = inst * indicesPerInstance
            for i in 0..<indicesPerInstance {
                idxPtr[outBase + i] = unitIndices[i] + vertBase
            }
        }
        self.indexBuffer = idx

        // ── Unit-egg sources for the kernel — float4 stride 16 ──────────
        let unitSize = vertsPerInstance * MemoryLayout<SIMD4<Float>>.stride
        guard
            let unitPos = device.makeBuffer(length: unitSize, options: .storageModeShared),
            let unitNrm = device.makeBuffer(length: unitSize, options: .storageModeShared)
        else { return nil }
        unitPos.label = "EggInstanced.unitPos"
        unitNrm.label = "EggInstanced.unitNrm"
        let upPtr = unitPos.contents().bindMemory(to: SIMD4<Float>.self, capacity: vertsPerInstance)
        let unPtr = unitNrm.contents().bindMemory(to: SIMD4<Float>.self, capacity: vertsPerInstance)
        for v in 0..<vertsPerInstance {
            upPtr[v] = SIMD4<Float>(unitPositions[v], 0)
            unPtr[v] = SIMD4<Float>(unitNormals[v], 0)
        }
        self.unitPosBuffer = unitPos
        self.unitNrmBuffer = unitNrm

        // ── SCNGeometry from the live MTLBuffers ────────────────────────
        // SceneKit consumes the MTLBuffer in place each frame (no copy).
        let posSource = SCNGeometrySource(
            buffer: pos,
            vertexFormat: .float3,
            semantic: .vertex,
            vertexCount: totalVerts,
            dataOffset: 0,
            dataStride: MemoryLayout<SIMD4<Float>>.stride
        )
        let nrmSource = SCNGeometrySource(
            buffer: nrm,
            vertexFormat: .float3,
            semantic: .normal,
            vertexCount: totalVerts,
            dataOffset: 0,
            dataStride: MemoryLayout<SIMD4<Float>>.stride
        )
        // UV + colour come from CPU-side Data (not buffer-backed) — SceneKit
        // re-uploads them when they're set, but we set them once.
        let uvData = Data(
            bytesNoCopy: uv.contents(),
            count: uvSize,
            deallocator: .none
        )
        let colData = Data(
            bytesNoCopy: col.contents(),
            count: colorSize,
            deallocator: .none
        )
        let uvSource = SCNGeometrySource(
            data: uvData,
            semantic: .texcoord,
            vectorCount: totalVerts,
            usesFloatComponents: true,
            componentsPerVector: 2,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<SIMD2<Float>>.stride
        )
        let colSource = SCNGeometrySource(
            data: colData,
            semantic: .color,
            vectorCount: totalVerts,
            usesFloatComponents: true,
            componentsPerVector: 4,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<SIMD4<Float>>.stride
        )

        let indexData = Data(
            bytesNoCopy: idx.contents(),
            count: indexSize,
            deallocator: .none
        )
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: totalIndices / 3,
            bytesPerIndex: MemoryLayout<Int32>.size
        )

        self.geometry = SCNGeometry(
            sources: [posSource, nrmSource, uvSource, colSource],
            elements: [element]
        )
        geometry.name = "EggInstanced"
    }

    /// Set the active instance count (kernel dispatches this many, rest stay
    /// at their off-screen init values). Clamped to `maxInstances`.
    public func setActiveInstanceCount(_ n: Int) {
        activeInstanceCount = max(0, min(maxInstances, n))
    }

    /// Bake an egg's diffuse colour into all of its vertices. Call once at
    /// population time; colour doesn't change per tick. Index is the
    /// instance slot in `[0, maxInstances)`.
    public func setInstanceColor(_ instance: Int, _ color: SIMD4<Float>) {
        guard instance >= 0, instance < maxInstances else { return }
        let base = instance * vertsPerInstance
        let ptr = colorBuffer.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: maxInstances * vertsPerInstance
        )
        for v in 0..<vertsPerInstance {
            ptr[base + v] = color
        }
    }

    /// Run the kernel to expand `transforms` × unit-egg into the per-vertex
    /// position + normal buffers SceneKit reads. Chain onto an existing
    /// command buffer (typically the same one running the motion kernel).
    public func dispatchExpand(
        transforms: MTLBuffer,
        in commandBuffer: MTLCommandBuffer
    ) {
        guard activeInstanceCount > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "EggInstancedRenderer.expand"
        enc.setComputePipelineState(pipeline)

        struct Uniforms {
            var eggCount: UInt32
            var vertsPerInstance: UInt32
            var _pad0: Float
            var _pad1: Float
        }
        var u = Uniforms(
            eggCount: UInt32(activeInstanceCount),
            vertsPerInstance: UInt32(vertsPerInstance),
            _pad0: 0, _pad1: 0
        )
        enc.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.setBuffer(transforms, offset: 0, index: 1)
        enc.setBuffer(unitPosBuffer, offset: 0, index: 2)
        enc.setBuffer(unitNrmBuffer, offset: 0, index: 3)
        enc.setBuffer(positionBuffer, offset: 0, index: 4)
        enc.setBuffer(normalBuffer, offset: 0, index: 5)

        let totalThreads = activeInstanceCount * vertsPerInstance
        let grid = MTLSize(width: totalThreads, height: 1, depth: 1)
        let group = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: group)
        enc.endEncoding()
    }
}
