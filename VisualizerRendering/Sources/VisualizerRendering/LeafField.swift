import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

// ── LEAF FIELD ───────────────────────────────────────────────────────────────
//
// GPU foliage system. One LeafField hosts N independent "species channels",
// each backed by:
//
//   • a SimBuffer<LeafInstance>     — one record per leaf (uploaded once)
//   • a position SimBuffer          — N×4 packed_float3, written each frame
//   • a normal   SimBuffer          — N×4 packed_float3, written each frame
//   • a static UV   MTLBuffer       — N×4 packed_float2
//   • a static index MTLBuffer      — N*2 triangles (UInt32 indices)
//   • a SCNGeometry with two SCNGeometrySource buffers aliased to the above
//   • a SCNNode in the scene
//
// Shared across all species:
//
//   • a SimBuffer<TreeTransform>   — one entry per tree (position + sway quat)
//     The CPU writes these each tick before the kernel dispatches.
//
// Per frame, ForestController calls `tick(time:tree:)` to upload tree
// transforms, then `encodeExpand(to:)` to dispatch one compute pass per
// species. SceneKit reads the resulting vertex buffers via SCNGeometrySource
// the same frame — zero CPU round-trip.
//
// Replaces the per-leaf-SCNNode approach. Where the old system needed
// ~22K SCNNodes for 10 trees, this needs 3 (one per species) and scales
// to 100K+ leaves with a constant CPU cost.

// ── Public data shapes (must mirror LeafField.metal exactly) ─────────────────

/// One leaf in the GPU instance buffer. Built CPU-side by TreeGeometry,
/// uploaded once per scene rebuild, never touched per-frame.
///
/// Per the ALIGNMENT RULE in PBDSolver.swift: a bare SIMD3<Float> followed
/// by trailing Floats would give Swift stride 64 vs Metal stride 48 with
/// `packed_float3`, and reads on the GPU would be misaligned — producing
/// occasional gigantic leaves where sizeW/sizeH alias into quaternion
/// components. Everything is packed into SIMD4<Float> slots instead.
public struct LeafInstance {
    /// xyz = tree-local position, w = treeIdx (bitcast UInt32→Float; recover
    /// in Metal via `as_type<uint>(localPosAndTreeIdx.w)`).
    public var localPosAndTreeIdx: SIMD4<Float>   // 16
    /// Leaf rotation in tree-local space.
    public var orientQuat: simd_quatf             // 16
    /// x = quad width (m), y = quad height (m), z = per-leaf wind phase
    /// (rad), w = unused pad.
    public var sizeAndPhase: SIMD4<Float>         // 16
                                                  // total stride: 48 bytes

    public init(localPos: SIMD3<Float>,
                treeIdx: UInt32,
                orientQuat: simd_quatf,
                sizeW: Float,
                sizeH: Float,
                phase: Float) {
        let treeIdxFloat = Float(bitPattern: treeIdx)
        self.localPosAndTreeIdx = SIMD4(localPos.x, localPos.y, localPos.z,
                                        treeIdxFloat)
        self.orientQuat = orientQuat
        self.sizeAndPhase = SIMD4(sizeW, sizeH, phase, 0)
    }
}

/// Per-tree transform. Updated every tick from the CPU side to reflect
/// the current whole-tree sway. Position never changes after spawn; quat
/// changes with the wind.
public struct TreeTransform {
    /// xyz = tree root world position, w = unused.
    public var worldPosAndPad: SIMD4<Float>       // 16
    /// Tree sway rotation (whole-tree wind).
    public var worldQuat: simd_quatf              // 16
                                                  // total stride: 32 bytes

    public init(worldPos: SIMD3<Float>, worldQuat: simd_quatf) {
        self.worldPosAndPad = SIMD4(worldPos.x, worldPos.y, worldPos.z, 0)
        self.worldQuat = worldQuat
    }
}

private struct LeafExpandUniforms {
    var leafCount: UInt32
    var time:      Float
    var windAmp:   Float
    var windFreq:  Float
}

// ── LeafField ─────────────────────────────────────────────────────────────────

@MainActor
public final class LeafField {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "LeafField")

    public let engine: SimEngine

    private let expandPipeline: MTLComputePipelineState

    /// Shared across all species — index → TreeTransform.
    private let treeBuffer: SimBuffer<TreeTransform>

    /// Per-species channels. Each species has its own instance buffer,
    /// vertex/normal buffers, geometry, node, and material.
    public private(set) var channels: [String: SpeciesChannel] = [:]

    /// Number of tree slots reserved at init. After `setTreeTransform` is
    /// called for indices 0..<treeCount, the kernel reads tree transforms
    /// by `inst.treeIdx`.
    public let treeCapacity: Int

    // ── Init ───────────────────────────────────────────────────────────

    public init?(engine: SimEngine, treeCapacity: Int) {
        let device = engine.device
        guard let pipe = engine.pipelineCache.pipelineState(
            name: "leafExpand", device: device
        ) else {
            Self.log.error("leafExpand pipeline missing — is LeafField.metal in the Shaders/ folder?")
            return nil
        }
        guard let treeBuf = SimBuffer<TreeTransform>(
            device: device, capacity: max(1, treeCapacity), label: "LeafField.trees"
        ) else { return nil }

        // Initialise tree slots to identity (zero position, identity quat).
        let identity = TreeTransform(
            worldPos: SIMD3(0, 0, 0),
            worldQuat: simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))
        )
        treeBuf.write(Array(repeating: identity, count: treeCapacity))

        self.engine = engine
        self.expandPipeline = pipe
        self.treeBuffer = treeBuf
        self.treeCapacity = treeCapacity
    }

    // ── Species channels ──────────────────────────────────────────────

    /// Create (or replace) a species channel with the given instances.
    /// Builds the GPU buffers + SCNNode and returns the node so the caller
    /// can add it to the scene. Subsequent calls for the same key replace
    /// the channel (releasing prior buffers).
    ///
    /// **IMPORTANT — set an explicit `node.boundingBox` after adding.**
    /// The returned node's vertex buffer is GPU-written every frame; at
    /// the moment SceneKit first samples it for an auto-BB the buffer
    /// is still zeroed, so SceneKit caches a degenerate point at origin
    /// and frustum-culls the entire LeafField as soon as the camera
    /// stops including origin. Provide an AABB that covers the world-
    /// space extent of your tree scatter + tallest crown + sway margin.
    public func setSpecies(
        key: String,
        instances: [LeafInstance],
        material: SCNMaterial
    ) -> SCNNode? {
        // Drop the prior channel's node from the scene before rebuilding.
        channels[key]?.node.removeFromParentNode()
        channels[key] = nil

        guard !instances.isEmpty else { return nil }
        guard let ch = SpeciesChannel(
            engine: engine, instances: instances, material: material, label: key
        ) else {
            Self.log.error("LeafField: failed to build species channel '\(key)'")
            return nil
        }
        channels[key] = ch
        return ch.node
    }

    // ── Tree transforms ───────────────────────────────────────────────

    /// Update one tree's world transform. Safe to call every frame.
    /// The kernel reads tree transforms during its next dispatch.
    public func setTreeTransform(
        index: Int, worldPos: SIMD3<Float>, worldQuat: simd_quatf
    ) {
        guard index >= 0 && index < treeCapacity else { return }
        let ptr = treeBuffer.contents
        ptr[index] = TreeTransform(worldPos: worldPos, worldQuat: worldQuat)
    }

    // ── Per-frame expand ──────────────────────────────────────────────

    /// Encode one compute dispatch per species into the given command
    /// buffer. Caller is responsible for committing the command buffer.
    /// `time` should advance monotonically (typically scene clock seconds).
    public func encodeExpand(
        to cb: MTLCommandBuffer,
        time: Float, windAmp: Float, windFreq: Float
    ) {
        for (_, ch) in channels {
            guard let enc = cb.makeComputeCommandEncoder() else { continue }
            enc.label = "LeafField.\(ch.label).expand"

            var u = LeafExpandUniforms(
                leafCount: UInt32(ch.leafCount),
                time: time,
                windAmp: windAmp,
                windFreq: windFreq
            )
            enc.setComputePipelineState(expandPipeline)
            enc.setBuffer(ch.instanceBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(treeBuffer.buffer,         offset: 0, index: 1)
            enc.setBytes(&u, length: MemoryLayout<LeafExpandUniforms>.stride, index: 2)
            enc.setBuffer(ch.positionBuffer.buffer,  offset: 0, index: 3)
            enc.setBuffer(ch.normalBuffer.buffer,    offset: 0, index: 4)

            let w = min(ch.leafCount, expandPipeline.maxTotalThreadsPerThreadgroup)
            enc.dispatchThreads(
                MTLSize(width: ch.leafCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
            )
            enc.endEncoding()
        }
    }
}

// ── SpeciesChannel ────────────────────────────────────────────────────────────
//
// All the GPU resources for a single species's leaves. Built once at scene
// setup, reused across the scene's lifetime.

@MainActor
public final class SpeciesChannel {

    public let label: String
    public let leafCount: Int
    public let node: SCNNode

    let instanceBuffer: SimBuffer<LeafInstance>
    let positionBuffer: SimBuffer<SIMD3<Float>>
    let normalBuffer:   SimBuffer<SIMD3<Float>>

    init?(engine: SimEngine,
          instances: [LeafInstance],
          material: SCNMaterial,
          label: String) {
        let device = engine.device
        let n = instances.count
        let vertexCount = n * 4

        guard let instBuf = SimBuffer<LeafInstance>(
            device: device, capacity: n, label: "LeafField.\(label).instances"
        ) else { return nil }
        instBuf.write(instances)

        guard let posBuf = SimBuffer<SIMD3<Float>>(
            device: device, capacity: vertexCount, label: "LeafField.\(label).positions"
        ) else { return nil }
        guard let normBuf = SimBuffer<SIMD3<Float>>(
            device: device, capacity: vertexCount, label: "LeafField.\(label).normals"
        ) else { return nil }

        // ── Static UV buffer ──────────────────────────────────────────
        //
        // Each quad's UVs are constant — (0,0)/(1,0)/(0,1)/(1,1). Built
        // once at init, never updated.
        let uvByteLen = MemoryLayout<SIMD2<Float>>.stride * vertexCount
        guard let uvBuf = device.makeBuffer(
            length: uvByteLen, options: .storageModeShared
        ) else { return nil }
        uvBuf.label = "LeafField.\(label).uvs"
        let uvPtr = uvBuf.contents().bindMemory(
            to: SIMD2<Float>.self, capacity: vertexCount
        )
        for i in 0..<n {
            uvPtr[i * 4 + 0] = SIMD2(0, 0)  // bottom-left
            uvPtr[i * 4 + 1] = SIMD2(1, 0)  // bottom-right
            uvPtr[i * 4 + 2] = SIMD2(0, 1)  // top-left
            uvPtr[i * 4 + 3] = SIMD2(1, 1)  // top-right
        }

        // ── Static index buffer ───────────────────────────────────────
        //
        // 2 triangles per quad, 6 indices each. UInt32 indices since
        // vertexCount easily exceeds UInt16.max for large leaf counts.
        // Winding follows SceneKit's CW front-face convention; both faces
        // are visible anyway via isDoubleSided in the material.
        var indices = [UInt32]()
        indices.reserveCapacity(n * 6)
        for i in 0..<n {
            let b = UInt32(i * 4)
            // corners: 0=BL, 1=BR, 2=TL, 3=TR
            indices.append(contentsOf: [b + 0, b + 2, b + 1,
                                        b + 1, b + 2, b + 3])
        }
        let idxByteLen = MemoryLayout<UInt32>.stride * indices.count
        guard let idxBuf = indices.withUnsafeBufferPointer({ ptr -> MTLBuffer? in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: idxByteLen,
                options: .storageModeShared
            )
        }) else { return nil }
        idxBuf.label = "LeafField.\(label).indices"

        // ── SCNGeometry binding ───────────────────────────────────────
        let stride12 = MemoryLayout<Float>.stride * 3  // packed_float3
        let posSource = SCNGeometrySource(
            buffer: posBuf.buffer, vertexFormat: .float3, semantic: .vertex,
            vertexCount: vertexCount, dataOffset: 0, dataStride: stride12
        )
        let normSource = SCNGeometrySource(
            buffer: normBuf.buffer, vertexFormat: .float3, semantic: .normal,
            vertexCount: vertexCount, dataOffset: 0, dataStride: stride12
        )
        let uvSource = SCNGeometrySource(
            buffer: uvBuf, vertexFormat: .float2, semantic: .texcoord,
            vertexCount: vertexCount, dataOffset: 0,
            dataStride: MemoryLayout<SIMD2<Float>>.stride
        )
        let element = SCNGeometryElement(
            buffer: idxBuf,
            primitiveType: .triangles,
            primitiveCount: indices.count / 3,
            bytesPerIndex: MemoryLayout<UInt32>.stride
        )

        let geom = SCNGeometry(
            sources: [posSource, normSource, uvSource],
            elements: [element]
        )
        geom.firstMaterial = material

        // Phase 4.13a + 4.17 — publish a GPU mesh descriptor so the
        // Illuminatorama scene extractor can bridge this leaf field
        // directly. Without it the leaves are invisible through the
        // overlay (the SCNGeometry's vertex source is
        // `MTLBuffer`-backed and the CPU-readback path returns empty
        // `data`). UV is essential here because leaves are textured
        // sprite-cards — the leaf material samples an alpha cutout +
        // tint texture, and synthesising UV = (0,0) collapses every
        // fragment into the texture's top-left corner.
        geom.illuminatoramaGPUMesh = IlluminatoramaGPUMeshDescriptor(
            positionBuffer: posBuf.buffer,
            normalBuffer: normBuf.buffer,
            positionStride: stride12,
            normalStride: stride12,
            vertexCount: vertexCount,
            bodyIndexBuffer: idxBuf,
            bodyIndexCount: indices.count,
            bodyIndexType: .uint32,
            capIndexBuffer: nil,
            capIndexCount: 0,
            capIndexType: .uint32,
            uvBuffer: uvBuf,
            uvStride: MemoryLayout<SIMD2<Float>>.stride
        )

        let n_ = SCNNode(geometry: geom)
        n_.name = "LeafField.\(label)"

        self.label = label
        self.leafCount = n
        self.node = n_
        self.instanceBuffer = instBuf
        self.positionBuffer = posBuf
        self.normalBuffer = normBuf
    }
}
