import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

// ── BARK RENDERER ────────────────────────────────────────────────────────────
//
// GPU-resident merged-mesh renderer for the Forest scene's tree bark. The
// counterpart to LeafField: where LeafField batches all leaves into a few
// SCN draw calls and animates them with a Metal compute kernel, BarkRenderer
// does the same for the bark (trunk + leader + every branch + every root)
// of a single tree.
//
// Architectural goal — push the bark animation onto the GPU so the per-tree
// per-frame work on the CPU is bounded by "compute one quaternion per tree"
// regardless of branch count. The Forest scene currently rotates each tree's
// wrapper SCNNode each frame (the wrapper rotation is uniform across the
// tree's children, so a single Euler-angle write per tree is enough); a
// per-tree BarkRenderer rotates each vertex on the GPU instead, with the
// same effective rotation, but in a form ready to extend with hierarchical
// wind in a follow-up (per-vertex branchAnchor / branchPhase / heightWeight
// for per-branch sway with different phases — see Bark.metal for the
// extension point).
//
// Today's animation is exactly the rotation the tree-wrapper SCNNode used
// to apply — a rigid rotation of every vertex around the tree's local origin
// by a quaternion the caller computes once per frame. The wrapper's wind-
// related Euler is zeroed when this renderer is active so the rotation
// isn't double-applied; the leaves continue to receive the matching
// quaternion via LeafField's per-tree TreeTransform, so bark and leaves
// stay in lock-step.
//
// The merged mesh is built once per scene-rebuild from a sequence of
// `BarkSegment` records (one per skeleton segment). Topology is fixed;
// position + normal buffers are MTLBuffer-backed and SceneKit reads them
// directly via SCNGeometrySource(buffer:vertexFormat:semantic:…). Rest pose
// + rest normal live in separate MTLBuffers so the kernel can rotate from
// the un-animated baseline each frame rather than drift through repeated
// applications of small angles.

/// Input record for `BarkRenderer.build` — one per skeleton segment.
/// Carries the tree-local spine control points + per-point radii. Same
/// payload the SceneForest skeleton's `BranchSegment` already holds; we
/// don't import the SceneForest package here to keep the GPU runtime free
/// of scene-specific dependencies.
public struct BarkSegment: Sendable {
    public let points: [SIMD3<Float>]
    public let radii:  [Float]

    public init(points: [SIMD3<Float>], radii: [Float]) {
        precondition(points.count == radii.count && points.count >= 2,
                     "BarkSegment requires ≥2 paired points/radii")
        self.points = points
        self.radii  = radii
    }
}

private struct BarkWindUniforms {
    var quat: SIMD4<Float>
    var vertexCount: UInt32
    var pad0: UInt32 = 0
    var pad1: UInt32 = 0
    var pad2: UInt32 = 0
}

@MainActor
public final class BarkRenderer {

    private static let log = Logger(
        subsystem: AppLog.subsystem, category: "BarkRenderer"
    )

    public let engine: SimEngine
    public let node: SCNNode
    public let geometry: SCNGeometry
    public let vertexCount: Int

    private let windPipeline: MTLComputePipelineState
    // Rest pose (CPU-written once, GPU-read every frame).
    //
    // Matches the LeafField pattern: SimBuffer<SIMD3<Float>> allocates
    // a 16-byte-stride buffer (Swift's SIMD3<Float> alignment) but we
    // ONLY use the first 12 bytes of every slot — Metal reads as
    // `packed_float3*` (stride 12), SCN reads at `dataStride: 12`. To
    // make this work, the rest pose is written by a raw `Float*`
    // pointer at packed 12-byte stride, NOT through `SimBuffer.write`
    // (which would memcpy the array's natural 16-byte-stride layout
    // into the same buffer the kernel reads at 12 — the variant of
    // the swift_metal_simd3_stride_mismatch bug that bit this file's
    // first cut).
    private let restPosBuffer:  SimBuffer<SIMD3<Float>>
    private let restNormBuffer: SimBuffer<SIMD3<Float>>
    // Animated pose (GPU-written each frame, SCN-read at draw).
    private let outPosBuffer:   SimBuffer<SIMD3<Float>>
    private let outNormBuffer:  SimBuffer<SIMD3<Float>>

    /// Build from a sequence of skeleton segments. Returns nil if the
    /// merged vertex count would overflow or buffer allocation fails.
    public init?(
        engine: SimEngine,
        segments: [BarkSegment],
        material: SCNMaterial,
        radialSegments: Int = 10
    ) {
        let device = engine.device
        guard let pipe = engine.pipelineCache.pipelineState(
            name: "barkWind", device: device
        ) else {
            Self.log.error(
                "barkWind pipeline missing — is Bark.metal in the Shaders/ folder?"
            )
            return nil
        }
        self.windPipeline = pipe

        // ── Walk segments + emit CPU-side vertex arrays ──────────────
        var restPos:  [SIMD3<Float>] = []
        var restNorm: [SIMD3<Float>] = []
        var uvs:      [SIMD2<Float>] = []
        var inds:     [UInt32] = []
        // Worst-case reservation so we don't rehash mid-build.
        var estVerts = 0
        var estTris  = 0
        for s in segments {
            estVerts += s.points.count * radialSegments
            estTris  += (s.points.count - 1) * radialSegments * 2
        }
        restPos.reserveCapacity(estVerts)
        restNorm.reserveCapacity(estVerts)
        uvs.reserveCapacity(estVerts)
        inds.reserveCapacity(estTris * 3)

        for seg in segments {
            let n = seg.points.count
            guard n >= 2 else { continue }
            let ringStart = UInt32(restPos.count)

            // Parallel-transport frame propagation along the spine so
            // adjacent rings share a continuous rotation — avoids the
            // visible-twist artifact you get if each ring picks its own
            // up-vector reference independently. Same pattern
            // RootGenerator.revolutionMesh uses.
            var prevX: SIMD3<Float>? = nil
            for ringIdx in 0..<n {
                let centre = seg.points[ringIdx]
                let r = max(0.001, seg.radii[ringIdx])
                let tan: SIMD3<Float>
                if ringIdx == 0 {
                    tan = simd_normalize(seg.points[1] - seg.points[0])
                } else if ringIdx == n - 1 {
                    tan = simd_normalize(
                        seg.points[ringIdx] - seg.points[ringIdx - 1]
                    )
                } else {
                    tan = simd_normalize(
                        seg.points[ringIdx + 1] - seg.points[ringIdx - 1]
                    )
                }
                let ringX: SIMD3<Float>
                let ringZ: SIMD3<Float>
                if let px = prevX {
                    let proj = px - tan * simd_dot(px, tan)
                    let pl = simd_length(proj)
                    if pl > 1e-4 {
                        ringX = proj / pl
                    } else {
                        let refUp: SIMD3<Float> = abs(tan.y) > 0.97
                            ? SIMD3<Float>(1, 0, 0)
                            : SIMD3<Float>(0, 1, 0)
                        ringX = simd_normalize(simd_cross(refUp, tan))
                    }
                    ringZ = simd_normalize(simd_cross(tan, ringX))
                } else {
                    let refUp: SIMD3<Float> = abs(tan.y) > 0.97
                        ? SIMD3<Float>(1, 0, 0)
                        : SIMD3<Float>(0, 1, 0)
                    ringX = simd_normalize(simd_cross(refUp, tan))
                    ringZ = simd_normalize(simd_cross(tan, ringX))
                }
                prevX = ringX

                let v = Float(ringIdx) / Float(max(1, n - 1))
                for radIdx in 0..<radialSegments {
                    let a = Float(radIdx) / Float(radialSegments) * 2 * .pi
                    let ca = cos(a), sa = sin(a)
                    let p = centre + ringX * (r * ca) + ringZ * (r * sa)
                    let normal = simd_normalize(
                        ringX * ca + ringZ * sa
                    )
                    restPos.append(p)
                    restNorm.append(normal)
                    uvs.append(SIMD2<Float>(
                        Float(radIdx) / Float(radialSegments), v
                    ))
                }
            }

            // Same winding as BarkMesh.assemble (the CPU sibling) —
            // verified visible there, so the two paths stay byte-for-byte
            // compatible at the topology level.
            for ringIdx in 0..<(n - 1) {
                for radIdx in 0..<radialSegments {
                    let next = (radIdx + 1) % radialSegments
                    let bL = ringStart
                        + UInt32(ringIdx * radialSegments + radIdx)
                    let bR = ringStart
                        + UInt32(ringIdx * radialSegments + next)
                    let tL = ringStart
                        + UInt32((ringIdx + 1) * radialSegments + radIdx)
                    let tR = ringStart
                        + UInt32((ringIdx + 1) * radialSegments + next)
                    inds.append(bL); inds.append(tL); inds.append(bR)
                    inds.append(bR); inds.append(tL); inds.append(tR)
                }
            }
        }

        let vertexCount = restPos.count
        guard vertexCount > 0 else {
            Self.log.error("BarkRenderer: empty skeleton — refusing to allocate")
            return nil
        }
        self.vertexCount = vertexCount

        // ── Allocate GPU buffers ─────────────────────────────────────
        guard
            let rPos  = SimBuffer<SIMD3<Float>>(
                device: device, capacity: vertexCount,
                label: "BarkRenderer.restPos"
            ),
            let rNorm = SimBuffer<SIMD3<Float>>(
                device: device, capacity: vertexCount,
                label: "BarkRenderer.restNorm"
            ),
            let oPos  = SimBuffer<SIMD3<Float>>(
                device: device, capacity: vertexCount,
                label: "BarkRenderer.outPos"
            ),
            let oNorm = SimBuffer<SIMD3<Float>>(
                device: device, capacity: vertexCount,
                label: "BarkRenderer.outNorm"
            )
        else {
            Self.log.error("BarkRenderer: GPU buffer allocation failed")
            return nil
        }
        self.restPosBuffer  = rPos
        self.restNormBuffer = rNorm
        self.outPosBuffer   = oPos
        self.outNormBuffer  = oNorm

        // Upload rest pose at PACKED 12-byte stride (x0,y0,z0,x1,y1,z1,…).
        // Going through `SimBuffer.write([SIMD3<Float>])` would memcpy at
        // Swift's natural 16-byte stride and break the contract with the
        // kernel (`packed_float3*`, stride 12) and with SCN (vertex
        // source `dataStride: 12`). The animated buffers are initialised
        // to the rest pose too, so the first frame (before the kernel
        // has dispatched) renders correctly instead of sampling zeros.
        func writePacked(_ src: [SIMD3<Float>], to buf: MTLBuffer) {
            let ptr = buf.contents().bindMemory(
                to: Float.self, capacity: src.count * 3
            )
            for i in 0..<src.count {
                ptr[i * 3 + 0] = src[i].x
                ptr[i * 3 + 1] = src[i].y
                ptr[i * 3 + 2] = src[i].z
            }
        }
        writePacked(restPos,  to: rPos.buffer)
        writePacked(restNorm, to: rNorm.buffer)
        writePacked(restPos,  to: oPos.buffer)
        writePacked(restNorm, to: oNorm.buffer)

        // ── Static UV buffer ─────────────────────────────────────────
        let uvByteLen = MemoryLayout<SIMD2<Float>>.stride * vertexCount
        guard let uvBuf = device.makeBuffer(
            length: uvByteLen, options: .storageModeShared
        ) else { return nil }
        uvBuf.label = "BarkRenderer.uvs"
        let uvPtr = uvBuf.contents().bindMemory(
            to: SIMD2<Float>.self, capacity: vertexCount
        )
        for i in 0..<vertexCount { uvPtr[i] = uvs[i] }

        // ── Static index buffer ──────────────────────────────────────
        let idxByteLen = MemoryLayout<UInt32>.stride * inds.count
        guard let idxBuf = device.makeBuffer(
            bytes: inds, length: idxByteLen, options: .storageModeShared
        ) else { return nil }
        idxBuf.label = "BarkRenderer.indices"

        // ── SCN geometry referencing the MTLBuffers ──────────────────
        //
        // Stride 12 (packed_float3) — matches what the kernel writes and
        // what LeafField does. The `.float3` vertexFormat MUST pair with
        // a 12-byte stride; SceneKit rejects float3 + stride 16 silently
        // (the bark mesh renders as nothing — no magenta, no error).
        let stride12 = MemoryLayout<Float>.stride * 3
        let posSource = SCNGeometrySource(
            buffer: oPos.buffer,
            vertexFormat: .float3, semantic: .vertex,
            vertexCount: vertexCount, dataOffset: 0, dataStride: stride12
        )
        let normSource = SCNGeometrySource(
            buffer: oNorm.buffer,
            vertexFormat: .float3, semantic: .normal,
            vertexCount: vertexCount, dataOffset: 0, dataStride: stride12
        )
        let uvSource = SCNGeometrySource(
            buffer: uvBuf,
            vertexFormat: .float2, semantic: .texcoord,
            vertexCount: vertexCount, dataOffset: 0,
            dataStride: MemoryLayout<SIMD2<Float>>.stride
        )
        let element = SCNGeometryElement(
            buffer: idxBuf,
            primitiveType: .triangles,
            primitiveCount: inds.count / 3,
            bytesPerIndex: MemoryLayout<UInt32>.stride
        )
        let geom = SCNGeometry(
            sources: [posSource, normSource, uvSource],
            elements: [element]
        )
        geom.firstMaterial = material
        // Phase 4.13a + 4.17 — publish a GPU mesh descriptor so the
        // Illuminatorama scene extractor bridges this trunk geometry
        // straight from its compute-fed MTLBuffers. UV is required
        // because BarkRenderer paints a bark normal/diffuse texture
        // through SCN's texcoord source — without UV bridging the
        // trunk would render as a flat-coloured cylinder through the
        // overlay.
        geom.illuminatoramaGPUMesh = IlluminatoramaGPUMeshDescriptor(
            positionBuffer: oPos.buffer,
            normalBuffer: oNorm.buffer,
            positionStride: stride12,
            normalStride: stride12,
            vertexCount: vertexCount,
            bodyIndexBuffer: idxBuf,
            bodyIndexCount: inds.count,
            bodyIndexType: .uint32,
            capIndexBuffer: nil,
            capIndexCount: 0,
            capIndexType: .uint32,
            uvBuffer: uvBuf,
            uvStride: MemoryLayout<SIMD2<Float>>.stride
        )
        self.geometry = geom

        let n = SCNNode(geometry: geom)
        n.name = "bark"
        self.node = n

        self.engine = engine
    }

    // ── Per-frame dispatch ─────────────────────────────────────────────

    /// Encode one bark-wind compute pass into the given command buffer.
    /// `quat` is the per-tree sway quaternion (the same one ForestController
    /// passes to LeafField.setTreeTransform). The kernel rotates every
    /// vertex's rest position + rest normal by `quat` and writes the
    /// result into the buffers SceneKit reads as the geometry's vertex
    /// source. Caller is responsible for committing the command buffer.
    public func encodeWind(
        to cb: MTLCommandBuffer, quat: simd_quatf
    ) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "BarkRenderer.barkWind"

        // simd_quatf's vector form is (x, y, z, w) — see Apple's
        // documentation. We pack it into a SIMD4<Float> matching Bark.metal's
        // expectation of `quat.xyz = axis, quat.w = w`.
        let v = quat.vector
        var u = BarkWindUniforms(
            quat: SIMD4<Float>(v.x, v.y, v.z, v.w),
            vertexCount: UInt32(vertexCount)
        )

        enc.setComputePipelineState(windPipeline)
        enc.setBuffer(restPosBuffer.buffer,  offset: 0, index: 0)
        enc.setBuffer(restNormBuffer.buffer, offset: 0, index: 1)
        enc.setBuffer(outPosBuffer.buffer,   offset: 0, index: 2)
        enc.setBuffer(outNormBuffer.buffer,  offset: 0, index: 3)
        enc.setBytes(&u, length: MemoryLayout<BarkWindUniforms>.stride, index: 4)

        let w = min(vertexCount, windPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: vertexCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
        enc.endEncoding()
    }
}
