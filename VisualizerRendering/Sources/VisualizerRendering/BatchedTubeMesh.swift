import Foundation
import Metal
import OSLog
import VisualizerCore

// ── BATCHED TUBE MESH ────────────────────────────────────────────────────────
//
// Many deforming PBD-tube franks rendered in ONE draw call instead of one per
// frank. Each `PBDTubeRenderer` currently registers its OWN GPU mesh, so the
// Illuminatorama renderer (which groups instances by meshKind and draws one
// `drawIndexedPrimitives` per group) issues N draws — multiplied across the
// G-buffer pass AND every shadow cascade. Measured ~0.37 ms/frank of pure
// draw-call/state overhead.
//
// THE FIX: one registered mesh whose position/normal/color buffers are SHARED
// across all franks. Slot `s` owns vertices [s*vertsPerTube, (s+1)*vertsPerTube).
// Each tube's expand kernel is redirected (via `PBDTubeRenderer.externalOutput`)
// to write into its slot. The index buffer replicates the single-tube body+cap
// pattern across all slots with a `slot*vertsPerTube` offset, so ONE
// `drawIndexedPrimitives` of the whole buffer draws every slot.
//
// CORRECTNESS: the mesh draws ALL `maxSlots` slots every frame (fixed
// indexCount). So every slot must hold EITHER a live frank's geometry (written
// by its tube's expand that frame) OR a collapsed degenerate point. A freed
// slot must be `collapse()`d so it renders as zero-area, off-screen triangles.
@MainActor
public final class BatchedTubeMesh {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "BatchedTubeMesh")

    public let maxSlots: Int
    public let ringCount: Int      // == a tube's visualRingCount
    public let ringSegments: Int
    public let vertsPerTube: Int   // ringCount*ringSegments + 2 (two cap poles)

    private let positionBuffer: MTLBuffer  // packed_float3, 12-byte stride
    private let normalBuffer: MTLBuffer    // packed_float3, 12-byte stride
    private let colorBuffer: MTLBuffer     // float4, 16-byte stride
    private let indexBuffer: MTLBuffer     // uint32

    /// The single registered Illuminatorama mesh — all franks draw through this.
    public let handle: IlluminatoramaMeshHandle

    // Far degenerate point a collapsed slot's vertices are written to.
    private static let degenerate = SIMD3<Float>(0, -100_000, 0)

    public init?(device: MTLDevice,
                 renderer: IlluminatoramaRenderer,
                 maxSlots: Int,
                 ringCount: Int,
                 ringSegments: Int) {
        guard maxSlots > 0, ringCount > 0, ringSegments > 0 else { return nil }
        self.maxSlots = maxSlots
        self.ringCount = ringCount
        self.ringSegments = ringSegments
        let vpt = ringCount * ringSegments + 2
        self.vertsPerTube = vpt

        let totalVerts = maxSlots * vpt
        guard
            let pos = device.makeBuffer(length: totalVerts * 12, options: .storageModeShared),
            let nrm = device.makeBuffer(length: totalVerts * 12, options: .storageModeShared),
            let col = device.makeBuffer(length: totalVerts * 16, options: .storageModeShared)
        else {
            Self.log.error("BatchedTubeMesh buffer alloc failed (slots=\(maxSlots), vpt=\(vpt))")
            return nil
        }
        pos.label = "BatchedTubeMesh.position"
        nrm.label = "BatchedTubeMesh.normal"
        col.label = "BatchedTubeMesh.color"
        self.positionBuffer = pos
        self.normalBuffer = nrm
        self.colorBuffer = col

        // ── ONE-tube index pattern (replicated per slot) ──────────────────────
        // Mirrors DynamicMeshIndexCache.tubeIndices EXACTLY so the batched topology
        // matches the per-frank path byte-for-byte. Body + caps are merged into a
        // single combined element here (the registered mesh has one element).
        let tubeVerts = ringCount * ringSegments
        let K = max(1, min(3, (ringCount - 2) / 3))
        let bodyStart = K
        let bodyEnd   = ringCount - K

        var oneTube = [UInt32]()
        oneTube.reserveCapacity((ringCount) * ringSegments * 6 + ringSegments * 6)

        // Body quad strips: between body rings only (K…bodyEnd-1).
        if bodyEnd - 1 > bodyStart {
            for s in bodyStart..<(bodyEnd - 1) {
                for r in 0..<ringSegments {
                    let next = (r + 1) % ringSegments
                    let a = UInt32(s * ringSegments + r)
                    let b = UInt32(s * ringSegments + next)
                    let c = UInt32((s + 1) * ringSegments + r)
                    let d = UInt32((s + 1) * ringSegments + next)
                    oneTube.append(contentsOf: [a, b, c, b, d, c])
                }
            }
        }
        // Cap quad strips: every strip on either end, including rim strips.
        let capStripRanges: [Range<Int>] = [
            0..<bodyStart,
            (bodyEnd - 1)..<(ringCount - 1)
        ]
        for range in capStripRanges {
            for s in range {
                for r in 0..<ringSegments {
                    let next = (r + 1) % ringSegments
                    let a = UInt32(s * ringSegments + r)
                    let b = UInt32(s * ringSegments + next)
                    let c = UInt32((s + 1) * ringSegments + r)
                    let d = UInt32((s + 1) * ringSegments + next)
                    oneTube.append(contentsOf: [a, b, c, b, d, c])
                }
            }
        }
        // Start cap fan: pole at tubeVerts, outward normal = −tangent.
        let cap0 = UInt32(tubeVerts)
        for r in 0..<ringSegments {
            oneTube.append(contentsOf: [cap0, UInt32((r + 1) % ringSegments), UInt32(r)])
        }
        // End cap fan: pole at tubeVerts+1, outward normal = +tangent.
        let cap1 = UInt32(tubeVerts + 1)
        let lastBase = (ringCount - 1) * ringSegments
        for r in 0..<ringSegments {
            oneTube.append(contentsOf: [cap1, UInt32(lastBase + r),
                                        UInt32(lastBase + (r + 1) % ringSegments)])
        }

        let indicesPerTube = oneTube.count
        var allIndices = [UInt32]()
        allIndices.reserveCapacity(indicesPerTube * maxSlots)
        for s in 0..<maxSlots {
            let base = UInt32(s * vpt)
            for idx in oneTube { allIndices.append(idx + base) }
        }

        guard let idxBuf = allIndices.withUnsafeBufferPointer({ ptr -> MTLBuffer? in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: MemoryLayout<UInt32>.stride * ptr.count,
                              options: .storageModeShared)
        }) else {
            Self.log.error("BatchedTubeMesh index buffer alloc failed")
            return nil
        }
        idxBuf.label = "BatchedTubeMesh.index"
        self.indexBuffer = idxBuf

        let desc = IlluminatoramaGPUMeshDescriptor(
            positionBuffer: pos,
            normalBuffer: nrm,
            positionStride: 12,
            normalStride: 12,
            vertexCount: totalVerts,
            bodyIndexBuffer: idxBuf,
            bodyIndexCount: allIndices.count,
            bodyIndexType: .uint32,
            colorBuffer: col,
            colorStride: 16)
        guard let h = renderer.registerGPUMesh(desc) else {
            Self.log.error("BatchedTubeMesh registerGPUMesh failed")
            return nil
        }
        self.handle = h

        // Every slot starts collapsed; live tubes overwrite their slot via expand.
        collapseAll()
    }

    /// The shared-buffer slice for `slot` — hand to a `PBDTubeRenderer` so its
    /// expand writes into [slot*vertsPerTube, (slot+1)*vertsPerTube).
    public func externalOutput(slot: Int) -> PBDTubeRenderer.ExternalTubeOutput {
        PBDTubeRenderer.ExternalTubeOutput(
            position: positionBuffer,
            normal: normalBuffer,
            color: colorBuffer,
            vertexBase: slot * vertsPerTube)
    }

    /// Write a slot's whole position range to a far degenerate point so it
    /// renders as zero-area, off-screen triangles. Positions are packed_float3
    /// (12-byte stride) — write 3 Floats per vertex, NOT a 16-byte SIMD3.
    public func collapse(slot: Int) {
        guard slot >= 0, slot < maxSlots else { return }
        let base = slot * vertsPerTube
        let p = positionBuffer.contents().advanced(by: base * 12)
            .bindMemory(to: Float.self, capacity: vertsPerTube * 3)
        let d = Self.degenerate
        for v in 0..<vertsPerTube {
            p[v * 3 + 0] = d.x
            p[v * 3 + 1] = d.y
            p[v * 3 + 2] = d.z
        }
    }

    public func collapseAll() {
        for s in 0..<maxSlots { collapse(slot: s) }
    }
}
