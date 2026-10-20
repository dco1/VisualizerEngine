import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    /// Bake the per-triangle GPU-diff maps (#60 item 1 + Phase D): owner
    /// instance, flag bits (frame-B half / deforming), OBJECT-space verts (rigid
    /// re-frame), and vertex-index triples (deform re-frame) per grouped-soup
    /// triangle, accumulated in lockstep with the soup build so global triangle
    /// order matches exactly. Any mismatch leaves the buffers nil → the CPU
    /// re-frame path stays in charge.
    func buildGPUDiffPerTriBuffers(triOwner: [UInt32],
                                           objVerts: [SIMD4<Float>],
                                           isFrameB: [Bool],
                                           triIdx: [SIMD4<UInt32>],
                                           deforming: [Bool]) {
        scs.surfGPUDiffChartOwnerBuffer = nil; scs.surfGPUDiffChartFrameBuffer = nil
        scs.surfGPUDiffTriOwnerBuffer = nil; scs.surfGPUDiffTriFlagsBuffer = nil
        scs.surfGPUDiffObjVertsBuffer = nil; scs.surfGPUDiffTriCount = 0
        scs.surfGPUDiffDeformIdxBuffer = nil; scs.surfGPUDiffDeformDispatches = []
        let triCount = triOwner.count
        guard triCount > 0, triCount == rts.rtTriangleCount,
              objVerts.count == triCount * 3, isFrameB.count == triCount,
              triIdx.count == triCount, deforming.count == triCount,
              scs.surfReframeTriPipeline != nil else { return }
        var flags = [UInt32](repeating: 0, count: triCount)
        for t in 0..<triCount {
            flags[t] = (isFrameB[t] ? Self.scFlagFrameB : 0)
                     | (deforming[t] ? Self.scFlagDeforming : 0)
        }
        guard let ob = device.makeBuffer(bytes: triOwner,
                  length: MemoryLayout<UInt32>.stride * triCount, options: .storageModeShared),
              let fb = device.makeBuffer(bytes: flags,
                  length: MemoryLayout<UInt32>.stride * triCount, options: .storageModeShared),
              let vb = device.makeBuffer(bytes: objVerts,
                  length: MemoryLayout<SIMD4<Float>>.stride * objVerts.count, options: .storageModeShared),
              let ib = device.makeBuffer(bytes: triIdx,
                  length: MemoryLayout<SIMD4<UInt32>>.stride * triCount, options: .storageModeShared)
        else { return }
        ob.label = "Illuminatorama.surfcache.gpuDiff.triOwner"
        fb.label = "Illuminatorama.surfcache.gpuDiff.triFlags"
        vb.label = "Illuminatorama.surfcache.gpuDiff.objVerts"
        ib.label = "Illuminatorama.surfcache.gpuDiff.triIdx"
        scs.surfGPUDiffTriOwnerBuffer = ob
        scs.surfGPUDiffTriFlagsBuffer = fb
        scs.surfGPUDiffObjVertsBuffer = vb
        scs.surfGPUDiffDeformIdxBuffer = ib
        scs.surfGPUDiffTriCount = triCount
        for sb in scs.surfGPUDiffStatsRing { memset(sb.contents(), 0, sb.length) }
    }

    /// Chart-path GPU-diff maps (#60 item 1): per-card owner instance (−1 =
    /// static / degenerate, never re-framed) + bake-pose object-space frame,
    /// flattened to 4×float4 per card for the kernel.
    func buildGPUDiffChartBuffers(objFrames: [SurfCardFrame], cardOwner: [Int32]) {
        scs.surfGPUDiffTriOwnerBuffer = nil; scs.surfGPUDiffTriFlagsBuffer = nil
        scs.surfGPUDiffObjVertsBuffer = nil; scs.surfGPUDiffTriCount = 0
        scs.surfGPUDiffChartOwnerBuffer = nil; scs.surfGPUDiffChartFrameBuffer = nil
        guard scs.surfCardCount > 0, objFrames.count == scs.surfCardCount,
              cardOwner.count == scs.surfCardCount,
              scs.surfReframeChartPipeline != nil else { return }
        var frames = [SIMD4<Float>](); frames.reserveCapacity(scs.surfCardCount * 4)
        for f in objFrames {
            frames.append(SIMD4(f.origin, 0)); frames.append(SIMD4(f.uAxis, 0))
            frames.append(SIMD4(f.vAxis, 0));  frames.append(SIMD4(f.normal, 0))
        }
        guard let ob = device.makeBuffer(bytes: cardOwner,
                  length: MemoryLayout<Int32>.stride * cardOwner.count, options: .storageModeShared),
              let fb = device.makeBuffer(bytes: frames,
                  length: MemoryLayout<SIMD4<Float>>.stride * frames.count, options: .storageModeShared)
        else { return }
        ob.label = "Illuminatorama.surfcache.gpuDiff.cardOwner"
        fb.label = "Illuminatorama.surfcache.gpuDiff.cardFrame"
        scs.surfGPUDiffChartOwnerBuffer = ob
        scs.surfGPUDiffChartFrameBuffer = fb
        for sb in scs.surfGPUDiffStatsRing { memset(sb.contents(), 0, sb.length) }
    }

    /// Phase D + #60 item 1 — the full GPU per-triangle dynamic-card pass.
    /// Returns false (CPU fallback) unless every map, pipeline, and deforming
    /// instance's live position buffer is resolvable; on success encodes the
    /// rigid kernel (diff + re-frame + dirty ownership for every non-deforming
    /// card) followed by one deform dispatch per deforming instance.
    func encodeGPUDiffPerTriPath(_ cb: MTLCommandBuffer,
                                         dirtyBuf: MTLBuffer, cardBuf: MTLBuffer) -> Bool {
        let deformingActive = scs.surfHasDeformingCards && !scs.noDeformCardsOverride
        guard surfaceCacheIncremental || deformingActive,
              let pipe = scs.surfReframeTriPipeline,
              let ownerBuf = scs.surfGPUDiffTriOwnerBuffer,
              let flagsBuf = scs.surfGPUDiffTriFlagsBuffer,
              let vertsBuf = scs.surfGPUDiffObjVertsBuffer,
              let statsBuf = surfGPUDiffStatsBuffer,
              let triCardBuf = scs.surfTriCardBuffer,
              scs.surfGPUDiffTriCount == rts.rtTriangleCount,
              scs.surfSoupBaseCPU.count == instances.count else { return false }
        // Resolve every deform dispatch's live position buffer up front — if any
        // is missing (solver unregistered ahead of the re-bake), take the CPU
        // path this frame rather than half-updating the dirty flags.
        var deformBindings: [(d: SurfDeformDispatch, positions: MTLBuffer, vertexCount: Int)] = []
        if deformingActive {
            guard scs.surfReframeDeformPipeline != nil, scs.surfGPUDiffDeformIdxBuffer != nil
            else { return false }
            for d in scs.surfGPUDiffDeformDispatches {
                guard d.instanceIndex < instances.count,
                      let task = fxs.gpuRepackTasks.first(where: { $0.kind == d.kind })
                else { return false }
                deformBindings.append((d, task.positionBuffer, task.vertexCount))
            }
        }
        encodeGPUDiffReframe(cb, pipeline: pipe, threadCount: scs.surfGPUDiffTriCount,
                             dirtyBuf: dirtyBuf, cardBuf: cardBuf,
                             ownerBuf: ownerBuf, statsBuf: statsBuf,
                             rigidEnabled: surfaceCacheIncremental,
                             triCardBuf: triCardBuf, triFlagsBuf: flagsBuf,
                             triVertsBuf: vertsBuf, tag: "[gpu]")
        if !deformBindings.isEmpty {
            encodeGPUDiffDeform(cb, bindings: deformBindings,
                                dirtyBuf: dirtyBuf, cardBuf: cardBuf, statsBuf: statsBuf)
        }
        return true
    }

    /// Phase D — encode the deforming-instance re-frames: one dispatch per
    /// deforming instance, threads = its local soup triangles, reading the
    /// solver's LIVE packed_float3 position buffer through the baked
    /// vertex-index triples. Runs after the rigid kernel (which skips
    /// deform-flagged triangles, so this pass solely owns these cards).
    private func encodeGPUDiffDeform(_ cb: MTLCommandBuffer,
                                     bindings: [(d: SurfDeformDispatch, positions: MTLBuffer, vertexCount: Int)],
                                     dirtyBuf: MTLBuffer, cardBuf: MTLBuffer,
                                     statsBuf: MTLBuffer) {
        guard let pipeline = scs.surfReframeDeformPipeline,
              let idxBuf = scs.surfGPUDiffDeformIdxBuffer,
              let triCardBuf = scs.surfTriCardBuffer,
              let flagsBuf = scs.surfGPUDiffTriFlagsBuffer else { return }
        guard let enc = timedComputeEncoder(cb, "surfcacheReframeDeform") else { return }
        enc.label = "Illuminatorama.surfcache.gpuDiff.deform"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(cardBuf, offset: 0, index: 0)
        enc.setBuffer(dirtyBuf, offset: 0, index: 1)
        enc.setBuffer(currentInstanceBuffer, offset: 0, index: 2)
        enc.setBuffer(triCardBuf, offset: 0, index: 3)
        enc.setBuffer(flagsBuf, offset: 0, index: 4)
        enc.setBuffer(idxBuf, offset: 0, index: 5)
        enc.setBuffer(statsBuf, offset: 0, index: 8)
        for b in bindings {
            var u = SCDeformUniforms(triBase: UInt32(b.d.triBase),
                                     triCount: UInt32(b.d.triCount),
                                     instanceIndex: UInt32(b.d.instanceIndex),
                                     vertexCount: UInt32(b.vertexCount))
            enc.setBuffer(b.positions, offset: 0, index: 6)
            enc.setBytes(&u, length: MemoryLayout<SCDeformUniforms>.stride, index: 7)
            dispatch1D(enc, pipeline: pipeline, count: b.d.triCount)
        }
        enc.endEncoding()
    }

    /// Encode the GPU-side moved-set detection + re-frame (#60 item 1). One
    /// dispatch; the surface-cache update kernel later in the same command
    /// buffer reads the re-framed cards + dirty flags (hazard tracking orders
    /// the two). Also drains this ring slot's atomic stats into the diagnostic
    /// sidecar — the slot was last written `maxFramesInFlight` frames ago, and the
    /// in-flight semaphore guarantees that frame has completed (DH-0537 #1).
    /// DH-0537 #1 — before a CPU write into a buffer the PREVIOUS frame's GPU passes may still
    /// be reading. Under `maxFramesInFlight == 2` that frame can be executing while this frame's
    /// CPU encode runs, so an in-place rewrite of `surfCardBuffer` / `surfCardDirtyBuffer` /
    /// `texelCard` / `cardRect` / `cardRequested` was a read-during-write hazard (the comments
    /// said "SAFE only while maxFramesInFlight == 1", and the pool had been 2 for a long time).
    /// Waiting drains the pipeline for that one frame, so only the paths that touch GPU-owned
    /// memory from the CPU pay it: the `VIZ_SURFCACHE_GPU_DIFF=0` fallback loops, the chart
    /// CPU re-frame, streaming residency and the feedback drain. The default GPU-diff lane
    /// writes those buffers from kernels on THIS frame's command buffer, which hazard
    /// tracking orders, and never calls this.
    func awaitInFlightFrameBeforeCPUCacheWrite() {
        gbs.lastCommittedFrameCB?.waitUntilCompleted()   // gpu-ok: opt-in CPU cache paths only, never the default per-frame lane
    }

    func encodeGPUDiffReframe(_ cb: MTLCommandBuffer,
                                      pipeline: MTLComputePipelineState,
                                      threadCount: Int,
                                      dirtyBuf: MTLBuffer, cardBuf: MTLBuffer,
                                      ownerBuf: MTLBuffer, statsBuf: MTLBuffer,
                                      rigidEnabled: Bool = true,
                                      chartFrameBuf: MTLBuffer? = nil,
                                      triCardBuf: MTLBuffer? = nil,
                                      triFlagsBuf: MTLBuffer? = nil,
                                      triVertsBuf: MTLBuffer? = nil,
                                      tag: String) {
        guard threadCount > 0 else { return }
        let stats = statsBuf.contents().bindMemory(to: UInt32.self, capacity: 4)
        if stats[0] > 0 || stats[1] > 0 || stats[2] > 0 {
            Self.recordSurfCacheStats("dynamic: moved=\(stats[0]) deformed=\(stats[2]) dirtiedCards=\(stats[1]) of \(scs.surfCardCount) \(tag)")
        }
        stats[0] = 0; stats[1] = 0; stats[2] = 0; stats[3] = 0
        var u = SCReframeUniforms(count: UInt32(threadCount),
                                  cardCount: UInt32(scs.surfCardCount),
                                  instanceCount: UInt32(instances.count),
                                  epsilon: 1e-5,   // matrix-diff threshold, mirrors the CPU loop
                                  rigidEnabled: rigidEnabled ? 1 : 0)
        guard let enc = timedComputeEncoder(cb, "surfcacheReframe") else { return }
        enc.label = "Illuminatorama.surfcache.gpuDiff"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(cardBuf, offset: 0, index: 0)
        enc.setBuffer(dirtyBuf, offset: 0, index: 1)
        enc.setBuffer(currentInstanceBuffer, offset: 0, index: 2)
        enc.setBuffer(previousInstanceBuffer, offset: 0, index: 3)
        enc.setBuffer(ownerBuf, offset: 0, index: 4)
        if let triCardBuf, let triFlagsBuf, let triVertsBuf {
            enc.setBuffer(triCardBuf, offset: 0, index: 5)
            enc.setBuffer(triFlagsBuf, offset: 0, index: 6)
            enc.setBuffer(triVertsBuf, offset: 0, index: 7)
            enc.setBytes(&u, length: MemoryLayout<SCReframeUniforms>.stride, index: 8)
            enc.setBuffer(statsBuf, offset: 0, index: 9)
        } else if let chartFrameBuf {
            enc.setBuffer(chartFrameBuf, offset: 0, index: 5)
            enc.setBytes(&u, length: MemoryLayout<SCReframeUniforms>.stride, index: 6)
            enc.setBuffer(statsBuf, offset: 0, index: 7)
        }
        dispatch1D(enc, pipeline: pipeline, count: threadCount)
        enc.endEncoding()
    }
}
