import Metal
import simd
import Signack

/// One placed instance's GPU-selected cut: the compacted index list a raster draw reads, and the
/// two indirect argument records that say how much of it to draw (`SignackGPUDrawArgs`).
///
/// Written every frame by `encode` on the host's render queue BEFORE the frame's raster passes, in
/// the same queue order — Metal's hazard tracking orders this frame's writes after the previous
/// frame's reads, so ONE output buffer is enough (S1 needed a 3-slot CPU ring).
public final class SignackGPUCut: @unchecked Sendable {
    public let hierarchy: SignackGPUHierarchy
    /// The index buffer the host's raster mesh draws (over the hierarchy's shared vertex buffer).
    public let outputIndices: MTLBuffer
    /// The cluster each output triangle came from (`outputIndices[3t..<3t+3]` ↔ `outputClusterIDs[t]`)
    /// — S3: a visibility-buffer triangle id names its cluster with one load (debug view, S4/S5).
    public let outputClusterIDs: MTLBuffer
    /// `SignackGPUDrawArgs` layout: primary record at 0, shadow at 32, soft at 64, soft dispatch at 96.
    public let drawArgs: MTLBuffer
    let flags: MTLBuffer
    let counters: MTLBuffer
    /// S4 — route this instance's sub-pixel on-screen triangles to the software rasterizer's range
    /// (set by the host each frame before `encode`; nil = none, exactly S2).
    public var routing: SignackGPURouting?

    public init?(hierarchy: SignackGPUHierarchy, device: MTLDevice, label: String = "Signack.cut") {
        guard let out = device.makeBuffer(length: max(4, hierarchy.maxCutIndices * 4), options: .storageModeShared),
              let clusterIDs = device.makeBuffer(length: max(4, hierarchy.maxCutIndices / 3 * 4), options: .storageModeShared),
              let args = device.makeBuffer(length: SignackGPUDrawArgs.bufferLength, options: .storageModeShared),
              let flags = device.makeBuffer(length: max(4, hierarchy.clusterCount), options: .storageModeShared),
              let counters = device.makeBuffer(length: 32, options: .storageModeShared)
        else { return nil }
        // Zeroed: an instance draws nothing until its first cut lands.
        memset(args.contents(), 0, args.length)
        memset(counters.contents(), 0, counters.length)
        memset(flags.contents(), 0, flags.length)
        out.label = "\(label).indices"; args.label = "\(label).args"
        flags.label = "\(label).flags"; counters.label = "\(label).counters"
        self.hierarchy = hierarchy
        outputIndices = out
        clusterIDs.label = "\(label).clusterIDs"
        outputClusterIDs = clusterIDs
        drawArgs = args
        self.flags = flags
        self.counters = counters
    }

    /// Bytes this instance holds on the GPU (its output + bookkeeping; the DAG is the type's).
    public var instanceBytes: Int { outputIndices.length + outputClusterIDs.length + drawArgs.length + flags.length + counters.length }

    /// Encode this instance's select → compact → finalize for `view` into `encoder` (serial).
    public func encode(view: SignackView, pipelines: SignackGPUPipelines, encoder enc: MTLComputeCommandEncoder) {
        var u = SignackGPUCutUniforms(view: view, clusterCount: hierarchy.clusterCount,
                                      capacity: hierarchy.maxCutIndices)
        // Routing needs a threadgroup per cluster with a thread per triangle (≤ 128).
        var route = SignackGPURouteUniforms(objectToClip: matrix_identity_float4x4, viewport: .zero, maxSpan: 0, vertexStride: 0)
        if let r = routing, r.maxArea > 0, pipelines.write.maxTotalThreadsPerThreadgroup >= 128 {
            u.softArea = r.maxArea
            route = SignackGPURouteUniforms(objectToClip: r.objectToClip, viewport: r.viewport, maxSpan: r.maxSpan,
                                            vertexStride: UInt32(r.vertexStride))
            enc.setBuffer(r.vertices, offset: 0, index: 9)
        } else {
            enc.setBuffer(hierarchy.indexBuffer, offset: 0, index: 9)   // bound, never read
        }
        enc.setBytes(&route, length: MemoryLayout<SignackGPURouteUniforms>.stride, index: 8)
        let n = hierarchy.clusterCount
        enc.setBytes(&u, length: MemoryLayout<SignackGPUCutUniforms>.stride, index: 0)
        enc.setBuffer(hierarchy.clusterBuffer, offset: 0, index: 1)
        enc.setBuffer(flags, offset: 0, index: 2)
        enc.setBuffer(counters, offset: 0, index: 3)
        enc.setBuffer(hierarchy.indexBuffer, offset: 0, index: 4)
        enc.setBuffer(outputIndices, offset: 0, index: 5)
        enc.setBuffer(drawArgs, offset: 0, index: 6)
        enc.setBuffer(outputClusterIDs, offset: 0, index: 7)

        enc.setComputePipelineState(pipelines.classify)
        let w = min(64, pipelines.classify.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.setComputePipelineState(pipelines.write)
        // ≤128 triangles per cluster: one thread per triangle (each copies its 3 indices' worth by stride).
        let tg = min(128, pipelines.write.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(MTLSize(width: n, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.setComputePipelineState(pipelines.finalize)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    }

    /// Indices the LAST COMPLETED frame drew in its primary (camera) pass — read from shared memory
    /// without waiting, so it may lag the GPU by the frames in flight. For census, not for logic.
    public var lastPrimaryIndexCount: Int {
        Int(drawArgs.contents().load(fromByteOffset: SignackGPUDrawArgs.primaryOffset, as: UInt32.self))
    }

    /// Indices the last completed frame routed to the software rasterizer (S4).
    public var lastSoftIndexCount: Int {
        Int(drawArgs.contents().load(fromByteOffset: SignackGPUDrawArgs.softOffset, as: UInt32.self))
    }

    /// Indices the last completed frame drew in its shadow passes (the whole LOD cut).
    public var lastShadowIndexCount: Int {
        Int(drawArgs.contents().load(fromByteOffset: SignackGPUDrawArgs.shadowOffset, as: UInt32.self))
    }

    /// The last completed selection, per cluster — `lodCut` (flag ≥ 1, what the shadow passes draw)
    /// and `onScreen` (flag 2, what the camera draws), as ascending cluster ids. Read it only after
    /// the command buffer that encoded it has completed (tests, diagnostics).
    public func readSelection() -> (lodCut: [UInt32], onScreen: [UInt32]) {
        let f = flags.contents().bindMemory(to: UInt8.self, capacity: hierarchy.clusterCount)
        var cut: [UInt32] = [], on: [UInt32] = []
        for i in 0..<hierarchy.clusterCount where f[i] != 0 {
            cut.append(UInt32(i))
            if f[i] == 2 { on.append(UInt32(i)) }
        }
        return (cut, on)
    }
}

/// Encode many instances' cuts into one compute pass (one serial encoder; three dispatches each).
public enum SignackGPUCutPass {
    public static func encode(_ work: [(cut: SignackGPUCut, view: SignackView)],
                              pipelines: SignackGPUPipelines, commandBuffer: MTLCommandBuffer) {
        guard !work.isEmpty, let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "Signack.cut"
        for w in work { w.cut.encode(view: w.view, pipelines: pipelines, encoder: enc) }
        enc.endEncoding()
    }
}
