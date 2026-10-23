import Metal
import simd
import Signack

/// **S6 — every instance of ONE baked type, LOD-picked, culled and compacted on the GPU in one
/// pass** (`Shaders/SignackInstances.metal`), so a planting of hundreds costs a few instanced draws.
///
/// The host keeps its instances' records (its own instance struct — opaque here, `instanceStride`
/// bytes) and world bounds in `source` / `bounds`, written ONCE when the planting changes. Each
/// frame `encode` sorts them into buckets — rung 0…K−1 of the type's `SignackRungs`, then the
/// impostor — in two lists (primary = on screen, shadow = all), and writes one indexed-instanced
/// indirect record per (bucket, list) into `drawArgs`. The host registers one raster mesh per
/// bucket over `lists` with `primaryOffset(bucket:)` / `shadowOffset(bucket:)`.
///
/// Memory: `lists` holds 2 × buckets × capacity records (every instance could land in any bucket):
/// 800 plants × 5 buckets × 384 B × 2 = 3 MB per type.
public final class SignackGPUInstanceBatch: @unchecked Sendable {
    public let capacity: Int
    public let instanceStride: Int
    public let rungs: [SignackRungs.Rung]
    public let impostorIndexCount: Int
    public var bucketCount: Int { rungs.count + (impostorIndexCount > 0 ? 1 : 0) }
    /// The host's instance records (capacity × stride), written by `setInstances`.
    public let source: MTLBuffer
    public let bounds: MTLBuffer
    /// The compacted records every bucket's draw reads (bind at offset 0).
    public let lists: MTLBuffer
    public let drawArgs: MTLBuffer
    let counters: MTLBuffer
    public private(set) var instanceCount = 0
    /// DH-1466 — this frame's Hi-Z occlusion source (world → the pyramid's clip), or nil = no test.
    public var occlusion: SignackGPUOcclusion?

    public static let maxBuckets = 8

    public init?(device: MTLDevice, rungs: SignackRungs, impostorIndexCount: Int, capacity: Int,
                 instanceStride: Int, label: String = "Signack.instances") {
        let buckets = rungs.rungs.count + (impostorIndexCount > 0 ? 1 : 0)
        guard capacity > 0, instanceStride % 16 == 0, !rungs.rungs.isEmpty, buckets <= Self.maxBuckets,
              let src = device.makeBuffer(length: capacity * instanceStride, options: .storageModeShared),
              let bnd = device.makeBuffer(length: capacity * MemoryLayout<SignackGPUInstanceBounds>.stride,
                                          options: .storageModeShared),
              let lst = device.makeBuffer(length: 2 * buckets * capacity * instanceStride, options: .storageModeShared),
              let args = device.makeBuffer(length: buckets * 64, options: .storageModeShared),
              let cnt = device.makeBuffer(length: 2 * Self.maxBuckets * 4, options: .storageModeShared)
        else { return nil }
        memset(args.contents(), 0, args.length)
        memset(cnt.contents(), 0, cnt.length)
        src.label = "\(label).source"; bnd.label = "\(label).bounds"; lst.label = "\(label).lists"
        args.label = "\(label).args"; cnt.label = "\(label).counters"
        self.capacity = capacity
        self.instanceStride = instanceStride
        self.rungs = rungs.rungs
        self.impostorIndexCount = impostorIndexCount
        source = src; bounds = bnd; lists = lst; drawArgs = args; counters = cnt
    }

    /// Byte offsets of a bucket's two records in `drawArgs`.
    public static func primaryOffset(bucket: Int) -> Int { bucket * 64 }
    public static func shadowOffset(bucket: Int) -> Int { bucket * 64 + 32 }

    /// Replace the instance set (≤ capacity): `write` fills `count` records at the raw pointer
    /// (stride `instanceStride`) and their bounds.
    public func setInstances(count: Int,
                             _ write: (UnsafeMutableRawPointer, UnsafeMutablePointer<SignackGPUInstanceBounds>) -> Void) {
        instanceCount = min(count, capacity)
        write(source.contents(),
              bounds.contents().bindMemory(to: SignackGPUInstanceBounds.self, capacity: capacity))
    }

    /// Bytes this batch holds on the GPU.
    public var gpuBytes: Int { source.length + bounds.length + lists.length + drawArgs.length + counters.length }

    /// Encode this frame's bucket sort for `view` (two dispatches) into `enc` (serial encoder).
    public func encode(view: SignackInstanceView, pipelines: SignackGPUPipelines, encoder enc: MTLComputeCommandEncoder) {
        var u = SignackGPUInstanceUniforms(view: view, batch: self)
        enc.setBytes(&u, length: MemoryLayout<SignackGPUInstanceUniforms>.stride, index: 0)
        enc.setBuffer(bounds, offset: 0, index: 1)
        enc.setBuffer(source, offset: 0, index: 2)
        enc.setBuffer(lists, offset: 0, index: 3)
        enc.setBuffer(counters, offset: 0, index: 4)
        enc.setBuffer(drawArgs, offset: 0, index: 5)
        SignackGPUOcclusion.bind(occlusion, enc, buffer: 6, texture: 0, fallback: pipelines.blankHiZ)
        if instanceCount > 0 {
            enc.setComputePipelineState(pipelines.instancesClassify)
            let w = min(64, pipelines.instancesClassify.maxTotalThreadsPerThreadgroup)
            enc.dispatchThreads(MTLSize(width: instanceCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        }
        enc.setComputePipelineState(pipelines.instancesFinalize)
        enc.dispatchThreads(MTLSize(width: bucketCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: bucketCount, height: 1, depth: 1))
    }

    /// Instances per bucket in the LAST COMPLETED frame (primary, shadow) — census, not logic.
    public var lastCounts: [(primary: Int, shadow: Int)] {
        (0..<bucketCount).map { b in
            (Int(drawArgs.contents().load(fromByteOffset: Self.primaryOffset(bucket: b) + 4, as: UInt32.self)),
             Int(drawArgs.contents().load(fromByteOffset: Self.shadowOffset(bucket: b) + 4, as: UInt32.self)))
        }
    }
}

/// Encode many batches' bucket sorts into one compute pass.
public enum SignackGPUInstancePass {
    public static func encode(_ work: [(batch: SignackGPUInstanceBatch, view: SignackInstanceView)],
                              pipelines: SignackGPUPipelines, commandBuffer: MTLCommandBuffer) {
        guard !work.isEmpty, let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "Signack.instances"
        for w in work { w.batch.encode(view: w.view, pipelines: pipelines, encoder: enc) }
        enc.endEncoding()
    }
}
