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
/// Memory: `lists` holds 2 × buckets × capacity records (every instance could land in any bucket),
/// then one region of `capacity` records per shadow view (DH-1493; an instance lands in one bucket,
/// so a view's buckets share it): 800 plants × (2 × 5 + 8) × 384 B = 5.5 MB per type.
///
/// **Per-light shadow culling (DH-1493).** `shadowViews` (world space, ≤ `SignackShadowViews.maxViews`)
/// adds one record per (view, bucket) at `shadowViewOffset(view:bucket:)`: only the instances whose
/// sphere can land in that view. Empty = exactly the two lists.
public final class SignackGPUInstanceBatch: @unchecked Sendable {
    public let capacity: Int
    public let instanceStride: Int
    public let rungs: [SignackRungs.Rung]
    public let impostorIndexCount: Int
    /// DH-1470 — where the impostor's indices start in the index buffer its bucket draws from: 0 when the
    /// impostor is its own mesh, the rungs' index count when a host packs every bucket into ONE buffer
    /// (one raster mesh drawing every bucket's record — see `IlluminatoramaMesh.additionalIndirectRecords`).
    public let impostorIndexStart: Int
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
    /// DH-1493 — this frame's shadow views (world-space frustums, the first `SignackShadowViews.maxViews`).
    public var shadowViews: [SignackFrustum] = []
    /// Per (view, bucket): count | range base | write cursor (3 × 64 uints).
    let viewBook: MTLBuffer
    /// Per instance: bucket | view mask << 8.
    let instView: MTLBuffer

    public static let maxBuckets = 8

    public init?(device: MTLDevice, rungs: SignackRungs, impostorIndexCount: Int, impostorIndexStart: Int = 0,
                 capacity: Int, instanceStride: Int, label: String = "Signack.instances") {
        let buckets = rungs.rungs.count + (impostorIndexCount > 0 ? 1 : 0)
        guard capacity > 0, instanceStride % 16 == 0, !rungs.rungs.isEmpty, buckets <= Self.maxBuckets,
              let src = device.makeBuffer(length: capacity * instanceStride, options: .storageModeShared),
              let bnd = device.makeBuffer(length: capacity * MemoryLayout<SignackGPUInstanceBounds>.stride,
                                          options: .storageModeShared),
              let lst = device.makeBuffer(length: (2 * buckets + SignackShadowViews.maxViews) * capacity * instanceStride,
                                          options: .storageModeShared),
              let args = device.makeBuffer(length: Self.drawArgsLength, options: .storageModeShared),
              let cnt = device.makeBuffer(length: 2 * Self.maxBuckets * 4, options: .storageModeShared),
              let book = device.makeBuffer(length: 3 * 64 * 4, options: .storageModeShared),
              let iv = device.makeBuffer(length: capacity * 4, options: .storageModeShared)
        else { return nil }
        memset(book.contents(), 0, book.length)
        book.label = "\(label).viewBook"; iv.label = "\(label).instView"
        viewBook = book; instView = iv
        memset(args.contents(), 0, args.length)
        memset(cnt.contents(), 0, cnt.length)
        src.label = "\(label).source"; bnd.label = "\(label).bounds"; lst.label = "\(label).lists"
        args.label = "\(label).args"; cnt.label = "\(label).counters"
        self.capacity = capacity
        self.instanceStride = instanceStride
        self.rungs = rungs.rungs
        self.impostorIndexCount = impostorIndexCount
        self.impostorIndexStart = max(0, impostorIndexStart)
        source = src; bounds = bnd; lists = lst; drawArgs = args; counters = cnt
    }

    /// Byte offsets of a bucket's two records in `drawArgs`.
    public static func primaryOffset(bucket: Int) -> Int { bucket * 64 }
    public static func shadowOffset(bucket: Int) -> Int { bucket * 64 + 32 }
    /// DH-1493 — byte offset of shadow view `view`'s record for `bucket` (`view` is the index into
    /// this frame's `shadowViews`).
    public static func shadowViewOffset(view: Int, bucket: Int) -> Int { maxBuckets * 64 + (view * maxBuckets + bucket) * 32 }
    static let drawArgsLength = maxBuckets * 64 + SignackShadowViews.maxViews * maxBuckets * 32

    /// Replace the instance set (≤ capacity): `write` fills `count` records at the raw pointer
    /// (stride `instanceStride`) and their bounds.
    public func setInstances(count: Int,
                             _ write: (UnsafeMutableRawPointer, UnsafeMutablePointer<SignackGPUInstanceBounds>) -> Void) {
        instanceCount = min(count, capacity)
        write(source.contents(),
              bounds.contents().bindMemory(to: SignackGPUInstanceBounds.self, capacity: capacity))
    }

    /// Bytes this batch holds on the GPU.
    public var gpuBytes: Int {
        source.length + bounds.length + lists.length + drawArgs.length + counters.length + viewBook.length + instView.length
    }

    /// Encode this frame's bucket sort for `view` (two dispatches) into `enc` (serial encoder).
    public func encode(view: SignackInstanceView, pipelines: SignackGPUPipelines, encoder enc: MTLComputeCommandEncoder) {
        var u = SignackGPUInstanceUniforms(view: view, batch: self)
        let views = Array(shadowViews.prefix(SignackShadowViews.maxViews))
        u.viewCount = UInt32(views.count)
        enc.setBytes(&u, length: MemoryLayout<SignackGPUInstanceUniforms>.stride, index: 0)
        SignackGPUInstanceShadowViews(views).bind(enc, index: 7)
        enc.setBuffer(viewBook, offset: 0, index: 8)
        enc.setBuffer(instView, offset: 0, index: 9)
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
            if !views.isEmpty {   // DH-1493 — the per-view copies
                enc.setComputePipelineState(pipelines.instancesViewsPrefix)
                enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                enc.setComputePipelineState(pipelines.instancesViewsWrite)
                let ww = min(64, pipelines.instancesViewsWrite.maxTotalThreadsPerThreadgroup)
                enc.dispatchThreads(MTLSize(width: instanceCount, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: ww, height: 1, depth: 1))
            }
        }
        enc.setComputePipelineState(pipelines.instancesFinalize)
        enc.dispatchThreads(MTLSize(width: bucketCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: bucketCount, height: 1, depth: 1))
    }

    /// DH-1493 — instances per bucket that shadow view `view`'s records drew in the last completed frame.
    public func lastShadowViewCounts(view: Int) -> [Int] {
        (0..<bucketCount).map {
            Int(drawArgs.contents().load(fromByteOffset: Self.shadowViewOffset(view: view, bucket: $0) + 4, as: UInt32.self))
        }
    }

    /// DH-1493 — the records shadow view `view` / `bucket` drew in the last completed frame (tests).
    public func shadowViewRecords(view: Int, bucket: Int) -> [Data] {
        let o = Self.shadowViewOffset(view: view, bucket: bucket)
        let p = drawArgs.contents()
        let n = Int(p.load(fromByteOffset: o + 4, as: UInt32.self))
        let base = Int(p.load(fromByteOffset: o + 16, as: UInt32.self))
        return (0..<n).map { Data(bytes: lists.contents() + (base + $0) * instanceStride, count: instanceStride) }
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
