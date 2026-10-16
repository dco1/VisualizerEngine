import Metal

/// **A still rendered as independent batches** (Daydream DH-1360, "render until clean").
///
/// Host-driven and opt-in: nothing here exists until a host creates one for a still, and the renderer
/// never calls it — so every host that does not is byte-identical. The host renders a still as G
/// batches, each a complete still on its own (its own radiance-cache build, GI and AO means, and its
/// own run of accumulated frames, the TAA accumulator restarting at blend 1), and between frames:
///
/// 1. `foldBatch()` after each batch: adds the batch's mean image — the TAA accumulator the last frame
///    wrote (`latestTAAHistoryTexture`, linear HDR at the internal size, before bloom and the tonemap) —
///    into an fp32 sum, and reduces it per tile together with the tonemapped picture and the G-buffer.
/// 2. `writeMeanToHistory()` at the end: the fp32 mean of every batch, written back into that
///    accumulator, so ONE frame rendered with the accumulator held (blend 0) prints it.
///
/// Why the export's mean is kept here and not in the accumulator: the accumulator is fp16, and a running
/// mean over hundreds of frames in fp16 stops moving (a step under half an ULP is rounded away — and in a
/// path tracer's right-skewed samples, that drops the small below-mean steps and keeps the rare large
/// ones, a bias). Batches of tens of frames stay well inside its precision; the sum across them is fp32.
/// The fp32 sum costs 16 B per internal pixel (147 MB at a 3840 × 2400 internal frame).
///
/// Tile statistics are per OUTPUT-pixel tile (`tileSize`), mapped onto the internal grid by the
/// supersample; see `IlluminatoramaStillBatches.metal` for the fields.
@MainActor
public final class IlluminatoramaStillBatches {

    /// One tile of one batch, decoded.
    public struct Tile: Equatable, Sendable {
        /// Mean luminance (Rec.709 Y) of the batch's mean image over the tile — linear, pre-exposure
        /// (× the renderer's `hdrPreExposure`; only ratios of it are meaningful).
        public var meanLuminance: Float
        /// Brightest pixel / mean (≥ 1; 1 for a black tile).
        public var peakToMean: Float
        /// max / min of the tile's 4 × 4 sub-block mean luminances (1 = flat).
        public var subBlockRatio: Float
        /// 1 − |Σ normal| / count over the tile's geometry pixels.
        public var normalSpread: Float
        public var skyFraction: Float
        public var emissiveFraction: Float
        /// Mean display luma code (0–255) of the tonemapped picture over the tile.
        public var displayCode: Float
        /// Share of the tile's output pixels with a channel ≥ 250.
        public var clippedFraction: Float
        /// Share of the tile's output pixels at luma code ≤ 2.
        public var darkFraction: Float
        /// The tile's share of the output frame's area.
        public var areaFraction: Float
    }

    /// Mirror of `StillTileParams` (IlluminatoramaStillBatches.metal) — 48 bytes.
    struct Params {
        var internalW: UInt32, internalH: UInt32, outputW: UInt32, outputH: UInt32
        var tilesX: UInt32, tilesY: UInt32, tileSize: Float, scaleX: Float
        var scaleY: Float, srgbOutput: UInt32, accumulate: UInt32, invBatches: Float
    }
    static let foldFields = 41
    static let displayFields = 4

    public let tileSize: Int
    public let tilesX: Int
    public let tilesY: Int
    /// Batches folded into the sum so far.
    public private(set) var batches = 0

    private let renderer: IlluminatoramaRenderer
    private let foldPipeline: MTLComputePipelineState
    private let displayPipeline: MTLComputePipelineState
    private let writePipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue
    private let sum: MTLBuffer
    private let foldTiles: MTLBuffer
    private let displayTiles: MTLBuffer
    private let internalW: Int, internalH: Int, outputW: Int, outputH: Int

    /// nil when a pipeline or a buffer cannot be made (the host then renders its fixed-frame still).
    /// The renderer's internal and output sizes must not change while this is alive.
    public init?(renderer: IlluminatoramaRenderer, tileSize: Int) {
        let device = renderer.device
        let cache = SimPipelineCache.shared
        guard let fold = cache.pipelineState(name: "illumi_still_batch_fold", device: device),
              let display = cache.pipelineState(name: "illumi_still_display_tiles", device: device),
              let write = cache.pipelineState(name: "illumi_still_write_mean", device: device) else { return nil }
        self.renderer = renderer
        self.tileSize = max(1, tileSize)
        internalW = renderer.width; internalH = renderer.height
        outputW = renderer.outputWidth; outputH = renderer.outputHeight
        tilesX = (outputW + self.tileSize - 1) / self.tileSize
        tilesY = (outputH + self.tileSize - 1) / self.tileSize
        guard let sum = device.makeBuffer(length: internalW * internalH * 16, options: .storageModePrivate),
              let ft = device.makeBuffer(length: tilesX * tilesY * Self.foldFields * 4, options: .storageModeShared),
              let dt = device.makeBuffer(length: tilesX * tilesY * Self.displayFields * 4, options: .storageModeShared)
        else { return nil }
        sum.label = "Illuminatorama.still.batchSum"
        self.sum = sum; foldTiles = ft; displayTiles = dt
        foldPipeline = fold; displayPipeline = display; writePipeline = write
        // The renderer's own queue: these passes order after any frame still in flight on it.
        queue = renderer.commandQueue
    }

    private func params() -> Params {
        Params(internalW: UInt32(internalW), internalH: UInt32(internalH),
               outputW: UInt32(outputW), outputH: UInt32(outputH),
               tilesX: UInt32(tilesX), tilesY: UInt32(tilesY), tileSize: Float(tileSize),
               scaleX: Float(internalW) / Float(max(outputW, 1)), scaleY: Float(internalH) / Float(max(outputH, 1)),
               srgbOutput: renderer.outputTexture.pixelFormat == .bgra8Unorm_srgb ? 1 : 0,
               accumulate: batches > 0 ? 1 : 0, invBatches: 1 / Float(max(batches, 1)))
    }

    private var sizesUnchanged: Bool {
        renderer.width == internalW && renderer.height == internalH
            && renderer.outputWidth == outputW && renderer.outputHeight == outputH
    }

    /// Fold the batch whose last frame the renderer has just COMPLETED (call after a blocking or awaited
    /// render): add its accumulator into the sum, and return its tiles in row-major order. Blocks until
    /// the GPU has finished (one small compute pass over the frame). Empty on a size change.
    public func foldBatch() -> [Tile] {
        guard sizesUnchanged, let cb = queue.makeCommandBuffer() else { return [] }
        cb.label = "Illuminatorama.still.fold"
        var p = params()
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let groups = MTLSize(width: tilesX, height: tilesY, depth: 1)
        if let enc = cb.makeComputeCommandEncoder() {
            enc.label = "Illuminatorama.still.fold"
            enc.setComputePipelineState(foldPipeline)
            enc.setTexture(renderer.latestTAAHistoryTexture, index: 0)
            enc.setTexture(renderer.gbufferNormalRoughnessTexture, index: 1)
            enc.setTexture(renderer.gbufferEmissionTexture, index: 2)
            enc.setTexture(renderer.gbufferDepthTexture, index: 3)
            enc.setBuffer(sum, offset: 0, index: 0)
            enc.setBuffer(foldTiles, offset: 0, index: 1)
            enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 2)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tgSize)
            enc.endEncoding()
        }
        if let enc = cb.makeComputeCommandEncoder() {
            enc.label = "Illuminatorama.still.display"
            enc.setComputePipelineState(displayPipeline)
            enc.setTexture(renderer.outputTexture, index: 0)
            enc.setBuffer(displayTiles, offset: 0, index: 0)
            enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 1)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tgSize)
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .completed else { return [] }
        batches += 1
        return decode()
    }

    /// Write the mean of every folded batch into the accumulator the NEXT frame reads as its history.
    /// The host then renders one frame with the accumulator held (TAA blend 0) to print it.
    @discardableResult
    public func writeMeanToHistory() -> Bool {
        guard batches > 0, sizesUnchanged, let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return false }
        cb.label = "Illuminatorama.still.writeMean"
        var p = params()
        enc.setComputePipelineState(writePipeline)
        enc.setTexture(renderer.latestTAAHistoryTexture, index: 0)
        enc.setBuffer(sum, offset: 0, index: 0)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 1)
        enc.dispatchThreads(MTLSize(width: internalW, height: internalH, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return cb.status == .completed
    }

    private func decode() -> [Tile] {
        let f = foldTiles.contents().bindMemory(to: Float.self, capacity: tilesX * tilesY * Self.foldFields)
        let d = displayTiles.contents().bindMemory(to: Float.self, capacity: tilesX * tilesY * Self.displayFields)
        let frame = Float(outputW * outputH)
        return (0..<(tilesX * tilesY)).map { t in
            let r = t * Self.foldFields, s = t * Self.displayFields
            let count = max(f[r + 1], 1)
            let mean = f[r] / count
            let geo = f[r + 6]
            let nLen = (f[r + 3] * f[r + 3] + f[r + 4] * f[r + 4] + f[r + 5] * f[r + 5]).squareRoot()
            var lo = Float.greatestFiniteMagnitude, hi: Float = 0
            for k in 0..<16 where f[r + 25 + k] > 0 {
                let m = f[r + 9 + k] / f[r + 25 + k]
                lo = min(lo, m); hi = max(hi, m)
            }
            let ratio: Float = hi <= 0 ? 1 : (lo > 0 ? hi / lo : 1000)
            let px = max(d[s + 1], 1)
            return Tile(meanLuminance: mean,
                        peakToMean: mean > 0 ? f[r + 2] / mean : 1,
                        subBlockRatio: ratio,
                        normalSpread: geo > 0 ? max(0, 1 - nLen / geo) : 0,
                        skyFraction: f[r + 8] / count,
                        emissiveFraction: f[r + 7] / count,
                        displayCode: d[s] / px,
                        clippedFraction: d[s + 2] / px,
                        darkFraction: d[s + 3] / px,
                        areaFraction: d[s + 1] / max(frame, 1))
        }
    }
}
