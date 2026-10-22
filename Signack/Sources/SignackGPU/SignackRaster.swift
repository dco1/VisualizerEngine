import Metal

/// Signack S4 — the software (compute) rasterizer for sub-pixel triangles, as one frame's
/// resources: the window-space triangle list the host's vertex stage appends to, and the per-pixel
/// nearest depth + id it resolves them into. Shaders and the full contract: `SignackRaster.metal`.
///
/// Per frame:
///   1. the host appends `SignackRasterTriangle`s to `triangles`, claiming slots with an atomic
///      increment of the uint at `state` offset 0 (`appendCounterOffset`); a claim past `capacity`
///      is dropped by the host and counted here;
///   2. `encode` resolves them — depth pass, then id pass — confined to the scissor;
///   3. the host's merge reads `depth` / `ids` inside the same scissor and resets every pixel it
///      reads to `emptyDepth` (self-clearing: no clear pass, and nothing outside the scissor is
///      ever written). A host without such a merge calls `encodeClear` instead.
public final class SignackRaster: @unchecked Sendable {
    public let device: MTLDevice
    /// Window-space triangles, `capacity` of them.
    public private(set) var triangles: MTLBuffer
    public private(set) var capacity: Int
    /// [0] append counter · [1] this frame's count · [2] dropped (cumulative) · [3] clipped by the
    /// span cap (cumulative) · the indirect dispatch at byte 32.
    public let state: MTLBuffer
    /// Float bits of the nearest soft depth per pixel (row-major, `width` per row), or `emptyDepth`.
    public private(set) var depth: MTLBuffer
    /// The winning triangle's id per pixel (`uint2`): host x with bit 31 = front-facing, host y.
    public private(set) var ids: MTLBuffer
    public private(set) var width: Int
    public private(set) var height: Int

    public static let emptyDepth: UInt32 = 0xFFFF_FFFF
    public static let appendCounterOffset = 0
    public static let threadsPerGroup = 64
    /// A triangle whose sample box is wider than this (px, per axis) is clipped to it and counted —
    /// the cut routes only clusters a few pixels across, so this is a backstop, not a budget.
    public var maxSpan = 64

    public init?(device: MTLDevice) {
        guard let state = device.makeBuffer(length: 48, options: .storageModeShared),
              let t = device.makeBuffer(length: 48, options: .storageModeShared),
              let d = device.makeBuffer(length: 4, options: .storageModeShared),
              let i = device.makeBuffer(length: 8, options: .storageModeShared) else { return nil }
        memset(state.contents(), 0, state.length)
        state.label = "Signack.raster.state"
        self.device = device
        self.state = state
        triangles = t; capacity = 1
        depth = d; ids = i
        width = 0; height = 0
    }

    /// Size the frame: a `width` × `height` target and room for `triangles` triangles. Grows only;
    /// a new target starts empty. False if Metal refused the memory (the host must not route soft).
    @discardableResult
    public func reserve(width w: Int, height h: Int, triangles n: Int) -> Bool {
        if n > capacity {
            guard let t = device.makeBuffer(length: n * MemoryLayout<SignackRasterTriangle>.stride,
                                            options: .storageModeShared) else { return false }
            t.label = "Signack.raster.triangles"
            triangles = t; capacity = n
        }
        if w != width || h != height {
            let px = max(w * h, 1)
            guard let d = device.makeBuffer(length: px * 4, options: .storageModeShared),
                  let i = device.makeBuffer(length: px * 8, options: .storageModeShared) else { return false }
            memset(d.contents(), 0xFF, d.length)
            d.label = "Signack.raster.depth"; i.label = "Signack.raster.ids"
            depth = d; ids = i; width = w; height = h
        }
        return true
    }

    /// Resolve this frame's appended triangles into `depth` / `ids`, writing only inside `scissor`.
    public func encode(commandBuffer: MTLCommandBuffer, scissor: SignackRasterRect, pipelines: SignackGPUPipelines) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "Signack.raster"
        encode(encoder: enc, scissor: scissor, pipelines: pipelines)
        enc.endEncoding()
    }

    /// The same, on the host's (serial) compute encoder — after its vertex stage, in one encoder.
    public func encode(encoder enc: MTLComputeCommandEncoder, scissor: SignackRasterRect, pipelines: SignackGPUPipelines) {
        var u = uniforms(scissor)
        enc.setBytes(&u, length: MemoryLayout<SignackRasterUniforms>.stride, index: 0)
        enc.setBuffer(state, offset: 0, index: 1)
        enc.setBuffer(triangles, offset: 0, index: 2)
        enc.setBuffer(depth, offset: 0, index: 3)
        enc.setBuffer(ids, offset: 0, index: 4)
        enc.setComputePipelineState(pipelines.rasterPrepare)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        let tg = MTLSize(width: Self.threadsPerGroup, height: 1, depth: 1)
        enc.setComputePipelineState(pipelines.rasterDepth)
        enc.dispatchThreadgroups(indirectBuffer: state, indirectBufferOffset: 32, threadsPerThreadgroup: tg)
        enc.setComputePipelineState(pipelines.rasterIDs)
        enc.dispatchThreadgroups(indirectBuffer: state, indirectBufferOffset: 32, threadsPerThreadgroup: tg)
    }

    /// Reset `scissor` to empty — for a host (or test) that reads the result without a merge.
    public func encodeClear(commandBuffer: MTLCommandBuffer, scissor: SignackRasterRect, pipelines: SignackGPUPipelines) {
        guard scissor.width > 0, scissor.height > 0, let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        var u = uniforms(scissor)
        enc.setBytes(&u, length: MemoryLayout<SignackRasterUniforms>.stride, index: 0)
        enc.setBuffer(depth, offset: 0, index: 3)
        enc.setComputePipelineState(pipelines.rasterClear)
        enc.dispatchThreads(MTLSize(width: scissor.width, height: scissor.height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
    }

    /// Cumulative counters (read after the frame completed): triangles dropped over capacity, and
    /// triangles clipped by `maxSpan` — both should stay 0. `lastCount` = triangles last resolved.
    public var stats: (lastCount: Int, dropped: Int, clipped: Int) {
        let p = state.contents().bindMemory(to: UInt32.self, capacity: 4)
        return (Int(p[1]), Int(p[2]), Int(p[3]))
    }

    private func uniforms(_ s: SignackRasterRect) -> SignackRasterUniforms {
        let x0 = max(0, min(s.x, width)), y0 = max(0, min(s.y, height))
        let x1 = max(x0, min(s.x + s.width, width)), y1 = max(y0, min(s.y + s.height, height))
        return SignackRasterUniforms(width: UInt32(width), height: UInt32(height), x0: UInt32(x0), y0: UInt32(y0),
                                     x1: UInt32(x1), y1: UInt32(y1), capacity: UInt32(capacity),
                                     maxSpan: UInt32(max(maxSpan, 1)))
    }
}
