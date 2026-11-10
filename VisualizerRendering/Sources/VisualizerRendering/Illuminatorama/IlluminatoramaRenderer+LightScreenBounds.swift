import Foundation
import Metal

/// **Tiled light masks for the deferred lighting kernel (Daydream DH-1625)** — the host half. The work is the
/// GPU's: `illumi_light_tiles` (`Shaders/IlluminatoramaLightBounds.metal`, where the mechanism and the
/// conservativeness argument live) runs one thread per 16 px tile just before `illumi_lighting`, reads the
/// tile's depth range and the frame's light buffers, and writes one bit per light per tile into
/// `shd.lightBoundsBuffer`, which the lighting kernel reads at `buffer(12)`. `lightScreenCullEnabled = false`
/// makes it write all-ones (the A/B lever; `VIZ_PERF_AB=nolightcull`). Nothing here is computed on the CPU.
extension IlluminatoramaRenderer {
    /// Keep in step with `kLightTileSize` / `kLightTileWords` (IlluminatoramaLightTiles.h).
    static let lightTileSize = 16, lightTileWords = 4

    /// Encode the tile kernel into `cb`, ahead of the lighting pass. Always leaves `shd.lightBoundsBuffer`
    /// holding a mask for every tile of the internal frame, so the lighting kernel binds it unconditionally.
    func encodeLightScreenBounds(_ cb: MTLCommandBuffer) {
        guard let depth = gbs.depthTexture else { return }
        let tilesWide = (depth.width + Self.lightTileSize - 1) / Self.lightTileSize
        let tilesHigh = (depth.height + Self.lightTileSize - 1) / Self.lightTileSize
        let length = max(1, tilesWide * tilesHigh) * Self.lightTileWords * MemoryLayout<UInt32>.stride
        /// "Every light, everywhere" — the pre-DH-1625 behaviour — for a frame the kernel cannot serve.
        func allLights() {
            let ones = [UInt32](repeating: .max, count: length / MemoryLayout<UInt32>.stride)
            shd.lightBoundsBuffer = ones.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
            shd.lightBoundsBuffer?.label = "Illuminatorama.lightTiles.allLights"
        }
        guard let pipe = engine.pipelineCache.pipelineState(name: "illumi_light_tiles", device: device) else {
            return allLights()
        }
        // Resized (never rewritten by the CPU): an in-flight frame keeps the buffer it was encoded with alive.
        if shd.lightBoundsBuffer?.length != length || shd.lightBoundsBuffer?.storageMode != .private {
            shd.lightBoundsBuffer = device.makeBuffer(length: length, options: .storageModePrivate)
            shd.lightBoundsBuffer?.label = "Illuminatorama.lightTiles"
        }
        guard let masks = shd.lightBoundsBuffer, let enc = timedComputeEncoder(cb, "lighting.tiles") else {
            return allLights()
        }
        enc.label = "Illuminatorama.lighting.tiles"
        enc.setComputePipelineState(pipe)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        enc.setBuffer(pointLightBuffer, offset: 0, index: 1)
        enc.setBuffer(spotLightBuffer, offset: 0, index: 3)
        enc.setBuffer(areaLightBuffer, offset: 0, index: 4)
        enc.setBuffer(masks, offset: 0, index: 12)
        var enabled: UInt32 = lightScreenCullEnabled ? 1 : 0
        enc.setBytes(&enabled, length: MemoryLayout<UInt32>.stride, index: 13)
        enc.setTexture(depth, index: 3)
        dispatch(enc, pipeline: pipe, width: tilesWide, height: tilesHigh)
        enc.endEncoding()
    }
}
