import Foundation
import Metal

/// **Per-light screen bounds for the deferred lighting kernel (Daydream DH-1625)** — the host half. The work
/// is the GPU's: `illumi_light_bounds` (`Shaders/IlluminatoramaLightBounds.metal`, where the mechanism and the
/// conservativeness argument live) runs one thread per light just before `illumi_lighting` and writes one NDC
/// rectangle per light into `shd.lightBoundsBuffer`, which the lighting kernel reads at `buffer(12)`.
/// `lightScreenCullEnabled = false` makes it write whole-screen rectangles (the A/B lever;
/// `VIZ_PERF_AB=nolightcull`). Nothing here is computed on the CPU.
extension IlluminatoramaRenderer {
    /// Encode the bounds kernel into `cb`, ahead of the lighting pass. Always leaves `shd.lightBoundsBuffer`
    /// bound-able (≥ one rectangle), so the lighting kernel can bind it unconditionally.
    func encodeLightScreenBounds(_ cb: MTLCommandBuffer) {
        let count = pointLights.count + spotLights.count + areaLights.count
        let stride = MemoryLayout<SIMD4<Float>>.stride
        if (shd.lightBoundsBuffer?.length ?? 0) < max(count, 1) * stride {
            // Grown, never rewritten in place by the CPU: an in-flight frame keeps the old buffer alive.
            let capacity = max(64, count * 2)
            shd.lightBoundsBuffer = device.makeBuffer(length: capacity * stride, options: .storageModePrivate)
            shd.lightBoundsBuffer?.label = "Illuminatorama.lightBounds"
        }
        guard count > 0 else { return }
        guard let bounds = shd.lightBoundsBuffer,
              let pipe = engine.pipelineCache.pipelineState(name: "illumi_light_bounds", device: device),
              let enc = timedComputeEncoder(cb, "lighting.bounds") else {
            // The kernel cannot run (pipeline or encoder missing): the lighting kernel must not read rectangles
            // nobody wrote, so hand it "every light, everywhere" — the pre-DH-1625 behaviour, never a cull.
            let whole = [SIMD4<Float>](repeating: SIMD4<Float>(-2, -2, 2, 2), count: max(64, count * 2))
            shd.lightBoundsBuffer = whole.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
            return
        }
        enc.label = "Illuminatorama.lighting.bounds"
        enc.setComputePipelineState(pipe)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        enc.setBuffer(pointLightBuffer, offset: 0, index: 1)
        enc.setBuffer(spotLightBuffer, offset: 0, index: 3)
        enc.setBuffer(areaLightBuffer, offset: 0, index: 4)
        enc.setBuffer(bounds, offset: 0, index: 12)
        var enabled: UInt32 = lightScreenCullEnabled ? 1 : 0
        enc.setBytes(&enabled, length: MemoryLayout<UInt32>.stride, index: 13)
        let group = min(count, pipe.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: group, height: 1, depth: 1))
        enc.endEncoding()
    }
}
