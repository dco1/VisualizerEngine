import Foundation
import Metal
import OSLog
import VisualizerCore

// ── NOISE VOLUME ─────────────────────────────────────────────────────────────
//
// Tileable 3D RGBA16F texture holding the FBM + Worley fields the volumetric
// cloud kernel used to compute inline per march step. One trilinear sample
// at the slab-march step replaces ~60 hash ops, which is the dominant win
// for the cloud renderer (avenue 4 in the perf plan).
//
// Channel packing:
//   R = fbm4(p)         — low-frequency cloud body (4 octaves)
//   G = 1 - worley(p*4) — high-frequency erosion mask, baked at the 4×
//                         density the cloud kernel asks for so reading both
//                         R and G needs ONE sample at noise-space position q
//   B, A = unused (reserved for future hi-freq detail / curl noise)
//
// The bake runs once at init via the `noiseBake` Metal kernel. Tileable
// hashing in the bake makes the volume periodic, so the kernel samples
// under `address::repeat` and never sees a seam at typical scales.
//
// Defaults: 128³ RGBA16F = 16 MB. Plenty for cumulus at horizontalScale =
// 0.039 (one period ≈ 3.3K world units). Bumping to 256³ costs 128 MB —
// not worth it unless we add hi-freq detail channels.
//
// ── TROUBLESHOOTING ──────────────────────────────────────────────────────
//
// "Clouds are all-black, all-white, or flicker as garbage on first frame"
//     The bake didn't complete (or didn't run) before the cloud kernel
//     read the volume. The init relies on same-MTLCommandQueue ordering
//     (see `init` for the contract). If that assumption ever breaks —
//     e.g. the cloud renderer is switched to its own command queue, or
//     a future change has the cloud kernel commit on a different queue
//     than the bake — the cloud kernel reads uninitialized GPU memory
//     and produces nonsense. Fix: add `cmd.waitUntilCompleted()` after
//     `cmd.commit()` in `init` to block until the bake finishes (one-
//     time cost at scene load, ~50 ms; not in the steady-state budget).
//
// "Clouds look right but there's a visible repeating pattern in the sky"
//     The tile is wrapping at a scale the viewer can see. Either
//     `horizontalScale` is too large (bigger = smaller-looking clouds =
//     more periods visible) or `tileSize` is too small for the scene's
//     visible XZ extent. At the default 128 tile and hScl = 0.039, one
//     period covers ~3.3K world units — invisible. If a scene needs
//     larger view distances, bump `tileSize` to 256 (8× memory cost).
//
// "noiseBake pipeline missing" in the log
//     `Shaders/NoiseBake.metal` failed to compile or wasn't included in
//     the package's metallib build. The texture exists but contains
//     undefined GPU memory; the cloud kernel will produce broken output
//     (see first symptom above). Check the Metal compile log in Xcode.

@MainActor
public final class NoiseVolume {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "NoiseVolume")

    /// Tileable 3D RGBA16F volume. R = fbm, G = erosion (1 - worley at 4×).
    /// Bind at `[[texture(1)]]` on the volumetric sky kernel.
    public let texture: MTLTexture

    /// Edge length of the cube. Cloud kernel samples at `(q / tileSize)`
    /// under `address::repeat`.
    public let tileSize: Int

    public init(engine: SimEngine = .shared, tileSize: Int = 128) {
        precondition(tileSize > 0 && tileSize <= 512, "NoiseVolume: tileSize out of range")
        self.tileSize = tileSize

        let device = engine.device
        let d = MTLTextureDescriptor()
        d.textureType = .type3D
        d.pixelFormat = .rgba16Float
        d.width = tileSize
        d.height = tileSize
        d.depth = tileSize
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        guard let tex = device.makeTexture(descriptor: d) else {
            preconditionFailure("NoiseVolume: makeTexture failed")
        }
        tex.label = "NoiseVolume.\(tileSize)cubed"
        self.texture = tex

        guard let pipeline = engine.pipelineCache.pipelineState(
            name: "noiseBake",
            device: device
        ) else {
            // Failure mode: texture exists but contains undefined GPU
            // memory because the bake never ran. Downstream symptom is
            // the cloud renderer producing all-black or flickering
            // garbage clouds. The log line is the breadcrumb — check
            // the Xcode Metal compile output for `Shaders/NoiseBake.metal`.
            Self.log.error("noiseBake pipeline missing — Shaders/NoiseBake.metal must compile")
            return
        }

        guard let cmd = engine.commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            // Same downstream symptom as the pipeline-missing case
            // above. Usually a device-out-of-memory situation; the
            // 16 MB texture allocation above should have already
            // failed if that were the cause, so this branch is
            // effectively unreachable in practice.
            Self.log.error("NoiseVolume: failed to acquire command/encoder for bake")
            return
        }
        cmd.label = "NoiseVolume.bake"
        enc.label = "noiseBake"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(tex, index: 0)
        var ts = UInt32(tileSize)
        enc.setBytes(&ts, length: MemoryLayout<UInt32>.size, index: 0)

        let tgw = min(pipeline.threadExecutionWidth, tileSize)
        let tgh = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / max(1, tgw), tileSize))
        let tgd = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / max(1, tgw * tgh), tileSize))
        let tg = MTLSize(width: tgw, height: tgh, depth: tgd)
        let grid = MTLSize(width: tileSize, height: tileSize, depth: tileSize)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        // Bake runs asynchronously on the GPU. We do NOT block with
        // `cmd.waitUntilCompleted()` because the contract here is that
        // the cloud renderer dispatches `volSkyRender` on the SAME
        // `engine.commandQueue`, and a single MTLCommandQueue executes
        // its command buffers in submission order — so by the time the
        // first cloud-kernel dispatch starts executing, the bake is
        // guaranteed to have finished. Skipping the CPU wait saves the
        // one-time ~50 ms blocking cost at scene load.
        //
        // IF YOU SEE BROKEN CLOUDS ON THE FIRST FRAME: the most likely
        // cause is that this contract has been broken — e.g. a future
        // refactor gave the cloud renderer its own queue, or moved the
        // bake to a different queue, or split the cloud kernel across
        // queues for some reason. The fix is either:
        //   1. Restore same-queue ordering (preferred), or
        //   2. Add `cmd.waitUntilCompleted()` on the line below — it
        //      makes init block until the bake finishes, which is
        //      always-correct at the cost of ~50 ms at scene load.
        // The same-queue assumption is verified by inspection at
        // `VolumetricCloudRenderer.init` where both `noiseVolume` and
        // `commandQueue` come from the same `engine` parameter.
    }
}
