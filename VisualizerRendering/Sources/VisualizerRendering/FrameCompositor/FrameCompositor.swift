import AppKit
import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── FRAME COMPOSITOR ────────────────────────────────────────────────────────
//
// A reusable "custom Metal renderer runs alongside SceneKit" scaffolding.
// Owns the offscreen HDR colour + depth attachments that scene passes write
// into, plus the bloom + ACES tonemap chain that resolves them into a final
// LDR texture the host displays via a camera-attached SceneKit quad.
//
// Same architectural pattern as `IlluminatoramaRenderer` (see issue #17 and
// `docs/illuminatorama/README.md`), but stripped to the basics that any
// scene-agnostic custom-Metal renderer needs. IlluminatoramaRenderer is
// deferred PBR for opaque meshes — wrong shape for forward-additive
// particle scenes like Fireworks+. This module is the shared substrate
// either kind of renderer can sit on top of.
//
// REUSABLE BY: Fireworks+ (particles + raymarched cloud), Napa Valley
// (volumetric balloons + sky), Dust (#28), Smoke, future Dayone sky
// renderer (#18). Any scene that wants to escape SceneKit's opaque
// forward pipeline and do its own thing.
//
// ── FRAME LIFECYCLE ─────────────────────────────────────────────────────────
//
//   let cb = engine.commandQueue.makeCommandBuffer()!
//
//   // 1. Clear HDR + depth, prepare attachments
//   compositor.beginFrame(into: cb, clearColor: skyColor)
//
//   // 2. Scene passes write to compositor.hdrColorTexture + depthTexture
//   //    via their own render encoders (passes own their pipelines).
//   particlePass.encode(into: cb, target: compositor)
//   cloudPass.encode(into: cb, target: compositor)
//   ...
//
//   // 3. Compose: bloom → tonemap → displayTexture (LDR)
//   compositor.composite(into: cb)
//
//   cb.commit()
//
//   // SceneKit's camera-facing quad has its diffuse bound to
//   // `compositor.displayTexture`, so the next SCN render shows the result.
//
// ── PASSES THAT WRITE INTO THE COMPOSITOR ───────────────────────────────────
//
// A pass takes the compositor as a render target (`target.hdrColorTexture`
// + `target.depthTexture`) and creates its own MTLRenderCommandEncoder
// with appropriate clear/load actions. The compositor's beginFrame has
// already done a single clear; later passes use load=`.load` to add on top.
//
// All passes share the depth attachment so they can occlude one another
// correctly. This is the whole reason for this module — it's how the
// cloud-vs-shell occlusion that screen-space overlays can't do becomes
// trivial. Every pass writes/reads the same depth buffer.

@MainActor
public final class FrameCompositor {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "FrameCompositor")

    public let device: MTLDevice
    public let engine: SimEngine
    public private(set) var size: SIMD2<Int>

    // ── Render targets ──────────────────────────────────────────────────────

    /// HDR colour attachment scene passes write into. RGBA16Float for HDR
    /// headroom — bloom/exposure needs values well above 1.
    public private(set) var hdrColorTexture: MTLTexture

    /// Depth attachment scene passes share so they occlude one another
    /// correctly. depth32Float for precision over large scene depth ranges.
    public private(set) var depthTexture: MTLTexture

    /// Final LDR output the host's SceneKit quad samples as its diffuse.
    /// rgba8Unorm to match what SCN expects from a material texture; bloom
    /// + ACES tonemap have already compressed HDR → LDR by this point.
    public private(set) var displayTexture: MTLTexture

    // ── Post-process tunables ───────────────────────────────────────────────

    public var bloomThreshold: Float = 1.0     // HDR luminance over this gets bloomed
    public var bloomIntensity: Float = 0.55    // Bloom layer's multiplier when added back
    public var exposure: Float = 1.0           // Scene exposure stop (linear multiplier)

    // ── Pipelines ───────────────────────────────────────────────────────────

    private let bloomThresholdPipeline: MTLComputePipelineState?
    private let bloomBlurHPipeline: MTLComputePipelineState?
    private let bloomBlurVPipeline: MTLComputePipelineState?
    private let tonemapPipeline: MTLComputePipelineState?

    // Half-res scratch textures for the bloom chain.
    private var bloomScratchA: MTLTexture
    private var bloomScratchB: MTLTexture

    /// 2-slot dispatch semaphore guarding the composite pass. Same pattern
    /// PR #35 introduced on VolumetricCloudRenderer to prevent the Metal
    /// command-queue saturation freeze (see
    /// [metal_command_queue_saturation.md]). Lets up to 2 composite passes
    /// be in-flight; further calls non-blocking-skip. The completion
    /// handler signals the semaphore so the next call has a slot.
    private let compositeSem = DispatchSemaphore(value: 2)

    // ── Construction ────────────────────────────────────────────────────────

    public init?(engine: SimEngine, size: SIMD2<Int>) {
        precondition(size.x > 0 && size.y > 0, "FrameCompositor: bad size")
        self.device = engine.device
        self.engine = engine
        self.size = size

        guard
            let hdr = Self.makeTexture(
                device: device, width: size.x, height: size.y,
                format: .rgba16Float,
                usage: [.renderTarget, .shaderRead, .shaderWrite],
                label: "FrameCompositor.hdr"
            ),
            let depth = Self.makeTexture(
                device: device, width: size.x, height: size.y,
                format: .depth32Float,
                usage: [.renderTarget, .shaderRead],
                label: "FrameCompositor.depth"
            ),
            let display = Self.makeTexture(
                device: device, width: size.x, height: size.y,
                format: .rgba8Unorm,
                usage: [.shaderRead, .shaderWrite],
                label: "FrameCompositor.display"
            ),
            let bloomA = Self.makeTexture(
                device: device, width: size.x / 2, height: size.y / 2,
                format: .rgba16Float,
                usage: [.shaderRead, .shaderWrite],
                label: "FrameCompositor.bloomScratchA"
            ),
            let bloomB = Self.makeTexture(
                device: device, width: size.x / 2, height: size.y / 2,
                format: .rgba16Float,
                usage: [.shaderRead, .shaderWrite],
                label: "FrameCompositor.bloomScratchB"
            )
        else {
            Self.log.error("FrameCompositor texture allocation failed")
            return nil
        }
        self.hdrColorTexture = hdr
        self.depthTexture = depth
        self.displayTexture = display
        self.bloomScratchA = bloomA
        self.bloomScratchB = bloomB

        // Compute pipelines for the bloom + tonemap chain. All read/write
        // the rgba16Float scratch textures except the final tonemap which
        // writes the LDR display texture.
        self.bloomThresholdPipeline = engine.pipeline("fcBloomThreshold")
        self.bloomBlurHPipeline = engine.pipeline("fcBloomBlurH")
        self.bloomBlurVPipeline = engine.pipeline("fcBloomBlurV")
        self.tonemapPipeline = engine.pipeline("fcTonemap")

        for (pl, name) in [
            (bloomThresholdPipeline, "fcBloomThreshold"),
            (bloomBlurHPipeline, "fcBloomBlurH"),
            (bloomBlurVPipeline, "fcBloomBlurV"),
            (tonemapPipeline, "fcTonemap"),
        ] where pl == nil {
            Self.log.error("FrameCompositor pipeline missing: \(name) — is FrameCompositor.metal in Shaders/?")
        }
    }

    /// Reallocate render-target textures at a new size. Cheap — drops the
    /// previous textures and makes fresh ones. Call when the host SCNView's
    /// drawable size changes (window resize).
    public func resize(to newSize: SIMD2<Int>) {
        guard newSize.x > 0, newSize.y > 0, newSize != size else { return }
        self.size = newSize
        guard
            let hdr = Self.makeTexture(
                device: device, width: newSize.x, height: newSize.y,
                format: .rgba16Float,
                usage: [.renderTarget, .shaderRead, .shaderWrite],
                label: "FrameCompositor.hdr"
            ),
            let depth = Self.makeTexture(
                device: device, width: newSize.x, height: newSize.y,
                format: .depth32Float,
                usage: [.renderTarget, .shaderRead],
                label: "FrameCompositor.depth"
            ),
            let display = Self.makeTexture(
                device: device, width: newSize.x, height: newSize.y,
                format: .rgba8Unorm,
                usage: [.shaderRead, .shaderWrite],
                label: "FrameCompositor.display"
            ),
            let bloomA = Self.makeTexture(
                device: device, width: newSize.x / 2, height: newSize.y / 2,
                format: .rgba16Float,
                usage: [.shaderRead, .shaderWrite],
                label: "FrameCompositor.bloomScratchA"
            ),
            let bloomB = Self.makeTexture(
                device: device, width: newSize.x / 2, height: newSize.y / 2,
                format: .rgba16Float,
                usage: [.shaderRead, .shaderWrite],
                label: "FrameCompositor.bloomScratchB"
            )
        else {
            Self.log.error("FrameCompositor resize allocation failed")
            return
        }
        self.hdrColorTexture = hdr
        self.depthTexture = depth
        self.displayTexture = display
        self.bloomScratchA = bloomA
        self.bloomScratchB = bloomB
    }

    // ── Frame lifecycle ─────────────────────────────────────────────────────

    /// Clear the HDR colour + depth attachments to start a new frame. The
    /// host's scene passes follow this with their own render encoders that
    /// `loadAction = .load` against these targets, building up the scene
    /// additively (or with proper depth-tested opaque writes).
    public func beginFrame(into cb: MTLCommandBuffer, clearColor: SIMD4<Float>) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = hdrColorTexture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(
            red:   Double(clearColor.x),
            green: Double(clearColor.y),
            blue:  Double(clearColor.z),
            alpha: Double(clearColor.w)
        )
        pass.depthAttachment.texture = depthTexture
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = 1.0
        // Trivial empty pass just to perform the clear; scene passes
        // follow with their own encoders that load these attachments.
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "FrameCompositor.clear"
        enc.endEncoding()
    }

    /// Run the bloom + tonemap chain against the HDR colour, writing the
    /// LDR result into `displayTexture`. Call after all scene passes have
    /// finished writing the HDR target.
    ///
    /// Non-blocking back-pressure guard: if 2 composite passes are already
    /// in-flight, this returns without dispatching to avoid saturating
    /// Metal's command-queue slots (the documented hang mode in
    /// [metal_command_queue_saturation.md]). The display texture keeps its
    /// previous-frame contents — visually a dropped frame, never a hang.
    public func composite(into cb: MTLCommandBuffer) {
        guard let bloomThresholdPipeline,
              let bloomBlurHPipeline,
              let bloomBlurVPipeline,
              let tonemapPipeline else { return }

        // Acquire a semaphore slot (non-blocking). If 2 composites are
        // already in flight, skip this frame's bloom/tonemap.
        guard compositeSem.wait(timeout: .now()) == .success else {
            Self.log.notice("FrameCompositor.composite skipped — back-pressure (2 in flight)")
            return
        }
        // Capture the semaphore in the completion handler so it's signalled
        // when the GPU finishes, freeing a slot for the next call.
        let capturedSem = compositeSem
        cb.addCompletedHandler { _ in capturedSem.signal() }

        // 1. Bloom threshold downsample: HDR full-res → half-res bright-only.
        do {
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.label = "FrameCompositor.bloomThreshold"
            enc.setComputePipelineState(bloomThresholdPipeline)
            enc.setTexture(hdrColorTexture, index: 0)
            enc.setTexture(bloomScratchA,   index: 1)
            var threshold = bloomThreshold
            enc.setBytes(&threshold, length: MemoryLayout<Float>.size, index: 0)
            let tg = MTLSize(width: 8, height: 8, depth: 1)
            let grid = MTLSize(width: bloomScratchA.width, height: bloomScratchA.height, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            enc.endEncoding()
        }

        // 2. Horizontal blur: scratchA → scratchB.
        do {
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.label = "FrameCompositor.bloomBlurH"
            enc.setComputePipelineState(bloomBlurHPipeline)
            enc.setTexture(bloomScratchA, index: 0)
            enc.setTexture(bloomScratchB, index: 1)
            let tg = MTLSize(width: 8, height: 8, depth: 1)
            let grid = MTLSize(width: bloomScratchB.width, height: bloomScratchB.height, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            enc.endEncoding()
        }

        // 3. Vertical blur: scratchB → scratchA (final bloom layer).
        do {
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.label = "FrameCompositor.bloomBlurV"
            enc.setComputePipelineState(bloomBlurVPipeline)
            enc.setTexture(bloomScratchB, index: 0)
            enc.setTexture(bloomScratchA, index: 1)
            let tg = MTLSize(width: 8, height: 8, depth: 1)
            let grid = MTLSize(width: bloomScratchA.width, height: bloomScratchA.height, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            enc.endEncoding()
        }

        // 4. Tonemap: HDR + bloom layer → LDR display. Compute kernel reads
        //    both, upsamples the half-res bloom bilinearly, adds it to the
        //    HDR with bloomIntensity, applies exposure + ACES, writes
        //    rgba8 to the display texture.
        do {
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.label = "FrameCompositor.tonemap"
            enc.setComputePipelineState(tonemapPipeline)
            enc.setTexture(hdrColorTexture, index: 0)
            enc.setTexture(bloomScratchA,   index: 1)
            enc.setTexture(displayTexture,  index: 2)
            var params = SIMD2<Float>(exposure, bloomIntensity)
            enc.setBytes(&params, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
            let tg = MTLSize(width: 8, height: 8, depth: 1)
            let grid = MTLSize(width: displayTexture.width, height: displayTexture.height, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            enc.endEncoding()
        }
    }

    // ── Utilities ───────────────────────────────────────────────────────────

    private static func makeTexture(
        device: MTLDevice,
        width: Int,
        height: Int,
        format: MTLPixelFormat,
        usage: MTLTextureUsage,
        label: String
    ) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format,
            width: max(1, width),
            height: max(1, height),
            mipmapped: false
        )
        d.usage = usage
        d.storageMode = .private
        let tex = device.makeTexture(descriptor: d)
        tex?.label = label
        return tex
    }

    // ── Render-pass descriptor helpers for scene-pass writers ───────────────

    /// Build a render-pass descriptor configured to write into the
    /// compositor's HDR colour + depth attachments. Use `loadAction = .load`
    /// so writes accumulate on top of the cleared / previously-written
    /// targets. Pass writers call this, then add their own pipeline state
    /// and draw calls.
    public func makeRenderPassDescriptor() -> MTLRenderPassDescriptor {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = hdrColorTexture
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depthTexture
        pass.depthAttachment.loadAction = .load
        pass.depthAttachment.storeAction = .store
        return pass
    }
}
