import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - SelectionState

/// Selection outline pass: mask, dilation, composite.
/// Owned by the renderer as `outl`; every member keeps the name it had on the renderer.
@MainActor
final class SelectionState {
    // ── Highlight outline pipeline state ─────────────────────────────
    var selectionBoxMaskPipeline: MTLRenderPipelineState?  // mask boxes, filtered by wantMode
    var selectionDilateHPipeline: MTLComputePipelineState?  // horizontal max-filter
    var selectionDilateVPipeline: MTLComputePipelineState?  // vertical max-filter
    var selectionCompositePipeline: MTLComputePipelineState?  // ring → HDR additive composite
    var selectionMaskTexture: MTLTexture?  // 0/1 per pixel (mask boxes rendered here)
    var selectionDilateHTexture: MTLTexture?  // horizontal-dilated intermediate
    var selectionDilatedTexture: MTLTexture?  // fully dilated result
    var selectionMaskInstanceBuffer: MTLBuffer?  // GPU copy of `highlightMaskInstances`
    var selectionDepthState: MTLDepthStencilState?
}

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    /// The look of ONE highlight mode's halo. Selection and hover differ in nothing but these
    /// three numbers, so they are one type with two values — not two parallel sets of properties
    /// that drift apart the moment a third mode (locked? conflicted?) shows up.
    public struct HighlightOutlineStyle: Equatable, Sendable {
        /// RGB in linear space, pre-intensity. This is what makes the mode identifiable at a
        /// glance, so it only survives if `intensity` keeps the ring off the clipping point.
        public var color: SIMD3<Float>
        /// How far the halo reaches from the silhouette, in **output** pixels — what the viewer
        /// actually sees, at any internal render scale (`resolvedOutlineRadius`). The ring ramps
        /// from full intensity against the element to nothing at this distance.
        public var width: Float
        /// PEAK additive brightness of the ring, in tonemapper-input units: 1.0 lands just under
        /// the ACES shoulder, so ~0.9 is a strong glow that still carries its hue and ~0.5 is a
        /// quiet hint. The composite divides out the frame's exposure, so this is a *displayed*
        /// brightness — the same in a night scene and a noon one.
        ///
        /// It used to be a hardcoded 7.0, chosen to shove the ring past `bloomThreshold` and let
        /// bloom do the feathering. That is what made the halo read as a hard WHITE band (7×
        /// anything with a channel at 1.0 clips long before the shoulder, so blue and amber were
        /// the same colour where it counted) sitting in a wash of its own bloom across the
        /// enclosed object — measured at 23–27% of the frame repainted around one sofa. The ring
        /// now feathers itself, so it does not need to be bloom-bait.
        public var intensity: Float

        public init(color: SIMD3<Float>, width: Float, intensity: Float) {
            self.color = color; self.width = width; self.intensity = intensity
        }

        /// A committed choice — the brighter, wider of the two.
        public static let selection = HighlightOutlineStyle(
            color: SIMD3(0.0, 0.7, 1.0), width: 3.0, intensity: 0.9)
        /// A passing hint. Deliberately quieter than `.selection`: the pointer sweeps across
        /// elements constantly, and a hover that shouts as loud as a selection makes the canvas
        /// flicker as you move the mouse.
        public static let hover = HighlightOutlineStyle(
            color: SIMD3(1.0, 0.78, 0.12), width: 2.0, intensity: 0.55)
    }

    /// A style's `width` in INTERNAL pixels — the space the mask textures live in. Never below 1,
    /// or the dilation degenerates and the halo disappears at low render scales.
    ///
    /// The width is authored in OUTPUT pixels and used to be consumed as-is against the mask
    /// textures, which are sized at the INTERNAL resolution — so the halo's apparent thickness was
    /// `width / internalRenderScale` and moved with a lever that has nothing to do with selection.
    /// A host that drops the render scale during a gesture — which Daydream Home does,
    /// edge-triggered on gesture start — got a halo that visibly FATTENS the instant you grab the
    /// thing it is drawn around and snaps thin when you let go. Across that app's shipped tiers one
    /// setting of 6 landed anywhere from 4.0 to 10.9 output px, a 2.7× spread, at its most obvious
    /// precisely when the outline is on screen.
    public func resolvedOutlineRadius(_ style: HighlightOutlineStyle) -> Int32 {
        Int32(max(1, (max(1, style.width) * internalRenderScale).rounded()))
    }

    /// Swift mirror of the Metal `SelectionOutlineParams` struct (must match byte-for-byte).
    struct SelectionOutlineParams {
        var colorIntensity: SIMD4<Float>
        var width: Int32; var height: Int32
        var ringGain: Float; var manualExposure: Float
        var autoExposure: Int32
        var _pad0: Int32 = 0
    }

    /// Feathered halo glow around selected / hovered objects. Runs after DOF so
    /// the outline stays sharp even when the object is out of focus; before bloom
    /// so the halo edge blooms into a soft glow.
    ///
    /// Each highlighted element supplies a SOLID bounding-box proxy via
    /// `highlightMaskInstances`, tagged `highlight == 1` (selected) or `2` (hover).
    /// The pass runs once per mode: rasterize the matching boxes into a 1-channel
    /// R8 mask, separable max-filter dilate, then composite `dilated − original`
    /// (a clean outer ring) additively into the HDR source bloom reads — in the
    /// selected colour, then the hover colour. Boxes (never the detailed meshes)
    /// keep the mask hole-free, so the ring traces the outline without internal
    /// edges or fill. Cull `.none` so a box's winding can never sparsen the mask.
    func encodeSelectionOutlinePass(_ cb: MTLCommandBuffer) {
        guard selectionOutlineEnabled else { return }
        guard let boxPipeline     = outl.selectionBoxMaskPipeline,
              let dilateHPipeline = outl.selectionDilateHPipeline,
              let dilateVPipeline = outl.selectionDilateVPipeline,
              let compPipeline    = outl.selectionCompositePipeline,
              let depthSt         = outl.selectionDepthState,
              let boxMesh         = meshes[.box] else { return }
        guard !highlightMaskInstances.isEmpty else { return }

        // Lazily allocate / resize mask textures at the internal render resolution.
        let w = gbs.depthTexture.width, h = gbs.depthTexture.height
        if outl.selectionMaskTexture == nil || outl.selectionMaskTexture!.width != w || outl.selectionMaskTexture!.height != h {
            let td = MTLTextureDescriptor()
            td.textureType = .type2D
            td.pixelFormat = .r8Unorm
            td.width = max(1, w); td.height = max(1, h)
            td.usage = [.shaderRead, .shaderWrite, .renderTarget]
            td.storageMode = .private
            outl.selectionMaskTexture   = device.makeTexture(descriptor: td)
            outl.selectionDilateHTexture = device.makeTexture(descriptor: td)
            outl.selectionDilatedTexture = device.makeTexture(descriptor: td)
            outl.selectionMaskTexture?.label    = "Illuminatorama.selectionMask"
            outl.selectionDilateHTexture?.label = "Illuminatorama.selectionDilateH"
            outl.selectionDilatedTexture?.label = "Illuminatorama.selectionDilated"
        }
        guard let maskTex = outl.selectionMaskTexture,
              let dilHTex = outl.selectionDilateHTexture,
              let dilTex  = outl.selectionDilatedTexture else { return }

        // Upload the mask boxes once; both mode passes index the same buffer and
        // the VS clips boxes whose `highlight` tag ≠ the mode being drawn.
        let instStride = MemoryLayout<IlluminatoramaInstance>.stride
        let needed = instStride * highlightMaskInstances.count
        if outl.selectionMaskInstanceBuffer == nil || outl.selectionMaskInstanceBuffer!.length < needed {
            outl.selectionMaskInstanceBuffer = device.makeBuffer(
                length: max(needed, instStride * 64), options: .storageModeShared)
            outl.selectionMaskInstanceBuffer?.label = "Illuminatorama.highlightMaskInstances"
        }
        guard let instBuf = outl.selectionMaskInstanceBuffer else { return }
        instBuf.contents().copyMemory(from: highlightMaskInstances, byteCount: needed)
        let boxCount = highlightMaskInstances.count

        // Mode 1 = selected (blue); mode 2 = hover (yellow). Skip a mode with no
        // boxes so we don't pay for an all-clear composite.
        let modes: [(mode: Int32, style: HighlightOutlineStyle)] = [
            (1, selectionOutline),
            (2, hoverOutline),
        ]

        for m in modes {
            guard highlightMaskInstances.contains(where: { $0.highlight == m.mode }) else { continue }
            // Each mode carries its own width, so the radius is per-pass, not per-frame.
            var radius = resolvedOutlineRadius(m.style)

            // ── Render pass: rasterize this mode's boxes into the mask ─────────
            let rp = MTLRenderPassDescriptor()
            rp.colorAttachments[0].texture      = maskTex
            rp.colorAttachments[0].loadAction   = .clear
            rp.colorAttachments[0].storeAction  = .store
            rp.colorAttachments[0].clearColor   = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            rp.depthAttachment.texture          = gbs.depthTexture
            rp.depthAttachment.loadAction       = .load    // G-buffer depth → occlusion
            rp.depthAttachment.storeAction      = .dontCare // read-only, don't modify

            guard let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { return }
            enc.label = "Illuminatorama.highlightMask.\(m.mode)"
            enc.setDepthStencilState(depthSt)
            enc.setCullMode(.none)   // a box is convex; depth test alone resolves the silhouette
            enc.setRenderPipelineState(boxPipeline)
            var wantMode = m.mode
            enc.setVertexBuffer(boxMesh.vertexBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(frameUniformBuffer,   offset: 0, index: 1)
            enc.setVertexBuffer(instBuf,              offset: 0, index: 2)
            enc.setVertexBytes(&wantMode, length: MemoryLayout<Int32>.stride, index: 3)
            enc.drawIndexedPrimitives(
                type: .triangle, indexCount: boxMesh.indexCount, indexType: boxMesh.indexType,
                indexBuffer: boxMesh.indexBuffer, indexBufferOffset: 0,
                instanceCount: boxCount, baseVertex: 0, baseInstance: 0)
            enc.endEncoding()

            // ── Compute: separable max-filter dilation → ring composite ───────
            guard let comp = cb.makeComputeCommandEncoder() else { return }
            comp.label = "Illuminatorama.highlightOutline.\(m.mode)"

            comp.setComputePipelineState(dilateHPipeline)
            comp.setTexture(maskTex, index: 0)
            comp.setTexture(dilHTex, index: 1)
            comp.setBytes(&radius, length: MemoryLayout<Int32>.stride, index: 0)
            dispatch(comp, pipeline: dilateHPipeline, width: w, height: h)

            comp.setComputePipelineState(dilateVPipeline)
            comp.setTexture(dilHTex, index: 0)
            comp.setTexture(dilTex,  index: 1)
            comp.setBytes(&radius, length: MemoryLayout<Int32>.stride, index: 0)
            dispatch(comp, pipeline: dilateVPipeline, width: w, height: h)

            // Ring → additive composite. `bloomTonemapSource` is dofOutputTexture
            // (if DOF ran) or displaySource. The dilation above already feathered the
            // ring, so the intensity is a look, not a lever for reaching the bloom
            // threshold — and staying under it is what keeps the halo off the rest of
            // the frame. `exposureBuffer` lets the kernel divide out the very exposure
            // the tonemap will apply (the estimator ran earlier in this command buffer).
            var params = SelectionOutlineParams(
                colorIntensity: SIMD4(m.style.color.x, m.style.color.y, m.style.color.z,
                                      m.style.intensity),
                width: Int32(w), height: Int32(h),
                ringGain: Float(radius + 1) / Float(radius),
                manualExposure: exposure,
                autoExposure: autoExposureEnabled ? 1 : 0)
            comp.setComputePipelineState(compPipeline)
            comp.setTexture(maskTex,            index: 0)
            comp.setTexture(dilTex,             index: 1)
            comp.setTexture(bloomTonemapSource, index: 2)
            comp.setBytes(&params, length: MemoryLayout<SelectionOutlineParams>.stride, index: 0)
            comp.setBuffer(pfx.exposureBuffer, offset: 0, index: 1)
            dispatch(comp, pipeline: compPipeline, width: w, height: h)

            comp.endEncoding()
        }
    }
}

// MARK: - Pipeline / resource construction (called from `init`, in the order it lists)

extension IlluminatoramaRenderer {
    /// Highlight outline pipelines (optional: silently off when the shaders are absent).
    @MainActor static func buildSelectionPipelines(into s: SelectionState, cache: SimPipelineCache, device: MTLDevice, library: MTLLibrary) {
        // ── Highlight outline pipelines (optional — silently no-ops if shaders absent) ──
        if let vs = library.makeFunction(name: "illumi_selection_box_vs"),
           let fs = library.makeFunction(name: "illumi_selection_mask_fs") {
            let d = MTLRenderPipelineDescriptor()
            d.label = "Illuminatorama.selectionBoxMask"
            d.vertexFunction = vs; d.fragmentFunction = fs
            d.colorAttachments[0].pixelFormat = .r8Unorm
            d.depthAttachmentPixelFormat = .depth32Float
            s.selectionBoxMaskPipeline = try? device.makeRenderPipelineState(descriptor: d) // gpu-ok: optional
        }
        s.selectionDilateHPipeline = cache.pipelineState(name: "illumi_selection_dilate_h", device: device)
        s.selectionDilateVPipeline = cache.pipelineState(name: "illumi_selection_dilate_v", device: device)
        s.selectionCompositePipeline = cache.pipelineState(name: "illumi_selection_composite", device: device)
        let selDepthDesc = MTLDepthStencilDescriptor()
        selDepthDesc.depthCompareFunction = .lessEqual
        selDepthDesc.isDepthWriteEnabled = false
        s.selectionDepthState = device.makeDepthStencilState(descriptor: selDepthDesc)
    }
}
