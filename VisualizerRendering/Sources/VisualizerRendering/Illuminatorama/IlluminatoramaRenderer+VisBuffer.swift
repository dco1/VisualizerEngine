import Foundation
import Metal
import OSLog
import simd

/// What the visibility-buffer resolve paints instead of the material (debug views).
public enum IlluminatoramaVisBufferDebug: UInt32, Sendable {
    case off = 0
    /// Each Signack cluster its own neon (the cut's per-triangle cluster id).
    case clusters = 1
    /// Each triangle its own neon.
    case triangles = 2
    /// Each opted-in draw (mesh) its own neon.
    case draws = 3
}

/// One opted-in draw as the resolve reads it — mirror of `VisBufferDraw` (IlluminatoramaVisBuffer.metal).
struct IlluminatoramaVisBufferDraw {
    var vertices: UInt64
    var indices: UInt64
    var clusterIDs: UInt64
    var instanceBase: UInt32
    var pad: UInt32 = 0
}

/// Mirror of `VisBufferResolveParams` (16 B).
struct IlluminatoramaVisBufferResolveParams {
    var targetSize: SIMD2<Float>
    var debugMode: UInt32
    var pad: UInt32 = 0
}

/// Signack S3 (Daydream DH-1461) — the VISIBILITY BUFFER: state for the draws in this file.
///
/// Meshes with `IlluminatoramaMesh.visBuffer` skip the conventional draw loop of the G-buffer pass.
/// At the END of that same render pass (after every conventional and impostor draw), they are
/// (a) rasterised into a memoryless id attachment (color 6) with the pass's own depth test, every
/// G-buffer attachment write-masked off, then (b) a fullscreen resolve reads each pixel's id from
/// tile memory (Apple-GPU programmable blending), reconstructs the surface where an id won, and
/// shades it with the conventional pass's own material function into the ordinary G-buffer
/// targets. One pass, no extra G-buffer load/store, nothing downstream changes; material cost is
/// once per covered pixel. Every G-buffer pipeline declares the id attachment (write-masked) so
/// the pass and its pipelines always agree (`declareVisBufferAttachment`). Shaders:
/// IlluminatoramaVisBuffer.metal.
final class VisBufferState {
    /// Host lever (A/B). Default OFF: on Apple GPUs hidden-surface removal already shades opaque
    /// foliage once per pixel, and the resolve (three vertex fetches + re-derived wind per pixel,
    /// over a SIMD group that mixes bark and leaf pixels) measured 1–3.5 ms slower on tree frames
    /// (DH-1461 S3). Env `ILLUMINATORAMA_VISBUFFER=1` / `=0` overrides it either way.
    var enabled = false
    var debug: IlluminatoramaVisBufferDebug = .off
    /// Overdraw instrument: count depth-passing id fragments and resolved pixels per frame.
    var countsOverdraw = false
    /// Pipelines keyed by (extended G-buffer, counting).
    var raster: [Int: MTLRenderPipelineState] = [:]
    var resolve: [Int: MTLRenderPipelineState] = [:]
    var resolveDepth: MTLDepthStencilState?
    var idTarget: MTLTexture?
    var zeroArgs: MTLBuffer?
    var counters: MTLBuffer?
    /// Set when Metal refused a pipeline or the GPU cannot read its tile — the path stays off.
    var unavailable = false
    /// The resolve reads its draw table with setFragmentBytes (≤ 4 KB).
    static let maxDraws = 4096 / MemoryLayout<IlluminatoramaVisBufferDraw>.stride
    static let log = Logger(subsystem: "Illuminatorama", category: "VisBuffer")
}

/// A draw group the G-buffer pass handed to the visibility buffer.
struct VisBufferDeferredDraw {
    let group: IlluminatoramaRenderer.MeshDrawGroup
    let mesh: IlluminatoramaMesh
    /// The group's world bounds (DH-0534's cull bounds) — the resolve is scissored to their union.
    let bounds: IlluminatoramaCullBounds
}

extension IlluminatoramaRenderer {
    /// A/B lever for Signack's visibility buffer (default OFF — see `VisBufferState.enabled` for the
    /// measurement). Off ⇒ opted-in meshes draw through the conventional G-buffer pass as before S3.
    public var visBufferEnabled: Bool {
        get { vbs.enabled }
        set { vbs.enabled = newValue }
    }
    /// Debug view: paint the resolved pixels by cluster / triangle / draw instead of their material.
    public var visBufferDebug: IlluminatoramaVisBufferDebug {
        get { vbs.debug }
        set { vbs.debug = newValue }
    }
    /// Overdraw instrument (test-only cost: an atomic per fragment). Read `visBufferOverdraw` after
    /// the frame completed.
    public var visBufferCountsOverdraw: Bool {
        get { vbs.countsOverdraw }
        set { vbs.countsOverdraw = newValue }
    }
    /// The last completed counting frame: id fragments that passed the depth test (what a
    /// conventional pass without hidden-surface removal would have shaded) and pixels resolved
    /// (what the visibility buffer shaded).
    public var visBufferOverdraw: (fragments: Int, pixels: Int)? {
        guard let c = vbs.counters else { return nil }
        let p = c.contents().bindMemory(to: UInt32.self, capacity: 2)
        return (Int(p[0]), Int(p[1]))
    }

    /// The G-buffer's albedo/metallic and motion-vector targets (internal size; private — blit
    /// them), for parity diagnostics of the visibility buffer against the conventional draw.
    public var gbufferAlbedoMetallicTextureForDiagnostics: MTLTexture { gbs.gbufferAlbedoMet }
    public var gbufferVelocityTextureForDiagnostics: MTLTexture { pfx.velocityTexture }

    /// Whether this device carries the id attachment at all (it needs tile-memory reads and
    /// memoryless targets — every Apple GPU). The G-buffer pipelines and pass both ask this, so
    /// they always agree on the attachment set.
    static func visBufferSupported(_ device: MTLDevice) -> Bool { device.supportsFamily(.apple4) }

    /// Declare the id attachment on a G-buffer-pass pipeline: present, never written by it.
    static func declareVisBufferAttachment(_ d: MTLRenderPipelineDescriptor, device: MTLDevice) {
        guard visBufferSupported(device) else { return }
        d.colorAttachments[6].pixelFormat = .rg32Uint
        d.colorAttachments[6].writeMask = []
    }

    /// Attach the memoryless id target (cleared to 0 = "no visibility-buffer geometry") to this
    /// frame's G-buffer pass. Free when nothing is deferred: memoryless, cleared in tile memory.
    func attachVisBufferTarget(_ pass: MTLRenderPassDescriptor) {
        guard Self.visBufferSupported(device) else { return }
        if vbs.idTarget?.width != width || vbs.idTarget?.height != height {
            let t = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Uint, width: width,
                                                             height: height, mipmapped: false)
            t.usage = .renderTarget
            t.storageMode = .memoryless
            vbs.idTarget = device.makeTexture(descriptor: t)
            vbs.idTarget?.label = "Illuminatorama.visBuffer.ids"
        }
        pass.colorAttachments[6].texture = vbs.idTarget
        pass.colorAttachments[6].loadAction = .clear
        pass.colorAttachments[6].storeAction = .dontCare
        pass.colorAttachments[6].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    }

    /// Whether this frame's opted-in meshes go through the visibility buffer. Decided before the
    /// G-buffer pass skips them, so a missing pipeline can never make a mesh vanish.
    func visBufferActive(extended: Bool) -> Bool {
        let env = Self.frameEnv("ILLUMINATORAMA_VISBUFFER")
        guard env == "1" || (vbs.enabled && env != "0"), !vbs.unavailable, vbs.idTarget != nil else { return false }
        return ensureVisBufferResources(extended: extended)
    }

    /// The G-buffer pass's question: does this group draw through the visibility buffer instead?
    func defersToVisBuffer(_ mesh: IlluminatoramaMesh, kind: MeshKind, count: Int) -> Bool {
        mesh.visBuffer && !mesh.tracedOnly && mesh.indexType == .uint32 && count <= 0xFFFF
            && interiorMappedMeshKinds[kind] == nil
            && !fxs.gpuRepackTasks.contains { $0.kind == kind && !$0.direct }
    }

    private func ensureVisBufferResources(extended: Bool) -> Bool {
        let key = (extended ? 1 : 0) | (vbs.countsOverdraw ? 2 : 0)
        if vbs.raster[key] == nil || vbs.resolve[key] == nil {
            guard Self.visBufferSupported(device), let library = engine.library else {
                vbs.unavailable = true
                return false
            }
            do {
                vbs.raster[key] = try makeVisBufferPipeline(library, resolve: false, extended: extended,
                                                            counting: vbs.countsOverdraw)
                vbs.resolve[key] = try makeVisBufferPipeline(library, resolve: true, extended: extended,
                                                             counting: vbs.countsOverdraw)
            } catch {
                VisBufferState.log.error("visibility buffer unavailable: \(error.localizedDescription)")
                vbs.unavailable = true
                return false
            }
        }
        if vbs.resolveDepth == nil {
            let d = MTLDepthStencilDescriptor()
            d.depthCompareFunction = .always
            d.isDepthWriteEnabled = false
            vbs.resolveDepth = device.makeDepthStencilState(descriptor: d)
        }
        if vbs.zeroArgs == nil {
            vbs.zeroArgs = device.makeBuffer(length: 32, options: .storageModeShared)
            vbs.counters = device.makeBuffer(length: 16, options: .storageModeShared)
            if let z = vbs.zeroArgs { memset(z.contents(), 0, z.length) }
        }
        return vbs.resolveDepth != nil && vbs.zeroArgs != nil
    }

    private func makeVisBufferPipeline(_ library: MTLLibrary, resolve: Bool, extended: Bool,
                                       counting: Bool) throws -> MTLRenderPipelineState {
        let c = MTLFunctionConstantValues()
        var f = false, ext = extended, count = counting
        c.setConstantValue(&f, type: .bool, index: 10)
        c.setConstantValue(&ext, type: .bool, index: 11)
        c.setConstantValue(&f, type: .bool, index: 12)
        c.setConstantValue(&count, type: .bool, index: 20)
        let d = MTLRenderPipelineDescriptor()
        d.label = "Illuminatorama.visBuffer.\(resolve ? "resolve" : "raster")\(extended ? ".ext" : "")"
        d.vertexFunction = try library.makeFunction(name: resolve ? "illumi_vis_resolve_vs" : "illumi_vis_vs",
                                                    constantValues: c)
        d.fragmentFunction = try library.makeFunction(name: resolve ? "illumi_vis_resolve_fs" : "illumi_vis_fs",
                                                      constantValues: c)
        let formats: [MTLPixelFormat] = [.rgba16Float, .rgba16Float, .rgba16Float, .rg16Float, .r32Uint]
            + (extended ? [.rgba16Float] : [])
        for (i, fmt) in formats.enumerated() {
            d.colorAttachments[i].pixelFormat = fmt
            d.colorAttachments[i].writeMask = resolve ? .all : []
        }
        d.colorAttachments[6].pixelFormat = .rg32Uint
        d.colorAttachments[6].writeMask = resolve ? [] : .all
        d.depthAttachmentPixelFormat = .depth32Float
        return try device.makeRenderPipelineState(descriptor: d)
    }

    /// The visibility-buffer draws, at the END of the G-buffer pass's encoder: id raster of `draws`
    /// against the pass's depth, then the fullscreen resolve into the G-buffer targets.
    func encodeVisBuffer(_ enc: MTLRenderCommandEncoder, _ draws: [VisBufferDeferredDraw], extended: Bool,
                         cull passCull: MTLCullMode?) {
        let key = (extended ? 1 : 0) | (vbs.countsOverdraw ? 2 : 0)
        guard !draws.isEmpty, let raster = vbs.raster[key], let resolve = vbs.resolve[key],
              let zero = vbs.zeroArgs else { return }
        if vbs.countsOverdraw, let c = vbs.counters { memset(c.contents(), 0, c.length) }
        enc.pushDebugGroup("Illuminatorama.visBuffer")
        let instStride = MemoryLayout<IlluminatoramaInstance>.stride

        // (a) ids — the G-buffer pass's own depth state and winding.
        enc.setRenderPipelineState(raster)
        enc.setDepthStencilState(gbs.depthState)
        enc.setFrontFacing(.counterClockwise)
        if vbs.countsOverdraw { enc.setFragmentBuffer(vbs.counters, offset: 0, index: 2) }
        var cull = passCull   // tracked like the pass does: a redundant set aborts under validation
        var table: [IlluminatoramaVisBufferDraw] = []
        for (slot, d) in draws.enumerated() {
            let want: MTLCullMode = d.mesh.doubleSided ? .none : .back
            if cull != want { cull = want; enc.setCullMode(want) }
            let off = instStride * d.group.start
            enc.setVertexBuffer(d.mesh.vertexBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
            enc.setVertexBuffer(currentInstanceBuffer, offset: off, index: 2)
            enc.setVertexBuffer(previousInstanceBuffer, offset: off, index: 4)
            var s = UInt32(slot)
            enc.setFragmentBytes(&s, length: 4, index: 0)
            if let args = d.mesh.indirectDrawBuffer {
                enc.setFragmentBuffer(args, offset: d.mesh.indirectDrawOffset, index: 1)
            } else {
                enc.setFragmentBuffer(zero, offset: 0, index: 1)
            }
            enc.drawMesh(d.mesh, instanceCount: d.group.count)
            table.append(IlluminatoramaVisBufferDraw(vertices: d.mesh.vertexBuffer.gpuAddress,
                                                     indices: d.mesh.indexBuffer.gpuAddress,
                                                     clusterIDs: d.mesh.visBufferClusterIDs?.gpuAddress ?? 0,
                                                     instanceBase: UInt32(d.group.start)))
            enc.useResource(d.mesh.vertexBuffer, usage: .read, stages: .fragment)
            enc.useResource(d.mesh.indexBuffer, usage: .read, stages: .fragment)
            if let ids = d.mesh.visBufferClusterIDs { enc.useResource(ids, usage: .read, stages: .fragment) }
        }

        // (b) resolve — one fullscreen triangle reading each pixel's id from tile memory, scissored
        // to the deferred draws' projected bounds: a pixel outside them has no id, and every
        // fragment launched there would still pay the resolve's register footprint to discard.
        enc.setScissorRect(visBufferScissor(draws))
        enc.setRenderPipelineState(resolve)
        enc.setDepthStencilState(vbs.resolveDepth)
        if cull != MTLCullMode.none { enc.setCullMode(.none) }
        var params = IlluminatoramaVisBufferResolveParams(targetSize: SIMD2(Float(width), Float(height)),
                                                          debugMode: vbs.debug.rawValue)
        enc.setFragmentBytes(&params, length: MemoryLayout<IlluminatoramaVisBufferResolveParams>.stride, index: 0)
        enc.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 1)
        enc.setFragmentBuffer(currentInstanceBuffer, offset: 0, index: 2)
        enc.setFragmentBuffer(albedoAtlas.uvScaleBuffer, offset: 0, index: 3)
        enc.setFragmentBuffer(previousInstanceBuffer, offset: 0, index: 4)
        enc.setFragmentBuffer(nonColorAtlas.uvScaleBuffer, offset: 0, index: 5)
        enc.setFragmentBuffer(albedoAtlas.sliceMeanBuffer, offset: 0, index: 7)
        enc.setFragmentBuffer(nonColorAtlas.sliceMeanBuffer, offset: 0, index: 8)
        table.withUnsafeBytes { enc.setFragmentBytes($0.baseAddress!, length: $0.count, index: 10) }
        if vbs.countsOverdraw { enc.setFragmentBuffer(vbs.counters, offset: 0, index: 11) }
        enc.setFragmentTexture(albedoAtlas.texture, index: 0)
        enc.setFragmentTexture(nonColorAtlas.texture, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.popDebugGroup()
    }

    /// The pixel rectangle covering every deferred draw's world bounds under this frame's camera
    /// (2 px of slack for the jitter); the whole target when any bound is open or crosses the eye.
    private func visBufferScissor(_ draws: [VisBufferDeferredDraw]) -> MTLScissorRect {
        let full = MTLScissorRect(x: 0, y: 0, width: width, height: height)
        let vp = gbs.gbufferCullVP
        var lo = SIMD2<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        for d in draws {
            guard d.bounds.isBounded else { return full }
            for k in 0..<8 {
                let p = SIMD3(k & 1 == 0 ? d.bounds.lo.x : d.bounds.hi.x, k & 2 == 0 ? d.bounds.lo.y : d.bounds.hi.y,
                              k & 4 == 0 ? d.bounds.lo.z : d.bounds.hi.z)
                let c = vp * SIMD4(p, 1)
                guard c.w > 1e-4 else { return full }
                let ndc = SIMD2(c.x, c.y) / c.w
                lo = simd_min(lo, ndc); hi = simd_max(hi, ndc)
            }
        }
        let W = Float(width), H = Float(height)
        let x0 = max(0, Int(((lo.x * 0.5 + 0.5) * W).rounded(.down)) - 2)
        let x1 = min(width, Int(((hi.x * 0.5 + 0.5) * W).rounded(.up)) + 2)
        let y0 = max(0, Int(((0.5 - hi.y * 0.5) * H).rounded(.down)) - 2)
        let y1 = min(height, Int(((0.5 - lo.y * 0.5) * H).rounded(.up)) + 2)
        guard x1 > x0, y1 > y0 else { return MTLScissorRect(x: 0, y: 0, width: 1, height: 1) }
        return MTLScissorRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    }
}
