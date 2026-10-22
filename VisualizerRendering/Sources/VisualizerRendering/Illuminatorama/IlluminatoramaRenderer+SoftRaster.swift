import Foundation
import Metal
import simd

/// A software (compute) rasterizer the renderer can hand sub-pixel triangles to — Signack S4
/// (Daydream DH-1461). The engine owns both ENDS: the vertex stage that writes window-space
/// triangles (`illumi_soft_vertex`, the same deformation as the hardware vertex stage) and the
/// merge that hands the result to the G-buffer pass's depth test (`illumi_soft_merge_fs`). The
/// rasterizer owns the middle — coverage, depth, ids — and its contract (Signack's
/// `SignackRaster.metal`): triangles are 48-byte `SignackRasterTriangle` records claimed with an
/// atomic increment of the uint at offset 0 of `stateBuffer`; after `encodeRaster` the nearest
/// depth (float bits, 0xFFFFFFFF = none) and the winning id (`uint2`) are in `depthBuffer` /
/// `idBuffer`, row-major, and the reader resets each pixel it reads. The engine does not import
/// Signack: the host attaches one (`IlluminatoramaRenderer.softRasterizer`).
public protocol IlluminatoramaSoftRasterizer: AnyObject {
    /// Size this frame's target and triangle list. false ⇒ the renderer draws the soft records with
    /// the hardware instead.
    func reserve(width: Int, height: Int, triangles: Int) -> Bool
    var triangleBuffer: MTLBuffer { get }
    var triangleCapacity: Int { get }
    var stateBuffer: MTLBuffer { get }
    var depthBuffer: MTLBuffer { get }
    var idBuffer: MTLBuffer { get }
    /// Resolve the appended triangles, writing only inside `scissor`, on `encoder` (serial).
    func encodeRaster(_ encoder: MTLComputeCommandEncoder, scissor: MTLScissorRect)
}

/// Mirror of `SoftVertexParams` (IlluminatoramaSoftRaster.metal, 48 B).
struct IlluminatoramaSoftVertexParams {
    var slot: UInt32
    var instanceBase: UInt32
    var capacity: UInt32
    var cullBack: UInt32
    var targetSize: SIMD2<Float>
    var pad = SIMD2<Float>(0, 0)
    var scissor: SIMD4<UInt32>
}

/// The software-raster half of `VisBufferState`.
final class SoftRasterState {
    /// Host lever (A/B), default OFF. Env `ILLUMINATORAMA_SOFTRASTER=1` / `=0` overrides it.
    var enabled = false
    /// Triangles projecting smaller than this (px², G-buffer resolution) go to the software rasterizer.
    var triangleArea: Float = 1
    /// TEST-ONLY instrument: draw the soft records NOWHERE (neither rasterizer) — what the hardware
    /// pass costs without the routed triangles.
    var dropForTest = false
    weak var rasterizer: IlluminatoramaSoftRasterizer?
    var vertex: MTLComputePipelineState?
    var merge: [Bool: MTLRenderPipelineState] = [:]
    var unavailable = false
}

/// What this frame's software pass did, for the G-buffer pass that follows it.
struct SoftRasterFrame {
    /// Visibility-buffer slots whose soft record was rasterised in software (draw no soft record).
    var slots: Set<Int>
    var scissor: MTLScissorRect
}

extension IlluminatoramaRenderer {
    /// A/B lever for Signack's software rasterizer (default OFF). It implies the visibility buffer
    /// for the meshes that opted into it.
    public var softRasterEnabled: Bool {
        get { vbs.soft.enabled }
        set { vbs.soft.enabled = newValue }
    }
    /// Projected triangle area (px², G-buffer resolution) under which a triangle is rasterised in
    /// software. The host routes by it (Signack's cut tests each on-screen triangle).
    public var softRasterTriangleArea: Float {
        get { vbs.soft.triangleArea }
        set { vbs.soft.triangleArea = newValue }
    }
    /// The rasterizer the host provides (Signack's). Held weakly — the host keeps it alive.
    public var softRasterizer: IlluminatoramaSoftRasterizer? {
        get { vbs.soft.rasterizer }
        set { vbs.soft.rasterizer = newValue }
    }
    /// TEST-ONLY: drop the soft-routed triangles entirely (the instrument for their hardware cost).
    public var softRasterDropForTest: Bool {
        get { vbs.soft.dropForTest }
        set { vbs.soft.dropForTest = newValue }
    }
    /// Whether the host should route soft clusters this frame: the lever (or env) is on and a
    /// rasterizer is attached. A host that routes while this is false loses nothing — the renderer
    /// then draws the soft records with the hardware.
    public var softRasterRequested: Bool {
        if vbs.soft.dropForTest { return true }
        let env = Self.frameEnv("ILLUMINATORAMA_SOFTRASTER")
        return (env == "1" || (vbs.soft.enabled && env != "0")) && vbs.soft.rasterizer != nil && !vbs.soft.unavailable
    }

    /// Before the G-buffer pass: the vertex stage for every deferred draw that carries a soft record,
    /// then the rasterizer itself. Returns the slots it covered (nil ⇒ none; their soft records are
    /// drawn by the hardware).
    func encodeSoftRaster(_ cb: MTLCommandBuffer, _ draws: [VisBufferDeferredDraw], extended: Bool) -> SoftRasterFrame? {
        guard softRasterRequested, !draws.isEmpty else { return nil }
        let slots = draws.indices.filter { i in
            let m = draws[i].mesh
            return m.softRasterDrawOffset != nil && m.softRasterDispatchOffset != nil && m.indirectDrawBuffer != nil
                && draws[i].group.count == 1 && i < 0x7FFE
        }
        guard !slots.isEmpty else { return nil }
        if vbs.soft.dropForTest { return SoftRasterFrame(slots: Set(slots), scissor: MTLScissorRect()) }
        guard let r = vbs.soft.rasterizer else { return nil }
        let capacity = slots.reduce(0) { $0 + draws[$1].mesh.indexCount / 3 }
        guard let vertex = softVertexPipeline(), softMergePipeline(extended: extended) != nil,
              r.reserve(width: width, height: height, triangles: capacity),
              let enc = timedComputeEncoder(cb, "softraster") else { return nil }
        enc.label = "Illuminatorama.softRaster"
        let scissor = visBufferScissor(draws)
        enc.setComputePipelineState(vertex)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 4)
        enc.setBuffer(currentInstanceBuffer, offset: 0, index: 5)
        enc.setBuffer(previousInstanceBuffer, offset: 0, index: 6)
        enc.setBuffer(r.triangleBuffer, offset: 0, index: 7)
        enc.setBuffer(r.stateBuffer, offset: 0, index: 8)
        for slot in slots {
            let d = draws[slot], m = d.mesh
            guard let args = m.indirectDrawBuffer, let rec = m.softRasterDrawOffset, let disp = m.softRasterDispatchOffset else { continue }
            var p = IlluminatoramaSoftVertexParams(
                slot: UInt32(slot), instanceBase: UInt32(d.group.start), capacity: UInt32(r.triangleCapacity),
                cullBack: m.doubleSided ? 0 : 1, targetSize: SIMD2(Float(width), Float(height)),
                scissor: SIMD4(UInt32(scissor.x), UInt32(scissor.y), UInt32(scissor.x + scissor.width),
                               UInt32(scissor.y + scissor.height)))
            enc.setBytes(&p, length: MemoryLayout<IlluminatoramaSoftVertexParams>.stride, index: 0)
            enc.setBuffer(m.vertexBuffer, offset: 0, index: 1)
            enc.setBuffer(m.indexBuffer, offset: 0, index: 2)
            enc.setBuffer(args, offset: rec, index: 3)
            enc.dispatchThreadgroups(indirectBuffer: args, indirectBufferOffset: disp,
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        r.encodeRaster(enc, scissor: scissor)
        enc.endEncoding()
        return SoftRasterFrame(slots: Set(slots), scissor: scissor)
    }

    /// Inside the G-buffer pass, after the hardware id raster: the soft result through the depth test
    /// into the id attachment. The caller restores its own state afterwards (the resolve sets all).
    func encodeSoftRasterMerge(_ enc: MTLRenderCommandEncoder, _ frame: SoftRasterFrame, extended: Bool) {
        guard frame.scissor.width > 0, let r = vbs.soft.rasterizer, let pso = softMergePipeline(extended: extended) else { return }
        enc.pushDebugGroup("Illuminatorama.softRaster.merge")
        enc.setScissorRect(frame.scissor)
        enc.setRenderPipelineState(pso)
        enc.setDepthStencilState(gbs.depthState)
        var w = UInt32(width)
        enc.setFragmentBytes(&w, length: 4, index: 0)
        enc.setFragmentBuffer(r.depthBuffer, offset: 0, index: 1)
        enc.setFragmentBuffer(r.idBuffer, offset: 0, index: 2)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.popDebugGroup()
    }

    private func softVertexPipeline() -> MTLComputePipelineState? {
        if let p = vbs.soft.vertex { return p }
        guard let library = engine.library else { vbs.soft.unavailable = true; return nil }
        let c = MTLFunctionConstantValues()
        var f = false
        c.setConstantValue(&f, type: .bool, index: 10)
        c.setConstantValue(&f, type: .bool, index: 11)
        c.setConstantValue(&f, type: .bool, index: 12)
        do {
            let fn = try library.makeFunction(name: "illumi_soft_vertex", constantValues: c)
            vbs.soft.vertex = try device.makeComputePipelineState(function: fn)
        } catch {
            VisBufferState.log.error("software raster vertex stage unavailable: \(error.localizedDescription)")
            vbs.soft.unavailable = true
        }
        return vbs.soft.vertex
    }

    private func softMergePipeline(extended: Bool) -> MTLRenderPipelineState? {
        if let p = vbs.soft.merge[extended] { return p }
        guard let library = engine.library else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.label = "Illuminatorama.softRaster.merge\(extended ? ".ext" : "")"
        d.vertexFunction = library.makeFunction(name: "illumi_vis_resolve_vs")
        d.fragmentFunction = library.makeFunction(name: "illumi_soft_merge_fs")
        let formats: [MTLPixelFormat] = [.rgba16Float, .rgba16Float, .rgba16Float, .rg16Float, .r32Uint]
            + (extended ? [.rgba16Float] : [])
        for (i, fmt) in formats.enumerated() {
            d.colorAttachments[i].pixelFormat = fmt
            d.colorAttachments[i].writeMask = []
        }
        d.colorAttachments[6].pixelFormat = .rg32Uint
        d.colorAttachments[6].writeMask = .all
        d.depthAttachmentPixelFormat = .depth32Float
        do { vbs.soft.merge[extended] = try device.makeRenderPipelineState(descriptor: d) } catch {
            VisBufferState.log.error("software raster merge unavailable: \(error.localizedDescription)")
            vbs.soft.unavailable = true
        }
        return vbs.soft.merge[extended]
    }
}
