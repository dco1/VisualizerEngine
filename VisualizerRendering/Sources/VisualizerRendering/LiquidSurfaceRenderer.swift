import Foundation
import Metal
import simd

// ── LIQUID SURFACE RENDERER ──────────────────────────────────────────────────
//
// Draws a simulated liquid's particles (`MLSMPMSolver.particleBuffer`, or any buffer of `MLSParticle`) as one continuous
// surface over a finished frame: the stream in the air, the splash and the pool are the same passes (Shaders/LiquidSurface.metal).
// The scene behind is refracted through it and absorbed by its depth, the room and the lamps reflect off it, aerated water
// (`MLSMPMSolver.aerationGain`) and spray are white.
//
// It composites onto `target` (the scene's finished colour) using `sceneDepth` (the scene's depth buffer, 0 near … 1 far, drawn
// with the same view and projection) so the scene hides the liquid where it is in front. No SceneKit; three render passes and
// five small compute passes a frame, independent of the simulation's grid.
//
//     let surface = LiquidSurfaceRenderer(engine: engine)
//     surface.particleRadius = solver.cellSize * 0.45
//     …each frame, after the scene is drawn and the solver stepped:
//     surface.encode(to: cb, particles: solver.particleBuffer.buffer, count: solver.particleBuffer.count,
//                    restDensity: solver.restDensity, view: v, projection: p, target: colour, sceneDepth: depth)

@MainActor
public final class LiquidSurfaceRenderer {

    /// A lamp: its light on the liquid's body and foam, and its highlight.
    public struct Light: Sendable {
        /// World position, or the direction TO the light when `directional`.
        public var position: SIMD3<Float>
        public var directional: Bool
        /// Its colour as it arrives at the liquid (display-referred, like the frame it is drawn onto).
        public var color: SIMD3<Float>
        public init(position: SIMD3<Float>, directional: Bool = false, color: SIMD3<Float>) {
            self.position = position; self.directional = directional; self.color = color
        }
    }

    /// What the liquid is.
    public struct Look: Sendable {
        /// Absorption per metre looked through, per channel (water: next to none; a red drink: green and blue go first).
        public var absorption = SIMD3<Float>(0.6, 0.15, 0.08)
        /// The colour the body scatters back where it is lit, and how much per metre (0 = clear).
        public var scatterColor = SIMD3<Float>(0.5, 0.7, 0.8)
        public var scatter: Float = 0
        /// Reflectance at normal incidence (water: 0.02).
        public var f0: Float = 0.02
        /// How far the scene behind is displaced by the surface (0 = none).
        public var refraction: Float = 0.6
        /// Gain on the environment's reflection, and the most a reflection or highlight may add (the frame is display-referred).
        public var environmentGain: Float = 1
        public var highlightCap: Float = 1.6
        /// Yaw of the environment map (radians).
        public var environmentYaw: Float = 0
        public var foamColor = SIMD3<Float>(0.95, 0.96, 0.97)
        /// How strongly aeration shows (0 = never white).
        public var foam: Float = 1.4
        /// How white a lone, fast drop is (spray).
        public var spray: Float = 0.5
        /// Light on the body and foam from the room at large.
        public var ambient: Float = 0.25
        public init() {}
    }

    public var look = Look()
    public var lights: [Light] = []
    /// The room, as an equirectangular map (nil = a dim room, lighter overhead).
    public var environment: MTLTexture?
    /// A particle's radius as drawn (world metres): about 0.9 of the particles' spacing fuses a body at rest density.
    public var particleRadius: Float = 0.004
    /// The smoothing's reach as a multiple of the radius, and how many rounds of it (each a horizontal and a vertical pass).
    public var smoothing: Float = 2.2
    public var smoothingRounds = 3
    /// Exposure time (s): a fast drop is drawn as long as it travels in it.
    public var shutter: Float = 1.0 / 250
    /// The most a sphere is stretched along its flight.
    public var maxStretch: Float = 3
    /// Spray and loose drops drawn into the same surface: buffers of `FoamParticle` (`FoamSolver.particleBuffer`, or any buffer
    /// laid out like it: position + life, velocity + age), how many each, and a drop's radius (world m; each is 0.45–1.35 of it).
    public var droplets: [(buffer: MTLBuffer, count: Int, radius: Float)] = []
    /// Draw nothing under this height (simulation space): for a pool that has a surface of its own, with this renderer drawing
    /// the stream and the splash above it, cut cleanly at that plane.
    public var clipBelow: Float?
    /// The outline: the depth pass draws each sphere this much larger, and the surface is then cut where the blurred thickness
    /// falls under `outlineThickness` radii: a smooth contour round the particles, where the plain union of their discs is
    /// scalloped (a thin stream showed its particles as beads down its sides).
    public var outlineDilation: Float = 1.4
    public var outlineThickness: Float = 0.3
    /// How much the surface rounds away at its silhouette (0 = flat to the edge, 1 = fully).
    public var edgeRounding: Float = 0.6

    private let device: MTLDevice
    private let library: MTLLibrary
    private let smoothPipeline: MTLComputePipelineState
    private let blurPipeline: MTLComputePipelineState
    private let depthPipeline: MTLRenderPipelineState
    private let thicknessPipeline: MTLRenderPipelineState
    private let dropDepthPipeline: MTLRenderPipelineState
    private let dropThicknessPipeline: MTLRenderPipelineState
    private var compositePipeline: MTLRenderPipelineState?
    private var compositeFormat: MTLPixelFormat = .invalid
    private let depthState: MTLDepthStencilState
    private let noDepthState: MTLDepthStencilState

    private var depthA: MTLTexture?, depthB: MTLTexture?, zBuffer: MTLTexture?
    private var thickA: MTLTexture?, thickB: MTLTexture?
    private var sceneCopy: MTLTexture?
    private var blackEnv: MTLTexture?

    struct Uniforms {
        var view = matrix_identity_float4x4
        var proj = matrix_identity_float4x4
        var invProj = matrix_identity_float4x4
        var invView = matrix_identity_float4x4
        var worldFromSim = matrix_identity_float4x4
        var simFromView = matrix_identity_float4x4
        var viewport = SIMD4<Float>.zero
        var particle = SIMD4<Float>.zero
        var absorption = SIMD4<Float>.zero
        var scatter = SIMD4<Float>.zero
        var optics = SIMD4<Float>.zero
        var foam = SIMD4<Float>.zero
        var filter = SIMD4<Float>.zero
        var envRot = SIMD4<Float>.zero
        var clip = SIMD4<Float>.zero
        var lightPos: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>) = (.zero, .zero, .zero, .zero)
        var lightCol: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>) = (.zero, .zero, .zero, .zero)
        var counts = SIMD4<UInt32>.zero
    }

    public init?(engine: SimEngine) {
        let device = engine.device
        guard let lib = engine.library,
              let smoothFn = lib.makeFunction(name: "liquidSmooth"), let blurFn = lib.makeFunction(name: "liquidThicknessBlur"),
              let splat = lib.makeFunction(name: "liquidSplatVS"), let depthFn = lib.makeFunction(name: "liquidDepthFS"),
              let thickFn = lib.makeFunction(name: "liquidThicknessFS"),
              let smooth = try? device.makeComputePipelineState(function: smoothFn),
              let blur = try? device.makeComputePipelineState(function: blurFn) else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.label = "LiquidSurface.depth"
        d.vertexFunction = splat; d.fragmentFunction = depthFn
        d.colorAttachments[0].pixelFormat = .r32Float
        d.depthAttachmentPixelFormat = .depth32Float
        let t = MTLRenderPipelineDescriptor()
        t.label = "LiquidSurface.thickness"
        t.vertexFunction = splat; t.fragmentFunction = thickFn
        t.colorAttachments[0].pixelFormat = .rg16Float
        t.colorAttachments[0].isBlendingEnabled = true
        t.colorAttachments[0].sourceRGBBlendFactor = .one; t.colorAttachments[0].destinationRGBBlendFactor = .one
        t.colorAttachments[0].sourceAlphaBlendFactor = .one; t.colorAttachments[0].destinationAlphaBlendFactor = .one
        let ds = MTLDepthStencilDescriptor()
        ds.depthCompareFunction = .less; ds.isDepthWriteEnabled = true
        let nds = MTLDepthStencilDescriptor()
        nds.depthCompareFunction = .always; nds.isDepthWriteEnabled = false
        guard let dp = try? device.makeRenderPipelineState(descriptor: d), let tp = try? device.makeRenderPipelineState(descriptor: t),
              let dropVS = lib.makeFunction(name: "liquidDropletVS") else { return nil }
        d.label = "LiquidSurface.dropDepth"; d.vertexFunction = dropVS
        t.label = "LiquidSurface.dropThickness"; t.vertexFunction = dropVS
        guard let ddp = try? device.makeRenderPipelineState(descriptor: d), let dtp = try? device.makeRenderPipelineState(descriptor: t),
              let dss = device.makeDepthStencilState(descriptor: ds), let ndss = device.makeDepthStencilState(descriptor: nds) else { return nil }
        self.device = device; library = lib
        smoothPipeline = smooth; blurPipeline = blur; depthPipeline = dp; thicknessPipeline = tp
        dropDepthPipeline = ddp; dropThicknessPipeline = dtp
        depthState = dss; noDepthState = ndss
    }

    private func texture(_ format: MTLPixelFormat, _ w: Int, _ h: Int, _ usage: MTLTextureUsage, _ label: String) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: max(1, w), height: max(1, h), mipmapped: false)
        d.usage = usage; d.storageMode = .private
        let t = device.makeTexture(descriptor: d)
        t?.label = label
        return t
    }

    private func prepare(target: MTLTexture) -> Bool {
        let w = target.width, h = target.height
        if depthA?.width != w || depthA?.height != h {
            depthA = texture(.r32Float, w, h, [.renderTarget, .shaderRead, .shaderWrite], "LiquidSurface.depthA")
            depthB = texture(.r32Float, w, h, [.shaderRead, .shaderWrite], "LiquidSurface.depthB")
            zBuffer = texture(.depth32Float, w, h, [.renderTarget], "LiquidSurface.z")
            thickA = texture(.rg16Float, w / 2, h / 2, [.renderTarget, .shaderRead, .shaderWrite], "LiquidSurface.thickA")
            thickB = texture(.rg16Float, w / 2, h / 2, [.shaderRead, .shaderWrite], "LiquidSurface.thickB")
        }
        if sceneCopy?.width != w || sceneCopy?.height != h || sceneCopy?.pixelFormat != target.pixelFormat {
            sceneCopy = texture(target.pixelFormat, w, h, [.shaderRead], "LiquidSurface.scene")
        }
        if compositePipeline == nil || compositeFormat != target.pixelFormat {
            guard let vs = library.makeFunction(name: "liquidQuadVS"), let fs = library.makeFunction(name: "liquidCompositeFS") else { return false }
            let c = MTLRenderPipelineDescriptor()
            c.label = "LiquidSurface.composite"
            c.vertexFunction = vs; c.fragmentFunction = fs
            c.colorAttachments[0].pixelFormat = target.pixelFormat
            c.colorAttachments[0].isBlendingEnabled = true
            c.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha; c.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            c.colorAttachments[0].sourceAlphaBlendFactor = .zero; c.colorAttachments[0].destinationAlphaBlendFactor = .one
            compositePipeline = try? device.makeRenderPipelineState(descriptor: c)
            compositeFormat = target.pixelFormat
        }
        if blackEnv == nil { blackEnv = texture(.rgba16Float, 1, 1, [.shaderRead], "LiquidSurface.noEnv") }
        return depthA != nil && depthB != nil && zBuffer != nil && thickA != nil && thickB != nil && sceneCopy != nil
            && compositePipeline != nil && blackEnv != nil
    }

    /// Draw the liquid over `target`. `view` and `projection` are the scene's (depth 0 near … 1 far, the camera looking down −z);
    /// `worldFromSim` places the simulation in the world (identity if it is simulated there).
    public func encode(to cb: MTLCommandBuffer, particles: MTLBuffer, count: Int, restDensity: Float,
                       view: simd_float4x4, projection: simd_float4x4, worldFromSim: simd_float4x4 = matrix_identity_float4x4,
                       target: MTLTexture, sceneDepth: MTLTexture) {
        guard count > 0, prepare(target: target),
              let depthA, let depthB, let zBuffer, let thickA, let thickB, let sceneCopy, let compositePipeline, let blackEnv else { return }
        let w = Float(target.width), h = Float(target.height)
        var u = Uniforms()
        u.view = view; u.proj = projection; u.invProj = projection.inverse; u.invView = view.inverse; u.worldFromSim = worldFromSim; u.simFromView = (view * worldFromSim).inverse
        u.viewport = SIMD4(w, h, 1 / w, 1 / h)
        u.particle = SIMD4(particleRadius, max(1, maxStretch), max(0, shutter), restDensity)
        u.absorption = SIMD4(look.absorption, 1)
        u.scatter = SIMD4(look.scatterColor, look.scatter)
        u.optics = SIMD4(look.f0, look.refraction, look.environmentGain, environment == nil ? 0 : 1)
        u.foam = SIMD4(look.foamColor, look.foam)
        // Rounds of a Gaussian add in quadrature: each is the whole reach over √rounds.
        u.filter = SIMD4(particleRadius * smoothing, particleRadius * 6, look.ambient, look.spray)
        // The thickness has no depth of its own to scale its blur by: the simulation's own distance stands in.
        let simAt = (view * worldFromSim).columns.3
        u.envRot = SIMD4(look.environmentYaw, look.highlightCap, edgeRounding, 1 / max(0.05, simd_length(SIMD3(simAt.x, simAt.y, simAt.z))))
        u.clip = SIMD4(clipBelow == nil ? 0 : 1, clipBelow ?? 0, particleRadius * outlineThickness, 1)
        // Enough rounds that a round's taps are a pixel or so apart (wider, and the smoothing is noise).
        let sigmaPixels = particleRadius * smoothing * projection.columns.1.y * h * 0.5 * u.envRot.w
        let rounds = min(6, max(max(1, smoothingRounds), Int((sigmaPixels / 9 * (sigmaPixels / 9)).rounded(.up))))
        u.filter.x = particleRadius * smoothing / Float(rounds).squareRoot()
        let ls = Array(lights.prefix(4))
        func lp(_ i: Int) -> SIMD4<Float> { i < ls.count ? SIMD4(ls[i].position, ls[i].directional ? 0 : 1) : .zero }
        func lc(_ i: Int) -> SIMD4<Float> { SIMD4(i < ls.count ? ls[i].color : .zero, 0) }
        u.lightPos = (lp(0), lp(1), lp(2), lp(3)); u.lightCol = (lc(0), lc(1), lc(2), lc(3))
        u.counts = SIMD4(UInt32(ls.count), 0, 0, 0)
        let stride = MemoryLayout<Uniforms>.stride

        if let blit = cb.makeBlitCommandEncoder() {
            blit.label = "LiquidSurface.sceneCopy"
            blit.copy(from: target, to: sceneCopy)
            blit.endEncoding()
        }

        // 1. The nearest surface.
        let dp = MTLRenderPassDescriptor()
        dp.colorAttachments[0].texture = depthA
        dp.colorAttachments[0].loadAction = .clear; dp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        dp.colorAttachments[0].storeAction = .store
        dp.depthAttachment.texture = zBuffer
        dp.depthAttachment.loadAction = .clear; dp.depthAttachment.clearDepth = 1; dp.depthAttachment.storeAction = .dontCare
        if let enc = cb.makeRenderCommandEncoder(descriptor: dp) {
            enc.label = "LiquidSurface.depth"
            enc.setRenderPipelineState(depthPipeline)
            enc.setDepthStencilState(depthState)
            var udepth = u
            udepth.clip.w = max(1, outlineDilation)
            enc.setVertexBuffer(particles, offset: 0, index: 0)
            enc.setVertexBytes(&udepth, length: stride, index: 1)
            enc.setFragmentBytes(&udepth, length: stride, index: 0)
            enc.setFragmentTexture(sceneDepth, index: 0)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: count)
            for drops in droplets where drops.count > 0 {
                var ud = udepth
                ud.lightCol.3.w = drops.radius
                enc.setRenderPipelineState(dropDepthPipeline)
                enc.setVertexBuffer(drops.buffer, offset: 0, index: 0)
                enc.setVertexBytes(&ud, length: stride, index: 1)
                enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: drops.count)
            }
            enc.endEncoding()
        }

        // 2. How much liquid, and how much of it white (half size).
        var uh = u
        uh.viewport = SIMD4(Float(thickA.width), Float(thickA.height), 1 / Float(thickA.width), 1 / Float(thickA.height))
        let tp = MTLRenderPassDescriptor()
        tp.colorAttachments[0].texture = thickA
        tp.colorAttachments[0].loadAction = .clear; tp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        tp.colorAttachments[0].storeAction = .store
        if let enc = cb.makeRenderCommandEncoder(descriptor: tp) {
            enc.label = "LiquidSurface.thickness"
            enc.setRenderPipelineState(thicknessPipeline)
            enc.setVertexBuffer(particles, offset: 0, index: 0)
            enc.setVertexBytes(&uh, length: stride, index: 1)
            enc.setFragmentBytes(&uh, length: stride, index: 0)
            enc.setFragmentTexture(sceneDepth, index: 0)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: count)
            for drops in droplets where drops.count > 0 {
                var ud = uh
                ud.lightCol.3.w = drops.radius
                enc.setRenderPipelineState(dropThicknessPipeline)
                enc.setVertexBuffer(drops.buffer, offset: 0, index: 0)
                enc.setVertexBytes(&ud, length: stride, index: 1)
                enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: drops.count)
            }
            enc.endEncoding()
        }

        // 3–4. Smooth both.
        if let enc = cb.makeComputeCommandEncoder() {
            enc.label = "LiquidSurface.smooth"
            func pass(_ pipeline: MTLComputePipelineState, _ a: MTLTexture, _ b: MTLTexture, _ axis: SIMD2<Int32>, _ uniforms: inout Uniforms) {
                var ax = axis
                enc.setComputePipelineState(pipeline)
                enc.setTexture(a, index: 0); enc.setTexture(b, index: 1)
                enc.setBytes(&uniforms, length: stride, index: 0)
                enc.setBytes(&ax, length: MemoryLayout<SIMD2<Int32>>.stride, index: 1)
                let tw = pipeline.threadExecutionWidth, th = max(1, pipeline.maxTotalThreadsPerThreadgroup / tw)
                enc.dispatchThreads(MTLSize(width: a.width, height: a.height, depth: 1), threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
            }
            for round in 0..<rounds {
                u.counts.y = UInt32(round)
                u.counts.z = round == rounds - 1 && rounds > 1 ? 1 : 0      // the last round is not jittered: it leaves no noise
                pass(smoothPipeline, depthA, depthB, SIMD2(1, 0), &u)
                pass(smoothPipeline, depthB, depthA, SIMD2(0, 1), &u)
            }
            pass(blurPipeline, thickA, thickB, SIMD2(1, 0), &uh)
            pass(blurPipeline, thickB, thickA, SIMD2(0, 1), &uh)
            enc.endEncoding()
        }

        // 5. Over the scene.
        let cp = MTLRenderPassDescriptor()
        cp.colorAttachments[0].texture = target
        cp.colorAttachments[0].loadAction = .load; cp.colorAttachments[0].storeAction = .store
        if let enc = cb.makeRenderCommandEncoder(descriptor: cp) {
            enc.label = "LiquidSurface.composite"
            enc.setRenderPipelineState(compositePipeline)
            enc.setFragmentBytes(&u, length: stride, index: 0)
            enc.setFragmentTexture(depthA, index: 0)
            enc.setFragmentTexture(thickA, index: 1)
            enc.setFragmentTexture(sceneCopy, index: 2)
            enc.setFragmentTexture(sceneDepth, index: 3)
            enc.setFragmentTexture(environment ?? blackEnv, index: 4)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }
    }
}
