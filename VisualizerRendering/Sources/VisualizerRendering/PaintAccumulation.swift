import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

/// PaintAccumulation — a persistent, never-cleared GPU paint surface for the
/// Tennis Ball Painter scene.
///
/// One `PaintAccumulation` owns one RGBA16F `MTLTexture` (the wall canvas, or the
/// floor spatter sheet). Tennis balls splat against the wall; each impact calls
/// `stamp(uv:colour:…)`, which dispatches a tiny `paintStamp` kernel that
/// composites a soft, ragged paint dab into the texture at the impact UV. The
/// texture is bound to the material's `diffuse.contents` (SceneKit samples an
/// `MTLTexture` directly), so the pointillist painting accumulates in GPU memory
/// and is never rebuilt CPU-side and never erased.
///
/// Drips are real: a stamp may `seedDrip(...)` a rivulet agent (position +
/// velocity + colour + life). `tick(dt:)` runs `paintDripStep`, which integrates
/// each agent DOWN the wall under gravity and stamps its wet streak into the same
/// texture — the streak is the integrated path of a gravity-advected agent, not a
/// scrolling-texture fake.
///
/// Plugs into the shared runtime: takes a `SimEngine`, pulls its pipelines from
/// `SimPipelineCache`, submits on the engine's command queue. No parallel device
/// stack, no per-frame `waitUntilCompleted`.
///
/// plausibility: real — the painting is a real accumulation buffer stamped at
///   physically-derived impact points with physically-derived colours; drips are
///   real gravity-advected agents. No scrolling-texture fake, no pre-baked image.
@MainActor
public final class PaintAccumulation {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "PaintAccumulation")

    private let engine: SimEngine
    private let stampPipeline: MTLComputePipelineState
    private let dripPipeline: MTLComputePipelineState
    private let clearPipeline: MTLComputePipelineState
    private let decayPipeline: MTLComputePipelineState
    private let resolvePipeline: MTLComputePipelineState
    private let floodPipeline: MTLComputePipelineState
    private let coveragePipeline: MTLComputePipelineState
    private let frontProfilePipeline: MTLComputePipelineState

    /// The persistent canvas. Bound to the surface material's `diffuse.contents`.
    public let texture: MTLTexture
    public let width: Int
    public let height: Int

    // ── Atlas binding (Illuminatorama native) ──────────────────────
    /// `bgra8Unorm` staging texture sized to the atlas slice; `paintResolve`
    /// encodes the linear RGBA16F canvas into sRGB bytes here, then we blit it
    /// into the renderer's albedo atlas slice (blit-compatible with the atlas's
    /// `bgra8Unorm_srgb` format). Lazily built at the atlas's slice size.
    private var stagingTexture: MTLTexture?
    private var stagingSize: Int = 0

    // ── Drip agents ────────────────────────────────────────────────
    private static let maxDrips = 1024
    private var dripBuffer: MTLBuffer
    private var dripCount: Int = 0
    private var nextDrip: Int = 0           // ring-buffer cursor

    /// Mirrors `PaintStamp` in PaintAccumulation.metal (float4-packed).
    private struct PaintStamp {
        var centerRadiusAlpha: SIMD4<Float>
        var colorEdge: SIMD4<Float>
        var seedFlick: SIMD4<Float>     // x=seed, yz=flick dir (UV), w=wetness
        var extra: SIMD4<Float>         // x=streakCount, yzw spare
    }
    private struct DecayUniforms {
        var keep: Float; var pad0: Float = 0; var pad1: Float = 0; var pad2: Float = 0
    }
    /// Mirrors `DripAgent` in PaintAccumulation.metal.
    private struct DripAgent {
        var posVel: SIMD4<Float>
        var colorLife: SIMD4<Float>
        var widthFlowPad: SIMD4<Float>
    }
    private struct DripUniforms {
        var dt: Float
        var gravity: Float
        var agentCount: UInt32
        var pad: Float
    }
    private struct ClearColor { var rgba: SIMD4<Float> }

    /// `baseColor` is the wall/floor base the texture is initialised to (e.g.
    /// near-white for the gallery wall). After init the texture is never cleared.
    public init?(engine: SimEngine, width: Int, height: Int, baseColor: SIMD3<Float>) {
        self.engine = engine
        self.width = width
        self.height = height
        guard let sp = engine.pipeline("paintStamp"),
              let dp = engine.pipeline("paintDripStep"),
              let cp = engine.pipeline("paintClear"),
              let dy = engine.pipeline("paintDecay"),
              let rs = engine.pipeline("paintResolve"),
              let fl = engine.pipeline("paintFlood"),
              let cv = engine.pipeline("paintCoverage"),
              let fp = engine.pipeline("paintFrontProfile") else {
            Self.log.error("paint pipeline lookup failed")
            return nil
        }
        self.stampPipeline = sp
        self.dripPipeline = dp
        self.clearPipeline = cp
        self.decayPipeline = dy
        self.resolvePipeline = rs
        self.floodPipeline = fl
        self.coveragePipeline = cv
        self.frontProfilePipeline = fp

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderWrite, .shaderRead]
        desc.storageMode = .private
        guard let tex = engine.device.makeTexture(descriptor: desc) else {
            Self.log.error("paint texture alloc failed")
            return nil
        }
        tex.label = "PaintAccumulation"
        self.texture = tex

        guard let db = engine.device.makeBuffer(
            length: MemoryLayout<DripAgent>.stride * Self.maxDrips,
            options: .storageModeShared) else {
            Self.log.error("drip buffer alloc failed")
            return nil
        }
        db.label = "PaintDripAgents"
        self.dripBuffer = db

        // One-time clear to the base colour.
        clear(to: baseColor)
    }

    /// Initialise (or re-initialise) the whole texture to a flat colour. Called
    /// once at setup; not part of the per-frame path.
    public func clear(to color: SIMD3<Float>) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = "PaintAccumulation.clear"
        enc.setComputePipelineState(clearPipeline)
        enc.setTexture(texture, index: 0)
        var c = ClearColor(rgba: SIMD4<Float>(color, 1))
        enc.setBytes(&c, length: MemoryLayout<ClearColor>.stride, index: 0)
        dispatch2D(enc, pipeline: clearPipeline)
        enc.endEncoding()
        cb.commit()
    }

    /// Stamp a paint dab at `uv` (in [0,1]). `radiusUV` is the dab radius in UV
    /// space, `alpha` the peak opacity, `colour` linear RGB. Tiny dispatch over a
    /// tile around the dab. Call this on a real ball→wall (or ball→floor) impact.
    ///
    /// `aspectY` (= faceHeight / faceWidth) keeps the dab round IN WORLD when the
    /// canvas is mapped onto a non-square face (the wide back wall). 1 = square.
    public func stamp(uv: SIMD2<Float>, radiusUV: Float, alpha: Float,
                      colour: SIMD3<Float>, edgeNoise: Float = 0.35,
                      flickDirUV: SIMD2<Float> = .zero, wetness: Float = 1.0,
                      streakCount: Int = 3, aspectY: Float = 1.0) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = "PaintAccumulation.stamp"
        enc.setComputePipelineState(stampPipeline)
        enc.setTexture(texture, index: 0)
        var s = PaintStamp(
            centerRadiusAlpha: SIMD4(uv.x, uv.y, radiusUV, alpha),
            colorEdge: SIMD4(colour, edgeNoise),
            seedFlick: SIMD4(Float.random(in: 0...1000), flickDirUV.x, flickDirUV.y, wetness),
            extra: SIMD4(Float(max(0, streakCount)), max(1e-3, aspectY), 0, 0))
        enc.setBytes(&s, length: MemoryLayout<PaintStamp>.stride, index: 0)
        // Dispatch the affected tile. The dab reaches ~1/aspectY further in v
        // (it's stretched tall in UV to read round on the wide face), so pad y by
        // that factor or the dab is clipped vertically.
        let pad = Int((radiusUV * 2.6 * Float(max(width, height))).rounded(.up)) + 2
        let padY = Int((Float(pad) / max(1e-3, aspectY)).rounded(.up)) + 2
        let cx = Int(uv.x * Float(width)), cy = Int(uv.y * Float(height))
        let x0 = max(0, cx - pad), y0 = max(0, cy - padY)
        let x1 = min(width, cx + pad), y1 = min(height, cy + padY)
        let gw = max(1, x1 - x0), gh = max(1, y1 - y0)
        let w = stampPipeline.threadExecutionWidth
        let h = max(1, stampPipeline.maxTotalThreadsPerThreadgroup / w)
        // The kernel reads gid in absolute texture space, so dispatch the full
        // affected origin via threadgroup offset isn't available; dispatch the
        // whole tile starting at (0,0) is wasteful — instead dispatch a grid
        // covering the tile but shift via the kernel's own uv math (absolute).
        // Simpler & correct: dispatch a grid of (gw,gh) won't map to absolute
        // coords. So dispatch the full texture region up to (x1,y1) — still
        // cheap since the kernel early-outs outside the dab.
        let grid = MTLSize(width: x1, height: y1, depth: 1)
        let tg = MTLSize(width: w, height: h, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
        _ = (x0, y0, gw, gh)   // (kept for clarity; tile origin handled in-kernel)
    }

    /// Seed a drip rivulet that will run down the wall from `uv`. `velUV` is the
    /// initial UV velocity (+y is DOWN). Ring-buffered.
    public func seedDrip(uv: SIMD2<Float>, velUV: SIMD2<Float>, colour: SIMD3<Float>,
                         life: Float, widthUV: Float, flow: Float) {
        let agent = DripAgent(
            posVel: SIMD4(uv.x, uv.y, velUV.x, velUV.y),
            colorLife: SIMD4(colour, life),
            widthFlowPad: SIMD4(widthUV, flow, 0, 0))
        let ptr = dripBuffer.contents().bindMemory(to: DripAgent.self, capacity: Self.maxDrips)
        ptr[nextDrip] = agent
        nextDrip = (nextDrip + 1) % Self.maxDrips
        // dripCount is the high-water mark of slots ever written (dead agents
        // early-out in the kernel, so it's safe to keep dispatching them).
        dripCount = max(dripCount, nextDrip == 0 ? Self.maxDrips : nextDrip)
    }

    /// Advance all live drips one step and stamp their streaks.
    public func tick(dt: Float) {
        guard dripCount > 0 else { return }
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = "PaintAccumulation.dripStep"
        enc.setComputePipelineState(dripPipeline)
        enc.setTexture(texture, index: 0)
        enc.setBuffer(dripBuffer, offset: 0, index: 0)
        var u = DripUniforms(dt: dt, gravity: 0.06, agentCount: UInt32(dripCount), pad: 0)
        enc.setBytes(&u, length: MemoryLayout<DripUniforms>.stride, index: 1)
        let w = dripPipeline.threadExecutionWidth
        let grid = MTLSize(width: dripCount, height: 1, depth: 1)
        let tg = MTLSize(width: min(w, dripCount), height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
    }

    /// Decay the per-texel WETNESS (alpha) channel toward 0 so fresh dabs read
    /// wet-glossy and dry to matte over ~2 s. `halfLife` is in seconds. RGB (the
    /// permanent painting) is untouched. Cheap full-texture pass; gate it so it
    /// only runs while something is still wet (caller tracks recent activity).
    public func decayWetness(dt: Float, halfLife: Float = 1.6) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = "PaintAccumulation.decay"
        enc.setComputePipelineState(decayPipeline)
        enc.setTexture(texture, index: 0)
        let keep = powf(0.5, max(1e-4, dt) / max(1e-3, halfLife))
        var u = DecayUniforms(keep: keep)
        enc.setBytes(&u, length: MemoryLayout<DecayUniforms>.stride, index: 0)
        dispatch2D(enc, pipeline: decayPipeline)
        enc.endEncoding()
        cb.commit()
    }

    /// Resolve the linear RGBA16F canvas into the renderer's albedo atlas
    /// `slice` (sRGB-encoded), so the accumulating painting is sampled directly
    /// by the Illuminatorama G-buffer pass — the native binding the Tennis Ball
    /// Painter wall / floor / per-ball stain textures use. Encodes onto `cb`
    /// (the caller's render command buffer, on the shared engine queue) so the
    /// resolve + blit are ordered BEFORE the g-buffer read with no extra sync.
    public func resolve(into atlas: IlluminatoramaTextureAtlas, slice: Int32,
                        on cb: MTLCommandBuffer) {
        let sliceSize = atlas.sliceSize
        if stagingTexture == nil || stagingSize != sliceSize {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: sliceSize, height: sliceSize,
                mipmapped: false)
            d.usage = [.shaderWrite, .shaderRead]
            d.storageMode = .private
            stagingTexture = engine.device.makeTexture(descriptor: d)
            stagingTexture?.label = "PaintAccumulation.staging"
            stagingSize = sliceSize
        }
        guard let staging = stagingTexture,
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "PaintAccumulation.resolve"
        enc.setComputePipelineState(resolvePipeline)
        enc.setTexture(texture, index: 0)
        enc.setTexture(staging, index: 1)
        let w = resolvePipeline.threadExecutionWidth
        let h = max(1, resolvePipeline.maxTotalThreadsPerThreadgroup / w)
        enc.dispatchThreads(MTLSize(width: sliceSize, height: sliceSize, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
        enc.endEncoding()
        atlas.blitLiveSlice(slice, from: staging, on: cb)
    }

    // ── Repaint flood front ────────────────────────────────────────
    /// Which way the new-colour wave flows across this canvas.
    public enum FloodAxis { case wallDown, floorRadial }

    /// Mirrors `FloodUniforms` in PaintAccumulation.metal.
    private struct FloodUniforms {
        var color: SIMD4<Float>   // rgb + wetness
        var front: SIMD4<Float>   // prevPos, curPos, seamV, mode
        var drip:  SIMD4<Float>   // seed, fingerAmp, fingerFreq, feather
    }

    /// Advance a repaint wave one frame: write `newColour` into the band the front
    /// swept between `prevPos` and `curPos` (in front-coordinate space — v for a wall
    /// flowing down, |v−seamV| for a floor spreading from the wall seam). Only the
    /// fresh band is written, so dabs that landed on already-covered texels survive.
    /// `fingerAmp`/`fingerFreq` give the leading edge its dripping rivulets.
    public func floodFront(newColour: SIMD3<Float>, axis: FloodAxis,
                           prevPos: Float, curPos: Float, seamV: Float = 0,
                           fingerAmp: Float = 0.12, fingerFreq: Float = 90,
                           feather: Float = 0.02, wetness: Float = 1.0,
                           seed: Float = 0) {
        guard curPos > prevPos else { return }
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = "PaintAccumulation.flood"
        enc.setComputePipelineState(floodPipeline)
        enc.setTexture(texture, index: 0)
        var u = FloodUniforms(
            color: SIMD4(newColour, wetness),
            front: SIMD4(prevPos, curPos, seamV, axis == .wallDown ? 0 : 1),
            drip:  SIMD4(seed, fingerAmp, fingerFreq, feather))
        enc.setBytes(&u, length: MemoryLayout<FloodUniforms>.stride, index: 0)
        dispatch2D(enc, pipeline: floodPipeline)
        enc.endEncoding()
        cb.commit()
    }

    /// Fraction of texels (0…1) whose RGB matches `colour` within `tolerance`.
    /// DIAGNOSTIC ONLY — synchronous read-back; never call on the render path.
    public func coverageFraction(of colour: SIMD3<Float>, tolerance: Float = 0.06) -> Float {
        guard let counter = engine.device.makeBuffer(
            length: MemoryLayout<UInt32>.stride * 2, options: .storageModeShared) else { return 0 }
        memset(counter.contents(), 0, MemoryLayout<UInt32>.stride * 2)
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return 0 }
        enc.setComputePipelineState(coveragePipeline)
        enc.setTexture(texture, index: 0)
        var t = SIMD4<Float>(colour, tolerance)
        enc.setBytes(&t, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        enc.setBuffer(counter, offset: 0, index: 1)
        dispatch2D(enc, pipeline: coveragePipeline)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: diagnostic read-back, not the render path
        let p = counter.contents().bindMemory(to: UInt32.self, capacity: 2)
        return Float(p[0]) / Float(max(1, p[1]))
    }

    /// Per-column deepest v (0…1) covered by `colour` (−1 = none). DIAGNOSTIC ONLY —
    /// synchronous read-back. Reveals the flood front's leading-edge geometry.
    public func frontProfile(of colour: SIMD3<Float>, tolerance: Float = 0.06) -> [Float] {
        guard let buf = engine.device.makeBuffer(
            length: MemoryLayout<Float>.stride * width, options: .storageModeShared) else { return [] }
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return [] }
        enc.setComputePipelineState(frontProfilePipeline)
        enc.setTexture(texture, index: 0)
        var t = SIMD4<Float>(colour, tolerance)
        enc.setBytes(&t, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        enc.setBuffer(buf, offset: 0, index: 1)
        let w = frontProfilePipeline.threadExecutionWidth
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: diagnostic read-back, not the render path
        let p = buf.contents().bindMemory(to: Float.self, capacity: width)
        return Array(UnsafeBufferPointer(start: p, count: width))
    }

    private func dispatch2D(_ enc: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState) {
        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        let tg = MTLSize(width: w, height: h, depth: 1)
        let grid = MTLSize(width: width, height: height, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
    }
}
