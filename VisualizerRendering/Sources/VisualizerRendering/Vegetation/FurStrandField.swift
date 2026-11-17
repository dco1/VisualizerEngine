import Foundation
import Metal
import simd

/// GPU fur on a breathing skin patch (sibling of `GrassTileField`).
///
/// One compute strand per hair: `furStrandGenerate` roots it on the SAME analytic skin function the skin
/// mesh uses — so a breath moves every root — clumps it into locks, combs it along the flow field,
/// bends it by the four surrounding GUIDE hairs and writes a camera-facing ribbon. Wind lives in the
/// guides: a coarse grid of damped springs (`furGuideStep`) pushed by a travelling gust field and
/// coupled to their neighbours, so a gust is a visible front crossing the coat. The ribbon's normal
/// slot carries the strand tangent with colour alpha 0.995 — the flag the deferred lighting pass turns
/// into the Chiang R/TT/TRT hair BSDF (`IlluminatoramaHairBSDF.h`). See `FurStrands.metal`.
///
/// // plausibility: shortcut — each rendered "strand" stands for a bundle of real fibres (a coat is
/// ~10⁴ hairs/cm²; this draws ~200/cm² (340k over 0.18 m²) as ≥0.7-px ribbons, with stochastic sub-pixel thinning and a
/// root-to-tip AO ramp instead of deep-opacity self-shadowing). Tried: real-density curves — ~100× the
/// vertex budget. Compromise NOT yet approved by Danny (autonomous run 2026-10-26); flagged in the report.
@MainActor
public final class FurStrandField {
    /// Ribbon segments per strand — mirror of `FUR_SEGMENTS` in FurStrands.metal.
    static let segments = 5
    static let vertsPerStrand = (segments + 1) * 2

    public struct Params {
        /// Skin patch size (m): along the spine × back→belly, and how high the arch rises across it.
        public var patchSize = SIMD2<Float>(0.52, 0.34)
        public var archHeight: Float = 0.05
        public var strandCount = 340_000
        public var undercoatLength: Float = 0.016
        public var guardLength: Float = 0.030
        public var guardFraction: Float = 0.14
        /// 0…1 — how far strands gather into locks.
        public var clump: Float = 0.55
        public var lockSize: Float = 0.006
        /// True fibre width (m); the ribbon is never drawn thinner than ~0.7 px.
        public var strandWidth: Float = 0.00030
        /// Comb direction in the skin tangent plane (rad, 0 = along the spine) and its waviness.
        public var combAngle: Float = 0.35
        public var waviness: Float = 0.35
        public var tint = SIMD3<Float>(1, 1, 1)
        public var colorJitter: Float = 0.35
        /// Fibre colours along the shaft (root band, mid-shaft, undercoat tip, guard tip) — one scheme's palette.
        public var rootColor = SIMD3<Float>(0.95, 0.62, 0.30)
        public var midColor = SIMD3<Float>(0.90, 0.36, 0.06)
        public var tipColor = SIMD3<Float>(0.62, 0.20, 0.04)
        public var guardTipColor = SIMD3<Float>(0.20, 0.07, 0.03)
        /// Tabby stripe strength 0…1, stripe frequency (cycles/m along the spine), belly lightening 0…1.
        public var tabby: Float = 0
        public var tabbyFrequency: Float = 55
        public var bellyLight: Float = 0.5
        /// Touch: contact radius (m).
        public var touchRadius: Float = 0.035
        // Wind
        public var windHeading: Float = 0.6
        public var windStrength: Float = 1.0
        public var gustScale: Float = 2.2
        public var gustSpeed: Float = 0.7
        // Breathing (the phase itself is passed to `step`, accumulated by the host so rate changes stay C¹)
        public var breathDepth: Float = 0.022
        public var ribcageCenter: Float = 0.45
        public var ribcageRadius: Float = 0.32
        // Guide-hair dynamics
        public var guideGrid = SIMD2<Int>(72, 48)
        public var springK: Float = 90
        public var damping: Float = 6
        public var coupling: Float = 55
        public init() {}
    }

    public var params: Params
    /// Contact point on the skin (world), eased pressure 0…1 and contact velocity (world xz, m/s) — written by
    /// the host each frame from the pointer; pressure 0 = no contact (the kernels then cost nothing extra).
    public var touchPoint = SIMD3<Float>(0, -10, 0)
    public var touchPressure: Float = 0
    public var touchVelocity = SIMD2<Float>(0, 0)
    public private(set) var strandMesh: IlluminatoramaMeshHandle!
    public private(set) var skinMesh: IlluminatoramaMeshHandle!

    private let engine: SimEngine
    private let guidePSO: MTLComputePipelineState
    private let skinPSO: MTLComputePipelineState
    private let strandPSO: MTLComputePipelineState
    private let uniformBuffer: MTLBuffer
    private let guideA: MTLBuffer
    private let guideB: MTLBuffer
    private let strandVertices: MTLBuffer
    private let skinVertices: MTLBuffer
    private let skinGrid = SIMD2<Int>(180, 120)
    private let strandCols: Int
    private let strandCount: Int
    private weak var renderer: IlluminatoramaRenderer?
    private var torn = false

    public init?(renderer: IlluminatoramaRenderer, engine: SimEngine = .shared, params: Params) {
        let dev = engine.device
        let cache = engine.pipelineCache
        guard let g = cache.pipelineState(name: "furGuideStep", device: dev),
              let s = cache.pipelineState(name: "furSkinGenerate", device: dev),
              let t = cache.pipelineState(name: "furStrandGenerate", device: dev),
              let ub = dev.makeBuffer(length: MemoryLayout<FurUniforms>.stride, options: .storageModeShared)
        else { return nil }
        let guides = params.guideGrid.x * params.guideGrid.y
        let n = max(1, params.strandCount)
        let vStride = MemoryLayout<IlluminatoramaVertex>.stride
        let sv = skinGrid.x * skinGrid.y
        guard let ga = dev.makeBuffer(length: guides * 16, options: .storageModeShared),
              let gb = dev.makeBuffer(length: guides * 16, options: .storageModeShared),
              let stv = dev.makeBuffer(length: n * Self.vertsPerStrand * vStride, options: .storageModeShared),
              let skv = dev.makeBuffer(length: sv * vStride, options: .storageModeShared),
              let dummy = dev.makeBuffer(length: 16, options: .storageModePrivate)
        else { return nil }
        memset(ga.contents(), 0, ga.length); memset(gb.contents(), 0, gb.length)
        memset(stv.contents(), 0, stv.length); memset(skv.contents(), 0, skv.length)

        var si = [UInt32](); si.reserveCapacity(n * Self.segments * 6)
        for i in 0..<n {
            let base = UInt32(i * Self.vertsPerStrand)
            for r in 0..<Self.segments {
                let b = base + UInt32(r * 2)
                si.append(contentsOf: [b, b + 2, b + 1, b + 1, b + 2, b + 3])
            }
        }
        var ki = [UInt32](); ki.reserveCapacity(skinGrid.x * skinGrid.y * 6)
        for y in 0..<(skinGrid.y - 1) {
            for x in 0..<(skinGrid.x - 1) {
                let a = UInt32(y * skinGrid.x + x), b = a + 1, c = a + UInt32(skinGrid.x), d = c + 1
                ki.append(contentsOf: [a, c, b, b, c, d])
            }
        }
        guard let sIdx = si.withUnsafeBytes({ dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
              let kIdx = ki.withUnsafeBytes({ dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        else { return nil }

        self.engine = engine; self.params = params; self.renderer = renderer
        guidePSO = g; skinPSO = s; strandPSO = t; uniformBuffer = ub
        guideA = ga; guideB = gb; strandVertices = stv; skinVertices = skv
        strandCount = n
        let aspect = params.patchSize.x / params.patchSize.y
        strandCols = max(1, Int((Float(n) * aspect).squareRoot().rounded(.up)))

        // Fill once before registering so the first presented frame already has fur.
        encode(dt: 1.0 / 60.0, time: 0, camera: renderer.camera.position, breathPhase: 0, substeps: 1)

        let sd = IlluminatoramaGPUMeshDescriptor(
            positionBuffer: dummy, normalBuffer: dummy, vertexCount: n * Self.vertsPerStrand,
            bodyIndexBuffer: sIdx, bodyIndexCount: si.count, bodyIndexType: .uint32,
            doubleSided: true, shadowCastsBothFaces: true, directVertexBuffer: stv)
        let kd = IlluminatoramaGPUMeshDescriptor(
            positionBuffer: dummy, normalBuffer: dummy, vertexCount: sv,
            bodyIndexBuffer: kIdx, bodyIndexCount: ki.count, bodyIndexType: .uint32,
            doubleSided: true, shadowCastsBothFaces: true, directVertexBuffer: skv)
        guard let h1 = renderer.registerGPUMesh(sd), let h2 = renderer.registerGPUMesh(kd) else { return nil }
        strandMesh = h1; skinMesh = h2
    }

    /// One frame: advance the guide hairs, rewrite the skin and every strand. `breathPhase` is in cycles.
    public func step(dt: Float, time: Float, camera: SIMD3<Float>, breathPhase: Float) {
        guard !torn else { return }
        encode(dt: dt, time: time, camera: camera, breathPhase: breathPhase, substeps: 2)
    }

    private func encode(dt: Float, time: Float, camera: SIMD3<Float>, breathPhase: Float, substeps: Int) {
        let p = params
        let cam = renderer?.camera
        let fov = cam?.fovYRadians ?? 0.66
        let h = Float(max(renderer?.outputHeight ?? 1080, 1))
        let foot = 2 * tan(fov * 0.5) / h
        let sub = max(substeps, 1)
        let u = FurUniforms(
            patch: SIMD4(p.patchSize.x, p.patchSize.y, p.archHeight, time),
            breath: SIMD4(breathPhase, p.breathDepth, p.ribcageCenter, p.ribcageRadius),
            coat: SIMD4(p.undercoatLength, p.guardLength, p.guardFraction, p.clump),
            look: SIMD4(p.strandWidth, p.lockSize, p.combAngle, p.waviness),
            wind: SIMD4(p.windHeading, p.windStrength, p.gustScale, p.gustSpeed),
            baseColor: SIMD4(p.tint, p.colorJitter),
            camera: SIMD4(camera, foot),
            counts: SIMD4(UInt32(strandCount), UInt32(strandCols), UInt32(p.guideGrid.x), UInt32(p.guideGrid.y)),
            skinGrid: SIMD4(UInt32(skinGrid.x), UInt32(skinGrid.y), 0, 0),
            sim: SIMD4(min(dt, 1.0 / 30.0) / Float(sub), p.springK, p.damping, p.coupling),
            colRoot: SIMD4(p.rootColor, 0), colMid: SIMD4(p.midColor, 0),
            colTip: SIMD4(p.tipColor, 0), colGuardTip: SIMD4(p.guardTipColor, 0),
            pattern: SIMD4(p.tabby, p.tabbyFrequency, p.bellyLight, 0),
            touch: SIMD4(touchPoint, p.touchRadius),
            touchState: SIMD4(touchPressure, touchVelocity.x, touchVelocity.y, 0))
        uniformBuffer.contents().bindMemory(to: FurUniforms.self, capacity: 1).pointee = u

        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = "Fur.generate"
        let gw = p.guideGrid.x, gh = p.guideGrid.y
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        var src = guideA, dst = guideB
        enc.setComputePipelineState(guidePSO)
        for _ in 0..<sub {
            enc.setBuffer(src, offset: 0, index: 0)
            enc.setBuffer(dst, offset: 0, index: 1)
            enc.setBuffer(uniformBuffer, offset: 0, index: 2)
            enc.dispatchThreads(MTLSize(width: gw, height: gh, depth: 1), threadsPerThreadgroup: tg)
            swap(&src, &dst)
        }
        // After an even number of substeps the newest state is back in `guideA`; after an odd one in `guideB`.
        enc.setComputePipelineState(skinPSO)
        enc.setBuffer(skinVertices, offset: 0, index: 0)
        enc.setBuffer(uniformBuffer, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: skinGrid.x, height: skinGrid.y, depth: 1), threadsPerThreadgroup: tg)
        enc.setComputePipelineState(strandPSO)
        enc.setBuffer(strandVertices, offset: 0, index: 0)
        enc.setBuffer(src, offset: 0, index: 1)
        enc.setBuffer(uniformBuffer, offset: 0, index: 3)
        let w = min(strandCount, strandPSO.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: strandCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
    }

    /// Skin height at world (x, z) — CPU mirror of `fur_skin` in FurStrands.metal (no touch dent), used only to
    /// pick where a pointer ray meets the coat. Keep in lockstep with the shader's arch + breath terms.
    public func skinHeight(x: Float, z: Float, breathPhase: Float) -> Float {
        let p = params
        let u = x / p.patchSize.x + 0.5, v = z / p.patchSize.y + 0.5
        let c = 2 * (v - 0.5)
        var y = p.archHeight * (1 - c * c)
        let b = 0.5 - 0.5 * cos(2 * Float.pi * breathPhase)
        let dx = (u - p.ribcageCenter) / max(p.ribcageRadius, 1e-3), dy = (v - 0.5) / max(p.ribcageRadius, 1e-3)
        let rib = exp(-(dx * dx + dy * dy))
        let t = (u - p.ribcageCenter) / (1.8 * max(p.ribcageRadius, 1e-3))
        y += p.breathDepth * b * (rib + 0.35 * exp(-t * t))
        return y
    }

    /// Where a ray meets the coat (the top of the fur, `furLift` m above the skin), or nil if it misses the patch.
    public func pick(origin: SIMD3<Float>, direction: SIMD3<Float>, breathPhase: Float) -> SIMD3<Float>? {
        let lift = params.guardLength * 0.6
        let half = SIMD2<Float>(params.patchSize.x, params.patchSize.y) * 0.5
        var prevT: Float = 0, prevD: Float? = nil
        var t: Float = 0.02
        while t < 4 {
            let q = origin + direction * t
            let d = q.y - (skinHeight(x: q.x, z: q.z, breathPhase: breathPhase) + lift)
            if let pd = prevD, pd > 0, d <= 0 {
                var lo = prevT, hi = t
                for _ in 0..<12 {
                    let mid = 0.5 * (lo + hi)
                    let m = origin + direction * mid
                    if m.y - (skinHeight(x: m.x, z: m.z, breathPhase: breathPhase) + lift) > 0 { lo = mid } else { hi = mid }
                }
                let hit = origin + direction * (0.5 * (lo + hi))
                guard abs(hit.x) <= half.x, abs(hit.z) <= half.y else { return nil }
                return hit
            }
            prevD = d; prevT = t
            t += 0.004
        }
        return nil
    }

    public func teardown() {
        guard !torn else { return }
        torn = true
        if let r = renderer {
            if let h = strandMesh { r.removeMesh(h.kind) }
            if let h = skinMesh { r.removeMesh(h.kind) }
        }
        renderer = nil
    }
}

/// Mirror of `FurUniforms` in FurStrands.metal — keep field order in lockstep.
struct FurUniforms {
    var patch: SIMD4<Float>
    var breath: SIMD4<Float>
    var coat: SIMD4<Float>
    var look: SIMD4<Float>
    var wind: SIMD4<Float>
    var baseColor: SIMD4<Float>
    var camera: SIMD4<Float>
    var counts: SIMD4<UInt32>
    var skinGrid: SIMD4<UInt32>
    var sim: SIMD4<Float>
    var colRoot: SIMD4<Float>
    var colMid: SIMD4<Float>
    var colTip: SIMD4<Float>
    var colGuardTip: SIMD4<Float>
    var pattern: SIMD4<Float>
    var touch: SIMD4<Float>
    var touchState: SIMD4<Float>
}
