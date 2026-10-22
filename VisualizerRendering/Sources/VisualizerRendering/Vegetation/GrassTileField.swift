import Metal
import simd

/// A GPU-generated, view-dependent grass field (the Ghost of Tsushima scheme).
///
/// Where `GrassRenderSet` simulates a fixed scatter of blades with XPBD, this field has no
/// simulation and no fixed scatter: each frame a compute kernel (`grassTileGenerate`) writes a
/// camera-facing ribbon along a Bézier for every blade SLOT, and the slots are arranged in
/// rings around the camera whose tiles double in size — so the blade COUNT per tile is constant,
/// density falls 4× per ring and the surviving blades widen. That is how a field can be dense
/// under your feet and still read as grass, not hairs, at 15 m. See `GrassTiles.metal`.
///
/// Blades are a pure function of their world cell, so the lawn is stable as the camera moves.
/// The mesh registers through the same GPU-mesh path as the PBD ribbons (G-buffer, shadow pass,
/// RT BLAS refit), with a per-vertex colour buffer the kernel writes.
@MainActor
public final class GrassTileField {

    public struct Params {
        /// One ring of tiles. `window` = cells per ring side (≥ 25 so the ring's disc fits it);
        /// `bladesPerCell` = blade slots per cell. Ring r's cell is `baseCellSize · 2^r`, so a ring
        /// covers a disc of radius `0.46 · window · cell`.
        public struct Ring { public var window: Int; public var bladesPerCell: Int
            public init(window: Int, bladesPerCell: Int) { self.window = window; self.bladesPerCell = bladesPerCell } }
        /// Near rings are dense; far rings use a smaller window — their cells are metres wide, so a
        /// 40-cell window still reaches tens of metres. Reach with the defaults ≈ 59 m.
        public var rings: [Ring] = [Ring(window: 64, bladesPerCell: 24), Ring(window: 64, bladesPerCell: 26),
                                    Ring(window: 64, bladesPerCell: 30), Ring(window: 48, bladesPerCell: 44),
                                    Ring(window: 48, bladesPerCell: 48), Ring(window: 40, bladesPerCell: 48),
                                    Ring(window: 40, bladesPerCell: 48), Ring(window: 40, bladesPerCell: 48)]
        /// Ribbon segments per blade (rows = segments + 1).
        public var segments: Int = 2
        /// Ring-0 cell size (m); ring r uses `baseCellSize · 2^r`.
        public var baseCellSize: Float = 0.04
        public var baseHalfWidth: Float = 0.0026
        public var heightMin: Float = 0.035
        public var heightRange: Float = 0.035
        /// Half-width multiplier per ring — what keeps the far field's mass as density falls.
        public var widenPerRing: Float = 1.4
        /// Wind tip amplitude (m) and heading (rad).
        public var windAmp: Float = 0.012
        public var windHeading: Float = atan2(0.6, 0.8)
        public var groundY: Float = 0
        /// Root → mid → tip colour, ALREADY divided by the instance albedo the host registers.
        public var rootColor: SIMD3<Float> = SIMD3(0.4, 0.4, 0.4)
        public var midColor: SIMD3<Float> = SIMD3(0.5, 0.5, 0.5)
        public var tipColor: SIMD3<Float> = SIMD3(0.6, 0.6, 0.6)
        /// Vertex alpha: 0 = the engine's foliage thin-sheet (backlit glow) flag, 1 = opaque.
        public var foliageSSS: Bool = true
        /// How far the outer rings converge on `midColor` (0 = never, 1 = fully).
        public var farTint: Float = 0.85
        /// Distance band (m) over which blades converge on `midColor` (scaled by `farTint`).
        public var farRange: SIMD2<Float> = SIMD2(2, 40)
        /// Brightness lift reached at the far end of `liftRange` (1 = none). A dense sward shades
        /// darker than the open ground it stands on, so far blades are lifted to meet it.
        public var farLift: Float = 1.9
        public var liftRange: SIMD2<Float> = SIMD2(3, 30)
        /// Blade HEIGHT growth per ring (×(1 + this·r)), so a widened far blade keeps a blade's
        /// proportions. 0 = every ring as tall as ring 0 (a mown lawn stays mown in the distance).
        public var heightPerRing: Float = 0.28
        /// Root darkening toward the ground: the lowest third of every blade is multiplied down to
        /// this (1 = none). Reeves & Blau root occlusion — the sward's base is in its own shadow.
        public var rootOcclusion: Float = 1.0
        /// How far a distant blade's shading normal is bent toward the ground's (0.2 near → this far),
        /// over `normalUpRange` (m). Near 1 a far blade lights exactly like the turf it stands on, so a
        /// low sun cannot pick it out as a bright fleck.
        public var farNormalUp: Float = 0.60
        public var normalUpRange: SIMD2<Float> = SIMD2(1.5, 45)
        /// Steep-view thinning: blades whose cos(elevation from the camera) is below `steepCos.x` are
        /// dropped, all kept above `steepCos.y`; it comes in over `steepRange` metres (horizontal) so
        /// the blades underfoot always stay. `x ≥ y` (the default) disables it.
        public var steepCos: SIMD2<Float> = SIMD2(1, 1)
        public var steepRange: SIMD2<Float> = SIMD2(2, 6)
        /// Fraction of the slots the output buffers can hold. Only blades inside the view cone are
        /// written (compacted), so the buffers need room for the worst view, not for every slot;
        /// a blade past capacity is dropped (far rings first, as they claim space last).
        public var visibleFraction: Float = 0.6
        public init() {}
    }

    public let bladeCount: Int
    public let vertexCount: Int
    public private(set) var meshHandle: IlluminatoramaMeshHandle!

    private let engine: SimEngine
    private let pipeline: MTLComputePipelineState
    private let uniformBuffer: MTLBuffer
    private let resetPipeline: MTLComputePipelineState
    private let finalizePipeline: MTLComputePipelineState
    /// Finished `IlluminatoramaVertex` records, written by the kernel (no engine repack).
    private let vertexBuffer: MTLBuffer
    private let counterBuffer: MTLBuffer
    private let drawArgsBuffer: MTLBuffer
    private let indexBuffer: MTLBuffer
    private let indexCount: Int
    /// Blades the vertex buffer can hold, and indices one blade draws.
    public let capacity: Int
    private let indicesPerBlade: Int
    private let maskTexture: MTLTexture
    private let groundTexture: MTLTexture
    private var params: Params
    private var maskRect: SIMD4<Float>
    private weak var renderer: IlluminatoramaRenderer?
    private var isTornDown = false

    /// - Parameters:
    ///   - maskTexture: `.r8Unorm` coverage (0 = no grass) over `maskMin … maskMin + maskSize` in XZ.
    ///   - groundTexture: `.r32Float` surface height over the same rect (blade roots bed onto it).
    public init?(renderer: IlluminatoramaRenderer, engine: SimEngine = .shared, params: Params,
                 maskTexture: MTLTexture, groundTexture: MTLTexture,
                 maskMin: SIMD2<Float>, maskSize: SIMD2<Float>) {
        let device = engine.device
        guard !params.rings.isEmpty, params.rings.count <= 8,
              params.rings.allSatisfy({ $0.window >= 25 && $0.bladesPerCell >= 1 }), params.segments >= 1,
              let pso = engine.pipelineCache.pipelineState(name: "grassTileGenerate", device: device),
              let uBuf = device.makeBuffer(length: MemoryLayout<GrassTileUniforms>.stride,
                                           options: .storageModeShared)
        else { return nil }

        let blades = params.rings.reduce(0) { $0 + $1.window * $1.window * $1.bladesPerCell }
        let capacity = max(1, min(blades, Int(Float(blades) * min(max(params.visibleFraction, 0.05), 1))))
        let rows = params.segments + 1
        let verts = capacity * rows * 2
        guard let vBuf = device.makeBuffer(length: verts * MemoryLayout<IlluminatoramaVertex>.stride,
                                           options: .storageModeShared),
              let cntBuf = device.makeBuffer(length: 4, options: .storageModeShared),
              let argBuf = device.makeBuffer(length: 20, options: .storageModeShared),
              let resetPSO = engine.pipelineCache.pipelineState(name: "grassTileReset", device: device),
              let finPSO = engine.pipelineCache.pipelineState(name: "grassTileFinalize", device: device),
              // The descriptor wants position/normal streams even though a direct mesh never reads them.
              let dummy = device.makeBuffer(length: 16, options: .storageModePrivate)
        else { return nil }

        var indices = [UInt32]()
        indices.reserveCapacity(capacity * params.segments * 6)
        for b in 0..<capacity {
            let base = UInt32(b * rows * 2)
            for i in 0..<params.segments {
                let bl = base + UInt32(i * 2)
                indices.append(contentsOf: [bl, bl + 2, bl + 1, bl + 1, bl + 2, bl + 3])
            }
        }
        guard let iBuf = indices.withUnsafeBufferPointer({ ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: MemoryLayout<UInt32>.stride * indices.count,
                              options: .storageModeShared)
        }) else { return nil }

        uBuf.label = "GrassTiles.uniforms"; vBuf.label = "GrassTiles.vertices"
        cntBuf.label = "GrassTiles.counter"; argBuf.label = "GrassTiles.drawArgs"; iBuf.label = "GrassTiles.indices"

        self.engine = engine
        self.pipeline = pso
        self.uniformBuffer = uBuf
        self.resetPipeline = resetPSO
        self.finalizePipeline = finPSO
        self.vertexBuffer = vBuf
        self.counterBuffer = cntBuf
        self.drawArgsBuffer = argBuf
        self.capacity = capacity
        self.indicesPerBlade = params.segments * 6
        self.indexBuffer = iBuf
        self.indexCount = indices.count
        self.maskTexture = maskTexture
        self.groundTexture = groundTexture
        self.params = params
        self.maskRect = SIMD4(maskMin.x, maskMin.y, 1 / max(maskSize.x, 1e-3), 1 / max(maskSize.y, 1e-3))
        self.renderer = renderer
        self.bladeCount = blades
        self.vertexCount = verts

        // Fill the buffers once BEFORE registering, so the first presented frame has real blades.
        encodeGenerate(camera: renderer.camera.position, time: 0)

        let desc = IlluminatoramaGPUMeshDescriptor(
            positionBuffer: dummy, normalBuffer: dummy,
            vertexCount: verts,
            bodyIndexBuffer: iBuf, bodyIndexCount: indices.count, bodyIndexType: .uint32,
            doubleSided: true,
            shadowCastsBothFaces: true,
            directVertexBuffer: vBuf,
            indirectDrawBuffer: argBuf)
        guard let handle = renderer.registerGPUMesh(desc) else { return nil }
        self.meshHandle = handle
    }

    /// The kernel writes through `encodeGenerate`; a tile field has no sim, so a "step" is one dispatch.
    ///   - windScale: multiplies `params.windAmp` — the host's gust strength / its calibration, so the
    ///     field obeys the same sway switch as the trees (0 freezes it).
    public func step(camera: SIMD3<Float>, time: Float, windScale: Float = 1) {
        guard !isTornDown else { return }
        encodeGenerate(camera: camera, time: time, windScale: windScale)
    }

    private func encodeGenerate(camera: SIMD3<Float>, time: Float, windScale: Float = 1, viewCull: Bool = true) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = "GrassTiles.generate"
        enc.label = "GrassTiles.generate"
        let p = params
        let planes = viewCull ? viewPlanes(camera: camera) : Array(repeating: SIMD4<Float>(0, 0, 0, 1e9), count: 4)
        uniformBuffer.contents().bindMemory(to: GrassTileUniforms.self, capacity: 1).pointee =
            GrassTileUniforms(
                camera: SIMD4(camera, time),
                shape: SIMD4(p.groundY, p.windAmp * windScale, p.windHeading, p.widenPerRing),
                sizes: SIMD4(p.baseCellSize, p.baseHalfWidth, p.heightMin, p.heightRange),
                maskRect: maskRect,
                rootC: SIMD4(p.rootColor, 0), midC: SIMD4(p.midColor, 0), tipC: SIMD4(p.tipColor, 0),
                misc: SIMD4(p.foliageSSS ? 0 : 1, p.farTint, p.farLift, p.heightPerRing),
                ringInfo: GrassTileUniforms.RingInfo(p.rings),
                ringCount: UInt32(p.rings.count), segments: UInt32(p.segments),
                capacity: UInt32(capacity), pad1: 0,
                plane0: planes[0], plane1: planes[1], plane2: planes[2], plane3: planes[3],
                tune: SIMD4(p.farRange.x, p.farRange.y, p.liftRange.x, p.liftRange.y),
                tune2: SIMD4(p.rootOcclusion, p.farNormalUp, p.normalUpRange.x, p.normalUpRange.y),
                tune3: SIMD4(p.steepCos.x, p.steepCos.y, p.steepRange.x, p.steepRange.y))
        enc.setComputePipelineState(resetPipeline)
        enc.setBuffer(counterBuffer, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(vertexBuffer, offset: 0, index: 0)
        enc.setBuffer(counterBuffer, offset: 0, index: 1)
        enc.setBuffer(uniformBuffer, offset: 0, index: 3)
        enc.setTexture(maskTexture, index: 0)
        enc.setTexture(groundTexture, index: 1)
        let w = min(bladeCount, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: bladeCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.setComputePipelineState(finalizePipeline)
        var info = SIMD2<UInt32>(UInt32(indicesPerBlade), UInt32(capacity))
        enc.setBuffer(counterBuffer, offset: 0, index: 0)
        enc.setBuffer(drawArgsBuffer, offset: 0, index: 1)
        enc.setBytes(&info, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
    }

    /// Four inward-facing planes through the camera bounding its view cone, widened ~20 % so a
    /// jittered/supersampled frame or a camera that moved since this step never exposes a missing blade.
    /// An orthographic camera gets planes that pass everything.
    private func viewPlanes(camera: SIMD3<Float>) -> [SIMD4<Float>] {
        guard let cam = renderer?.camera, !cam.isOrthographic else {
            return Array(repeating: SIMD4<Float>(0, 0, 0, 1e9), count: 4)
        }
        let f = simd_normalize(cam.target - cam.position)
        let r = simd_normalize(simd_cross(f, cam.up))
        let u = simd_cross(r, f)
        let tanY = tan(cam.fovYRadians * 0.5) * 1.2
        let tanX = tanY * max(cam.aspect, 1) * 1.1
        func plane(_ n: SIMD3<Float>) -> SIMD4<Float> {
            let nn = simd_normalize(n)
            return SIMD4(nn, -simd_dot(nn, camera))
        }
        return [plane(r + f * tanX), plane(-r + f * tanX), plane(u + f * tanY), plane(-u + f * tanY)]
    }

    /// Blades the last generate kept (drains the queue first).
    public func keptBladeCount() -> Int {
        if let cb = engine.commandQueue.makeCommandBuffer() { cb.commit(); cb.waitUntilCompleted() }
        return min(Int(counterBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee), capacity)
    }

    /// Every kept blade as (root midpoint, tip midpoint), sorted by root so two generations compare
    /// index-for-index (the kernel's output order is whatever the atomics produced).
    private func readKeptBlades() -> [(root: SIMD3<Float>, tip: SIMD3<Float>)] {
        let kept = keptBladeCount()
        let rows = params.segments + 1
        let verts = vertexBuffer.contents().bindMemory(to: IlluminatoramaVertex.self, capacity: capacity * rows * 2)
        var out: [(root: SIMD3<Float>, tip: SIMD3<Float>)] = []
        out.reserveCapacity(kept)
        for b in 0..<kept {
            let v = b * rows * 2
            out.append(((verts[v].position + verts[v + 1].position) * 0.5,
                        (verts[v + (rows - 1) * 2].position + verts[v + (rows - 1) * 2 + 1].position) * 0.5))
        }
        return out.sorted { a, b in
            a.root.x != b.root.x ? a.root.x < b.root.x : (a.root.z != b.root.z ? a.root.z < b.root.z : a.root.y < b.root.y)
        }
    }

    /// TEST-OBSERVABLE: world position of every KEPT blade's root (all views — no view cull applied by the
    /// caller's generate; see `bladeRoots(cameraAt:)`). Drains the sim queue first.
    public func readBladeRoots() -> [SIMD3<Float>] { readKeptBlades().map(\.root) }

    /// TEST-OBSERVABLE: tip positions (last row midpoint) of every KEPT blade, sorted by root, after
    /// regenerating at `time` with the given wind scale — lets a gate compare two times/scales.
    public func bladeTips(cameraAt camera: SIMD3<Float>, time: Float, windScale: Float) -> [SIMD3<Float>] {
        encodeGenerate(camera: camera, time: time, windScale: windScale, viewCull: false)
        return readKeptBlades().map(\.tip)
    }

    /// TEST-OBSERVABLE: the roots this field would grow with the camera at `camera` (regenerates with NO
    /// view cull, then reads). The field is view-dependent by design — it only exists around the viewer —
    /// so a test asks "what grows HERE" rather than "what grows in the yard".
    public func bladeRoots(cameraAt camera: SIMD3<Float>) -> [SIMD3<Float>] {
        encodeGenerate(camera: camera, time: 0, viewCull: false)
        return readBladeRoots()
    }

    /// TEST-OBSERVABLE: blades kept for the CURRENT view (view cull on) with the camera at `camera`.
    public func keptBladeCount(cameraAt camera: SIMD3<Float>) -> Int {
        encodeGenerate(camera: camera, time: 0)
        return keptBladeCount()
    }

    public func teardown() {
        guard !isTornDown else { return }
        isTornDown = true
        renderer?.removeMesh(meshHandle.kind)
        renderer = nil
    }
}

/// Mirror of `GrassTileUniforms` in GrassTiles.metal — keep field order in lockstep.
struct GrassTileUniforms {
    var camera: SIMD4<Float>
    var shape: SIMD4<Float>
    var sizes: SIMD4<Float>
    var maskRect: SIMD4<Float>
    var rootC: SIMD4<Float>
    var midC: SIMD4<Float>
    var tipC: SIMD4<Float>
    var misc: SIMD4<Float>
    var ringInfo: RingInfo
    var ringCount: UInt32
    var segments: UInt32
    var capacity: UInt32
    var pad1: UInt32
    var plane0: SIMD4<Float>, plane1: SIMD4<Float>, plane2: SIMD4<Float>, plane3: SIMD4<Float>
    var tune: SIMD4<Float>
    var tune2: SIMD4<Float>
    var tune3: SIMD4<Float>

    /// Eight `uint4` (x = window, y = blades per cell) laid out as a Metal `uint4[8]`.
    struct RingInfo {
        var r0: SIMD4<UInt32>, r1: SIMD4<UInt32>, r2: SIMD4<UInt32>, r3: SIMD4<UInt32>
        var r4: SIMD4<UInt32>, r5: SIMD4<UInt32>, r6: SIMD4<UInt32>, r7: SIMD4<UInt32>
        init(_ rings: [GrassTileField.Params.Ring]) {
            func v(_ i: Int) -> SIMD4<UInt32> {
                i < rings.count ? SIMD4(UInt32(rings[i].window), UInt32(rings[i].bladesPerCell), 0, 0) : .zero
            }
            r0 = v(0); r1 = v(1); r2 = v(2); r3 = v(3); r4 = v(4); r5 = v(5); r6 = v(6); r7 = v(7)
        }
    }
}
