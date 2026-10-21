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
        public var rings: [Ring] = [Ring(window: 64, bladesPerCell: 20), Ring(window: 64, bladesPerCell: 20),
                                    Ring(window: 64, bladesPerCell: 22), Ring(window: 48, bladesPerCell: 32),
                                    Ring(window: 48, bladesPerCell: 30), Ring(window: 40, bladesPerCell: 28),
                                    Ring(window: 40, bladesPerCell: 26), Ring(window: 40, bladesPerCell: 24)]
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
        public init() {}
    }

    public let bladeCount: Int
    public let vertexCount: Int
    public private(set) var meshHandle: IlluminatoramaMeshHandle!

    private let engine: SimEngine
    private let pipeline: MTLComputePipelineState
    private let uniformBuffer: MTLBuffer
    private let positionBuffer: MTLBuffer
    private let normalBuffer: MTLBuffer
    private let colorBuffer: MTLBuffer
    private let indexBuffer: MTLBuffer
    private let indexCount: Int
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
        let rows = params.segments + 1
        let verts = blades * rows * 2
        let vecBytes = MemoryLayout<Float>.stride * 3
        guard let pBuf = device.makeBuffer(length: verts * vecBytes, options: .storageModeShared),
              let nBuf = device.makeBuffer(length: verts * vecBytes, options: .storageModeShared),
              let cBuf = device.makeBuffer(length: verts * MemoryLayout<SIMD4<Float>>.stride,
                                           options: .storageModeShared)
        else { return nil }

        var indices = [UInt32]()
        indices.reserveCapacity(blades * params.segments * 6)
        for b in 0..<blades {
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

        uBuf.label = "GrassTiles.uniforms"; pBuf.label = "GrassTiles.positions"
        nBuf.label = "GrassTiles.normals"; cBuf.label = "GrassTiles.colors"; iBuf.label = "GrassTiles.indices"

        self.engine = engine
        self.pipeline = pso
        self.uniformBuffer = uBuf
        self.positionBuffer = pBuf
        self.normalBuffer = nBuf
        self.colorBuffer = cBuf
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
            positionBuffer: pBuf, normalBuffer: nBuf,
            positionStride: vecBytes, normalStride: vecBytes,
            vertexCount: verts,
            bodyIndexBuffer: iBuf, bodyIndexCount: indices.count, bodyIndexType: .uint32,
            colorBuffer: cBuf,
            doubleSided: true,
            shadowCastsBothFaces: true)
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

    private func encodeGenerate(camera: SIMD3<Float>, time: Float, windScale: Float = 1) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = "GrassTiles.generate"
        enc.label = "GrassTiles.generate"
        let p = params
        uniformBuffer.contents().bindMemory(to: GrassTileUniforms.self, capacity: 1).pointee =
            GrassTileUniforms(
                camera: SIMD4(camera, time),
                shape: SIMD4(p.groundY, p.windAmp * windScale, p.windHeading, p.widenPerRing),
                sizes: SIMD4(p.baseCellSize, p.baseHalfWidth, p.heightMin, p.heightRange),
                maskRect: maskRect,
                rootC: SIMD4(p.rootColor, 0), midC: SIMD4(p.midColor, 0), tipC: SIMD4(p.tipColor, 0),
                misc: SIMD4(p.foliageSSS ? 0 : 1, p.farTint, 0, 0),
                ringInfo: GrassTileUniforms.RingInfo(p.rings),
                ringCount: UInt32(p.rings.count), segments: UInt32(p.segments), pad0: 0, pad1: 0)
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(positionBuffer, offset: 0, index: 0)
        enc.setBuffer(normalBuffer, offset: 0, index: 1)
        enc.setBuffer(colorBuffer, offset: 0, index: 2)
        enc.setBuffer(uniformBuffer, offset: 0, index: 3)
        enc.setTexture(maskTexture, index: 0)
        enc.setTexture(groundTexture, index: 1)
        let w = min(bladeCount, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: bladeCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
    }

    /// TEST-OBSERVABLE: world position of every KEPT blade's root. Drains the sim queue first, so the
    /// read sees this frame's generation. A culled slot writes coincident verts (zero width); a kept
    /// blade's root row is `2·halfWidth` wide.
    public func readBladeRoots() -> [SIMD3<Float>] {
        if let cb = engine.commandQueue.makeCommandBuffer() { cb.commit(); cb.waitUntilCompleted() }
        let rows = params.segments + 1
        let pos = positionBuffer.contents().bindMemory(to: Float.self, capacity: vertexCount * 3)
        var roots: [SIMD3<Float>] = []
        for b in 0..<bladeCount {
            let v = b * rows * 2 * 3
            let l = SIMD3<Float>(pos[v], pos[v + 1], pos[v + 2])
            let r = SIMD3<Float>(pos[v + 3], pos[v + 4], pos[v + 5])
            if simd_length(r - l) > 1e-6 { roots.append((l + r) * 0.5) }
        }
        return roots
    }

    /// TEST-OBSERVABLE: tip positions (last row midpoint) of every KEPT blade, in slot order, after
    /// regenerating at `time` with the given wind scale — lets a gate compare two times/scales.
    public func bladeTips(cameraAt camera: SIMD3<Float>, time: Float, windScale: Float) -> [SIMD3<Float>] {
        encodeGenerate(camera: camera, time: time, windScale: windScale)
        if let cb = engine.commandQueue.makeCommandBuffer() { cb.commit(); cb.waitUntilCompleted() }
        let rows = params.segments + 1
        let pos = positionBuffer.contents().bindMemory(to: Float.self, capacity: vertexCount * 3)
        var tips: [SIMD3<Float>] = []
        for b in 0..<bladeCount {
            let v0 = b * rows * 2 * 3
            let w = SIMD3<Float>(pos[v0], pos[v0 + 1], pos[v0 + 2]) - SIMD3<Float>(pos[v0 + 3], pos[v0 + 4], pos[v0 + 5])
            if simd_length(w) <= 1e-6 { continue }
            let v = (b * rows * 2 + (rows - 1) * 2) * 3
            tips.append((SIMD3<Float>(pos[v], pos[v + 1], pos[v + 2]) + SIMD3<Float>(pos[v + 3], pos[v + 4], pos[v + 5])) * 0.5)
        }
        return tips
    }

    /// TEST-OBSERVABLE: the roots this field would grow with the camera at `camera` (regenerates, then
    /// reads). The field is view-dependent by design — it only exists around the viewer — so a test
    /// asks "what grows HERE" rather than "what grows in the yard".
    public func bladeRoots(cameraAt camera: SIMD3<Float>) -> [SIMD3<Float>] {
        encodeGenerate(camera: camera, time: 0)
        return readBladeRoots()
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
    var pad0: UInt32
    var pad1: UInt32

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
