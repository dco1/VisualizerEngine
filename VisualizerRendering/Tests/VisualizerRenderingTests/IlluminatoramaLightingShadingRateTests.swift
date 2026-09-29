import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// The lighting pass's DEFOCUS-AWARE SHADING RATE (`lightingDefocusShadingMinCoC`, VZ-0197), run on
/// the REAL `illumi_lighting` kernel compiled from source:
///
/// • the COARSE variant (a thread per 2×2 quad) and the SKIP variant (a thread per pixel) write every
///   pixel of the frame exactly once between them — a pre-filled NaN sentinel survives nowhere;
/// • a quad is lit once only where the rule allows it — its 16×16 tile defocused throughout (the
///   tile map ≥ the gate), all four pixels geometry, their view depths within 2 %, their normals
///   within ~8° of the leader's, the quad inside the frame — and there all four pixels carry the top-left pixel's radiance from the ordinary
///   kernel; everywhere else (in-focus tiles, sky in the quad, a silhouette, a ragged frame edge)
///   every pixel equals the ordinary kernel's own result.
/// And the tile map itself: `illumi_dof_shading_tiles` is the least |CoC| of each tile's geometry
/// (sky ignored) under the DOF's own `coc_radius`.
@MainActor
final class IlluminatoramaLightingShadingRateTests: XCTestCase {

    private let W = 72, H = 50          // not multiples of 16 or even-by-tile: ragged edge tiles + quads
    private let tile = 16
    private let gate: Float = 8

    private func shaderURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/\(name)")
    }

    /// A real perspective (the unprojection the coarse test uses must be exercised) and its device
    /// depth for a view distance `z` down the axis.
    private let cam = IlluminatoramaCamera(position: .zero, target: SIMD3(0, 0, -1), up: SIMD3(0, 1, 0),
                                           fovYRadians: 0.6, aspect: 72.0 / 50.0, zNear: 0.05, zFar: 500)
    private func deviceDepth(viewZ z: Float) -> Float {
        let c = cam.projectionMatrix * SIMD4<Float>(0, 0, -z, 1)
        return c.z / c.w
    }

    // ── The scene: view depth per pixel (0 = sky) and the tile map ────────────────────────────────

    /// View distance of pixel (x, y), or 0 for sky. A far wall (3 m, a gentle slope) with: sky in the
    /// top-right tile; a silhouette (a post at 4.5 m) crossing the defocused tiles; an in-focus block.
    private func viewZ(_ x: Int, _ y: Int) -> Float {
        if (x >= 56 || (x >= 37 && x < 46)) && y < 12 { return 0 }         // sky (one patch in a defocused tile)
        if x == 21 || x == 22 { return 4.5 }                               // a post: silhouettes at 20|21, 22|23
        if x == 9 && y >= 20 && y < 24 { return 3.4 }                      // a 1-px ledge (quad straddle)
        if x >= 40 && x < 56 && y >= 16 && y < 32 { return 0.62 }          // in focus
        return 3.0 + 0.002 * Float(x) + 0.001 * Float(y)                   // the wall (well within 2 %)
    }
    /// The tile map the lighting reads (hand-authored here; the DOF kernel is tested separately):
    /// tiles in columns 0…2 are defocused (20 px) except (2, 1), which holds the in-focus block.
    private func tileValue(_ tx: Int, _ ty: Int) -> Float {
        if tx == 2 && ty == 1 { return 0.5 }
        if tx >= 3 { return 2 }                                            // column 3, 4: sharp-ish
        return 20
    }

    /// The G-buffer's oct-encoded normal: one orientation everywhere except a CREASE (column 30 and
    /// a 3-px patch at (12…14, 40…42)) tilted ~25°, so quads straddling it must be lit per pixel.
    private func normalEnc(_ x: Int, _ y: Int) -> SIMD2<Float> {
        let crease = x == 30 || (x >= 12 && x <= 14 && y >= 40 && y <= 42)
        return crease ? SIMD2(0.72, 0.62) : SIMD2(0.5, 0.62)
    }
    /// CPU mirror of `octDecode` on the half-precision value the kernel reads.
    private func normalAt(_ x: Int, _ y: Int) -> SIMD3<Float> {
        let e0 = normalEnc(x, y)
        var e = SIMD2(Float(Float16(e0.x)), Float(Float16(e0.y))) * 2 - 1
        var n = SIMD3(e.x, e.y, 1 - abs(e.x) - abs(e.y))
        if n.z < 0 {
            let s = SIMD2<Float>(n.x >= 0 ? 1 : -1, n.y >= 0 ? 1 : -1)
            e = (SIMD2<Float>(1, 1) - SIMD2(abs(n.y), abs(n.x))) * s
            n = SIMD3(e.x, e.y, n.z)
        }
        return simd_normalize(n)
    }

    private func isCoarseQuad(_ qx: Int, _ qy: Int) -> Bool {
        guard qx + 1 < W, qy + 1 < H else { return false }
        guard tileValue(qx / tile, qy / tile) >= gate else { return false }
        let zs = [viewZ(qx, qy), viewZ(qx + 1, qy), viewZ(qx, qy + 1), viewZ(qx + 1, qy + 1)]
        guard zs.allSatisfy({ $0 > 0 }) else { return false }
        guard zs.max()! - zs.min()! <= 0.02 * zs.min()! else { return false }
        let n0 = normalAt(qx, qy)
        return [(qx + 1, qy), (qx, qy + 1), (qx + 1, qy + 1)].allSatisfy { simd_dot(n0, normalAt($0.0, $0.1)) >= 0.99 }
    }

    // ── Running the kernel ────────────────────────────────────────────────────────────────────────

    private struct Rig {
        let device: MTLDevice, queue: MTLCommandQueue, lib: MTLLibrary
    }

    private func rig() throws -> Rig {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = shaderURL("IlluminatoramaLighting.metal")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no shader source") }
        return Rig(device: device, queue: queue, lib: try MetalSourceLoader.makeLibrary(device: device, contentsOf: url))
    }

    /// `illumi_lighting` with every feature constant off (sun without shadows, ambient, emission) and
    /// the shading-rate constants as given.
    private func pipeline(_ r: Rig, coarse: Bool?, skip: Bool?) throws -> MTLComputePipelineState {
        let cv = MTLFunctionConstantValues()
        var off = false
        for i in 0..<6 { cv.setConstantValue(&off, type: .bool, index: i) }
        if var c = coarse { cv.setConstantValue(&c, type: .bool, index: 6) }
        if var k = skip { cv.setConstantValue(&k, type: .bool, index: 7) }
        let fn = try r.lib.makeFunction(name: "illumi_lighting", constantValues: cv)
        var refl: MTLComputePipelineReflection?
        let p = try r.device.makeComputePipelineState(function: fn, options: [.bindingInfo], reflection: &refl)  // gpu-ok: test harness
        usedBindings[ObjectIdentifier(p)] = Set((refl?.bindings ?? []).filter(\.isUsed).map { "\($0.type == .buffer ? "b" : "t")\($0.index)" })
        return p
    }
    /// The argument slots each pipeline actually reads (API validation rejects binding any other).
    private var usedBindings: [ObjectIdentifier: Set<String>] = [:]

    private func texture(_ r: Rig, _ fmt: MTLPixelFormat, _ w: Int, _ h: Int, type: MTLTextureType = .type2D,
                         mips: Int = 1, slices: Int = 1, usage: MTLTextureUsage = [.shaderRead, .shaderWrite],
                         shared: Bool = true) throws -> MTLTexture {
        let d = MTLTextureDescriptor()
        d.textureType = type
        d.pixelFormat = fmt
        d.width = w; d.height = h
        d.mipmapLevelCount = mips
        if type == .type2DArray { d.arrayLength = slices }
        d.usage = usage
        d.storageMode = shared ? .shared : .private
        return try XCTUnwrap(r.device.makeTexture(descriptor: d))
    }

    private func fill16(_ t: MTLTexture, _ f: (Int, Int) -> SIMD4<Float>) {
        var px = [Float16](repeating: 0, count: t.width * t.height * 4)
        for y in 0..<t.height { for x in 0..<t.width {
            let v = f(x, y), i = (y * t.width + x) * 4
            px[i] = Float16(v.x); px[i + 1] = Float16(v.y); px[i + 2] = Float16(v.z); px[i + 3] = Float16(v.w)
        } }
        px.withUnsafeBytes { t.replace(region: MTLRegionMake2D(0, 0, t.width, t.height), mipmapLevel: 0,
                                       withBytes: $0.baseAddress!, bytesPerRow: t.width * 8) }
    }

    /// Lights the synthetic G-buffer. `variant` nil = the ordinary kernel over every pixel; otherwise
    /// the coarse + skip pair. Returns rgba (Float) per pixel.
    private func light(_ r: Rig, shadingRate: Bool) throws -> [SIMD4<Float>] {
        let dev = r.device
        // G-buffer: albedo varies per pixel (so replication is visible), a fixed up-and-out normal.
        let albedo = try texture(r, .rgba16Float, W, H)
        fill16(albedo) { x, y in SIMD4(0.2 + 0.6 * Float(x % 7) / 7, 0.3 + 0.5 * Float(y % 5) / 5, 0.25, 0) }
        let normal = try texture(r, .rgba16Float, W, H)
        fill16(normal) { x, y in let e = self.normalEnc(x, y); return SIMD4<Float>(e.x, e.y, 0.5, 1) }  // oct-encoded, roughness 0.5, opaque
        let emission = try texture(r, .rgba16Float, W, H)
        fill16(emission) { x, y in (x + y) % 11 == 0 ? SIMD4(0.8, 0.1, 0.05, 0) : .zero }
        let material = try texture(r, .rgba16Float, W, H)
        fill16(material) { _, _ in .zero }
        // Depth: via a shared buffer → blit (depth textures are private).
        let depth = try texture(r, .depth32Float, W, H, usage: [.shaderRead, .renderTarget], shared: false)
        var dz = [Float](repeating: 1, count: W * H)
        for y in 0..<H { for x in 0..<W {
            let z = viewZ(x, y)
            dz[y * W + x] = z > 0 ? deviceDepth(viewZ: z) : 1.0
        } }
        let depthBuf = try XCTUnwrap(dev.makeBuffer(bytes: dz, length: dz.count * 4, options: .storageModeShared))
        let out = try texture(r, .rgba16Float, W, H)
        fill16(out) { _, _ in SIMD4(repeating: .nan) }                     // the sentinel
        let ao = try texture(r, .rgba16Float, (W + 1) / 2, (H + 1) / 2)
        fill16(ao) { _, _ in SIMD4(1, 1, 1, 1) }
        let sky = try texture(r, .rgba16Float, 8, 4)
        fill16(sky) { _, _ in SIMD4(0.4, 0.5, 0.7, 1) }
        let cube = try texture(r, .rgba16Float, 4, 4, type: .typeCube, usage: [.shaderRead])
        let cubeMips = try texture(r, .rgba16Float, 4, 4, type: .typeCube, mips: 3, usage: [.shaderRead])
        let shadow = try texture(r, .depth32Float, 4, 4, type: .type2DArray, slices: 3,
                                 usage: [.shaderRead, .renderTarget], shared: false)
        let dfg = try texture(r, .rg16Float, 4, 4, usage: [.shaderRead])
        let small = try texture(r, .rgba16Float, 1, 1)
        let specOut = try texture(r, .rgba16Float, 1, 1), diffOut = try texture(r, .rgba16Float, 1, 1)
        let irrPrev = try texture(r, .rgba16Float, W, H), irrCur = try texture(r, .rgba16Float, W, H)
        let ltc = try texture(r, .rgba32Float, 4, 4, usage: [.shaderRead])
        let layer = try texture(r, .r32Uint, W, H)
        var ones = [UInt32](repeating: 0xFFFF_FFFF, count: W * H)
        ones.withUnsafeMutableBytes { layer.replace(region: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0,
                                                    withBytes: $0.baseAddress!, bytesPerRow: W * 4) }
        let sss = try texture(r, .rgba16Float, W, H)
        let tw = (W + tile - 1) / tile, th = (H + tile - 1) / tile
        let tiles = try texture(r, .r32Float, tw, th)
        var tv = [Float](repeating: 0, count: tw * th)
        for ty in 0..<th { for tx in 0..<tw { tv[ty * tw + tx] = tileValue(tx, ty) } }
        tv.withUnsafeMutableBytes { tiles.replace(region: MTLRegionMake2D(0, 0, tw, th), mipmapLevel: 0,
                                                  withBytes: $0.baseAddress!, bytesPerRow: tw * 4) }

        // Frame uniforms: zeroed, then the fields this lane reads.
        let fuLen = MemoryLayout<IlluminatoramaFrameUniforms>.stride
        let fuBuf = try XCTUnwrap(dev.makeBuffer(length: fuLen, options: .storageModeShared))
        memset(fuBuf.contents(), 0, fuLen)
        let fu = fuBuf.contents().bindMemory(to: IlluminatoramaFrameUniforms.self, capacity: 1)
        let proj = cam.projectionMatrix, view = cam.viewMatrix
        fu.pointee.projection = proj
        fu.pointee.invProjection = proj.inverse
        fu.pointee.view = view
        fu.pointee.invView = view.inverse
        fu.pointee.viewProjection = proj * view
        fu.pointee.invViewProjection = (proj * view).inverse
        fu.pointee.cameraWorldPos = cam.position
        fu.pointee.directionalLightDir = simd_normalize(SIMD3(0.3, 0.8, 0.5))
        fu.pointee.directionalLightColor = SIMD3(2, 1.8, 1.5)
        fu.pointee.ambientColor = SIMD3(0.2, 0.25, 0.3)
        fu.pointee.exposure = 1
        fu.pointee.iblIntensity = 1
        fu.pointee.lightingCoarseMinCoC = shadingRate ? (gateOverride ?? gate) : 0
        let zeros = try XCTUnwrap(dev.makeBuffer(length: 1 << 16, options: .storageModeShared))
        memset(zeros.contents(), 0, 1 << 16)

        let cb = try XCTUnwrap(r.queue.makeCommandBuffer())
        let blit = try XCTUnwrap(cb.makeBlitCommandEncoder())
        blit.copy(from: depthBuf, sourceOffset: 0, sourceBytesPerRow: W * 4, sourceBytesPerImage: W * H * 4,
                  sourceSize: MTLSize(width: W, height: H, depth: 1), to: depth, destinationSlice: 0,
                  destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        let textures: [MTLTexture] = [albedo, normal, emission, depth, out, ao, sky, cube, cubeMips, shadow, dfg,
                                      small, small, shadow, irrPrev, irrCur, ltc, ltc, layer, shadow, sss, material,
                                      specOut, diffOut, tiles]
        func run(_ p: MTLComputePipelineState, _ w: Int, _ h: Int) {
            // One encoder per variant, binding exactly the slots it reads (API validation rejects
            // unused and redundant bindings).
            guard let e = cb.makeComputeCommandEncoder() else { XCTFail("no encoder"); return }
            let used = usedBindings[ObjectIdentifier(p)] ?? []
            for (i, t) in textures.enumerated() where used.contains("t\(i)") { e.setTexture(t, index: i) }
            for i in 0...7 where used.contains("b\(i)") { e.setBuffer(i == 0 ? fuBuf : zeros, offset: 0, index: i) }
            e.setComputePipelineState(p)
            e.dispatchThreads(MTLSize(width: w, height: h, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: p.threadExecutionWidth,
                                                             height: max(1, min(8, p.maxTotalThreadsPerThreadgroup / p.threadExecutionWidth)),
                                                             depth: 1))
            e.endEncoding()
        }
        if shadingRate {
            run(try pipeline(r, coarse: true, skip: false), (W + 1) / 2, (H + 1) / 2)
            run(try pipeline(r, coarse: false, skip: true), W, H)
        } else {
            run(try pipeline(r, coarse: nil, skip: nil), W, H)
        }
        cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        XCTAssertNil(cb.error, "lighting failed: \(String(describing: cb.error))")
        var px = [Float16](repeating: 0, count: W * H * 4)
        px.withUnsafeMutableBytes { out.getBytes($0.baseAddress!, bytesPerRow: W * 8,
                                                 from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0) }
        return (0..<(W * H)).map { SIMD4(Float(px[$0 * 4]), Float(px[$0 * 4 + 1]), Float(px[$0 * 4 + 2]), Float(px[$0 * 4 + 3])) }
    }

    private func close(_ a: SIMD4<Float>, _ b: SIMD4<Float>) -> Bool {
        simd_reduce_max(simd_abs(a - b)) <= 2e-3 * max(1, simd_reduce_max(simd_abs(b)))
    }

    // ── Tests ─────────────────────────────────────────────────────────────────────────────────────

    func testCoarseAndSkipPartitionTheFrameAndReplicateOnlyWhereAllowed() throws {
        let r = try rig()
        let ref = try light(r, shadingRate: false)
        let img = try light(r, shadingRate: true)
        var coarseQuads = 0, perPixelQuads = 0, silhouetteInCoarseTile = 0, skyInCoarseTile = 0, creaseInCoarseTile = 0
        for qy in stride(from: 0, to: H, by: 2) {
            for qx in stride(from: 0, to: W, by: 2) {
                let pts = [(qx, qy), (qx + 1, qy), (qx, qy + 1), (qx + 1, qy + 1)].filter { $0.0 < W && $0.1 < H }
                for (x, y) in pts {
                    let v = img[y * W + x]
                    XCTAssertFalse(v.x.isNaN || v.y.isNaN || v.z.isNaN, "pixel (\(x),\(y)) was never written")
                }
                if isCoarseQuad(qx, qy) {
                    coarseQuads += 1
                    let lead = ref[qy * W + qx]
                    for (x, y) in pts {
                        XCTAssertTrue(close(img[y * W + x], lead),
                                      "coarse quad (\(qx),\(qy)): (\(x),\(y)) = \(img[y * W + x]) ≠ leader \(lead)")
                    }
                } else {
                    perPixelQuads += 1
                    if tileValue(qx / tile, qy / tile) >= gate {
                        let zs = pts.map { viewZ($0.0, $0.1) }
                        let flat = zs.allSatisfy { $0 > 0 } && zs.max()! - zs.min()! <= 0.02 * zs.min()!
                        if zs.contains(0) { skyInCoarseTile += 1 }
                        else if flat && pts.count == 4 { creaseInCoarseTile += 1 }
                        else { silhouetteInCoarseTile += 1 }
                    }
                    for (x, y) in pts {
                        XCTAssertTrue(close(img[y * W + x], ref[y * W + x]),
                                      "per-pixel quad (\(qx),\(qy)): (\(x),\(y)) = \(img[y * W + x]) ≠ \(ref[y * W + x])")
                    }
                }
            }
        }
        // The probe must exercise every branch of the rule.
        XCTAssertGreaterThan(coarseQuads, 100)
        XCTAssertGreaterThan(perPixelQuads, 100)
        XCTAssertGreaterThan(silhouetteInCoarseTile, 0, "a silhouette / ragged edge inside a defocused tile")
        XCTAssertGreaterThan(skyInCoarseTile, 0, "sky inside a defocused tile")
        XCTAssertGreaterThan(creaseInCoarseTile, 0, "a crease (normals disagree) on one flat surface in a defocused tile")
        // …and the replication must actually change something (albedo varies per pixel).
        let differs = (0..<(W * H)).contains { !close(img[$0], ref[$0]) }
        XCTAssertTrue(differs, "the coarse lane changed nothing — the probe is vacuous")
    }

    /// Gate 0 (the host's knob off, or no DOF): the coarse lane writes nothing and the skip lane lights
    /// every pixel exactly as the ordinary kernel does.
    func testGateOffLightsEveryPixelAsTheOrdinaryKernel() throws {
        let r = try rig()
        let ref = try light(r, shadingRate: false)
        // shadingRate: true with a zero gate: set by running the pair with the uniform at 0.
        let img = try lightPairWithZeroGate(r)
        for i in 0..<(W * H) {
            XCTAssertTrue(close(img[i], ref[i]), "pixel \(i % W),\(i / W): \(img[i]) ≠ \(ref[i])")
        }
    }

    private func lightPairWithZeroGate(_ r: Rig) throws -> [SIMD4<Float>] {
        let saved = gateOverride
        gateOverride = 0
        defer { gateOverride = saved }
        return try light(r, shadingRate: true)
    }
    private var gateOverride: Float? = nil

    // ── The tile map ──────────────────────────────────────────────────────────────────────────────

    /// Mirrors the Metal `DOFParams` (stride 144).
    private struct DOFParams {
        var invProjection: simd_float4x4
        var focusDist: Float, cocCoefficient: Float, maxRadius: Float
        var blades: Float = 0, bladeRotation: Float = 0, catsEye: Float = 0
        var width: UInt32, height: UInt32, tileW: UInt32, tileH: UInt32, tileSize: UInt32
        var prefilterScale: Float = 0, cocFloor: Float, subjectAware: Float = 0
        var fastGather: Float = 0, halfMinCoC: Float = 3, quarterMinCoC: Float = 0
    }

    func testShadingTileMapIsTheLeastCoCOfEachTilesGeometry() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = shaderURL("IlluminatoramaDOF.metal")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no shader") }
        let extra = """

        kernel void test_abs_coc(texture2d<float, access::read> gDepth [[texture(0)]],
                                 device float* out [[buffer(1)]], constant DOFParams& p [[buffer(0)]],
                                 uint2 gid [[thread_position_in_grid]]) {
            if (gid.x >= p.width || gid.y >= p.height) return;
            float d = gDepth.read(gid).r;
            out[gid.y * p.width + gid.x] = d >= 0.99999f ? -1.0f : abs(coc_radius(view_z(d, gid, p), p));
        }
        """
        let lib = try device.makeLibrary(source: MetalSourceLoader.source(contentsOf: url) + extra, options: nil)
        let tilesP = try device.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "illumi_dof_shading_tiles")))  // gpu-ok: test harness
        let cocP = try device.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "test_abs_coc")))  // gpu-ok: test harness
        let tw = (W + tile - 1) / tile, th = (H + tile - 1) / tile
        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: W, height: H, mipmapped: false)
        dd.usage = [.shaderRead]; dd.storageMode = .shared
        let depth = try XCTUnwrap(device.makeTexture(descriptor: dd))
        var dz = [Float](repeating: 1, count: W * H)
        for y in 0..<H { for x in 0..<W { let z = viewZ(x, y); dz[y * W + x] = z > 0 ? deviceDepth(viewZ: z) : 1 } }
        dz.withUnsafeBytes { depth.replace(region: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0,
                                           withBytes: $0.baseAddress!, bytesPerRow: W * 4) }
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: tw, height: th, mipmapped: false)
        td.usage = [.shaderRead, .shaderWrite]; td.storageMode = .shared
        let tiles = try XCTUnwrap(device.makeTexture(descriptor: td))
        let cocBuf = try XCTUnwrap(device.makeBuffer(length: W * H * 4, options: .storageModeShared))
        var p = DOFParams(invProjection: cam.projectionMatrix.inverse, focusDist: 0.62, cocCoefficient: 73.1,
                          maxRadius: 32, width: UInt32(W), height: UInt32(H), tileW: UInt32(tw), tileH: UInt32(th),
                          tileSize: UInt32(tile), cocFloor: 0.15)
        XCTAssertEqual(MemoryLayout<DOFParams>.stride, 144)
        let cb = try XCTUnwrap(queue.makeCommandBuffer())
        let e = try XCTUnwrap(cb.makeComputeCommandEncoder())
        e.setComputePipelineState(tilesP)
        e.setTexture(depth, index: 0); e.setTexture(tiles, index: 1)
        e.setBytes(&p, length: MemoryLayout<DOFParams>.stride, index: 0)
        e.dispatchThreadgroups(MTLSize(width: tw, height: th, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        e.setComputePipelineState(cocP)
        e.setBuffer(cocBuf, offset: 0, index: 1)
        e.dispatchThreads(MTLSize(width: W, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        e.endEncoding()
        cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        XCTAssertNil(cb.error)
        var got = [Float](repeating: 0, count: tw * th)
        got.withUnsafeMutableBytes { tiles.getBytes($0.baseAddress!, bytesPerRow: tw * 4,
                                                    from: MTLRegionMake2D(0, 0, tw, th), mipmapLevel: 0) }
        let coc = cocBuf.contents().bindMemory(to: Float.self, capacity: W * H)
        var sawSkyOnlyMin = false, sawSharp = false
        for ty in 0..<th { for tx in 0..<tw {
            var m = Float.infinity
            for y in (ty * tile)..<min(H, (ty + 1) * tile) { for x in (tx * tile)..<min(W, (tx + 1) * tile) {
                let c = coc[y * W + x]
                if c >= 0 { m = min(m, c) }
            } }
            let want = min(m, 65504)
            XCTAssertEqual(got[ty * tw + tx], want, accuracy: 1e-4 * max(1, want), "tile (\(tx),\(ty))")
            if want < 1 { sawSharp = true }
            if tx == tw - 1 && ty == 0 { sawSkyOnlyMin = want > 1 }       // sky + wall: the wall's CoC, not ∞
        } }
        XCTAssertTrue(sawSharp, "the in-focus block's tile must read sharp")
        XCTAssertTrue(sawSkyOnlyMin, "a tile with sky must report its geometry's CoC")
    }
}
