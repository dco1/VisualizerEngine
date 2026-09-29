import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// The RT glass DEFOCUS TRACE (`rtGlassDefocusTraceFactor`, VZ-0197): the pane is traced once per
/// k × k pixels where the lens will spread it over a disc anyway, and the full-resolution glass
/// pass reconstructs that result. Its two building blocks are tested here on the GPU, compiled from
/// the real `IlluminatoramaGlassRT.metal` (+ the DOF's own `coc_radius` from
/// `IlluminatoramaDOF.metal` for the gate's cross-check):
///
/// • `glassDefocusUpsample` — reconstructs a smooth field exactly (bilinear of a linear ramp), and
///   NEVER mixes in a texel the low-res pass did not cover (the pane's edge, a declined gate),
///   however bright; a pixel with no covered neighbour reports coverage 0 (→ full trace).
/// • `glassDefocusCoC` — the gate — equals the DOF's |CoC| for the pane's own depth when the pane
///   is behind the focus plane, is never larger than the |CoC| the DOF then applies to anything
///   seen THROUGH the pane (deeper still), and is 0 in front of the focus plane (where the DOF
///   follows the opaque depth behind the glass and may keep the pixel sharp).
/// Plus the Swift ↔ Metal mirror of the uniform block the new fields were appended to.
@MainActor
final class IlluminatoramaGlassDefocusTraceTests: XCTestCase {

    private static let testKernels = """

    // ── test kernels (appended by IlluminatoramaGlassDefocusTraceTests) ──
    kernel void test_glass_defocus_upsample(texture2d<float, access::read> lo [[texture(0)]],
                                            device float4* out [[buffer(0)]],
                                            constant float4& p [[buffer(1)]],   // x = k, y = full W, z = full H
                                            uint2 gid [[thread_position_in_grid]]) {
        uint w = uint(p.y), h = uint(p.z);
        if (gid.x >= w || gid.y >= h) return;
        out[gid.y * w + gid.x] = glassDefocusUpsample(lo, float2(gid) + 0.5, p.x);
    }
    kernel void test_glass_defocus_coc(constant GlassRTUniforms& u [[buffer(0)]],
                                       const device float4* pts [[buffer(1)]],
                                       device float* out [[buffer(2)]],
                                       uint i [[thread_position_in_grid]]) {
        out[i] = glassDefocusCoC(pts[i].xyz, u);
    }
    kernel void test_glass_uniform_size(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
        if (i == 0) out[0] = uint(sizeof(GlassRTUniforms));
    }
    """

    private static let dofTestKernel = """

    kernel void test_dof_coc(constant DOFParams& p [[buffer(0)]],
                             const device float* z [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
        out[i] = coc_radius(z[i], p);
    }
    """

    /// Mirrors the Metal `DOFParams` (stride 144) — as `IlluminatoramaDOFSubjectTests` does.
    private struct DOFParams {
        var invProjection: simd_float4x4 = matrix_identity_float4x4
        var focusDist: Float, cocCoefficient: Float, maxRadius: Float
        var blades: Float = 0, bladeRotation: Float = 0, catsEye: Float = 0
        var width: UInt32 = 1, height: UInt32 = 1, tileW: UInt32 = 1, tileH: UInt32 = 1, tileSize: UInt32 = 16
        var prefilterScale: Float = 0, cocFloor: Float, subjectAware: Float = 0
        var fastGather: Float = 0, halfMinCoC: Float = 3, quarterMinCoC: Float = 0
    }

    private func shaderURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/\(name)")
    }

    private func library(_ device: MTLDevice, file: String, appending extra: String) throws -> MTLLibrary {
        let url = shaderURL(file)
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no shader source") }
        let src = try MetalSourceLoader.source(contentsOf: url) + extra
        return try device.makeLibrary(source: src, options: nil)
    }

    private func setUpGlass() throws -> (MTLDevice, MTLCommandQueue, MTLLibrary) {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard device.supportsRaytracing else { throw XCTSkip("the glass RT shader needs ray tracing") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let lib = try library(device, file: "IlluminatoramaGlassRT.metal", appending: Self.testKernels)
        return (device, queue, lib)
    }

    private func pipe(_ device: MTLDevice, _ lib: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
        let fn = try XCTUnwrap(lib.makeFunction(name: name), "missing kernel \(name)")
        return try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
    }

    // ── Layout ────────────────────────────────────────────────────────────────

    func testUniformBlockMirrorsTheShader() throws {
        let (device, queue, lib) = try setUpGlass()
        let p = try pipe(device, lib, "test_glass_uniform_size")
        let out = try XCTUnwrap(device.makeBuffer(length: 16, options: .storageModeShared))
        let cb = try XCTUnwrap(queue.makeCommandBuffer())
        let e = try XCTUnwrap(cb.makeComputeCommandEncoder())
        e.setComputePipelineState(p)
        e.setBuffer(out, offset: 0, index: 0)
        e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        e.endEncoding()
        cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        let metalSize = Int(out.contents().load(as: UInt32.self))
        XCTAssertEqual(MemoryLayout<IlluminatoramaGlassRTUniforms>.stride, metalSize,
                       "Swift IlluminatoramaGlassRTUniforms no longer mirrors Metal GlassRTUniforms")
        // The low-res pass hands this block over by value (setFragmentBytes): it must fit.
        XCTAssertLessThanOrEqual(metalSize, 4096)
    }

    // ── Reconstruction ────────────────────────────────────────────────────────

    /// Runs `glassDefocusUpsample` over a full-res grid for a low-res image.
    private func upsample(k: Int, lowW: Int, lowH: Int, fullW: Int, fullH: Int,
                          texel: (Int, Int) -> SIMD4<Float>) throws -> [SIMD4<Float>] {
        let (device, queue, lib) = try setUpGlass()
        let p = try pipe(device, lib, "test_glass_defocus_upsample")
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: lowW, height: lowH, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .shared
        let lo = try XCTUnwrap(device.makeTexture(descriptor: d))
        var px = [Float](repeating: 0, count: lowW * lowH * 4)
        for y in 0..<lowH { for x in 0..<lowW {
            let t = texel(x, y); let i = (y * lowW + x) * 4
            px[i] = t.x; px[i + 1] = t.y; px[i + 2] = t.z; px[i + 3] = t.w
        } }
        px.withUnsafeBytes { lo.replace(region: MTLRegionMake2D(0, 0, lowW, lowH), mipmapLevel: 0,
                                        withBytes: $0.baseAddress!, bytesPerRow: lowW * 16) }
        let out = try XCTUnwrap(device.makeBuffer(length: fullW * fullH * 16, options: .storageModeShared))
        var params = SIMD4<Float>(Float(k), Float(fullW), Float(fullH), 0)
        let cb = try XCTUnwrap(queue.makeCommandBuffer())
        let e = try XCTUnwrap(cb.makeComputeCommandEncoder())
        e.setComputePipelineState(p)
        e.setTexture(lo, index: 0)
        e.setBuffer(out, offset: 0, index: 0)
        e.setBytes(&params, length: 16, index: 1)
        e.dispatchThreads(MTLSize(width: fullW, height: fullH, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        e.endEncoding()
        cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        XCTAssertNil(cb.error)
        let f = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: fullW * fullH)
        return (0..<(fullW * fullH)).map { f[$0] }
    }

    /// A linear ramp is reproduced exactly wherever the four neighbours are covered: the
    /// reconstruction adds no error of its own to a smooth pane (and the texel centres sit where
    /// the low-res pass rasterised them — a half-texel slip would show as a constant offset).
    func testFullyCoveredLinearRampIsReconstructedExactly() throws {
        for k in [2, 3, 4] {
            let lowW = 12, lowH = 9, fullW = lowW * k, fullH = lowH * k
            func ramp(_ u: Float, _ v: Float) -> SIMD3<Float> { SIMD3(0.5 + 0.25 * u, 2.0 - 0.1 * v, 0.03 * u + 0.07 * v) }
            let img = try upsample(k: k, lowW: lowW, lowH: lowH, fullW: fullW, fullH: fullH) { x, y in
                SIMD4(ramp(Float(x), Float(y)), 1)
            }
            var worst: Float = 0
            for y in 0..<fullH { for x in 0..<fullW {
                // Low-res continuous coordinate of this full-res pixel centre.
                let u = (Float(x) + 0.5) / Float(k) - 0.5, v = (Float(y) + 0.5) / Float(k) - 0.5
                guard u >= 0, v >= 0, u <= Float(lowW - 1), v <= Float(lowH - 1) else { continue }
                let got = img[y * fullW + x]
                XCTAssertEqual(got.w, 1, accuracy: 1e-4, "full coverage expected at (\(x),\(y)) k=\(k)")
                worst = max(worst, simd_reduce_max(simd_abs(SIMD3(got.x, got.y, got.z) - ramp(u, v))))
            } }
            XCTAssertLessThan(worst, 1e-4, "k=\(k): bilinear reconstruction of a linear ramp is off by \(worst)")
        }
    }

    /// The pane's edge: texels the low-res pass did not cover carry a HUGE colour here, and not a
    /// trace of it may reach any reconstructed pixel; a pixel whose four neighbours are all
    /// uncovered reports coverage 0 so the full-res pass traces it itself.
    func testUncoveredTexelsNeverBleedIn() throws {
        let k = 4, lowW = 10, lowH = 6, fullW = lowW * k, fullH = lowH * k
        let pane = SIMD3<Float>(0.2, 0.3, 0.4)
        let img = try upsample(k: k, lowW: lowW, lowH: lowH, fullW: fullW, fullH: fullH) { x, _ in
            x < 5 ? SIMD4(pane, 1) : SIMD4(1000, 1000, 1000, 0)
        }
        var covered = 0, uncovered = 0
        for y in 0..<fullH { for x in 0..<fullW {
            let got = img[y * fullW + x]
            if got.w > 0 {
                covered += 1
                XCTAssertLessThan(simd_reduce_max(simd_abs(SIMD3(got.x, got.y, got.z) - pane)), 1e-4,
                                  "uncovered texel bled into (\(x),\(y)): \(got)")
            } else {
                uncovered += 1
                XCTAssertEqual(SIMD3(got.x, got.y, got.z), .zero)
                // Only pixels whose nearest low-res column is past the pane edge may be uncovered.
                let u = (Float(x) + 0.5) / Float(k) - 0.5
                XCTAssertGreaterThanOrEqual(u, 4.0, "a pixel beside covered texels reported no coverage at x=\(x)")
            }
        } }
        XCTAssertGreaterThan(covered, 0)
        XCTAssertGreaterThan(uncovered, 0, "the probe must reach past the pane edge")
    }

    // ── The gate ──────────────────────────────────────────────────────────────

    func testGateIsTheDOFsOwnCoCBehindFocusAndZeroInFront() throws {
        let (device, queue, lib) = try setUpGlass()
        let glassP = try pipe(device, lib, "test_glass_defocus_coc")
        let dofLib = try library(device, file: "IlluminatoramaDOF.metal", appending: Self.dofTestKernel)
        let dofP = try pipe(device, dofLib, "test_dof_coc")

        // A camera at the origin looking down −Z (DOF depth = distance along the view axis),
        // focused at 0.62 m with the Digital Clock's hero lens on a 1620-tall frame.
        let focus: Float = 0.62, coef: Float = 73.1, maxR: Float = 32, floorPx: Float = 0.15
        var u = IlluminatoramaGlassRTUniforms()
        u.cameraWorldPos = .zero
        u.cameraForward = SIMD4(0, 0, -1, 0)
        u.defocusLens = SIMD4(focus, coef, maxR, floorPx)
        var dof = DOFParams(focusDist: focus, cocCoefficient: coef, maxRadius: maxR, cocFloor: floorPx)

        // Pane points at depths across the focus plane, some off-axis (the gate measures depth
        // along the axis, as the DOF does — never the Euclidean range).
        let depths: [Float] = [0.2, 0.4, 0.6, 0.62, 0.63, 0.7, 0.9, 1.1, 1.6, 3, 40]
        var pts: [SIMD4<Float>] = []
        for (i, z) in depths.enumerated() {
            let off = Float(i % 3) * 0.1 * z
            pts.append(SIMD4(off, -off * 0.5, -z, 0))
        }
        // The DOF evaluated at each pane depth, and at a surface 0.3 m BEHIND it (seen through it).
        let dofZ = depths + depths.map { $0 + 0.3 }

        let ptsBuf = try XCTUnwrap(device.makeBuffer(bytes: pts, length: pts.count * 16, options: .storageModeShared))
        let gateOut = try XCTUnwrap(device.makeBuffer(length: pts.count * 4, options: .storageModeShared))
        let zBuf = try XCTUnwrap(device.makeBuffer(bytes: dofZ, length: dofZ.count * 4, options: .storageModeShared))
        let dofOut = try XCTUnwrap(device.makeBuffer(length: dofZ.count * 4, options: .storageModeShared))
        let cb = try XCTUnwrap(queue.makeCommandBuffer())
        do {
            let e = try XCTUnwrap(cb.makeComputeCommandEncoder())
            e.setComputePipelineState(glassP)
            e.setBytes(&u, length: MemoryLayout<IlluminatoramaGlassRTUniforms>.stride, index: 0)
            e.setBuffer(ptsBuf, offset: 0, index: 1)
            e.setBuffer(gateOut, offset: 0, index: 2)
            e.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
            e.endEncoding()
        }
        do {
            let e = try XCTUnwrap(cb.makeComputeCommandEncoder())
            e.setComputePipelineState(dofP)
            e.setBytes(&dof, length: MemoryLayout<DOFParams>.stride, index: 0)
            e.setBuffer(zBuf, offset: 0, index: 1)
            e.setBuffer(dofOut, offset: 0, index: 2)
            e.dispatchThreads(MTLSize(width: dofZ.count, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            e.endEncoding()
        }
        cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        XCTAssertNil(cb.error)
        XCTAssertEqual(MemoryLayout<DOFParams>.stride, 144)
        let gate = gateOut.contents().bindMemory(to: Float.self, capacity: pts.count)
        let coc = dofOut.contents().bindMemory(to: Float.self, capacity: dofZ.count)
        for (i, z) in depths.enumerated() {
            let g = gate[i]
            let dofHere = coc[i], dofBehind = coc[depths.count + i]
            if z <= focus {
                XCTAssertEqual(g, 0, "z=\(z): a pane in front of (or on) the focus plane must never take the reduced trace")
            } else {
                // Behind focus the DOF's CoC is negative (far field); the gate is its magnitude.
                XCTAssertLessThan(dofHere, 0, "z=\(z): the DOF should call this far field")
                XCTAssertEqual(g, abs(dofHere), accuracy: 1e-3 * max(1, abs(dofHere)), "z=\(z): gate ≠ the DOF's |CoC|")
                XCTAssertLessThanOrEqual(g, abs(dofBehind) + 1e-4,
                                         "z=\(z): what the DOF blurs THROUGH the pane is blurred less than the gate assumed")
            }
        }
    }
}
