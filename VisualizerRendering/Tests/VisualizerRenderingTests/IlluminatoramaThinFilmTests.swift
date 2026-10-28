import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// **Daydream DH-1014 — thin-film iridescence on opaque surfaces, on the shipped shader code.**
///
/// The film is a view-angle TINT on the base's F0 (`illumiThinFilmTint`, IlluminatoramaMaterialLayers.h)
/// that every lobe reads through `layeredBaseF0`. The probe kernels are appended to
/// IlluminatoramaLighting.metal so they call the kernel's own helpers.
///
/// Gates:
///   • weight 0 is EXACTLY float3(1) (the default instance changes nothing);
///   • the tint has unit mean across the channels at weight 1 — the film redistributes reflected
///     energy between channels, it does not invent any;
///   • it SHIFTS WITH VIEW ANGLE and with THICKNESS (otherwise it is not iridescence);
///   • the weight blends linearly;
///   • the white furnace holds per channel with a film on: a white dielectric under any film returns 1
///     on the direct lobe (the base's diffuse weight takes the per-channel complement of the tinted F0).
@MainActor
final class IlluminatoramaThinFilmTests: XCTestCase {

    private static let probeKernel = """

    // ── DH-1014 thin-film probes (appended by IlluminatoramaThinFilmTests) ──
    static inline float dh1014RadInv(uint bits) {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return float(bits) * 2.3283064365386963e-10;
    }
    static inline float3 dh1014SampleGGX(float a2, float3 V, float u1, float u2) {
        float ph = 2.0 * M_PI_F * u2;
        float ct = sqrt((1.0 - u1) / (1.0 + (a2 - 1.0) * u1));
        float st = sqrt(max(0.0, 1.0 - ct * ct));
        float3 H = float3(st * cos(ph), st * sin(ph), ct);
        return 2.0 * dot(V, H) * H - V;
    }
    static inline float dh1014PdfGGX(float a2, float3 V, float3 H) {
        float dd = H.z * H.z * (a2 - 1.0) + 1.0;
        return (a2 / (M_PI_F * dd * dd)) * H.z / (4.0 * max(dot(V, H), 1e-6));
    }
    // in: [cosTheta, thicknessNm, ior, weight] [metallic, 0, 0, 0] ; out: [tint.rgb, 0]
    kernel void dh1014Tint(device const float4* inp [[buffer(0)]], device float4* outp [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
        float4 a = inp[gid * 2], b = inp[gid * 2 + 1];
        MaterialLayers m = illumiDecodeMaterialLayers(float4(0.0), float4(0.0), float4(0.0), float4(0.0),
                                                      float4(a.w, a.y, a.z, 0.0));
        outp[gid] = float4(illumiThinFilmTint(m, a.x, b.x), 0.0);
    }
    // DH-1302 — the translucency record's decode. in: [weight, r, g, b] ; out: [trans, weight, colorSet, color.r + color.g/10 + color.b/100]
    kernel void dh1302Decode(device const float4* inp [[buffer(0)]], device float4* outp [[buffer(1)]],
                             uint gid [[thread_position_in_grid]]) {
        MaterialLayers m = illumiDecodeMaterialLayers(float4(0.0), float4(0.0), float4(0.0), float4(0.0),
                                                      float4(0.0), inp[gid]);
        outp[gid] = float4(m.trans ? 1.0 : 0.0, m.transWeight, m.transColorSet ? 1.0 : 0.0,
                           m.transColor.x + m.transColor.y / 10.0 + m.transColor.z / 100.0);
    }
    // White dielectric (albedo 1, roughness r) under a film: ∫ brdf·cos over the hemisphere, multiscatter on.
    // in: [r, NdotV, thicknessNm, weight] ; out: [direct.rgb, 0]
    kernel void dh1014Furnace(device const float4* inp [[buffer(0)]], device float4* outp [[buffer(1)]],
                              texture2d<half, access::sample> dfg [[texture(0)]],
                              uint gid [[thread_position_in_grid]]) {
        float4 a = inp[gid];
        float r = a.x, mu = a.y;
        MaterialLayers m = illumiDecodeMaterialLayers(float4(0.0), float4(0.0), float4(0.0), float4(0.0),
                                                      float4(a.w, a.z, 1.4, 0.0));
        float3 tint = illumiThinFilmTint(m, mu, 0.0);
        float3 N = float3(0, 0, 1);
        float3 V = float3(sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu);
        constexpr sampler ls(filter::linear, address::clamp_to_edge);
        float4 lut = float4(dfg.sample(ls, float2(mu, r)));
        float3 albedo = float3(1.0);
        LayeredEnergy le = layeredEnergy(albedo, 0.0, lut, 0.0, 0.0, true, false, float3(0.04), false, m.film, tint);
        const uint NS = 4096u;
        float3 direct = float3(0.0);
        float a2 = r * r * r * r;
        for (uint s = 0u; s < 2u * NS; ++s) {
            uint i = s % NS;
            float u1 = (float(i) + 0.5) / float(NS), u2 = dh1014RadInv(i);
            float3 L;
            if (s < NS) { float rr = sqrt(u1), ph = 2.0 * M_PI_F * u2; L = float3(rr * cos(ph), rr * sin(ph), sqrt(max(0.0, 1.0 - u1))); }
            else { L = dh1014SampleGGX(a2, V, u1, u2); }
            if (L.z <= 0.0) continue;
            L = normalize(L);
            float3 H = normalize(V + L);
            float w = 1.0 / (float(NS) * (L.z / M_PI_F + dh1014PdfGGX(a2, V, H)));
            direct += brdf(N, V, L, albedo, 0.0, r, float3(1.0), 0.0, float3(0.0),
                           0.0, nullptr, kClothSheenRoughness, le) * w;
        }
        outp[gid] = float4(direct, 0.0);
    }
    """

    private struct Rig { let device: MTLDevice; let queue: MTLCommandQueue
                         let tint: MTLComputePipelineState; let furnace: MTLComputePipelineState
                         let decode: MTLComputePipelineState; let lut: MTLTexture }
    private static var cachedRig: Rig?

    private static func shaderURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/\(name)")
    }

    private func rig() throws -> Rig {
        if let r = Self.cachedRig { return r }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let bakeLib = try MetalSourceLoader.makeLibrary(device: device, contentsOf: Self.shaderURL("IlluminatoramaIBLBake.metal"))
        guard let bakeFn = bakeLib.makeFunction(name: "illumi_dfg_bake") else { throw XCTSkip("no bake kernel") }
        let bake = try device.makeComputePipelineState(function: bakeFn)  // gpu-ok: test harness
        let n = 128
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: n, height: n, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        guard let lut = device.makeTexture(descriptor: desc),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no LUT") }
        enc.setComputePipelineState(bake)
        enc.setTexture(lut, index: 0)
        var remap: UInt32 = IlluminatoramaRenderer.dfgIBLGeometryRemapDefault ? 1 : 0
        enc.setBytes(&remap, length: 4, index: 0)
        enc.dispatchThreads(MTLSize(width: n, height: n, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        let src = try MetalSourceLoader.source(contentsOf: Self.shaderURL("IlluminatoramaLighting.metal")) + Self.probeKernel
        let lib = try device.makeLibrary(source: src, options: nil)
        guard let tf = lib.makeFunction(name: "dh1014Tint"), let ff = lib.makeFunction(name: "dh1014Furnace"),
              let df = lib.makeFunction(name: "dh1302Decode") else { throw XCTSkip("probe missing") }
        let r = Rig(device: device, queue: queue,
                    tint: try device.makeComputePipelineState(function: tf),      // gpu-ok: test harness
                    furnace: try device.makeComputePipelineState(function: ff),   // gpu-ok: test harness
                    decode: try device.makeComputePipelineState(function: df),    // gpu-ok: test harness
                    lut: lut)
        Self.cachedRig = r
        return r
    }

    /// Runs `pso` over `packed` (float4 records, `perRow` in / 1 out each).
    private func run(_ pso: MTLComputePipelineState, _ packed: [SIMD4<Float>], perRow: Int, withLUT: Bool = false) throws -> [SIMD3<Float>] {
        let rig = try rig()
        let rows = packed.count / perRow
        guard let ib = rig.device.makeBuffer(bytes: packed, length: packed.count * 16, options: .storageModeShared),
              let ob = rig.device.makeBuffer(length: rows * 16, options: .storageModeShared),
              let cb = rig.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no buffers") }
        enc.setComputePipelineState(pso)
        enc.setBuffer(ib, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        if withLUT { enc.setTexture(rig.lut, index: 0) }
        enc.dispatchThreads(MTLSize(width: rows, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(32, rows), height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: rows)
        return (0..<rows).map { SIMD3(o[$0].x, o[$0].y, o[$0].z) }
    }

    private func tints(_ cases: [(cos: Float, d: Float, ior: Float, w: Float, metal: Float)]) throws -> [SIMD3<Float>] {
        var packed: [SIMD4<Float>] = []
        for c in cases { packed.append(SIMD4(c.cos, c.d, c.ior, c.w)); packed.append(SIMD4(c.metal, 0, 0, 0)) }
        return try run(try rig().tint, packed, perRow: 2)
    }

    private func say(_ s: String) { print(s); fflush(stdout) }

    func testWeightZeroIsExactlyNoFilm() throws {
        let t = try tints([(1, 400, 1.4, 0, 0), (0.3, 200, 1.8, 0, 1)])
        for v in t { XCTAssertEqual(v, SIMD3(1, 1, 1), "no film weight ⇒ the tint is exactly 1") }
    }

    func testTintHasUnitMeanAcrossChannels() throws {
        var cases: [(cos: Float, d: Float, ior: Float, w: Float, metal: Float)] = []
        for mu in [Float(1.0), 0.7, 0.4, 0.15] { for d in [Float(120), 250, 400, 650, 900] { for ior in [Float(1.33), 1.45, 2.4] {
            cases.append((mu, d, ior, 1, 0)); cases.append((mu, d, ior, 1, 1))
        } } }
        for (c, v) in zip(cases, try tints(cases)) {
            XCTAssertEqual((v.x + v.y + v.z) / 3, 1, accuracy: 1e-4, "unit mean at \(c)")
            XCTAssertGreaterThanOrEqual(min(v.x, v.y, v.z), 0, "no negative reflectance at \(c)")
        }
    }

    /// Iridescence = colour that moves with the view angle and with the film's thickness.
    func testTintShiftsWithAngleAndThickness() throws {
        let t = try tints([(1.0, 400, 1.4, 1, 0), (0.5, 400, 1.4, 1, 0), (0.15, 400, 1.4, 1, 0),
                           (1.0, 200, 1.4, 1, 0), (1.0, 500, 1.4, 1, 0)])
        say("── DH-1014 thin-film tint (film 1.4, base 1.5): cos 1 / .5 / .15 at 400 nm; 200 / 500 nm at cos 1 ──")
        for (i, v) in t.enumerated() { say(String(format: "  [%d] %.3f %.3f %.3f", i, v.x, v.y, v.z)) }
        func spread(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float { simd_reduce_max(simd_abs(a - b)) }
        XCTAssertGreaterThan(spread(t[0], t[2]), 0.05, "the colour moves with the view angle")
        XCTAssertGreaterThan(spread(t[0], t[1]), 0.02, "…monotonically away from normal incidence")
        XCTAssertGreaterThan(spread(t[3], t[4]), 0.05, "the colour moves with the film's thickness")
    }

    func testWeightBlendsLinearly() throws {
        let t = try tints([(0.6, 350, 1.4, 1, 0), (0.6, 350, 1.4, 0.5, 0)])
        let expect = (SIMD3<Float>(1, 1, 1) + t[0]) * 0.5
        XCTAssertEqual(simd_reduce_max(simd_abs(t[1] - expect)), 0, accuracy: 1e-5)
    }

    /// THE GATE: a white dielectric under any film returns 1 per channel on the direct lobe.
    func testWhiteFurnaceUnderAFilm() throws {
        var rows: [SIMD4<Float>] = []
        let cfg: [(r: Float, d: Float, w: Float)] = [(0.3, 250, 1), (0.3, 500, 1), (0.6, 400, 1), (0.3, 400, 0.5)]
        for c in cfg { for mu in [Float(1.0), 0.7, 0.4, 0.15] { rows.append(SIMD4(c.r, mu, c.d, c.w)) } }
        let out = try run(try rig().furnace, rows, perRow: 1, withLUT: true)
        say("── DH-1014 white furnace — white dielectric under a thin film, direct arm, per channel ──")
        for (i, v) in out.enumerated() {
            let c = cfg[i / 4]
            say(String(format: "  r %.1f d %.0f w %.1f mu %.2f → %.3f %.3f %.3f", c.r, c.d, c.w, rows[i].y, v.x, v.y, v.z))
            for ch in [v.x, v.y, v.z] {
                XCTAssertEqual(ch, 1.0, accuracy: 0.04, "direct furnace under a film \(c) at row \(i)")
            }
        }
    }

    /// DH-1302 — the translucency record decodes: weight 0 ⇒ no layer; weight with no colour ⇒ the albedo
    /// (colour unset); a colour is clamped to [0, 1]; the weight saturates.
    func testTranslucencyRecordDecodes() throws {
        let t = try run(try rig().decode, [SIMD4(0, 0.5, 0.5, 0.5), SIMD4(0.4, 0, 0, 0), SIMD4(0.6, 0.9, 0.5, 0.2), SIMD4(3, 2, 0, 0)], perRow: 1)
        XCTAssertEqual(t[0].x, 0, "weight 0 ⇒ none, whatever the colour")
        XCTAssertEqual(t[1].x, 1); XCTAssertEqual(t[1].y, 0.4, accuracy: 1e-6); XCTAssertEqual(t[1].z, 0, "no colour ⇒ the albedo")
        XCTAssertEqual(t[2].y, 0.6, accuracy: 1e-6); XCTAssertEqual(t[2].z, 1)
        XCTAssertEqual(t[3].y, 1, accuracy: 1e-6, "the weight saturates")
        let i = IlluminatoramaInstance(modelMatrix: matrix_identity_float4x4)
        XCTAssertEqual(i.translucency, .zero)
        var j = i; j.translucencyWeight = 0.5; j.translucencyColor = SIMD3(1, 0.5, 0.25)
        XCTAssertEqual(j.translucency, SIMD4(0.5, 1, 0.5, 0.25))
    }

    func testInstanceFieldsRoundTrip() {
        var i = IlluminatoramaInstance(modelMatrix: matrix_identity_float4x4)
        XCTAssertEqual(i.thinFilm, .zero)
        XCTAssertEqual(i.thinFilmThickness, 400); XCTAssertEqual(i.thinFilmIOR, 1.4)
        i.thinFilmWeight = 0.8; i.thinFilmThickness = 320; i.thinFilmIOR = 2.4
        XCTAssertEqual(i.thinFilm, SIMD4(0.8, 320, 2.4, 0))
        i.thinFilmThickness = 400; i.thinFilmIOR = 1.4
        XCTAssertEqual(i.thinFilm, SIMD4(0.8, 0, 0, 0), "defaults are stored as 0")
        XCTAssertEqual(MemoryLayout<IlluminatoramaInstance>.stride, 464)
    }
}
