import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// **Daydream DH-1309 — the coat as a real layer: the white furnace with a coat, on the shipped code.**
///
/// A coat (OpenPBR's) is a dielectric GGX lobe over everything below it, and the base below keeps
/// only what that lobe did not reflect: `1 − w·E_coat(μo)`, the coat lobe's own directional albedo
/// from the DFG LUT — the direct lobe's under the lights, the environment lobe's under the IBL. So a
/// WHITE base under any coat returns exactly 1 under a uniform light. The Phase-7 clearcoat could
/// not: it lit the coat by the sun and the IBL only, and charged the base `cc·Schlick(N·V)`.
///
/// Probe appended to IlluminatoramaLighting.metal (so `brdf`, `layeredWithCoat`, `layeredCoatDirect`,
/// `layeredRTWeights` are the kernel's own), integrated over the hemisphere — MIS of cosine, GGX at the
/// base roughness and GGX at the COAT's roughness (a polish lobe is far too narrow for the other two)
/// — 3 × 4096 points, with the engine's own DFG bake.
///
/// Rows (white dielectric base, layered multiscatter ON, N·V 1/.7/.4/.15):
///   • direct  — ∫ brdf(V, L)·cos dω_L with the coat layer: base under the coat + the coat lobe;
///   • env     — the IBL arm: (kD·albedo + specular)·(1 − w·E_env) + w·E_env;
///   • maximum — the path lane at the primary: albedo × `layeredRTWeights(…coat)`.diffuse + the traced
///               reflection (the base's specular share, under the coat) + the coat's environment lobe.
@MainActor
final class IlluminatoramaCoatLayerFurnaceTests: XCTestCase {

    private static let probeKernel = """

    // ── DH-1309 coat-layer furnace probe (appended by IlluminatoramaCoatLayerFurnaceTests) ──
    static inline float dh1309RadInv(uint bits) {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return float(bits) * 2.3283064365386963e-10;
    }
    static inline float3 dh1309SampleGGX(float a2, float3 V, float u1, float u2) {
        float ph = 2.0 * M_PI_F * u2;
        float ct = sqrt((1.0 - u1) / (1.0 + (a2 - 1.0) * u1));
        float st = sqrt(max(0.0, 1.0 - ct * ct));
        float3 H = float3(st * cos(ph), st * sin(ph), ct);
        return 2.0 * dot(V, H) * H - V;
    }
    static inline float dh1309PdfGGX(float a2, float3 V, float3 H) {
        float dd = H.z * H.z * (a2 - 1.0) + 1.0;
        return (a2 / (M_PI_F * dd * dd)) * H.z / (4.0 * max(dot(V, H), 1e-6));
    }
    // in:  [r, NdotV, coatW, coatR] [albedo.rgb, coatF0] [coatTint.rgb, 0] [darken.rgb, 0]
    // out: [direct.rgb, 0] [env.rgb, 0] [maximum.rgb, 0] [underDirect, underEnv, EcEnv, 0]
    kernel void dh1309Furnace(device const float4* inp [[buffer(0)]],
                              device float4* outp [[buffer(1)]],
                              texture2d<half, access::sample> dfg [[texture(0)]],
                              uint gid [[thread_position_in_grid]]) {
        float4 a = inp[gid * 4], b = inp[gid * 4 + 1];
        float r = a.x, mu = a.y, cw = a.z, cr = a.w;
        float3 albedo = b.xyz; float cf0 = b.w;
        float3 tint = inp[gid * 4 + 2].xyz, darken = inp[gid * 4 + 3].xyz;
        float3 N = float3(0, 0, 1);
        float3 V = float3(sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu);
        constexpr sampler ls(filter::linear, address::clamp_to_edge);
        float4 lut = float4(dfg.sample(ls, float2(mu, r)));
        float4 lutC = float4(dfg.sample(ls, float2(mu, cr)));
        LayeredEnergy le = layeredEnergy(albedo, 0.0, lut, 0.0, 0.0, true, false);
        le = layeredWithCoat(le, cw, cr, cf0, lutC, tint, darken);

        const uint NS = 4096u;
        float3 direct = float3(0.0);
        float a2 = r * r * r * r, ac2 = pow(max(cr * cr, 1e-4), 2.0);
        for (uint s = 0u; s < 3u * NS; ++s) {
            uint i = s % NS;
            float u1 = (float(i) + 0.5) / float(NS), u2 = dh1309RadInv(i);
            float3 L;
            if (s < NS) {
                float rr = sqrt(u1), ph = 2.0 * M_PI_F * u2;
                L = float3(rr * cos(ph), rr * sin(ph), sqrt(max(0.0, 1.0 - u1)));
            } else if (s < 2u * NS) {
                L = dh1309SampleGGX(a2, V, u1, u2);
            } else {
                L = dh1309SampleGGX(ac2, V, u1, u2);
            }
            if (L.z <= 0.0) continue;
            L = normalize(L);
            float3 H = normalize(V + L);
            float pdf = L.z / M_PI_F + dh1309PdfGGX(a2, V, H) + dh1309PdfGGX(ac2, V, H);
            float w = 1.0 / (float(NS) * pdf);
            direct += brdf(N, V, L, albedo, 0.0, r, float3(1.0), 0.0, float3(0.0),
                           0.0, nullptr, kClothSheenRoughness, le) * w;
        }

        float3 F0 = float3(0.04);
        float3 kD = le.diffuseEnv * le.underCoatEnv * le.coatDarken;
        float3 specW = layeredEnvSpecular(F0, lut.xy) * le.underCoatEnv;
        float3 coatEnv = float3(le.coatWeight * le.coatEnvAlbedo);
        float3 env = kD * albedo + specW + coatEnv;

        float3 fresLegacy = F0 + (1.0 - F0) * pow(1.0 - mu, 5.0);
        LayeredRTWeights rw = layeredRTWeights(kPathFlagLayeredMultiscatter, albedo, 0.0, r, mu, 0.0h, dfg,
                                               fresLegacy, 0.0, F0, cw, cr, cf0, tint, darken);
        float3 maximum = albedo * rw.diffuse + rw.reflection + coatEnv;   // the traced reflection replaces the base's share

        outp[gid * 4]     = float4(direct, 0.0);
        outp[gid * 4 + 1] = float4(env, 0.0);
        outp[gid * 4 + 2] = float4(maximum, 0.0);
        outp[gid * 4 + 3] = float4(le.underCoatDirect.g, le.underCoatEnv.g, le.coatEnvAlbedo, rw.reflRoughness);
    }

    // DH-1299 — fuzz over coat over base: a WHITE cloth (sheen s, nap α) under the sheen, with a coat
    // between. in: [r, NdotV, coatW, coatR] [sheenStrength, sheenRoughness, 0, 0]; out: [direct, env, 0, 0]
    kernel void dh1299Furnace(device const float4* inp [[buffer(0)]], device float4* outp [[buffer(1)]],
                              texture2d<half, access::sample> dfg [[texture(0)]],
                              uint gid [[thread_position_in_grid]]) {
        float4 a = inp[gid * 2], b = inp[gid * 2 + 1];
        float r = a.x, mu = a.y, cw = a.z, cr = a.w, ss = b.x, sr = b.y;
        float3 albedo = float3(1.0);
        float3 N = float3(0, 0, 1);
        float3 V = float3(sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu);
        constexpr sampler ls(filter::linear, address::clamp_to_edge);
        float4 lut = float4(dfg.sample(ls, float2(mu, r)));
        float4 lutC = float4(dfg.sample(ls, float2(mu, cr)));
        float sheenA = float(dfg.sample(ls, float2(mu, sr)).a);
        LayeredEnergy le = layeredEnergy(albedo, 0.0, lut, sheenA, ss, true, true);
        le = layeredWithCoat(le, cw, cr, 0.04, lutC, float3(1.0), float3(1.0));
        const uint NS = 4096u;
        float direct = 0.0;
        float a2 = r * r * r * r, ac2 = pow(max(cr * cr, 1e-4), 2.0);
        for (uint s = 0u; s < 3u * NS; ++s) {
            uint i = s % NS;
            float u1 = (float(i) + 0.5) / float(NS), u2 = dh1309RadInv(i);
            float3 L;
            if (s < NS) { float rr = sqrt(u1), ph = 2.0 * M_PI_F * u2; L = float3(rr * cos(ph), rr * sin(ph), sqrt(max(0.0, 1.0 - u1))); }
            else if (s < 2u * NS) { L = dh1309SampleGGX(a2, V, u1, u2); }
            else { L = dh1309SampleGGX(ac2, V, u1, u2); }
            if (L.z <= 0.0) continue;
            L = normalize(L);
            float3 H = normalize(V + L);
            float w = 1.0 / (float(NS) * (L.z / M_PI_F + dh1309PdfGGX(a2, V, H) + dh1309PdfGGX(ac2, V, H)));
            direct += brdf(N, V, L, albedo, 0.0, r, float3(1.0), 0.0, float3(0.0), ss, nullptr, sr, le).g * w;
        }
        float3 F0 = float3(0.04);
        float3 under = le.underCoatEnv * le.baseUnderSheen;
        float env = (le.diffuseEnv * le.coatDarken * albedo * under + layeredEnvSpecular(F0, lut.xy) * under
                     + le.coatWeight * le.coatEnvAlbedo * le.baseUnderSheen
                     + clothSheenColor(albedo) * ss * le.sheenAlbedo).g;
        outp[gid] = float4(direct, env, 0.0, 0.0);
    }

    // DH-1298 — the coat-optics helpers on the shipped header. in: [albedo.rgb, coatW] [specIOR, coatIOR, darkening, 0]
    // out: [darken.rgb, coatF0] [baseF0UnderCoat.rgb, 0]
    kernel void dh1298Optics(device const float4* inp [[buffer(0)]], device float4* outp [[buffer(1)]],
                             uint gid [[thread_position_in_grid]]) {
        float4 a = inp[gid * 2], b = inp[gid * 2 + 1];
        MaterialLayers m = illumiDecodeMaterialLayers(float4(0.0, a.w, 0.08, 0.0), float4(0.0, b.x, b.y, b.z),
                                                     float4(0.0), float4(0.0));
        outp[gid * 2]     = float4(illumiCoatDarkening(m, a.xyz, a.w), illumiCoatF0(m));
        outp[gid * 2 + 1] = float4(illumiDielectricF0UnderCoat(m, a.w), 0.0);
    }
    """

    struct Row {
        var r: Float; var mu: Float; var coatW: Float; var coatR: Float
        var albedo: SIMD3<Float> = SIMD3(repeating: 1); var coatF0: Float = 0.04
        var tint: SIMD3<Float> = SIMD3(repeating: 1); var darken: SIMD3<Float> = SIMD3(repeating: 1)
    }
    struct Out { var direct, env, maximum: SIMD3<Float>; var underDirect, underEnv, ecEnv, reflRough: Float }
    private struct Rig { let device: MTLDevice; let queue: MTLCommandQueue
                         let probe: MTLComputePipelineState; let lut: MTLTexture }
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
        guard let fn = lib.makeFunction(name: "dh1309Furnace") else { throw XCTSkip("probe missing") }
        let probe = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        let r = Rig(device: device, queue: queue, probe: probe, lut: lut)
        Self.cachedRig = r
        return r
    }

    func run(_ rows: [Row]) throws -> [Out] {
        let rig = try rig()
        var packed: [SIMD4<Float>] = []
        for r in rows {
            packed.append(SIMD4(r.r, r.mu, r.coatW, r.coatR))
            packed.append(SIMD4(r.albedo, r.coatF0))
            packed.append(SIMD4(r.tint, 0))
            packed.append(SIMD4(r.darken, 0))
        }
        guard let ib = rig.device.makeBuffer(bytes: packed, length: packed.count * 16, options: .storageModeShared),
              let ob = rig.device.makeBuffer(length: packed.count * 16, options: .storageModeShared),
              let cb = rig.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no buffers") }
        enc.setComputePipelineState(rig.probe)
        enc.setBuffer(ib, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        enc.setTexture(rig.lut, index: 0)
        enc.dispatchThreads(MTLSize(width: rows.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(32, rows.count), height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: packed.count)
        func v(_ k: Int) -> SIMD3<Float> { SIMD3(o[k].x, o[k].y, o[k].z) }
        return (0..<rows.count).map { i in
            Out(direct: v(i * 4), env: v(i * 4 + 1), maximum: v(i * 4 + 2),
                underDirect: o[i * 4 + 3].x, underEnv: o[i * 4 + 3].y, ecEnv: o[i * 4 + 3].z, reflRough: o[i * 4 + 3].w)
        }
    }

    static let views: [Float] = [1.0, 0.7, 0.4, 0.15]
    private func say(_ s: String) { print(s); fflush(stdout) }
    private func f(_ x: Float) -> String { String(format: "%.3f", x) }

    /// THE GATE: a white base under a coat — every coat weight and roughness the library ships
    /// (polish 0.05–0.10, satin 0.3), under a smooth and a rough base — returns 1 on every arm.
    func testWhiteFurnaceUnderACoat() throws {
        let coats: [(w: Float, r: Float)] = [(1.0, 0.05), (1.0, 0.08), (0.3, 0.08), (1.0, 0.30), (0.5, 0.60)]
        for base in [Float(0.3), 0.8] {
            var rows: [Row] = []
            for c in coats { for mu in Self.views { rows.append(Row(r: base, mu: mu, coatW: c.w, coatR: c.r)) } }
            let out = try run(rows)
            say("── DH-1309 white furnace — white DIELECTRIC base r \(base) under a coat; per N·V 1/.7/.4/.15 ──")
            say("  coat w / r  | direct                  | env                     | maximum                 | base keeps (direct)")
            for (k, c) in coats.enumerated() {
                let o = Array(out[(k * Self.views.count)..<((k + 1) * Self.views.count)])
                func list(_ kp: KeyPath<Out, SIMD3<Float>>) -> String { o.map { f($0[keyPath: kp].y) }.joined(separator: "/") }
                say(String(format: "  %.1f / %.2f  | ", c.w, c.r) + list(\.direct) + " | " + list(\.env) + " | " + list(\.maximum)
                    + " | " + o.map { f($0.underDirect) }.joined(separator: "/"))
                for row in o {
                    XCTAssertEqual(row.direct.y, 1.0, accuracy: 0.02, "direct furnace under coat \(c) base \(base)")
                    XCTAssertEqual(row.env.y, 1.0, accuracy: 0.01, "env furnace under coat \(c) base \(base)")
                    XCTAssertEqual(row.maximum.y, 1.0, accuracy: 0.01, "Maximum furnace under coat \(c) base \(base)")
                    XCTAssertEqual(row.reflRough, base, accuracy: 1e-6, "the traced reflection is the base's, under the coat")
                }
                XCTAssertGreaterThan(o[3].underDirect, 0.3, "the coat cannot take everything at grazing")
                XCTAssertLessThan(o[3].underDirect, o[0].underDirect, "the coat reflects more at grazing")
            }
        }
    }

    /// DH-1298 — a coat of any IOR keeps the white furnace (water 1.33, glaze 1.54, a dense 1.8 lacquer),
    /// with or without the darkening (white is never darkened).
    func testWhiteFurnaceUnderCoatsOfEveryIOR() throws {
        let iors: [Float] = [1.33, 1.54, 1.8]
        var rows: [Row] = []
        for n in iors { for mu in Self.views {
            let f0 = pow((n - 1) / (n + 1), 2)
            rows.append(Row(r: 0.5, mu: mu, coatW: 1, coatR: 0.08, coatF0: f0))
        } }
        let out = try run(rows)
        say("── DH-1298 white furnace — coat w 1 r .08 at coat IOR 1.33 / 1.54 / 1.8 (base r .5); per N·V ──")
        for (k, n) in iors.enumerated() {
            let o = Array(out[(k * Self.views.count)..<((k + 1) * Self.views.count)])
            say(String(format: "  ior %.2f | direct ", n) + o.map { f($0.direct.y) }.joined(separator: "/")
                + " | env " + o.map { f($0.env.y) }.joined(separator: "/") + " | maximum " + o.map { f($0.maximum.y) }.joined(separator: "/"))
            for row in o {
                XCTAssertEqual(row.direct.y, 1, accuracy: 0.02); XCTAssertEqual(row.env.y, 1, accuracy: 0.01)
                XCTAssertEqual(row.maximum.y, 1, accuracy: 0.01)
            }
        }
    }

    /// DH-1298 — the coat-optics helpers against a CPU mirror of OpenPBR's forms: the base's diffuse
    /// darkening ρ(1 − K)/(1 − ρK), K = 1 − (1 − Ē_c)/n², blended by w × coat_darkening; the coat F0;
    /// the base's F0 at the RELATIVE index. White is never darkened; a darker albedo darkens more.
    func testCoatOpticsHelpers() throws {
        let rig = try rig()
        guard let fn = (try? rig.device.makeLibrary(source: MetalSourceLoader.source(contentsOf: Self.shaderURL("IlluminatoramaLighting.metal")) + Self.probeKernel, options: nil))?.makeFunction(name: "dh1298Optics"),
              let pso = try? rig.device.makeComputePipelineState(function: fn) else { throw XCTSkip("no optics probe") }  // gpu-ok: test harness
        let cases: [(albedo: Float, w: Float, nb: Float, nc: Float, dk: Float)] = [
            (1.0, 1, 1.5, 1.5, 1), (0.5, 1, 1.5, 1.5, 1), (0.2, 1, 1.5, 1.5, 1), (0.5, 0.3, 1.56, 1.56, 1),
            (0.5, 1, 1.5, 1.5, 0), (0.5, 1, 1.56, 1.33, 0.5), (0.2, 1, 1.56, 1.33, 1)]
        var packed: [SIMD4<Float>] = []
        for c in cases { packed.append(SIMD4(c.albedo, c.albedo, c.albedo, c.w)); packed.append(SIMD4(c.nb, c.nc, c.dk, 0)) }
        guard let ib = rig.device.makeBuffer(bytes: packed, length: packed.count * 16, options: .storageModeShared),
              let ob = rig.device.makeBuffer(length: packed.count * 16, options: .storageModeShared),
              let cb = rig.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no buffers") }
        enc.setComputePipelineState(pso); enc.setBuffer(ib, offset: 0, index: 0); enc.setBuffer(ob, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: cases.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: cases.count, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: packed.count)
        say("── DH-1298 coat optics — albedo, w, n_base, n_coat, darkening → diffuse factor, coat F0, base F0 ──")
        for (i, c) in cases.enumerated() {
            func f0(_ n: Float) -> Float { pow((n - 1) / (n + 1), 2) }
            let fc = f0(c.nc), eavg = fc + (1 - fc) / 21, K = max(0, 1 - (1 - eavg) / (c.nc * c.nc))
            let d = (1 - K) / (1 - c.albedo * K), dk = 1 + (d - 1) * min(1, c.w * c.dk)
            let nEff = c.nb + (c.nb / c.nc - c.nb) * c.w
            say(String(format: "  ρ %.1f w %.1f n %.2f/%.2f dk %.1f → darken %.3f (cpu %.3f)  F0c %.4f  F0base %.4f (cpu %.4f)",
                       c.albedo, c.w, c.nb, c.nc, c.dk, o[i * 2].x, dk, o[i * 2].w, o[i * 2 + 1].x, f0(nEff)))
            XCTAssertEqual(o[i * 2].x, dk, accuracy: 1e-4)
            XCTAssertEqual(o[i * 2].w, fc, accuracy: 1e-6)
            XCTAssertEqual(o[i * 2 + 1].x, f0(nEff), accuracy: 1e-5)
        }
        XCTAssertEqual(o[0].x, 1, accuracy: 1e-6, "white is never darkened")
        XCTAssertLessThan(o[2 * 2].x, o[1 * 2].x, "a darker albedo darkens more under the coat")
        XCTAssertEqual(o[4 * 2].x, 1, accuracy: 1e-6, "coat_darkening 0 ⇒ none")
    }

    /// DH-1299 — fuzz over coat over base: a white cloth with a coat between (glazed chintz, a satin
    /// over a lacquer) returns 1 on the direct and environment arms — the sheen layer charges everything
    /// below it (coat included) its albedo, the coat everything below it.
    func testWhiteFurnaceFuzzOverCoat() throws {
        let rig = try rig()
        guard let fn = (try? rig.device.makeLibrary(source: MetalSourceLoader.source(contentsOf: Self.shaderURL("IlluminatoramaLighting.metal")) + Self.probeKernel, options: nil))?.makeFunction(name: "dh1299Furnace"),
              let pso = try? rig.device.makeComputePipelineState(function: fn) else { throw XCTSkip("no fuzz probe") }  // gpu-ok: test harness
        let cases: [(s: Float, a: Float, w: Float, r: Float)] = [(0.30, 0.30, 0.35, 0.12), (0.85, 0.45, 0.35, 0.12), (0.30, 0.18, 1.0, 0.08)]
        var packed: [SIMD4<Float>] = []
        for c in cases { for mu in Self.views { packed.append(SIMD4(0.6, mu, c.w, c.r)); packed.append(SIMD4(c.s, c.a, 0, 0)) } }
        let n = packed.count / 2
        guard let ib = rig.device.makeBuffer(bytes: packed, length: packed.count * 16, options: .storageModeShared),
              let ob = rig.device.makeBuffer(length: n * 16, options: .storageModeShared),
              let cb = rig.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no buffers") }
        enc.setComputePipelineState(pso); enc.setBuffer(ib, offset: 0, index: 0); enc.setBuffer(ob, offset: 0, index: 1)
        enc.setTexture(rig.lut, index: 0)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(32, n), height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: n)
        say("── DH-1299 white furnace — fuzz over coat over base (base r .6); per N·V 1/.7/.4/.15 ──")
        for (k, c) in cases.enumerated() {
            let rows = (0..<Self.views.count).map { o[k * Self.views.count + $0] }
            say(String(format: "  sheen %.2f nap %.2f | coat %.2f r %.2f | direct ", c.s, c.a, c.w, c.r)
                + rows.map { f($0.x) }.joined(separator: "/") + " | env " + rows.map { f($0.y) }.joined(separator: "/"))
            for row in rows {
                XCTAssertEqual(row.x, 1, accuracy: 0.02, "direct, fuzz over coat \(c)")
                XCTAssertEqual(row.y, 1, accuracy: 0.01, "env, fuzz over coat \(c)")
            }
        }
    }

    /// No coat ⇒ the layered rows are exactly the uncoated ones (the coat terms are 1 / 0).
    func testNoCoatIsIdentity() throws {
        let coated = try run(Self.views.map { Row(r: 0.5, mu: $0, coatW: 0, coatR: 0.08) })
        for o in coated {
            XCTAssertEqual(o.underDirect, 1); XCTAssertEqual(o.underEnv, 1); XCTAssertEqual(o.ecEnv, 0)
            XCTAssertEqual(o.reflRough, 0.5, accuracy: 1e-6)
        }
    }
}
