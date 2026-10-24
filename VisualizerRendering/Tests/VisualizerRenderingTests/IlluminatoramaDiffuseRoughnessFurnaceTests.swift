import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// **Daydream DH-1301 — diffuse roughness (EON), the white furnace and the lane-agreement check,
/// run on the shipped shader code.**
///
/// The EON lobe (`IlluminatoramaDiffuseRoughness.h`, Portsmouth et al. 2025 — OpenPBR's
/// `base_diffuse_roughness`) is energy-preserving by construction: a white surface sends back
/// exactly what it receives at every roughness and every view. This test holds the shipped code to
/// that, the same way `IlluminatoramaLayeredBSDFFurnaceTests` holds the specular stack, by
/// appending a probe to IlluminatoramaLighting.metal (so `brdf`, `layeredEnergy`,
/// `layeredRTWeights` and the EON helpers are the kernel's own) and integrating over the
/// hemisphere (MIS of cosine + GGX sampling, 2 × 4096 points), with the engine's own DFG bake.
///
/// Rows (white albedo, dielectric base, base roughness 0.5 and 1.0, N·V 1/.7/.4/.15):
///   • direct  — ∫ brdf(V, L)·cos dω_L with the pixel's diffuse roughness set (every punctual
///               light's lobe), layered multiscatter ON — the energy-conserving configuration;
///   • env     — the IBL arm: kD·(E_EON/ρ)·albedo + specular;
///   • maximum — the path lane at the primary: albedo × `layeredRTWeights(…, diffuseRoughness)`
///               (traced irradiance = 1) + the deferred specular IBL.
/// …and the AGREEMENT check that keeps the lanes on one footing: for a COLOURED albedo the
/// direct lobe's own hemispherical integral (the full EON lobe a lamp sees) must equal the
/// directional albedo the integrated sources use (`illumiEONAlbedoRatio`) — so a lamp-lit and a
/// sky-lit wall of the same paint are the same paint.
@MainActor
final class IlluminatoramaDiffuseRoughnessFurnaceTests: XCTestCase {

    private static let probeKernel = """

    // ── DH-1301 diffuse-roughness furnace probe (appended by IlluminatoramaDiffuseRoughnessFurnaceTests) ──
    static inline float dh1301RadInv(uint bits) {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return float(bits) * 2.3283064365386963e-10;
    }
    // in:  [r, NdotV, diffuseRoughness, flags] [albedo.rgb, 0]
    // out: [direct, env, maximum, 0] [diffuseLobeIntegral, albedoRatio, reciprocity, 0]
    kernel void dh1301Furnace(device const float4* inp [[buffer(0)]],
                              device float4* outp [[buffer(1)]],
                              texture2d<half, access::sample> dfg [[texture(0)]],
                              uint gid [[thread_position_in_grid]]) {
        float4 a = inp[gid * 2], b = inp[gid * 2 + 1];
        float r = a.x, mu = a.y, dr = a.z; uint flags = uint(a.w + 0.5);
        float3 albedo = b.xyz;
        float3 N = float3(0, 0, 1);
        float3 V = float3(sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu);
        bool ms = (flags & kFrameFlagLayeredMultiscatter) != 0u;
        constexpr sampler ls(filter::linear, address::clamp_to_edge);
        float4 lut = float4(dfg.sample(ls, float2(mu, r)));
        LayeredEnergy le = ms ? layeredEnergy(albedo, 0.0, lut, 0.0, 0.0, true, false) : layeredOff();
        le = layeredWithDiffuseRoughness(le, albedo, dr, mu);

        const uint NS = 4096u;
        float direct = 0.0, lobe = 0.0, recip = 0.0;
        float a2 = r * r * r * r;
        for (uint s = 0u; s < 2u * NS; ++s) {
            uint i = s % NS;
            float u1 = (float(i) + 0.5) / float(NS), u2 = dh1301RadInv(i);
            float3 L;
            if (s < NS) {
                float rr = sqrt(u1), ph = 2.0 * M_PI_F * u2;
                L = float3(rr * cos(ph), rr * sin(ph), sqrt(max(0.0, 1.0 - u1)));
            } else {
                float ph = 2.0 * M_PI_F * u2;
                float ct = sqrt((1.0 - u1) / (1.0 + (a2 - 1.0) * u1));
                float st = sqrt(max(0.0, 1.0 - ct * ct));
                float3 H = float3(st * cos(ph), st * sin(ph), ct);
                L = 2.0 * dot(V, H) * H - V;
            }
            if (L.z <= 0.0) continue;
            L = normalize(L);
            float3 H = normalize(V + L);
            float pc = L.z / M_PI_F;
            float dd = H.z * H.z * (a2 - 1.0) + 1.0;
            float pg = (a2 / (M_PI_F * dd * dd)) * H.z / (4.0 * max(dot(V, H), 1e-6));
            float w = 1.0 / (float(NS) * (pc + pg));
            float3 f = brdf(N, V, L, albedo, 0.0, r, float3(1.0), 0.0, float3(0.0),
                            0.0, nullptr, kClothSheenRoughness, le);
            direct += f.g * w;
            // The diffuse lobe alone, relative to Lambert: ∫ (f_EON/ρ)·cos dω = E_EON/ρ.
            float3 ratio = illumiEONLambertRatio(albedo, dr, L.z, mu, dot(L, V));
            lobe += ratio.g * (L.z / M_PI_F) * w;
            // Reciprocity: f(L, V) == f(V, L).
            float3 swapped = illumiEONLambertRatio(albedo, dr, mu, L.z, dot(L, V));
            recip = max(recip, abs(ratio.g - swapped.g));
        }

        float3 F0 = float3(0.04);
        float3 kD = (1.0 - fresnelSchlickRoughness(mu, F0, r));
        if (ms) kD = le.diffuseEnv;
        kD *= le.eonAlbedoRatio;
        float3 specW = layeredEnvSpecular(F0, lut.xy);
        float env = (kD * albedo + specW).g;

        uint pathFlags = ms ? kPathFlagLayeredMultiscatter : 0u;
        float3 fresLegacy = F0 + (1.0 - F0) * pow(1.0 - mu, 5.0);
        LayeredRTWeights rw = layeredRTWeights(pathFlags, albedo, 0.0, r, mu, 0.0h, dfg, fresLegacy, dr);
        float maximum = (albedo * rw.diffuse + specW).g;

        outp[gid * 2]     = float4(direct, env, maximum, 0.0);
        outp[gid * 2 + 1] = float4(lobe, le.eonAlbedoRatio.g, recip, 0.0);
    }
    """

    private struct Row { var r: Float; var mu: Float; var dr: Float; var flags: UInt32; var albedo: SIMD3<Float> }
    private struct Out { var direct, env, maximum, lobe, ratio, recip: Double }
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
        guard let fn = lib.makeFunction(name: "dh1301Furnace") else { throw XCTSkip("probe missing") }
        let probe = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        let r = Rig(device: device, queue: queue, probe: probe, lut: lut)
        Self.cachedRig = r
        return r
    }

    private func run(_ rows: [Row]) throws -> [Out] {
        let rig = try rig()
        var packed: [SIMD4<Float>] = []
        for r in rows {
            packed.append(SIMD4(r.r, r.mu, r.dr, Float(r.flags)))
            packed.append(SIMD4(r.albedo, 0))
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
        return (0..<rows.count).map { i in
            Out(direct: Double(o[i * 2].x), env: Double(o[i * 2].y), maximum: Double(o[i * 2].z),
                lobe: Double(o[i * 2 + 1].x), ratio: Double(o[i * 2 + 1].y), recip: Double(o[i * 2 + 1].z))
        }
    }

    static let views: [Float] = [1.0, 0.7, 0.4, 0.15]
    static let diffuseRoughnesses: [Float] = [0, 0.25, 0.5, 0.75, 1.0]
    static let ms: UInt32 = 4      // kFrameFlagLayeredMultiscatter

    private func say(_ s: String) { print(s); fflush(stdout) }
    private func f(_ x: Double) -> String { String(format: "%.3f", x) }

    /// THE GATE: a white dielectric with diffuse roughness returns 1.000 on every arm, at every
    /// diffuse roughness, every view, base roughness 0.5 and 1.0 — and diffuse roughness 0 is the
    /// pre-DH-1301 row exactly.
    func testWhiteFurnaceWithDiffuseRoughness() throws {
        for base in [Float(0.5), 1.0] {
            var rows: [Row] = []
            for dr in Self.diffuseRoughnesses { for mu in Self.views {
                rows.append(Row(r: base, mu: mu, dr: dr, flags: Self.ms, albedo: SIMD3(repeating: 1)))
            } }
            let out = try run(rows)
            say("── DH-1301 white furnace — white DIELECTRIC, base roughness \(base), layered multiscatter ON; per N·V 1/.7/.4/.15 ──")
            say("  diffuse r | direct                  | env                     | maximum")
            for (k, dr) in Self.diffuseRoughnesses.enumerated() {
                let o = Array(out[(k * Self.views.count)..<((k + 1) * Self.views.count)])
                func list(_ kp: KeyPath<Out, Double>) -> String { o.map { f($0[keyPath: kp]) }.joined(separator: "/") }
                say(String(format: "  %.2f      | ", dr) + list(\.direct) + " | " + list(\.env) + " | " + list(\.maximum))
                for row in o {
                    XCTAssertEqual(row.direct, 1.0, accuracy: 0.02, "direct furnace dr \(dr) base \(base)")
                    XCTAssertEqual(row.env, 1.0, accuracy: 0.01, "env furnace dr \(dr) base \(base)")
                    XCTAssertEqual(row.maximum, 1.0, accuracy: 0.01, "Maximum furnace dr \(dr) base \(base)")
                    XCTAssertEqual(row.ratio, 1.0, accuracy: 1e-4, "a white surface's EON albedo ratio is 1")
                }
            }
            // dr = 0 must be the Lambert rows bit for bit (the helpers return the literal 1.0).
            let zero = Array(out[0..<Self.views.count])
            let lambert = try run(Self.views.map { Row(r: base, mu: $0, dr: -1, flags: Self.ms, albedo: SIMD3(repeating: 1)) })
            for (z, l) in zip(zero, lambert) {
                XCTAssertEqual(z.direct, l.direct, "diffuse roughness 0 must be byte-identical to Lambert")
                XCTAssertEqual(z.env, l.env); XCTAssertEqual(z.maximum, l.maximum)
            }
        }
    }

    /// The two halves of the model must be ONE lobe: for a coloured albedo, the full EON lobe a lamp
    /// sees, integrated over the hemisphere, is the directional albedo the integrated sources (IBL,
    /// area lights, traced GI, path vertices) use. Also: the lobe is reciprocal, and rough diffuse
    /// darkens a coloured albedo head-on (multiple scattering saturates it) while white stays 1.
    func testDirectLobeAgreesWithIntegratedSourcesForColouredAlbedo() throws {
        var rows: [Row] = []
        let albedos: [Float] = [0.8, 0.5, 0.2]
        for a in albedos { for dr in Self.diffuseRoughnesses where dr > 0 { for mu in Self.views {
            rows.append(Row(r: 0.8, mu: mu, dr: dr, flags: 0, albedo: SIMD3(repeating: a)))
        } } }
        let out = try run(rows)
        say("── DH-1301 EON lobe ∫ vs directional albedo E_EON/ρ (per N·V 1/.7/.4/.15) ──")
        var k = 0
        for a in albedos { for dr in Self.diffuseRoughnesses where dr > 0 {
            let o = Array(out[k..<(k + Self.views.count)]); k += Self.views.count
            say(String(format: "  ρ %.1f r %.2f | lobe ", a, dr) + o.map { f($0.lobe) }.joined(separator: "/")
                  + " | E/ρ " + o.map { f($0.ratio) }.joined(separator: "/")
                  + String(format: " | reciprocity err %.1e", o.map(\.recip).max()!))
            for row in o {
                XCTAssertEqual(row.lobe, row.ratio, accuracy: 0.003, "lamp lobe vs integrated-source albedo, ρ \(a) r \(dr)")
                XCTAssertLessThan(row.recip, 1e-4, "EON must be reciprocal")
                XCTAssertLessThanOrEqual(row.ratio, 1.0 + 1e-4, "a coloured rough surface can only lose to multiple scattering")
            }
            XCTAssertLessThan(o[0].ratio, o[3].ratio, "head-on darker than grazing for ρ < 1")
        } }
    }

    /// The ray-traced rows' carriage: `rtPackBaseLane` against a mirror of `secondaryMaterial` /
    /// `secondaryDiffuseRoughness` (IlluminatoramaSecondary.h). A pure dielectric with a diffuse
    /// roughness decodes as metalness 0 + its roughness + that diffuse roughness; everything else is
    /// the DH-1308 lane bit for bit.
    func testBaseLaneCarriesDiffuseRoughnessOnlyForPureDielectrics() {
        func decode(_ w: Float) -> (m: Float, r: Float, dr: Float) {
            let x = -w - 1, aq = (x * 0.5).rounded(.down)
            let a = aq * 0.001, r = min(max(x - 2 * aq, 0), 1)
            return a > 1.0005 ? (0, r, min(max(a - 1.001, 0), 1)) : (a, r, 0)
        }
        for dr in stride(from: Float(0), through: 1, by: 0.05) { for r in stride(from: Float(0), through: 1, by: 0.037) {
            var inst = IlluminatoramaInstance(modelMatrix: matrix_identity_float4x4, metallic: 0, roughness: r)
            inst.diffuseRoughness = dr
            let d = decode(IlluminatoramaRenderer.rtPackBaseLane(inst))
            XCTAssertEqual(d.m, 0); XCTAssertEqual(d.r, min(r, 0.999), accuracy: 6e-4)
            XCTAssertEqual(d.dr, dr, accuracy: 6e-4)
            if dr == 0 {
                XCTAssertEqual(IlluminatoramaRenderer.rtPackBaseLane(inst), IlluminatoramaRenderer.rtPackMaterialLane(0, r))
            }
        } }
        var metal = IlluminatoramaInstance(modelMatrix: matrix_identity_float4x4, metallic: 0.3, roughness: 0.5)
        metal.diffuseRoughness = 1
        XCTAssertEqual(IlluminatoramaRenderer.rtPackBaseLane(metal), IlluminatoramaRenderer.rtPackMaterialLane(0.3, 0.5),
                       "a partly metallic row keeps its metalness and is Lambert")
        XCTAssertEqual(MemoryLayout<IlluminatoramaInstance>.stride, 432)
    }
}
