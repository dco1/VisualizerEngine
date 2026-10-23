import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// **Daydream DH-1300 — the per-material layer parameters (IlluminatoramaMaterialLayers.h): the white
/// furnace, the identity contract and the carriage, run on the shipped shader code.**
///
/// The specular layer's IOR, weight and colour set the dielectric F0 every lobe mixes from. The
/// layered energy bookkeeping (DH-1014) weights the diffuse base by `1 − E_spec(F0)` computed from
/// that same F0, so a WHITE base returns exactly 1 per channel whatever the IOR or the tint. This
/// test holds the shipped code to that the way `IlluminatoramaLayeredBSDFFurnaceTests` and
/// `IlluminatoramaDiffuseRoughnessFurnaceTests` do: a probe appended to IlluminatoramaLighting.metal
/// (so `brdf`, `layeredEnergy`, `layeredRTWeights` and the layer decode are the kernel's own),
/// integrated over the hemisphere (MIS of cosine + GGX sampling, 2 × 4096 points), with the engine's
/// own DFG bake.
///
/// Rows (white dielectric base, roughness 0.3 and 0.8, N·V 1/.7/.4/.15), per channel:
///   • direct  — ∫ brdf(V, L)·cos dω_L with the layer's F0 (every punctual light), multiscatter ON;
///   • env     — the IBL arm: kD·albedo + layeredEnvSpecular(F0);
///   • maximum — the path lane at the primary: albedo × `layeredRTWeights(…, F0)` + the specular IBL.
@MainActor
final class IlluminatoramaMaterialLayersFurnaceTests: XCTestCase {

    private static let probeKernel = """

    // ── DH-1300 material-layer furnace probe (appended by IlluminatoramaMaterialLayersFurnaceTests) ──
    static inline float dh1300RadInv(uint bits) {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return float(bits) * 2.3283064365386963e-10;
    }
    // in:  [r, NdotV, flags, 0] [albedo.rgb, 0] [materialExt] [specularTint]
    // out: [direct.rgb, 0] [env.rgb, 0] [maximum.rgb, 0] [F0.rgb, any]
    kernel void dh1300Furnace(device const float4* inp [[buffer(0)]],
                              device float4* outp [[buffer(1)]],
                              texture2d<half, access::sample> dfg [[texture(0)]],
                              uint gid [[thread_position_in_grid]]) {
        float4 a = inp[gid * 4];
        float r = a.x, mu = a.y; uint flags = uint(a.z + 0.5);
        float3 albedo = inp[gid * 4 + 1].xyz;
        MaterialLayers ml = illumiDecodeMaterialLayers(inp[gid * 4 + 2], inp[gid * 4 + 3]);
        float3 dielF0 = illumiDielectricF0(ml);
        float3 N = float3(0, 0, 1);
        float3 V = float3(sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu);
        bool ms = (flags & kFrameFlagLayeredMultiscatter) != 0u;
        constexpr sampler ls(filter::linear, address::clamp_to_edge);
        float4 lut = float4(dfg.sample(ls, float2(mu, r)));
        LayeredEnergy le = layeredOff();
        le.dielectricF0 = dielF0;
        if (ms) le = layeredEnergy(albedo, 0.0, lut, 0.0, 0.0, true, false, dielF0);

        const uint NS = 4096u;
        float3 direct = float3(0.0);
        float a2 = r * r * r * r;
        for (uint s = 0u; s < 2u * NS; ++s) {
            uint i = s % NS;
            float u1 = (float(i) + 0.5) / float(NS), u2 = dh1300RadInv(i);
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
            direct += brdf(N, V, L, albedo, 0.0, r, float3(1.0), 0.0, float3(0.0),
                           0.0, nullptr, kClothSheenRoughness, le) * w;
        }

        float3 kD = (1.0 - fresnelSchlickRoughness(mu, dielF0, r));
        if (ms) kD = le.diffuseEnv;
        float3 specW = layeredEnvSpecular(dielF0, lut.xy);
        float3 env = kD * albedo + specW;

        uint pathFlags = ms ? kPathFlagLayeredMultiscatter : 0u;
        float3 fresLegacy = dielF0 + (1.0 - dielF0) * pow(1.0 - mu, 5.0);
        LayeredRTWeights rw = layeredRTWeights(pathFlags, albedo, 0.0, r, mu, 0.0h, dfg, fresLegacy, 0.0, dielF0);
        float3 maximum = albedo * rw.diffuse + specW;

        outp[gid * 4]     = float4(direct, 0.0);
        outp[gid * 4 + 1] = float4(env, 0.0);
        outp[gid * 4 + 2] = float4(maximum, 0.0);
        outp[gid * 4 + 3] = float4(dielF0, ml.any ? 1.0 : 0.0);
    }
    """

    struct Row {
        var r: Float; var mu: Float; var flags: UInt32
        var albedo: SIMD3<Float> = SIMD3(repeating: 1)
        var materialExt: SIMD4<Float> = .zero
        var specularTint: SIMD4<Float> = .zero
    }
    struct Out { var direct, env, maximum, f0: SIMD3<Float>; var any: Bool }
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
        guard let fn = lib.makeFunction(name: "dh1300Furnace") else { throw XCTSkip("probe missing") }
        let probe = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        let r = Rig(device: device, queue: queue, probe: probe, lut: lut)
        Self.cachedRig = r
        return r
    }

    func run(_ rows: [Row]) throws -> [Out] {
        let rig = try rig()
        var packed: [SIMD4<Float>] = []
        for r in rows {
            packed.append(SIMD4(r.r, r.mu, Float(r.flags), 0))
            packed.append(SIMD4(r.albedo, 0))
            packed.append(r.materialExt)
            packed.append(r.specularTint)
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
            Out(direct: v(i * 4), env: v(i * 4 + 1), maximum: v(i * 4 + 2), f0: v(i * 4 + 3), any: o[i * 4 + 3].w > 0.5)
        }
    }

    static let views: [Float] = [1.0, 0.7, 0.4, 0.15]
    static let ms: UInt32 = 4      // kFrameFlagLayeredMultiscatter

    private func say(_ s: String) { print(s); fflush(stdout) }
    private func f(_ x: Float) -> String { String(format: "%.3f", x) }
    private func rgb(_ v: SIMD3<Float>) -> String { v.x == v.y && v.y == v.z ? f(v.x) : "(\(f(v.x)),\(f(v.y)),\(f(v.z)))" }

    /// The specular layers DH-1300 ships: water, the default, a glaze, calcite, melamine, a high-index
    /// gem, a half-weight specular, and a tinted one.
    static let specularCases: [(name: String, ext: SIMD4<Float>, tint: SIMD4<Float>)] = [
        ("ior 1.33 (water)",     SIMD4(0, 1.33, 0, 0), .zero),
        ("ior 1.54 (glaze)",     SIMD4(0, 1.54, 0, 0), .zero),
        ("ior 1.57 (calcite)",   SIMD4(0, 1.57, 0, 0), .zero),
        ("ior 1.65 (melamine)",  SIMD4(0, 1.65, 0, 0), .zero),
        ("ior 2.42 (diamond)",   SIMD4(0, 2.42, 0, 0), .zero),
        ("weight 0.5",           .zero, SIMD4(1, 1, 1, 0.5)),
        ("colour (.4,.7,1) ×2",  SIMD4(0, 1.6, 0, 0), SIMD4(0.4, 0.7, 1.0, 2.0)),
    ]

    /// THE GATE: a white dielectric returns 1.000 per channel on every arm for every specular layer —
    /// the base takes exactly what the (possibly tinted) specular did not reflect.
    func testWhiteFurnaceWithPerMaterialSpecular() throws {
        for base in [Float(0.3), 0.8] {
            var rows: [Row] = []
            for c in Self.specularCases { for mu in Self.views {
                rows.append(Row(r: base, mu: mu, flags: Self.ms, materialExt: c.ext, specularTint: c.tint))
            } }
            let out = try run(rows)
            say("── DH-1300 white furnace — white DIELECTRIC, roughness \(base), layered multiscatter ON; per N·V 1/.7/.4/.15 ──")
            say("  layer                | F0                  | direct                  | env                     | maximum")
            for (k, c) in Self.specularCases.enumerated() {
                let o = Array(out[(k * Self.views.count)..<((k + 1) * Self.views.count)])
                func list(_ kp: KeyPath<Out, SIMD3<Float>>) -> String { o.map { rgb($0[keyPath: kp]) }.joined(separator: "/") }
                say("  " + c.name.padding(toLength: 20, withPad: " ", startingAt: 0) + " | " + rgb(o[0].f0)
                    + " | " + list(\.direct) + " | " + list(\.env) + " | " + list(\.maximum))
                for row in o {
                    XCTAssertTrue(row.any, "\(c.name) carries layer parameters")
                    for ch in 0..<3 {
                        XCTAssertEqual(row.direct[ch], 1.0, accuracy: 0.02, "direct furnace \(c.name) r \(base) ch \(ch)")
                        XCTAssertEqual(row.env[ch], 1.0, accuracy: 0.01, "env furnace \(c.name) r \(base) ch \(ch)")
                        XCTAssertEqual(row.maximum[ch], 1.0, accuracy: 0.01, "Maximum furnace \(c.name) r \(base) ch \(ch)")
                    }
                }
            }
        }
    }

    /// The F0 the layer decode hands every lobe: ((n − 1)/(n + 1))² × weight × colour, and EXACTLY the
    /// literal 0.04 — not F0(1.5) recomputed — for an instance that carries none (the identity contract
    /// the whole lane rests on). Also: an untouched instance's rows are bit-identical to the legacy
    /// `layeredOff()` / `layeredEnergy` call the kernel made before DH-1300.
    func testDecodeAndIdentity() throws {
        let none = try run(Self.views.map { Row(r: 0.5, mu: $0, flags: Self.ms) })
        for o in none {
            XCTAssertFalse(o.any)
            XCTAssertEqual(o.f0, SIMD3(repeating: Float(0.04)), "no layer parameters ⇒ the literal 0.04")
        }
        let water = try run([Row(r: 0.5, mu: 1, flags: Self.ms, materialExt: SIMD4(0, 1.33, 0, 0))])[0]
        XCTAssertEqual(water.f0.x, pow((1.33 - 1) / (1.33 + 1), 2), accuracy: 1e-6)
        let tinted = try run([Row(r: 0.5, mu: 1, flags: Self.ms, specularTint: SIMD4(0.5, 1, 0.25, 0.8))])[0]
        XCTAssertEqual(tinted.f0.x, 0.04 * 0.5 * 0.8, accuracy: 1e-5)
        XCTAssertEqual(tinted.f0.z, 0.04 * 0.25 * 0.8, accuracy: 1e-5)
        // The multiscatter-OFF (legacy) arm with no layer: the same numbers the kernel always produced.
        let legacy = try run(Self.views.map { Row(r: 0.5, mu: $0, flags: 0) })
        for o in legacy { XCTAssertEqual(o.f0, SIMD3(repeating: Float(0.04))) }
    }

    /// The carriage: the Swift accessors store the defaults as the zero sentinel (so an untouched
    /// instance carries no layer and the G-buffer writes none), and the float4 offsets the RT kernel
    /// reads the record by match the struct (the Metal side static_asserts the same numbers).
    func testSwiftCarriage() {
        var inst = IlluminatoramaInstance(modelMatrix: matrix_identity_float4x4)
        XCTAssertEqual(inst.specularIOR, 1.5); XCTAssertEqual(inst.specularWeight, 1)
        XCTAssertEqual(inst.specularColor, SIMD3(1, 1, 1))
        inst.specularIOR = 1.5; inst.specularWeight = 1; inst.specularColor = SIMD3(1, 1, 1)
        XCTAssertEqual(inst.materialExt, .zero, "the default IOR is stored as the sentinel")
        XCTAssertEqual(inst.specularTint, .zero, "the default tint is stored as the sentinel")
        inst.specularIOR = 1.33
        XCTAssertEqual(inst.materialExt.y, 1.33)
        inst.specularWeight = 0.5
        XCTAssertEqual(inst.specularTint, SIMD4(1, 1, 1, 0.5))
        inst.specularColor = SIMD3(0.2, 0.4, 0.6)
        XCTAssertEqual(inst.specularTint, SIMD4(0.2, 0.4, 0.6, 0.5))
        XCTAssertEqual(MemoryLayout<IlluminatoramaInstance>.stride, 416)
        XCTAssertEqual(MemoryLayout<IlluminatoramaInstance>.offset(of: \.metallic), 9 * 16)
        XCTAssertEqual(MemoryLayout<IlluminatoramaInstance>.offset(of: \.materialExt), 24 * 16)
        XCTAssertEqual(MemoryLayout<IlluminatoramaInstance>.offset(of: \.specularTint), 25 * 16)
    }
}
