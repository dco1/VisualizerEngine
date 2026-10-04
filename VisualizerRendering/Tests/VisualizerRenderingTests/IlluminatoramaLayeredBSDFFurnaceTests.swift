import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// **Daydream DH-1014 — the white furnace, run on the shipped shader code.**
///
/// A white surface inside a uniform environment of radiance 1 must send back exactly 1 — at
/// every roughness, every view angle, metal or dielectric — because a passive surface can
/// neither create light nor (if it is white) absorb it. Any departure is energy the BSDF
/// invents or loses, and the departure grows with roughness on a single-scatter microfacet
/// lobe: the light that would have bounced twice between microfacets is simply dropped.
///
/// What runs here is NOT a re-derivation. The DFG LUT is baked by the engine's own
/// `illumi_dfg_bake` (IlluminatoramaIBLBake.metal, compiled from source), and a probe kernel
/// appended to IlluminatoramaLighting.metal integrates the deferred kernel's own `brdf()` —
/// every punctual / area light's lobe — over the hemisphere (MIS of cosine and GGX sampling,
/// 2 × 4096 Hammersley points), evaluates the environment arm from the same helpers the kernel
/// calls (`layeredEnvSpecular`, `layeredEnergy`), and evaluates the ray-traced kernel's
/// replacement weights (`layeredRTWeights`) the way Maximum composites them. Each row is
/// measured twice in ONE run — host flags off (the historical expressions) and on (DH-1014) —
/// so the before/after is the same LUT, the same device, the same binary.
///
/// The three arms (all for a white albedo, F0 = 1 metal or F0 = 0.04 dielectric):
///   • direct   — ∫ brdf(V, L)·cos dω_L: the sun / lamps / fills / portals, under a uniform light;
///   • env      — the IBL arm: kD·E_irr·albedo + specEnv·envBRDF, with E_irr = specEnv = 1;
///   • maximum  — the path lane at the primary: traced irradiance (= 1 in a furnace) × albedo ×
///                the RT diffuse weight, plus the deferred specular IBL the lane keeps.
@MainActor
final class IlluminatoramaLayeredBSDFFurnaceTests: XCTestCase {

    private static let probeKernel = """

    // ── DH-1014 white-furnace probe (appended by IlluminatoramaLayeredBSDFFurnaceTests) ──
    static inline float dh1014RadInv(uint bits) {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return float(bits) * 2.3283064365386963e-10;
    }
    // in: [r, NdotV, metallic, flags] [sheenStrength, sheenRoughness, 0, 0]
    // out: [direct, env, maximum, maximumReflReplaced] [directSpecOnly, directPeakClipped, 0, 0]
    kernel void dh1014Furnace(device const float4* inp [[buffer(0)]],
                              device float4* outp [[buffer(1)]],
                              texture2d<half, access::sample> dfg [[texture(0)]],
                              uint gid [[thread_position_in_grid]]) {
        float4 a = inp[gid * 2], b = inp[gid * 2 + 1];
        float r = a.x, mu = a.y, metal = a.z; uint flags = uint(a.w + 0.5);   // a plain float: a bit-cast small uint is a denormal the GPU may flush
        float sheenS = b.x, sheenR = b.y;
        float3 albedo = float3(1.0);
        float3 N = float3(0, 0, 1);
        float3 V = float3(sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu);
        bool ms = (flags & kFrameFlagLayeredMultiscatter) != 0u;
        bool sh = (flags & kFrameFlagLayeredSheen) != 0u;
        constexpr sampler ls(filter::linear, address::clamp_to_edge);
        float4 lut = float4(dfg.sample(ls, float2(mu, r)));
        float  sheenA = (sh && sheenS > 0.0) ? float(dfg.sample(ls, float2(mu, sheenR)).a) : 0.0;
        LayeredEnergy le = layeredOff();
        if (ms || sh) le = layeredEnergy(albedo, metal, lut, sheenA, sheenS, ms, sh);

        // ── direct: ∫ brdf·cos dω, balance-heuristic MIS of cosine + GGX-NDF sampling ──
        const uint NS = 4096u;
        float direct = 0.0, directSpec = lut.z;
        float a2 = r * r * r * r;
        for (uint s = 0u; s < 2u * NS; ++s) {
            uint i = s % NS;
            float u1 = (float(i) + 0.5) / float(NS), u2 = dh1014RadInv(i);
            float3 L;
            if (s < NS) {                                   // cosine
                float rr = sqrt(u1), ph = 2.0 * M_PI_F * u2;
                L = float3(rr * cos(ph), rr * sin(ph), sqrt(max(0.0, 1.0 - u1)));
            } else {                                        // GGX NDF (half-vector)
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
            // The IDEAL GGX density (no guard) — the sampler's own pdf.
            float dd = H.z * H.z * (a2 - 1.0) + 1.0;
            float Dideal = a2 / (M_PI_F * dd * dd);
            float pg = Dideal * H.z / (4.0 * max(dot(V, H), 1e-6));
            float w = 1.0 / (float(NS) * (pc + pg));
            float3 f = brdf(N, V, L, albedo, metal, r, float3(1.0), 0.0, float3(0.0),
                            sheenS, nullptr, sheenR, le);
            direct += f.g * w;
        }

        // ── env: the IBL arm's composition (IlluminatoramaLighting.metal, `kLightingIBLEnabled`)
        float3 F0 = mix(float3(0.04), albedo, metal);
        float3 Fr = fresnelSchlickRoughness(mu, F0, r);
        float3 kD = (1.0 - Fr) * (1.0 - metal);
        if (ms) kD = le.diffuseEnv;
        if (sh) kD *= le.baseUnderSheen;
        float3 specW = layeredEnvSpecular(F0, lut.xy);
        if (sh) specW *= le.baseUnderSheen;
        float sheenE = sh ? le.sheenAlbedo : clothSheenEnvAlbedo(mu);
        float3 sheenEnv = (sheenS > 0.0) ? clothSheenColor(albedo) * sheenS * sheenE : float3(0.0);
        float env = (kD * albedo + specW + sheenEnv).g;

        // ── maximum: the path lane owns the diffuse; the deferred keeps specular IBL + sheen ──
        uint pathFlags = (ms ? kPathFlagLayeredMultiscatter : 0u) | (sh ? kPathFlagLayeredSheen : 0u);
        // emission.alpha as the G-buffer packs it: -(band + strength).
        half emA = sheenS > 0.0 ? half(-(float(clothSheenBandForRoughness(sheenR)) + min(sheenS, 0.98))) : 0.0h;
        float3 fresLegacy = F0 + (1.0 - F0) * pow(1.0 - mu, 5.0);
        LayeredRTWeights w = layeredRTWeights(pathFlags, albedo, metal, r, mu, emA, dfg, fresLegacy);
        float maximum = (albedo * w.diffuse + specW + sheenEnv).g;
        // …and where an RT reflection ray HIT: the reflection replaces the specular IBL.
        float maximumRefl = (albedo * w.diffuse + w.reflection + sheenEnv).g;

        // How much of the direct lobe the D guard (`+1e-7` in distributionGGX) clips at the peak.
        float dpk = a2 / (M_PI_F * a2 * a2 + 1e-7);
        float clip = dpk * (M_PI_F * a2);              // D_guarded(peak) / D_ideal(peak)
        outp[gid * 2]     = float4(direct, env, maximum, maximumRefl);
        outp[gid * 2 + 1] = float4(directSpec, clip, 0.0, 0.0);
    }
    """

    private struct Row { var r: Float; var mu: Float; var metal: Float; var flags: UInt32
                         var sheen: Float = 0; var sheenR: Float = 0.30 }
    private struct Out { var direct: Double; var env: Double; var maximum: Double; var maxRefl: Double
                         var directSpec: Double; var peak: Double; }

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
        // 1. The engine's own DFG bake, compiled from source, at the renderer's size and remap.
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
        // 2. The lighting kernel's source + the probe.
        let src = try MetalSourceLoader.source(contentsOf: Self.shaderURL("IlluminatoramaLighting.metal")) + Self.probeKernel
        let lib = try device.makeLibrary(source: src, options: nil)
        guard let fn = lib.makeFunction(name: "dh1014Furnace") else { throw XCTSkip("probe missing") }
        let probe = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        let r = Rig(device: device, queue: queue, probe: probe, lut: lut)
        Self.cachedRig = r
        return r
    }

    private func run(_ rows: [Row]) throws -> [Out] {
        let rig = try rig()
        var packed: [SIMD4<Float>] = []
        for r in rows {
            packed.append(SIMD4(r.r, r.mu, r.metal, Float(r.flags)))
            packed.append(SIMD4(r.sheen, r.sheenR, 0, 0))
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
                maxRefl: Double(o[i * 2].w), directSpec: Double(o[i * 2 + 1].x), peak: Double(o[i * 2 + 1].y))
        }
    }

    static let roughnesses: [Float] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    static let views: [Float] = [1.0, 0.7, 0.4, 0.15]
    static let off: UInt32 = 0
    static let ms: UInt32 = 4      // kFrameFlagLayeredMultiscatter
    static let msSheen: UInt32 = 12

    /// Mean over the view sweep, the number the report quotes per roughness.
    private func sweep(metal: Float, flags: UInt32, sheen: Float = 0, sheenR: Float = 0.30)
        throws -> [(r: Float, rows: [Out])] {
        var rows: [Row] = []
        for r in Self.roughnesses { for mu in Self.views {
            rows.append(Row(r: r, mu: mu, metal: metal, flags: flags, sheen: sheen, sheenR: sheenR))
        } }
        let out = try run(rows)
        return Self.roughnesses.enumerated().map { (k, r) in
            (r, Array(out[(k * Self.views.count)..<((k + 1) * Self.views.count)]))
        }
    }

    /// stdout is block-buffered under `swift test`; an unflushed table is lost when the runner exits.
    private func say(_ s: String) { print(s); fflush(stdout) }

    private func f(_ x: Double) -> String { String(format: "%.3f", x) }

    /// THE GATE: every arm at 1.000 with the layered model on, across roughness 0.2…1 and view
    /// angles N·V 1…0.15, for a white metal and a white dielectric — and the before/after table.
    func testWhiteFurnaceEveryArmEveryRoughness() throws {
        for (label, metal) in [("white METAL (F0 = 1)", Float(1)), ("white DIELECTRIC (F0 = 0.04)", Float(0))] {
            let before = try sweep(metal: metal, flags: Self.off)
            let after  = try sweep(metal: metal, flags: Self.ms)
            say("── DH-1014 white furnace — \(label); mean over N·V {1, .7, .4, .15} [min…max] ──")
            say("  rough | direct before → after        | env before → after          | maximum before → after       | D-guard peak kept")
            for (k, r) in Self.roughnesses.enumerated() {
                func stat(_ rows: [Out], _ kp: KeyPath<Out, Double>) -> String {
                    let v = rows.map { $0[keyPath: kp] }
                    return "\(f(v.reduce(0, +) / Double(v.count))) [\(f(v.min()!))…\(f(v.max()!))]"
                }
                let b = before[k].rows, a = after[k].rows
                say(String(format: "  %.2f  | ", r) + stat(b, \.direct) + " → " + stat(a, \.direct)
                      + " | " + stat(b, \.env) + " → " + stat(a, \.env)
                      + " | " + stat(b, \.maximum) + " → " + stat(a, \.maximum)
                      + " | " + f(a[0].peak))
                // The gate. Below r ≈ 0.2 the direct lobe is limited by `distributionGGX`'s
                // `+1e-7` peak guard, not by scattering (the last column: the share of the ideal
                // peak the guard keeps) — reported, and filed as DH-1306.
                for row in a {
                    XCTAssertEqual(row.env, 1.0, accuracy: 0.01, "env furnace \(label) r \(r)")
                    XCTAssertEqual(row.maximum, 1.0, accuracy: 0.01, "Maximum furnace \(label) r \(r)")
                    if r >= 0.2 {
                        XCTAssertEqual(row.direct, 1.0, accuracy: 0.02, "direct furnace \(label) r \(r)")
                    }
                }
            }
        }
    }

    /// The single-scatter loss the compensation exists to put back, stated as a number for a
    /// white metal under a uniform LIGHT (the direct lobe, flags off): it must grow with
    /// roughness — the signature of missing multiple scattering — and the compensated lobe
    /// must not.
    func testSingleScatterLossGrowsWithRoughnessAndCompensationRemovesIt() throws {
        let before = try sweep(metal: 1, flags: Self.off)
        let after  = try sweep(metal: 1, flags: Self.ms)
        func mean(_ rows: [Out]) -> Double { rows.map(\.direct).reduce(0, +) / Double(rows.count) }
        let b05 = mean(before[5].rows), b10 = mean(before[10].rows), b03 = mean(before[3].rows)
        XCTAssertLessThan(b10, b05, "single-scatter loss must grow with roughness")
        XCTAssertLessThan(b05, b03 + 1e-3, "single-scatter loss must grow with roughness")
        XCTAssertLessThan(b10, 0.9, "a rough white metal must visibly lose energy single-scatter")
        XCTAssertEqual(mean(after[10].rows), 1.0, accuracy: 0.02)
        XCTAssertEqual(mean(after[5].rows), 1.0, accuracy: 0.02)
    }

    /// Cloth (increment 3): white velvet — strength 0.85, nap 0.45 — must also return 1.0 once
    /// the sheen is a layer (its albedo comes off the base), and returned MORE than it received
    /// while the sheen was bolted on top.
    func testWhiteClothFurnaceWithSheenLayer() throws {
        let before = try sweep(metal: 0, flags: Self.off, sheen: 0.85, sheenR: 0.45)
        let after  = try sweep(metal: 0, flags: Self.msSheen, sheen: 0.85, sheenR: 0.45)
        say("── DH-1014 white furnace — white VELVET (sheen 0.85, nap 0.45), dielectric base ──")
        say("  rough | direct before → after | env before → after | maximum before → after (per N·V 1/.7/.4/.15)")
        for (k, r) in Self.roughnesses.enumerated() where k % 2 == 0 || k == 10 {
            let b = before[k].rows, a = after[k].rows
            func list(_ v: [Out], _ kp: KeyPath<Out, Double>) -> String { v.map { f($0[keyPath: kp]) }.joined(separator: "/") }
            say(String(format: "  %.2f  | ", r) + list(b, \.direct) + " → " + list(a, \.direct)
                  + " | " + list(b, \.env) + " → " + list(a, \.env)
                  + " | " + list(b, \.maximum) + " → " + list(a, \.maximum))
            for row in a where r >= 0.2 {
                XCTAssertEqual(row.env, 1.0, accuracy: 0.01, "velvet env furnace r \(r)")
                XCTAssertEqual(row.maximum, 1.0, accuracy: 0.01, "velvet Maximum furnace r \(r)")
                XCTAssertEqual(row.direct, 1.0, accuracy: 0.02, "velvet direct furnace r \(r)")
            }
            // The bolted-on sheen invented light at grazing, where its lobe lives.
            XCTAssertGreaterThan(b.last!.env, 1.02, "the bolted-on sheen should over-return at grazing")
        }
    }

    /// The LUT's sheen channel against an independent CPU integral of the same Charlie·Neubelt
    /// lobe (midpoint quadrature in θ, φ) — so `.a` is the integral it claims to be, and the
    /// hand-fitted curve it replaces is printed beside it (the magnitude change, on record).
    func testSheenLUTMatchesCPUIntegral() throws {
        let rig = try rig()
        // Read the LUT back (private → shared blit).
        let n = rig.lut.width
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: n, height: n, mipmapped: false)
        desc.storageMode = .shared
        guard let dst = rig.device.makeTexture(descriptor: desc), let cb = rig.queue.makeCommandBuffer(),
              let bl = cb.makeBlitCommandEncoder() else { throw XCTSkip("no blit") }
        bl.copy(from: rig.lut, to: dst); bl.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        var raw = [UInt16](repeating: 0, count: n * n * 4)
        raw.withUnsafeMutableBytes { dst.getBytes($0.baseAddress!, bytesPerRow: n * 8, from: MTLRegionMake2D(0, 0, n, n), mipmapLevel: 0) }
        func texel(_ mu: Double, _ a: Double) -> SIMD4<Double> {
            let x = min(n - 1, max(0, Int(mu * Double(n)))), y = min(n - 1, max(0, Int(a * Double(n))))
            let b = (y * n + x) * 4
            return SIMD4((0..<4).map { Double(Float(Float16(bitPattern: raw[b + $0]))) })
        }
        func cpuSheen(_ mu: Double, _ alpha: Double) -> Double {
            let v = SIMD3(sqrt(max(0, 1 - mu * mu)), 0, mu)
            var acc = 0.0
            let nt = 256, np = 128
            for i in 0..<nt { for j in 0..<np {
                let ct = (Double(i) + 0.5) / Double(nt)            // uniform in cosθ
                let st = sqrt(max(0, 1 - ct * ct)), ph = (Double(j) + 0.5) / Double(np) * 2 * .pi
                let l = SIMD3(st * cos(ph), st * sin(ph), ct)
                let h = simd_normalize(v + l)
                let inv = 1 / max(alpha, 1e-3)
                let s2 = max(1 - h.z * h.z, 1e-7)
                let d = (2 + inv) * pow(s2, inv * 0.5) / (2 * .pi)
                let vis = 1 / max(4 * (ct + mu - ct * mu), 1e-5)
                acc += d * vis * ct
            } }
            return acc * 2 * .pi / Double(nt * np)
        }
        say("── DH-1014 / DH-0597 sheen albedo: LUT .a vs CPU integral vs the old fitted curve ──")
        for alpha in [0.18, 0.30, 0.45, 0.60] {
            var line = String(format: "  α %.2f |", alpha)
            for mu in [0.1, 0.3, 0.5, 0.9] {
                // Texel centres: evaluate the CPU integral at the texel the readback picked.
                let tx = (floor(mu * Double(n)) + 0.5) / Double(n), ty = (floor(alpha * Double(n)) + 0.5) / Double(n)
                let lutA = texel(mu, alpha).w, cpu = cpuSheen(tx, ty)
                let curve = 0.08 + 0.92 * pow(1 - mu, 4)
                line += String(format: " μ %.1f: LUT %.4f CPU %.4f (curve %.4f) |", mu, lutA, cpu, curve)
                XCTAssertEqual(lutA, cpu, accuracy: max(0.004, 0.02 * cpu), "sheen LUT α \(alpha) μ \(mu)")
            }
            say(line)
        }
        // The two env-split channels must still be what they were (rg unchanged by the format).
        // Sanity on the other channels: a mirror keeps everything, a rough lobe at grazing much
        // less, and the direct lobe (analytic k, a heavier G) never more than the env lobe.
        for (mu, r) in [(0.95, 0.05), (0.5, 0.5), (0.15, 1.0)] {
            let t = texel(mu, r)
            say(String(format: "  LUT (N·V %.2f, r %.2f): A %.4f  B %.4f  A+B %.4f  Ed %.4f  Es %.4f", mu, r, t.x, t.y, t.x + t.y, t.z, t.w))
            XCTAssertLessThan(t.x + t.y, 1.01); XCTAssertLessThan(t.z, 1.01)
            XCTAssertLessThanOrEqual(t.z, t.x + t.y + 0.01, "the analytic-k lobe cannot out-reflect the k = α/2 lobe")
        }
        XCTAssertGreaterThan(texel(0.95, 0.05).z, 0.98, "a near-mirror keeps its energy")
        XCTAssertLessThan(texel(0.15, 1.0).z, 0.6, "a rough lobe at grazing loses most of it single-scatter")
    }
}
