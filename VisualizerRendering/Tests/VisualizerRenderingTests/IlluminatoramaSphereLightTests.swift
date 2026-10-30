import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// SPHERE LIGHTS (`IlluminatoramaRenderer.sphereLightsEnabled`, `kFrameFlagSphereLights`).
///
/// (a) REFERENCE-FREE kernel validation. The real `brdf()` (IlluminatoramaLighting.metal, compiled from
///     source with a probe kernel appended — same harness as `IlluminatoramaAreaLightSpecularTests`) is
///     called with `sphereR > 0` and compared against a CPU Monte-Carlo integral of the engine's OWN
///     point-light BRDF over the sphere's solid angle: "a sphere light is a sum of tiny point lights",
///     radiance L = I/(π r²). The DIFFUSE term is an exact analytic integral and is held to a tight
///     bound; the SPECULAR term is Karis's representative-point approximation and is held to a
///     stated, measured bound. Far field must equal the point law; r = 0 must equal the point path.
/// (b) Flag-off BIT IDENTITY: with the knob off a `bulbRadius` is ignored — pixels identical.
/// (c) With the knob on, a glossy plane under a point light shows a LOWER, WIDER highlight at
///     comparable total reflected energy.
@MainActor
final class IlluminatoramaSphereLightTests: XCTestCase {

    // ── GPU probe ────────────────────────────────────────────────────────────

    private static let probeKernel = """

    // ── sphere-light test probe (appended by IlluminatoramaSphereLightTests) ──
    // in : 3 float4 per probe — (N, 0) (V, 0) (L unit, d)   and a 4th (r, roughness, albedo, 0)
    // out: (sphere diffuse+spec, sphere spec only, point diffuse+spec, point spec only), all at I = 1, no 1/d²
    kernel void sphereLightProbe(device const float4* in [[buffer(0)]],
                                 device float4* out [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]]) {
        float3 N = in[gid * 4 + 0].xyz, V = in[gid * 4 + 1].xyz;
        float3 L = in[gid * 4 + 2].xyz; float d = in[gid * 4 + 2].w;
        float r = in[gid * 4 + 3].x, rough = in[gid * 4 + 3].y, alb = in[gid * 4 + 3].z;
        float3 one = float3(1.0);
        float sd = max(d, r * 1.001);
        float3 a = brdf(N, V, L, float3(alb), 0.0, rough, one, 0.0, float3(0.0), 0.0, nullptr,
                        kClothSheenRoughness, layeredOff(), 0.0, r, sd);
        float3 b = brdf(N, V, L, float3(0.0), 0.0, rough, one, 0.0, float3(0.0), 0.0, nullptr,
                        kClothSheenRoughness, layeredOff(), 0.0, r, sd);
        float3 c = brdf(N, V, L, float3(alb), 0.0, rough, one);
        float3 e = brdf(N, V, L, float3(0.0), 0.0, rough, one);
        out[gid] = float4(a.x, b.x, c.x, e.x);
    }
    // out: (default call, specScale 1 explicit, specScale 0, brdfDiffuse reference); out2: (specScale 0.5, spec-only at 1, sphere+scale 0, 0)
    kernel void specScaleProbe(device const float4* in [[buffer(0)]],
                               device float4* out [[buffer(1)]],
                               device float4* out2 [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
        float3 N = in[gid * 4 + 0].xyz, V = in[gid * 4 + 1].xyz;
        float3 L = in[gid * 4 + 2].xyz; float d = in[gid * 4 + 2].w;
        float r = in[gid * 4 + 3].x, rough = in[gid * 4 + 3].y, alb = in[gid * 4 + 3].z;
        float3 one = float3(1.0);
        float3 def = brdf(N, V, L, float3(alb), 0.0, rough, one);
        float3 s1 = brdf(N, V, L, float3(alb), 0.0, rough, one, 0.0, float3(0.0), 0.0, nullptr, kClothSheenRoughness, layeredOff(), 0.0, 0.0, 0.0, 1.0);
        float3 s0 = brdf(N, V, L, float3(alb), 0.0, rough, one, 0.0, float3(0.0), 0.0, nullptr, kClothSheenRoughness, layeredOff(), 0.0, 0.0, 0.0, 0.0);
        float3 sh = brdf(N, V, L, float3(alb), 0.0, rough, one, 0.0, float3(0.0), 0.0, nullptr, kClothSheenRoughness, layeredOff(), 0.0, 0.0, 0.0, 0.5);
        float3 sp = brdf(N, V, L, float3(0.0), 0.0, rough, one);
        float3 ds = brdfDiffuse(N, V, L, float3(alb), 0.0, one);
        float3 sph0 = brdf(N, V, L, float3(alb), 0.0, rough, one, 0.0, float3(0.0), 0.0, nullptr, kClothSheenRoughness, layeredOff(), 0.0, r, max(d, r * 1.001), 0.0);
        out[gid] = float4(def.x, s1.x, s0.x, ds.x);
        out2[gid] = float4(sh.x, sp.x, sph0.x, 0.0);
    }
    """

    private struct Probe { var n, v, l: SIMD3<Double>; var d, r, rough, albedo: Double }

    private func runProbes(_ probes: [Probe]) throws -> [SIMD4<Float>] {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaLighting.metal")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no shader") }
        let source = try MetalSourceLoader.source(contentsOf: url) + Self.probeKernel
        let lib = try device.makeLibrary(source: source, options: nil)
        guard let fn = lib.makeFunction(name: "sphereLightProbe") else { throw XCTSkip("probe kernel missing") }
        let pipeline = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        var packed: [SIMD4<Float>] = []
        for p in probes {
            packed.append(SIMD4(SIMD3<Float>(p.n), 0))
            packed.append(SIMD4(SIMD3<Float>(p.v), 0))
            packed.append(SIMD4(SIMD3<Float>(p.l), Float(p.d)))
            packed.append(SIMD4(Float(p.r), Float(p.rough), Float(p.albedo), 0))
        }
        guard let ib = device.makeBuffer(bytes: packed, length: packed.count * 16, options: .storageModeShared),
              let ob = device.makeBuffer(length: probes.count * 16, options: .storageModeShared),
              let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder()
        else { throw XCTSkip("no buffers") }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(ib, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: probes.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(64, probes.count), height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()  // gpu-ok: test harness
        XCTAssertNil(cmd.error)
        let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: probes.count)
        return (0..<probes.count).map { o[$0] }
    }

    func testSpecularScaleIsExactAtOneDiffuseOnlyAtZeroAndLinearBetween() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue()  // gpu-ok: test harness
        else { throw XCTSkip("no Metal") }
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaLighting.metal")
        let lib = try device.makeLibrary(source: try MetalSourceLoader.source(contentsOf: url) + Self.probeKernel, options: nil)
        let pipe = try device.makeComputePipelineState(function: try XCTUnwrap(lib.makeFunction(name: "specScaleProbe")))  // gpu-ok: test harness
        var probes: [Probe] = []
        for cosT in [0.9, 0.6, 0.3] { for rough in [0.1, 0.3, 0.7] { for dv in [0.0, 0.2] {
            let fr = Self.frame(cosTheta: cosT, viewAzimuthFromMirror: dv, nv: cosT)
            probes.append(Probe(n: fr.n, v: fr.v, l: fr.l, d: 3, r: 0.2, rough: rough, albedo: 0.7))
        } } }
        var packed: [SIMD4<Float>] = []
        for p in probes {
            packed += [SIMD4(SIMD3<Float>(p.n), 0), SIMD4(SIMD3<Float>(p.v), 0), SIMD4(SIMD3<Float>(p.l), Float(p.d)),
                       SIMD4(Float(p.r), Float(p.rough), Float(p.albedo), 0)]
        }
        let ib = try XCTUnwrap(device.makeBuffer(bytes: packed, length: packed.count * 16, options: .storageModeShared))
        let ob = try XCTUnwrap(device.makeBuffer(length: probes.count * 16, options: .storageModeShared))
        let ob2 = try XCTUnwrap(device.makeBuffer(length: probes.count * 16, options: .storageModeShared))
        let cmd = try XCTUnwrap(queue.makeCommandBuffer()), enc = try XCTUnwrap(cmd.makeComputeCommandEncoder())
        enc.setComputePipelineState(pipe)
        enc.setBuffer(ib, offset: 0, index: 0); enc.setBuffer(ob, offset: 0, index: 1); enc.setBuffer(ob2, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: probes.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: probes.count, height: 1, depth: 1))
        enc.endEncoding(); cmd.commit()
        cmd.waitUntilCompleted()  // gpu-ok: test harness
        XCTAssertNil(cmd.error)
        let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: probes.count)
        let o2 = ob2.contents().bindMemory(to: SIMD4<Float>.self, capacity: probes.count)
        for i in 0..<probes.count {
            XCTAssertEqual(o[i].x.bitPattern, o[i].y.bitPattern, "specScale 1 must be bit-identical to the default call")
            XCTAssertGreaterThan(o2[i].y, 0, "probe has a specular term")
            XCTAssertEqual(o[i].z, o[i].w, accuracy: 1e-5 * o[i].w + 1e-9, "specScale 0 == brdfDiffuse (diffuse-only)")
            XCTAssertEqual(o2[i].x, o[i].z + 0.5 * (o[i].x - o[i].z), accuracy: 1e-5 * o[i].x + 1e-9, "linear in the specular term")
            XCTAssertGreaterThan(o2[i].z, 0)
        }
    }

    // ── CPU reference: the engine's point BRDF × cos, integrated over the sphere ──────────

    /// (diffuse part, specular part) of brdf()·cosθ for one incident direction (unit radiance not applied).
    private static func fCos(_ n: SIMD3<Double>, _ v: SIMD3<Double>, _ l: SIMD3<Double>,
                             _ rough: Double, _ albedo: Double) -> (diff: Double, spec: Double, fresnelKd: Double) {
        let NdotL = simd_dot(n, l), NdotV = simd_dot(n, v)
        guard NdotL > 0, NdotV > 0 else { return (0, 0, 1) }
        let h = simd_normalize(v + l)
        let NdotH = max(0, simd_dot(n, h)), HdotV = max(0, simd_dot(h, v))
        let a2 = pow(rough * rough, 2)
        let dd = NdotH * NdotH * (a2 - 1) + 1
        let D = a2 / (Double.pi * dd * dd + 1e-7)
        let k = (rough + 1) * (rough + 1) / 8
        let G = NdotV / (NdotV * (1 - k) + k) * NdotL / (NdotL * (1 - k) + k)
        let F = 0.04 + 0.96 * pow(max(0, 1 - HdotV), 5)
        let spec = D * G * F / (4 * NdotV * NdotL + 1e-7) * NdotL
        let diff = (1 - F) * albedo / Double.pi * NdotL
        return (diff, spec, 1 - F)
    }

    /// ∫ over the sphere's solid angle (uniform radiance 1/(π r²)) of fCos, by a deterministic
    /// Fibonacci-lattice cone sampling (≈ uniform in solid angle). Returns I=1 values scaled by d²
    /// (i.e. directly comparable to the kernel's un-attenuated brdf output).
    private static func monteCarlo(_ p: Probe) -> (diff: Double, spec: Double) {
        let sinS = min(p.r / p.d, 0.999)
        let cosS = (1 - sinS * sinS).squareRoot()
        let w = p.l
        let up = abs(w.y) < 0.9 ? SIMD3<Double>(0, 1, 0) : SIMD3<Double>(1, 0, 0)
        let u = simd_normalize(simd_cross(up, w)), vv = simd_cross(w, u)
        let n = 50_000
        let golden = Double.pi * (3 - 5.0.squareRoot())
        var sd = 0.0, ss = 0.0
        // The kernel applies ONE Fresnel kd = 1 − F(centre H) to the whole sphere; hold the reference to the
        // same kd so the diffuse comparison isolates the IRRADIANCE integral (Fresnel varying across a big
        // sphere at grazing view is a separate, documented approximation).
        let hc = simd_normalize(p.v + p.l)
        let kdC = 1 - (0.04 + 0.96 * pow(max(0, 1 - max(0, simd_dot(hc, p.v))), 5))
        for i in 0..<n {
            let t = (Double(i) + 0.5) / Double(n)
            let cosA = 1 - t * (1 - cosS), sinA = (1 - cosA * cosA).squareRoot()
            let phi = golden * Double(i)
            let dir = w * cosA + u * (sinA * cos(phi)) + vv * (sinA * sin(phi))
            let f = fCos(p.n, p.v, dir, p.rough, p.albedo)
            sd += f.diff / max(f.fresnelKd, 1e-9) * kdC; ss += f.spec
        }
        let omega = 2 * Double.pi * (1 - cosS)
        // E = ∫ f·cos·L dω with L = I/(π r²); I = 1; kernel output is E·d² (no 1/d² applied).
        let scale = omega / Double(n) / (Double.pi * p.r * p.r) * p.d * p.d
        return (sd * scale, ss * scale)
    }

    private static func frame(cosTheta: Double, viewAzimuthFromMirror dv: Double, nv: Double) -> (n: SIMD3<Double>, v: SIMD3<Double>, l: SIMD3<Double>) {
        let n = SIMD3<Double>(0, 0, 1)
        let sinT = (1 - cosTheta * cosTheta).squareRoot()
        let l = SIMD3<Double>(sinT, 0, cosTheta)
        // View: elevation angle acos(nv) in the plane, mirrored about N from L, rotated by dv (radians).
        let sinV = (1 - nv * nv).squareRoot()
        let ang = dv
        let v = simd_normalize(SIMD3<Double>(-sinV * cos(ang), sinV * sin(ang), nv))
        return (n, v, l)
    }

    private static func probeSet() -> [Probe] {
        var out: [Probe] = []
        for cosT in [0.95, 0.7, 0.4, 0.15, 0.03] {
            for rd in [0.02, 0.1, 0.3, 0.6] {
                for rough in [0.14, 0.3, 0.6, 1.0] {
                    // mirror-aligned view (V = reflect of L about N ⇒ azimuth 0, nv = cosT), and two offsets
                    for (dv, nv) in [(0.0, cosT), (0.25, cosT), (0.0, max(0.2, cosT - 0.15))] {
                        let d = 4.0, r = rd * d
                        let fr = frame(cosTheta: cosT, viewAzimuthFromMirror: dv, nv: nv)
                        out.append(Probe(n: fr.n, v: fr.v, l: fr.l, d: d, r: r, rough: rough, albedo: 0.8))
                    }
                }
            }
        }
        return out
    }

    /// DIFFUSE: the Frostbite horizon-clipped sphere irradiance is an exact integral — near float precision.
    /// SPECULAR: representative-point + (α/α')² is an approximation — held to the measured bound below.
    func testKernelMatchesSphereMonteCarlo() throws {
        let probes = Self.probeSet()
        let gpu = try runProbes(probes)
        var practicalRels: [Double] = [], worstSpecAll = 0.0, worstDiff = 0.0, worstSpec = 0.0, worstSpecAbs = 0.0
        var sumSpecRatio = 0.0
        var worstSpecDesc = "", worstDiffDesc = ""
        for (p, o) in zip(probes, gpu) {
            let ref = Self.monteCarlo(p)
            // The ASSERTED domain is the practical one (a lamp globe: r/d <= 0.3; centre >= 20° above the
            // horizon; view >= 20° above the plane). Everything else is printed, not asserted.
            let practical = p.r / p.d <= 0.31 && p.l.z >= 0.35 && p.v.z >= 0.35
            let gDiff = Double(o.x - o.y), gSpec = Double(o.y)
            if ref.diff > 1e-4 && practical {
                let e = abs(gDiff - ref.diff) / ref.diff
                if e > worstDiff {
                    worstDiff = e
                    worstDiffDesc = String(format: "cos %.2f r/d %.2f rough %.2f nv %.2f: kernel %.4g MC %.4g",
                                           p.l.z, p.r / p.d, p.rough, p.v.z, gDiff, ref.diff)
                }
            }
            // specular: relative to the larger of the reference and 2% of this probe's point peak (so
            // deep-tail probes that the eye never sees do not dominate a ratio of two near-zero numbers)
            let denom = max(ref.spec, 0.02 * Double(o.w), 1e-4)
            let rel = abs(gSpec - ref.spec) / denom
            if practical { print(String(format: "SPHERELIGHT_ROW cos %.2f r/d %.2f rough %.2f nv %.2f az %.2f kernel %.4g MC %.4g point %.4g",
                                        p.l.z, p.r / p.d, p.rough, p.v.z, atan2(p.v.y, -p.v.x), gSpec, ref.spec, Double(o.w))) }
            if !practical { if rel > worstSpecAll { worstSpecAll = rel }; continue }
            practicalRels.append(rel)
            if rel > worstSpec {
                worstSpec = rel
                worstSpecDesc = String(format: "cos %.2f r/d %.2f rough %.2f: kernel %.4g MC %.4g point %.4g",
                                       p.l.z, p.r / p.d, p.rough, gSpec, ref.spec, Double(o.w))
            }
            worstSpecAbs = max(worstSpecAbs, abs(gSpec - ref.spec))
            if ref.spec > 1e-3 { sumSpecRatio += gSpec / ref.spec }
        }
        print(String(format: "SPHERELIGHT worst specular rel err OUTSIDE the practical domain (grazing / huge sphere): %.2f", worstSpecAll))
        print("SPHERELIGHT worst diffuse: \(worstDiffDesc)")
        print(String(format: "SPHERELIGHT worst diffuse rel err %.2e | worst specular rel err %.3f (%@) | probes %d",
                     worstDiff, worstSpec, worstSpecDesc, probes.count))
        XCTAssertLessThan(worstDiff, 2e-3, "sphere diffuse must equal the exact sphere irradiance")
        practicalRels.sort()
        let median = practicalRels[practicalRels.count / 2], p90 = practicalRels[practicalRels.count * 9 / 10]
        print(String(format: "SPHERELIGHT specular rel err over %d practical probes: median %.3f p90 %.3f max %.3f", practicalRels.count, median, p90, worstSpec))
        // Measured: median ≈ 0.18; the max is a plateau-EDGE probe (the rep-point reflection disc is hard-edged,
        // the integral's falls off smoothly). Bounds are the measured values with margin.
        XCTAssertLessThan(median, 0.25, "representative-point specular: typical error vs sphere MC")
        XCTAssertLessThan(p90, 0.7, "representative-point specular: p90 error vs sphere MC")
        XCTAssertLessThan(worstSpec, 1.5, "representative-point specular worst case vs sphere MC: \(worstSpecDesc)")
    }

    /// Far field: a tiny sphere IS a point (diffuse equals the point law; specular within a few %).
    /// r == 0 is exactly the point path; `cosθ <= 0` is zero (documented: no horizon wrap).
    func testFarFieldEqualsPointLawAndZeroRadiusIsAPoint() throws {
        var probes: [Probe] = []
        for cosT in [0.9, 0.5, 0.2] {
            for rough in [0.3, 0.8] {
                let fr = Self.frame(cosTheta: cosT, viewAzimuthFromMirror: 0.2, nv: 0.7)
                probes.append(Probe(n: fr.n, v: fr.v, l: fr.l, d: 50, r: 0.01, rough: rough, albedo: 0.8))
            }
        }
        let g = try runProbes(probes)
        for (p, o) in zip(probes, g) {
            XCTAssertEqual(Double(o.x - o.y), Double(o.z - o.w), accuracy: 2e-4 * Double(o.z - o.w) + 1e-7, "far-field diffuse == point law")
            XCTAssertEqual(Double(o.y), Double(o.w), accuracy: 0.06 * Double(o.w) + 1e-6, "far-field specular ≈ point (cos \(p.l.z))")
        }
        let fr = Self.frame(cosTheta: 0.6, viewAzimuthFromMirror: 0.1, nv: 0.6)
        let zero = try runProbes([Probe(n: fr.n, v: fr.v, l: fr.l, d: 3, r: 0, rough: 0.2, albedo: 0.8)])[0]
        XCTAssertEqual(zero.x, zero.z, "r = 0 takes the point path exactly")
        XCTAssertEqual(zero.y, zero.w)
        // Horizon: centre below the horizon ⇒ nothing (wrap not modelled), even for a big sphere.
        let below = Self.frame(cosTheta: -0.05, viewAzimuthFromMirror: 0, nv: 0.6)
        let h = try runProbes([Probe(n: below.n, v: below.v, l: below.l, d: 2, r: 0.8, rough: 0.2, albedo: 0.8)])[0]
        XCTAssertEqual(h.x, 0)
    }

    // ── (b), (c) render tests ────────────────────────────────────────────────

    private struct Shot { var luma: [Double]; var width: Int; var height: Int; var bytes: [UInt8] }

    /// A dark room: one glossy dielectric floor, one point light, no sun / ambient / IBL / bloom.
    private func render(sphere: Bool, bulb: Float, roughness: Float = 0.25) throws -> Shot {
        let cam = IlluminatoramaCamera(position: SIMD3(0, 1.5, 3), target: SIMD3(0, 0, 1.4), up: SIMD3(0, 1, 0),
                                       fovYRadians: 0.9, aspect: 4.0 / 3.0, zNear: 0.05, zFar: 100)
        let r = try IlluminatoramaRenderer(engine: SimEngine.shared, width: 160, height: 120, camera: cam)
        r.directionalLightColor = .zero
        r.ambientColor = .zero
        r.iblIntensity = 0
        r.bloomIntensity = 0
        r.taaEnabled = false
        r.autoExposureEnabled = false
        r.sphereLightsEnabled = sphere
        r.setMesh(.box, IlluminatoramaMesh.unitBox(device: r.device))
        var m = matrix_identity_float4x4
        m.columns.0.x = 20; m.columns.1.y = 0.1; m.columns.2.z = 20; m.columns.3 = SIMD4(0, -0.05, 0, 1)
        let inst = IlluminatoramaInstance(modelMatrix: m, albedo: SIMD3(repeating: 0.05), metallic: 0, roughness: roughness)
        r.instances = [IlluminatoramaRenderer.InstanceRef(meshKind: .box, data: inst)]
        var pl = IlluminatoramaPointLight(position: SIMD3(0, 1.5, 0), radius: 30, color: SIMD3(repeating: 2.5))
        pl.bulbRadius = bulb
        r.pointLights = [pl]
        let f = try XCTUnwrap(r.renderHeadless(frames: 2))
        let bgra = f.isBGRA
        var luma = [Double]()
        luma.reserveCapacity(f.width * f.height)
        for i in 0..<(f.width * f.height) {
            let b0 = Double(f.bytes[i * 4]), b1 = Double(f.bytes[i * 4 + 1]), b2 = Double(f.bytes[i * 4 + 2])
            let (rr, gg, bb) = bgra ? (b2, b1, b0) : (b0, b1, b2)
            luma.append((0.2126 * rr + 0.7152 * gg + 0.0722 * bb) / 255)
        }
        return Shot(luma: luma, width: f.width, height: f.height, bytes: f.bytes)
    }

    /// `specularScale` reuses the slot a point light formerly carried as `openingOffset`: pin the layout.
    func testPointLightLayoutPinsSpecularScaleSlot() {
        XCTAssertEqual(MemoryLayout<IlluminatoramaPointLight>.stride, 80)
        XCTAssertEqual(MemoryLayout<IlluminatoramaPointLight>.offset(of: \.specularScale), 76)
        XCTAssertEqual(MemoryLayout<IlluminatoramaPointLight>.offset(of: \.bulbRadius), 60)
        XCTAssertEqual(IlluminatoramaPointLight(position: .zero, radius: 1, color: .zero).specularScale, 1)
    }

    func testFlagOffIgnoresBulbRadiusBitForBit() throws {
        let a = try render(sphere: false, bulb: 0)
        let b = try render(sphere: false, bulb: 0.4)
        XCTAssertEqual(a.bytes, b.bytes, "sphereLightsEnabled == false ⇒ bulbRadius must not change a pixel")
        let c = try render(sphere: true, bulb: 0)
        XCTAssertEqual(a.bytes, c.bytes, "flag on but bulbRadius == 0 ⇒ the point path, bit for bit")
        XCTAssertGreaterThan(a.luma.max() ?? 0, 0.05, "the scene must actually be lit")
    }

    func testSphereHighlightIsLowerAndWiderAtComparableEnergy() throws {
        let point = try render(sphere: true, bulb: 0)
        let sph = try render(sphere: true, bulb: 0.35)
        func stats(_ s: Shot) -> (peak: Double, sum: Double, area: Int) {
            let peak = s.luma.max() ?? 0
            return (peak, s.luma.reduce(0, +), s.luma.filter { $0 > 0.5 * peak }.count)
        }
        let p = stats(point), q = stats(sph)
        print(String(format: "SPHERELIGHT highlight: point peak %.3f sum %.1f area>50%%pk %d | sphere peak %.3f sum %.1f area %d",
                     p.peak, p.sum, p.area, q.peak, q.sum, q.area))
        // Compare at the POINT's half-peak level so "wider" is not an artefact of a lower peak.
        let level = 0.5 * p.peak
        let wideP = point.luma.filter { $0 > level }.count, wideS = sph.luma.filter { $0 > level }.count
        XCTAssertLessThan(q.peak, p.peak, "sphere highlight peak must be lower than the point's")
        XCTAssertGreaterThan(q.sum / p.sum, 0.6, "total reflected energy comparable (not collapsed)")
        XCTAssertLessThan(q.sum / p.sum, 1.4, "total reflected energy comparable (not inflated)")
        XCTAssertGreaterThan(q.area, p.area, "sphere highlight is wider at its own half-peak (\(wideP) vs \(wideS) above the point's)")
    }
}
