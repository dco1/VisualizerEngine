import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// VZ-0141 — the rectangular area light's MOST-REPRESENTATIVE-POINT specular (the
/// `areaLTCOverride = false` lane Digital Clock, Vintage Diner Ultra and Daydream Home's
/// window portals ship on) must be ENERGY-BOUNDED and agree with the actual integral of the
/// engine's GGX over the light.
///
/// The shipped form evaluated the punctual GGX at the representative point and multiplied it
/// by the light's form factor, so a light that covers the whole lobe returned the lobe's PEAK
/// times the light's solid angle instead of the lobe's energy. On Digital Clock's glossy bezel
/// lip (roughness 0.08) under its 1.0 × 1.2 m window portal that was 3.55 × colour against an
/// integral of 0.022 × colour — 158 × too bright, 22 × the ceiling any passive reflector obeys —
/// in a band ≈ 1 px wide, which aliased into a dashed white line and bloomed into a dot grid.
///
/// Runs the REAL `evalAreaLight` (IlluminatoramaLighting.metal, compiled from source with a
/// probe kernel appended) against a CPU reference: the engine's own point-light BRDF (GGX D,
/// Schlick-GGX G with k = (r+1)²/8, Schlick F) integrated over the rectangle by MIS of
/// light-area and GGX-lobe sampling — "an area light is a sum of tiny point lights".
///
/// UNITS. The engine's area light carries its own radiometric scale: `ltcPolygonForm` (and the
/// LTC branch) return the clamped-cosine integral × `s`. The test MEASURES `s` from the kernel's
/// own diffuse form factor on a small head-on light rather than hard-coding it, and holds the
/// specular to the same `s` — so the invariant is "specular and diffuse ride one scale", and a
/// future fix of that scale (VZ-0143; see `kAreaLightFormScale`) that forgets the MRP goes red.
@MainActor
final class IlluminatoramaAreaLightSpecularTests: XCTestCase {

    // ── GPU probe ────────────────────────────────────────────────────────────

    private struct Probe {
        let pipeline: MTLComputePipelineState
        let queue: MTLCommandQueue
        let ltcMat: MTLTexture
        let ltcMag: MTLTexture
        let device: MTLDevice
    }

    /// One receiver: world position, unit normal, unit view vector, perceptual roughness.
    private struct Receiver {
        var p: SIMD3<Double>
        var n: SIMD3<Double>
        var v: SIMD3<Double>
        var roughness: Double
    }

    /// Per receiver: (MRP specular, LTC specular, diffuse form factor, rect solid angle).
    private struct Sample { var mrp: Double; var ltc: Double; var ff: Double; var omega: Double }

    private static let probeKernel = """

    // ── VZ-0141 test probe (appended by IlluminatoramaAreaLightSpecularTests) ──
    // The probe's own solid-angle helper (Van Oosterom–Strackee), independent of the kernel's.
    static inline float vz0141ProbeSolidAngle(float3 p0, float3 p1, float3 p2, float3 p3) {
        float3 a = normalize(p0), b = normalize(p1), c = normalize(p2), d = normalize(p3);
        float t0 = atan2(abs(dot(a, cross(b, c))), 1.0 + dot(a, b) + dot(b, c) + dot(c, a));
        float t1 = atan2(abs(dot(a, cross(c, d))), 1.0 + dot(a, c) + dot(c, d) + dot(d, a));
        return 2.0 * (t0 + t1);
    }
    kernel void vz0141AreaSpecProbe(device const AreaLight* light [[buffer(0)]],
                                    device const float4*   recv  [[buffer(1)]],
                                    device float4*          out   [[buffer(2)]],
                                    texture2d<float> ltcMat [[texture(0)]],
                                    texture2d<float> ltcMag [[texture(1)]],
                                    uint gid [[thread_position_in_grid]]) {
        AreaLight al = light[0];
        float4 a = recv[gid * 3 + 0], b = recv[gid * 3 + 1], c = recv[gid * 3 + 2];
        float3 P = a.xyz, N = b.xyz, V = c.xyz;
        float  r = a.w;
        // albedo 0, metallic 0 ⇒ diffuse = 0 and F0 = 0.04: the return is the specular alone
        // (× colour 1 × the radius window, ≈ 1 at the test's 10 km radius).
        float3 mrp = evalAreaLight(al, P, N, V, float3(0.0), 0.0, r, ltcMat, ltcMag, false);
        float3 ltc = evalAreaLight(al, P, N, V, float3(0.0), 0.0, r, ltcMat, ltcMag, true);
        float3 p0 = al.center - al.ex - al.ey - P, p1 = al.center - al.ex + al.ey - P;
        float3 p2 = al.center + al.ex + al.ey - P, p3 = al.center + al.ex - al.ey - P;
        float  ff = ltcPolygonForm(N, p0, p1, p2, p3, al.twoSided > 0.5);
        out[gid] = float4(mrp.x, ltc.x, ff, vz0141ProbeSolidAngle(p0, p1, p2, p3));
    }
    """

    private func makeProbe() throws -> Probe {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaLighting.metal")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no shader") }
        let source = try MetalSourceLoader.source(contentsOf: url) + Self.probeKernel
        let lib = try device.makeLibrary(source: source, options: nil)
        guard let fn = lib.makeFunction(name: "vz0141AreaSpecProbe") else {
            throw XCTSkip("probe kernel missing")
        }
        let pipeline = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        guard let mat = Self.lutTexture(device, IlluminatoramaLTCTable.baked.mat),
              let mag = Self.lutTexture(device, IlluminatoramaLTCTable.baked.mag)
        else { throw XCTSkip("no LTC LUT textures") }
        return Probe(pipeline: pipeline, queue: queue, ltcMat: mat, ltcMag: mag, device: device)
    }

    /// The baked LTC table as an `rgba16Float` texture, converted to HALF floats on upload.
    /// Built here rather than taken from `IlluminatoramaLTC.makeLUTs`, which (VZ-0144, open as of
    /// this test) hands `replace(region:)` the table's 32-bit floats for a 16-bit texture — the GPU reads
    /// each float's two halves as two texels' worth of components (measured: 13/1024 matrix and
    /// 7/1024 magnitude components non-finite, the rest off by up to 6.5e4). The MRP-vs-LTC
    /// comparison needs the LTC lane the table describes, not that upload.
    private static func lutTexture(_ device: MTLDevice, _ texels: [SIMD4<Float>]) -> MTLTexture? {
        let n = IlluminatoramaLTCTable.baked.size
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: n, height: n,
                                                            mipmapped: false)
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        guard texels.count == n * n, let tex = device.makeTexture(descriptor: desc) else { return nil }
        let half = texels.map { SIMD4<Float16>($0) }
        half.withUnsafeBytes {
            tex.replace(region: MTLRegionMake2D(0, 0, n, n), mipmapLevel: 0,
                        withBytes: $0.baseAddress!, bytesPerRow: n * MemoryLayout<SIMD4<Float16>>.stride)
        }
        return tex
    }

    private func run(_ probe: Probe, light: IlluminatoramaAreaLight, _ recv: [Receiver]) throws -> [Sample] {
        var lights = [light]
        var packed: [SIMD4<Float>] = []
        for r in recv {
            packed.append(SIMD4(SIMD3<Float>(r.p), Float(r.roughness)))
            packed.append(SIMD4(SIMD3<Float>(r.n), 0))
            packed.append(SIMD4(SIMD3<Float>(r.v), 0))
        }
        let dev = probe.device
        guard let lb = dev.makeBuffer(bytes: &lights, length: MemoryLayout<IlluminatoramaAreaLight>.stride,
                                      options: .storageModeShared),
              let rb = dev.makeBuffer(bytes: packed, length: packed.count * 16, options: .storageModeShared),
              let ob = dev.makeBuffer(length: recv.count * 16, options: .storageModeShared),
              let cmd = probe.queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder()
        else { throw XCTSkip("no buffers") }
        enc.setComputePipelineState(probe.pipeline)
        enc.setBuffer(lb, offset: 0, index: 0)
        enc.setBuffer(rb, offset: 0, index: 1)
        enc.setBuffer(ob, offset: 0, index: 2)
        enc.setTexture(probe.ltcMat, index: 0)
        enc.setTexture(probe.ltcMag, index: 1)
        enc.dispatchThreads(MTLSize(width: recv.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(64, recv.count), height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()  // gpu-ok: test harness
        let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: recv.count)
        return (0..<recv.count).map { Sample(mrp: Double(o[$0].x), ltc: Double(o[$0].y),
                                            ff: Double(o[$0].z), omega: Double(o[$0].w)) }
    }

    // ── CPU reference ────────────────────────────────────────────────────────

    private struct Rect { var c: SIMD3<Double>; var ex: SIMD3<Double>; var ey: SIMD3<Double> }

    /// The engine's point-light specular BRDF × cos (brdf() in the lighting kernel), with the
    /// NDF properly normalised (the shader's `+1e-7` peak guard is a separate matter — the MRP
    /// normalises against the energy of a unit lobe, so the plateau target is the true one).
    private static func fCos(_ n: SIMD3<Double>, _ v: SIMD3<Double>, _ l: SIMD3<Double>, _ r: Double) -> Double {
        let NdotL = simd_dot(n, l), NdotV = simd_dot(n, v)
        guard NdotL > 0, NdotV > 0 else { return 0 }
        let h = simd_normalize(v + l)
        let NdotH = max(0, simd_dot(n, h)), HdotV = max(0, simd_dot(h, v))
        let a2 = pow(r * r, 2)
        let d = NdotH * NdotH * (a2 - 1) + 1
        let D = a2 / (Double.pi * d * d)
        let k = (r + 1) * (r + 1) / 8
        let G = NdotV / (NdotV * (1 - k) + k) * NdotL / (NdotL * (1 - k) + k)
        let F = 0.04 + 0.96 * pow(max(0, 1 - HdotV), 5)
        return D * G * F / (4 * NdotV * NdotL) * NdotL
    }

    private static func lobePdf(_ n: SIMD3<Double>, _ v: SIMD3<Double>, _ l: SIMD3<Double>, _ r: Double) -> Double {
        let h = simd_normalize(v + l)
        let NdotH = max(0, simd_dot(n, h)), HdotV = max(1e-9, simd_dot(h, v))
        let a2 = pow(r * r, 2)
        let d = NdotH * NdotH * (a2 - 1) + 1
        return a2 / (Double.pi * d * d) * NdotH / (4 * HdotV)
    }

    /// ∫_rect f·cos dω by MIS (balance) of stratified light-area and GGX-D sampling.
    private static func reference(_ rect: Rect, _ rc: Receiver, strata m: Int = 96) -> Double {
        let nL = simd_normalize(simd_cross(rect.ex, rect.ey))
        let ux = simd_length(rect.ex), uy = simd_length(rect.ey)
        let area = 4 * ux * uy
        guard simd_dot(nL, rc.p - rect.c) > 0 else { return 0 }
        func lightPdf(_ dist2: Double, _ l: SIMD3<Double>) -> Double {
            dist2 / (area * max(1e-12, abs(simd_dot(l, nL))))
        }
        var sum = 0.0
        let inv = 1.0 / Double(m)
        // Light-area sampling.
        for i in 0..<m {
            for j in 0..<m {
                let u = (Double(i) + 0.5) * inv * 2 - 1, w = (Double(j) + 0.5) * inv * 2 - 1
                let x = rect.c + rect.ex * u + rect.ey * w
                let d = x - rc.p
                let dist2 = simd_length_squared(d)
                let l = d / dist2.squareRoot()
                let f = fCos(rc.n, rc.v, l, rc.roughness)
                if f > 0 { sum += f / (lightPdf(dist2, l) + lobePdf(rc.n, rc.v, l, rc.roughness)) }
            }
        }
        // GGX-D sampling of the half vector.
        let a2 = pow(rc.roughness * rc.roughness, 2)
        let t = simd_normalize(simd_cross(rc.n, abs(rc.n.y) < 0.99 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)))
        let b = simd_cross(rc.n, t)
        for i in 0..<m {
            for j in 0..<m {
                let u1 = (Double(i) + 0.5) * inv, u2 = (Double(j) + 0.5) * inv
                let cos2 = (1 - u1) / (1 + (a2 - 1) * u1)
                let ct = cos2.squareRoot(), st = max(0, 1 - cos2).squareRoot(), ph = 2 * Double.pi * u2
                let h = t * (st * cos(ph)) + b * (st * sin(ph)) + rc.n * ct
                let l = -rc.v + 2 * simd_dot(h, rc.v) * h
                let den = simd_dot(l, nL)
                guard abs(den) > 1e-12 else { continue }
                let tt = simd_dot(rect.c - rc.p, nL) / den
                guard tt > 0 else { continue }
                let q = rc.p + l * tt - rect.c
                guard abs(simd_dot(q, rect.ex / ux)) <= ux, abs(simd_dot(q, rect.ey / uy)) <= uy else { continue }
                let f = fCos(rc.n, rc.v, l, rc.roughness)
                if f > 0 { sum += f / (lightPdf(tt * tt, l) + lobePdf(rc.n, rc.v, l, rc.roughness)) }
            }
        }
        return sum * inv * inv
    }

    /// Exact solid angle of a small rect seen head-on at distance `z` (for the scale probe).
    private static func headOnSolidAngle(halfX a: Double, halfY b: Double, z: Double) -> Double {
        4 * atan(a * b / (z * (a * a + b * b + z * z).squareRoot()))
    }

    private func light(_ r: Rect) -> IlluminatoramaAreaLight {
        IlluminatoramaAreaLight(center: SIMD3<Float>(r.c), ex: SIMD3<Float>(r.ex), ey: SIMD3<Float>(r.ey),
                                color: SIMD3<Float>(1, 1, 1), radius: 10_000)
    }

    /// The engine's area-light radiometric scale, measured: `ltcPolygonForm` of a small light
    /// seen head-on over its exact projected solid angle / π. (A physically unit-scaled form
    /// factor gives 1.)
    private func measuredScale(_ probe: Probe) throws -> Double {
        let rect = Rect(c: .zero, ex: SIMD3(0.02, 0, 0), ey: SIMD3(0, 0.02, 0))
        let rc = Receiver(p: SIMD3(0, 0, 1), n: SIMD3(0, 0, -1), v: SIMD3(0, 0, -1), roughness: 0.5)
        let s = try run(probe, light: light(rect), [rc])[0]
        let omega = Self.headOnSolidAngle(halfX: 0.02, halfY: 0.02, z: 1)
        XCTAssertEqual(s.omega, omega, accuracy: omega * 1e-3, "the probe's solid angle is off for a head-on rect")
        return s.ff * Double.pi / omega    // projected ≈ Ω here (cos ≈ 1 across a 2.3° light)
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    /// The VZ-0141 configuration itself: Digital Clock's hero camera, the top outer bezel
    /// round-over (roughness 0.08) and the 1.0 × 1.2 m daylight portal. Sweep the round-over's
    /// normal from front (+Z) to up (+Y) through the window's reflection band.
    func testDigitalClockBezelUnderTheWindowPortal() throws {
        let probe = try makeProbe()
        let s = try measuredScale(probe)
        let rect = Rect(c: SIMD3(-0.10, 1.40, 0.02), ex: SIMD3(0.5, 0, 0), ey: SIMD3(0, 0.6, 0))
        let p = SIMD3<Double>(-0.06, 0.7215, 0.1858)
        let yaw = 30.0 * Double.pi / 180, pitch = -3.0 * Double.pi / 180
        let pivot = SIMD3<Double>(-0.06, 0.7115, 0.185)
        let cam = pivot + 0.62 * SIMD3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch))
        let v = simd_normalize(cam - p)
        var recv: [Receiver] = []
        for deg in stride(from: 0.0, through: 85.0, by: 1.0) {
            let ph = deg * Double.pi / 180
            let n = SIMD3<Double>(0, sin(ph), cos(ph))
            if simd_dot(n, v) > 0.01 { recv.append(Receiver(p: p, n: n, v: v, roughness: 0.08)) }
        }
        let got = try run(probe, light: light(rect), recv)
        var worstBand = 0.0, peak = 0.0, peakRef = 0.0
        for (rc, g) in zip(recv, got) {
            let ref = Self.reference(rect, rc) * s
            peak = max(peak, g.mrp); peakRef = max(peakRef, ref)
            XCTAssertTrue(g.mrp.isFinite && g.mrp >= 0, "non-finite / negative MRP specular")
            // Inside the reflection band (the light fully covers the lobe) the MRP must be the
            // integral, not the lobe peak × the light's solid angle.
            let phiDeg = asin(rc.n.y) * 180 / Double.pi
            if phiDeg >= 50, phiDeg <= 73 { worstBand = max(worstBand, abs(g.mrp / ref - 1)) }
        }
        print(String(format: "VZ-0141 bezel: scale s = %.5f (1/2π = %.5f); MRP peak %.4f·sL vs reference peak %.4f·sL; worst in-band error %.1f%%",
                     s, 1 / (2 * Double.pi), peak / s, peakRef / s, worstBand * 100))
        XCTAssertLessThan(worstBand, 0.15, "MRP specular in the window's reflection band strays from the integral")
        // A passive reflector cannot return more radiance than its source (spec ≤ s·L); the
        // shipped form returned 3.55 L here — 22 × that ceiling.
        XCTAssertLessThan(peak, 1.0 * s, "MRP specular exceeds the light's own radiance")
        XCTAssertLessThan(peak, 1.3 * peakRef, "MRP highlight is brighter than the integral allows")
    }

    /// Randomised receivers against a 5 cm softbox, a 30 × 20 cm panel and a 1.0 × 1.2 m window
    /// over roughness 0.08 … 0.8: bounded everywhere, and accurate in the two limits the MRP
    /// is exact in (a light that covers the lobe; a lobe that covers the light).
    func testEnergyBoundAndBothLimitsAcrossLightsAndRoughness() throws {
        let probe = try makeProbe()
        let s = try measuredScale(probe)
        var rng = SplitMix(seed: 0x0141)
        let lights: [(String, Rect)] = [
            ("5 cm softbox", Rect(c: .zero, ex: SIMD3(0.025, 0, 0), ey: SIMD3(0, 0.025, 0))),
            ("30×20 cm", Rect(c: .zero, ex: SIMD3(0.15, 0, 0), ey: SIMD3(0, 0.10, 0))),
            ("1.0×1.2 m window", Rect(c: .zero, ex: SIMD3(0.5, 0, 0), ey: SIMD3(0, 0.6, 0))),
        ]
        for (name, rect) in lights {
            for rough in [0.08, 0.15, 0.3, 0.5, 0.8] {
                var recv: [Receiver] = []
                while recv.count < 48 {
                    let dir = simd_normalize(SIMD3(rng.gauss(), rng.gauss(), abs(rng.gauss()) + 0.2))
                    let p = dir * rng.uniform(0.3, 3.0)
                    let n = simd_normalize(simd_normalize(-p) + SIMD3(rng.gauss(), rng.gauss(), rng.gauss()) * 0.8)
                    var v: SIMD3<Double>
                    if rng.uniform(0, 1) < 0.6 {
                        let tgt = rect.ex * rng.uniform(-1.3, 1.3) + rect.ey * rng.uniform(-1.3, 1.3)
                        let lr = simd_normalize(tgt - p)
                        v = simd_normalize(-lr + 2 * simd_dot(n, lr) * n
                                           + SIMD3(rng.gauss(), rng.gauss(), rng.gauss()) * 0.05)
                    } else {
                        v = simd_normalize(n + SIMD3(rng.gauss(), rng.gauss(), rng.gauss()) * 0.7)
                    }
                    if simd_dot(n, v) >= 0.05 { recv.append(Receiver(p: p, n: n, v: v, roughness: rough)) }
                }
                let got = try run(probe, light: light(rect), recv)
                var ratios: [Double] = []
                var maxSpec = 0.0, maxRef = 0.0, overshoot = 0.0
                for (rc, g) in zip(recv, got) {
                    let ref = Self.reference(rect, rc, strata: 48) * s
                    XCTAssertTrue(g.mrp.isFinite && g.mrp >= 0, "\(name) r=\(rough): non-finite MRP")
                    maxSpec = max(maxSpec, g.mrp); maxRef = max(maxRef, ref)
                    overshoot = max(overshoot, g.mrp - 2 * ref)
                    if ref > 1e-4 * s { ratios.append(g.mrp / ref) }
                }
                ratios.sort()
                let median = ratios.isEmpty ? 1 : ratios[ratios.count / 2]
                print(String(format: "VZ-0141 %@ r=%.2f: MRP/ref median %.3f, max MRP %.4f·sL vs max ref %.4f·sL, overshoot %.4f·sL",
                             name, rough, median, maxSpec / s, maxRef / s, overshoot / s))
                XCTAssertLessThan(maxSpec, 1.0 * s, "\(name) r=\(rough): MRP specular exceeds the light's radiance")
                XCTAssertLessThan(overshoot, 0.08 * s, "\(name) r=\(rough): MRP overshoots the integral")
                // The two exact limits: a rough lobe over a small light (the shipped form was
                // 0.2–0.4 × here — it multiplied an extra cos and dropped π), and a glossy lobe
                // inside a large light (the shipped form was 11–13 × here).
                if rough >= 0.3, name.hasPrefix("5 cm") {
                    XCTAssertEqual(median, 1.0, accuracy: 0.2, "\(name) r=\(rough): small-light limit off")
                }
                if rough <= 0.15, name.hasPrefix("1.0") {
                    XCTAssertEqual(median, 1.0, accuracy: 0.25, "\(name) r=\(rough): covered-lobe limit off")
                }
            }
        }
    }

    /// Where the light covers the lobe and the view is well off grazing — the regime the LTC
    /// lane is validated in — the MRP and the LTC lane must agree on BRIGHTNESS (they differ
    /// only in lobe shape). This is the check that the MRP rides the same area-light scale as
    /// the LTC branch; flipping `areaLTCOverride` must not change how bright a highlight is.
    func testMRPAgreesWithLTCWhereTheLightCoversTheLobe() throws {
        let probe = try makeProbe()
        let rect = Rect(c: .zero, ex: SIMD3(0.5, 0, 0), ey: SIMD3(0, 0.6, 0))
        var recv: [Receiver] = []
        for rough in [0.1, 0.15, 0.2, 0.3] {
            for tiltDeg in [0.0, 20.0, 40.0] {
                // Receiver 0.8 m in front, normal tilted toward +X; view chosen so the mirror
                // direction hits the light's centre.
                let p = SIMD3<Double>(0.0, 0.0, 0.8)
                let t = tiltDeg * Double.pi / 180
                let n = SIMD3<Double>(sin(t) * 0.5, 0, -cos(t))
                let nn = simd_normalize(n)
                let l = simd_normalize(-p)
                let v = simd_normalize(-l + 2 * simd_dot(nn, l) * nn)
                recv.append(Receiver(p: p, n: nn, v: v, roughness: rough))
            }
        }
        let got = try run(probe, light: light(rect), recv)
        for (rc, g) in zip(recv, got) {
            let ratio = g.mrp / max(g.ltc, 1e-9)
            print(String(format: "VZ-0141 MRP vs LTC r=%.2f NdotV=%.2f: MRP %.5f LTC %.5f ratio %.2f",
                         rc.roughness, simd_dot(rc.n, rc.v), g.mrp, g.ltc, ratio))
            XCTAssertEqual(ratio, 1.0, accuracy: 0.15,
                           "MRP and LTC disagree on the brightness of a light that covers the lobe")
        }
    }
}

/// Deterministic generator (SplitMix64) so the sweep is reproducible.
private struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func uniform(_ a: Double, _ b: Double) -> Double {
        a + (b - a) * Double(next() >> 11) / Double(1 << 53)
    }
    mutating func gauss() -> Double {
        let u1 = max(uniform(0, 1), 1e-12), u2 = uniform(0, 1)
        return (-2 * log(u1)).squareRoot() * cos(2 * Double.pi * u2)
    }
}
