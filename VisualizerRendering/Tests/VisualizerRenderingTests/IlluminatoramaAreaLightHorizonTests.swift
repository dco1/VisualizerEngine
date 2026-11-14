import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// DH-1686 — a horizon-split area light (a window portal: sky above the receiver's horizon, ground
/// below) against a CPU integral of the same rectangle.
///
/// Runs the REAL `areaLightHorizonForm` / `evalAreaLight` (IlluminatoramaLighting.metal, compiled
/// from source with a probe kernel appended) for a vertical 1.2 × 1.4 m window and receivers on
/// the floor, the ceiling and the walls around it. Holds:
/// - the split's TOTAL is the unsplit form factor (`ltcPolygonForm`) — the cut edge cancels;
/// - its SKY SHARE matches a Monte-Carlo integral of the clamped cosine over the part of the
///   rectangle above the receiver's horizontal plane;
/// - a ceiling (above the window head) sees only ground, a floor (below the sill) only sky;
/// - `evalAreaLight` with `belowColor == color` is the unsplit light, and with `belowColor` 0 it
///   is `color × share` of it.
@MainActor
final class IlluminatoramaAreaLightHorizonTests: XCTestCase {

    private static let probeKernel = """

    // ── DH-1686 test probe (appended by IlluminatoramaAreaLightHorizonTests) ──
    kernel void dh1686HorizonProbe(device const AreaLight* light [[buffer(0)]],
                                   device const float4*   recv  [[buffer(1)]],
                                   device float4*          out   [[buffer(2)]],
                                   texture2d<float> ltcMat [[texture(0)]],
                                   uint gid [[thread_position_in_grid]]) {
        AreaLight al = light[0];
        float3 P = recv[gid * 2 + 0].xyz, N = recv[gid * 2 + 1].xyz;
        float3 V = N;
        float3 p0 = al.center - al.ex - al.ey - P, p1 = al.center - al.ex + al.ey - P;
        float3 p2 = al.center + al.ex + al.ey - P, p3 = al.center + al.ex - al.ey - P;
        float  ff = ltcPolygonForm(N, p0, p1, p2, p3, al.twoSided > 0.5);
        float2 hf = areaLightHorizonForm(N, P, al.center, al.ex, al.ey, al.twoSided > 0.5);
        // Diffuse only: albedo 1, roughness 1, F0 0.04 — the specular is small and is read from
        // the SAME colour blend, so the ratio test below holds for the whole return.
        float3 lit = evalAreaLight(al, P, N, V, float3(1.0), 0.0, 1.0, ltcMat, ltcMat, false);
        out[gid] = float4(ff, hf.x, hf.y, lit.x);
    }
    """

    private struct Probe { let pipeline: MTLComputePipelineState; let queue: MTLCommandQueue; let device: MTLDevice; let lut: MTLTexture }

    private func makeProbe() throws -> Probe {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaLighting.metal")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no shader") }
        let source = try MetalSourceLoader.source(contentsOf: url) + Self.probeKernel
        let lib = try device.makeLibrary(source: source, options: nil)
        guard let fn = lib.makeFunction(name: "dh1686HorizonProbe") else { throw XCTSkip("probe kernel missing") }
        let pipeline = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 4, height: 4, mipmapped: false)
        d.storageMode = .shared
        guard let lut = device.makeTexture(descriptor: d) else { throw XCTSkip("no texture") }
        return Probe(pipeline: pipeline, queue: queue, device: device, lut: lut)
    }

    /// Per receiver (P, N): (ltcPolygonForm, split total, sky share, evalAreaLight red).
    private func run(_ probe: Probe, _ light: IlluminatoramaAreaLight,
                     _ recv: [(SIMD3<Float>, SIMD3<Float>)]) throws -> [SIMD4<Float>] {
        var lights = [light]
        var packed: [SIMD4<Float>] = []
        for (p, n) in recv { packed.append(SIMD4(p, 1)); packed.append(SIMD4(simd_normalize(n), 0)) }
        let dev = probe.device
        guard let lb = dev.makeBuffer(bytes: &lights, length: MemoryLayout<IlluminatoramaAreaLight>.stride, options: .storageModeShared),
              let rb = dev.makeBuffer(bytes: packed, length: packed.count * 16, options: .storageModeShared),
              let ob = dev.makeBuffer(length: recv.count * 16, options: .storageModeShared),
              let cmd = probe.queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder()
        else { throw XCTSkip("no buffers") }
        enc.setComputePipelineState(probe.pipeline)
        enc.setBuffer(lb, offset: 0, index: 0)
        enc.setBuffer(rb, offset: 0, index: 1)
        enc.setBuffer(ob, offset: 0, index: 2)
        enc.setTexture(probe.lut, index: 0)
        enc.dispatchThreads(MTLSize(width: recv.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(64, recv.count), height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()  // gpu-ok: test harness
        let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: recv.count)
        return (0..<recv.count).map { o[$0] }
    }

    /// A 1.2 m wide × 1.4 m tall window in the plane z = 0, sill 0.9 m, head 2.3 m, emitting
    /// toward +z (into the room). `ex` along the wall, `ey` up.
    private static func window(color: SIMD3<Float>, below: SIMD3<Float>?) -> IlluminatoramaAreaLight {
        var ex = SIMD3<Float>(0.6, 0, 0)
        let ey = SIMD3<Float>(0, 0.7, 0)
        if simd_dot(simd_cross(ex, ey), SIMD3(0, 0, 1)) < 0 { ex = -ex }
        var al = IlluminatoramaAreaLight(center: SIMD3(0, 1.6, 0), ex: ex, ey: ey, color: color, radius: 1e4)
        if let below { al.belowColor = below; al.horizonSplit = 1 }
        return al
    }

    /// CPU: (whole, above-horizon) clamped-cosine form factors of the window rect at (P, N), by a
    /// stratified midpoint sum over the rectangle (cosR · cosL / d² dA / π, then × π/(2π) to the
    /// engine's `ltcPolygonForm` scale, which is 1/(2π) of the Lambert form factor — VZ-0143).
    private static func cpuForms(_ P: SIMD3<Double>, _ N: SIMD3<Double>) -> (Double, Double) {
        let m = 400
        var whole = 0.0, up = 0.0
        let w = 1.2, h = 1.4, dA = (w / Double(m)) * (h / Double(m))
        for i in 0..<m { for j in 0..<m {
            let q = SIMD3<Double>(-0.6 + (Double(i) + 0.5) * w / Double(m), 0.9 + (Double(j) + 0.5) * h / Double(m), 0)
            let d = q - P
            let r2 = simd_length_squared(d)
            let l = d / sqrt(r2)
            let cosR = max(0, simd_dot(N, l))
            let cosL = max(0, -l.z)          // the window emits toward +z: from P (z > 0) l.z < 0
            let v = cosR * cosL / r2 * dA / .pi
            whole += v
            if q.y > P.y { up += v }
        } }
        return (whole / 2, up / 2)   // Lambert form factor F → the engine's F/2 (= F·π/(2π))
    }

    private static let receivers: [(String, SIMD3<Float>, SIMD3<Float>)] = [
        ("ceiling above the head",  SIMD3(0.3, 2.6, 1.5),  SIMD3(0, -1, 0)),
        ("ceiling deep in room",    SIMD3(-0.5, 2.6, 4.0), SIMD3(0, -1, 0)),
        ("floor near",              SIMD3(0.2, 0.0, 1.2),  SIMD3(0, 1, 0)),
        ("floor far",               SIMD3(1.0, 0.0, 4.0),  SIMD3(0, 1, 0)),
        ("facing wall mid",         SIMD3(0.0, 1.5, 4.0),  SIMD3(0, 0, -1)),
        ("facing wall high",        SIMD3(0.5, 2.2, 4.0),  SIMD3(0, 0, -1)),
        ("side wall mid",           SIMD3(1.5, 1.4, 1.0),  SIMD3(-1, 0, 0)),
        ("side wall low",           SIMD3(-1.5, 0.5, 2.0), SIMD3(1, 0, 0)),
        ("tilted desk",             SIMD3(0.0, 0.75, 1.0), SIMD3(0, 0.8, -0.6)),
    ]

    func testSplitTotalIsTheUnsplitFormFactorAndTheShareMatchesTheIntegral() throws {
        let probe = try makeProbe()
        let recv = Self.receivers.map { ($0.1, $0.2) }
        let out = try run(probe, Self.window(color: SIMD3(1, 1, 1), below: SIMD3(0, 0, 0)), recv)
        let plain = try run(probe, Self.window(color: SIMD3(1, 1, 1), below: nil), recv)
        var scale: Double? = nil      // the engine's form-factor units vs the Lambert integral (VZ-0143)
        for (k, r) in Self.receivers.enumerated() {
            let o = out[k]
            let (whole, up) = Self.cpuForms(SIMD3<Double>(r.1), simd_normalize(SIMD3<Double>(r.2)))
            print("[dh1686] " + r.0.padding(toLength: 24, withPad: " ", startingAt: 0)
                  + String(format: " ff %.5f split %.5f cpu %.5f  share %.4f cpu %.4f  lit %.5f / %.5f",
                           o.x, o.y, whole, o.z, whole > 0 ? up / whole : 0, o.w, plain[k].w))
            XCTAssertEqual(o.y, o.x, accuracy: max(1e-6, o.x * 1e-3), "\(r.0): the split's total must be the unsplit light")
            guard whole > 1e-4 else { continue }
            let s = Double(o.x) / whole
            if let scale { XCTAssertEqual(s, scale, accuracy: scale * 0.01, "\(r.0): form factor vs the CPU integral") }
            else { scale = s }
            XCTAssertEqual(Double(o.z), up / whole, accuracy: 0.01, "\(r.0): sky share vs the CPU integral")
            // belowColor 0 ⇒ the light is the sky share of the unsplit light.
            XCTAssertEqual(o.w, plain[k].w * o.z, accuracy: max(1e-6, plain[k].w * 1e-3), "\(r.0): evalAreaLight ≠ share × unsplit")
        }
        XCTAssertEqual(out[0].z, 0, accuracy: 1e-6, "a ceiling above the window head sees only ground")
        XCTAssertEqual(out[2].z, 1, accuracy: 1e-6, "a floor below the sill sees only sky")
        XCTAssertEqual(out[0].w, 0, accuracy: 1e-7, "…so with a black ground it takes nothing from the window")
    }

    func testEqualColoursAreTheUnsplitLight() throws {
        let probe = try makeProbe()
        let c = SIMD3<Float>(0.8, 0.9, 1.1)
        let split = try run(probe, Self.window(color: c, below: c), Self.receivers.map { ($0.1, $0.2) })
        let plain = try run(probe, Self.window(color: c, below: nil), Self.receivers.map { ($0.1, $0.2) })
        for k in 0..<Self.receivers.count {
            XCTAssertEqual(split[k].w, plain[k].w, accuracy: max(1e-6, plain[k].w * 1e-3), Self.receivers[k].0)
        }
    }
}
