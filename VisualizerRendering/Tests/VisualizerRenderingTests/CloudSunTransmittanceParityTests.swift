import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// DH-1492 — the CPU reading of the cloud deck's shadow (`IlluminatoramaRenderer.cloudSunTransmittance`,
/// what a host's exposure meter / daylight intake reads) is the SAME lookup the shaders make: the shipped
/// `IlluminatoramaCloudShadow.h` compiled from source and run over a scatter of points, compared value by
/// value against the Swift mirror on one synthetic map (header + a seeded field), every branch exercised —
/// off, sun too low, outside the square, the edge fade, the height limit, partial strength.
final class CloudSunTransmittanceParityTests: XCTestCase {

    private static func shaderURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/\(name)")
    }

    private static let probeKernel = """

    struct CloudProbe { float4 p; float4 ld; };
    kernel void dh1492CloudProbe(const device float* cs [[buffer(0)]],
                                 const device CloudProbe* q [[buffer(1)]],
                                 device float* outT [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
        outT[gid] = illumiCloudSunTransmittance(cs, q[gid].p.xyz, q[gid].ld.xyz);
    }
    """

    /// A 64² map over 400 m centred at (30, −20): a smooth seeded field in 0.05…1 so bilinear weights matter.
    private static func map(strength: Float, heightLimit: Float, enabled: Bool = true) -> [Float] {
        let res = 64
        var m: [Float] = [30, -20, 400, enabled ? 1 : 0, Float(res), 2, strength, heightLimit]
        for y in 0..<res {
            for x in 0..<res {
                let v = 0.525 + 0.475 * sin(Float(x) * 0.37 + 1.3) * cos(Float(y) * 0.23 - 0.4)
                m.append(v)
            }
        }
        return m
    }

    private static func probes() -> [(SIMD3<Float>, SIMD3<Float>)] {
        var rng = SystemRandomNumberGenerator()
        var out: [(SIMD3<Float>, SIMD3<Float>)] = []
        let suns: [SIMD3<Float>] = [simd_normalize(SIMD3(0.3, 0.8, -0.5)), simd_normalize(SIMD3(-0.6, 0.35, 0.2)),
                                    SIMD3(0, 1, 0), simd_normalize(SIMD3(1, 0.01, 0))]   // last: below the 0.02 gate
        for i in 0..<2048 {
            // Over, at and past the square's edge (±200 m around the centre, ±260 m sampled).
            let x = Float.random(in: -230...290, using: &rng)
            let z = Float.random(in: -280...240, using: &rng)
            let y = Float.random(in: 0...12, using: &rng)
            out.append((SIMD3(x, y, z), suns[i % suns.count]))
        }
        return out
    }

    @MainActor
    func testCPUMirrorMatchesTheShippedShaderLookup() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {  // gpu-ok: test harness
            throw XCTSkip("no Metal")
        }
        let src = "#include <metal_stdlib>\nusing namespace metal;\n"
            + (try MetalSourceLoader.source(contentsOf: Self.shaderURL("IlluminatoramaCloudShadow.h"))) + Self.probeKernel
        let lib = try device.makeLibrary(source: src, options: nil)
        guard let fn = lib.makeFunction(name: "dh1492CloudProbe") else { throw XCTSkip("probe missing") }
        let pipe = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        let pts = Self.probes()
        let rows = pts.flatMap { [SIMD4<Float>($0.0, 0), SIMD4<Float>($0.1, 0)] }

        let maps: [(String, [Float])] = [
            ("full", Self.map(strength: 1, heightLimit: 0)),
            ("partial strength", Self.map(strength: 0.6, heightLimit: 0)),
            ("ground only", Self.map(strength: 1, heightLimit: 4)),
            ("disabled", Self.map(strength: 1, heightLimit: 0, enabled: false)),
        ]
        for (name, m) in maps {
            guard let mb = device.makeBuffer(bytes: m, length: m.count * 4, options: .storageModeShared),
                  let qb = device.makeBuffer(bytes: rows, length: rows.count * 16, options: .storageModeShared),
                  let ob = device.makeBuffer(length: pts.count * 4, options: .storageModeShared),
                  let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no buffers") }
            enc.setComputePipelineState(pipe)
            enc.setBuffer(mb, offset: 0, index: 0); enc.setBuffer(qb, offset: 0, index: 1); enc.setBuffer(ob, offset: 0, index: 2)
            enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
            let gpu = ob.contents().bindMemory(to: Float.self, capacity: pts.count)
            var worst: Float = 0, shaded = 0, unity = 0
            m.withUnsafeBufferPointer { cs in
                for (i, (p, ld)) in pts.enumerated() {
                    let cpu = IlluminatoramaRenderer.cloudSunTransmittance(header: cs.baseAddress!, count: cs.count,
                                                                           at: p, towardSun: ld)
                    worst = max(worst, abs(cpu - gpu[i]))
                    if cpu < 0.999 { shaded += 1 } else { unity += 1 }
                }
            }
            print(String(format: "DH-1492 cloud parity [%@]: %d probes, max |cpu-gpu| %.2e, shaded %d, unity %d",
                         name, pts.count, worst, shaded, unity))
            XCTAssertLessThan(worst, 1e-4, "\(name): CPU mirror drifted from IlluminatoramaCloudShadow.h")
            if name == "disabled" { XCTAssertEqual(shaded, 0, "disabled map must read exactly 1") }
            else { XCTAssertGreaterThan(shaded, pts.count / 8, "\(name): the probes must exercise the shaded branch") }
        }
    }
}
