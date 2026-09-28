import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// The opt-in HUE-STABLE TOE of the display transform (`IlluminatoramaRenderer.hueStableToe`,
/// IlluminatoramaTonemap.metal `hueStableDisplay` / `hueStableToeWeight`), on the real shader
/// functions compiled from source:
///  • it is the shipped transform exactly for a GREY (so the toe joins the curve seamlessly and a
///    neutral frame prints the same);
///  • for a coloured pixel several stops under mid-grey it keeps the scene's LINEAR channel ratios,
///    where AgX's per-channel toe raises them to a high power (measured: Digital Clock's 19:40 sky,
///    linear 1 : 0.66 : 1.63, printed near-pure violet) — the defect this exists for;
///  • it stays in gamut and monotone in exposure;
///  • the knee weight is exactly 1 above `hi` (the shipped path) and 0 below `lo`.
@MainActor
final class IlluminatoramaHueStableToeTests: XCTestCase {

    private static let kernel = """
    kernel void toeProbe(device float4* out [[buffer(0)]], device const float4* inp [[buffer(1)]],
                         constant float4& knee [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
        float3 x = inp[tid].xyz;
        out[tid * 2 + 0] = float4(displayTransform(x, 2u), hueStableToeWeight(x, knee.x, knee.y));
        out[tid * 2 + 1] = float4(hueStableDisplay(x, displayTransform(x, 2u)), 0.0);
    }
    """

    private struct Probe { let agx: SIMD3<Float>; let weight: Float; let toe: SIMD3<Float> }

    private func probe(_ inputs: [SIMD3<Float>], knee: SIMD2<Float> = SIMD2(0.1, 0.4)) throws -> [Probe] {
        let engine = SimEngine.shared
        let device = engine.device
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaTonemap.metal")
        let src = try MetalSourceLoader.source(contentsOf: url)
        let lib = try device.makeLibrary(source: src + "\n" + Self.kernel, options: nil)
        guard let fn = lib.makeFunction(name: "toeProbe") else { throw XCTSkip("probe missing") }
        let pso = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        var inp = inputs.map { SIMD4<Float>($0, 0) }
        var k = SIMD4<Float>(knee.x, knee.y, 0, 0)
        let n = inputs.count
        guard let ib = device.makeBuffer(bytes: &inp, length: n * 16, options: .storageModeShared),
              let ob = device.makeBuffer(length: n * 32, options: .storageModeShared),
              let cb = engine.commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw XCTSkip("no resources") }
        enc.setComputePipelineState(pso)
        enc.setBuffer(ob, offset: 0, index: 0)
        enc.setBuffer(ib, offset: 0, index: 1)
        enc.setBytes(&k, length: 16, index: 2)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(n, 64), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: test harness
        if let e = cb.error { throw e }
        let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: n * 2)
        return (0..<n).map { i in
            Probe(agx: SIMD3(o[2 * i].x, o[2 * i].y, o[2 * i].z), weight: o[2 * i].w,
                  toe: SIMD3(o[2 * i + 1].x, o[2 * i + 1].y, o[2 * i + 1].z))
        }
    }

    func testGreyIsTheShippedTransform() throws {
        let greys = stride(from: -12.0 as Float, through: 3, by: 0.5).map { SIMD3<Float>(repeating: pow(2, $0)) }
        for (g, p) in zip(greys, try probe(greys)) {
            let l = 0.2126 * p.agx.x + 0.7152 * p.agx.y + 0.0722 * p.agx.z
            for c in 0..<3 {
                XCTAssertEqual(p.toe[c], p.agx.max(), accuracy: max(1e-6, l * 1e-4), "grey \(g.x): the toe is the transform's own peak")
                XCTAssertEqual(p.agx[c], l, accuracy: max(2e-5, l * 0.01), "AgX keeps a grey grey (inset/outset rows sum to 1)")
                XCTAssertEqual(p.toe[c], p.agx[c], accuracy: max(2e-5, l * 0.01), "grey \(g.x): the toe IS the shipped transform")
            }
        }
    }

    func testToeKeepsTheSceneRatiosWhereAgXCrushesThem() throws {
        // The 19:40 zenith probe's chromaticity, 5 stops under mid-grey (where the lifted night
        // frame lands), and the LED's red at the pool's level.
        let lavender = SIMD3<Float>(1, 0.66, 1.63) / 1.63 * 0.18 / 32
        let red = SIMD3<Float>(1, 0.035, 0.012) * 0.02
        let p = try probe([lavender, red])
        // Toe: G/B exactly the scene's.
        XCTAssertEqual(p[0].toe.y / p[0].toe.z, 0.66 / 1.63, accuracy: 1e-4)
        XCTAssertEqual(p[0].toe.x / p[0].toe.z, 1 / 1.63, accuracy: 1e-4)
        XCTAssertEqual(p[1].toe.y / p[1].toe.x, 0.035, accuracy: 1e-4)
        // …at the brightness AgX gives the dominant channel (the red keeps its level).
        XCTAssertEqual(p[1].toe.x, p[1].agx.max(), accuracy: 1e-6)
        // AgX: the ratios raised to a high power (the defect) — G/B far under the scene's 0.40.
        XCTAssertLessThan(p[0].agx.y / p[0].agx.z, 0.25, "AgX's per-channel toe crushes the minor channels")
        // The toe's peak channel is the transform's own peak (in gamut, same black level).
        XCTAssertLessThanOrEqual(max(p[0].toe.x, max(p[0].toe.y, p[0].toe.z)), 1)
    }

    func testToeIsMonotoneInGamutAndTheKneeIsExact() throws {
        let chroma = SIMD3<Float>(0.3, 0.2, 0.9)
        let scales = stride(from: -14.0 as Float, through: 4, by: 0.25).map { pow(2, $0) }
        let p = try probe(scales.map { chroma * $0 }, knee: SIMD2(0.05, 0.3))
        var prev = SIMD3<Float>(repeating: -1)
        for (s, q) in zip(scales, p) {
            // Monotone wherever the toe can be used (below the knee's top, with headroom). Far
            // over white AgX's own peak channel is not monotone (its outset matrix), but there the
            // knee weight is exactly 1 and the toe is never evaluated into the frame.
            if q.weight < 1 || s < 4 {
                XCTAssertTrue(q.toe.x >= prev.x && q.toe.y >= prev.y && q.toe.z >= prev.z, "monotone at \(s)")
            }
            XCTAssertTrue(q.toe.max() <= 1 && q.toe.min() >= 0, "in gamut at \(s)")
            prev = q.toe
            let x = chroma * s
            let b = max(0.2126 * x.x + 0.7152 * x.y + 0.0722 * x.z, 0.5 * x.max())
            if b >= 0.3 { XCTAssertEqual(q.weight, 1, "above hi: exactly the shipped path") }
            if b <= 0.05 { XCTAssertEqual(q.weight, 0, "below lo: the toe alone") }
        }
    }
}
