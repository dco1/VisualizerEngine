import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// AO v2 / progressive sampling (Daydream DH-0887): the parts that can be checked without a scene.
///
/// - the uniform mirrors match their Metal twins byte for byte (a silent mismatch reads every field
///   after the first difference as garbage — the class of bug a `_pad` rename once shipped);
/// - the accumulator's filter schedule;
/// - the shared sampler (`IlluminatoramaSampling.h`) run ON THE GPU through a probe kernel: the
///   Owen-scrambled Sobol points are a (0,m,2)-net at every power-of-two prefix (the property that
///   makes a still's average stratified whenever it stops), scrambles differ per pixel, and the
///   cosine mapping is unit length with E[cos θ] = 2/3.
final class IlluminatoramaAOAccumulatorTests: XCTestCase {

    func testUniformMirrorsMatchTheMetalLayouts() {
        // Metal: `static_assert(sizeof(RTAOv2Uniforms) == 128)` in IlluminatoramaAOAccumulate.metal.
        XCTAssertEqual(MemoryLayout<IlluminatoramaAOAccumulator.RTAOv2Uniforms>.stride, 128)
        XCTAssertEqual(MemoryLayout<IlluminatoramaAOAccumulator.AccumulateUniforms>.stride, 16)
        XCTAssertEqual(MemoryLayout<IlluminatoramaAOAccumulator.DenoiseUniforms>.stride, 80)
    }

    func testDenoiseStrengthEasesFromFullToAHalfFloor() {
        XCTAssertEqual(IlluminatoramaAOAccumulator.denoiseStrength(frames: 1), 1.0)
        var last: Float = 2
        for k in 1...256 {
            let f = IlluminatoramaAOAccumulator.denoiseStrength(frames: k)
            XCTAssertLessThanOrEqual(f, last, "never grows as the mean fills")
            XCTAssertGreaterThanOrEqual(f, 0.5, "never fades below the floor")
            last = f
        }
        XCTAssertEqual(IlluminatoramaAOAccumulator.denoiseStrength(frames: 256), 0.5)
    }

    // MARK: - GPU probe of the sampling header

    private static let probeKernel = """

    kernel void aoSamplingProbe(device float4* out [[buffer(0)]],
                                constant uint& count [[buffer(1)]],
                                uint gid [[thread_position_in_grid]]) {
        if (gid >= count * 4u) return;
        uint pixel = gid / count;             // four pixels, one sequence each
        uint i = gid % count;
        uint seed = illumiPixelSeed(uint2(pixel * 37u, pixel * 11u), 0u);
        // Pixels 0–1 read the plain sequence, 2–3 the index-shuffled one every call site uses:
        // both must be (0,m,2)-nets at every power-of-two prefix.
        float2 u = pixel < 2u ? illumiSobolOwen2D(i, seed) : illumiSobolOwen2DShuffled(i, seed);
        float3 d = illumiCosineHemisphereAxisSafe(normalize(float3(0.3, 0.9, 0.1)), u);
        out[gid] = float4(u, dot(d, normalize(float3(0.3, 0.9, 0.1))), length(d));
    }
    """

    private func runProbe(count: Int) throws -> [SIMD4<Float>] {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaSampling.h")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no header") }
        let source = try MetalSourceLoader.source(contentsOf: url) + Self.probeKernel
        let lib = try device.makeLibrary(source: source, options: nil)
        guard let fn = lib.makeFunction(name: "aoSamplingProbe") else { throw XCTSkip("probe missing") }
        let pipeline = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        let n = count * 4
        guard let buf = device.makeBuffer(length: n * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw XCTSkip("allocation failed") }
        var c = UInt32(count)
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBytes(&c, length: 4, index: 1)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(n, 64), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()   // gpu-ok: test harness
        let p = buf.contents().bindMemory(to: SIMD4<Float>.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    /// Every power-of-two prefix of each pixel's sequence puts exactly one point in every
    /// elementary box 2^-a × 2^-b (a + b = m): the (0,m,2)-net property Owen scrambling preserves —
    /// including with the INDEX shuffled (an aligned block of indices maps onto an aligned block),
    /// which is what lets the sampler decorrelate from the base-2 TAA jitter without losing it.
    func testOwenSobolPrefixesAreStratified() throws {
        let count = 256
        let r = try runProbe(count: count)
        for px in 0..<4 {
            let pts = (0..<count).map { SIMD2(r[px * count + $0].x, r[px * count + $0].y) }
            for m in [2, 4, 6, 8] {
                let n = 1 << m
                for a in 0...m {
                    let b = m - a
                    var cells = Set<Int>()
                    for p in pts.prefix(n) {
                        let cx = min((1 << a) - 1, Int(p.x * Float(1 << a)))
                        let cy = min((1 << b) - 1, Int(p.y * Float(1 << b)))
                        cells.insert(cx * (1 << b) + cy)
                    }
                    XCTAssertEqual(cells.count, n, "pixel \(px): prefix \(n) not stratified in \(1 << a)×\(1 << b) boxes")
                }
            }
        }
        // Per-pixel scrambles differ (no shared pattern across the screen).
        XCTAssertNotEqual(r[0].x, r[count].x)
        XCTAssertNotEqual(r[count].x, r[2 * count].x)
    }

    func testCosineMappingIsUnitAndCosineWeighted() throws {
        let count = 1024
        let r = try runProbe(count: count)
        var meanCos = 0.0
        for i in 0..<count {
            XCTAssertEqual(r[i].w, 1.0, accuracy: 1e-4, "direction must be unit length")
            XCTAssertGreaterThanOrEqual(r[i].z, -1e-5, "direction must be in the hemisphere")
            meanCos += Double(r[i].z)
        }
        meanCos /= Double(count)
        XCTAssertEqual(meanCos, 2.0 / 3.0, accuracy: 0.01, "cosine-weighted hemisphere ⇒ E[cos θ] = 2/3")
    }
}
