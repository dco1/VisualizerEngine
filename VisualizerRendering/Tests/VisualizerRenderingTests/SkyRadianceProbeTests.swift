import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// `VolumetricCloudRenderer.radianceProbes` — the `skyRadianceProbe` kernel (compiled from source:
/// `swift test` builds no metallib) fed the renderer's own packing
/// (`radianceProbeGPUData`), on synthetic equirect skies with analytic answers:
///  • a uniform sky returns its radiance (E/π of a constant sky is the constant);
///  • a sky bright above the horizon and black below, seen by a VERTICAL plane, returns exactly
///    half (the window case: half its hemisphere is ground);
///  • a sky whose radiance is max(0, y), seen by an up-facing plane, returns ∫cos²/π = 2/3;
///  • the directional convention is the dome's: a +X-only bright half is seen by a +X plane, not
///    a −X one (the equirect writer's `dirToEquirectUV`);
///  • an excluded cone drops only its samples (a uniform sky stays exact; the kept fraction is the
///    cone's cosine-weighted share, sin²α about the normal);
///  • the reading store never rolls back to an older generation.
@MainActor
final class SkyRadianceProbeTests: XCTestCase {

    private struct Harness {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let pso: MTLComputePipelineState
    }

    private func makeHarness() throws -> Harness {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/SkyRadianceProbe.metal")
        let lib = try MetalSourceLoader.makeLibrary(device: device, contentsOf: url)
        guard let fn = lib.makeFunction(name: "skyRadianceProbe") else { throw XCTSkip("no kernel") }
        let pso = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        return Harness(device: device, queue: queue, pso: pso)
    }

    /// An rgba16Float equirect (the dome's format and convention) from a direction function.
    private func makeSky(_ h: Harness, width: Int = 512, height: Int = 256,
                         _ radiance: (SIMD3<Float>) -> SIMD3<Float>) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width,
                                                         height: height, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .shared
        let t = h.device.makeTexture(descriptor: d)!
        var px = [Float16](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                // Inverse of dirToEquirectUV: u = atan2(z, x)/2π, v = 0.5 − asin(y)/π.
                let u = (Float(x) + 0.5) / Float(width), v = (Float(y) + 0.5) / Float(height)
                let phi = u * 2 * .pi, el = (0.5 - v) * .pi
                let dir = SIMD3<Float>(cos(el) * cos(phi), sin(el), cos(el) * sin(phi))
                let c = radiance(dir)
                let i = (y * width + x) * 4
                px[i] = Float16(c.x); px[i + 1] = Float16(c.y); px[i + 2] = Float16(c.z); px[i + 3] = 1
            }
        }
        px.withUnsafeBytes { raw in
            t.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                      withBytes: raw.baseAddress!, bytesPerRow: width * 8)
        }
        return t
    }

    private func run(_ h: Harness, _ sky: MTLTexture,
                     _ probes: [VolumetricCloudRenderer.RadianceProbe]) -> [SIMD4<Float>] {
        let gpu = VolumetricCloudRenderer.radianceProbeGPUData(probes)
        XCTAssertEqual(gpu.count * MemoryLayout<SIMD4<Float>>.stride, probes.count * 32,
                       "SkyRadianceProbeIn is 32 bytes")
        let out = h.device.makeBuffer(length: probes.count * 16, options: .storageModeShared)!
        let cb = h.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(h.pso)
        enc.setTexture(sky, index: 0)
        gpu.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        enc.setBuffer(out, offset: 0, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: probes.count, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: test harness
        XCTAssertNil(cb.error)
        let p = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: probes.count)
        return (0..<probes.count).map { p[$0] }
    }

    func testUniformSkyReturnsItsRadiance() throws {
        let h = try makeHarness()
        let c = SIMD3<Float>(0.2, 0.5, 1.25)
        let sky = makeSky(h) { _ in c }
        let r = run(h, sky, [.init(normal: SIMD3(0, 0, -1)), .init(normal: SIMD3(0, 1, 0)),
                             .init(normal: simd_normalize(SIMD3(1, -2, 3)))])
        for v in r {
            XCTAssertEqual(v.x, c.x, accuracy: c.x * 2e-3)
            XCTAssertEqual(v.y, c.y, accuracy: c.y * 2e-3)
            XCTAssertEqual(v.z, c.z, accuracy: c.z * 2e-3)
            XCTAssertEqual(v.w, 1, accuracy: 1e-6)
        }
    }

    func testVerticalWindowSeesHalfItsHemisphereAsGround() throws {
        let h = try makeHarness()
        let sky = makeSky(h) { d in d.y > 0 ? SIMD3(repeating: 1) : .zero }
        let r = run(h, sky, [.init(normal: SIMD3(0, 0, -1)), .init(normal: SIMD3(1, 0, 0))])
        for v in r { XCTAssertEqual(v.x, 0.5, accuracy: 0.01, "a vertical plane: half sky, half ground") }
    }

    func testCosineSkyGivesTwoThirdsOnAnUpFacingPlane() throws {
        let h = try makeHarness()
        let sky = makeSky(h) { d in SIMD3(repeating: max(0, d.y)) }
        let r = run(h, sky, [.init(normal: SIMD3(0, 1, 0))])
        XCTAssertEqual(r[0].x, 2.0 / 3.0, accuracy: 0.01, "E/π of L = cosθ is ∫cos²θ dω / π = 2/3")
    }

    func testDirectionConventionMatchesTheDome() throws {
        let h = try makeHarness()
        let sky = makeSky(h) { d in d.x > 0 ? SIMD3(1, 0, 0) : SIMD3(0, 0, 1) }
        let r = run(h, sky, [.init(normal: SIMD3(1, 0, 0)), .init(normal: SIMD3(-1, 0, 0))])
        XCTAssertGreaterThan(r[0].x, 0.98); XCTAssertLessThan(r[0].z, 0.02)
        XCTAssertGreaterThan(r[1].z, 0.98); XCTAssertLessThan(r[1].x, 0.02)
    }

    func testExcludedConeDropsOnlyItsSamples() throws {
        let h = try makeHarness()
        let c = SIMD3<Float>(0.7, 0.7, 0.7)
        let sky = makeSky(h) { _ in c }
        let alpha: Float = 20 * .pi / 180
        let n = SIMD3<Float>(0, 0, -1)
        let r = run(h, sky, [.init(normal: n, excludeDirection: n, excludeHalfAngle: alpha)])
        XCTAssertEqual(r[0].x, c.x, accuracy: 2e-3, "the mean of the rest of a uniform sky is the sky")
        // Cosine-weighted share of a cone of half-angle α about the normal = sin²α.
        XCTAssertEqual(r[0].w, 1 - sin(alpha) * sin(alpha), accuracy: 0.02)
    }

    func testStoreNeverRollsBack() {
        let store = RadianceProbeStore()
        func reading(_ g: UInt64) -> VolumetricCloudRenderer.RadianceProbeReading {
            .init(probes: [], radiance: [SIMD3(repeating: Float(g))], keptFraction: [1], generation: g)
        }
        store.publish(reading(3))
        store.publish(reading(2))
        XCTAssertEqual(store.latest?.generation, 3)
        store.publish(reading(4))
        XCTAssertEqual(store.latest?.generation, 4)
    }

    func testNoProbesMeansNoData() {
        XCTAssertTrue(VolumetricCloudRenderer.radianceProbeGPUData([]).isEmpty)
    }
}
