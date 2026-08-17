import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Pins the equirect WRITER's azimuth convention as a number.
///
/// `volSkyRender` (VolumetricSky.metal) writes the sky dome/IBL equirect with
/// +X at the u = 0 seam: `az = u·2π`, `rayDir = (cos e·cos az, sin e, cos e·sin az)`,
/// i.e. a reader must use `u = frac(atan2(dir.z, dir.x) / 2π)` — the convention
/// of `dirToEquirectUV` in IlluminatoramaCommon.h. A second, +0.5-offset
/// variant of that helper used to ship in IlluminatoramaSecondary.h /
/// IlluminatoramaRT.metal / IlluminatoramaSurfaceCache.metal, which put every
/// RT sky sample (RT glass, RT GI, surface cache) 180° in azimuth from where
/// the raster draws the sky. This test renders the writer with a known sun and
/// asserts the sun's argmax column lands at `frac(atan2(toSun.z, toSun.x)/2π)`
/// — NOT +0.5 from it — so the two conventions can never silently diverge again.
///
/// Runs the REAL kernel: compiled from source via MetalSourceLoader (SwiftPM
/// does not compile .metal — Bundle.module carries raw sources under
/// `swift test`), packed with the renderer's own `SkyUniforms`.
final class VolumetricSkyEquirectConventionTests: XCTestCase {

    func testSunLandsAtAtan2OverTwoPiColumn() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness

        let shaderURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/VolumetricSky.metal")
        guard FileManager.default.fileExists(atPath: shaderURL.path) else { throw XCTSkip("no shader") }
        let lib = try MetalSourceLoader.makeLibrary(device: device, contentsOf: shaderURL)
        guard let fn = lib.makeFunction(name: "volSkyRender") else {
            return XCTFail("volSkyRender missing from VolumetricSky.metal")
        }
        let pipeline = try device.makeComputePipelineState(function: fn)

        // Sun at azimuth 0.30 of a turn, 15° above the horizon. Under the
        // writer's convention the disc's column is u ≈ 0.30; under the buggy
        // +0.5 reader convention it would be 0.80 — unambiguously separated.
        let phi: Float = 2.0 * .pi * 0.30
        let elev: Float = 15.0 * .pi / 180.0
        let toSun = SIMD3<Float>(cos(elev) * cos(phi), sin(elev), cos(elev) * sin(phi))
        let expectedU = { () -> Float in
            var u = atan2(toSun.z, toSun.x) / (2.0 * Float.pi)
            if u < 0 { u += 1 }
            return u
        }()
        let expectedV = 0.5 - asin(toSun.y) / .pi

        var params = VolumetricCloudRenderer.Params()
        params.sunDir = -toSun               // params.sunDir is light-travel direction
        params.coverage = 0                  // no clouds occluding the disc
        params.density = 0
        params.debugNeonBackground = false
        params.celestialsInDome = false      // Daydream's configuration: sun only
        let uniforms = SkyUniforms(params: params, time: 0)

        // IBL-resolution target — the texture the RT paths actually sample.
        let W = 256, H = 128
        let td = MTLTextureDescriptor()
        td.pixelFormat = .rgba16Float
        td.width = W; td.height = H
        td.usage = [.shaderWrite, .shaderRead]
        td.storageMode = .shared
        guard let target = device.makeTexture(descriptor: td) else { throw XCTSkip("no texture") }

        // Zeroed 3D noise volume (clouds are off; the binding just has to be valid).
        let nd = MTLTextureDescriptor()
        nd.textureType = .type3D
        nd.pixelFormat = .rgba16Float
        nd.width = 4; nd.height = 4; nd.depth = 4
        nd.usage = [.shaderRead]
        nd.storageMode = .shared
        guard let noise = device.makeTexture(descriptor: nd) else { throw XCTSkip("no noise texture") }

        guard let uBuf = device.makeBuffer(length: MemoryLayout<SkyUniforms>.stride, options: .storageModeShared),
              let lightBuf = device.makeBuffer(length: 256, options: .storageModeShared)
        else { throw XCTSkip("no buffers") }
        uBuf.contents().bindMemory(to: SkyUniforms.self, capacity: 1).pointee = uniforms

        guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
            throw XCTSkip("no command buffer")
        }
        enc.setComputePipelineState(pipeline)
        enc.setTexture(target, index: 0)
        enc.setTexture(noise, index: 1)
        enc.setBuffer(uBuf, offset: 0, index: 0)
        enc.setBuffer(lightBuf, offset: 0, index: 1)
        let tgw = min(pipeline.threadExecutionWidth, W)
        let tgh = max(1, pipeline.maxTotalThreadsPerThreadgroup / max(1, tgw))
        enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgw, height: min(tgh, H), depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()  // gpu-ok: test harness

        // Read back and take the argmax-luminance texel.
        var px = [Float16](repeating: 0, count: W * H * 4)
        px.withUnsafeMutableBytes { raw in
            target.getBytes(raw.baseAddress!,
                            bytesPerRow: W * 4 * MemoryLayout<Float16>.stride,
                            from: MTLRegionMake2D(0, 0, W, H),
                            mipmapLevel: 0)
        }
        var bestLum: Float = -1
        var bestX = 0, bestY = 0
        for y in 0..<H {
            for x in 0..<W {
                let i = (y * W + x) * 4
                let lum = 0.2126 * Float(px[i]) + 0.7152 * Float(px[i + 1]) + 0.0722 * Float(px[i + 2])
                if lum > bestLum { bestLum = lum; bestX = x; bestY = y }
            }
        }
        XCTAssertGreaterThan(bestLum, 0, "writer produced a black frame — kernel did not run")

        let uMax = (Float(bestX) + 0.5) / Float(W)
        let vMax = (Float(bestY) + 0.5) / Float(H)
        // Wrap-aware distance in u. Tolerance 0.05 of a turn (±18°) covers the
        // disc + halo comfortably; the buggy convention misses by 0.5 exactly.
        var du = abs(uMax - expectedU)
        if du > 0.5 { du = 1.0 - du }
        print("EQUIRECT_WRITER: sun argmax u=\(uMax) v=\(vMax) expected u=\(expectedU) v=\(expectedV) lum=\(bestLum)")
        XCTAssertLessThan(du, 0.05,
            "sun column is \(du) of a turn from atan2(z,x)/2π — the writer's convention moved; " +
            "every dirToEquirectUV reader (Common.h + the RT copies) must move with it")
        XCTAssertEqual(vMax, expectedV, accuracy: 0.06,
            "sun row disagrees with v = 0.5 − asin(y)/π")
    }
}
