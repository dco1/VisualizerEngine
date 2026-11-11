import XCTest
import simd
import Metal
@testable import VisualizerRendering

/// Coverage 0 on the cumulus deck is a CLEAR sky. The weather field used to add up to 0.58 local
/// coverage on its own, so a host's "Clear" still drew scattered puffs (Daydream DH-1664).
final class CumulusClearSkyTests: XCTestCase {

    @MainActor
    private func dome(coverage: Float, density: Float) -> [Float16] {
        let cloud = VolumetricCloudRenderer(engine: .shared, resolution: SIMD2(128, 64), iblResolution: SIMD2(32, 16))
        var p = VolumetricCloudRenderer.Params()
        p.cloudModel = .cumulus
        p.sunDir = -simd_normalize(SIMD3<Float>(0.4, 0.35, 0.5))
        p.cloudBaseY = 1200
        p.cloudTopY = 2500
        p.horizontalScale = 1.0 / 3600
        p.coverage = coverage
        p.density = density
        p.cloudLightingFromAtmosphere = true
        cloud.render(params: p)
        let fence = SimEngine.shared.commandQueue.makeCommandBuffer()!
        fence.commit(); fence.waitUntilCompleted()   // gpu-ok: test-only drain before reading results
        let t = cloud.outputTexture
        var px = [Float16](repeating: 0, count: t.width * t.height * 4)
        px.withUnsafeMutableBytes {
            t.getBytes($0.baseAddress!, bytesPerRow: t.width * 8, from: MTLRegionMake2D(0, 0, t.width, t.height), mipmapLevel: 0)
        }
        return px
    }

    @MainActor
    func testCoverageZeroDrawsNoCloud() {
        let clear = dome(coverage: 0, density: 0.035)
        let none = dome(coverage: 0, density: 0)
        var worst: Float = 0
        for i in clear.indices { worst = max(worst, abs(Float(clear[i]) - Float(none[i]))) }
        XCTAssertLessThan(worst, 1e-3, "coverage 0 still put cloud in the dome (max channel difference \(worst))")
        // …and the deck is still there once there is coverage to draw (the test can see a cloud).
        let fair = dome(coverage: 0.38, density: 0.035)
        var moved: Float = 0
        for i in fair.indices { moved = max(moved, abs(Float(fair[i]) - Float(none[i]))) }
        XCTAssertGreaterThan(moved, 0.01, "a fair-weather deck must differ from an empty sky")
    }
}
