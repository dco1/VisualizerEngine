import XCTest
import simd
@testable import VisualizerRendering

/// `cloudLightingFromAtmosphere` lights the deck in the atmosphere's own units, on the GPU
/// (`volSkyCloudLight`). The failure it fixes was clouds DARKER than the blue sky behind them.
final class NishitaCloudLightingTests: XCTestCase {
    func testTheFlagIsOffByDefaultAndLegacyPackingIsUnchanged() {
        var p = VolumetricCloudRenderer.Params()
        XCTAssertFalse(p.cloudLightingFromAtmosphere)
        let legacy = SkyUniforms(params: p, time: 0)
        XCTAssertEqual(legacy.skyGrade.z, 0)
        XCTAssertEqual(legacy.skyHorizon, SIMD4(p.skyHorizon, p.hazePower))
        p.cloudLightingFromAtmosphere = true
        let lit = SkyUniforms(params: p, time: 0)
        XCTAssertEqual(lit.skyGrade.z, 1)
        XCTAssertEqual(lit.skyHorizon, SIMD4(p.cloudGroundAlbedo, p.hazePower))
        p.atmosphere = .proceduralGradient
        XCTAssertEqual(SkyUniforms(params: p, time: 0).skyGrade.z, 0, "needs the nishita sky")
        // The host upload stops short of the GPU-written tail.
        XCTAssertEqual(SkyUniforms.hostPrefixLength, MemoryLayout<SkyUniforms>.stride - 32)
    }

    /// Run the real prepass and compare its numbers to the physics: a white Lambertian facing a
    /// 48° sun must be several times brighter than the zenith sky, the sun irradiance mildly
    /// warm and most of `intensity`, the fill bluish and far below the sun.
    @MainActor
    func testGPUPrepassLightsTheDeckInSkyUnits() throws {
        let engine = SimEngine.shared
        let cloud = VolumetricCloudRenderer(engine: engine, resolution: SIMD2(64, 32), iblResolution: SIMD2(32, 16))
        var p = VolumetricCloudRenderer.Params()
        let toSun = simd_normalize(SIMD3<Float>(0.3, sin(48 * Float.pi / 180), 0.6))
        p.sunDir = -toSun
        p.atmosphereIntensity = 20
        p.cloudBaseY = 1300
        p.cloudLightingFromAtmosphere = true
        cloud.render(params: p)
        // Test-only sync: drain the queue so the tail can be inspected.
        let fence = engine.commandQueue.makeCommandBuffer()!
        fence.commit(); fence.waitUntilCompleted()   // gpu-ok: test-only drain before reading results
        let u = cloud.skyUniformsBuffer.contents().bindMemory(to: SkyUniforms.self, capacity: 1).pointee
        let luma = SIMD3<Float>(0.2126, 0.7152, 0.0722)
        let sun = SIMD3<Float>(u.cloudLitSun.x, u.cloudLitSun.y, u.cloudLitSun.z)
        let amb = SIMD3<Float>(u.cloudLitAmbient.x, u.cloudLitAmbient.y, u.cloudLitAmbient.z)
        XCTAssertEqual(u.cloudLitSun.w, 1, "the prepass ran")
        XCTAssertGreaterThan(sun.x, sun.z, "sunlight through air is warm")
        XCTAssertGreaterThan(simd_dot(sun, luma), 20 * 0.6)
        XCTAssertLessThan(simd_dot(sun, luma), 20 * 1.01)
        XCTAssertGreaterThan(amb.z, amb.x, "the fill is sky-blue")
        XCTAssertLessThan(simd_dot(amb, luma), simd_dot(sun, luma) / 4)
        let white = sun * toSun.y / .pi + amb
        XCTAssertGreaterThan(simd_dot(white, luma), 3 * simd_dot(amb, luma) * 0.9)
    }
}
