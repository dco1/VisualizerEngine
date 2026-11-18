import XCTest
import simd
import Metal
@testable import VisualizerRendering

/// DH-1708 — a dome bake spread over `domeBakeStrips` render calls writes the same texels as the
/// one-dispatch bake (the `…Tile` kernels are the same bodies with a row offset), and costs one strip
/// per call, not the whole dome.
final class TiledDomeBakeTests: XCTestCase {

    @MainActor
    private func params(_ cumulus: Bool, ms: Bool) -> VolumetricCloudRenderer.Params {
        var p = VolumetricCloudRenderer.Params()
        p.sunDir = -simd_normalize(SIMD3<Float>(0.4, 0.35, 0.5))
        p.coverage = 0.4
        if cumulus {
            p.cloudModel = .cumulus; p.cloudBaseY = 1200; p.cloudTopY = 2500
            p.horizontalScale = 1.0 / 3600; p.density = 0.035; p.cloudLightingFromAtmosphere = true
        }
        if ms { p.atmosphere = .nishita; p.atmosphereMultipleScattering = true }
        return p
    }

    @MainActor
    private func read(_ cloud: VolumetricCloudRenderer) -> [Float16] {
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
    private func check(cumulus: Bool, ms: Bool) throws {
        let p = params(cumulus, ms: ms)
        let whole = VolumetricCloudRenderer(engine: .shared, resolution: SIMD2(128, 64), iblResolution: SIMD2(32, 16))
        whole.minRenderInterval = 0
        whole.render(params: p)
        let want = read(whole)

        let tiled = VolumetricCloudRenderer(engine: .shared, resolution: SIMD2(128, 64), iblResolution: SIMD2(32, 16))
        tiled.minRenderInterval = 0
        tiled.domeBakeStrips = 8
        // The strip kernel compiles in the background; give it until it is ready (a nil sends the call to the whole bake).
        let deadline = Date().addingTimeInterval(20)
        tiled.prewarmTiledBake(params: p)
        while Date() < deadline {
            RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.02))
            tiled.prewarmTiledBake(params: p)
            if tiled.tiledBakeReady(for: p) { break }
        }
        try XCTSkipUnless(tiled.tiledBakeReady(for: p), "tile kernel never became ready")
        // One call per strip: after the first call only the top 8 rows are baked.
        tiled.render(params: p)
        let first = read(tiled)
        let rowsDone = (0..<64).filter { y in (0..<128 * 4).contains { first[y * 128 * 4 + $0] != 0 } }.count
        XCTAssertEqual(rowsDone, 8, "the first call must bake exactly one strip, got \(rowsDone) rows")
        for _ in 0..<7 { tiled.render(params: p); _ = read(tiled) }   // a strip is skipped while both bake slots are busy
        let got = read(tiled)
        var worst: Float = 0
        for i in got.indices { worst = max(worst, abs(Float(got[i]) - Float(want[i]))) }
        XCTAssertEqual(worst, 0, accuracy: 1e-3, "tiled dome differs from the one-dispatch dome (max channel difference \(worst))")
    }

    @MainActor func testTiledEqualsWholeBuiltinSky() throws { try check(cumulus: false, ms: false) }
    @MainActor func testTiledEqualsWholeMultipleScatteringCumulus() throws { try check(cumulus: true, ms: true) }
}
