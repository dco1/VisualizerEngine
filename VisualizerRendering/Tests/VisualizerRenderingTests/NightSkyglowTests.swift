import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// ARTIFICIAL SKYGLOW and the STAR LIMITING MAGNITUDE of the physical night sky
/// (`VolumetricCloudRenderer.Params.artificialSkyglow…` / `starLimitingMagnitude`):
///  • the photometric helpers — mag/arcsec² → the night unit, SQM → naked-eye limiting magnitude,
///    the gradient exponent for a horizon/zenith ratio — against their published anchors;
///  • the dome (the renderer's own buffer, through the MS harness): with the skyglow on, the moonless
///    zenith gains exactly `artificialSkyglow × nightRadiance` of luminance in a warm colour, the
///    horizon is `artificialSkyglowHorizonRatio` × the zenith, and OFF is bit-identical;
///  • the star field (the real header, compiled from source): a limiting magnitude removes the
///    stars fainter than it and leaves the brighter ones exactly as they were.
@MainActor
final class NightSkyglowTests: XCTestCase {

    // ── Photometry ──────────────────────────────────────────────────────────

    func testSkyBrightnessHelpersMatchTheirAnchors() {
        // 1 sr = 4.2545e10 arcsec²: a magnitude-0 source per sr is 26.57 mag/arcsec².
        XCTAssertEqual(NightSkyEphemeris.skyRadiance(magnitudesPerSquareArcsecond: 26.5725), 1, accuracy: 0.01)
        // The header's own anchors: a natural dark zenith (21.8) ≈ 81, a full moon's sky (18.0) ≈ 2690.
        XCTAssertEqual(NightSkyEphemeris.skyRadiance(magnitudesPerSquareArcsecond: 21.8), 81, accuracy: 2)
        XCTAssertEqual(NightSkyEphemeris.skyRadiance(magnitudesPerSquareArcsecond: 18.0), 2690, accuracy: 30)
        // SQM → NELM: ≈ 6.6 under a pristine 22, ≈ 4.0 under a city's 18, monotone.
        XCTAssertEqual(NightSkyEphemeris.nakedEyeLimitingMagnitude(skyMagnitudesPerSquareArcsecond: 22), 6.6, accuracy: 0.1)
        XCTAssertEqual(NightSkyEphemeris.nakedEyeLimitingMagnitude(skyMagnitudesPerSquareArcsecond: 18), 4.0, accuracy: 0.1)
        var prev: Float = -.infinity
        for mu in stride(from: 16.0, through: 22.0, by: 0.25) {
            let m = NightSkyEphemeris.nakedEyeLimitingMagnitude(skyMagnitudesPerSquareArcsecond: mu)
            XCTAssertGreaterThan(m, prev)
            prev = m
        }
        // The gradient exponent reproduces the requested horizon/zenith ratio.
        for r: Float in [1, 2.5, 4, 6] {
            let p = NightSkyEphemeris.skyglowGradientExponent(horizonToZenith: r)
            XCTAssertEqual(pow(1.15 / 0.15, p), r, accuracy: r * 1e-4)
        }
    }

    // ── The dome ────────────────────────────────────────────────────────────

    private static let luma = SIMD3<Float>(0.2126, 0.7152, 0.0722)

    /// A moonless physical night (sun 30° down), airglow and zodiacal light off, so the only night
    /// glow is the skyglow under test.
    private static func night(skyglow: Float, ratio: Float = 4) -> VolumetricCloudRenderer.Params {
        var p = NishitaMultipleScatteringTests.bareSky(el: -30, ms: true)
        p.atmosphereIntensity = 20
        p.nightSkyModel = .physical
        p.moonIntensity = 0
        p.airglow = 0
        p.zodiacalLight = 0
        p.nightRadiance = 2.54e-6 / 1048          // Digital Clock's photometric night unit
        p.artificialSkyglow = skyglow
        p.artificialSkyglowHorizonRatio = ratio
        return p
    }

    func testSkyglowAddsItsZenithRadianceAndHorizonGradient() throws {
        let h = try NishitaMultipleScatteringTests.Harness()
        let S = NightSkyEphemeris.skyRadiance(magnitudesPerSquareArcsecond: 18)
        let up = SIMD3<Float>(0, 1, 0)
        let horizon = simd_normalize(SIMD3<Float>(1, 0.0005, 0.3))
        let off = try h.sky(Self.night(skyglow: 0), [up, horizon])
        let on = try h.sky(Self.night(skyglow: S), [up, horizon])
        let added = simd_dot(on[0] - off[0], Self.luma)
        let want = S * Self.night(skyglow: S).nightRadiance
        XCTAssertEqual(added, want, accuracy: want * 0.02, "zenith: the skyglow's own luminance, added")
        let addedH = simd_dot(on[1] - off[1], Self.luma)
        XCTAssertEqual(addedH / added, 4, accuracy: 0.1, "horizon ÷ zenith = the requested ratio")
        // A warm (≈ 3500 K) glow: red over green over blue.
        let c = on[0] - off[0]
        XCTAssertGreaterThan(c.x, c.y)
        XCTAssertGreaterThan(c.y, c.z)
        // And it dominates the moonless atmosphere at a city's level (it is the night sky there).
        XCTAssertGreaterThan(added, 10 * simd_dot(off[0], Self.luma))
    }

    func testSkyglowOffIsBitIdentical() throws {
        let h = try NishitaMultipleScatteringTests.Harness()
        let dirs = [SIMD3<Float>(0, 1, 0), simd_normalize(SIMD3<Float>(1, 0.2, 0.3)),
                    simd_normalize(SIMD3<Float>(-0.4, 0.02, -1))]
        var a = Self.night(skyglow: 0)
        a.artificialSkyglowHorizonRatio = 4
        var b = a
        b.artificialSkyglowHorizonRatio = 9          // ignored while the glow is off
        b.artificialSkyglowTemperature = 2200
        let x = try h.sky(a, dirs), y = try h.sky(b, dirs)
        for i in 0..<dirs.count { XCTAssertEqual(x[i], y[i], "ray \(i)") }
        // The packing: zero ⇒ the shader's branch never runs.
        XCTAssertEqual(SkyUniforms(params: a, time: 0).nightSkyE.x, 0)
        var legacy = a
        legacy.nightSkyModel = .legacy
        legacy.artificialSkyglow = 2690
        legacy.starLimitingMagnitude = 4
        XCTAssertEqual(SkyUniforms(params: legacy, time: 0).nightSkyE.x, 0, "physical model only")
        XCTAssertEqual(SkyUniforms(params: legacy, time: 0).nightSkyE.w, 0, "physical model only")
    }

    // ── The star field ──────────────────────────────────────────────────────

    private struct GPU {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let library: MTLLibrary
    }

    /// Every lattice star evaluated at its OWN direction (pixel 1e-5 rad, so no neighbour reaches
    /// it), with no limit and with `limit`: (mag, value without limit, value with limit).
    private func starsAtThemselves(limit: Float) throws -> [SIMD3<Float>] {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaNightSky.h")
        let header = try MetalSourceLoader.source(contentsOf: url)
        let lib = try device.makeLibrary(source: header + "\n" + """
        kernel void starsSelf(device float4* out [[buffer(0)]], device atomic_uint* count [[buffer(1)]],
                              constant float& limit [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
            const int N = 99;
            int3 c = int3(int(tid % N), int((tid / N) % N), int(tid / (N * N))) - 49;
            if (fabs(length(float3(c) + 0.5f) - kStarShellR) > 1.5f) return;
            uint ch = nightCellHash(c);
            int n = nightStarCount(c, ch);
            for (int i = 0; i < n; ++i) {
                NightStar s = nightStarInCell(c, ch, i);
                if (!s.valid || s.dir.z < 0.2f) continue;      // world = equatorial; above the horizon
                float a = dot(nightStarsPhysical(s.dir, s.dir.z, 1e-5f, 1.0f, 1e-3f, 0.0f, 0.0f), float3(1));
                float b = dot(nightStarsPhysical(s.dir, s.dir.z, 1e-5f, 1.0f, 1e-3f, 0.0f, 0.0f, limit), float3(1));
                uint k = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
                if (k < 40000u) out[k] = float4(s.mag, a, b, 0);
            }
        }
        """, options: nil)
        guard let fn = lib.makeFunction(name: "starsSelf") else { throw XCTSkip("kernel missing") }
        let pso = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        let cap = 40000
        var lim = limit
        guard let out = device.makeBuffer(length: cap * 16, options: .storageModeShared),
              let count = device.makeBuffer(length: 4, options: .storageModeShared),
              let lb = device.makeBuffer(bytes: &lim, length: 4, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no resources") }
        memset(count.contents(), 0, 4)
        enc.setComputePipelineState(pso)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.setBuffer(count, offset: 0, index: 1)
        enc.setBuffer(lb, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: 99 * 99 * 99, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: test harness
        if let e = cb.error { throw e }
        let n = min(Int(count.contents().load(as: UInt32.self)), cap)
        let p = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: cap)
        return (0..<n).map { SIMD3(p[$0].x, p[$0].y, p[$0].z) }
    }

    func testLimitingMagnitudeRemovesOnlyTheFaintStars() throws {
        let stars = try starsAtThemselves(limit: 4)
        XCTAssertGreaterThan(stars.count, 3000)
        var bright = 0, faint = 0
        for s in stars {
            let mag = s.x, unlimited = s.y, limited = s.z
            if mag < 3.4 {
                XCTAssertEqual(limited, unlimited, "a star brighter than the limit is untouched (m \(mag))")
                bright += 1
            } else if mag > 4.6 {
                XCTAssertEqual(limited, 0, "a star fainter than the limit is gone (m \(mag))")
                XCTAssertGreaterThan(unlimited, 0, "…though the per-pixel washout alone kept it (m \(mag))")
                faint += 1
            }
        }
        XCTAssertGreaterThan(bright, 50)
        XCTAssertGreaterThan(faint, 1000)
        // limit 0 = no limit: identical to the default call.
        for s in try starsAtThemselves(limit: 0) { XCTAssertEqual(s.y, s.z) }
    }
}
