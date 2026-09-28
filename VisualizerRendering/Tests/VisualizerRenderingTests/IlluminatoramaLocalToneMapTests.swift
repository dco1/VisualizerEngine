import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Local tone mapping (`IlluminatoramaLocalToneMapping`, `IlluminatoramaLocalToneMap.metal`) on
/// synthetic HDR images: the REAL kernels, compiled from source (SwiftPM's `swift test` builds no
/// metallib), run through the real `IlluminatoramaLocalToneMapPass`.
///
/// The invariants the operator is sold on:
///  • OFF is free and exact — strength 0 is never encoded;
///  • a flat neighbourhood gets exactly the documented lift `d − f(d)` (so a host can calibrate
///    against measured levels);
///  • MONOTONE — a brighter input never prints darker (a 22-stop log ramp stays ordered);
///  • NO HALOS — across a 13-stop step edge and round a bright disc (an LED on a dark room), the
///    dark side's gain stays within a threshold of its far-field gain right up to the edge;
///  • detail = 1 keeps local texture exactly (the gain is a function of the smooth base only);
///  • the anchor is in the TONEMAP's units — the metered exposure is read from the renderer's
///    ExposureState buffer, not assumed to be 1.
@MainActor
final class IlluminatoramaLocalToneMapTests: XCTestCase {

    // ── Harness ─────────────────────────────────────────────────────────────

    private struct Harness {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let pass: IlluminatoramaLocalToneMapPass
    }

    private func makeHarness() throws -> Harness {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaLocalToneMap.metal")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no shader") }
        let lib = try MetalSourceLoader.makeLibrary(device: device, contentsOf: url)
        let pass = IlluminatoramaLocalToneMapPass(device: device) { name in
            guard let fn = lib.makeFunction(name: name) else { return nil }
            return try? device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        }
        XCTAssertTrue(pass.isAvailable, "every LTM kernel must compile")
        return Harness(device: device, queue: queue, pass: pass)
    }

    /// An rgba16Float image from per-pixel linear RGB.
    private func makeImage(_ h: Harness, width: Int, height: Int,
                           _ rgb: (Int, Int) -> SIMD3<Float>) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width,
                                                         height: height, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .shared
        let t = h.device.makeTexture(descriptor: d)!
        var px = [Float16](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let c = rgb(x, y)
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

    /// Run the pass; returns the output's linear RGB (nil when the pass encoded nothing).
    private func run(_ h: Harness, _ src: MTLTexture, _ s: IlluminatoramaLocalToneMapping,
                     metered: Float = 1, hostExposure: Float = 1, autoExposure: Bool = false)
        -> [SIMD3<Float>]? {
        let expo = h.device.makeBuffer(length: 16, options: .storageModeShared)!
        expo.contents().storeBytes(of: SIMD4<Float>(0, metered, 0, 1 / 60), as: SIMD4<Float>.self)
        let cb = h.queue.makeCommandBuffer()!
        guard let out = h.pass.encode(cb, source: src, exposureBuffer: expo, hostExposure: hostExposure,
                                      autoExposure: autoExposure, settings: s,
                                      makeEncoder: { _ in cb.makeComputeCommandEncoder() }) else {
            cb.commit()
            return nil
        }
        let w = out.width, hgt = out.height
        let buf = h.device.makeBuffer(length: w * hgt * 8, options: .storageModeShared)!
        let blit = cb.makeBlitCommandEncoder()!
        blit.copy(from: out, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: hgt, depth: 1), to: buf, destinationOffset: 0,
                  destinationBytesPerRow: w * 8, destinationBytesPerImage: w * hgt * 8)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: test harness
        XCTAssertNil(cb.error, "LTM command buffer failed: \(String(describing: cb.error))")
        let p = buf.contents().bindMemory(to: Float16.self, capacity: w * hgt * 4)
        return (0..<(w * hgt)).map { i in SIMD3(Float(p[i * 4]), Float(p[i * 4 + 1]), Float(p[i * 4 + 2])) }
    }

    /// The operator's own brightness metric: max(luma, ½·max channel).
    private func brightness(_ c: SIMD3<Float>) -> Float {
        max(0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z, 0.5 * max(c.x, max(c.y, c.z)))
    }

    /// A grey whose metric brightness is 2^stops.
    private func grey(_ stops: Float) -> SIMD3<Float> { SIMD3(repeating: pow(2, stops)) }

    private let W = 384, H = 256
    /// A neighbourhood of 0.1 × 256 ≈ 26 px (cells of 3 × 9 = 27 px).
    private var settings: IlluminatoramaLocalToneMapping {
        IlluminatoramaLocalToneMapping(strength: 0.7, radius: 0.1, edgeStops: 1, detail: 1,
                                       anchor: 0.18, knee: 2, maxLift: 10)
    }

    // ── OFF is free and exact ───────────────────────────────────────────────

    func testStrengthZeroIsNotEncodedAndLiftsNothing() throws {
        let h = try makeHarness()
        let src = makeImage(h, width: 64, height: 32) { _, _ in SIMD3(0.001, 0.002, 0.003) }
        let off = IlluminatoramaLocalToneMapping()
        XCTAssertFalse(off.isEnabled)
        XCTAssertNil(run(h, src, off), "strength 0 must not encode a pass (byte-identical frames)")
        for d: Float in [0, 1, 5, 12, 20] { XCTAssertEqual(off.lift(stopsUnderAnchor: d), 0) }
    }

    // ── a flat neighbourhood gets exactly the documented lift ───────────────

    func testFlatFieldLiftMatchesTheCurve() throws {
        let h = try makeHarness()
        let s = settings
        let anchorStops = log2(s.anchor)
        for level: Float in [-1, -3, -6, -10, -14, -18] {
            let src = makeImage(h, width: 128, height: 96) { _, _ in grey(level) }
            let out = try XCTUnwrap(run(h, src, s))
            let gain = log2(brightness(out[48 * 128 + 64]) / brightness(grey(level)))
            let want = s.lift(stopsUnderAnchor: anchorStops - level)
            XCTAssertEqual(gain, want, accuracy: 0.02, "flat field at \(level) stops")
        }
        // Above the anchor: untouched.
        XCTAssertEqual(s.lift(stopsUnderAnchor: -1), 0)
    }

    // ── monotone ────────────────────────────────────────────────────────────

    /// A 16-stop ramp over fp16's NORMAL range (−14 … +2 stops; 0.031 stops a pixel ≫ fp16's
    /// 0.0014-stop quantum). Below −14 stops fp16 goes subnormal and a smooth ramp quantises into
    /// flat plateaus up to 0.09 stops wide; along a plateau the input is constant while the
    /// neighbourhood keeps brightening, so a LOCAL operator lifts its right end a hair less than its
    /// left (measured −0.004 stops over a 2-px plateau at −19 stops) — equal inputs, not an inversion.
    func testLogRampStaysMonotone() throws {
        let h = try makeHarness()
        let w = 512, hgt = 64
        let src = makeImage(h, width: w, height: hgt) { x, _ in grey(-14 + 16 * Float(x) / Float(w - 1)) }
        let out = try XCTUnwrap(run(h, src, settings))
        var worst: Float = 0
        for y in stride(from: 0, to: hgt, by: 7) {
            for x in 1..<w {
                let a = log2(brightness(out[y * w + x - 1])), b = log2(brightness(out[y * w + x]))
                worst = min(worst, b - a)
            }
        }
        // fp16 quantisation of the output alone is ~0.0014 stops; anything worse is a reversal.
        XCTAssertGreaterThan(worst, -0.003, "a brighter input printed darker by \(-worst) stops")
        // …and it actually compressed: 22 stops in, fewer out.
        let span = log2(brightness(out[w - 1])) - log2(brightness(out[0]))
        XCTAssertLessThan(span, 16 - 4, "ramp span out \(span) stops")
    }

    // ── no halos ────────────────────────────────────────────────────────────

    /// The dark half of a 13-stop step (−14 | −1): every dark column's gain within `haloLimit`
    /// of the far-field gain, right up to the edge; the bright half (above the anchor) untouched.
    func testNoHaloAcrossAStepEdge() throws {
        let h = try makeHarness()
        let edge = W / 2 + 5          // deliberately off the cell grid
        let src = makeImage(h, width: W, height: H) { x, _ in grey(x < edge ? -14 : -1) }
        let out = try XCTUnwrap(run(h, src, settings))
        let y = H / 2
        func gain(_ x: Int) -> Float { log2(brightness(out[y * W + x]) / brightness(grey(x < edge ? -14 : -1))) }
        let far = gain(8)
        XCTAssertGreaterThan(far, 5, "the dark field must actually be lifted (got \(far) stops)")
        var worst: Float = 0
        for x in 0..<edge { worst = max(worst, abs(gain(x) - far)) }
        XCTAssertLessThan(worst, Self.haloLimit, "dark-side halo \(worst) stops at the step")
        var bright: Float = 0
        for x in edge..<W { bright = max(bright, abs(gain(x))) }
        XCTAssertLessThan(bright, Self.haloLimit, "bright side moved \(bright) stops")
        print("LTM step edge: far-field lift \(far) stops, worst dark-side deviation \(worst), bright \(bright)")
    }

    /// An LED-like bright disc (r = 12 px, −1 stops) on a dark field (−15): the ring of dark
    /// pixels round it gets the same lift as the far field.
    func testNoHaloRoundABrightDisc() throws {
        let h = try makeHarness()
        let cx = Float(W) * 0.5 + 3.3, cy = Float(H) * 0.5 - 2.7, r: Float = 12
        let src = makeImage(h, width: W, height: H) { x, y in
            let inside = simd_length(SIMD2(Float(x) + 0.5 - cx, Float(y) + 0.5 - cy)) < r
            return grey(inside ? -1 : -15)
        }
        let out = try XCTUnwrap(run(h, src, settings))
        let far = log2(brightness(out[10 * W + 10]) / brightness(grey(-15)))
        var worst: Float = 0
        for y in 0..<H {
            for x in 0..<W {
                let dist = simd_length(SIMD2(Float(x) + 0.5 - cx, Float(y) + 0.5 - cy))
                guard dist >= r + 1 else { continue }     // dark pixels, from 1 px outside the rim
                let g = log2(brightness(out[y * W + x]) / brightness(grey(-15)))
                worst = max(worst, abs(g - far))
            }
        }
        XCTAssertLessThan(worst, Self.haloLimit, "halo round the disc: \(worst) stops")
        print("LTM disc: far-field lift \(far) stops, worst ring deviation \(worst)")
    }

    /// The halo probe is not vacuous: with the range kernel opened up to 12 stops the grid is a
    /// plain spatial blur, and the same step then shows the classic dark ring on the dim side.
    func testHaloProbeSeesAPlainBlurHalo() throws {
        let h = try makeHarness()
        let edge = W / 2 + 5
        let src = makeImage(h, width: W, height: H) { x, _ in grey(x < edge ? -14 : -1) }
        var blur = settings
        blur.edgeStops = 12
        let out = try XCTUnwrap(run(h, src, blur))
        let y = H / 2
        func gain(_ x: Int) -> Float { log2(brightness(out[y * W + x]) / brightness(grey(-14))) }
        let halo = gain(8) - gain(edge - 2)
        XCTAssertGreaterThan(halo, 1, "a non-edge-aware base must darken the dim side at the edge")
        print("LTM plain-blur control: dim-side halo \(halo) stops at the edge")
    }

    /// A visible halo is ~0.1 stop (7 %) on a smooth dark field.
    private static let haloLimit: Float = 0.1

    // ── detail = 1 keeps texture ────────────────────────────────────────────

    /// A ±0.5-stop checker (4 px) at −12 stops — texture as strong as `edgeStops` (1), the worst
    /// case: the bilateral base takes in ≈ 5 % of it (a pixel slices the grid at its own level), so
    /// the texture's 1-stop swing comes out within 0.05 stops. Weaker texture is kept closer still.
    func testDetailOneKeepsLocalContrast() throws {
        let h = try makeHarness()
        let src = makeImage(h, width: W, height: H) { x, y in grey(-12 + (((x / 4) + (y / 4)) % 2 == 0 ? 0.5 : -0.5)) }
        let out = try XCTUnwrap(run(h, src, settings))
        let y = H / 2
        var maxDev: Float = 0
        for x in stride(from: 64, to: W - 64, by: 4) {
            let a = log2(brightness(out[y * W + x])), b = log2(brightness(out[y * W + x + 4]))
            maxDev = max(maxDev, abs(abs(a - b) - 1))
        }
        XCTAssertLessThan(maxDev, 0.05, "checker amplitude changed by \(maxDev) stops")
        print("LTM checker: 1-stop texture swing changed by \(maxDev) stops")
    }

    // ── hue preserved ───────────────────────────────────────────────────────

    func testGainIsHuePreserving() throws {
        let h = try makeHarness()
        let blue = SIMD3<Float>(0.02, 0.05, 0.4) * pow(2, -13)
        let src = makeImage(h, width: 128, height: 96) { _, _ in blue }
        let out = try XCTUnwrap(run(h, src, settings))[48 * 128 + 64]
        XCTAssertGreaterThan(out.z / blue.z, 8, "the blue field must be lifted")
        XCTAssertEqual(out.x / out.z, blue.x / blue.z, accuracy: 0.002)
        XCTAssertEqual(out.y / out.z, blue.y / blue.z, accuracy: 0.002)
    }

    // ── the anchor is in the tonemap's exposed units ───────────────────────

    func testAnchorRidesTheMeteredExposure() throws {
        let h = try makeHarness()
        let s = settings
        let src = makeImage(h, width: 128, height: 96) { _, _ in grey(-10) }
        // Metered ×4 (+2 EV) and host ×0.5 (−1 EV): the field is exposed at −9 stops.
        let out = try XCTUnwrap(run(h, src, s, metered: 4, hostExposure: 0.5, autoExposure: true))
        let gain = log2(brightness(out[48 * 128 + 64]) / brightness(grey(-10)))
        XCTAssertEqual(gain, s.lift(stopsUnderAnchor: log2(s.anchor) + 9), accuracy: 0.02)
        // Auto-exposure off: the metered value is ignored.
        let out2 = try XCTUnwrap(run(h, src, s, metered: 4, hostExposure: 0.5, autoExposure: false))
        let gain2 = log2(brightness(out2[48 * 128 + 64]) / brightness(grey(-10)))
        XCTAssertEqual(gain2, s.lift(stopsUnderAnchor: log2(s.anchor) + 11), accuracy: 0.02)
    }

    // ── the curve itself ────────────────────────────────────────────────────

    func testCurveIsMonotoneKneedAndClamped() {
        let s = IlluminatoramaLocalToneMapping(strength: 0.6, knee: 3, maxLift: 7)
        var prevOut: Float = .infinity
        var prevLift: Float = -.infinity
        for i in 0...600 {
            let d = Float(i) * 0.05
            let lift = s.lift(stopsUnderAnchor: d)
            let out = -d + lift                                // output base, relative to the anchor
            XCTAssertLessThanOrEqual(out, prevOut + 1e-5, "output base must fall as input falls")
            XCTAssertGreaterThanOrEqual(lift, prevLift - 1e-5, "the lift never shrinks going darker")
            prevOut = out
            prevLift = lift
            XCTAssertLessThanOrEqual(lift, 7 + 1e-5, "the ceiling is never exceeded")
        }
        // The ceiling is reached, smoothly: 30 stops down the lift is within 1 % of it.
        XCTAssertEqual(s.lift(stopsUnderAnchor: 30), 7, accuracy: 0.07)
        // Within the knee contrast is nearly kept (slope ≈ 1 at the anchor).
        XCTAssertLessThan(s.lift(stopsUnderAnchor: 0.5), 0.05)
        // Far below (ceiling out of the way), the slope is 1 − strength.
        var open = s
        open.maxLift = 40
        let slope = (open.lift(stopsUnderAnchor: 14) - open.lift(stopsUnderAnchor: 12)) / 2
        XCTAssertEqual(slope, 0.6, accuracy: 0.02)
    }

    // ── the second segment (floorLevel) ─────────────────────────────────────

    /// A night frame's two dim levels — a window 12 stops under the anchor and the room it lights
    /// 6 stops under THAT — printed through one far-tail slope (0.1) collapse to ~1 stop apart;
    /// with the floor at the window's level they keep `floorSlope` of their 6 stops. Measured on
    /// the real kernel as flat fields (exposure 2⁻¹⁰, so no input sits in fp16 subnormals).
    func testFloorSegmentRestoresContrastBelowTheFloor() throws {
        let h = try makeHarness()
        let hostExposure: Float = pow(2, -10)
        var s = IlluminatoramaLocalToneMapping(strength: 0.9, radius: 0.1, edgeStops: 1, detail: 1,
                                               anchor: 0.18, knee: 2.5, maxLift: 16)
        let anchorStops = log2(s.anchor)
        // Scene levels: exposed = scene × 2⁻¹⁰.
        let windowExposed = anchorStops - 12, roomExposed = windowExposed - 6
        func printed(_ exposedStops: Float, _ st: IlluminatoramaLocalToneMapping) throws -> Float {
            let scene = exposedStops + 10
            let src = makeImage(h, width: 128, height: 96) { _, _ in grey(scene) }
            let out = try XCTUnwrap(run(h, src, st, hostExposure: hostExposure))
            return log2(brightness(out[48 * 128 + 64]) * hostExposure)
        }
        let oneSlope = try printed(windowExposed, s) - printed(roomExposed, s)
        XCTAssertLessThan(oneSlope, 1.2, "one far-tail slope of 0.1 squashes 6 stops to ~0.6")
        s.floorLevel = pow(2, windowExposed + 10)     // scene-referred: the window's own level
        s.floorSlope = 0.5
        let window = try printed(windowExposed, s), room = try printed(roomExposed, s)
        XCTAssertGreaterThan(window - room, 2.5, "below the floor the 6 stops print at ~slope 0.5")
        // The kernel draws the documented curve (the Swift mirror, with the floor's own stops).
        let dF = anchorStops - windowExposed
        for e: Float in [windowExposed + 3, windowExposed, windowExposed - 3, roomExposed] {
            let got = try printed(e, s) - e
            let want = s.lift(stopsUnderAnchor: anchorStops - e, floorStopsUnderAnchor: dF)
            XCTAssertEqual(got, want, accuracy: 0.03, "flat field at \(e) exposed stops")
        }
    }

    func testFloorSegmentKeepsTheCurveMonotoneAndOffIsExact() {
        var s = IlluminatoramaLocalToneMapping(strength: 0.9, knee: 2.5, maxLift: 14)
        let base = s
        s.floorLevel = 1e-5
        s.floorSlope = 0.55
        var prevOut: Float = .infinity
        for i in 0...600 {
            let d = Float(i) * 0.05
            let out = -d + s.lift(stopsUnderAnchor: d, floorStopsUnderAnchor: 12)
            XCTAssertLessThan(out, prevOut + 1e-5, "output must fall as input falls (d = \(d))")
            prevOut = out
            XCTAssertLessThanOrEqual(s.lift(stopsUnderAnchor: d, floorStopsUnderAnchor: 12), 14 + 1e-5)
            // Without a floor stop (nil) the curve is the original one exactly.
            XCTAssertEqual(s.lift(stopsUnderAnchor: d), base.lift(stopsUnderAnchor: d))
        }
        // Well below the floor the output slope is floorSlope (ceiling out of the way).
        var open = s
        open.maxLift = 60
        let slope = 1 - (open.lift(stopsUnderAnchor: 20, floorStopsUnderAnchor: 8)
                         - open.lift(stopsUnderAnchor: 16, floorStopsUnderAnchor: 8)) / 4
        XCTAssertEqual(slope, 0.55, accuracy: 0.02)
    }

    // ── mesopic vision ──────────────────────────────────────────────────────

    /// CIE 191 on the kernel: a red field at photopic luminance keeps its colour; the same red at
    /// ~0.01 cd/m² loses most of its chroma and falls toward its rod response (dark for red); a
    /// blue-grey at the same level keeps its luminance (rods favour blue). Off ⇒ exact.
    func testMesopicFollowsAbsoluteLuminance() throws {
        let h = try makeHarness()
        let red = SIMD3<Float>(0.5, 0.0175, 0.006)       // the LED's chromaticity
        let blueGrey = SIMD3<Float>(0.30, 0.34, 0.42)
        // strength small so the gain barely moves the level; nits per unit chosen per case.
        func out(_ c: SIMD3<Float>, nitsPerUnit: Float, mesopic: Float) throws -> SIMD3<Float> {
            var s = IlluminatoramaLocalToneMapping(strength: 0.01, radius: 0.1, edgeStops: 1, detail: 1,
                                                   anchor: 0.18, knee: 2, maxLift: 1)
            s.mesopic = mesopic
            s.mesopicNitsPerUnit = nitsPerUnit
            let src = makeImage(h, width: 96, height: 64) { _, _ in c }
            let o = try XCTUnwrap(run(h, src, s))
            let base = try XCTUnwrap(run(h, src, { var t = s; t.mesopic = 0; return t }()))
            // Divide the (tiny) LTM gain back out: compare the mesopic colour with the input.
            let g = brightness(base[32 * 96 + 48]) / brightness(c)
            return o[32 * 96 + 48] / g
        }
        func luma(_ c: SIMD3<Float>) -> Float { 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }
        func chroma(_ c: SIMD3<Float>) -> Float { (max(c.x, max(c.y, c.z)) - min(c.x, min(c.y, c.z))) / max(c.x, max(c.y, c.z)) }
        // Red at luma × nits ≈ 20 cd/m² — photopic: unchanged (m = 1).
        let bright = try out(red, nitsPerUnit: 20 / luma(red), mesopic: 1)
        XCTAssertEqual(bright.x, red.x, accuracy: red.x * 0.01)
        XCTAssertEqual(chroma(bright), chroma(red), accuracy: 0.01)
        // Red at ≈ 0.003 cd/m² — scotopic: grey-ish and much darker (rods barely see red).
        let dim = try out(red, nitsPerUnit: 0.003 / luma(red), mesopic: 1)
        XCTAssertLessThan(chroma(dim), 0.2 * chroma(red) + 0.01, "chroma gone at scotopic levels")
        XCTAssertLessThan(luma(dim), 0.3 * luma(red), "a red light is dark to rods")
        // Blue-grey at the same dim level keeps (or gains) its luminance.
        let bg = try out(blueGrey, nitsPerUnit: 0.003 / luma(blueGrey), mesopic: 1)
        XCTAssertGreaterThan(luma(bg), 0.9 * luma(blueGrey))
        // Off is exact.
        let off = try out(red, nitsPerUnit: 0.003 / luma(red), mesopic: 0)
        let red16 = SIMD3<Float>(Float(Float16(red.x)), Float(Float16(red.y)), Float(Float16(red.z)))
        XCTAssertEqual(off, red16, "mesopic 0 leaves the (fp16) input exactly as it was")
    }
}
