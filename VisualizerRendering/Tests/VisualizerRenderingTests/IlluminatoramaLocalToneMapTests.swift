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

    // ── Round 3: adaptation from the frame ─────────────────────────────────

    /// A frame exposed within the tolerance of the key is copied EXACTLY (no lift, no mesopic);
    /// an under-exposed one is lifted, with the lift bounded by gain × (deficit − tolerance).
    func testAdaptationFromFrameIsExactWithinToleranceAndBoundedByTheDeficit() throws {
        let h = try makeHarness()
        var s = settings
        s.maxLift = 14
        s.adaptationFromFrame = true
        s.adaptationTargetEV = -2.2
        s.adaptationTolerance = 1
        s.mesopic = 1
        s.mesopicNitsPerUnit = 1e-3            // deep mesopic — would shift colour if it ran
        // Half the frame at −1.5, half at −4 (log-mean −2.75): 0.55 stop under the key < tolerance.
        let lit = makeImage(h, width: 128, height: 96) { x, _ in x < 64 ? grey(-1.5) : SIMD3(0.01, 0.02, 0.06) * 0.0625 / 0.0197 }
        let out = try XCTUnwrap(run(h, lit, s))
        let ref = (0..<(128 * 96)).map { i -> SIMD3<Float> in
            let x = i % 128
            let c = x < 64 ? grey(-1.5) : SIMD3<Float>(0.01, 0.02, 0.06) * 0.0625 / 0.0197
            return SIMD3(Float(Float16(c.x)), Float(Float16(c.y)), Float(Float16(c.z)))
        }
        XCTAssertEqual(out, ref, "within tolerance the pass is an exact copy")
        XCTAssertEqual(h.pass.lastLiftCeiling, 0, "within tolerance the host sees a zero ceiling")
        // A uniformly dark frame at −12, tolerance 5: ceiling 9.8 − 5 = 4.8 stops, under the
        // curve's own 5.28 ⇒ the deficit is what binds (through the 1-stop soft ceiling).
        var t = s; t.mesopic = 0; t.adaptationTolerance = 5
        let dark = makeImage(h, width: 128, height: 96) { _, _ in grey(-12) }
        let o2 = try XCTUnwrap(run(h, dark, t))
        let gain = log2(brightness(o2[48 * 128 + 64]) / brightness(grey(-12)))
        let curve = t.lift(stopsUnderAnchor: log2(t.anchor) + 12)
        XCTAssertGreaterThan(curve, 5, "precondition: the curve alone would lift past the ceiling")
        let soft = curve - log(1 + exp(curve - 4.8))
        XCTAssertEqual(gain, soft, accuracy: 0.05, "lift = the curve under the deficit's soft ceiling")
        // With the default tolerance the same frame is lifted by the curve (ceiling 8.8 > 5.28).
        var u = t; u.adaptationTolerance = 1
        let o3 = try XCTUnwrap(run(h, dark, u))
        let g3 = log2(brightness(o3[48 * 128 + 64]) / brightness(grey(-12)))
        XCTAssertEqual(g3, curve - log(1 + exp(curve - 8.8)), accuracy: 0.05)
        // The host reads that same adaptation state back (a shared-buffer load, no wait) — what a
        // scene keys its print grade on instead of a second ramp.
        XCTAssertEqual(h.pass.lastLiftCeiling, 8.8, accuracy: 0.02, "host-visible ceiling = gain·(deficit − tolerance)")
        // A PHOTOPIC field (1 unit = 1e6 cd/m²: the −12 frame is ~244 cd/m²) with the photopic
        // level at 10 cd/m²: +4.6 stops of tolerance ⇒ ceiling 4.2 < the curve's 5.28 — and at
        // 1e9 cd/m²/unit the field is so bright the pass is an exact copy.
        var v = u; v.adaptationPhotopicNits = 10; v.mesopicNitsPerUnit = 1e6
        let o4 = try XCTUnwrap(run(h, dark, v))
        let g4 = log2(brightness(o4[48 * 128 + 64]) / brightness(grey(-12)))
        XCTAssertLessThan(g4, g3 - 0.8, "photopic tolerance lowers the ceiling: \(g4) vs \(g3)")
        v.mesopicNitsPerUnit = 1e9
        let o5 = try XCTUnwrap(run(h, dark, v))
        XCTAssertEqual(o5[48 * 128 + 64], SIMD3(repeating: Float(Float16(Float(0.000244140625)))), "a bright field is left as printed")
    }

    // ── Round 3: histogram-adjusted tail ───────────────────────────────────

    /// A bright key region, an EMPTY 8-stop gap, then two populated dim regions 2.6 stops apart
    /// (a sky and a facade): the fixed tail squashes their difference to ~0.1 stop; the histogram
    /// tail keeps most of it (populated bins keep slope ~1) and compresses the empty gap instead.
    func testHistogramTailKeepsPopulatedContrastAndCompressesGaps() throws {
        let h = try makeHarness()
        let w = 192, hgt = 96
        let src = makeImage(h, width: w, height: hgt) { x, _ in
            x < 64 ? grey(-1) : (x < 128 ? grey(-12) : grey(-14.6))
        }
        var fixed = settings
        fixed.strength = 0.95
        fixed.maxLift = 14
        var hist = fixed
        hist.histogramTail = true
        hist.gapSlope = 0.05
        hist.populatedFraction = 0.05
        func diff(_ o: [SIMD3<Float>]) -> Float {
            log2(brightness(o[48 * w + 96]) / brightness(o[48 * w + 170]))
        }
        let a = diff(try XCTUnwrap(run(h, src, fixed)))
        let b = diff(try XCTUnwrap(run(h, src, hist)))
        XCTAssertLessThan(a, 0.5, "the fixed tail squashes the two dim regions: \(a)")
        XCTAssertGreaterThan(b, 1.2, "the histogram tail keeps their contrast: \(b)")
        // …and still lifts them (the gap is what got compressed).
        let o = try XCTUnwrap(run(h, src, hist))
        XCTAssertGreaterThan(log2(brightness(o[48 * w + 96]) / brightness(grey(-12))), 4)
    }

    /// The fit lands the floor luminance on the requested print level.
    func testHistogramTailFitsTheFloorLuminanceOntoThePrintLevel() throws {
        let h = try makeHarness()
        let w = 192, hgt = 96
        // Populated everywhere from −1 down to −20 (a ramp): the fit has to squeeze.
        let src = makeImage(h, width: w, height: hgt) { x, _ in grey(-1 - 19 * Float(x) / Float(w - 1)) }
        var s = settings
        s.strength = 0.95
        s.maxLift = 20
        s.histogramTail = true
        s.populatedFraction = 0.01
        s.minPopulatedScale = 0.02
        s.mesopicNitsPerUnit = 1          // 1 frame unit = 1 cd/m²
        s.printFloorNits = pow(2, -20)    // the ramp's dark end
        s.printFloorLevel = 0.18 / 16     // lands 4 stops under the anchor
        let o = try XCTUnwrap(run(h, src, s))
        let end = log2(brightness(o[48 * w + w - 1]))
        XCTAssertEqual(end, log2(0.18 / 16), accuracy: 0.6, "the floor prints where it was aimed: \(end)")
        // Monotone along the ramp.
        var worst: Float = 0
        for x in 1..<w { worst = min(worst, log2(brightness(o[48 * w + x - 1])) - log2(brightness(o[48 * w + x]))) }
        XCTAssertGreaterThan(worst, -0.01)
    }

    // ── Round 3: blue-shift mesopic + pixel floor ──────────────────────────

    /// Jensen et al. 2000: a neutral grey at scotopic luminance goes to the blue-shift tint; at
    /// photopic luminance it is untouched; the red LED colour at scotopic luminance goes dark.
    func testBlueShiftMesopicFollowsLogLuminance() throws {
        let h = try makeHarness()
        func out(_ c: SIMD3<Float>, nits: Float) throws -> SIMD3<Float> {
            var s = IlluminatoramaLocalToneMapping(strength: 0.01, radius: 0.1, edgeStops: 1, detail: 1,
                                                   anchor: 0.18, knee: 2, maxLift: 1)
            s.mesopic = 1
            s.mesopicModel = .blueShift
            s.mesopicTint = IlluminatoramaLocalToneMapping.jensenBlueShiftTint
            let Y = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
            s.mesopicNitsPerUnit = nits / Y
            let src = makeImage(h, width: 96, height: 64) { _, _ in c }
            let o = try XCTUnwrap(run(h, src, s))
            let base = try XCTUnwrap(run(h, src, { var t = s; t.mesopic = 0; return t }()))
            return o[32 * 96 + 48] / (brightness(base[32 * 96 + 48]) / brightness(c))
        }
        let grey = SIMD3<Float>(0.2, 0.2, 0.2)
        let photopic = try out(grey, nits: 30)
        XCTAssertEqual(photopic.x, 0.2, accuracy: 0.003); XCTAssertEqual(photopic.z, 0.2, accuracy: 0.003)
        let scotopic = try out(grey, nits: 0.001)
        XCTAssertGreaterThan(scotopic.z / scotopic.x, 2.4, "rod vision reads blue: \(scotopic)")
        let red = try out(SIMD3(0.5, 0.0175, 0.006), nits: 0.001)
        XCTAssertLessThan(0.2126 * red.x + 0.7152 * red.y + 0.0722 * red.z, 0.3 * 0.1316, "a red light goes dark")
    }

    /// Fix for the blue rim round a lit patch: with `mesopicPixelFloor` a pixel bright enough to be
    /// photopic is never rod-tinted by a dark neighbourhood's base.
    func testMesopicPixelFloorRemovesTheRimTint() throws {
        let h = try makeHarness()
        let w = 128, hgt = 64
        // A bright patch (1 cd/m²-ish) beside a very dark floor.
        let src = makeImage(h, width: w, height: hgt) { x, _ in x < 64 ? SIMD3(0.3, 0.29, 0.28) : SIMD3(0.0003, 0.0003, 0.0003) }
        var s = IlluminatoramaLocalToneMapping(strength: 0.01, radius: 0.1, edgeStops: 1, detail: 1,
                                               anchor: 0.18, knee: 2, maxLift: 1)
        s.mesopic = 1
        s.mesopicModel = .blueShift
        s.mesopicTint = IlluminatoramaLocalToneMapping.jensenBlueShiftTint
        s.mesopicNitsPerUnit = 10         // patch ≈ 2.9 cd/m², floor ≈ 0.003
        func rimBR(_ floorOn: Bool) throws -> Float {
            var t = s; t.mesopicPixelFloor = floorOn
            let o = try XCTUnwrap(run(h, src, t))
            // The patch's own pixels next to the edge: B − R relative to their level.
            let c = o[32 * w + 62]
            return (c.z - c.x) / max(c.x, 1e-6)
        }
        let without = try rimBR(false), with = try rimBR(true)
        XCTAssertLessThan(with, without + 1e-4)
        XCTAssertLessThan(with, 0.02, "a photopic pixel stays its own colour at the rim: \(with)")
    }

    // ── Round 4: the mesopic print is monotone across a red light's edge (VZ-0194) ─────────

    /// A red LED's light falling off across a surface also lit by dim blue moonlight: radiance
    /// rises steadily across the edge (ambient + t·red, t exponential), so the PRINT must too. The
    /// old rod response (Larson et al. 1997's V = Y·[1.33(1 + (Y+Z)/X) − 1.68]) is not additive —
    /// adding red raises X and the mixture's rod luminance falls BELOW the moonlight's own — so
    /// where a pixel was already red-dominated but still mesopic it printed near-black (Digital
    /// Clock 23:47: (21,26,40) · (2,2,4) · (31,6,11)). `mesopicRodModel = .additive` sums per-primary
    /// rod responses (Larson's own at each primary), and the seam is gone.
    private func mesopicRamp(_ model: IlluminatoramaLocalToneMapping.MesopicRodModel, floor: Bool = true) throws -> [Float] {
        let h = try makeHarness()
        let w = 256, hgt = 32
        let ambient = SIMD3<Float>(0.0045, 0.006, 0.010)          // moonlit blue-grey, ≈ 0.006 cd/m²
        let red = SIMD3<Float>(1, 0.035, 0.012)                    // the LED red
        let src = makeImage(h, width: w, height: hgt) { x, _ in
            ambient + red * (1e-3 * pow(10, 4 * Float(x) / Float(w - 1)))
        }
        var s = IlluminatoramaLocalToneMapping(strength: 0.01, radius: 0.1, edgeStops: 1, detail: 1,
                                               anchor: 0.18, knee: 2, maxLift: 1)
        s.mesopic = 1
        s.mesopicModel = .blueShift
        s.mesopicTint = IlluminatoramaLocalToneMapping.jensenBlueShiftTint
        s.mesopicLogRange = SIMD2(-2, 0.6)
        s.mesopicPixelFloor = floor
        s.mesopicNitsPerUnit = 1
        s.mesopicRodModel = model
        let o = try XCTUnwrap(run(h, src, s))
        return (0..<w).map { x in let c = o[16 * w + x]; return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }
    }

    /// Largest drop in printed luma along the ramp, relative to the level before it (0 = monotone),
    /// and whether any pixel prints darker than both its neighbours by more than fp16 noise.
    private func worstDip(_ luma: [Float]) -> (drop: Float, pit: Bool) {
        var drop: Float = 0, pit = false
        var peak: Float = 0
        for (i, v) in luma.enumerated() {
            peak = max(peak, v)
            drop = max(drop, (peak - v) / max(peak, 1e-9))
            if i > 0, i + 1 < luma.count, v < luma[i - 1] * 0.995, v < luma[i + 1] * 0.995 { pit = true }
        }
        return (drop, pit)
    }

    func testMesopicPrintIsMonotoneAcrossARedLightEdge() throws {
        let larson = worstDip(try mesopicRamp(.larson))
        // The probe is not vacuous: the old rod response prints a pit.
        XCTAssertGreaterThan(larson.drop, 0.05, "the Larson rod response should dip across the edge (probe check)")
        for floor in [true, false] {
            let additive = worstDip(try mesopicRamp(.additive, floor: floor))
            XCTAssertLessThan(additive.drop, 0.01, "additive rods: the print must rise with the radiance (floor \(floor))")
            XCTAssertFalse(additive.pit, "no pixel darker than both neighbours (floor \(floor))")
        }
    }

    /// A red light's TERMINATOR on a moonlit surface: the moonlight everywhere, the red light
    /// switching on over 3 px and then a small brighter red feature — the Digital Clock toy's red
    /// side. Radiance never falls left to right, so the print must not either; with the weight from
    /// the surround's plain adaptation level (`mesopicSpatialAdaptation`), additive rods and the
    /// cone-signal floor it does not.
    func testRedTerminatorPrintsNoSeamWithSpatialAdaptation() throws {
        let h = try makeHarness()
        let w = 256, hgt = 64
        let ambient = SIMD3<Float>(0.004, 0.0055, 0.0095)
        let red = SIMD3<Float>(1, 0.035, 0.012)
        let src = makeImage(h, width: w, height: hgt) { x, _ in
            let t = max(0, min(1, Float(x - 120) / 3))
            let feature: Float = (x >= 140 && x < 150) ? 3 : 1           // a small brighter ridge
            return ambient + red * (0.02 * t * feature)
        }
        func row(_ spatial: Bool, _ cone: Bool, _ additive: Bool) throws -> [Float] {
            var s = IlluminatoramaLocalToneMapping(strength: 0.01, radius: 0.1, edgeStops: 1, detail: 1,
                                                   anchor: 0.18, knee: 2, maxLift: 1)
            s.mesopic = 1
            s.mesopicModel = .blueShift
            s.mesopicTint = IlluminatoramaLocalToneMapping.jensenBlueShiftTint
            s.mesopicLogRange = SIMD2(-2, 0.6)
            s.mesopicPixelFloor = true
            s.mesopicNitsPerUnit = 1
            s.mesopicRodModel = additive ? .additive : .larson
            s.mesopicConeFloor = cone
            s.mesopicSpatialAdaptation = spatial
            let o = try XCTUnwrap(run(h, src, s))
            return (0..<w).map { x in let c = o[32 * w + x]; return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }
        }
        // Pits in the band where the red light comes on: a pixel darker than both neighbours
        // (beyond fp16 noise), or a print below the moonlit side's level after the red is on.
        func pits(_ l: [Float]) -> (count: Int, worst: Float) {
            let moon = l[100]
            var n = 0, worst: Float = 1
            for x in 118..<200 {
                if l[x] < l[x - 1] * 0.99, l[x] < l[x + 1] * 0.99 { n += 1 }
                if x > 124 { worst = min(worst, l[x] / moon) }
            }
            return (n, worst)
        }
        let old = pits(try row(false, false, false)), new = pits(try row(true, true, true))
        print("red terminator: old (edge-aware weight, Larson) pits \(old.count) min/moon \(old.worst); new pits \(new.count) min/moon \(new.worst)")
        XCTAssertLessThan(old.worst, 0.97, "probe: the old weight + Larson rods dip below the moonlit side")
        XCTAssertEqual(new.count, 0, "no pixel prints darker than both neighbours")
        XCTAssertGreaterThan(new.worst, 0.99, "once the red light is on, nothing prints darker than the moonlit side")
    }

    /// The additive rod response is Larson's exactly at each primary and at D65 grey (within
    /// 0.3 %), so single-colour regions (a red LED face, a grey wall) print as before.
    func testAdditiveRodMatchesLarsonOnPrimaries() {
        for c in [SIMD3<Float>(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1), SIMD3(1, 1, 1)] {
            let l = IlluminatoramaLocalToneMapping.larsonRodLuminance(c)
            let a = IlluminatoramaLocalToneMapping.additiveRodLuminance(c)
            XCTAssertEqual(a, l, accuracy: 0.003 * max(l, 1e-3), "\(c)")
        }
        // …and additive where Larson is not: red added to moonlight never LOWERS the rod response.
        let moon = SIMD3<Float>(0.0045, 0.006, 0.010), red = SIMD3<Float>(1, 0.035, 0.012) * 0.01
        XCTAssertLessThan(IlluminatoramaLocalToneMapping.larsonRodLuminance(moon + red),
                          IlluminatoramaLocalToneMapping.larsonRodLuminance(moon), "probe: Larson falls")
        XCTAssertGreaterThanOrEqual(IlluminatoramaLocalToneMapping.additiveRodLuminance(moon + red),
                                    IlluminatoramaLocalToneMapping.additiveRodLuminance(moon))
    }

    // ── Display inverse ─────────────────────────────────────────────────────

    func testDisplayInverseRoundTrips() {
        for look in [true, false] {
            for code: Float in [3, 6, 20, 60, 120] {
                let x = IlluminatoramaDisplayInverse.exposedLevel(printingAt: code, look: look)
                XCTAssertEqual(IlluminatoramaDisplayInverse.printedCode(exposed: x, look: look), code, accuracy: 0.05)
            }
        }
        // The punchy toe is the crushing one: sRGB 6 needs a brighter exposed level than bare AgX.
        XCTAssertGreaterThan(IlluminatoramaDisplayInverse.exposedLevel(printingAt: 6, look: true),
                             2 * IlluminatoramaDisplayInverse.exposedLevel(printingAt: 6, look: false))
        // Mid-grey prints near sRGB 99 through punchy AgX (the table in DigitalClockLook).
        XCTAssertEqual(IlluminatoramaDisplayInverse.printedCode(exposed: 0.18, look: true), 99, accuracy: 1.5)
    }
}
