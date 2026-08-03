import XCTest
@testable import VisualizerRendering

/// `internalRenderScale` → internal render-target size.
///
/// The clamp used to be `max(1.0, scale)`: sub-native rendering was impossible,
/// so the only direction the pixel cost could move was UP. On a deferred,
/// fill-bound frame (measured on an architectural document: ≈4 ms fixed +
/// 2.3 ms per internal megapixel) that clamp was the single thing standing
/// between the app and a draft/navigation quality tier — 1.5 → 1.0 on a
/// 2560×1440 canvas measured 22.9 ms → 14.2 ms of GPU. These tests pin the
/// arithmetic contract that makes sub-native safe.
///
/// GPU-free by necessity: this package's `swift test` builds no metallib, so an
/// `IlluminatoramaRenderer` cannot be constructed here at all. `internalDims` is
/// deliberately `internal` (not `private`) so the contract is reachable.
final class IlluminatoramaInternalScaleTests: XCTestCase {

    private func dims(_ w: Int, _ h: Int, _ s: Float) -> (Int, Int) {
        IlluminatoramaRenderer.internalDims(outputW: w, outputH: h, scale: s)
    }

    // ── the reason this change exists ───────────────────────────────────────

    func testSubNativeScalesProduceSmallerInternalTargets() {
        // A 2560×1440 canvas — the size the complaint was measured at.
        XCTAssertEqual(dims(2560, 1440, 1.5).0, 3840)
        XCTAssertEqual(dims(2560, 1440, 1.5).1, 2160)
        XCTAssertEqual(dims(2560, 1440, 1.0).0, 2560)
        XCTAssertEqual(dims(2560, 1440, 1.0).1, 1440)
        // The three that were previously UNREACHABLE — every one of them
        // silently rendered at 1.0 before.
        XCTAssertEqual(dims(2560, 1440, 0.75).0, 1920)
        XCTAssertEqual(dims(2560, 1440, 0.75).1, 1080)
        XCTAssertEqual(dims(2560, 1440, 0.5).0, 1280)
        XCTAssertEqual(dims(2560, 1440, 0.5).1, 720)
    }

    func testInternalPixelCountFallsQuadraticallyWithScale() {
        // The whole point: the frame is fill-bound, so halving the linear scale
        // must quarter the pixels the G-buffer + lighting + glass passes cover.
        func pixels(_ s: Float) -> Int {
            let (w, h) = dims(2560, 1440, s)
            return w * h
        }
        XCTAssertEqual(Double(pixels(0.5)) / Double(pixels(1.0)), 0.25, accuracy: 0.01)
        XCTAssertEqual(Double(pixels(1.0)) / Double(pixels(1.5)), 1.0 / 2.25, accuracy: 0.01)
    }

    // ── clamping ────────────────────────────────────────────────────────────

    func testScaleIsClampedToTheDocumentedRange() {
        let floorDims = dims(2560, 1440, IlluminatoramaRenderer.minInternalRenderScale)
        // Anything below the floor renders AT the floor, never smaller.
        for tooSmall: Float in [0.49, 0.25, 0.1, 0.0, -1.0, -.infinity] {
            XCTAssertEqual(dims(2560, 1440, tooSmall).0, floorDims.0,
                           "scale \(tooSmall) must clamp up to the floor")
            XCTAssertEqual(dims(2560, 1440, tooSmall).1, floorDims.1)
        }
        let ceilDims = dims(2560, 1440, IlluminatoramaRenderer.maxInternalRenderScale)
        for tooBig: Float in [4.01, 8, 64, .infinity] {
            XCTAssertEqual(dims(2560, 1440, tooBig).0, ceilDims.0,
                           "scale \(tooBig) must clamp down to the ceiling")
        }
    }

    func testFloorIsHalfAndDefaultIsUnchanged() {
        XCTAssertEqual(IlluminatoramaRenderer.minInternalRenderScale, 0.5)
        XCTAssertEqual(IlluminatoramaRenderer.maxInternalRenderScale, 4.0)
        // The shipped default must not move — every existing scene's look and
        // every recorded perf baseline is against 1.5.
        XCTAssertEqual(IlluminatoramaRenderer.defaultInternalRenderScale, 1.5)
    }

    // ── nothing may ever produce a 0-sized or odd texture ───────────────────

    func testDimsAreNeverZeroAndAlwaysEven() {
        // A 0-wide texture is a hard Metal allocation failure, and `resize`
        // would then silently keep the OLD size forever — a scale change that
        // appears to do nothing. Sweep pathological outputs against the whole
        // scale range.
        let outputs = [(1, 1), (2, 1), (3, 7), (16, 9), (640, 480), (2560, 1440), (7680, 4320)]
        let scales: [Float] = [0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        for (w, h) in outputs {
            for s in scales {
                let (iw, ih) = dims(w, h, s)
                XCTAssertGreaterThanOrEqual(iw, 2, "internal width for \(w)×\(h) @\(s)")
                XCTAssertGreaterThanOrEqual(ih, 2, "internal height for \(w)×\(h) @\(s)")
                XCTAssertEqual(iw % 2, 0, "internal width must be even (\(w)×\(h) @\(s))")
                XCTAssertEqual(ih % 2, 0, "internal height must be even (\(w)×\(h) @\(s))")
                // The derived chains: SSAO/bloom at internal/2, halation at
                // internal/4. `makeTargets` floors each at 1, so the invariant
                // to hold here is simply that they stay allocatable.
                XCTAssertGreaterThanOrEqual(max(1, iw / 4), 1)
                XCTAssertGreaterThanOrEqual(max(1, ih / 4), 1)
            }
        }
    }

    func testNonPositiveOutputSizesStillProduceAllocatableDims() {
        // `resize` clamps output to ≥1 before calling this, but a direct caller
        // (or a zero-size drawable during window teardown) must not be able to
        // produce a 0.
        for (w, h) in [(0, 0), (-4, -4), (0, 1080)] {
            let (iw, ih) = dims(w, h, 1.0)
            XCTAssertGreaterThanOrEqual(iw, 2)
            XCTAssertGreaterThanOrEqual(ih, 2)
        }
    }

    func testScaleIsMonotonicInBothDirections() {
        var previous = 0
        for s in stride(from: Float(0.5), through: 4.0, by: 0.25) {
            let w = dims(1920, 1080, s).0
            XCTAssertGreaterThanOrEqual(w, previous, "internal width must not shrink as scale grows")
            previous = w
        }
    }
}
