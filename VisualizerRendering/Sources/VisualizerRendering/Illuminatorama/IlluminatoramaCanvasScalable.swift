import CoreGraphics
import Metal
import VisualizerCore
import simd

/// A native Illuminatorama scene whose fixed render canvas can be resized live.
///
/// Native scenes render at a FIXED canvas (`renderer.outputWidth × outputHeight`,
/// always 16:9 — e.g. 2880×1620, 1920×1080, 1280×720) and cover-blit it into the
/// live window. When the window drawable is larger than that canvas the image is
/// upscaled (soft). Conforming lets the toolbar's render-scale popover dial the
/// canvas up (sharper / 1:1 with the window) or down (faster) without relaunching.
///
/// The only per-scene requirement is exposing the renderer; the resize logic is a
/// default implementation here. Each native controller's `renderer` is `private`,
/// so the witness (`illuminatoramaRenderer { renderer }`) must live in the
/// controller's own module — declared as a one-line conforming extension per
/// controller, the same way `NativeMetalScene` is wired.
@MainActor
public protocol IlluminatoramaCanvasScalable: SceneController {
    var illuminatoramaRenderer: IlluminatoramaRenderer { get }
}

/// Every `NativeMetalScene` that is also `IlluminatoramaCanvasScalable` gets
/// `setRecorderFrameTap` for free: all it takes is forwarding the tap to the
/// renderer's `onFramePresented` hook, which `present(to:)` already calls each
/// frame. No per-scene boilerplate needed.
public extension NativeMetalScene where Self: IlluminatoramaCanvasScalable {
    func setRecorderFrameTap(_ tap: ((MTLTexture, MTLCommandQueue) -> Void)?) {
        illuminatoramaRenderer.onFramePresented = tap
    }
}

public extension IlluminatoramaCanvasScalable {
    /// Reference width that canvas scale 1.0 maps to (a 16:9 "4K-ish" canvas).
    /// Scale is expressed against this so the number is comparable across scenes
    /// regardless of each one's shipped default canvas.
    var canvasReferenceWidth: CGFloat { 2880 }

    /// Current fixed render-canvas resolution in pixels.
    var canvasPixelSize: SIMD2<Int> {
        SIMD2(illuminatoramaRenderer.outputWidth, illuminatoramaRenderer.outputHeight)
    }

    /// Current canvas scale relative to `canvasReferenceWidth` (1.0 ≈ 2880 wide).
    var canvasScale: CGFloat {
        CGFloat(illuminatoramaRenderer.outputWidth) / canvasReferenceWidth
    }

    /// Resize the fixed render canvas to `scale × reference`, preserving the
    /// scene's current aspect ratio (so framing never changes — only sharpness).
    ///
    /// This calls `IlluminatoramaRenderer.resize`, which reallocates every GPU
    /// target and resets temporal accumulation (a brief TAA/RT reconverge). That
    /// is acceptable for a deliberate, occasional user action — it is NOT the
    /// per-frame "match the renderer to the drawable" anti-pattern that flashes
    /// noise on every window drag (see docs/known-issues/illuminatorama-viewport-
    /// cover.md). The controller's tick rebinds `scene.background.contents` to the
    /// reallocated output texture and re-derives `camera.aspect` (unchanged here),
    /// so no extra bookkeeping is needed at the call site.
    func setCanvasScale(_ scale: CGFloat) {
        let s = min(1.5, max(0.25, scale))
        let r = illuminatoramaRenderer
        let aspect = r.outputHeight > 0
            ? CGFloat(r.outputWidth) / CGFloat(r.outputHeight)
            : 16.0 / 9.0
        let w = max(2, (Int((canvasReferenceWidth * s).rounded()) / 2) * 2)
        let h = max(2, (Int((CGFloat(w) / aspect).rounded()) / 2) * 2)
        guard w != r.outputWidth || h != r.outputHeight else { return }
        r.resize(width: w, height: h)
    }
}
