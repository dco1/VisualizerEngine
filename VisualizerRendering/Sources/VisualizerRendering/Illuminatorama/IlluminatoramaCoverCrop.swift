import SceneKit
import simd

public extension IlluminatoramaRenderer {
    /// **The default viewport behaviour for a SceneKit-background native
    /// Illuminatorama scene** (one whose `scene.background.contents` is bound to
    /// `renderer.outputTexture`). Object-fit: **cover** — keep the renderer at its
    /// FIXED canvas and cover-crop that fixed-aspect image to the live viewport:
    /// fills the `SCNView`, never stretches, never letterboxes/pillarboxes.
    ///
    /// Call this ONCE at the **end of `tick()`**, AFTER every
    /// `scene.background.contents = …` write — SceneKit silently resets
    /// `contentsTransform` to identity on each `contents` assignment (including the
    /// pool-rotation rebind), so a transform set earlier in the tick is clobbered.
    ///
    /// ```swift
    /// // end of tick(), after render() + any background.contents rebind:
    /// if let size = viewportSizeProvider?(), size.x > 0, size.y > 0 {
    ///     renderer.applyBackgroundCoverCrop(to: scene,
    ///                                       viewportWidth: size.x, viewportHeight: size.y)
    /// }
    /// ```
    ///
    /// Do **NOT** `renderer.resize(width:height:)` to the live drawable/viewport to
    /// "fill" the window — that reframes the camera as the window is dragged and
    /// reallocates every GPU target (resetting TAA / AO / RT temporal history, which
    /// flashes noise on every drag). The fixed-canvas + cover-crop pattern here is
    /// the one true default. The MTKView-present sibling path gets the identical
    /// crop for free inside `present(to:)` — no per-scene call needed there.
    ///
    /// See `docs/known-issues/illuminatorama-viewport-cover.md`.
    @MainActor
    func applyBackgroundCoverCrop(to scene: SCNScene, viewportWidth: Int, viewportHeight: Int) {
        guard viewportWidth > 0, viewportHeight > 0 else { return }
        let uv = coverUVRect(viewportWidth: viewportWidth, viewportHeight: viewportHeight)
        var t = SCNMatrix4Identity
        t.m11 = SCNFloat(uv.x); t.m22 = SCNFloat(uv.y)
        t.m41 = SCNFloat(uv.z); t.m42 = SCNFloat(uv.w)
        scene.background.contentsTransform = t
    }
}
