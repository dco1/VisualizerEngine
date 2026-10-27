import CoreGraphics
import simd

extension IlluminatoramaRenderer {
    /// Cover-blit-aware screen→world ray — the one place a view point becomes a pick ray.
    /// The renderer cover-crops its output onto the drawable on present, so a raw viewport→NDC
    /// mapping is off whenever the render aspect ≠ the view aspect; inverting `coverUVRect`
    /// first keeps picking aligned with what is drawn.
    ///
    /// `viewPoint` is in the view's points with a TOP-LEFT origin (flip AppKit's native
    /// bottom-left `y` as `viewport.height - y` first). `viewport` is the drawable size the
    /// cover rect is computed for. Perspective cameras give one origin and a fanned direction;
    /// orthographic ones give one direction and an origin that slides across the frame.
    public func pickRay(viewPoint vp: CGPoint, viewport: CGSize)
        -> (origin: SIMD3<Float>, dir: SIMD3<Float>) {
        let cam = camera
        let w = max(Float(viewport.width), 1), h = max(Float(viewport.height), 1)
        let uv = coverUVRect(viewportWidth: Int(w), viewportHeight: Int(h))
        let cu = Float(vp.x) / w * uv.x + uv.z, cv = Float(vp.y) / h * uv.y + uv.w
        let ndcX = 2 * cu - 1, ndcY = 1 - 2 * cv
        let f = simd_normalize(cam.target - cam.position)
        let s = simd_normalize(simd_cross(f, cam.up))
        let up = simd_cross(s, f)
        let e = cam.halfExtents(atViewDistance: 1)
        switch cam.projection {
        case .perspective:
            return (cam.position, simd_normalize(f + ndcX * e.width * s + ndcY * e.height * up))
        case .orthographic:
            return (cam.position + ndcX * e.width * s + ndcY * e.height * up, f)
        }
    }
}
