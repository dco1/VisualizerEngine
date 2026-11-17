import XCTest
import simd
@testable import VisualizerRendering

/// Stable sun cascades: as the camera pans or turns, each cascade's shadow-texel grid must stay put in
/// the world. When it slid by a fraction of a texel per frame, every shadow edge's staircase
/// re-rasterised differently and shadows quivered under a moving camera (Munimyni's orbiting board).
@MainActor
final class IlluminatoramaCascadeSnapTests: XCTestCase {

    /// Munimyni's board camera: orthographic, pitched 34°, 70 back, turned by `yaw` round `target`.
    private func boardCamera(yaw: Float, target: SIMD3<Float>) -> IlluminatoramaCamera {
        let pitch: Float = 34 * .pi / 180
        let back = SIMD3(sinf(yaw) * cosf(pitch), sinf(pitch), cosf(yaw) * cosf(pitch))
        return IlluminatoramaCamera(position: target + back * 70, target: target, up: SIMD3(0, 1, 0),
                                    fovYRadians: 0.5, aspect: 1.6, zNear: 20, zFar: 160,
                                    projection: .orthographic(halfHeight: 0.85))
    }

    /// Where a fixed world point lands inside its shadow texel, per cascade, for a sweep of camera poses.
    private func subTexelSpread(snap: Bool) throws -> Float {
        let r = try IlluminatoramaRenderer(engine: .shared, width: 160, height: 100,
                                           camera: boardCamera(yaw: 0, target: .zero), internalRenderScale: 1)
        r.sharedTAAOverride = false
        r.taaEnabled = false                       // no DH-0858 sub-texel jitter: measure the fit alone
        r.shadowMaxDistance = 160
        r.directionalLightDirection = simd_normalize(SIMD3(-0.45, 0.85, 0.35))
        r.sunCascadeTexelSnapEnabled = snap
        let res = Float(r.shd.shadowMap.width)
        let probe = SIMD4<Float>(0.3, 0.05, -0.2, 1)
        var worst: Float = 0
        for c in 0..<IlluminatoramaRenderer.cascadeCount {
            var first: SIMD2<Float>?
            for k in 0..<24 {
                // A slow turn plus a small pan — sub-texel steps, the way an eased turn moves.
                r.camera = boardCamera(yaw: .pi / 4 + Float(k) * 0.0037, target: SIMD3(Float(k) * 0.0011, 0, 0.35))
                r.updateCascades()
                let p = r.shd.cascadeVPs[c] * probe
                let texel = (SIMD2(p.x, p.y) / p.w * 0.5 + 0.5) * res
                let frac = texel - texel.rounded(.down)
                if let f0 = first {
                    let d = abs(frac - f0)
                    worst = max(worst, min(d.x, 1 - d.x), min(d.y, 1 - d.y))
                } else { first = frac }
            }
        }
        return worst
    }

    func testCascadeTexelGridHoldsStillWhileTheCameraMoves() throws {
        XCTAssertLessThan(try subTexelSpread(snap: true), 0.02, "snapped cascades keep a world point at the same sub-texel spot")
    }

    /// Negative control: the same sweep without snapping does slide the grid, so the test above can see the bug.
    func testUnsnappedCascadesSlideTheTexelGrid() throws {
        XCTAssertGreaterThan(try subTexelSpread(snap: false), 0.1)
    }
}
