import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// The CPU halves of DH-1668's shadow reuse, each checked against the thing it stands in for:
///
///  · `IlluminatoramaMesh.shapeDiff` — the region a mesh swap dirties must contain every triangle
///    that is not common to both meshes, and must not grow to the whole mesh when one part moved;
///  · `texelRect` + `scissorFrustum` — a point the region's frustum keeps must project inside the
///    texel rectangle, and one it rejects must project outside it (the visibility cull in a region
///    pass is only allowed to be a pure skip).
///
/// The real-Metal half — a dragged wall's frame matches a freshly rasterised one, bit for bit — is
/// Daydream's `HouseRenderBridgeGPUTests_SunCascadeDrag`.
@MainActor
final class IlluminatoramaSunCascadeRegionTests: XCTestCase {

    /// `n` unit quads along +x, quad `i` spanning x ∈ [2i, 2i+1], each lifted by `lift(i)` in y.
    private func strip(_ device: MTLDevice, _ n: Int, lift: (Int) -> Float = { _ in 0 }) -> IlluminatoramaMesh {
        var v: [IlluminatoramaVertex] = [], idx: [UInt16] = []
        for i in 0..<n {
            let x = Float(2 * i), y = lift(i), b = UInt16(v.count)
            for (dx, dz) in [(0, 0), (1, 0), (1, 1), (0, 1)] as [(Float, Float)] {
                v.append(IlluminatoramaVertex(position: SIMD3(x + dx, y, dz), normal: SIMD3(0, 1, 0), uv: .zero))
            }
            idx += [b, b + 2, b + 1, b, b + 3, b + 2]
        }
        return IlluminatoramaMesh(device: device, vertices: v, indices: idx)
    }

    func testShapeDiffBoundsOnlyTheMovedPart() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let a = strip(device, 6)
        XCTAssertEqual(IlluminatoramaMesh.shapeDiff(a, strip(device, 6)), .identical,
                       "byte-identical geometry in a new mesh changes no depth")
        // Quad 3 (x ∈ [6, 7]) lifted 0.5 m: the region is that quad, old pose and new.
        guard case .region(let lo, let hi) = IlluminatoramaMesh.shapeDiff(a, strip(device, 6) { $0 == 3 ? 0.5 : 0 }) else {
            return XCTFail("a moved part must yield a bounded region")
        }
        XCTAssertEqual(lo, SIMD3<Float>(6, 0, 0))
        XCTAssertEqual(hi, SIMD3<Float>(7, 0.5, 1))
        // A part added at the end: only it.
        guard case .region(let lo2, let hi2) = IlluminatoramaMesh.shapeDiff(a, strip(device, 7)) else {
            return XCTFail("an appended part must yield a bounded region")
        }
        XCTAssertEqual(lo2.x, 12); XCTAssertEqual(hi2.x, 13)
        // Different raster flags cannot be compared triangle by triangle.
        let b = strip(device, 6); b.shadowCastsBothFaces = true
        XCTAssertEqual(IlluminatoramaMesh.shapeDiff(a, b), .whole)
    }

    func testRegionFrustumKeepsExactlyWhatProjectsIntoTheRectangle() {
        // A cascade-like ortho light matrix looking down −y over a 40 m square.
        let view = IlluminatoramaRenderer.lookAtRH(eye: SIMD3(0, 50, 0), target: .zero, up: SIMD3(0, 0, 1))
        let proj = IlluminatoramaRenderer.orthoRH(left: -20, right: 20, bottom: -20, top: 20, near: 0, far: 100)
        let vp = proj * view
        let size = 2048
        var box = IlluminatoramaCullBounds.empty
        box.formUnion(center: SIMD3(3, 1, -4), radius: 1)
        guard let rect = IlluminatoramaRenderer.texelRect(of: [box], vp: vp, width: size, height: size) else {
            return XCTFail("a box inside the volume must project to a rectangle")
        }
        XCTAssertLessThan(rect.width * rect.height, size * size / 50, "a 2 m box is a small part of a 40 m slice")
        let sub = IlluminatoramaRenderer.scissorFrustum(vp, rect, size: size)
        func texel(_ p: SIMD3<Float>) -> SIMD2<Float> {
            let q = vp * SIMD4(p, 1)
            return SIMD2((q.x / q.w + 1) * 0.5 * Float(size), (1 - q.y / q.w) * 0.5 * Float(size))
        }
        func inside(_ m: simd_float4x4, _ p: SIMD3<Float>) -> Bool {
            let q = m * SIMD4(p, 1)
            return abs(q.x) <= q.w && abs(q.y) <= q.w && q.z >= 0 && q.z <= q.w
        }
        var rng = SystemRandomNumberGenerator()
        var kept = 0
        for _ in 0..<4000 {
            let p = SIMD3<Float>(Float.random(in: -8...8, using: &rng), Float.random(in: -2...6, using: &rng),
                                 Float.random(in: -12...4, using: &rng))
            let t = texel(p)
            // (Skip the float-rounding band on the rectangle's own edges.)
            let edge = [t.x - Float(rect.x), Float(rect.x + rect.width) - t.x,
                        t.y - Float(rect.y), Float(rect.y + rect.height) - t.y].map(abs).min() ?? 1
            if edge < 0.01 { continue }
            let inRect = t.x >= Float(rect.x) && t.x <= Float(rect.x + rect.width)
                && t.y >= Float(rect.y) && t.y <= Float(rect.y + rect.height)
            XCTAssertEqual(inside(sub, p), inRect && inside(vp, p), "region frustum and rectangle disagree at \(p)")
            if inside(sub, p) { kept += 1 }
        }
        XCTAssertGreaterThan(kept, 0, "the sample must reach the region")
        // Everything in the changed box itself projects into the rectangle.
        for i in 0..<8 {
            let p = SIMD3<Float>((i & 1) == 0 ? box.lo.x : box.hi.x, (i & 2) == 0 ? box.lo.y : box.hi.y,
                                 (i & 4) == 0 ? box.lo.z : box.hi.z)
            XCTAssertTrue(inside(sub, p), "a corner of the changed box fell outside its own region")
        }
    }
}
