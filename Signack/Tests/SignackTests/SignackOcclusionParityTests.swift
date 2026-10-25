import XCTest
import Metal
import simd
@testable import Signack
import SignackGPU

/// DH-1466's parity gate: the GPU Hi-Z pyramid (`SignackHiZ`) equals the CPU reference level for
/// level, and the cut kernel's occlusion test hides EXACTLY the on-screen clusters the CPU reference
/// (`SignackOcclusion.isOccluded`) hides — and only from the camera record (the shadow record still
/// holds every cluster of the cut).
final class SignackOcclusionParityTests: XCTestCase {

    func testHiZAndOcclusionMatchTheCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let pipelines = SignackGPUPipelines.shared(device: device) else { throw XCTSkip("no Metal") }
        // An odd-sized depth: far background, a near "house" slab over the middle, a noisy band.
        let W = 301, H = 187
        var depth = [Float](repeating: 1, count: W * H)
        var rng = SystemRandomNumberGenerator()
        for y in 0..<H { for x in 0..<W {
            if x > 70 && x < 230 && y > 40 && y < 170 { depth[y * W + x] = 0.92 }
            else if y > 150 { depth[y * W + x] = Float.random(in: 0.95...0.999, using: &rng) }
        } }
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: W, height: H, mipmapped: false)
        td.usage = [.shaderRead, .renderTarget]; td.storageMode = .private
        guard let tex = device.makeTexture(descriptor: td),
              let up = device.makeBuffer(bytes: depth, length: depth.count * 4, options: .storageModeShared),
              let cb0 = queue.makeCommandBuffer(), let blit = cb0.makeBlitCommandEncoder() else { return XCTFail("setup") }
        blit.copy(from: up, sourceOffset: 0, sourceBytesPerRow: W * 4, sourceBytesPerImage: W * H * 4,
                  sourceSize: MTLSize(width: W, height: H, depth: 1), to: tex, destinationSlice: 0,
                  destinationLevel: 0, destinationOrigin: MTLOrigin())
        blit.endEncoding()
        let hiz = SignackHiZ(device: device)
        XCTAssertTrue(hiz.encodeBuild(depth: tex, pipelines: pipelines, commandBuffer: cb0))
        cb0.commit(); cb0.waitUntilCompleted()

        // Pyramid parity, every level.
        let cpu = SignackOcclusion.pyramid(depth: depth, width: W, height: H)
        XCTAssertEqual(hiz.mipCount, cpu.count)
        guard let t = hiz.texture, let cb1 = queue.makeCommandBuffer(), let b1 = cb1.makeBlitCommandEncoder() else { return XCTFail() }
        var reads: [MTLBuffer] = []
        for (l, lv) in cpu.enumerated() {
            let buf = device.makeBuffer(length: lv.width * lv.height * 4, options: .storageModeShared)!
            b1.copy(from: t, sourceSlice: 0, sourceLevel: l, sourceOrigin: MTLOrigin(),
                    sourceSize: MTLSize(width: lv.width, height: lv.height, depth: 1), to: buf,
                    destinationOffset: 0, destinationBytesPerRow: lv.width * 4, destinationBytesPerImage: lv.width * lv.height * 4)
            reads.append(buf)
        }
        b1.endEncoding(); cb1.commit(); cb1.waitUntilCompleted()
        for (l, lv) in cpu.enumerated() {
            let g = Array(UnsafeBufferPointer(start: reads[l].contents().bindMemory(to: Float.self, capacity: lv.depth.count), count: lv.depth.count))
            XCTAssertEqual(g, lv.depth, "pyramid level \(l) (\(lv.width)×\(lv.height))")
        }

        // Cut parity: an icosphere at several places — wholly behind the slab, beside it, straddling.
        let ico = SignackTestMeshes.icosphere(subdivisions: 5)
        let h = SignackHierarchyBuilder().build(positions: ico.positions, indices: ico.indices, simplifier: SignackEdgeCollapse())
        guard let gh = SignackGPUHierarchy(device: device, hierarchy: h),
              let cut = SignackGPUCut(hierarchy: gh, device: device) else { return XCTFail("cut") }
        let proj = SignackGPUParityTests.perspective(fovY: 0.8, aspect: Float(W) / Float(H), near: 0.1, far: 100)
        var hid = 0, compared = 0, ties = 0
        for (dx, dist) in [(Float(0), Float(40)), (0, 8), (-3.5, 30), (2.5, 25), (0, 12), (6, 30)] {
            let eye = h.bounds.center + SIMD3(dx, 0.3, dist)
            let viewM = SignackGPUParityTests.lookAt(eye: eye, target: eye - SIMD3(0, 0, 1))
            let vp = proj * viewM
            var view = SignackView.perspective(position: eye, fovY: 0.8, viewportHeight: Float(H), threshold: 1)
            view.frustum = SignackFrustum(clip: vp)
            cut.occlusion = SignackGPUOcclusion(hiz: hiz, toClip: vp, dilatePixels: 2)
            guard let cb = queue.makeCommandBuffer() else { return XCTFail() }
            SignackGPUCutPass.encode([(cut, view)], pipelines: pipelines, commandBuffer: cb)
            cb.commit(); cb.waitUntilCompleted()
            let sel = cut.readSelectionWithOcclusion()
            let onOrHidden = Set(sel.onScreen).union(sel.occluded)
            let want = Set(onOrHidden.filter {
                let b = h.clusters[Int($0)].bounds
                return SignackOcclusion.isOccluded(center: SIMD3(b.x, b.y, b.z), radius: b.w, toClip: vp, levels: cpu,
                                                   depthSize: SIMD2(Float(W), Float(H)), dilate: 2)
            })
            for c in want.symmetricDifference(sel.occluded) {   // a float tie only: depth within 1e-5 of the max
                ties += 1
                XCTAssertLessThan(ties, 3, "cluster \(c): GPU occluded \(sel.occluded.contains(c)) vs CPU \(want.contains(c))")
            }
            XCTAssertEqual(cut.lastShadowIndexCount, 3 * h.triangleCount(sel.lodCut), "shadow record keeps hidden clusters")
            XCTAssertEqual(cut.lastPrimaryIndexCount, 3 * h.triangleCount(sel.onScreen), "camera record drops them")
            XCTAssertEqual(cut.lastOccludedClusterCount, sel.occluded.count)
            hid += sel.occluded.count; compared += onOrHidden.count
        }
        cut.occlusion = nil
        print("Signack Hi-Z parity: \(cpu.count) pyramid levels equal; \(hid) of \(compared) on-screen clusters hidden, \(ties) ties, library \(pipelines.librarySource)")
        XCTAssertGreaterThan(hid, 20, "the views must hide clusters")
        XCTAssertLessThan(hid, compared, "and leave some visible")
    }
}
