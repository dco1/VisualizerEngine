import XCTest
import simd
import CoreGraphics
@testable import VisualizerRendering

/// Small renderers (gallery cards, thumbnails): the init knobs that keep one from paying for a full-size
/// renderer's atlases and targets, and fewer sun cascades — without moving anything for a default renderer.
@MainActor
final class IlluminatoramaSmallRendererTests: XCTestCase {

    private func camera() -> IlluminatoramaCamera {
        IlluminatoramaCamera(position: SIMD3(0, 1.2, 2.4), target: SIMD3(0, 0.2, 0), up: SIMD3(0, 1, 0),
                             fovYRadians: .pi / 4, aspect: 1, zNear: 0.05, zFar: 50)
    }

    /// A box on a slab under a high sun — a shadow on the slab beside the box.
    private func stage(_ r: IlluminatoramaRenderer) {
        r.setMesh(.box, IlluminatoramaMesh.unitBox(device: r.device))
        let slab = simd_float4x4(diagonal: SIMD4(4, 0.02, 4, 1))
        var box = simd_float4x4(diagonal: SIMD4(0.5, 0.5, 0.5, 1))
        box.columns.3 = SIMD4(0, 0.26, 0, 1)
        r.instances = [.init(meshKind: .box, data: IlluminatoramaInstance(modelMatrix: slab)),
                       .init(meshKind: .box, data: IlluminatoramaInstance(modelMatrix: box))]
        r.directionalLightDirection = simd_normalize(SIMD3(0.6, 1, 0.3))
        r.directionalLightColor = SIMD3(repeating: 3)
        r.shadowsEnabled = true
        r.sharedTAAOverride = false
        r.autoExposureEnabled = false   // frames compared across settings must not drift with metering
        r.exposure = 0.5
    }

    private func meanLuma(_ f: IlluminatoramaFrame) -> Double {
        var s = 0.0
        for i in stride(from: 0, to: f.bytes.count, by: 4) { s += Double(f.bytes[i]) + Double(f.bytes[i + 1]) + Double(f.bytes[i + 2]) }
        return s / Double(f.bytes.count / 4 * 3)
    }

    private func meanAbsDiff(_ a: IlluminatoramaFrame, _ b: IlluminatoramaFrame) -> Double {
        var s = 0.0
        for i in 0..<min(a.bytes.count, b.bytes.count) { s += abs(Double(a.bytes[i]) - Double(b.bytes[i])) }
        return s / Double(a.bytes.count)
    }

    func testDefaultsAreUnchanged() throws {
        let r = try IlluminatoramaRenderer(engine: .shared, width: 160, height: 120, camera: camera())
        XCTAssertEqual(r.albedoAtlas.capacity, 64)
        XCTAssertEqual(r.nonColorAtlas.capacity, 64)
        XCTAssertEqual(r.sunShadowCascadeCount, 3)
        if IlluminatoramaRenderer.canvasBudget.maxInternalRenderScale == nil {
            XCTAssertEqual(r.internalRenderScale, IlluminatoramaRenderer.defaultInternalRenderScale)
        }
    }

    func testLightInitAllocatesASmallRenderer() throws {
        let full = try IlluminatoramaRenderer(engine: .shared, width: 300, height: 300, camera: camera())
        let light = try IlluminatoramaRenderer(engine: .shared, width: 300, height: 300, camera: camera(),
                                               shadowMapResolution: 1024, textureAtlasCapacity: 4,
                                               internalRenderScale: 1)
        XCTAssertEqual(light.albedoAtlas.capacity, 4)
        XCTAssertEqual(light.internalRenderScale, 1)
        XCTAssertEqual(light.width, light.outputWidth)
        XCTAssertGreaterThan(full.initTiming.totalAllocatedMB, 150)
        XCTAssertLessThan(light.initTiming.totalAllocatedMB, 80,
                          "light init allocated \(light.initTiming.totalAllocatedMB) MB:\n\(light.initTiming.summary)")
    }

    func testASmallAtlasGrowsOnDemand() throws {
        let r = try IlluminatoramaRenderer(engine: .shared, width: 64, height: 64, camera: camera(),
                                           textureAtlasCapacity: 2, internalRenderScale: 1)
        let space = CGColorSpaceCreateDeviceRGB()
        var slices: [Int32] = []
        for k in 0..<5 {
            let ctx = try XCTUnwrap(CGContext(data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 32,
                                              space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
            ctx.setFillColor(CGColor(red: CGFloat(k) / 5, green: 0.5, blue: 0.2, alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
            slices.append(try XCTUnwrap(r.albedoAtlas.register(contents: try XCTUnwrap(ctx.makeImage()))))
        }
        XCTAssertEqual(Set(slices).count, 5)
        XCTAssertGreaterThanOrEqual(r.albedoAtlas.capacity, 5)
    }

    /// One cascade over a short range still casts the box's shadow, and lands close to the three-cascade picture;
    /// going back to three restores it.
    func testOneCascadeStillShadows() throws {
        let r = try IlluminatoramaRenderer(engine: .shared, width: 160, height: 160, camera: camera(),
                                           internalRenderScale: 1)
        stage(r)
        r.shadowMaxDistance = 6
        let three = try XCTUnwrap(r.renderHeadless(frames: 4))
        r.sunShadowCascadeCount = 1
        XCTAssertEqual(r.shadowViewsThisFrame().filter { if case .cascade = $0.view { true } else { false } }.count, 1)
        let one = try XCTUnwrap(r.renderHeadless(frames: 4))
        r.shadowsEnabled = false
        let none = try XCTUnwrap(r.renderHeadless(frames: 4))
        r.shadowsEnabled = true
        r.sunShadowCascadeCount = 3
        let threeAgain = try XCTUnwrap(r.renderHeadless(frames: 4))

        let shadow = meanAbsDiff(one, none)     // what the shadow changes
        XCTAssertGreaterThan(meanLuma(none), meanLuma(one), "one cascade's frame is not darker than no shadow")
        XCTAssertGreaterThan(shadow, 0.2, "one cascade cast no shadow")
        XCTAssertLessThan(meanAbsDiff(one, three), shadow / 2, "one cascade's picture is far from three's")
        XCTAssertLessThan(meanAbsDiff(three, threeAgain), 0.05, "three cascades did not come back")
        r.sunShadowCascadeCount = 7
        XCTAssertEqual(r.sunShadowCascadeCount, 3)
        r.sunShadowCascadeCount = 0
        XCTAssertEqual(r.sunShadowCascadeCount, 1)
    }
}
