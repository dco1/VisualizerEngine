import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// **Lean lighting** (IlluminatoramaRenderer+LeanLighting.swift): the lighting kernel and the tonemap compiled
/// on the frame's own flags and features must draw the same pixels as the uber kernels they replace, and the
/// masks must keep every feature whose uniform can reach the kernel.
@MainActor
final class IlluminatoramaLeanLightingTests: XCTestCase {

    private func frame(lean: Bool, pointLight: Bool, plain: Bool = false) throws -> IlluminatoramaFrame {
        let cam = IlluminatoramaCamera(position: SIMD3(0, 1, 4), target: .zero, up: SIMD3(0, 1, 0),
                                       fovYRadians: .pi / 3, aspect: 4.0 / 3.0, zNear: 0.05, zFar: 100)
        let r = try IlluminatoramaRenderer(engine: SimEngine.shared, width: 128, height: 96, camera: cam)
        let rig = IlluminatoramaStudio.productShot
        rig.apply(to: r)
        r.setMesh(.box, IlluminatoramaMesh.unitBox(device: r.device))
        r.instances = rig.instances()
        if pointLight {
            r.pointLights = [IlluminatoramaPointLight(position: SIMD3(1, 1.5, 1), radius: 6, color: SIMD3(4, 3, 2))]
        }
        r.taaJitterPixels = 0
        r.blockingLightingCompile = true            // the specialised variant, not the uber fallback
        r.leanLightingMaterials = .none
        r.leanLightingEnabled = lean
        r.plainSurfacesOnly = plain
        if lean { XCTAssertFalse(r.prewarmLeanLighting(addingFeatures: [IlluminatoramaRenderer.leanFeaturePointLights]),
                                 "nothing to prewarm from before a frame") }
        let f = try XCTUnwrap(r.renderHeadless(frames: 3))
        if lean { XCTAssertTrue(r.prewarmLeanLighting(addingFeatures: [IlluminatoramaRenderer.leanFeaturePointLights])) }
        if lean {
            let masks = try XCTUnwrap(r.activeLeanMasks, "the lean variant must be the one that ran")
            XCTAssertEqual(masks.y & 1 != 0, pointLight, "the point-light bit follows the scene")
        }
        return f
    }

    func testLeanVariantsDrawTheUberKernelsPixels() throws {
        for point in [false, true] {
            let uber = try frame(lean: false, pointLight: point)
            let lean = try frame(lean: true, pointLight: point)
            XCTAssertEqual(uber.bytes.count, lean.bytes.count)
            var worst = 0
            for i in 0..<uber.bytes.count { worst = max(worst, abs(Int(uber.bytes[i]) - Int(lean.bytes[i]))) }
            XCTAssertLessThanOrEqual(worst, 1, "lean lighting changed pixels (point light: \(point)), worst \(worst)/255")
        }
    }

    /// The studio rig's boxes are plain vertex-colour materials: the plain-surface G-buffer draws the same frame.
    func testPlainSurfaceGBufferDrawsTheSameFrame() throws {
        let uber = try frame(lean: false, pointLight: true)
        let plain = try frame(lean: false, pointLight: true, plain: true)
        var worst = 0
        for i in 0..<uber.bytes.count { worst = max(worst, abs(Int(uber.bytes[i]) - Int(plain.bytes[i]))) }
        XCTAssertLessThanOrEqual(worst, 1, "plain surfaces changed pixels, worst \(worst)/255")
    }

    func testMasksKeepEveryReachableFeature() {
        var u = IlluminatoramaFrameUniforms.leanTestDefault
        XCTAssertEqual(IlluminatoramaRenderer.leanFeatureMask(u, materials: .none), 0)
        XCTAssertEqual(IlluminatoramaRenderer.leanFeatureMask(u, materials: .all), IlluminatoramaLeanMaterials.all.rawValue)
        u.pointLightCount = 2; u.directionalLightCount = 1; u.sssStrength = 0.3; u.interiorMask = 4
        XCTAssertEqual(IlluminatoramaRenderer.leanFeatureMask(u, materials: .none), (1 << 0) | (1 << 3) | (1 << 5) | (1 << 6))
        var t = IlluminatoramaFrameUniforms.leanTestDefault
        t.debandDitherEnabled = 1; t.vignetteStrength = 0.2; t.highlightTemperatureK = 5000
        XCTAssertEqual(IlluminatoramaRenderer.leanTonemapMask(t), (1 << 19) | (1 << 16) | (1 << 14))
    }
}

private extension IlluminatoramaFrameUniforms {
    /// Every lean-gated field at its "off" value.
    static var leanTestDefault: Self {
        let raw = UnsafeMutableRawPointer.allocate(byteCount: MemoryLayout<Self>.size, alignment: MemoryLayout<Self>.alignment)
        defer { raw.deallocate() }
        raw.initializeMemory(as: UInt8.self, repeating: 0, count: MemoryLayout<Self>.size)
        var u = raw.load(as: Self.self)
        u.shadowTemperatureK = 6500; u.highlightTemperatureK = 6500; u.debandDitherEnabled = 0
        return u
    }
}
