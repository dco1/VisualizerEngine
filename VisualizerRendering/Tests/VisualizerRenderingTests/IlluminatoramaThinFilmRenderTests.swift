import XCTest
import simd
import Metal
@testable import VisualizerRendering

/// **DH-1014 / DH-1302 on the real renderer** — the shipped G-buffer, lighting, SSR and RT kernels (not a
/// probe), through `IlluminatoramaRenderer.renderHeadless`, on the product studio.
///
/// The contract each lever keeps: with it OFF, an instance that CARRIES the parameter renders byte-for-byte
/// the same as one that does not (an opted-out host is untouched); with it ON but the weight 0, the same;
/// with it ON and the weight up, the picture moves — in the right way (a film adds COLOUR, translucency
/// adds LIGHT).
@MainActor
final class IlluminatoramaThinFilmRenderTests: XCTestCase {

    private func makeRenderer() throws -> IlluminatoramaRenderer {
        let cam = IlluminatoramaCamera(position: SIMD3(0.0, 0.55, 1.7), target: SIMD3(-0.2, 0.32, 0.0), up: SIMD3(0, 1, 0),
                                       fovYRadians: 0.9, aspect: 4.0 / 3.0, zNear: 0.05, zFar: 50)
        let r = try IlluminatoramaRenderer(engine: SimEngine.shared, width: 320, height: 240, camera: cam)
        let rig = IlluminatoramaStudio.productShot
        rig.apply(to: r)
        r.setMesh(.box, IlluminatoramaMesh.unitBox(device: r.device))
        r.setMesh(.sphere, IlluminatoramaMesh.unitSphere(device: r.device, meridians: 64, parallels: 32))
        r.layeredSpecularEnabled = true
        return r
    }

    /// The studio + a polished metal sphere (the film's subject — a film on a 4 % dielectric F0 is nearly invisible) + a thin panel (translucency's subject).
    private func scene(film: Float = 0, translucency: Float = 0) -> [IlluminatoramaRenderer.InstanceRef] {
        var s = matrix_identity_float4x4
        s.columns.0.x = 0.3; s.columns.1.y = 0.3; s.columns.2.z = 0.3; s.columns.3 = SIMD4(0.1, 0.3, 0, 1)
        var sphere = IlluminatoramaInstance(modelMatrix: s, albedo: SIMD3(0.78, 0.78, 0.80), metallic: 1, roughness: 0.14)
        sphere.thinFilmWeight = film
        sphere.thinFilmThickness = 380
        var p = matrix_identity_float4x4
        p.columns.0.x = 0.02; p.columns.1.y = 0.5; p.columns.2.z = 0.5; p.columns.3 = SIMD4(-0.55, 0.3, 0, 1)
        var panel = IlluminatoramaInstance(modelMatrix: p, albedo: SIMD3(0.8, 0.45, 0.2), metallic: 0, roughness: 0.6)
        panel.translucencyWeight = translucency
        return IlluminatoramaStudio.productShot.instances()
            + [.init(meshKind: .sphere, data: sphere), .init(meshKind: .box, data: panel)]
    }

    private func shoot(_ r: IlluminatoramaRenderer, _ label: String, film: Float = 0, translucency: Float = 0) throws -> [UInt8] {
        r.instances = scene(film: film, translucency: translucency)
        let frame = try XCTUnwrap(r.renderHeadless(frames: 6), "\(label): no frame")
        if let dir = ProcessInfo.processInfo.environment["THINFILM_OUT"] {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            _ = r.renderHeadlessPNG(to: "\(dir)/\(label).png", frames: 1)
        }
        return frame.bytes
    }

    private func diff(_ a: [UInt8], _ b: [UInt8]) -> (changed: Int, maxDelta: Int, meanChroma: Double) {
        var changed = 0, maxD = 0, chroma = 0.0
        for i in stride(from: 0, to: a.count, by: 4) {
            let d = max(abs(Int(a[i]) - Int(b[i])), abs(Int(a[i + 1]) - Int(b[i + 1])), abs(Int(a[i + 2]) - Int(b[i + 2])))
            if d > 2 { changed += 1; maxD = max(maxD, d) }
            let ca = Double(max(a[i], a[i + 1], a[i + 2])) - Double(min(a[i], a[i + 1], a[i + 2]))
            let cb = Double(max(b[i], b[i + 1], b[i + 2])) - Double(min(b[i], b[i + 1], b[i + 2]))
            chroma += cb - ca
        }
        return (changed, maxD, chroma / Double(a.count / 4))
    }

    func testThinFilmIsInertWhenOffAndColoursTheSphereWhenOn() throws {
        let r = try makeRenderer()
        r.thinFilmEnabled = false
        let base = try shoot(r, "thinfilm-baseline")
        let offWithFilm = try shoot(r, "thinfilm-off-instance-carries-film", film: 1)
        r.thinFilmEnabled = true
        let onNoFilm = try shoot(r, "thinfilm-on-weight0")
        let onFilm = try shoot(r, "thinfilm-on-weight1", film: 1)
        let a = diff(base, offWithFilm), b = diff(base, onNoFilm), c = diff(base, onFilm)
        print("DH-1014 render: off+film changed \(a.changed) px · on+w0 changed \(b.changed) px · on+w1 changed \(c.changed) px (max Δ \(c.maxDelta), mean chroma Δ \(String(format: "%.3f", c.meanChroma)))")
        XCTAssertEqual(a.changed, 0, "lever OFF: an instance that carries a film renders exactly as one that does not")
        XCTAssertEqual(b.changed, 0, "lever ON, weight 0: nothing moves")
        XCTAssertGreaterThan(c.changed, 150, "lever ON, weight 1: the sphere's reflection is recoloured")
        XCTAssertGreaterThan(c.meanChroma, 0, "a film adds colour")
    }

    func testTranslucencyIsInertWhenOffAndLightsThePanelWhenOn() throws {
        let r = try makeRenderer()
        r.translucencyEnabled = false
        let base = try shoot(r, "translucency-baseline")
        let offWith = try shoot(r, "translucency-off-instance-carries-it", translucency: 0.8)
        r.translucencyEnabled = true
        let onZero = try shoot(r, "translucency-on-weight0")
        let on = try shoot(r, "translucency-on-weight0.8", translucency: 0.8)
        let a = diff(base, offWith), b = diff(base, onZero), c = diff(base, on)
        print("DH-1302 render: off+t changed \(a.changed) px · on+w0 changed \(b.changed) px · on+w.8 changed \(c.changed) px (max Δ \(c.maxDelta))")
        XCTAssertEqual(a.changed, 0, "lever OFF: an instance that carries translucency renders exactly as one that does not")
        XCTAssertEqual(b.changed, 0, "lever ON, weight 0: nothing moves")
        XCTAssertGreaterThan(c.changed, 20, "lever ON, weight 0.8: the panel lets light through")
        var brighter = 0
        for i in stride(from: 0, to: base.count, by: 4) where Int(on[i]) + Int(on[i + 1]) + Int(on[i + 2]) > Int(base[i]) + Int(base[i + 1]) + Int(base[i + 2]) + 6 { brighter += 1 }
        XCTAssertGreaterThanOrEqual(brighter, c.changed * 8 / 10, "translucency only ever ADDS light (energy: never more than arrives)")
    }
}
