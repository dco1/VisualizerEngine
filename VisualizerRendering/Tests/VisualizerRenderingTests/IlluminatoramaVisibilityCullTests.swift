import XCTest
import simd
@testable import VisualizerRendering

/// GPU-free soundness of visibility culling (DH-0534). Culling is only allowed to be a pure skip,
/// so both halves are checked against the thing they stand in for, not against themselves:
///
///  · the clip test is checked against Metal's clip rule applied to sampled points — a box the
///    test excludes must contain no point inside −w ≤ x,y ≤ w, 0 ≤ z ≤ w;
///  · the instance bounds are checked against Swift copies of `applyTreeWind` and `applySway`
///    (IlluminatoramaGBuffer.metal) — every displaced vertex must land inside the bound.
///
/// The app-side `HouseRenderBridgeGPUTests+VisibilityCulling` is the real-Metal half: zero
/// changed pixels, cull on vs off.
final class IlluminatoramaVisibilityCullTests: XCTestCase {

    // MARK: - shader mirrors (keep in step with IlluminatoramaGBuffer.metal)

    private func applyTreeWind(_ wp: SIMD3<Float>, _ attr: SIMD4<Float>, _ time: Float,
                               _ strength: Float, _ heading: Float) -> SIMD3<Float> {
        let sway = attr.x
        if sway < 0.0001 || strength <= 0 { return wp }
        var p = wp
        let wdir = SIMD2<Float>(cos(heading), sin(heading))
        let gust = 0.55 + 0.45 * sin(time * 0.43 + simd_dot(SIMD2(wp.x, wp.z), wdir) * 0.07)
        let macro = sin(time * 0.9 + attr.y)
        let bend = sway * strength * (0.5 + gust) * macro
        p.x += wdir.x * bend; p.z += wdir.y * bend
        p.y -= sway * strength * 0.18 * abs(macro)
        let flutter = attr.z
        if flutter > 0.0001 {
            p.x += flutter * strength * 0.05 * sin(time * 6.5 + attr.y * 4.0 + p.y * 1.1)
            p.z += flutter * strength * 0.05 * sin(time * 5.7 + attr.y * 3.0 + p.x * 1.1)
            p.y += flutter * strength * 0.035 * sin(time * 6.0 + attr.y * 3.5)
        }
        return p
    }

    private func applySway(_ wp: SIMD3<Float>, _ m: simd_float4x4, _ mode: Int32,
                           _ lean: Float, _ jostle: Float, _ time: Float) -> SIMD3<Float> {
        if mode == 0 { return wp }
        let pivotY: Float = mode == 2 ? 0 : -0.5
        let p4 = m * SIMD4<Float>(0, pivotY, 0, 1)
        let pivot = SIMD3<Float>(p4.x, p4.y, p4.z)
        let a4 = m * SIMD4<Float>(0, 0, 1, 0)
        let axis = simd_normalize(SIMD3<Float>(a4.x, a4.y, a4.z))
        var angle = lean
        if mode == 2 { angle = lean * sin(time * 2.65 + pivot.x * 1.7 + pivot.z * 2.3) }
        let c = cos(angle), s = sin(angle)
        let r = wp - pivot
        let rot = r * c + simd_cross(axis, r) * s + axis * simd_dot(axis, r) * (1 - c)
        return pivot + rot + SIMD3<Float>(0, jostle, 0)
    }

    // MARK: - helpers

    private struct LCG {
        var state: UInt64
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 24)
        }
        mutating func range(_ lo: Float, _ hi: Float) -> Float { lo + (hi - lo) * next() }
        mutating func unitVector() -> SIMD3<Float> {
            while true {
                let v = SIMD3<Float>(range(-1, 1), range(-1, 1), range(-1, 1))
                let l = simd_length(v)
                if l > 0.05 && l <= 1 { return v / l }
            }
        }
    }

    private func randomModel(_ g: inout LCG) -> simd_float4x4 {
        let q = simd_quatf(angle: g.range(0, 2 * .pi), axis: g.unitVector())
        let s = SIMD3<Float>(g.range(0.1, 3), g.range(0.1, 3), g.range(0.1, 3))
        var m = simd_float4x4(q) * simd_float4x4(diagonal: SIMD4(s, 1))
        m.columns.3 = SIMD4(g.range(-40, 40), g.range(-5, 10), g.range(-40, 40), 1)
        return m
    }

    private func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let ys = 1 / tan(fovY * 0.5), xs = ys / aspect, zs = far / (near - far)
        return simd_float4x4(columns: (SIMD4(xs, 0, 0, 0), SIMD4(0, ys, 0, 0),
                                       SIMD4(0, 0, zs, -1), SIMD4(0, 0, zs * near, 0)))
    }

    private func lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let z = simd_normalize(eye - target)
        let x = simd_normalize(simd_cross(up, z))
        let y = simd_cross(z, x)
        return simd_float4x4(columns: (SIMD4(x.x, y.x, z.x, 0), SIMD4(x.y, y.y, z.y, 0),
                                       SIMD4(x.z, y.z, z.z, 0),
                                       SIMD4(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1)))
    }

    private func insideClip(_ vp: simd_float4x4, _ p: SIMD3<Float>) -> Bool {
        let c = vp * SIMD4<Float>(p, 1)
        return c.x >= -c.w && c.x <= c.w && c.y >= -c.w && c.y <= c.w && c.z >= 0 && c.z <= c.w
    }

    // MARK: - clip volume

    func testClipVolumeKeepsWhatIsInFrontAndDropsWhatIsBehind() {
        let vp = perspective(fovY: 1.0, aspect: 1.6, near: 0.1, far: 100)
            * lookAt(eye: .zero, target: SIMD3(0, 0, -1), up: SIMD3(0, 1, 0))
        let clip = IlluminatoramaClipVolume(vp)
        func box(_ c: SIMD3<Float>, _ r: Float) -> IlluminatoramaCullBounds {
            var b = IlluminatoramaCullBounds.empty; b.formUnion(center: c, radius: r); return b
        }
        XCTAssertFalse(clip.excludes(box(SIMD3(0, 0, -10), 1)), "in front, centred")
        XCTAssertTrue(clip.excludes(box(SIMD3(0, 0, 10), 1)), "behind the eye")
        XCTAssertTrue(clip.excludes(box(SIMD3(0, 0, -200), 1)), "past the far plane")
        XCTAssertTrue(clip.excludes(box(SIMD3(60, 0, -10), 1)), "far off to the right")
        XCTAssertFalse(clip.excludes(box(SIMD3(0, 0, 0), 1)), "straddling the eye")
        XCTAssertFalse(clip.excludes(.unbounded), "unbounded always draws")

        // Orthographic (the cascade shape): a box beyond the far plane is dropped, one inside kept.
        let ortho = simd_float4x4(columns: (SIMD4(0.1, 0, 0, 0), SIMD4(0, 0.1, 0, 0),
                                            SIMD4(0, 0, -0.05, 0), SIMD4(0, 0, 0, 1)))   // x,y ∈ ±10, z ∈ [0, 20] down −Z
        let oc = IlluminatoramaClipVolume(ortho)
        XCTAssertFalse(oc.excludes(box(SIMD3(0, 0, -10), 1)))
        XCTAssertTrue(oc.excludes(box(SIMD3(0, 0, -25), 1)))
        XCTAssertTrue(oc.excludes(box(SIMD3(0, 0, 3), 1)))
    }

    /// Never exclude a box that holds a visible point — over random cameras and boxes.
    func testClipVolumeNeverExcludesABoxWithAPointInside() {
        var g = LCG(state: 0x5EED_0534)
        var excluded = 0, checked = 0
        for _ in 0..<400 {
            let eye = SIMD3<Float>(g.range(-20, 20), g.range(0, 10), g.range(-20, 20))
            let vp = perspective(fovY: g.range(0.3, 2.9), aspect: g.range(0.5, 2), near: g.range(0.005, 1),
                                 far: g.range(2, 150))
                * lookAt(eye: eye, target: eye + g.unitVector(), up: SIMD3(0.01, 1, 0))
            let clip = IlluminatoramaClipVolume(vp)
            for _ in 0..<40 {
                var b = IlluminatoramaCullBounds.empty
                b.formUnion(center: eye + g.unitVector() * g.range(0, 60), radius: g.range(0.01, 8))
                guard clip.excludes(b) else { continue }
                excluded += 1
                for _ in 0..<200 {
                    let p = SIMD3<Float>(g.range(b.lo.x, b.hi.x), g.range(b.lo.y, b.hi.y), g.range(b.lo.z, b.hi.z))
                    checked += 1
                    XCTAssertFalse(insideClip(vp, p), "an excluded box held a point inside the clip volume")
                }
            }
        }
        XCTAssertGreaterThan(excluded, 1000, "the sweep must actually exclude boxes or it proves nothing")
        print("DH-0534 clip soundness — \(excluded) excluded boxes, \(checked) sampled points, 0 inside")
    }

    // MARK: - instance bounds

    /// Every vertex position the vertex stage can emit — wind, then sway, at any time — lies
    /// inside `worldSphere`, for random matrices, sway modes, leans, jostles and wind weights.
    func testInstanceBoundsContainEveryWindAndSwayDisplacement() {
        var g = LCG(state: 0xB0B5_0534)
        let local = (center: SIMD3<Float>(0.1, -0.2, 0.05), radius: Float(0.9))
        var worstSlack: Float = .infinity
        for trial in 0..<600 {
            var inst = IlluminatoramaInstance(modelMatrix: randomModel(&g))
            inst.swayMode = Int32(trial % 3)
            inst.swayLean = g.range(-2, 2)
            inst.swayJostle = g.range(-0.3, 0.3)
            inst.windScale = trial % 2 == 0 ? g.range(0, 2) : 0
            let strength = g.range(0, 1.5)
            let windMax = (sway: g.range(1, 4), flutter: g.range(1, 3))
            guard let s = IlluminatoramaCullBounds.worldSphere(local: local, instance: inst,
                                                               treeWindStrength: strength,
                                                               windMax: { windMax }) else {
                return XCTFail("a finite instance must have a finite bound")
            }
            for _ in 0..<60 {
                let obj = local.center + g.unitVector() * g.range(0, local.radius)
                let attr = SIMD4<Float>(g.range(-1, windMax.sway), g.range(0, 6), g.range(-1, windMax.flutter), 1)
                let time = g.range(0, 500), heading = g.range(0, 2 * .pi)
                let w4 = inst.modelMatrix * SIMD4<Float>(obj, 1)
                var wp = applyTreeWind(SIMD3(w4.x, w4.y, w4.z), attr, time, strength * inst.windScale, heading)
                wp = applySway(wp, inst.modelMatrix, inst.swayMode, inst.swayLean, inst.swayJostle, time)
                let d = simd_length(wp - s.center)
                worstSlack = min(worstSlack, s.radius - d)
                XCTAssertLessThanOrEqual(d, s.radius, "trial \(trial): a displaced vertex escaped its bound")
            }
        }
        print("DH-0534 instance-bound soundness — worst slack \(worstSlack) m (must be ≥ 0)")
    }

    func testDegenerateMatrixIsUnboundedNotCulled() {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4(.nan, 0, 0, 1)
        let inst = IlluminatoramaInstance(modelMatrix: m)
        XCTAssertNil(IlluminatoramaCullBounds.worldSphere(local: (.zero, 1), instance: inst,
                                                          treeWindStrength: 0, windMax: { nil }))
    }
}
