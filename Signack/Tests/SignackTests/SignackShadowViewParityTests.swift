import XCTest
import Metal
import simd
@testable import Signack
import SignackGPU

/// DH-1465's parity gate: the GPU's per-light shadow-view cull (`signack_cut_classify`'s view mask,
/// `signack_cut_views_prefix`, the per-view copies and records) selects EXACTLY the clusters the CPU
/// reference (`SignackShadowViews.select`) selects for every view, each view's record holds exactly
/// those clusters' indices (or is the whole-cut shadow record when the view sees the whole cut or the
/// pool is full), and the S2 primary/shadow records are unchanged by it.
final class SignackShadowViewParityTests: XCTestCase {

    func testShadowViewCullMatchesTheCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let pipelines = SignackGPUPipelines.shared(device: device) else { throw XCTSkip("no Metal") }
        let ico = SignackTestMeshes.icosphere(subdivisions: 5)
        let (cm, atoms, cards) = SignackTestMeshes.cards(6000)
        let hierarchies: [(String, SignackHierarchy)] = [
            ("icosphere", SignackHierarchyBuilder().build(positions: ico.positions, indices: ico.indices,
                                                         simplifier: SignackEdgeCollapse())),
            ("cards", SignackHierarchyBuilder().build(positions: cm.positions, indices: cm.indices, atoms: atoms,
                                                     simplifier: SignackCardSimplifier(cards: cards, seed: 7))),
        ]
        var compared = 0, culledViews = 0, fullViews = 0, ties = 0
        for (name, h) in hierarchies {
            guard let gh = SignackGPUHierarchy(device: device, hierarchy: h) else { return XCTFail("buffers") }
            // A full pool and a tiny one (forces the whole-cut fallback for later views).
            for pool in [gh.maxCutIndices, gh.maxCutIndices / 20] {
                guard let cut = SignackGPUCut(hierarchy: gh, device: device, shadowViewPool: pool) else { return XCTFail("cut") }
                let views = SignackGPUParityTests.views(radius: h.bounds.w, center: h.bounds.center)
                for (vi, view) in views.enumerated() where vi % 3 == 0 {
                    let shadow = Self.shadowViews(radius: h.bounds.w, center: h.bounds.center, seed: vi)
                    cut.shadowViews = shadow
                    guard let cb = queue.makeCommandBuffer() else { return XCTFail("command buffer") }
                    SignackGPUCutPass.encode([(cut, view)], pipelines: pipelines, commandBuffer: cb)
                    cb.commit(); cb.waitUntilCompleted()
                    let sel = cut.readSelection()
                    let gpuViews = cut.readShadowViewSelection(shadow.count)
                    let cpuViews = SignackShadowViews.select(h, view: view, shadowViews: shadow)
                    let shadowCount = cut.lastShadowIndexCount
                    XCTAssertEqual(shadowCount, 3 * h.triangleCount(sel.lodCut), "\(name): S2 shadow record unchanged")
                    let out = cut.outputIndices.contents().bindMemory(to: UInt32.self, capacity: gh.maxCutIndices + pool)
                    for v in 0..<shadow.count {
                        // Membership is decided on the GPU's own LOD cut (ties there are S2's to forgive).
                        let cpu = Set(cpuViews[v]).intersection(sel.lodCut), gpu = Set(gpuViews[v])
                        for c in cpu.symmetricDifference(gpu) {
                            let cl = h.clusters[Int(c)], r = cl.bounds.w + view.cullPadding
                            let tie = shadow[v].planes.contains {
                                abs(simd_dot(SIMD3($0.x, $0.y, $0.z), cl.bounds.center) + $0.w + r) <= 1e-5 * max(1, h.bounds.w)
                            }
                            XCTAssertTrue(tie, "\(name) view \(v): cluster \(c) GPU \(gpu.contains(c)) CPU \(cpu.contains(c))")
                            ties += 1
                        }
                        let rec = cut.shadowViewRecord(v)
                        if rec.start == 0 {   // the whole-cut fallback
                            XCTAssertEqual(rec.count, shadowCount, "\(name) view \(v): fallback is the shadow record")
                            fullViews += 1
                        } else {
                            XCTAssertGreaterThanOrEqual(rec.start, gh.maxCutIndices, "\(name) view \(v): pool range")
                            XCTAssertLessThanOrEqual(rec.start + rec.count, gh.maxCutIndices + pool)
                            let got = Array(UnsafeBufferPointer(start: out + rec.start, count: rec.count))
                            XCTAssertEqual(got.sorted(), h.flatten(gpuViews[v]).sorted(), "\(name) view \(v): its indices")
                            if gpuViews[v].count < sel.lodCut.count { culledViews += 1 }
                        }
                    }
                    for v in shadow.count..<SignackShadowViews.maxViews {
                        XCTAssertEqual(cut.shadowViewRecord(v).count, shadowCount, "unused slot \(v) = shadow record")
                    }
                    compared += 1
                }
            }
        }
        print("Signack shadow-view parity: \(compared) cuts, \(culledViews) culled views, \(fullViews) whole-cut fallbacks, \(ties) plane ties")
        XCTAssertGreaterThan(culledViews, 20, "the views must cull")
        XCTAssertGreaterThan(fullViews, 5, "the views must exercise the fallback")
    }

    /// Three sun-cascade-like ortho boxes (a growing slice of the object along a light direction)
    /// plus a spot-like perspective cone aimed past the object's edge.
    static func shadowViews(radius r: Float, center c: SIMD3<Float>, seed: Int) -> [SignackFrustum] {
        let sun = simd_normalize(SIMD3<Float>(0.4 + 0.1 * Float(seed % 5), 1, 0.3))
        let eye = c + sun * 4 * r
        let view = SignackGPUParityTests.lookAt(eye: eye, target: c)
        var out: [SignackFrustum] = []
        for (k, half) in [Float(0.25), 0.6, 1.5].enumerated() {
            let off = SIMD3<Float>(Float(k) * 0.3 * r - 0.4 * r, 0.2 * r, 0)
            out.append(SignackFrustum(clip: ortho(halfWidth: half * r, near: 0.1 * r, far: 9 * r)
                                      * translate(-off) * view))
        }
        let spotEye = c + SIMD3(-2.5 * r, 1.5 * r, 0.5 * r)
        out.append(SignackFrustum(clip: SignackGPUParityTests.perspective(fovY: 0.5, aspect: 1, near: 0.05, far: 20 * r)
                                  * SignackGPUParityTests.lookAt(eye: spotEye, target: c + SIMD3(0, 0.6 * r, 0.4 * r))))
        return out
    }

    static func ortho(halfWidth w: Float, near: Float, far: Float) -> simd_float4x4 {
        simd_float4x4(SIMD4(1 / w, 0, 0, 0), SIMD4(0, 1 / w, 0, 0), SIMD4(0, 0, -1 / (far - near), 0),
                      SIMD4(0, 0, -near / (far - near), 1))
    }

    static func translate(_ t: SIMD3<Float>) -> simd_float4x4 {
        var m = matrix_identity_float4x4; m.columns.3 = SIMD4(t, 1); return m
    }
}
