import XCTest
import Metal
import simd
@testable import Signack
import SignackGPU

/// S2's numeric parity gate: the GPU cut pass (`SignackCut.metal`) selects EXACTLY the clusters the
/// CPU reference (`SignackSelection`) selects — the LOD cut and the frustum-culled on-screen subset —
/// and compacts exactly their indices, shadow-only clusters first. Runs the real kernel under a bare
/// `swift test` (the library is compiled from source when no metallib is bundled).
///
/// A mismatch is tolerated only where the CPU itself sits on a float tie (projected error within
/// 1e-5 relative of the threshold, or a sphere within 1e-5 of a plane) — anywhere else it fails.
final class SignackGPUParityTests: XCTestCase {

    func testGPUCutMatchesTheCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw XCTSkip("no Metal device")
        }
        guard let pipelines = SignackGPUPipelines.shared(device: device) else {
            return XCTFail("SignackGPU pipelines failed to build")
        }
        let ico = SignackTestMeshes.icosphere(subdivisions: 5)
        let uv = SignackTestMeshes.uvSphere(meridians: 96, parallels: 48)
        let (cm, atoms, cards) = SignackTestMeshes.cards(6000)
        let hierarchies: [(String, SignackHierarchy)] = [
            ("icosphere", SignackHierarchyBuilder().build(positions: ico.positions, indices: ico.indices,
                                                         simplifier: SignackEdgeCollapse())),
            ("uvSphere", SignackHierarchyBuilder().build(positions: uv.positions, indices: uv.indices,
                                                        simplifier: SignackEdgeCollapse())),
            ("cards", SignackHierarchyBuilder().build(positions: cm.positions, indices: cm.indices, atoms: atoms,
                                                     simplifier: SignackCardSimplifier(cards: cards, seed: 7))),
        ]
        var compared = 0, ties = 0, culledSome = 0, mixed = 0
        for (name, h) in hierarchies {
            guard let gh = SignackGPUHierarchy(device: device, hierarchy: h),
                  let cut = SignackGPUCut(hierarchy: gh, device: device) else { return XCTFail("buffers") }
            for view in Self.views(radius: h.bounds.w, center: h.bounds.center) {
                guard let cb = queue.makeCommandBuffer() else { return XCTFail("command buffer") }
                SignackGPUCutPass.encode([(cut, view)], pipelines: pipelines, commandBuffer: cb)
                cb.commit(); cb.waitUntilCompleted()
                let gpu = cut.readSelection()
                let cpuCut = SignackSelection.select(h, view: view)
                let cpuOn = view.frustum == nil ? cpuCut : SignackSelection.select(h, view: view, cull: true)
                ties += try Self.diff(gpu.lodCut, cpuCut, h, view, cull: false, "\(name) LOD cut")
                ties += try Self.diff(gpu.onScreen, cpuOn, h, view, cull: true, "\(name) on-screen")
                if cpuOn.count < cpuCut.count { culledSome += 1 }
                if Set(cpuCut.map { h.clusters[Int($0)].level }).count > 1 { mixed += 1 }

                // Compaction: shadow range = the whole cut, primary range = the on-screen clusters,
                // off-screen ones first. Same index multiset as the CPU flatten.
                XCTAssertEqual(cut.lastShadowIndexCount, 3 * h.triangleCount(gpu.lodCut), "\(name) shadow count")
                XCTAssertEqual(cut.lastPrimaryIndexCount, 3 * h.triangleCount(gpu.onScreen), "\(name) primary count")
                let out = cut.outputIndices.contents().bindMemory(to: UInt32.self, capacity: gh.maxCutIndices)
                let drawn = Array(UnsafeBufferPointer(start: out, count: cut.lastShadowIndexCount))
                XCTAssertEqual(drawn.sorted(), h.flatten(gpu.lodCut).sorted(), "\(name) compacted indices")
                let offStart = cut.lastShadowIndexCount - cut.lastPrimaryIndexCount
                let onSet = Set(gpu.onScreen)
                XCTAssertEqual(Array(drawn[offStart...]).sorted(), h.flatten(onSet.sorted()).sorted(),
                               "\(name) primary range holds exactly the on-screen clusters")
                compared += 1
            }
        }
        print("SignackGPU parity: \(compared) views × 3 meshes compared, \(ties) float-tie mismatches, "
              + "\(culledSome) views frustum-culled clusters, \(mixed) mixed-level cuts, library \(pipelines.librarySource)")
        XCTAssertGreaterThan(culledSome, 10, "the views must exercise frustum culling")
        XCTAssertGreaterThan(mixed, 10, "the views must exercise mixed-level cuts")
    }

    /// Fails on any mismatch that is not a float tie; returns how many ties it forgave.
    private static func diff(_ gpu: [UInt32], _ cpu: [UInt32], _ h: SignackHierarchy, _ v: SignackView,
                             cull: Bool, _ what: String) throws -> Int {
        let a = Set(gpu), b = Set(cpu)
        var ties = 0
        for c in a.symmetricDifference(b) {
            let cl = h.clusters[Int(c)]
            func nearTie(_ s: SignackSphere, _ e: Float) -> Bool {
                let p = v.projectedError(s, e)
                return p.isFinite && abs(p - v.threshold) <= 1e-5 * max(v.threshold, 1)
            }
            var tie = nearTie(cl.lodSphere, cl.lodError) || nearTie(cl.parentSphere, cl.parentError)
            if cull, let f = v.frustum {
                let r = cl.bounds.w + v.cullPadding
                tie = tie || f.planes.contains { abs(simd_dot(SIMD3($0.x, $0.y, $0.z), cl.bounds.center) + $0.w + r) <= 1e-5 }
            }
            XCTAssertTrue(tie, "\(what): cluster \(c) (level \(cl.level)) GPU \(a.contains(c)) vs CPU \(b.contains(c))")
            ties += 1
        }
        return ties
    }

    /// Distances from inside to far, several thresholds, and look directions that put the mesh
    /// centred, half out, and mostly out of a padded frustum (plus no-frustum and orthographic).
    static func views(radius r: Float, center: SIMD3<Float>) -> [SignackView] {
        var out: [SignackView] = []
        for d: Float in [0.4, 1.3, 2.5, 6, 20, 80] {
            for px: Float in [0.5, 1, 4, 16] {
                let eye = center + SIMD3(d * r, 0.3 * d * r, 0.2 * r)
                var plain = SignackView.perspective(position: eye, fovY: 0.8, viewportHeight: 1200, threshold: px)
                out.append(plain)
                for aim: Float in [0, 0.6, 1.4] {   // radians off the mesh centre
                    let toward = simd_normalize(center - eye)
                    let side = simd_normalize(simd_cross(toward, SIMD3(0, 1, 0)))
                    let look = simd_normalize(toward * cos(aim) + side * sin(aim))
                    let view = lookAt(eye: eye, target: eye + look)
                    let proj = perspective(fovY: 0.8, aspect: 1.6, near: 0.01, far: 1000)
                    plain.frustum = SignackFrustum(clip: proj * view)
                    plain.cullPadding = 0.02 * r
                    out.append(plain)
                }
            }
        }
        out.append(.orthographic(halfHeight: 2 * r, viewportHeight: 1200, threshold: 1))
        return out
    }

    static func lookAt(eye: SIMD3<Float>, target: SIMD3<Float>) -> simd_float4x4 {
        let f = simd_normalize(target - eye), s = simd_normalize(simd_cross(f, SIMD3(0, 1, 0))), u = simd_cross(s, f)
        return simd_float4x4(SIMD4(s.x, u.x, -f.x, 0), SIMD4(s.y, u.y, -f.y, 0), SIMD4(s.z, u.z, -f.z, 0),
                             SIMD4(-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1))
    }

    static func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let y = 1 / tan(fovY * 0.5), x = y / aspect, zr = far - near
        return simd_float4x4(SIMD4(x, 0, 0, 0), SIMD4(0, y, 0, 0), SIMD4(0, 0, -far / zr, -1),
                             SIMD4(0, 0, -(far * near) / zr, 0))
    }

    /// The Swift mirrors must match the shader's static_asserted sizes.
    func testGPUStructMirrorsMatchTheShader() {
        XCTAssertEqual(MemoryLayout<SignackGPUCluster>.stride, 64)
        XCTAssertEqual(MemoryLayout<SignackGPUCutUniforms>.stride, 144)
        XCTAssertEqual(MemoryLayout<SignackGPUCluster>.offset(of: \.indexStart), 48)
        XCTAssertEqual(MemoryLayout<SignackGPUCutUniforms>.offset(of: \.camera), 96)
        XCTAssertEqual(MemoryLayout<SignackGPUCutUniforms>.offset(of: \.threshold), 112)
    }
}
