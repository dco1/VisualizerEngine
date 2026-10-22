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
                // S3: every output triangle carries the id of the cluster it was copied from — the
                // per-cluster multiset of (cluster, index) pairs equals the DAG's.
                let tris = cut.lastShadowIndexCount / 3
                let ids = cut.outputClusterIDs.contents().bindMemory(to: UInt32.self, capacity: max(tris, 1))
                var stamped: [UInt32: [UInt32]] = [:]
                for t in 0..<tris { stamped[ids[t], default: []] += drawn[3 * t ..< 3 * t + 3] }
                XCTAssertEqual(Set(stamped.keys), Set(gpu.lodCut), "\(name) cluster ids name the cut")
                for (c, idx) in stamped where idx.sorted() != h.flatten([c]).sorted() {
                    XCTFail("\(name): triangles stamped \(c) are not that cluster's"); break
                }
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

    /// S4 — per-triangle soft routing: an on-screen triangle projecting under `maxArea` px² (box
    /// under `maxSpan`) lands in the soft range `[C − Q, C)`, every other on-screen triangle in the
    /// primary range, and shadow + soft together are exactly the LOD cut.
    func testSoftRoutingSplitsTheOnScreenCut() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let pipelines = SignackGPUPipelines.shared(device: device) else { throw XCTSkip("no Metal") }
        let (cm, atoms, cards) = SignackTestMeshes.cards(6000)
        let h = SignackHierarchyBuilder().build(positions: cm.positions, indices: cm.indices, atoms: atoms,
                                                simplifier: SignackCardSimplifier(cards: cards, seed: 7))
        let src: [SIMD3<Float>] = cm.positions
        let pos: [SIMD3<Float>] = src + h.derivedVertices.map { $0.position(source: src[Int($0.source)]) }
        guard let gh = SignackGPUHierarchy(device: device, hierarchy: h),
              let cut = SignackGPUCut(hierarchy: gh, device: device),
              let vbuf = device.makeBuffer(bytes: pos, length: pos.count * MemoryLayout<SIMD3<Float>>.stride,
                                           options: .storageModeShared) else { return XCTFail("buffers") }
        let viewport = SIMD2<Float>(1920, 1200)
        var split = 0
        for d: Float in [1.5, 4, 10, 30, 90] {
            for aim: Float in [0, 0.5] {
                let c = h.bounds.center, r = h.bounds.w
                let eye = c + SIMD3(d * r, 0.3 * d * r, 0.2 * r)
                let toward = simd_normalize(c - eye), side = simd_normalize(simd_cross(toward, SIMD3(0, 1, 0)))
                let m = Self.perspective(fovY: 0.8, aspect: 1.6, near: 0.01, far: 1000)
                    * Self.lookAt(eye: eye, target: eye + simd_normalize(toward * cos(aim) + side * sin(aim)))
                var view = SignackView.perspective(position: eye, fovY: 0.8, viewportHeight: viewport.y, threshold: 1)
                view.frustum = SignackFrustum(clip: m)
                cut.routing = SignackGPURouting(objectToClip: m, viewport: viewport, maxArea: 1, maxSpan: 4,
                                                vertices: vbuf, vertexStride: MemoryLayout<SIMD3<Float>>.stride)
                guard let cb = queue.makeCommandBuffer() else { return XCTFail("command buffer") }
                SignackGPUCutPass.encode([(cut, view)], pipelines: pipelines, commandBuffer: cb)
                cb.commit(); cb.waitUntilCompleted()
                let sel = cut.readSelection()
                // The CPU's routing of every on-screen triangle (ties within 1e-3 px² are skipped).
                var wantSoft: [[UInt32]] = [], wantHW: [[UInt32]] = [], ties = Set<[UInt32]>()
                for ci in sel.onScreen {
                    let cl = h.clusters[Int(ci)]
                    for k in stride(from: Int(cl.indexStart), to: Int(cl.indexStart) + 3 * Int(cl.triangleCount), by: 3) {
                        let tri = Array(h.indices[k ..< k + 3])
                        let (small, tie) = Self.routeSoft(tri.map { pos[Int($0)] }, m, viewport)
                        if tie { ties.insert(tri.sorted()) }
                        if small { wantSoft.append(tri.sorted()) } else { wantHW.append(tri.sorted()) }
                    }
                }
                let out = cut.outputIndices.contents().bindMemory(to: UInt32.self, capacity: gh.maxCutIndices)
                func tris(_ start: Int, _ count: Int) -> [[UInt32]] {
                    stride(from: start, to: start + count, by: 3).map { Array(UnsafeBufferPointer(start: out + $0, count: 3)).sorted() }
                }
                let cap = gh.maxCutIndices
                let gotSoft = tris(cap - cut.lastSoftIndexCount, cut.lastSoftIndexCount)
                let gotHW = tris(cut.lastShadowIndexCount - cut.lastPrimaryIndexCount, cut.lastPrimaryIndexCount)
                func key(_ v: [[UInt32]]) -> [[UInt32]] { v.filter { !ties.contains($0) }.sorted { $0.lexicographicallyPrecedes($1) } }
                XCTAssertEqual(key(gotSoft), key(wantSoft), "d \(d) aim \(aim): the soft range")
                XCTAssertEqual(key(gotHW), key(wantHW), "d \(d) aim \(aim): the primary range")
                XCTAssertEqual(cut.lastShadowIndexCount + cut.lastSoftIndexCount, 3 * h.triangleCount(sel.lodCut))
                let groups = Int(cut.drawArgs.contents().load(fromByteOffset: SignackGPUDrawArgs.softDispatchOffset, as: UInt32.self))
                XCTAssertEqual(groups, (cut.lastSoftIndexCount / 3 + 63) / 64)
                if !wantSoft.isEmpty && !wantHW.isEmpty { split += 1 }
            }
        }
        cut.routing = nil
        XCTAssertGreaterThanOrEqual(split, 3, "the views must split some cuts both ways")
    }

    /// The CPU reading of the routing test: projected area < 1 px² and box < 4 px; `tie` within 1e-3.
    static func routeSoft(_ p: [SIMD3<Float>], _ m: simd_float4x4, _ viewport: SIMD2<Float>) -> (Bool, Bool) {
        var px: [SIMD2<Float>] = []
        for v in p {
            let c: SIMD4<Float> = m * SIMD4<Float>(v, 1)
            if c.w <= 1e-6 { return (false, false) }
            px.append(SIMD2<Float>(c.x / c.w, c.y / c.w) * 0.5 * viewport)
        }
        let e1: SIMD2<Float> = px[1] - px[0], e2: SIMD2<Float> = px[2] - px[0]
        let area: Float = 0.5 * abs(e1.x * e2.y - e1.y * e2.x)
        let lo: SIMD2<Float> = simd_min(px[0], simd_min(px[1], px[2]))
        let hi: SIMD2<Float> = simd_max(px[0], simd_max(px[1], px[2]))
        let span: Float = max(hi.x - lo.x, hi.y - lo.y)
        return (area < 1 && span < 4, abs(area - 1) < 1e-3 || abs(span - 4) < 1e-3)
    }

    /// The Swift mirrors must match the shader's static_asserted sizes.
    func testGPUStructMirrorsMatchTheShader() {
        XCTAssertEqual(MemoryLayout<SignackGPUCluster>.stride, 64)
        XCTAssertEqual(MemoryLayout<SignackGPUCutUniforms>.stride, 144)
        XCTAssertEqual(MemoryLayout<SignackGPUCluster>.offset(of: \.indexStart), 48)
        XCTAssertEqual(MemoryLayout<SignackGPUCutUniforms>.offset(of: \.camera), 96)
        XCTAssertEqual(MemoryLayout<SignackGPUCutUniforms>.offset(of: \.threshold), 112)
        XCTAssertEqual(MemoryLayout<SignackGPUCutUniforms>.offset(of: \.softArea), 136)
        XCTAssertEqual(MemoryLayout<SignackRasterTriangle>.stride, 48)
    }
}
