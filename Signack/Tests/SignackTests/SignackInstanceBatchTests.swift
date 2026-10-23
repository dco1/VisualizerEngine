import XCTest
import Metal
import simd
@testable import Signack
@testable import SignackGPU

/// S6: the batched instance pass (`SignackInstances.metal`) puts every instance in the SAME bucket
/// and list the CPU reference does, copies its record intact, and writes the indexed-instanced
/// records the host draws. Plus the pass's cost at 50 / 200 / 800 instances.
final class SignackInstanceBatchTests: XCTestCase {

    static func reference(_ b: SignackGPUInstanceBounds, _ v: SignackInstanceView, rungs: [SignackRungs.Rung],
                          impostor: Bool) -> (Int, Bool)? {
        SignackGPUInstanceBatch.referenceBucket(b, view: v, rungs: rungs, impostor: impostor).map { ($0.bucket, $0.onScreen) }
    }

    static func rungs() -> SignackRungs {
        let (m, atoms, cards) = SignackTestMeshes.cards(3000)
        let h = SignackHierarchyBuilder().build(positions: m.positions, indices: m.indices, atoms: atoms,
                                               simplifier: SignackCardSimplifier(cards: cards, seed: 3))
        return SignackRungs(h, maxRungs: 4)
    }

    static func view(_ aspect: Float = 1.6) -> SignackInstanceView {
        let eye = SIMD3<Float>(0, 1.6, 0), fov: Float = 0.9
        let viewM = simd_float4x4(lookAt: eye, target: SIMD3(0, 1, -10))
        let proj = simd_float4x4(perspectiveFov: fov, aspect: aspect, near: 0.1, far: 500)
        return SignackInstanceView(position: eye, pixelScale: 1200 / (2 * tan(fov / 2)), threshold: 1,
                                   impostorPixels: 48, frustum: SignackFrustum(clip: proj * viewM), maxDistance: 180)
    }

    func testBucketsMatchTheCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let pipelines = SignackGPUPipelines.shared(device: device) else { throw XCTSkip("no Metal") }
        let rungs = Self.rungs()
        XCTAssertGreaterThanOrEqual(rungs.rungs.count, 3)
        for k in 1..<rungs.rungs.count {
            XCTAssertLessThanOrEqual(rungs.rungs[k].triangleCount * 2, rungs.rungs[k - 1].triangleCount + 1)
            XCTAssertGreaterThan(rungs.rungs[k].error, rungs.rungs[k - 1].error)
        }
        let n = 800, stride = 16
        guard let batch = SignackGPUInstanceBatch(device: device, rungs: rungs, impostorIndexCount: 24,
                                                  capacity: n, instanceStride: stride) else { return XCTFail("batch") }
        var bounds: [SignackGPUInstanceBounds] = []
        var s: UInt64 = 99
        func r() -> Float { s = s &* 6364136223846793005 &+ 1442695040888963407; return Float(s >> 40) / Float(1 << 24) }
        for _ in 0..<n {
            let scale = 0.4 + r()
            bounds.append(SignackGPUInstanceBounds(center: SIMD3((r() - 0.5) * 300, 0.5, (r() - 0.5) * 300),
                                                   radius: 3 * scale, scale: scale))
        }
        batch.setInstances(count: n) { raw, b in
            for i in 0..<n {
                raw.storeBytes(of: SIMD4<UInt32>(UInt32(i), 7, 7, 7), toByteOffset: i * stride, as: SIMD4<UInt32>.self)
                b[i] = bounds[i]
            }
        }
        let view = Self.view()
        guard let cb = queue.makeCommandBuffer() else { return XCTFail("cb") }
        SignackGPUInstancePass.encode([(batch, view)], pipelines: pipelines, commandBuffer: cb)
        cb.commit(); cb.waitUntilCompleted()

        let buckets = batch.bucketCount
        var expect = [[Set<UInt32>]](repeating: [Set<UInt32>](repeating: [], count: buckets), count: 2)
        for i in 0..<n {
            guard let (b, on) = Self.reference(bounds[i], view, rungs: rungs.rungs, impostor: true) else { continue }
            if on { expect[0][b].insert(UInt32(i)) }
            expect[1][b].insert(UInt32(i))
        }
        let lists = batch.lists.contents()
        var used = Set<Int>()
        for b in 0..<buckets {
            for list in 0..<2 {
                let off = list == 0 ? SignackGPUInstanceBatch.primaryOffset(bucket: b) : SignackGPUInstanceBatch.shadowOffset(bucket: b)
                let a = batch.drawArgs.contents().advanced(by: off).bindMemory(to: UInt32.self, capacity: 5)
                let count = Int(a[1]), base = Int(a[4])
                XCTAssertEqual(base, (list * buckets + b) * n)
                if b < rungs.rungs.count {
                    let r = list == 1 ? min(b + 2, rungs.rungs.count - 1) : b   // shadows: two rungs coarser
                    XCTAssertEqual(Int(a[0]), rungs.rungs[r].indexCount); XCTAssertEqual(Int(a[2]), rungs.rungs[r].indexStart)
                } else { XCTAssertEqual(Int(a[0]), 24); XCTAssertEqual(a[2], 0) }
                var got = Set<UInt32>()
                for k in 0..<count {
                    let rec = lists.load(fromByteOffset: (base + k) * stride, as: SIMD4<UInt32>.self)
                    XCTAssertEqual(rec.y, 7, "record copied intact")
                    got.insert(rec.x)
                }
                XCTAssertEqual(got, expect[list][b], "bucket \(b) list \(list)")
                if count > 0 { used.insert(b) }
            }
        }
        XCTAssertGreaterThanOrEqual(used.count, 3, "the view exercises several rungs and the impostor")
        XCTAssertTrue(used.contains(buckets - 1), "far plants go to the impostor")
    }

    /// DH-1493 — per-light shadow views: every (view, bucket) record holds exactly the instances the CPU
    /// reference puts there (in that bucket, sphere touching the view), each once, records intact; the
    /// primary and shadow lists are unchanged by the views; views straddle, contain and miss instances.
    func testShadowViewRecordsMatchTheCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let pipelines = SignackGPUPipelines.shared(device: device) else { throw XCTSkip("no Metal") }
        let rungs = Self.rungs()
        let n = 800, stride = 16
        guard let batch = SignackGPUInstanceBatch(device: device, rungs: rungs, impostorIndexCount: 24,
                                                  capacity: n, instanceStride: stride) else { return XCTFail("batch") }
        var bounds: [SignackGPUInstanceBounds] = []
        var s: UInt64 = 1493
        func r() -> Float { s = s &* 6364136223846793005 &+ 1442695040888963407; return Float(s >> 40) / Float(1 << 24) }
        for _ in 0..<n {
            let scale = 0.4 + r()
            bounds.append(SignackGPUInstanceBounds(center: SIMD3((r() - 0.5) * 300, 0.5, (r() - 0.5) * 300),
                                                   radius: 3 * scale, scale: scale))
        }
        batch.setInstances(count: n) { raw, b in
            for i in 0..<n {
                raw.storeBytes(of: SIMD4<UInt32>(UInt32(i), 7, 7, 7), toByteOffset: i * stride, as: SIMD4<UInt32>.self)
                b[i] = bounds[i]
            }
        }
        // Three nested sun-cascade boxes (a low sun from the south-west) and one spot cone.
        let sunView = simd_float4x4(lookAt: SIMD3(-60, 80, 60), target: SIMD3(0, 0, -10))
        var views: [SignackFrustum] = [20, 60, 160].map {
            SignackFrustum(clip: simd_float4x4(orthoHalf: $0, depth: 400) * sunView)
        }
        views.append(SignackFrustum(clip: simd_float4x4(perspectiveFov: 0.8, aspect: 1, near: 0.1, far: 120)
                                         * simd_float4x4(lookAt: SIMD3(10, 30, -20), target: SIMD3(15, 0, -40))))
        batch.shadowViews = views
        let view = Self.view()
        guard let cb = queue.makeCommandBuffer() else { return XCTFail("cb") }
        SignackGPUInstancePass.encode([(batch, view)], pipelines: pipelines, commandBuffer: cb)
        cb.commit(); cb.waitUntilCompleted()

        let buckets = batch.bucketCount
        var line = "DH-1493 per-view instances (of \(n)):"
        for (v, f) in views.enumerated() {
            var total = 0
            for b in 0..<buckets {
                let off = SignackGPUInstanceBatch.shadowViewOffset(view: v, bucket: b)
                let a = batch.drawArgs.contents().advanced(by: off).bindMemory(to: UInt32.self, capacity: 5)
                if b < rungs.rungs.count {
                    let rr = min(b + 2, rungs.rungs.count - 1)
                    XCTAssertEqual(Int(a[0]), rungs.rungs[rr].indexCount, "view \(v) bucket \(b): the shadow rung")
                    XCTAssertEqual(Int(a[2]), rungs.rungs[rr].indexStart)
                } else { XCTAssertEqual(Int(a[0]), 24) }
                var expect = Set<UInt32>()
                for i in 0..<n {
                    guard let (bb, _) = Self.reference(bounds[i], view, rungs: rungs.rungs, impostor: true), bb == b,
                          SignackGPUInstanceBatch.referenceInView(bounds[i], f) else { continue }
                    expect.insert(UInt32(i))
                }
                var got: [UInt32] = []
                for rec in batch.shadowViewRecords(view: v, bucket: b) {
                    let w = rec.withUnsafeBytes { $0.load(as: SIMD4<UInt32>.self) }
                    XCTAssertEqual(w.y, 7, "record copied intact")
                    got.append(w.x)
                }
                XCTAssertEqual(got.count, Set(got).count, "view \(v) bucket \(b): each instance once")
                XCTAssertEqual(Set(got), expect, "view \(v) bucket \(b)")
                total += got.count
            }
            line += " v\(v) \(total)"
            XCTAssertGreaterThan(total, 0, "view \(v) must hold plants")
            XCTAssertLessThan(total, n, "view \(v) must cull plants")
        }
        print(line)
        // The two lists are untouched by the views.
        for b in 0..<buckets {
            let a = batch.drawArgs.contents().advanced(by: SignackGPUInstanceBatch.shadowOffset(bucket: b))
                .bindMemory(to: UInt32.self, capacity: 5)
            var expect = 0
            for i in 0..<n where Self.reference(bounds[i], view, rungs: rungs.rungs, impostor: true)?.0 == b { expect += 1 }
            XCTAssertEqual(Int(a[1]), expect, "shadow list bucket \(b)")
        }
        // A second frame with the views gone writes no view records again (the book is reset each frame).
        batch.shadowViews = []
        guard let cb2 = queue.makeCommandBuffer() else { return XCTFail("cb") }
        SignackGPUInstancePass.encode([(batch, view)], pipelines: pipelines, commandBuffer: cb2)
        cb2.commit(); cb2.waitUntilCompleted()
        let book = batch.viewBook.contents().bindMemory(to: UInt32.self, capacity: 192)
        XCTAssertEqual((0..<64).map { book[$0] }.reduce(0, +), 0, "per-view counts reset after finalize")
    }

    /// The pass's GPU cost (start→end of a command buffer holding only it), 6 species × N each.
    func testBatchedPassCost() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let pipelines = SignackGPUPipelines.shared(device: device) else { throw XCTSkip("no Metal") }
        let rungs = Self.rungs()
        var report = "S6 batched instance pass, 6 batches (species variants), stride 384 B, GPU ms (median of 40):\n"
        for total in [50, 200, 800] {
            let per = max(1, total / 6)
            var batches: [SignackGPUInstanceBatch] = []
            for _ in 0..<6 {
                guard let b = SignackGPUInstanceBatch(device: device, rungs: rungs, impostorIndexCount: 24,
                                                      capacity: per, instanceStride: 384) else { return XCTFail() }
                var s: UInt64 = UInt64(batches.count + 1)
                func r() -> Float { s = s &* 6364136223846793005 &+ 1442695040888963407; return Float(s >> 40) / Float(1 << 24) }
                b.setInstances(count: per) { _, bb in
                    for i in 0..<per { bb[i] = SignackGPUInstanceBounds(center: SIMD3((r() - 0.5) * 40, 0.5, -r() * 40), radius: 1, scale: 1) }
                }
                batches.append(b)
            }
            var times: [Double] = []
            for _ in 0..<40 {
                guard let cb = queue.makeCommandBuffer() else { return XCTFail() }
                SignackGPUInstancePass.encode(batches.map { ($0, Self.view()) }, pipelines: pipelines, commandBuffer: cb)
                cb.commit(); cb.waitUntilCompleted()
                times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
            }
            times.sort()
            report += "  \(per * 6) instances: \(String(format: "%.4f", times[times.count / 2])) ms\n"
        }
        print(report)
        try? report.write(toFile: NSTemporaryDirectory() + "signack-s6-batch-cost.txt", atomically: true, encoding: .utf8)
    }
}

private extension simd_float4x4 {
    init(lookAt eye: SIMD3<Float>, target: SIMD3<Float>) {
        let f = simd_normalize(target - eye), s = simd_normalize(simd_cross(f, SIMD3(0, 1, 0))), u = simd_cross(s, f)
        self.init(SIMD4(s.x, u.x, -f.x, 0), SIMD4(s.y, u.y, -f.y, 0), SIMD4(s.z, u.z, -f.z, 0),
                  SIMD4(-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1))
    }
    /// Metal-style orthographic box, ±half in x/y, depth 0…depth in front of the eye.
    init(orthoHalf h: Float, depth: Float) {
        self.init(SIMD4(1 / h, 0, 0, 0), SIMD4(0, 1 / h, 0, 0), SIMD4(0, 0, -1 / depth, 0), SIMD4(0, 0, 0, 1))
    }
    init(perspectiveFov fov: Float, aspect: Float, near: Float, far: Float) {
        let y = 1 / tan(fov / 2), x = y / aspect, z = far / (near - far)
        self.init(SIMD4(x, 0, 0, 0), SIMD4(0, y, 0, 0), SIMD4(0, 0, z, -1), SIMD4(0, 0, z * near, 0))
    }
}
