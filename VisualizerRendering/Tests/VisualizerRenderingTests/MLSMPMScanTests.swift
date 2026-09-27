import Metal
import XCTest
import simd
@testable import VisualizerRendering

/// `mlsCellOffsetsScan` — the per-substep exclusive prefix sum that bins MLS-MPM particles by
/// cell: exact against a CPU scan (sizes that end mid-block, mid-simdgroup and on a block edge),
/// the counts left zeroed for the scatter pass, and fast (the single-thread loop it replaced took
/// 2.2–4.2 ms at 27k cells).
@MainActor
final class MLSMPMScanTests: XCTestCase {
    func testScanIsExactZeroesCountsAndIsFast() throws {
        let engine = SimEngine.shared
        let pso = try XCTUnwrap(engine.pipeline("mlsCellOffsetsScan"))
        var rng = SystemRandomNumberGenerator()
        for res in [SIMD3<UInt32>(31, 28, 31), SIMD3<UInt32>(8, 8, 16), SIMD3<UInt32>(7, 5, 3), SIMD3<UInt32>(1, 1, 1)] {
            let cells = Int(res.x * res.y * res.z)
            let counts = try XCTUnwrap(engine.device.makeBuffer(length: cells * 4, options: .storageModeShared))
            let offs = try XCTUnwrap(engine.device.makeBuffer(length: (cells + 1) * 4, options: .storageModeShared))
            let cp = counts.contents().bindMemory(to: UInt32.self, capacity: cells)
            var input: [UInt32] = []
            for i in 0 ..< cells { let v = UInt32.random(in: 0 ... 12, using: &rng); cp[i] = v; input.append(v) }
            var u = MLSUniforms(gridRes: SIMD4(res, 0), dxParams: .zero, matParams: .zero, boundsMin: .zero,
                                boundsMax: .zero, gravity: .zero, plasticA: .zero, plasticB: .zero)
            // A warm-up dispatch first (the first use of a pipeline pays for its setup).
            for warm in [true, false] where warm {
                let wcb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
                let wenc = try XCTUnwrap(wcb.makeComputeCommandEncoder())
                wenc.setComputePipelineState(pso)
                wenc.setBuffer(offs, offset: 0, index: 0)   // scans the (zero) offsets buffer in place
                wenc.setBuffer(offs, offset: 0, index: 1)
                wenc.setBytes(&u, length: MemoryLayout<MLSUniforms>.stride, index: 2)
                wenc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: min(1024, pso.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                wenc.endEncoding()
                wcb.commit(); wcb.waitUntilCompleted()   // gpu-ok: test-time warm-up
            }
            let cb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(pso)
            enc.setBuffer(counts, offset: 0, index: 0)
            enc.setBuffer(offs, offset: 0, index: 1)
            enc.setBytes(&u, length: MemoryLayout<MLSUniforms>.stride, index: 2)
            let w = min(1024, pso.maxTotalThreadsPerThreadgroup)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()   // gpu-ok: test-time readback
            XCTAssertNil(cb.error)
            let op = offs.contents().bindMemory(to: UInt32.self, capacity: cells + 1)
            var sum: UInt32 = 0, wrong = 0, nonzero = 0
            for i in 0 ..< cells {
                if op[i] != sum { wrong += 1 }
                sum += input[i]
                if cp[i] != 0 { nonzero += 1 }
            }
            XCTAssertEqual(wrong, 0, "offsets wrong at \(res)")
            XCTAssertEqual(op[cells], sum, "total at \(res)")
            XCTAssertEqual(nonzero, 0, "counts not zeroed at \(res)")
            let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            print("MLSMPMScan: \(cells) cells in \(ms) ms")
            if cells > 20_000 { XCTAssertLessThan(ms, 0.5, "scan cost") }
        }
    }
}
