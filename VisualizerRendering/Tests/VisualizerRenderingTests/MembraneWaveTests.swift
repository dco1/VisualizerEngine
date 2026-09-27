import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// GPU test for `membrane_step` (MembraneWave.metal): a poke in the middle of a flat,
/// square, pinned-rim membrane spreads symmetrically (no axis bias from the ping-pong
/// update) and dies away under damping (the solver is stable at the host's substep size).
final class MembraneWaveTests: XCTestCase {
    struct Params { var rows: UInt32, cols: UInt32, wrapCols: UInt32, impulseCount: UInt32
                    var dt: Float, c2: Float, damping: Float, uniformForce: Float }
    struct Impulse { var gridPos: SIMD2<Float>; var radius: Float; var velocity: Float }

    func testPokeSpreadsSymmetricallyAndDecays() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let src = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/MembraneWave.metal")
        let lib = try device.makeLibrary(source: try String(contentsOf: src, encoding: .utf8), options: nil)
        let pso = try device.makeComputePipelineState(function: lib.makeFunction(name: "membrane_step")!)   // gpu-ok: test-only
        let queue = device.makeCommandQueue()!   // gpu-ok: test-only
        let n = 41
        var mask = [Float](repeating: 1, count: n * n)
        for i in 0 ..< n { mask[i] = 0; mask[(n - 1) * n + i] = 0; mask[i * n] = 0; mask[i * n + n - 1] = 0 }
        let inv = [SIMD2<Float>](repeating: SIMD2(1, 1), count: n * n)   // unit spacing
        let zeros = [Float](repeating: 0, count: n * n)
        let hA = device.makeBuffer(bytes: zeros, length: n * n * 4)!, hB = device.makeBuffer(bytes: zeros, length: n * n * 4)!
        let v = device.makeBuffer(bytes: zeros, length: n * n * 4)!
        let mk = device.makeBuffer(bytes: mask, length: n * n * 4)!
        let isb = device.makeBuffer(bytes: inv, length: n * n * 8)!
        func run(_ steps: Int, poke: Bool) -> [Float] {
            var cur = [hA, hB]
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            enc.setBuffer(v, offset: 0, index: 1)
            enc.setBuffer(mk, offset: 0, index: 2); enc.setBuffer(isb, offset: 0, index: 3)
            var imp = Impulse(gridPos: SIMD2(20, 20), radius: 2, velocity: 1)
            enc.setBytes(&imp, length: MemoryLayout<Impulse>.stride, index: 5)
            for s in 0 ..< steps {
                var p = Params(rows: UInt32(n), cols: UInt32(n), wrapCols: 0, impulseCount: poke && s == 0 ? 1 : 0,
                               dt: 0.02, c2: 20, damping: 1.5, uniformForce: 0)   // c·dt/Δx ≈ 0.09
                enc.setBuffer(cur[0], offset: 0, index: 0)
                enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 4)
                enc.setBuffer(cur[1], offset: 0, index: 6)
                enc.dispatchThreads(MTLSize(width: n * n, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                cur.swapAt(0, 1)
            }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()   // gpu-ok: test-only readback
            let ptr = cur[0].contents().bindMemory(to: Float.self, capacity: n * n)
            return (0 ..< n * n).map { ptr[$0] }
        }
        let early = run(40, poke: true)          // even step count ⇒ result lands in hA
        var asym: Float = 0, peak: Float = 0
        for r in 0 ..< n { for c in 0 ..< n {
            asym = max(asym, abs(early[r * n + c] - early[c * n + r]))
            peak = max(peak, abs(early[r * n + c]))
        } }
        XCTAssertGreaterThan(peak, 1e-3, "the poke should move the membrane")
        XCTAssertLessThan(asym, peak * 1e-3, "propagation must be symmetric")
        let late = run(2000, poke: false)
        XCTAssertLessThan(late.map(abs).max()!, peak * 0.05, "damping must settle the membrane")
    }
}
