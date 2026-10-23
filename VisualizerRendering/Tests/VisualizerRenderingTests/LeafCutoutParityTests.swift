import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// DH-1487 — the leaf cutout: the shipped `IlluminatoramaLeafCutout.h` compiled from source agrees with
/// the Swift mirror point for point (so the bake's leaf areas are the shader's), every species cuts a
/// real share of its card, and the no-cutout code keeps everything.
final class LeafCutoutParityTests: XCTestCase {

    private static func shaderURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/\(name)")
    }

    private static let probeKernel = """

    kernel void dh1487CutoutProbe(const device float2* q [[buffer(0)]], device uint* keep [[buffer(1)]],
                                  uint gid [[thread_position_in_grid]]) {
        keep[gid] = illumiLeafCutoutKeep(q[gid]) ? 1u : 0u;
    }
    """

    @MainActor
    func testShaderAgreesWithTheSwiftMirror() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {  // gpu-ok: test harness
            throw XCTSkip("no Metal")
        }
        let src = "#include <metal_stdlib>\nusing namespace metal;\n"
            + (try MetalSourceLoader.source(contentsOf: Self.shaderURL("IlluminatoramaLeafCutout.h"))) + Self.probeKernel
        let lib = try device.makeLibrary(source: src, options: nil)
        guard let fn = lib.makeFunction(name: "dh1487CutoutProbe") else { throw XCTSkip("probe missing") }
        let pipe = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        var pts: [SIMD2<Float>] = []
        for code in 0...4 {
            for j in 0..<160 { for i in 0..<120 {
                let v = (Float(j) + 0.37) / 160, u = -1 + 2 * (Float(i) + 0.61) / 120
                pts.append(SIMD2(u, code == 0 ? 0 : v + 2 * Float(code)))
            } }
        }
        guard let qb = device.makeBuffer(bytes: pts, length: pts.count * 8, options: .storageModeShared),
              let ob = device.makeBuffer(length: pts.count * 4, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no buffers") }
        enc.setComputePipelineState(pipe); enc.setBuffer(qb, offset: 0, index: 0); enc.setBuffer(ob, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        let gpu = ob.contents().bindMemory(to: UInt32.self, capacity: pts.count)
        var mismatch = 0
        for (i, p) in pts.enumerated() where (gpu[i] != 0) != IlluminatoramaLeafCutout.keep(p) { mismatch += 1 }
        print("DH-1487 cutout parity: \(pts.count) points, \(mismatch) mismatches")
        // Float transcendental ulps can flip a point lying on an outline; a handful in 96 000 is that.
        XCTAssertLessThanOrEqual(mismatch, 20, "the Swift mirror drifted from IlluminatoramaLeafCutout.h")
    }

    func testEverySpeciesCutsAShareOfItsCard() {
        let card = IlluminatoramaLeafCutout.area(code: 0)
        XCTAssertEqual(card, 2 * (0.15 * 0.95 / 2 + 0.70 * 0.95 + 0.15 * 0.95 / 2), accuracy: 0.01)
        var line = "DH-1487 leaf areas (card units; card \(card), ovate 1.066):"
        for code in 1...4 {
            let a = IlluminatoramaLeafCutout.area(code: code)
            line += String(format: " %d:%.3f", code, a)
            XCTAssertGreaterThan(a, 0.35 * card, "code \(code) keeps too little of its card")
            XCTAssertLessThan(a, 0.85 * card, "code \(code) cuts too little to be a shape")
        }
        print(line)
    }
}
