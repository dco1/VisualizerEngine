import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// `exposureMeterGridSize` (VZ-0197): the auto-exposure meter samples 256 × 32 cells by LINEAR
/// index over a (w/8 × h/8) grid in normalised UV, so the strip of the frame it meters depends on
/// the grid size it is handed (VZ-0172). The REAL `illumi_exposure_estimate`, on the same scene (a
/// frame whose brightness falls from top to bottom) rendered at two internal scales of one canvas:
/// handed each frame's own size it meters two different strips; handed one pinned grid it meters
/// the same positions, and the two scales agree.
@MainActor
final class IlluminatoramaExposureMeterGridTests: XCTestCase {

    private static let fillKernel = """

    kernel void test_fill_gradient(texture2d<half, access::write> out [[texture(0)]],
                                   uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        float v = (float(gid.y) + 0.5) / float(out.get_height());
        // Bright window at the top, a dark room below: 6 stops over the frame height.
        float L = exp2(2.0 - 6.0 * v);
        out.write(half4(half3(L), 1.0h), gid);
    }
    """

    private func meter(_ device: MTLDevice, _ queue: MTLCommandQueue, _ lib: MTLLibrary,
                       frame: SIMD2<Int>, grid: SIMD2<Int>) throws -> Float {
        let fill = try device.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "test_fill_gradient")))  // gpu-ok: test harness
        let est = try device.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "illumi_exposure_estimate")))  // gpu-ok: test harness
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: frame.x, height: frame.y,
                                                         mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
        let tex = try XCTUnwrap(device.makeTexture(descriptor: d))
        let state = try XCTUnwrap(device.makeBuffer(length: 16, options: .storageModeShared))
        state.contents().storeBytes(of: SIMD4<Float>(0, 1, 0, 1.0 / 60), as: SIMD4<Float>.self)
        let hist = try XCTUnwrap(device.makeBuffer(length: 4096, options: .storageModeShared))
        let cb = try XCTUnwrap(queue.makeCommandBuffer())
        do {
            let e = try XCTUnwrap(cb.makeComputeCommandEncoder())
            e.setComputePipelineState(fill)
            e.setTexture(tex, index: 0)
            e.dispatchThreads(MTLSize(width: frame.x, height: frame.y, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            e.endEncoding()
        }
        do {
            let e = try XCTUnwrap(cb.makeComputeCommandEncoder())
            e.setComputePipelineState(est)
            e.setTexture(tex, index: 0)
            e.setBuffer(state, offset: 0, index: 0)
            var size = SIMD2<UInt32>(UInt32(grid.x), UInt32(grid.y))
            e.setBytes(&size, length: 8, index: 1)
            var params = SIMD4<Float>(-2.2, 0.5, 3, 0.05)
            e.setBytes(&params, length: 16, index: 2)
            var params2 = SIMD4<Float>(0, 0, 0, 0)
            e.setBytes(&params2, length: 16, index: 3)
            var params3 = SIMD4<Float>(0, 0.5, 0, 0)
            e.setBytes(&params3, length: 16, index: 4)
            e.setBuffer(hist, offset: 0, index: 5)
            e.setThreadgroupMemoryLength(256 * 4, index: 0)
            e.setThreadgroupMemoryLength(256 * 4, index: 1)
            e.setThreadgroupMemoryLength(64 * 4, index: 2)
            e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            e.endEncoding()
        }
        cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        XCTAssertNil(cb.error)
        // ExposureState.newTargetLogLum (slot 2): the log2 luminance the meter targeted this frame.
        return state.contents().load(fromByteOffset: 8, as: Float.self)
    }

    func testPinnedGridMetersTheSamePositionsAtEveryScale() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaTonemap.metal")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no shader") }
        let lib = try device.makeLibrary(source: MetalSourceLoader.source(contentsOf: url) + Self.fillKernel, options: nil)
        // A 2880 × 1620 canvas at internal scale 1.5 (the still) and 1.0 (the live frame).
        let still = SIMD2(4320, 2430), live = SIMD2(2880, 1620)
        let atStill = try meter(device, queue, lib, frame: still, grid: still)
        let liveOwnGrid = try meter(device, queue, lib, frame: live, grid: live)
        let livePinned = try meter(device, queue, lib, frame: live, grid: still)
        print("meter log2 L — still \(atStill), live (own grid) \(liveOwnGrid), live (pinned) \(livePinned)")
        // The bug the pin exists for: the same frame at two scales meters two different strips.
        XCTAssertGreaterThan(abs(liveOwnGrid - atStill), 0.1, "the probe must reproduce the scale-dependent meter")
        // The pin: the same UV positions, so the same answer (bilinear taps of the same gradient).
        XCTAssertEqual(livePinned, atStill, accuracy: 0.01)
    }
}
