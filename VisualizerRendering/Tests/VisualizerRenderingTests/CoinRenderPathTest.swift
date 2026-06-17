import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Diagnoses "coins invisible in the app": runs the FULL render path (solver step
/// → coinDeriveTransforms → coin_expand_instances) headlessly and reads back the
/// renderer's output position buffer. If instance 0's expanded geometry sits near
/// the coin's COM, the GPU render data is correct and the invisibility is a
/// SceneKit-side issue (bounds/material). If it's parked at −100000, the expand
/// path is broken.
@MainActor
final class CoinRenderPathTest: XCTestCase {
    func testExpandWritesVisibleGeometry() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let engine = SimEngine(device: device)

        // Compile both CoinDEM kernels (solver + expand) from source.
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard let src = try? String(contentsOf: dir, encoding: .utf8) else { throw XCTSkip("no shader") }
        let lib = try device.makeLibrary(source: src, options: nil)

        guard let solver = CoinDEMSolver(engine: engine, library: lib, maxCoins: 16,
                                         coinRadius: 0.12, halfThickness: 0.009,
                                         boundsMin: SIMD3(-2, -2, -2), boundsMax: SIMD3(2, 3, 2)),
              let queue = device.makeCommandQueue()  // gpu-ok: test harness
        else { throw XCTSkip("solver init") }
        solver.floorY = -1.5
        solver.setColliders([.plane(normal: SIMD3(0,1,0), offset: 0)])

        // Build the renderer with an expand pipeline from the SAME library.
        let mesh = CoinMesh.make(radius: 0.12, halfThickness: 0.009, radialSegments: 12)
        let renderer = try makeRenderer(engine: engine, lib: lib, mesh: mesh, maxInstances: 16)

        let slot = solver.spawn(at: SIMD3(0, 0.5, 0))
        XCTAssertEqual(slot, 0)
        renderer.setActiveInstanceCount(solver.highWater)

        for _ in 0..<60 {
            guard let cb = queue.makeCommandBuffer() else { return }
            solver.encode(to: cb, wallDt: 1.0/60.0)
            renderer.dispatchExpand(transforms: solver.transformBuffer, bodyType: solver.bodyTypeBuffer, in: cb)
            cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        }

        let c = renderer.debugInstanceCentroid(0)
        print("RENDER_PATH_CENTROID = \(c)")
        XCTAssertGreaterThan(c.y, -100.0, "instance 0 still parked off-screen — expand path broken")
        // Coin rests near floorY + h = 0.009.
        XCTAssertEqual(c.y, 0.009, accuracy: 0.05, "expanded coin not at the rest height")
        XCTAssertLessThan(abs(c.x), 0.05)
        XCTAssertLessThan(abs(c.z), 0.05)
    }

    // Renderer needs its expand pipeline from the runtime-compiled lib; the
    // production init resolves it via Bundle.module which the SwiftPM CLI lacks.
    private func makeRenderer(engine: SimEngine, lib: MTLLibrary,
                              mesh: CoinMesh.Mesh, maxInstances: Int) throws -> CoinInstancedRenderer {
        guard let fn = lib.makeFunction(name: "coin_expand_instances"),
              let pipeline = try? engine.device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        else { throw XCTSkip("no expand fn") }
        guard let r = CoinInstancedRenderer(engine: engine, pipeline: pipeline,
                                            mesh: mesh, maxInstances: maxInstances) else {
            throw XCTSkip("renderer init")
        }
        return r
    }
}
