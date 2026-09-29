import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// `illumi_dof_tile_parallel` (the fast gather's max-CoC tile reduction, VZ-0197) against the
/// serial `illumi_dof_tile` it replaces there: the SAME tile map, bit for bit, on a depth field
/// with a near subject, a far wall, sky and a ragged frame edge (partial tiles), for the
/// threadgroup shapes the renderer can pick.
@MainActor
final class IlluminatoramaDOFTileParallelTests: XCTestCase {

    /// Mirrors the Metal `DOFParams` (stride 144).
    private struct Params {
        var invProjection: simd_float4x4
        var focusDist: Float, cocCoefficient: Float, maxRadius: Float
        var blades: Float = 0, bladeRotation: Float = 0, catsEye: Float = 0
        var width: UInt32, height: UInt32, tileW: UInt32, tileH: UInt32, tileSize: UInt32
        var prefilterScale: Float = 1, cocFloor: Float, subjectAware: Float = 1
        var fastGather: Float = 1, halfMinCoC: Float = 3, quarterMinCoC: Float = 0
    }

    func testParallelTileMapIsBitIdenticalToTheSerialOne() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaDOF.metal")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no shader") }
        let lib = try MetalSourceLoader.makeLibrary(device: device, contentsOf: url)
        let serial = try device.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "illumi_dof_tile")))  // gpu-ok: test harness
        let par = try device.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "illumi_dof_tile_parallel")))  // gpu-ok: test harness

        // A frame whose size is NOT a multiple of the tile (partial tiles on both edges).
        let W = 203, H = 117, tile = 16
        let tw = (W + tile - 1) / tile, th = (H + tile - 1) / tile
        // A real perspective projection, so the kernels' own unprojection is exercised.
        let cam = IlluminatoramaCamera(position: .zero, target: SIMD3(0, 0, -1), up: SIMD3(0, 1, 0),
                                       fovYRadians: 0.47, aspect: Float(W) / Float(H), zNear: 0.01, zFar: 2000)
        let proj = cam.projectionMatrix
        func depthAt(viewZ z: Float) -> Float {        // device depth of a point `z` m down the axis
            let c = proj * SIMD4<Float>(0, 0, -z, 1)
            return c.z / c.w
        }
        var dz = [Float](repeating: 0, count: W * H)
        var rng = SystemRandomNumberGenerator()
        for y in 0..<H {
            for x in 0..<W {
                let fx = Float(x) / Float(W), fy = Float(y) / Float(H)
                let z: Float
                if (fx - 0.3) * (fx - 0.3) + (fy - 0.6) * (fy - 0.6) < 0.02 { z = 0.55 + 0.02 * Float.random(in: 0...1, using: &rng) }   // near subject
                else if fy < 0.25 { z = 1e9 }                                                                                             // sky
                else { z = 0.7 + 1.5 * fy + 0.3 * Float.random(in: 0...1, using: &rng) }                                                 // wall / table
                dz[y * W + x] = z > 1e8 ? 1.0 : depthAt(viewZ: z)
            }
        }
        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: W, height: H, mipmapped: false)
        dd.usage = [.shaderRead]; dd.storageMode = .shared
        let depth = try XCTUnwrap(device.makeTexture(descriptor: dd))
        dz.withUnsafeBytes { depth.replace(region: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0,
                                           withBytes: $0.baseAddress!, bytesPerRow: W * 4) }
        var p = Params(invProjection: proj.inverse, focusDist: 0.62, cocCoefficient: 73.1, maxRadius: 32,
                       width: UInt32(W), height: UInt32(H), tileW: UInt32(tw), tileH: UInt32(th),
                       tileSize: UInt32(tile), cocFloor: 0.15)

        func tileMap(_ run: (MTLComputeCommandEncoder, MTLTexture) -> Void) throws -> [SIMD2<Float>] {
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float, width: tw, height: th, mipmapped: false)
            td.usage = [.shaderRead, .shaderWrite]; td.storageMode = .shared
            let out = try XCTUnwrap(device.makeTexture(descriptor: td))
            let cb = try XCTUnwrap(queue.makeCommandBuffer())
            let e = try XCTUnwrap(cb.makeComputeCommandEncoder())
            e.setTexture(depth, index: 0)
            e.setTexture(out, index: 1)
            e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
            run(e, out)
            e.endEncoding()
            cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
            XCTAssertNil(cb.error)
            var v = [SIMD2<Float>](repeating: .zero, count: tw * th)
            v.withUnsafeMutableBytes { out.getBytes($0.baseAddress!, bytesPerRow: tw * 8,
                                                    from: MTLRegionMake2D(0, 0, tw, th), mipmapLevel: 0) }
            return v
        }
        XCTAssertEqual(MemoryLayout<Params>.stride, 144)
        let ref = try tileMap { e, _ in
            e.setComputePipelineState(serial)
            e.dispatchThreads(MTLSize(width: tw, height: th, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        }
        XCTAssertGreaterThan(ref.map(\.x).max() ?? 0, 10, "the probe must hold defocused tiles")
        XCTAssertGreaterThan(ref.map(\.y).max() ?? 0, 0, "…and near-field ones")
        for side in [16, 8, 4] {
            let got = try tileMap { e, _ in
                e.setComputePipelineState(par)
                e.dispatchThreadgroups(MTLSize(width: tw, height: th, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: side, height: side, depth: 1))
            }
            for i in 0..<(tw * th) {
                XCTAssertEqual(got[i].x.bitPattern, ref[i].x.bitPattern, "tile \(i) max|CoC| differs (side \(side)): \(got[i]) vs \(ref[i])")
                XCTAssertEqual(got[i].y.bitPattern, ref[i].y.bitPattern, "tile \(i) near max differs (side \(side)): \(got[i]) vs \(ref[i])")
            }
        }
    }
}
