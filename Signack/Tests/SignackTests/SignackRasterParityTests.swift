import XCTest
import Metal
import simd
import SignackGPU

/// S4's coverage gate: the software rasterizer (`SignackRaster.metal`) covers EXACTLY the pixels the
/// HARDWARE rasterizer covers, with the same front-facing bit, and both agree with an independent CPU
/// reading of the rule (`SignackRasterReference`). If the rule were off — a different fill
/// convention, a different snap — a soft triangle and the hardware triangle beside it would leave a
/// crack or double-cover the pixels on their shared edge.
///
/// The mesh is a watertight planar triangulation of sub-pixel to few-pixel triangles, vertices on
/// the 1/256 grid, with whole rows/columns of vertices exactly on pixel centres (so edges pass
/// exactly through samples and the tie rule decides), half the triangles wound the other way.
final class SignackRasterParityTests: XCTestCase {

    func testCoverageMatchesTheHardwareRasterizerAndTheCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw XCTSkip("no Metal device")
        }
        guard let pipelines = SignackGPUPipelines.shared(device: device) else {
            return XCTFail("SignackGPU pipelines failed to build")
        }
        let W = 64, H = 64
        for (spacing, seed) in [(Float(0.7), UInt64(1)), (1.3, 2), (2.6, 3), (0.45, 4)] {
            let tris = Self.mesh(width: W, height: H, spacing: spacing, seed: seed)
            let cpu = SignackRasterReference.rasterize(tris, width: W, height: H)
            // The rule is watertight: every interior sample is owned by exactly one triangle.
            let soft = try Self.soft(tris, W, H, device, queue, pipelines)
            let hard = try Self.hardware(tris, W, H, device, queue)
            var covered = 0, mismatchCPU = 0, mismatchHW = 0, frontHW = 0, depthBits = 0
            for i in 0..<(W * H) {
                let c = cpu[i], s = soft[i], h = hard[i]
                if c != nil { covered += 1 }
                if c?.triangle != s?.triangle || c?.front != s?.front { mismatchCPU += 1 }
                if let c, let s, c.depth.bitPattern != s.depth.bitPattern { depthBits += 1 }
                if s?.triangle != h?.triangle {
                    mismatchHW += 1
                    func show(_ k: Int?) -> String {
                        guard let k else { return "none" }
                        let t = tris[k]
                        return "#\(k) (\(t.v0x * 256),\(t.v0y * 256)) (\(t.v1x * 256),\(t.v1y * 256)) (\(t.v2x * 256),\(t.v2y * 256))"
                    }
                    print("  px (\(i % W),\(i / W)) centre (\((i % W) * 256 + 128),\((i / W) * 256 + 128)): soft \(show(s?.triangle)) · hardware \(show(h?.triangle))")
                }
                else if let s, let h, s.front != h.front { frontHW += 1 }
            }
            print(String(format: "SignackRaster parity, spacing %.2f px, %d triangles: %d / %d px covered, "
                         + "soft≠CPU %d, soft≠hardware %d (front-facing %d), depth bits ≠ CPU %d",
                         spacing, tris.count, covered, W * H, mismatchCPU, mismatchHW, frontHW, depthBits))
            XCTAssertEqual(covered, W * H, "spacing \(spacing): the mesh covers the target, so every centre is owned")
            XCTAssertEqual(mismatchCPU, 0, "spacing \(spacing): GPU kernel vs CPU reading of the rule")
            XCTAssertEqual(depthBits, 0, "spacing \(spacing): depth must round identically (the id pass relies on it)")
            XCTAssertEqual(mismatchHW, 0, "spacing \(spacing): soft coverage vs the hardware rasterizer")
            XCTAssertEqual(frontHW, 0, "spacing \(spacing): front-facing vs the hardware")
        }
    }

    /// Overlapping triangles at different depths: the nearest wins, as in the CPU reference.
    func testNearestDepthWins() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let pipelines = SignackGPUPipelines.shared(device: device) else { throw XCTSkip("no Metal") }
        var rng = SplitMix(seed: 9)
        let W = 48, H = 40
        var tris: [SignackRasterTriangle] = []
        for k in 0..<4000 {
            let c = SIMD2(Float.random(in: 0..<Float(W), using: &rng), Float.random(in: 0..<Float(H), using: &rng))
            func v() -> SIMD3<Float> {
                let p = c + SIMD2(Float.random(in: -2...2, using: &rng), Float.random(in: -2...2, using: &rng))
                return SIMD3(p, Float.random(in: 0.1...0.9, using: &rng))
            }
            tris.append(SignackRasterTriangle(v(), v(), v(), idX: k % 3 == 0 ? SignackRasterTriangle.cullBackFaces : 0,
                                              idY: UInt32(k)))
        }
        let cpu = SignackRasterReference.rasterize(tris, width: W, height: H)
        let soft = try Self.soft(tris, W, H, device, queue, pipelines)
        var bad = 0, covered = 0
        for i in 0..<(W * H) {
            if cpu[i] != nil { covered += 1 }
            // A tie in depth may go either way (the documented race); anything else must match.
            if cpu[i]?.depth != soft[i]?.depth || (cpu[i]?.triangle != soft[i]?.triangle
                && !Self.tied(cpu[i], soft[i], tris)) { bad += 1 }
        }
        XCTAssertGreaterThan(covered, W * H / 2)
        XCTAssertEqual(bad, 0)
    }

    // MARK: - Harness

    struct Sample { var triangle: Int; var front: Bool; var depth: Float }

    static func tied(_ a: SignackRasterReference.Sample?, _ b: Sample?, _ t: [SignackRasterTriangle]) -> Bool {
        guard let a, let b else { return false }
        return a.depth == b.depth
    }

    /// The software rasterizer's nearest triangle per pixel.
    static func soft(_ tris: [SignackRasterTriangle], _ W: Int, _ H: Int, _ device: MTLDevice,
                     _ queue: MTLCommandQueue, _ pipelines: SignackGPUPipelines) throws -> [Sample?] {
        guard let raster = SignackRaster(device: device), raster.reserve(width: W, height: H, triangles: tris.count),
              let cb = queue.makeCommandBuffer() else { throw XCTSkip("raster buffers") }
        tris.withUnsafeBytes { raster.triangles.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        raster.state.contents().storeBytes(of: UInt32(tris.count), toByteOffset: SignackRaster.appendCounterOffset, as: UInt32.self)
        raster.encode(commandBuffer: cb, scissor: SignackRasterRect(x: 0, y: 0, width: W, height: H), pipelines: pipelines)
        cb.commit(); cb.waitUntilCompleted()
        XCTAssertEqual(raster.stats.lastCount, tris.count)
        XCTAssertEqual(raster.stats.clipped, 0)
        let d = raster.depth.contents().bindMemory(to: UInt32.self, capacity: W * H)
        let ids = raster.ids.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: W * H)
        return (0..<(W * H)).map { i in
            d[i] == SignackRaster.emptyDepth ? nil
                : Sample(triangle: Int(ids[i].y), front: ids[i].x & 0x8000_0000 != 0, depth: Float(bitPattern: d[i]))
        }
    }

    /// The hardware rasterizer's coverage of the same triangles (no depth test — the partition has
    /// no overlap): each fragment writes its primitive id + 1 and the front-facing bit.
    static func hardware(_ tris: [SignackRasterTriangle], _ W: Int, _ H: Int, _ device: MTLDevice,
                         _ queue: MTLCommandQueue) throws -> [Sample?] {
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        struct VO { float4 p [[position]]; };
        vertex VO vs(uint v [[vertex_id]], const device float4* pos [[buffer(0)]]) { VO o; o.p = pos[v]; return o; }
        fragment uint2 fs(VO in [[stage_in]], uint pid [[primitive_id]], bool ff [[front_facing]]) {
            return uint2(ff ? 1u : 0u, pid + 1u);
        }
        """
        let lib = try device.makeLibrary(source: src, options: nil)
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = lib.makeFunction(name: "vs"); d.fragmentFunction = lib.makeFunction(name: "fs")
        d.colorAttachments[0].pixelFormat = .rg32Uint
        let pso = try device.makeRenderPipelineState(descriptor: d)
        var pos: [SIMD4<Float>] = []
        for t in tris {
            for (x, y, z) in [(t.v0x, t.v0y, t.v0z), (t.v1x, t.v1y, t.v1z), (t.v2x, t.v2y, t.v2z)] {
                pos.append(SIMD4(x / Float(W) * 2 - 1, 1 - y / Float(H) * 2, z, 1))
            }
        }
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Uint, width: W, height: H, mipmapped: false)
        td.usage = [.renderTarget, .shaderRead]; td.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: td),
              let pbuf = device.makeBuffer(bytes: pos, length: pos.count * 16, options: .storageModeShared),
              let cb = queue.makeCommandBuffer() else { throw XCTSkip("hardware target") }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = tex
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { throw XCTSkip("encoder") }
        enc.setRenderPipelineState(pso)
        enc.setFrontFacing(.counterClockwise)
        enc.setCullMode(.none)
        enc.setVertexBuffer(pbuf, offset: 0, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: pos.count)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        var px = [SIMD2<UInt32>](repeating: .zero, count: W * H)
        tex.getBytes(&px, bytesPerRow: W * 8, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0)
        return px.map { $0.y == 0 ? nil : Sample(triangle: Int($0.y) - 1, front: $0.x == 1, depth: 0) }
    }

    /// A jittered grid over (and past) the target, split into two triangles per cell. Every third
    /// row and column sits exactly on pixel centres; the rest are jittered and snapped to 1/256 px.
    static func mesh(width W: Int, height H: Int, spacing s: Float, seed: UInt64) -> [SignackRasterTriangle] {
        var rng = SplitMix(seed: seed)
        let nx = Int((Float(W) + 2) / s) + 2, ny = Int((Float(H) + 2) / s) + 2
        func snap(_ v: Float) -> Float { (v * 256).rounded() / 256 }
        var grid = [[SIMD3<Float>]](repeating: [], count: ny)
        for j in 0..<ny {
            for i in 0..<nx {
                var x = -1 + Float(i) * s, y = -1 + Float(j) * s
                // Pin to the nearest centre only where that moves the vertex less than the jitter
                // would — a larger move folds the triangulation (overlap is not what is under test).
                let cx = x.rounded(.down) + 0.5, cy = y.rounded(.down) + 0.5
                if i % 3 == 0 && abs(cx - x) <= 0.3 * s { x = cx } else if i > 0 && i < nx - 1 { x += Float.random(in: -0.3...0.3, using: &rng) * s }
                if j % 3 == 0 && abs(cy - y) <= 0.3 * s { y = cy } else if j > 0 && j < ny - 1 { y += Float.random(in: -0.3...0.3, using: &rng) * s }
                if i == 0 { x = -1.5 }; if j == 0 { y = -1.5 }
                grid[j].append(SIMD3(snap(x), snap(y), 0.25 + 0.5 * Float(i + j) / Float(nx + ny)))
            }
        }
        // Columns pinned to centres can collide with a jittered neighbour: keep x/y monotone.
        for j in 0..<ny { for i in 1..<nx where grid[j][i].x <= grid[j][i - 1].x { grid[j][i].x = grid[j][i - 1].x + 1.0 / 256 } }
        for i in 0..<nx { for j in 1..<ny where grid[j][i].y <= grid[j - 1][i].y { grid[j][i].y = grid[j - 1][i].y + 1.0 / 256 } }
        var tris: [SignackRasterTriangle] = []
        for j in 0..<(ny - 1) {
            for i in 0..<(nx - 1) {
                let a = grid[j][i], b = grid[j][i + 1], c = grid[j + 1][i], d = grid[j + 1][i + 1]
                let diag = (i + j) % 2 == 0
                for (p, q, r) in diag ? [(a, b, d), (a, d, c)] : [(a, b, c), (b, d, c)] {
                    let n = UInt32(tris.count)
                    tris.append(n % 2 == 0 ? SignackRasterTriangle(p, q, r, idX: 0, idY: n)
                                           : SignackRasterTriangle(p, r, q, idX: 0, idY: n))
                }
            }
        }
        return tris
    }
}

/// SplitMix64 — deterministic test randomness.
struct SplitMix: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
