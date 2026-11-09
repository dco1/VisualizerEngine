import CoreGraphics
import ImageIO
import Metal
import UniformTypeIdentifiers
import XCTest
import simd
@testable import VisualizerRendering

/// Films a pour into a cup with `MLSMPMSolver.emitters` + `LiquidSurfaceRenderer`, for looking at (not an assertion beyond
/// "it ran and drew something"). Off unless `LIQUID_FILM_DIR` names a folder: PNG frames go to `<dir>/<view>/f0000.png`.
/// `LIQUID_FILM_FRAMES`, `LIQUID_FILM_SIZE`, `LIQUID_FILM_CELL` (m), `LIQUID_FILM_VIEW=wide|close|side`.
@MainActor
final class LiquidPourFilmTests: XCTestCase {

    private static let sceneShader = """
    #include <metal_stdlib>
    using namespace metal;
    struct SU { float4x4 invVP; float4x4 vp; float4 eye; float4 cup; };   // cup: x inner radius, y height, z wall
    struct VO { float4 position [[position]]; float2 uv; };
    vertex VO scene_vs(uint id [[vertex_id]]) {
        float2 c = float2((id << 1) & 2, id & 2);
        return VO{ float4(c * 2 - 1, 0, 1), float2(c.x, 1 - c.y) };
    }
    struct FO { float4 color [[color(0)]]; float depth [[depth(any)]]; };
    inline float3 shade(float3 p, float3 n, float3 albedo) {
        float3 L = normalize(float3(0.5, 0.9, 0.35)), L2 = normalize(float3(-0.7, 0.4, -0.3));
        return albedo * (0.10 + 0.85 * saturate(dot(n, L)) + 0.25 * saturate(dot(n, L2)) * float3(0.6, 0.75, 1.0));
    }
    fragment FO scene_fs(VO in [[stage_in]], constant SU& u [[buffer(0)]]) {
        float4 f = u.invVP * float4(in.uv.x * 2 - 1, 1 - in.uv.y * 2, 1, 1);
        float3 ro = u.eye.xyz, rd = normalize(f.xyz / f.w - ro);
        float R = u.cup.x, H = u.cup.y, Ro = R + u.cup.z;
        float best = 1e9; float3 n = float3(0), alb = float3(0);
        // The table.
        if (rd.y < 0) {
            float t = (-u.cup.z - ro.y) / rd.y; float3 p = ro + rd * t;
            float2 g = abs(fract(p.xz * 12.5) - 0.5);
            float line = smoothstep(0.47, 0.5, max(g.x, g.y));
            best = t; n = float3(0, 1, 0); alb = mix(float3(0.16, 0.15, 0.14), float3(0.30, 0.28, 0.26), line) * exp(-length(p.xz) * 1.2);
        }
        // The cup's outside.
        float a = dot(rd.xz, rd.xz), b = dot(ro.xz, rd.xz), c = dot(ro.xz, ro.xz);
        float disc = b * b - a * (c - Ro * Ro);
        if (disc > 0 && a > 1e-9) {
            float t = (-b - sqrt(disc)) / a; float3 p = ro + rd * t;
            if (t > 0 && t < best && p.y > -u.cup.z && p.y < H) { best = t; n = normalize(float3(p.x, 0, p.z)); alb = float3(0.78, 0.76, 0.72); }
        }
        // Its rim, inside wall and floor, seen through the opening.
        if (rd.y < 0) {
            float t = (H - ro.y) / rd.y; float3 p = ro + rd * t; float r = length(p.xz);
            if (t > 0 && t < best && r < Ro) {
                if (r > R) { best = t; n = float3(0, 1, 0); alb = float3(0.80, 0.78, 0.74); }
                else {
                    float di = b * b - a * (c - R * R);
                    float ti = (-b + sqrt(max(di, 0.0))) / a, tf = (0.0 - ro.y) / rd.y;
                    if (tf < ti) {
                        float3 q = ro + rd * tf; best = tf; n = float3(0, 1, 0);
                        float2 g = floor(q.xz * 50.0);
                        alb = fmod(g.x + g.y, 2.0) == 0.0 ? float3(0.72, 0.70, 0.66) : float3(0.46, 0.45, 0.43);
                    } else {
                        float3 q = ro + rd * ti; best = ti; n = -normalize(float3(q.x, 0, q.z));
                        alb = float3(0.70, 0.68, 0.64) * (fract(q.y * 50.0) < 0.5 ? 1.0 : 0.82);
                    }
                }
            }
        }
        FO o;
        if (best > 1e8) { o.color = float4(float3(0.012, 0.012, 0.016) + 0.03 * saturate(rd.y), 1); o.depth = 1; return o; }
        float3 p = ro + rd * best;
        float3 col = shade(p, n, alb);
        col = col / (1 + col);
        o.color = float4(pow(col * 1.6, float3(1 / 2.2)), 1);
        float4 cl = u.vp * float4(p, 1);
        o.depth = cl.z / cl.w;
        return o;
    }
    """

    private struct SceneUniforms { var invVP: simd_float4x4, vp: simd_float4x4, eye: SIMD4<Float>, cup: SIMD4<Float> }

    private func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let ys = 1 / tan(fovY / 2), xs = ys / aspect, zs = far / (near - far)
        return simd_float4x4(SIMD4(xs, 0, 0, 0), SIMD4(0, ys, 0, 0), SIMD4(0, 0, zs, -1), SIMD4(0, 0, zs * near, 0))
    }
    private func lookAt(_ eye: SIMD3<Float>, _ at: SIMD3<Float>) -> simd_float4x4 {
        let z = simd_normalize(eye - at), x = simd_normalize(simd_cross(SIMD3(0, 1, 0), z)), y = simd_cross(z, x)
        return simd_float4x4(SIMD4(x.x, y.x, z.x, 0), SIMD4(x.y, y.y, z.y, 0), SIMD4(x.z, y.z, z.z, 0),
                             SIMD4(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1))
    }

    /// A black studio with two soft boxes and a strip light, equirectangular.
    private func studio(_ device: MTLDevice) -> MTLTexture? {
        let w = 512, h = 256
        var px = [Float](repeating: 0, count: w * h * 4)
        for y in 0..<h { for x in 0..<w {
            let u = (Float(x) + 0.5) / Float(w), v = (Float(y) + 0.5) / Float(h)
            let el = (0.5 - v) * 180, az = (u - 0.5) * 360
            func box(_ a: Float, _ e: Float, _ wa: Float, _ we: Float) -> Float {
                let dx = abs(az - a) / wa, dy = abs(el - e) / we
                return max(0, min(1, (1 - dx) * 6)) * max(0, min(1, (1 - dy) * 6))
            }
            let c = 0.02 + 5.0 * box(40, 55, 28, 16) + 2.5 * box(-70, 35, 14, 30) + 3.0 * box(160, 60, 40, 8)
            let i = (y * w + x) * 4
            px[i] = c; px[i + 1] = c * 0.98; px[i + 2] = c * 0.95; px[i + 3] = 1
        } }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: w, height: h, mipmapped: false)
        d.usage = .shaderRead
        let t = device.makeTexture(descriptor: d)
        t?.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: px, bytesPerRow: w * 16)
        return t
    }

    private func writePNG(_ t: MTLTexture, to url: URL) {
        var bytes = [UInt8](repeating: 0, count: t.width * t.height * 4)
        t.getBytes(&bytes, bytesPerRow: t.width * 4, from: MTLRegionMake2D(0, 0, t.width, t.height), mipmapLevel: 0)
        let info = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        guard let ctx = CGContext(data: &bytes, width: t.width, height: t.height, bitsPerComponent: 8, bytesPerRow: t.width * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: info), let img = ctx.makeImage(),
              let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(dst, img, nil)
        CGImageDestinationFinalize(dst)
    }

    func testFilmAPourIntoACup() throws {
        let env = ProcessInfo.processInfo.environment
        guard let dirPath = env["LIQUID_FILM_DIR"] else { throw XCTSkip("set LIQUID_FILM_DIR to film") }
        let frames = Int(env["LIQUID_FILM_FRAMES"] ?? "") ?? 300
        let size = Int(env["LIQUID_FILM_SIZE"] ?? "") ?? 720
        let cell = Float(env["LIQUID_FILM_CELL"] ?? "") ?? 0.008
        let viewName = env["LIQUID_FILM_VIEW"] ?? "wide"
        let dir = URL(fileURLWithPath: dirPath).appendingPathComponent(env["LIQUID_FILM_NAME"] ?? viewName)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let engine = SimEngine.shared, device = engine.device
        let R: Float = 0.11, H: Float = 0.10, nozzleY: Float = 0.50
        let solver = try XCTUnwrap(MLSMPMSolver(engine: engine, boundsMin: SIMD3(-0.16, -0.03, -0.16), boundsMax: SIMD3(0.16, 0.56, 0.16),
                                                cellSize: cell, maxParticles: 80_000))
        solver.vessel = MLSMPMSolver.Vessel(floorY: 0, profile: [SIMD2(0, R), SIMD2(H, R)], friction: 0)
        solver.bulkModulus = Float(env["LIQUID_FILM_K"] ?? "") ?? 60
        solver.gamma = 3
        solver.viscosity = 0.01
        solver.volumeCorrection = 0.05
        solver.thrownRelax = 0.2; solver.dryFloor = true
        solver.noTension = (env["LIQUID_FILM_TENSION"] ?? "") != "1"
        solver.cohesion = Float(env["LIQUID_FILM_COHESION"] ?? "") ?? 0.02
        solver.settleDamping = Float(env["LIQUID_FILM_DAMP"] ?? "") ?? 0.999
        solver.restDensity = 8
        solver.substepsPerFrame = 4
        solver.aerationGain = 0.7
        solver.reservePool(totalParticles: 80_000)
        let speed: Float = 1.0, radius: Float = Float(env["LIQUID_FILM_RADIUS"] ?? "") ?? 0.011
        solver.maxExpectedSpeed = solver.fallSpeed(height: nozzleY, initial: speed)
        let flow = MLSMPMSolver.Emitter.fullFlow(radius: radius, speed: speed)

        let surface = try XCTUnwrap(LiquidSurfaceRenderer(engine: engine))
        surface.particleRadius = cell * 0.5 * (Float(env["LIQUID_FILM_PR"] ?? "") ?? 0.95)
        surface.environment = studio(device)
        surface.lights = [.init(position: SIMD3(0.5, 0.9, 0.35), directional: true, color: SIMD3(1.0, 0.97, 0.92)),
                          .init(position: SIMD3(-0.7, 0.4, -0.3), directional: true, color: SIMD3(0.25, 0.32, 0.45))]
        if env["LIQUID_FILM_DRINK"] == "1" {
            surface.look.absorption = SIMD3(6, 38, 46)
            surface.look.scatterColor = SIMD3(0.75, 0.08, 0.05); surface.look.scatter = 14
        }

        let spray = try XCTUnwrap(FoamSolver(engine: engine, fluidSolver: solver, capacity: 4000))
        spray.lifeMax = 0.5; spray.drag = 1.2
        spray.spawnMinSpeed = 0.5; spray.spawnMaxDensity = 7; spray.spawnMinAeration = 0.35
        spray.spawnProb = Float(env["LIQUID_FILM_SPRAY"] ?? "") ?? 0.02
        spray.spawnVelJitter = 0.8; spray.spawnUpKick = 1.6; spray.spawnPositionJitter = cell
        surface.droplets = [(spray.particleBuffer.buffer, spray.particleBuffer.capacity, cell * 0.22)]

        let lib = try device.makeLibrary(source: Self.sceneShader, options: nil)
        let sd = MTLRenderPipelineDescriptor()
        sd.vertexFunction = lib.makeFunction(name: "scene_vs"); sd.fragmentFunction = lib.makeFunction(name: "scene_fs")
        sd.colorAttachments[0].pixelFormat = .bgra8Unorm
        sd.depthAttachmentPixelFormat = .depth32Float
        let scenePipeline = try device.makeRenderPipelineState(descriptor: sd)
        let dsd = MTLDepthStencilDescriptor(); dsd.depthCompareFunction = .always; dsd.isDepthWriteEnabled = true
        let depthState = try XCTUnwrap(device.makeDepthStencilState(descriptor: dsd))
        let cd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: size, height: size, mipmapped: false)
        cd.usage = [.renderTarget, .shaderRead]; cd.storageMode = .shared
        let colour = try XCTUnwrap(device.makeTexture(descriptor: cd))
        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: size, height: size, mipmapped: false)
        dd.usage = [.renderTarget, .shaderRead]; dd.storageMode = .private
        let depth = try XCTUnwrap(device.makeTexture(descriptor: dd))

        let eye: SIMD3<Float>, at: SIMD3<Float>, fov: Float
        switch viewName {
        case "close": eye = SIMD3(0.20, 0.24, 0.26); at = SIMD3(0.01, 0.05, 0.0); fov = 30
        case "side": eye = SIMD3(0.0, 0.30, 0.62); at = SIMD3(0, 0.22, 0); fov = 40
        default: eye = SIMD3(0.30, 0.42, 0.44); at = SIMD3(0, 0.10, 0); fov = 36
        }
        let view = lookAt(eye, at), proj = perspective(fovY: fov * .pi / 180, aspect: 1, near: 0.02, far: 10)
        var su = SceneUniforms(invVP: (proj * view).inverse, vp: proj * view, eye: SIMD4(eye, 1), cup: SIMD4(R, H, 0.006, 0))

        var gpuSim: [Double] = [], gpuDraw: [Double] = []
        for f in 0..<frames {
            let t = Float(f) / 60
            // Pour for 3 s from 0.3 s, the nozzle circling slowly; then let it settle.
            let pouring = t > 0.3 && t < 3.3
            let a = 0.9 * t
            let nozzle = SIMD3<Float>(0.045 * cos(a), nozzleY, 0.045 * sin(a))
            solver.emitters = [.init(position: nozzle, velocity: SIMD3(0, -speed, 0), radius: radius, flowRate: pouring ? flow : 0,
                                     velocityJitter: Float(env["LIQUID_FILM_JITTER"] ?? "") ?? 0.02)]
            let cb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
            solver.encode(to: cb, wallDt: 1.0 / 60)
            spray.encode(to: cb, wallDt: 1.0 / 60)
            cb.commit()
            let cb2 = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
            let rp = MTLRenderPassDescriptor()
            rp.colorAttachments[0].texture = colour; rp.colorAttachments[0].loadAction = .clear; rp.colorAttachments[0].storeAction = .store
            rp.depthAttachment.texture = depth; rp.depthAttachment.loadAction = .clear; rp.depthAttachment.clearDepth = 1
            rp.depthAttachment.storeAction = .store
            let enc = try XCTUnwrap(cb2.makeRenderCommandEncoder(descriptor: rp))
            enc.setRenderPipelineState(scenePipeline); enc.setDepthStencilState(depthState)
            enc.setFragmentBytes(&su, length: MemoryLayout<SceneUniforms>.stride, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
            cb2.commit()
            let cb3 = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
            surface.encode(to: cb3, particles: solver.particleBuffer.buffer, count: solver.particleBuffer.count,
                           restDensity: solver.restDensity, view: view, projection: proj, target: colour, sceneDepth: depth)
            cb3.commit()
            cb3.waitUntilCompleted()   // gpu-ok: test-time readback
            XCTAssertNil(cb.error); XCTAssertNil(cb3.error)
            gpuSim.append((cb.gpuEndTime - cb.gpuStartTime) * 1000); gpuDraw.append((cb3.gpuEndTime - cb3.gpuStartTime) * 1000)
            writePNG(colour, to: dir.appendingPathComponent(String(format: "f%04d.png", f)))
        }
        gpuSim.sort(); gpuDraw.sort()
        let st = solver.pourStats
        print(String(format: "LIQUID FILM %@: %d frames, cell %.1f mm, %d substeps, live %d (dropped %d), sim GPU p50 %.2f ms p90 %.2f, draw GPU p50 %.2f ms p90 %.2f",
                     viewName, frames, cell * 1000, solver.effectiveSubsteps(wallDt: 1.0 / 60), st.live, st.dropped,
                     gpuSim[gpuSim.count / 2], gpuSim[gpuSim.count * 9 / 10], gpuDraw[gpuDraw.count / 2], gpuDraw[gpuDraw.count * 9 / 10]))
        XCTAssertGreaterThan(st.live, 1000)
    }
}
