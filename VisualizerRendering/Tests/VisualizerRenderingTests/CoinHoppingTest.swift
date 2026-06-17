import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Instrumentation for the intermittent "hopping" the scene shows: settle a pile,
/// then run a reciprocating pusher plate and log every coin that gains a
/// significant UPWARD velocity (gravity only pulls down, so positive vy means a
/// contact shoved it up = a hop). Prints the hop rate, the worst offenders, and a
/// trace of a few sample coins so we can see WHERE the energy comes from.
@MainActor
final class CoinHoppingTest: XCTestCase {
    private static let R: Float = 0.05
    private static let h: Float = 0.0037

    private func makeLib(_ d: MTLDevice) throws -> MTLLibrary {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard let s = try? String(contentsOf: url, encoding: .utf8) else { throw XCTSkip("no shader") }
        return try d.makeLibrary(source: s, options: nil)
    }

    func testHoppingInstrumentation() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let engine = SimEngine(device: device)
        let lib = try makeLib(device)
        guard let solver = CoinDEMSolver(engine: engine, library: lib, maxCoins: 320,
                                         coinRadius: Self.R, halfThickness: Self.h,
                                         boundsMin: SIMD3(-1, -0.5, -1.5), boundsMax: SIMD3(1, 2, 1.5)),
              let queue = device.makeCommandQueue()  // gpu-ok: test harness
        else { throw XCTSkip("init") }
        solver.floorY = 0

        let halfX: Float = 0.5, t: Float = 0.03
        // Open-front bin (back + sides), like the shelf the pusher sweeps toward.
        let staticCols: [CoinStaticCollider] = [
            .plane(normal: SIMD3(0, 1, 0), offset: 0),
            .box(center: SIMD3(halfX + t, 0.5, 0), halfExtents: SIMD3(t, 0.5, 1.0)),
            .box(center: SIMD3(-halfX - t, 0.5, 0), halfExtents: SIMD3(t, 0.5, 1.0)),
            .box(center: SIMD3(0, 0.5, -1.0 - t), halfExtents: SIMD3(halfX + t, 0.5, t)),
        ]
        solver.setColliders(staticCols)

        var seed: UInt64 = 99
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40)/Float(1<<24) }
        for _ in 0..<240 {
            solver.spawn(at: SIMD3((rnd()-0.5)*2*halfX, 0.1 + rnd()*0.5, (rnd()-0.5)*1.6),
                         tumble: SIMD3((rnd()-0.5)*4,(rnd()-0.5)*4,(rnd()-0.5)*4))
        }
        func frame(_ cols: [CoinStaticCollider]) {
            solver.setColliders(cols)
            guard let cb = queue.makeCommandBuffer() else { return }
            solver.encode(to: cb, wallDt: 1.0/60.0); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test
        }
        for _ in 0..<240 { frame(staticCols) }   // settle

        let ptr = solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: solver.highWater)
        let hopThresh: Float = 0.5
        var hopCoinFrames = 0, totalActiveFrames = 0
        var worstVy: Float = 0
        var hopWithPusher = 0, hopNoPusher = 0

        // 480 frames: reciprocating pusher plate sweeping -Z→+Z.
        for f in 0..<480 {
            let phase = Float(f % 90) / 90.0
            let tri = phase < 0.5 ? phase*2 : (1-phase)*2
            let adv = tri*tri*(3-2*tri) * 0.45
            let plate = CoinStaticCollider.pusherPlate(
                center: SIMD3(0, 0.2, -0.9 + adv), halfExtents: SIMD3(halfX, 0.2, 0.04))
            frame(staticCols + [plate])

            var frameHops = 0
            for i in 0..<solver.highWater where ptr[i].posInvMass.w != 0 {
                totalActiveFrames += 1
                let vy = ptr[i].vel.y
                if vy > worstVy { worstVy = vy }
                if vy > hopThresh { hopCoinFrames += 1; frameHops += 1
                    // near the plate front?
                    if abs(ptr[i].posInvMass.z - (-0.9 + adv)) < 0.2 { hopWithPusher += 1 } else { hopNoPusher += 1 }
                }
            }
            if f % 80 == 0 {
                let c0 = ptr[0]
                print(String(format: "f%3d adv=%.2f hops=%d coin0 y=%.4f vy=%.3f", f, adv, frameHops, c0.posInvMass.y, c0.vel.y))
            }
        }
        let rate = Double(hopCoinFrames) / Double(max(1, totalActiveFrames)) * 100
        print(String(format: "HOPS: coin-frames vy>%.1f = %d (%.3f%% of active)  worstVy=%.2f  nearPlate=%d far=%d",
                     hopThresh, hopCoinFrames, rate, worstVy, hopWithPusher, hopNoPusher))
        // Not a strict gate yet — this run is to read the numbers and locate the source.
    }
}
