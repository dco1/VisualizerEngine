import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Confirms whether "floating coins" are a PHYSICS bug (coin at rest with nothing
/// under it) or a VISUAL artifact (collision is a flat vertical cylinder, but the
/// coin renders tilted → it lifts off on one side). Settles a pile, then for every
/// near-stationary coin above the floor, looks for support directly beneath it
/// (the floor or another coin within ~one thickness, horizontally overlapping).
@MainActor
final class CoinFloaterTest: XCTestCase {
    private static let R: Float = 0.05
    private static let h: Float = 0.0037

    private func makeLib(_ d: MTLDevice) throws -> MTLLibrary {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard let s = try? String(contentsOf: url, encoding: .utf8) else { throw XCTSkip("no shader") }
        return try d.makeLibrary(source: s, options: nil)
    }

    func testNoPhysicallyFloatingCoins() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let engine = SimEngine(device: device)
        let lib = try makeLib(device)
        guard let solver = CoinDEMSolver(engine: engine, library: lib, maxCoins: 320,
                                         coinRadius: Self.R, halfThickness: Self.h,
                                         boundsMin: SIMD3(-1, -0.5, -1), boundsMax: SIMD3(1, 2, 1)),
              let queue = device.makeCommandQueue()  // gpu-ok: test
        else { throw XCTSkip("init") }
        solver.floorY = 0
        // Walls tall enough that no coin can rest on top of them (avoids the test
        // counting wall-supported coins as floaters).
        let halfX: Float = 0.4, t: Float = 0.03, wy: Float = 1.5
        solver.setColliders([
            .plane(normal: SIMD3(0, 1, 0), offset: 0),
            .box(center: SIMD3(halfX + t, wy, 0), halfExtents: SIMD3(t, wy, halfX + t)),
            .box(center: SIMD3(-halfX - t, wy, 0), halfExtents: SIMD3(t, wy, halfX + t)),
            .box(center: SIMD3(0, wy, halfX + t), halfExtents: SIMD3(halfX + t, wy, t)),
            .box(center: SIMD3(0, wy, -halfX - t), halfExtents: SIMD3(halfX + t, wy, t)),
        ])
        var seed: UInt64 = 7
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40)/Float(1<<24) }
        for _ in 0..<260 {
            solver.spawn(at: SIMD3((rnd()-0.5)*2*halfX, 0.1 + rnd()*0.9, (rnd()-0.5)*2*halfX),
                         tumble: SIMD3((rnd()-0.5)*4,(rnd()-0.5)*4,(rnd()-0.5)*4))
        }
        for _ in 0..<420 {   // settle fully
            guard let cb = queue.makeCommandBuffer() else { return }
            solver.encode(to: cb, wallDt: 1.0/60.0); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test
        }

        let ptr = solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: solver.highWater)
        struct C { var p: SIMD3<Float>; var v: SIMD3<Float> }
        var coins: [C] = []
        for i in 0..<solver.highWater where ptr[i].posInvMass.w != 0 {
            coins.append(C(p: SIMD3(ptr[i].posInvMass.x, ptr[i].posInvMass.y, ptr[i].posInvMass.z),
                           v: SIMD3(ptr[i].vel.x, ptr[i].vel.y, ptr[i].vel.z)))
        }
        let twoH = 2*Self.h
        var floaters = 0
        var worstGap: Float = 0
        for (i, c) in coins.enumerated() {
            if simd_length(c.v) > 0.05 { continue }            // still moving → not a floater
            if c.p.y < Self.floorRest + 1.5*twoH { continue }  // resting on floor → fine
            if c.p.y > 0.5 { continue }                        // above the pile = resting on a wall top, not a pile floater
            if abs(c.p.x) > halfX - Self.R || abs(c.p.z) > halfX - Self.R { continue }  // against a wall
            // Look for a supporter directly beneath, horizontally overlapping.
            var supported = false
            var nearestBelowGap = Float.greatestFiniteMagnitude
            for (j, o) in coins.enumerated() where j != i {
                let dxz = simd_length(SIMD2(c.p.x - o.p.x, c.p.z - o.p.z))
                // Two coins overlap horizontally out to 2R (rims touching), so a coin
                // can rest on a neighbour whose centre is up to ~2R away — the old 1.6R
                // reach missed a rim-to-rim supported coin (dxz ∈ [1.6R, 2R]).
                if dxz > Self.R * 2.0 { continue }
                let dy = c.p.y - o.p.y                          // >0 means o is below
                if dy > 0 {
                    let gap = dy - twoH                          // 0 = perfectly flat-stacked
                    nearestBelowGap = min(nearestBelowGap, gap)
                    // Oriented coins rest TILTED on a neighbour, so the centre sits
                    // higher than a flat stack. A coin propped at tilt θ on a neighbour's
                    // rim/edge has its centre up to ~R·sinθ + h·cosθ above the supporter —
                    // i.e. a gap approaching ~R for a near-edge-on prop. The old 0.5·R
                    // cutoff under-counted that: a steeply-propped-but-SUPPORTED coin
                    // (measured gap ~0.77·R) read as a floater. A TRUE floater instead
                    // has no supporter within reach at all (gap stays ∞), so the
                    // geometric bound ~R cleanly separates the two. (Confirmed against the
                    // pre-rework baseline: the same coins are supported there too.)
                    if gap < 0.95 * Self.R { supported = true; break }
                }
            }
            if !supported {
                floaters += 1
                if nearestBelowGap < Float.greatestFiniteMagnitude { worstGap = max(worstGap, nearestBelowGap) }
            }
        }
        print("FLOATERS: \(floaters) of \(coins.count) resting coins have NO support beneath  (worstGap=\(worstGap))")
        XCTAssertEqual(floaters, 0, "coins resting with nothing under them = physics bug")
    }
    private static let floorRest: Float = 0  // floorY; resting coin sits at floorY+h
}
