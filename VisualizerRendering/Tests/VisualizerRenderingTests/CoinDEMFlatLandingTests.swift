import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// VZ-0201: a flat-bottomed hull landing FLAT on a static box must not sink into it. The polytope
/// narrowphase's Gregorius edge tie (`sepE > 0.9·maxF + absTol`) is written for overlap; within the
/// speculative margin (pair apart, maxF > 0) it let an edge pair whose axis is parallel to the face
/// normal win, and the manifold became one point midway between the base's edge and the box's far
/// rim — the Digital Clock's flat-based worker sank up to 2.6 mm on 8 of 12 drop heights. The fix
/// (`apartFaceFirst`) is opt-in with the manifold solve, so every other world is bit-identical.
/// Numbers are PRINTed with a `FLATLAND_` prefix.
@MainActor
final class CoinDEMFlatLandingTests: XCTestCase {

    private static var cached: (SimEngine, MTLLibrary, MTLCommandQueue)?
    private static func shared() throws -> (SimEngine, MTLLibrary, MTLCommandQueue) {
        if let c = cached { return c }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness queue
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard FileManager.default.fileExists(atPath: shader.path) else {
            throw XCTSkip("CoinDEM.metal not found at \(shader.path)")
        }
        let c = (SimEngine(device: device), try MetalSourceLoader.makeLibrary(device: device, contentsOf: shader), queue)
        cached = c
        return c
    }

    /// A 38 mm "stump" figure: a 12-gon base Ø21 mm with a 0.8 mm chamfer, a body to 18.5 mm, a
    /// head / hat taper to 38 mm — the shape class of the Digital Clock worker.
    static func stumpPoints() -> [SIMD3<Float>] {
        let rings: [(r: Float, y: Float)] = [(0.0103, 0), (0.0111, 0.0008), (0.0108, 0.0185), (0.0102, 0.0308), (0.0060, 0.0370), (0.0020, 0.0386)]
        let g = 1 / cos(Float.pi / 12)
        var out: [SIMD3<Float>] = []
        for ring in rings {
            for i in 0..<12 {
                let a = 2 * Float.pi * Float(i) / 12
                out.append(SIMD3(ring.r * g * sin(a), ring.y, ring.r * g * cos(a)))
            }
        }
        return out
    }

    /// The Digital Clock's §G configuration with its stage-B opt-ins.
    func makeSolver(manifold: Bool) throws -> (CoinDEMSolver, MTLCommandQueue, CoinDEMSolver.HullHandle) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: 8, coinRadius: 0.03, halfThickness: 0.03,
                                    boundsMin: SIMD3(-0.3, -0.1, -0.3), boundsMax: SIMD3(0.3, 0.3, 0.3))
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = 9.81
        s.fixedDt = 1.0 / 180
        s.maxSubsteps = 10
        s.velocityIterations = 6
        s.colorRounds = 16
        s.islandUnionRounds = 6
        s.warmStart = manifold
        s.manifoldSolve = manifold
        s.linDamping = 0.99995
        s.angDamping = 0.9998
        s.frictionCoeff = 0.5
        s.accumulatedRollingResistance = true
        s.rollingResistance = 0.024
        s.scaleAwareDeadStop = true
        s.restitution = 0.15
        s.restThreshold = 0.14
        s.contactSlop = 1e-4
        s.baumgarteBeta = 0.2
        s.speculativeMargin = 2.5e-3
        s.maxSpeed = 3
        s.maxHSpeed = 3
        s.maxOmega = 60
        s.floorY = -0.5
        s.sleepEnabled = true
        s.sleepFrames = 20
        s.sleepLinVel = 2e-4
        s.setColliders([.box(center: SIMD3(0, -0.011, 0), halfExtents: SIMD3(0.28, 0.011, 0.21), friction: 0.5, restitution: 0.2)])
        guard let h = s.registerHull(vertices: Self.stumpPoints()) else { throw XCTSkip("hull") }
        XCTAssertTrue(s.hullHasTopology(h.index), "the stump must take the exact polytope path")
        return (s, queue, h)
    }

    /// Deepest the base goes below the box top (m) over 0.33 s after a flat drop of `drop` m.
    func dropSink(drop: Float, manifold: Bool) throws -> Float {
        let (s, q, h) = try makeSolver(manifold: manifold)
        let base = SIMD3<Float>(0, drop, 0)
        guard let w = s.spawnHull(at: base + h.comOffset, hull: h, orient: SIMD4(h.principalRotation.imag, h.principalRotation.real),
                                  mass: 0.0113, friction: 0.6, restitution: 0.15) else { XCTFail("spawn"); return 0 }
        var lowest: Float = 1
        var st: [CoinBodyState] = []
        let g = 1 / cos(Float.pi / 12)
        for _ in 0..<60 {
            guard let cb = q.makeCommandBuffer() else { break }
            s.encode(to: cb, wallDt: 1.0 / 180)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test reads every substep synchronously
            s.readStates([w], into: &st)
            let qt = st[0].orientation * h.principalRotation.inverse
            let c = st[0].position - simd_act(qt, h.comOffset)
            for i in 0..<12 {
                let a = 2 * Float.pi * Float(i) / 12
                lowest = min(lowest, c.y + simd_act(qt, SIMD3(0.0103 * g * sin(a), 0, 0.0103 * g * cos(a))).y)
            }
        }
        return -lowest
    }

    /// With the manifold solve (the opt-in the fix rides on), a flat landing never sinks: 12 drop
    /// heights spanning a whole substep of approach (0.33 mm apart, 11.00–14.63 mm).
    func testFlatLandingDoesNotSinkWithTheManifoldSolve() throws {
        var worst: Float = 0
        var line = ""
        for k in 0..<12 {
            let drop = Float(0.0110) + Float(k) * 0.00033
            let sink = try dropSink(drop: drop, manifold: true)
            worst = max(worst, sink)
            line += String(format: " %.2f:%.3f", drop * 1000, sink * 1000)
        }
        print("FLATLAND_ manifold+warm, drop mm:sink mm" + line)
        XCTAssertLessThan(worst, 5e-5, "a flat base landing flat sank \(worst * 1000) mm into the box (VZ-0201)")
    }
}
