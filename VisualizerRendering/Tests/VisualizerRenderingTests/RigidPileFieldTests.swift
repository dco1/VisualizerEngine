import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Tests for `RigidPileField` — the general mixed-shape "pile of mess" facade. GPU
/// cases use the same runtime-compiled-library seam as the other CoinDEM tests.
@MainActor
final class RigidPileFieldTests: XCTestCase {

    private static func makeLibrary(_ device: MTLDevice) throws -> MTLLibrary {
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard let src = try? String(contentsOf: shader, encoding: .utf8) else {
            throw XCTSkip("CoinDEM.metal not found at \(shader.path)")
        }
        return try device.makeLibrary(source: src, options: nil)
    }

    private func makeField(maxBodies: Int, bodyScale: Float,
                           colliders: [CoinStaticCollider]) throws
        -> (RigidPileField, MTLCommandQueue)
    {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)
        var cfg = RigidPileField.Config(maxBodies: maxBodies, bodyScale: bodyScale,
                                        bounds: (SIMD3(-1, -0.5, -1), SIMD3(1, 2, 1)))
        cfg.floorY = 0
        guard let field = RigidPileField(engine: engine, library: lib, config: cfg,
                                         colliders: colliders),
              let queue = device.makeCommandQueue()  // gpu-ok: test harness queue
        else { throw XCTSkip("field/queue init failed") }
        return (field, queue)
    }

    private func step(_ field: RigidPileField, _ queue: MTLCommandQueue, frames: Int) {
        for _ in 0..<frames {
            guard let cb = queue.makeCommandBuffer() else { return }
            field.encode(to: cb, dt: 1.0 / 60.0)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
        }
    }

    // ── A heap of MIXED shapes (discs + spheres + rods + boxes) settles together in a
    // bin — no explosion, tunnelling, or NaN. The whole point of the facade. ─────────
    func testMixedShapePileSettles() throws {
        let n = 20
        let bodyScale: Float = 0.07                    // ≥ the biggest body's bounding radius
        let bin = RigidPileField.bin(innerHalf: SIMD2(0.3, 0.3), floorY: 0)
        let (field, queue) = try makeField(maxBodies: n + 2, bodyScale: bodyScale, colliders: bin)
        var seed: UInt64 = 0xC0FFEE
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for i in 0..<n {
            let p = SIMD3<Float>((rnd()-0.5)*0.4, 0.2 + rnd()*0.7, (rnd()-0.5)*0.4)
            let tumble = SIMD3<Float>((rnd()-0.5)*4, (rnd()-0.5)*4, (rnd()-0.5)*4)
            switch i % 4 {
            case 0: field.dropDisc(at: p, radius: 0.05, halfThickness: 0.006, tumble: tumble, type: 0)
            case 1: field.dropSphere(at: p, radius: 0.04, tumble: tumble, type: 1)
            case 2: field.dropRod(at: p, radius: 0.015, halfLength: 0.06, tumble: tumble, type: 2)
            default: field.dropBox(at: p, halfExtents: SIMD3(0.04, 0.03, 0.04), tumble: tumble, type: 3)
            }
        }
        step(field, queue, frames: 420)

        let r = CoinDiagnostics.measure(field.solver)   // KE / belowFloor / maxY shape-agnostic
        print("MIXED_PILE n=\(r.activeCount) KE=\(r.kineticEnergy) belowFloor=\(r.belowFloorCount) maxY=\(r.maxY)")
        XCTAssertEqual(r.activeCount, n)
        XCTAssertFalse(r.kineticEnergy.isNaN, "no NaN blow-up")
        XCTAssertEqual(r.belowFloorCount, 0, "nothing tunnelled the floor")
        XCTAssertLessThan(r.kineticEnergy, 1.2, "the mixed heap settles (no jitter/explosion)")
        XCTAssertLessThan(r.maxY, 0.45, "the heap stays in the bin, not launched")
    }

    // ── Pile of Mess rest gate. Reproduces the shipped scene config (176 mixed
    // coins/marbles/dice, bin 0.55, iter 8 × sub 4, contactRelax 0.5) and measures the
    // settled heap's kinetic energy. Guards the two real contact fixes in CoinDEM.metal:
    //   • disc↔box phantom-corner rejection (a disc collided as its square bounding box
    //     had 4 phantom corner columns the round disc lacks → coin↔die rest limit cycle)
    //   • sphere↔disc / sphere↔box real closest-point contact (the old bounding-sphere
    //     fallback over-separated → marble-on-coin/die rest limit cycle)
    // A regression in either re-introduces a pervasive jitter (KE ≫ 1) or, if a contact
    // is made too stiff, an explosion. Run 3× because the GPU Jacobi solve is non-
    // deterministic (atomic scatter ordering) and a dense mixed pack occasionally forms
    // a small jam pocket; we assert NONE explode and the BEST run settles quiet. ──────
    func testPileOfMessRestQuiet() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)

        let coinR0: Float = 0.075, coinH0: Float = 0.022, marbleR0: Float = 0.052

        // Returns (avgRestKE, avgAwake, maxPen, belowFloor) for one settled run of the
        // shipped scene config. `kinds` ⊆ [0=coin,1=marble,2=die] (for isolation runs).
        func runConfig(label: String, fill: Int = 176, innerHalf: Float = 0.55,
                       contactRelax: Float = 0.5, constraint: Bool = false, kinds: [Int] = [0, 1, 2])
            -> (ke: Float, awake: Float, pen: Float, below: Int)
        {
            var cfg = RigidPileField.Config(maxBodies: fill + 14, bodyScale: 0.12,
                                            bounds: (SIMD3(-1.1, -0.5, -1.1), SIMD3(1.1, 2.6, 1.1)))
            cfg.floorY = 0
            cfg.restitution = 0.18
            cfg.friction = 0.86
            cfg.maxHorizontalSpeed = 2.5
            cfg.iterations = 8
            cfg.maxSubsteps = 4
            cfg.contactRelax = contactRelax
            cfg.useConstraintSolver = constraint
            if constraint { cfg.frictionCoeff = 0.5 }   // real Coulomb friction → repose
            guard let field = RigidPileField(engine: engine, library: lib, config: cfg,
                                             colliders: RigidPileField.bin(innerHalf: SIMD2(innerHalf, innerHalf), floorY: 0)),
                  let queue = device.makeCommandQueue()  // gpu-ok: test harness queue
            else { return (0, 0, 0, 0) }

            var spawnCount = 0
            func rnd(_ s: inout UInt64) -> Float { s ^= s >> 30; s &*= 0xBF58476D1CE4E5B9; s ^= s >> 27; return Float(s >> 40) / Float(1 << 24) }
            func spawnOne() {
                spawnCount += 1
                var h = UInt64(bitPattern: Int64(spawnCount)) &* 0x9E3779B97F4A7C15 &+ 0xD1B54A32D192ED03
                let kind = kinds[Int(rnd(&h) * Float(kinds.count)) % kinds.count]
                let p = SIMD3<Float>((rnd(&h) - 0.5) * 0.8, 1.7 + rnd(&h) * 0.3, (rnd(&h) - 0.5) * 0.8)
                let tumble = SIMD3<Float>((rnd(&h) - 0.5) * 5, (rnd(&h) - 0.5) * 5, (rnd(&h) - 0.5) * 5)
                switch kind {
                case 0: field.dropDisc(at: p, radius: coinR0, halfThickness: coinH0, tumble: tumble, type: 0)
                case 1: field.dropSphere(at: p, radius: marbleR0, tumble: tumble, type: 1)
                default: let he = 0.038 + rnd(&h) * 0.012; field.dropBox(at: p, halfExtents: SIMD3(he, he, he), tumble: tumble, type: 2)
                }
            }
            for f in 0..<900 {
                if f < 400 && field.activeCount < fill { spawnOne() }   // ≤1 body/frame: gentle, like the scene
                guard let cb = queue.makeCommandBuffer() else { break }
                field.encode(to: cb, dt: 1.0 / 60.0)
                cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness reads synchronously
            }
            let n = field.solver.highWater
            var keSum: Float = 0, awakeSum = 0, penMax: Float = 0, below = 0
            for _ in 0..<12 {
                guard let cb = queue.makeCommandBuffer() else { break }
                field.encode(to: cb, dt: 1.0 / 60.0)
                cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness reads synchronously
                let r = CoinDiagnostics.measure(field.solver)
                keSum += r.kineticEnergy; penMax = max(penMax, r.maxPenetration); below = max(below, r.belowFloorCount)
                let ptr = field.solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: max(1, n))
                for i in 0..<n where ptr[i].posInvMass.w != 0 {
                    let v = SIMD3(ptr[i].vel.x, ptr[i].vel.y, ptr[i].vel.z)
                    if simd_length(v) > 0.01 { awakeSum += 1 }
                }
            }
            let ke = keSum / 12, awake = Float(awakeSum) / 12
            print(String(format: "PILEOFMESS_REST %-14@ avgKE=%.4f avgAwake=%.1f maxPen=%.4f below=%d",
                         label as NSString, ke, awake, penMax, below))
            return (ke, awake, penMax, below)
        }

        // Legacy Jacobi path (baseline) — bounded, but jitters/jam-pockets at this density.
        _ = runConfig(label: "legacy", constraint: false)

        // CONSTRAINT solver — the new graph-colored sequential-impulse engine. At the
        // SAME shipped 176 @ 0.55 density it must settle DEAD-STILL, every run (no jam
        // pocket, no body-count / bin dodge). This is the fix for the original bug.
        var worstKE: Float = 0
        for r in 0..<3 {
            let m = runConfig(label: "constraint #\(r)", constraint: true)
            XCTAssertEqual(m.below, 0, "nothing tunnelled the floor")
            XCTAssertLessThanOrEqual(m.awake, 6, "constraint solver: a settled pile has ~zero awake bodies")
            worstKE = max(worstKE, m.ke)
        }
        XCTAssertLessThan(worstKE, 0.05, "constraint solver settles the full-density mixed pile dead-still, every run")
    }

    // ── The cull line recycles settled bodies (a payout). ─────────────────────────
    func testCullBelowLine() throws {
        let bin = RigidPileField.bin(innerHalf: SIMD2(0.3, 0.3), floorY: 0)
        let (field, queue) = try makeField(maxBodies: 12, bodyScale: 0.06, colliders: bin)
        for i in 0..<8 {                               // a spread 4×2 grid → one settled layer
            let x = Float(i % 4) * 0.13 - 0.195
            let z = Float(i / 4) * 0.13 - 0.065
            field.dropSphere(at: SIMD3(x, 0.15, z), radius: 0.04)
        }
        step(field, queue, frames: 180)               // settle on the floor (COM ≈ 0.04)
        XCTAssertEqual(field.activeCount, 8)
        let culled = field.cull(belowY: 0.1)           // everything resting below the payout line
        XCTAssertEqual(culled, 8, "all settled bodies are below the cull line → recycled")
        XCTAssertEqual(field.activeCount, 0)
    }
}
