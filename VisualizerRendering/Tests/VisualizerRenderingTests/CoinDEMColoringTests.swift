import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Engine gate for VZ-0160 (stage B2): the SPECULATIVE contact colouring
/// (`coloringScheme = .speculative`, CoinDEMSolver+Coloring.swift; CoinDEM.metal
/// coinColorTentative / coinColorResolve). Jones–Plassmann colours at most one contact per body
/// per round, so a heap whose bodies touch 20–30 others needs 50–60 rounds and leaves thousands
/// of contacts to the uncoloured tail at the usual 16–24. Each test colours the SAME contact
/// graph (one generate pass from a frozen state) with both schemes, and checks the three things
/// a colouring must be for the parallel Gauss–Seidel solve: COMPLETE (every contact coloured,
/// so every contact is solved in the sweep), PROPER (no two contacts that share a dynamic body
/// in one colour — they would race), and REPRODUCIBLE (a pure function of the contact graph).
/// Measured values are PRINTed with a `COLOR_` prefix.
@MainActor
final class CoinDEMColoringTests: XCTestCase {

    // ── Harness ───────────────────────────────────────────────────────────────

    private static var cached: (SimEngine, MTLLibrary, MTLCommandQueue)?
    private static func shared() throws -> (SimEngine, MTLLibrary, MTLCommandQueue) {
        if let c = cached { return c }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness queue
        let url = CoinDEMFootTests.shaderURL
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("CoinDEM.metal not found") }
        let c = (SimEngine(device: device), try MetalSourceLoader.makeLibrary(device: device, contentsOf: url), queue)
        cached = c
        return c
    }

    @discardableResult
    private func step(_ s: CoinDEMSolver, _ q: MTLCommandQueue, frames: Int, wallDt: Float = 1.0 / 60,
                      perFrame: ((Int) -> Void)? = nil) -> [Double] {
        var gpu: [Double] = []
        for f in 0..<frames {
            guard let cb = q.makeCommandBuffer() else { break }
            s.encode(to: cb, wallDt: wallDt)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            if cb.status == .error { XCTFail("command buffer error: \(String(describing: cb.error))"); break }
            gpu.append(max(0, cb.gpuEndTime - cb.gpuStartTime) * 1000)
            perFrame?(f)
        }
        return gpu
    }

    /// One colouring of the solver's current contact graph, read back.
    struct Coloring {
        var contacts = 0
        var uncoloured = 0
        var colours = 0                          // highest colour + 1
        var conflicts = 0                        // pairs sharing a dynamic body with one colour
        var maxDegree = 0
        var byIdentity: [String: [Int]] = [:]    // contact identity → its colour(s)
    }

    /// Generate + colour the current state once (nothing is stepped) with `scheme` / `rounds`.
    func colourOnce(_ s: CoinDEMSolver, _ scheme: CoinDEMSolver.ColoringScheme, rounds: Int) -> Coloring {
        s.coloringScheme = scheme
        if scheme == .speculative { s.speculativeColorRounds = rounds } else { s.colorRounds = rounds }
        let n = s.generateContactsNow(color: true)
        let p = s.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: s.maxContacts)
        var r = Coloring()
        r.contacts = n
        var perBody: [UInt32: [Int]] = [:]       // body → colours of its coloured contacts
        var degree: [UInt32: Int] = [:]          // body → all its contacts
        for i in 0..<n {
            let c = p[i]
            let col = Int(c.tan2.w)
            let key = "\(c.meta.x) \(c.meta.y) \(c.meta.z) \(c.meta.w)"
            r.byIdentity[key, default: []].append(col)
            degree[c.meta.x, default: 0] += 1
            if c.meta.y != 0xFFFF_FFFF { degree[c.meta.y, default: 0] += 1 }
            if col < 0 { r.uncoloured += 1; continue }
            r.colours = max(r.colours, col + 1)
            perBody[c.meta.x, default: []].append(col)
            if c.meta.y != 0xFFFF_FFFF { perBody[c.meta.y, default: []].append(col) }
        }
        r.maxDegree = degree.values.max() ?? 0
        for (_, cols) in perBody {
            var seen: [Int: Int] = [:]
            for c in cols { seen[c, default: 0] += 1 }
            r.conflicts += seen.values.reduce(0) { $0 + ($1 > 1 ? $1 - 1 : 0) }
        }
        for k in r.byIdentity.keys { r.byIdentity[k]!.sort() }
        return r
    }

    func line(_ tag: String, _ c: Coloring) -> String {
        "\(tag): contacts \(c.contacts), max degree \(c.maxDegree), uncoloured \(c.uncoloured), colours \(c.colours), conflicts \(c.conflicts)"
    }

    // ── Worlds ────────────────────────────────────────────────────────────────

    /// A resting n×n grid of 10 mm cubes on a plane, 50 µm apart (inside the speculative margin,
    /// so every cube has floor + side manifolds) — VZ-0160's cube case: per-point contacts give
    /// the middle cubes degree ≈ 20.
    private func cubeGrid(n: Int = 7) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: n * n + 8,
                                    coinRadius: 0.03, halfThickness: 0.03,
                                    boundsMin: SIMD3(-0.4, -0.1, -0.4), boundsMax: SIMD3(0.4, 0.4, 0.4))
        else { throw XCTSkip("solver init failed") }
        CoinDEMFootTests.applyClockConfig(s)
        s.sleepEnabled = false
        s.speculativeMargin = 2e-4
        s.setColliders([.plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.15)])
        let he: Float = 0.005
        for i in 0..<n { for k in 0..<n {
            _ = try XCTUnwrap(s.spawnBox(at: SIMD3(-0.1 + Float(i) * (2 * he + 5e-5), he + 5e-5, -0.1 + Float(k) * (2 * he + 5e-5)),
                                         halfExtents: SIMD3(he, he, he), mass: 0.01))
        } }
        step(s, queue, frames: 20)
        return (s, queue)
    }

    /// A Daydream-Home-style egg heap: `count` eggs (its egg: L = 62 mm, fat 0.390·L, tip
    /// 0.245·L, centres 0.365·L apart) rained into a square bin of half-width `binHalf` with its
    /// solver config (1/120 s × 4 substeps × 4 iterations, speculative margin 30 mm) and left to
    /// pile up. 300 eggs in a 0.6 m bin: a heap ~4 eggs deep whose busiest egg has ~40 contacts
    /// (the 30 mm margin counts near neighbours too) — inside the 64-entry contact list.
    func eggHeap(count: Int = 300, binHalf: Float = 0.30, seed: UInt64 = 0xE66_5EED) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        let L: Float = 0.062
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: count + 8,
                                    coinRadius: 0.56 * L, halfThickness: 0.56 * L,
                                    boundsMin: SIMD3(-binHalf - 0.1, -0.1, -binHalf - 0.1),
                                    boundsMax: SIMD3(binHalf + 0.1, 1.6, binHalf + 0.1))
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = 9.8
        s.fixedDt = 1.0 / 120
        s.maxSubsteps = 4
        s.velocityIterations = 4
        s.frictionCoeff = 0.42
        s.rollingResistance = 0.02
        s.restitution = 0.22
        s.linDamping = 0.999
        s.maxSpeed = 9; s.maxHSpeed = 4; s.maxOmega = 18
        s.floorY = 0
        s.speculativeMargin = 0.03
        s.quadraticDrag = 0.18
        s.dragRefRadius = 0.390 * L
        s.sleepEnabled = true
        s.setColliders(RigidPileField.bin(innerHalf: SIMD2(binHalf, binHalf), floorY: 0))
        var state = seed
        func rnd() -> Float { state = state &* 6364136223846793005 &+ 1442695040888963407; return Float(state >> 40) / Float(1 << 24) }
        var spawned = 0
        var frame = 0
        while spawned < count || frame < count / 6 + 120 {
            for _ in 0..<6 where spawned < count {
                let q = simd_quatf(angle: rnd() * 6.28, axis: simd_normalize(SIMD3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) + 0.001))
                _ = s.spawnEgg(at: SIMD3((rnd() - 0.5) * 1.6 * binHalf, 0.5 + rnd() * 0.9, (rnd() - 0.5) * 1.6 * binHalf),
                               fatRadius: 0.390 * L, tipRadius: 0.245 * L, centerDistance: 0.365 * L,
                               orient: SIMD4(q.imag, q.real),
                               tumble: SIMD3((rnd() - 0.5) * 6, (rnd() - 0.5) * 6, (rnd() - 0.5) * 6))
                spawned += 1
            }
            step(s, queue, frames: 1)
            frame += 1
        }
        s.wakeAll()      // the settled heap sleeps; asleep pairs generate no contacts (VZ-0152)
        return (s, queue)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 1. The cube grid: complete in 6 rounds where Jones–Plassmann at 16 is not.
    // ══════════════════════════════════════════════════════════════════════════

    func testCubeGridColoursCompletelyInSixRounds() throws {
        let (s, _) = try cubeGrid()
        let jp16 = colourOnce(s, .jonesPlassmann, rounds: 16)
        let jpAll = colourOnce(s, .jonesPlassmann, rounds: 400)
        let spec = colourOnce(s, .speculative, rounds: 6)
        let (s2, _) = try cubeGrid()
        let spec2 = colourOnce(s2, .speculative, rounds: 6)
        print("COLOR_1 " + line("JP@16", jp16) + " | " + line("JP@400", jpAll) + " | " + line("spec@6", spec)
              + " | same colours in a second world: \(spec.byIdentity == spec2.byIdentity)")
        XCTAssertGreaterThan(jpAll.maxDegree, 16, "precondition: bodies busier than the JP round budget")
        XCTAssertGreaterThan(jp16.uncoloured, 0, "control: Jones–Plassmann at 16 rounds leaves contacts uncoloured (VZ-0160)")
        XCTAssertEqual(jpAll.uncoloured, 0, "control: JP does finish, given ~2.5× the max degree in rounds")
        XCTAssertEqual(spec.contacts, jp16.contacts, "the same contact graph")
        XCTAssertEqual(spec.uncoloured, 0, "speculative: every contact coloured in 6 rounds")
        XCTAssertEqual(spec.conflicts, 0, "speculative: proper — no two contacts of one body share a colour")
        XCTAssertEqual(jpAll.conflicts, 0)
        XCTAssertLessThanOrEqual(spec.colours, CoinDEMSolver.maxColors)
        XCTAssertLessThanOrEqual(Double(spec.colours), 1.4 * Double(jpAll.colours), "at most ~40 % more colours than a complete JP")
        XCTAssertEqual(spec.byIdentity, spec2.byIdentity, "reproducible: the same graph gets the same colours")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 2. A Daydream-Home-style egg heap: the same, where JP at 24 is not complete.
    // ══════════════════════════════════════════════════════════════════════════

    func testEggHeapColoursCompletelyInSixRounds() throws {
        let (s, _) = try eggHeap(count: 300, binHalf: 0.30)
        let jp24 = colourOnce(s, .jonesPlassmann, rounds: 24)
        let jpAll = colourOnce(s, .jonesPlassmann, rounds: 400)
        let spec4 = colourOnce(s, .speculative, rounds: 4)
        let spec = colourOnce(s, .speculative, rounds: 6)
        print("COLOR_2 " + line("JP@24", jp24) + " | " + line("JP@400", jpAll) + " | " + line("spec@4", spec4) + " | " + line("spec@6", spec))
        XCTAssertGreaterThan(jpAll.maxDegree, 24, "precondition: bodies busier than the JP round budget")
        XCTAssertLessThan(jpAll.maxDegree, 64, "…inside the per-body contact list (no overflow)")
        XCTAssertGreaterThan(jp24.uncoloured, 0, "control: Jones–Plassmann at 24 rounds leaves contacts uncoloured (VZ-0160)")
        XCTAssertEqual(spec.uncoloured, 0, "speculative: every contact coloured in 6 rounds")
        XCTAssertEqual(spec.conflicts, 0, "speculative: proper")
        XCTAssertEqual(jpAll.conflicts, 0)
        XCTAssertLessThanOrEqual(Double(spec.colours), 1.4 * Double(jpAll.colours), "at most ~40 % more colours than a complete JP")
        XCTAssertEqual(s.colorStats.listOverflow, 0)
    }
}
