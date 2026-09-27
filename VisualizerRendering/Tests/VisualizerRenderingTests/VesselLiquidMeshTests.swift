import Metal
import XCTest
import simd
@testable import VisualizerRendering

/// `VesselLiquidMesh` — the GPU liquid body of a vessel of revolution (VesselLiquid.metal).
@MainActor
final class VesselLiquidMeshTests: XCTestCase {

    /// In a cylinder of radius 0.5, a tilted surface meets the wall where the plane says it
    /// must (level ± slope·R), every vertex stays inside the vessel, and every triangle winds
    /// with its normals (outward from the liquid).
    func testTiltedSurfaceMeetsTheWallAndWindsOutward() throws {
        let engine = SimEngine.shared
        let R: Float = 0.5
        let liquid = try XCTUnwrap(VesselLiquidMesh(engine: engine, innerProfile: [SIMD2(0, R), SIMD2(1, R)],
                                                    floorY: 0.02, segments: 32, surfaceRings: 6, wallRings: 8))
        let cb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
        liquid.encode(VesselLiquidMesh.Surface(level: 0.5, tilt: SIMD2(0.2, 0)), into: cb)
        cb.commit()
        cb.waitUntilCompleted()   // gpu-ok: test-time readback
        XCTAssertNil(cb.error)

        let n = liquid.vertexCount
        let pp = liquid.positionBuffer.contents().bindMemory(to: Float.self, capacity: n * 3)
        let np = liquid.normalBuffer.contents().bindMemory(to: Float.self, capacity: n * 3)
        func P(_ i: Int) -> SIMD3<Float> { SIMD3(pp[i * 3], pp[i * 3 + 1], pp[i * 3 + 2]) }
        func N(_ i: Int) -> SIMD3<Float> { SIMD3(np[i * 3], np[i * 3 + 1], np[i * 3 + 2]) }

        // Surface centre on the axis at the level; outer ring at azimuth 0 (+x) and 16 (−x).
        XCTAssertEqual(P(0).y, 0.5, accuracy: 1e-4)
        let outer = 1 + (6 - 1) * 32
        XCTAssertEqual(P(outer).x, R, accuracy: 1e-3)
        XCTAssertEqual(P(outer).y, 0.5 + 0.2 * R, accuracy: 1e-3, "+x wall contact")
        XCTAssertEqual(P(outer + 16).y, 0.5 - 0.2 * R, accuracy: 1e-3, "−x wall contact")
        // The wall's top ring meets the surface's outer ring.
        let wallTop = 1 + 6 * 32 + (8 - 1) * 32
        XCTAssertEqual(simd_distance(P(wallTop), P(outer)), 0, accuracy: 1e-3)

        for i in 0 ..< n {
            let p = P(i)
            XCTAssertLessThanOrEqual(simd_length(SIMD2(p.x, p.z)), R + 1e-3)
            XCTAssertGreaterThanOrEqual(p.y, 0.02 - 1e-4)
        }
        let ip = liquid.indexBuffer.contents().bindMemory(to: UInt32.self, capacity: liquid.indexCount)
        var disagree = 0, checked = 0
        for t in stride(from: 0, to: liquid.indexCount, by: 3) {
            let i0 = Int(ip[t]), i1 = Int(ip[t + 1]), i2 = Int(ip[t + 2])
            let f = simd_cross(P(i1) - P(i0), P(i2) - P(i0))
            guard simd_length(f) > 1e-10 else { continue }
            checked += 1
            if simd_dot(f, N(i0) + N(i1) + N(i2)) < 0 { disagree += 1 }
        }
        XCTAssertGreaterThan(checked, 100)
        XCTAssertEqual(disagree, 0, "inside-out triangles: \(disagree)/\(checked)")
    }

    // ── The surface of a SIMULATED liquid (encode(density:)) ──────────────────────────────

    /// A cylinder of liquid settled in an MLS-MPM vessel sim, extracted: the surface is level
    /// and smooth at rest; under a steady effective gravity tilted 10° it settles perpendicular
    /// to it (slope tan 10°); it always meets the wall above the floor; triangles wind outward.
    func testSimulatedSurfaceIsLevelSmoothAndFollowsGravity() throws {
        let engine = SimEngine.shared
        let R: Float = 0.5
        let solver = try XCTUnwrap(MLSMPMSolver(engine: engine, boundsMin: SIMD3(-0.66, -0.1, -0.66),
                                                boundsMax: SIMD3(0.66, 1.2, 0.66), cellSize: 0.06, maxParticles: 40_000))
        solver.vessel = MLSMPMSolver.Vessel(floorY: 0.02, profile: [SIMD2(0.02, R), SIMD2(1.1, R)])
        solver.seedVessel(level: 0.5)
        solver.bulkModulus = 150; solver.gamma = 3; solver.viscosity = 0
        solver.volumeCorrection = 0.1; solver.noTension = true; solver.settleDamping = 0.97
        let liquid = try XCTUnwrap(VesselLiquidMesh(engine: engine, innerProfile: [SIMD2(0.02, R), SIMD2(1.1, R)],
                                                    floorY: 0.02, segments: 48, surfaceRings: 12, wallRings: 10))
        func run(_ frames: Int, gravityTilt deg: Float) throws {
            let t = deg * .pi / 180
            solver.gravity = 9.81
            solver.gravityDirection = SIMD3(sin(t), -cos(t), 0)
            for _ in 0 ..< frames {
                solver.substepsPerFrame = 5
                let cb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
                solver.encode(to: cb, wallDt: 1.0 / 60)
                cb.commit()
            }
            let cb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
            liquid.encode(density: VesselLiquidMesh.DensityGrid(buffer: solver.gridMassBuffer, origin: solver.boundsMin,
                                                               cellSize: solver.cellSize, resolution: solver.gridResolution,
                                                               iso: 0.5 * solver.restDensity), into: cb)
            cb.commit()
            cb.waitUntilCompleted()   // gpu-ok: test-time readback
            XCTAssertNil(cb.error)
        }
        let pp = liquid.positionBuffer.contents().bindMemory(to: Float.self, capacity: liquid.vertexCount * 3)
        let np = liquid.normalBuffer.contents().bindMemory(to: Float.self, capacity: liquid.vertexCount * 3)
        func P(_ i: Int) -> SIMD3<Float> { SIMD3(pp[i * 3], pp[i * 3 + 1], pp[i * 3 + 2]) }
        func N(_ i: Int) -> SIMD3<Float> { SIMD3(np[i * 3], np[i * 3 + 1], np[i * 3 + 2]) }
        let surf = 1 + liquid.surfaceRings * liquid.segments

        // At rest: level, smooth normals, the contact ring meets the wall above the floor.
        try run(150, gravityTilt: 0)
        let ys = (0 ..< surf).map { P($0).y }
        let spread = (ys.max() ?? 0) - (ys.min() ?? 0)
        let tiltN = (0 ..< surf).map { acos(min(1, N($0).y)) * 180 / .pi }.max() ?? 0
        print("VesselLiquid(density): rest level \(ys.reduce(0, +) / Float(ys.count)), spread \(spread) m, max normal tilt \(tiltN)°")
        // The interior's ripple (inner 70 % of the radius — what a mirror-dark liquid shows as
        // a broken reflection) apart from the rim, where the glass correction shapes it.
        let interior = (0 ..< surf).filter { simd_length(SIMD2(P($0).x, P($0).z)) < 0.7 * R }
        let tiltsIn = interior.map { acos(min(1, N($0).y)) * 180 / .pi }
        let rmsIn = (tiltsIn.map { $0 * $0 }.reduce(0, +) / Float(max(1, tiltsIn.count))).squareRoot()
        print(String(format: "VesselLiquid(density): interior normal tilt rms %.2f° max %.2f°", rmsIn, tiltsIn.max() ?? 0))
        // Measured: 1 blur pass 0.38°, 2 passes 0.28° (the default), 3 → 0.32°, 4 → 0.48°.
        XCTAssertLessThan(rmsIn, 0.4, "resting interior ripple")
        for r in 0 ..< liquid.surfaceRings {
            let idx = (0 ..< liquid.segments).map { 1 + r * liquid.segments + $0 }
            let tilts = idx.map { acos(min(1, N($0).y)) * 180 / .pi }
            let hs = idx.map { P($0).y }
            print(String(format: "VesselLiquid(density):   ring %2d r %.3f  height %.3f…%.3f  tilt max %.1f° mean %.1f°", r,
                         simd_length(SIMD2(P(idx[0]).x, P(idx[0]).z)), hs.min() ?? 0, hs.max() ?? 0,
                         tilts.max() ?? 0, tilts.reduce(0, +) / Float(tilts.count)))
        }
        XCTAssertLessThan(spread, 0.015, "resting surface not level")
        // (Two blur passes + a three-node slope stencil: the particle ripple reads ≈ 1°; one pass
        // and a 1.5-node stencil left ≈ 2.4°, which a dark mirror-like liquid printed as a
        // hammered, radially streaked reflection of the sky.)
        XCTAssertLessThan(tiltN, 2, "resting surface normals")
        for j in 0 ..< liquid.segments {
            XCTAssertGreaterThan(P(1 + (liquid.surfaceRings - 1) * liquid.segments + j).y, 0.2, "contact line at the floor")
        }

        // Steady effective gravity leaning 10° toward +x: the surface settles perpendicular to it,
        // the liquid piling up on the +x side (slope +tan 10°).
        solver.settleDamping = 0.95
        try run(180, gravityTilt: 10)
        var sx: Float = 0, sxx: Float = 0, sy: Float = 0, sxy: Float = 0, n: Float = 0
        for i in 0 ..< surf {
            let q = P(i)
            guard simd_length(SIMD2(q.x, q.z)) < 0.35 else { continue }
            sx += q.x; sxx += q.x * q.x; sy += q.y; sxy += q.x * q.y; n += 1
        }
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        print("VesselLiquid(density): 10° gravity tilt → surface slope \(slope) (tan 10° = \(tan(10 * Float.pi / 180)))")
        XCTAssertEqual(slope, tan(10 * .pi / 180), accuracy: 0.03, "surface ⟂ effective gravity")

        // Winding: every triangle agrees with its normals.
        let ip = liquid.indexBuffer.contents().bindMemory(to: UInt32.self, capacity: liquid.indexCount)
        var disagree = 0, checked = 0
        for t in stride(from: 0, to: liquid.indexCount, by: 3) {
            let i0 = Int(ip[t]), i1 = Int(ip[t + 1]), i2 = Int(ip[t + 2])
            let f = simd_cross(P(i1) - P(i0), P(i2) - P(i0))
            guard simd_length(f) > 1e-10 else { continue }
            checked += 1
            if simd_dot(f, N(i0) + N(i1) + N(i2)) < 0 { disagree += 1 }
        }
        XCTAssertEqual(disagree, 0, "inside-out triangles: \(disagree)/\(checked)")
    }

    /// A ROUND-BELLIED vessel (a coffee pot's: a small floor disc, the wall flaring out above it
    /// to a belly, narrowing to the neck) — the liquid at rest must still read level right up to
    /// the glass, all the way round. Columns near the wall stand on the flaring bottom, not on
    /// the floor disc: a column walk that starts at the floor's height begins inside the glass
    /// there, reads dry, and either drops the contact line to the floor (a curtain down the
    /// wall) or fakes a steep slope that droops the surface into the glass (measured in Vintage
    /// Diner Ultra's pot, on the side where the node lattice sits closest to the wall).
    func testBelliedVesselSurfaceStaysLevelToTheWall() throws {
        let engine = SimEngine.shared
        // (y, r): floor disc r 0.38 at y 0.022, flaring to a 0.58 belly at 0.26, then narrowing.
        var profile: [SIMD2<Float>] = []
        for i in 0 ... 40 {
            let y = 0.022 + Float(i) * 0.025
            let r: Float = y < 0.26 ? 0.38 + 0.20 * sin(.pi / 2 * (y - 0.022) / 0.238)
                                    : 0.58 - 0.12 * (y - 0.26) / 0.76
            profile.append(SIMD2(y, r))
        }
        let rMax = (profile.map(\.y).max() ?? 0.58) + 0.12       // the pot's own domain (an off-axis lattice)
        let solver = try XCTUnwrap(MLSMPMSolver(engine: engine, boundsMin: SIMD3(-rMax, -0.1, -rMax),
                                                boundsMax: SIMD3(rMax, 1.2, rMax), cellSize: 0.06, maxParticles: 40_000))
        solver.vessel = MLSMPMSolver.Vessel(floorY: 0.022, profile: profile, friction: 0)
        solver.seedVessel(level: 0.6)
        solver.bulkModulus = 150; solver.gamma = 3; solver.viscosity = 0
        solver.volumeCorrection = 0.1; solver.noTension = true; solver.settleDamping = 0.97
        let liquid = try XCTUnwrap(VesselLiquidMesh(engine: engine, innerProfile: profile, floorY: 0.022,
                                                    segments: 64, surfaceRings: 16, wallRings: 14))
        solver.gravity = 9.81
        solver.gravityDirection = SIMD3(0, -1, 0)
        for _ in 0 ..< 150 {
            solver.substepsPerFrame = 5
            let cb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
            solver.encode(to: cb, wallDt: 1.0 / 60)
            cb.commit()
        }
        let cb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
        liquid.encode(density: VesselLiquidMesh.DensityGrid(buffer: solver.gridMassBuffer, origin: solver.boundsMin,
                                                           cellSize: solver.cellSize, resolution: solver.gridResolution,
                                                           iso: 0.5 * solver.restDensity), into: cb)
        cb.commit()
        cb.waitUntilCompleted()   // gpu-ok: test-time readback
        XCTAssertNil(cb.error)

        let pp = liquid.positionBuffer.contents().bindMemory(to: Float.self, capacity: liquid.vertexCount * 3)
        func P(_ i: Int) -> SIMD3<Float> { SIMD3(pp[i * 3], pp[i * 3 + 1], pp[i * 3 + 2]) }
        let segs = liquid.segments, rings = liquid.surfaceRings
        let centre = P(0).y
        let contact = (0 ..< segs).map { P(1 + (rings - 1) * segs + $0).y }
        let outer = (rings - 4 ..< rings).flatMap { r in (0 ..< segs).map { P(1 + r * segs + $0).y } }
        print(String(format: "VesselLiquid(bellied): centre %.3f, contact %.3f…%.3f, outer rings %.3f…%.3f",
                     centre, contact.min() ?? 0, contact.max() ?? 0, outer.min() ?? 0, outer.max() ?? 0))
        for (label, j) in [("+x", 0), ("+z", segs / 4), ("−x", segs / 2), ("−z", 3 * segs / 4)] {
            let run = (0 ..< rings).map { String(format: "%.3f", P(1 + $0 * segs + j).y) }.joined(separator: " ")
            print("VesselLiquid(bellied):   \(label): \(run)")
        }
        for (j, y) in contact.enumerated() {
            XCTAssertEqual(y, centre, accuracy: 0.025, "contact line at azimuth \(j) of \(segs) is not level with the surface")
        }
        XCTAssertLessThan((outer.max() ?? 0) - (outer.min() ?? 0), 0.03, "the surface droops or climbs near the glass")

        // Sloshing (a walk's lurch: effective gravity leaning 15° for 0.3 s, then free): the
        // contact line rides up and down the glass but never tears — it never drops to the floor
        // and neighbouring azimuths never part by more than a slosh can put between them.
        solver.settleDamping = 1
        func frames(_ k: Int, lean deg: Float) throws {
            let t = deg * .pi / 180
            solver.gravityDirection = SIMD3(sin(t), -cos(t), 0)
            for _ in 0 ..< k {
                let cb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
                solver.encode(to: cb, wallDt: 1.0 / 60)
                cb.commit()
            }
        }
        try frames(18, lean: 15)
        var lowest: Float = 1, worstStep: Float = 0, highest: Float = 0
        for _ in 0 ..< 24 {
            try frames(5, lean: 0)
            let cb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
            liquid.encode(density: VesselLiquidMesh.DensityGrid(buffer: solver.gridMassBuffer, origin: solver.boundsMin,
                                                               cellSize: solver.cellSize, resolution: solver.gridResolution,
                                                               iso: 0.5 * solver.restDensity), into: cb)
            cb.commit()
            cb.waitUntilCompleted()   // gpu-ok: test-time readback
            let c = (0 ..< segs).map { P(1 + (rings - 1) * segs + $0).y }
            lowest = min(lowest, c.min() ?? 0)
            highest = max(highest, c.max() ?? 0)
            for j in 0 ..< segs { worstStep = max(worstStep, abs(c[j] - c[(j + 1) % segs])) }
        }
        print(String(format: "VesselLiquid(bellied): sloshing — contact %.3f…%.3f, worst step between azimuths %.3f",
                     lowest, highest, worstStep))
        XCTAssertGreaterThan(lowest, 0.3, "the contact line tore down to the floor while sloshing")
        XCTAssertLessThan(worstStep, 0.06, "the contact line tore between neighbouring azimuths while sloshing")
    }

    /// A walked coffee pot (Vintage Diner Ultra's decanter: its real inner profile — a 0.41 m
    /// floor disc rounding out to a 0.59 m belly within a few cm, narrowing to the neck — and its
    /// scene tuning): the heavy walk rocks the body ±7° and bobs it ±2.4 m/s² while the liquid
    /// sloshes; the surface's contact line with the glass must ride the slosh without ever tearing
    /// down to the floor or apart between neighbouring azimuths (in the scene it did, most seconds).
    func testWalkedCoffeePotContactNeverTears() throws {
        let engine = SimEngine.shared
        let profile: [SIMD2<Float>] = [SIMD2(0.0220, 0.4082), SIMD2(0.0257, 0.4242), SIMD2(0.0283, 0.4425), SIMD2(0.0316, 0.4611), SIMD2(0.0355, 0.4781), SIMD2(0.0403, 0.4914), SIMD2(0.0459, 0.5024), SIMD2(0.0524, 0.5129), SIMD2(0.0596, 0.5224), SIMD2(0.0677, 0.5311), SIMD2(0.0769, 0.5392), SIMD2(0.0871, 0.5466), SIMD2(0.0985, 0.5534), SIMD2(0.1115, 0.5595), SIMD2(0.1258, 0.5651), SIMD2(0.1413, 0.5701), SIMD2(0.1577, 0.5745), SIMD2(0.1746, 0.5785), SIMD2(0.1922, 0.5819), SIMD2(0.2107, 0.5848), SIMD2(0.2300, 0.5870), SIMD2(0.2498, 0.5887), SIMD2(0.2699, 0.5897), SIMD2(0.2900, 0.5900), SIMD2(0.3101, 0.5896), SIMD2(0.3304, 0.5886), SIMD2(0.3510, 0.5869), SIMD2(0.3722, 0.5845), SIMD2(0.3940, 0.5817), SIMD2(0.4166, 0.5783), SIMD2(0.4400, 0.5743), SIMD2(0.4643, 0.5698), SIMD2(0.4892, 0.5647), SIMD2(0.5143, 0.5592), SIMD2(0.5395, 0.5531), SIMD2(0.5643, 0.5468), SIMD2(0.5888, 0.5398), SIMD2(0.6134, 0.5323), SIMD2(0.6381, 0.5242), SIMD2(0.6630, 0.5159), SIMD2(0.6880, 0.5074), SIMD2(0.7131, 0.4991), SIMD2(0.7382, 0.4908), SIMD2(0.7636, 0.4821), SIMD2(0.7892, 0.4734), SIMD2(0.8145, 0.4648), SIMD2(0.8394, 0.4565), SIMD2(0.8636, 0.4490), SIMD2(0.8870, 0.4420), SIMD2(0.9100, 0.4355), SIMD2(0.9325, 0.4294), SIMD2(0.9542, 0.4237), SIMD2(0.9750, 0.4184), SIMD2(0.9949, 0.4136), SIMD2(1.0143, 0.4091), SIMD2(1.0331, 0.4051), SIMD2(1.0511, 0.4015), SIMD2(1.0679, 0.3984), SIMD2(1.0830, 0.3956), SIMD2(1.0964, 0.3933), SIMD2(1.1081, 0.3915), SIMD2(1.1183, 0.3902), SIMD2(1.1268, 0.3893), SIMD2(1.1338, 0.3888), SIMD2(1.1391, 0.3884), SIMD2(1.1431, 0.3881)]
        let floorY: Float = 0.022, rMax = (profile.map(\.y).max() ?? 0.59) + 0.12
        let solver = try XCTUnwrap(MLSMPMSolver(engine: engine, boundsMin: SIMD3(-rMax, -0.1, -rMax),
                                                boundsMax: SIMD3(rMax, 1.145 + 0.14, rMax), cellSize: 0.06,
                                                maxParticles: 32_000))
        solver.vessel = MLSMPMSolver.Vessel(floorY: floorY, profile: profile, friction: 0)
        solver.bulkModulus = 150; solver.gamma = 3; solver.viscosity = 0
        solver.noTension = true; solver.volumeCorrection = 0.1
        let liquid = try XCTUnwrap(VesselLiquidMesh(engine: engine, innerProfile: profile, floorY: floorY,
                                                    segments: 64, surfaceRings: 16, wallRings: 14))
        // Poured and pre-settled as the scene does (0.8 s of damped steps in one go).
        solver.seedVessel(level: 0.6)
        solver.gravity = 9.81; solver.gravityDirection = SIMD3(0, -1, 0)
        solver.settleDamping = 0.97; solver.substepsPerFrame = 240
        var cb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
        solver.encode(to: cb, wallDt: 0.8)
        cb.commit()
        solver.settleDamping = 1
        let segs = liquid.segments, rings = liquid.surfaceRings
        let pp = liquid.positionBuffer.contents().bindMemory(to: Float.self, capacity: liquid.vertexCount * 3)
        var lowest: Float = 1, highest: Float = 0, worstStep: Float = 0, tears = 0, samples = 0
        var t: Float = 0
        for f in 0 ..< 600 {
            // The walk (0.5 Hz stride): the body's roll tips gravity ±7°, a sway and the bob.
            let roll = 0.12 * sin(.pi * t)
            let g = SIMD3<Float>(9.81 * sin(roll) + 0.4 * sin(.pi * t + 0.5), -9.81 * cos(roll) - 2.4 * sin(2 * .pi * t),
                                 0.8 * sin(2 * .pi * t))
            solver.gravity = simd_length(g); solver.gravityDirection = g / simd_length(g)
            solver.substepsPerFrame = 5
            cb = try XCTUnwrap(engine.commandQueue.makeCommandBuffer())
            solver.encode(to: cb, wallDt: 1.0 / 60)
            if f % 5 == 4 {
                liquid.encode(density: VesselLiquidMesh.DensityGrid(buffer: solver.gridMassBuffer, origin: solver.boundsMin,
                                                                   cellSize: solver.cellSize, resolution: solver.gridResolution,
                                                                   iso: 0.5 * solver.restDensity), into: cb)
                cb.commit()
                cb.waitUntilCompleted()   // gpu-ok: test-time readback
                let c = (0 ..< segs).map { pp[(1 + (rings - 1) * segs + $0) * 3 + 1] }
                lowest = min(lowest, c.min() ?? 0); highest = max(highest, c.max() ?? 0)
                var step: Float = 0
                for j in 0 ..< segs { step = max(step, abs(c[j] - c[(j + 1) % segs])) }
                worstStep = max(worstStep, step)
                samples += 1
                if (c.min() ?? 0) < 0.3 || step > 0.06 { tears += 1 }
            } else {
                cb.commit()
            }
            t += 1.0 / 60
        }
        print(String(format: "VesselLiquid(pot walk): contact %.3f…%.3f, worst azimuth step %.3f, torn in %d of %d frames",
                     lowest, highest, worstStep, tears, samples))
        XCTAssertEqual(tears, 0, "the contact line tore (to the floor, or apart between azimuths)")
    }
}
