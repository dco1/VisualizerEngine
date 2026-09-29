import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Engine gate for CoinDEMSolver+Actuation (Digital Clock phase 2, build step 1):
/// the ballasted "Weeble" egg, rigid-velocity actuation, per-body wake, physical-unit
/// joint drive setters, pooled joints / native welds, read-backs, and a perf fence
/// over a clock-like world. Every GPU case runs the real solver through the same
/// runtime-compiled-library seam as the other CoinDEM suites, at toy scale with the
/// scene's §G configuration (1/180 s, 6 velocity iterations, sleep on).
///
/// Measured numbers are PRINTed with an `ACT_` prefix so a run's log is the record.
@MainActor
final class CoinDEMActuationTests: XCTestCase {

    // ── Harness (from CoinDEMGenericEngineTests.makeSolver) ──────────────────

    private static func makeLibrary(_ device: MTLDevice) throws -> MTLLibrary {
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard FileManager.default.fileExists(atPath: shader.path) else {
            throw XCTSkip("CoinDEM.metal not found at \(shader.path)")
        }
        return try MetalSourceLoader.makeLibrary(device: device, contentsOf: shader)
    }

    /// One device / engine / runtime-compiled CoinDEM library for the whole suite (the
    /// park-tilt bisection builds dozens of solvers; recompiling the shader for each
    /// would dominate the run).
    private static var cached: (SimEngine, MTLLibrary, MTLCommandQueue)?
    private static func shared() throws -> (SimEngine, MTLLibrary, MTLCommandQueue) {
        if let c = cached { return c }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness queue
        let c = (SimEngine(device: device), try makeLibrary(device), queue)
        cached = c
        return c
    }

    static let g: Float = 9.81

    /// The Digital Clock worker (§B1): 0.9 mm ABS shell + 6.5 mm lead cap fill.
    static let worker = CoinBallastedEgg(fatRadius: 0.0115, tipRadius: 0.0075, centerDistance: 0.017,
                                         shellThickness: 0.0009, shellDensity: 1050,
                                         ballastFillHeight: 0.0065, ballastDensity: 11300)

    /// The scene's solver knobs (§G), applied to a fresh solver.
    private static func applyClockConfig(_ s: CoinDEMSolver, dt: Float, iterations: Int,
                                         rollingResistance: Float? = nil) {
        s.solverMode = .constraint
        s.gravity = g
        s.fixedDt = dt
        s.maxSubsteps = 10
        s.velocityIterations = iterations
        // §G said 10 (fallback 14). Measured in the fence world: 10 rounds leave 135–320
        // contacts uncoloured at 1/180 × 6 and 11–184 at 1/240 × 8 / 1/120 × 8 — an
        // uncoloured contact is never solved. 14 colours all of them in every config, for
        // +0.4 ms; 16 keeps a margin at the same cost.
        s.colorRounds = 16
        s.islandUnionRounds = 6
        s.warmStart = false
        s.linDamping = 0.99995
        s.angDamping = 0.9998
        s.frictionCoeff = 0.5
        s.rollingResistance = rollingResistance ?? (0.024 / Float(iterations))
        s.restitution = 0.15
        s.restThreshold = 0.14
        s.contactSlop = 1e-4
        s.baumgarteBeta = 0.2
        s.speculativeMargin = 2e-4
        s.maxSpeed = 3
        s.maxHSpeed = 3
        s.maxOmega = 60
        s.floorY = -0.5
        s.sleepEnabled = true
        s.sleepFrames = 20
        s.sleepLinVel = 2e-4
    }

    /// `maxRadius` must cover the largest body's bounding radius (the broadphase cell
    /// is 2·maxRadius, so every contacting pair lands in a 3×3×3 neighbourhood).
    private func makeSolver(maxCoins: Int = 16, maxRadius: Float = 0.03,
                            boundsMin: SIMD3<Float> = SIMD3(-0.4, -0.1, -0.4),
                            boundsMax: SIMD3<Float> = SIMD3(0.4, 0.4, 0.4),
                            dt: Float = 1.0 / 180, iterations: Int = 6,
                            rollingResistance: Float? = nil,
                            colliders: [CoinStaticCollider]? = nil) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        guard let solver = CoinDEMSolver(engine: engine, library: lib, maxCoins: maxCoins,
                                         coinRadius: maxRadius, halfThickness: maxRadius,
                                         boundsMin: boundsMin, boundsMax: boundsMax)
        else { throw XCTSkip("solver init failed") }
        Self.applyClockConfig(solver, dt: dt, iterations: iterations, rollingResistance: rollingResistance)
        solver.setColliders(colliders ?? [.plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.15)])
        return (solver, queue)
    }

    /// Encode + commit + wait `frames` frames; returns each frame's physics GPU time (ms,
    /// gpuEndTime − gpuStartTime; 0 for a frame the whole-world sleep skipped).
    @discardableResult
    private func step(_ solver: CoinDEMSolver, _ queue: MTLCommandQueue, frames: Int,
                      wallDt: Float = 1.0 / 60, perFrame: ((Int) -> Void)? = nil) -> [Double] {
        var gpu: [Double] = []
        gpu.reserveCapacity(frames)
        for f in 0..<frames {
            guard let cb = queue.makeCommandBuffer() else { break }
            solver.encode(to: cb, wallDt: wallDt)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            if cb.status == .error { XCTFail("command buffer error: \(String(describing: cb.error))"); break }
            gpu.append(max(0, cb.gpuEndTime - cb.gpuStartTime) * 1000)
            perFrame?(f)
        }
        return gpu
    }

    // ── Small maths helpers ───────────────────────────────────────────────────

    static func up(_ q: simd_quatf) -> SIMD3<Float> { simd_act(q, SIMD3<Float>(0, 1, 0)) }
    static func tiltDeg(_ q: simd_quatf) -> Float { acos(min(max(up(q).y, -1), 1)) * 180 / .pi }
    /// Rotation angle of a quaternion, degrees — from the vector part: 2·acos(|w|) in
    /// float cannot resolve below ≈ 0.04° (one ulp of w near 1), which is most of the
    /// #13 weld gate (0.05°).
    static func angleDeg(_ q: simd_quatf) -> Float {
        let n = q.normalized
        return 2 * atan2(simd_length(n.imag), abs(n.real)) * 180 / .pi
    }
    static func v4(_ q: simd_quatf) -> SIMD4<Float> { SIMD4(q.imag, q.real) }
    static func deg(_ d: Float) -> Float { d * .pi / 180 }

    /// Relative pose of `b` in `a`'s frame: (position, rotation).
    private func relPose(_ s: CoinDEMSolver, _ a: Int, _ b: Int) -> (SIMD3<Float>, simd_quatf) {
        let qa = s.orientation(of: a)!, qb = s.orientation(of: b)!
        return (simd_act(qa.inverse, s.position(of: b)! - s.position(of: a)!), qa.inverse * qb)
    }
    private func drift(_ s: CoinDEMSolver, _ a: Int, _ b: Int,
                       from ref: (SIMD3<Float>, simd_quatf)) -> (mm: Float, deg: Float) {
        let now = relPose(s, a, b)
        return (simd_length(now.0 - ref.0) * 1000, Self.angleDeg(now.1 * ref.1.inverse))
    }

    // ── The segment-bar hull (§A1: 4 rings × 8 plan stations, inscribed) ─────

    /// `midRingsFirst` was the input-order workaround for CoinHullMath's old absolute
    /// visibility epsilon (the ring-by-ring order dropped 3 extreme vertices at mm scale).
    /// Since VZ-0148 both orders give the same exact hull; kept so the fence world and
    /// testBarHullRegistrationKeepsAll32Vertices can exercise both.
    static func barHullPoints(midRingsFirst: Bool = true) -> [SIMD3<Float>] {
        let mm: Float = 0.001
        // Plan stations (mm), CCW: 2 per tip on the 0.9 mm tip fillet (±22.5° off the
        // bisector), 1 per shoulder (fillet midpoint, just inside the sharp corner).
        let st: [SIMD2<Float>] = [SIMD2(12.558, -0.344), SIMD2(12.558, 0.344), SIMD2(9.47, 3.43),
                                  SIMD2(-9.47, 3.43), SIMD2(-12.558, 0.344), SIMD2(-12.558, -0.344),
                                  SIMD2(-9.47, -3.43), SIMD2(9.47, -3.43)]
        func inset(_ d: Float) -> [SIMD2<Float>] {
            let n = st.count
            return (0..<n).map { i in
                let p0 = st[(i + n - 1) % n], p1 = st[i], p2 = st[(i + 1) % n]
                let e0 = simd_normalize(p1 - p0), e1 = simd_normalize(p2 - p1)
                let n0 = SIMD2<Float>(-e0.y, e0.x), n1 = SIMD2<Float>(-e1.y, e1.x)   // inward (CCW)
                return p1 + d * (n0 + n1) / (1 + simd_dot(n0, n1))
            }
        }
        var pts: [SIMD3<Float>] = []
        let rings: [(Float, Float)] = midRingsFirst ? [(-2.25, 0), (2.25, 0), (-2.75, 0.5), (2.75, 0.5)]
                                                    : [(-2.75, 0.5), (-2.25, 0), (2.25, 0), (2.75, 0.5)]
        for (z, off) in rings {
            for p in inset(off) { pts.append(SIMD3(p.x * mm, p.y * mm, z * mm)) }
        }
        return pts
    }

    static let barMass: Float = 0.00103

    /// Spawn a bar hull at its DESIGN pose (R_d, t_d): q = R_d ⊗ R_p, x = t_d + R_d·comOffset.
    @discardableResult
    private func spawnBar(_ s: CoinDEMSolver, _ h: CoinDEMSolver.HullHandle,
                          _ rd: simd_quatf, _ td: SIMD3<Float>) -> Int? {
        let q = rd * h.principalRotation
        return s.spawnHull(at: td + simd_act(rd, h.comOffset), hull: h, orient: Self.v4(q),
                           mass: Self.barMass, friction: 0.3, restitution: 0.2)
    }

    /// Bar orientations: local x = long axis, y = width, z = face normal.
    static let barHorizontal = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    static let barVertical = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1))      // x→Y, y→−X, z→Z
    static let barFacingX = simd_quatf(simd_float3x3(columns: (SIMD3(0, 1, 0), SIMD3(0, 0, 1), SIMD3(1, 0, 0))))

    // ══════════════════════════════════════════════════════════════════════════
    // 1–3: ballasted-egg mass properties (CPU)
    // ══════════════════════════════════════════════════════════════════════════

    /// Volume of the two-sphere hull, closed form (fat cap + tangent frustum + tip cap).
    static func sweptVolume(_ r1: Double, _ r2: Double, _ c: Double) -> Double {
        let s = (r1 - r2) / c, k = (1 - s * s).squareRoot()
        let ya = r1 * s, yb = c + r2 * s
        let capA = Double.pi * ((r1 * r1 * ya - ya * ya * ya / 3) - (r1 * r1 * (-r1) - (-r1) * (-r1) * (-r1) / 3))
        let ub = yb - c
        let capB = Double.pi * ((r2 * r2 * r2 - r2 * r2 * r2 / 3) - (r2 * r2 * ub - ub * ub * ub / 3))
        let ra = r1 * k, rb = r2 * k
        let frustum = Double.pi * (yb - ya) / 3 * (ra * ra + ra * rb + rb * rb)
        return capA + frustum + capB
    }

    /// #1 — shell 0 + fill 0 is a uniform solid: must reproduce spawnEgg's own
    /// integration (eggProperties) within 1%, and its mass must be ρ·V (closed form).
    func testBallastedEggDegeneratesToUniform() {
        let rho: Float = 1000
        for (r1, r2, c) in [(Float(0.055), Float(0.040), Float(0.050)),
                            (0.0115, 0.0075, 0.017), (0.03, 0.03, 0.08)] {
            let spec = CoinBallastedEgg(fatRadius: r1, tipRadius: r2, centerDistance: c, shellThickness: 0,
                                        shellDensity: rho, ballastFillHeight: 0, ballastDensity: 11300)
            let p = CoinDEMSolver.ballastedEggProperties(spec)
            let u = CoinDEMSolver.eggProperties(fatRadius: r1, tipRadius: r2, centerDistance: c)
            let vol = Self.sweptVolume(Double(r1), Double(r2), Double(c))
            print("ACT_1 r=(\(r1),\(r2),\(c)) m=\(p.mass) ρV=\(Double(rho) * vol) yFat=\(p.yFat)/\(u.yFat) yTip=\(p.yTip)/\(u.yTip) k=\(p.invInertiaK)/\(u.invInertiaK)")
            XCTAssertEqual(Double(p.mass), Double(rho) * vol, accuracy: Double(rho) * vol * 0.005, "m = ρ·V")
            XCTAssertEqual(p.yFat, u.yFat, accuracy: abs(u.yFat) * 0.01 + 1e-7)
            XCTAssertEqual(p.yTip, u.yTip, accuracy: abs(u.yTip) * 0.01 + 1e-7)
            XCTAssertEqual(p.comBelowFatCenter, u.yFat, accuracy: abs(u.yFat) * 0.01 + 1e-7, "d = yFat (both −y_c)")
            for (a, b) in [(p.invInertiaK.x, u.invInertiaK.x), (p.invInertiaK.y, u.invInertiaK.y),
                           (p.invInertiaK.z, u.invInertiaK.z)] {
                XCTAssertEqual(a, b, accuracy: b * 0.01)
            }
            XCTAssertEqual(p.boundingRadius, max(abs(u.yFat) + r1, abs(u.yTip) + r2), accuracy: 1e-5)
        }
    }

    /// #2 — r1 = r2, c = 0 is a sphere: a spherical shell + a spherical-cap ballast has
    /// closed-form V, centroid and inertia (antiderivatives, not the solver's quadrature).
    func testBallastedSphereShellAndCapMatchClosedForm() {
        let R = 0.02, t = 0.002, h = 0.006, rhoS = 1000.0, rhoB = 8000.0
        let spec = CoinBallastedEgg(fatRadius: Float(R), tipRadius: Float(R), centerDistance: 0,
                                    shellThickness: Float(t), shellDensity: Float(rhoS),
                                    ballastFillHeight: Float(h), ballastDensity: Float(rhoB))
        let p = CoinDEMSolver.ballastedEggProperties(spec)
        let a = R - t, pi = Double.pi
        // Shell: M = ρ·4/3π(R³−a³), F = 0, I (any centroidal axis) = 8π/15·ρ(R⁵−a⁵).
        let mS = rhoS * 4.0 / 3.0 * pi * (R * R * R - a * a * a)
        let iS = 8.0 * pi / 15.0 * rhoS * (pow(R, 5) - pow(a, 5))
        // Cap y ∈ [−a, −a+h] of the inner sphere, r² = a² − y².
        func anti(_ f: (Double) -> Double) -> Double { f(-a + h) - f(-a) }
        let mB = rhoB * pi * anti { y in a * a * y - y * y * y / 3 }
        let fB = rhoB * pi * anti { y in a * a * y * y / 2 - y * y * y * y / 4 }
        let aB = rhoB * pi / 2 * anti { y in pow(a, 4) * y - 2 * a * a * pow(y, 3) / 3 + pow(y, 5) / 5 }
        let dB = rhoB * pi * anti { y in pow(a, 4) * y / 4 + a * a * pow(y, 3) / 6 - 3 * pow(y, 5) / 20 }
        let m = mS + mB, yc = fB / m
        let iAxis = iS + aB, iDiam = iS + dB - m * yc * yc
        let gotIDiam = Double(p.mass / p.invInertiaK.x), gotIAxis = Double(p.mass / p.invInertiaK.y)
        print("ACT_2 m=\(p.mass)/\(m) d=\(p.comBelowFatCenter)/\(-yc) Idiam=\(gotIDiam)/\(iDiam) Iaxis=\(gotIAxis)/\(iAxis)")
        XCTAssertEqual(Double(p.mass), m, accuracy: m * 0.005, "V (mass)")
        XCTAssertEqual(Double(p.comBelowFatCenter), -yc, accuracy: abs(yc) * 0.005, "centroid")
        XCTAssertEqual(gotIDiam, iDiam, accuracy: iDiam * 0.005, "I_diam")
        XCTAssertEqual(gotIAxis, iAxis, accuracy: iAxis * 0.005, "I_axis")
        // Shell only: centred, isotropic.
        var hollow = spec; hollow.ballastFillHeight = 0
        let q = CoinDEMSolver.ballastedEggProperties(hollow)
        XCTAssertEqual(Double(q.mass), mS, accuracy: mS * 0.005)
        XCTAssertEqual(q.comBelowFatCenter, 0, accuracy: 1e-7)
        XCTAssertEqual(Double(q.mass / q.invInertiaK.y), iS, accuracy: iS * 0.005)
    }

    /// #3 — the §B1 worker: m 14.67 g, d 4.87 mm, k (19872, 24711, 19872), yFat 4.87,
    /// yTip 21.87, bound 29.37 mm.
    func testWorkerSpecNumbers() {
        let p = CoinDEMSolver.ballastedEggProperties(Self.worker)
        let iDiam = p.mass / p.invInertiaK.x, iAxis = p.mass / p.invInertiaK.y
        print("ACT_3 m=\(p.mass * 1000) g d=\(p.comBelowFatCenter * 1000) mm yFat=\(p.yFat * 1000) yTip=\(p.yTip * 1000) bound=\(p.boundingRadius * 1000) k=\(p.invInertiaK) Idiam=\(iDiam) Iaxis=\(iAxis)")
        XCTAssertEqual(p.mass, 0.01467, accuracy: 0.01467 * 0.01)
        XCTAssertEqual(p.comBelowFatCenter, 0.00487, accuracy: 0.00487 * 0.01)
        XCTAssertEqual(p.yFat, 0.00487, accuracy: 0.00487 * 0.01)
        XCTAssertEqual(p.yTip, 0.02187, accuracy: 0.02187 * 0.01)
        XCTAssertEqual(p.boundingRadius, 0.02937, accuracy: 0.02937 * 0.01)
        XCTAssertEqual(p.invInertiaK.x, 19872, accuracy: 19872 * 0.01)
        XCTAssertEqual(p.invInertiaK.y, 24711, accuracy: 24711 * 0.01)
        XCTAssertEqual(p.invInertiaK.z, 19872, accuracy: 19872 * 0.01)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 4–6: the Weeble in the solver
    // ══════════════════════════════════════════════════════════════════════════

    /// Weeble resting on the y = 0 plane at `tilt` about +X (fat sphere touching).
    @discardableResult
    private func spawnWorker(_ s: CoinDEMSolver, at xz: SIMD2<Float> = .zero, y0: Float = 0,
                             tilt: Float = 0, lift: Float = 0) -> Int? {
        let q = simd_quatf(angle: tilt, axis: SIMD3(1, 0, 0))
        return s.spawnBallastedEgg(Self.worker,
                                   fatCenter: SIMD3(xz.x, y0 + Self.worker.fatRadius + lift, xz.y),
                                   orient: q, friction: 0.6, restitution: 0.15)
    }

    /// Signed tilt about +X (the plane the tests rock in).
    static func signedTiltDeg(_ q: simd_quatf) -> Float {
        let u = up(q)
        return atan2(u.z, u.y) * 180 / .pi
    }

    /// #4 — from 85° with no velocity the ballasted egg self-rights (< 3° within 3 s);
    /// the uniform egg of the same outer shape lies down (control, cf. CoinDEMEggTests:114).
    func testWeebleSelfRightsFrom85Degrees() throws {
        let (s, q) = try makeSolver()
        let w = try XCTUnwrap(spawnWorker(s, at: SIMD2(-0.1, 0), tilt: Self.deg(85)))
        // Control: uniform egg, same outer shape and mass.
        let props = CoinDEMSolver.ballastedEggProperties(Self.worker)
        let u = CoinDEMSolver.eggProperties(fatRadius: 0.0115, tipRadius: 0.0075, centerDistance: 0.017)
        let qc = simd_quatf(angle: Self.deg(85), axis: SIMD3(1, 0, 0))
        let fatC = SIMD3<Float>(0.1, 0.0115, 0)
        let ctrl = try XCTUnwrap(s.spawnEgg(at: fatC - simd_act(qc, SIMD3(0, u.yFat, 0)),
                                            fatRadius: 0.0115, tipRadius: 0.0075, centerDistance: 0.017,
                                            orient: Self.v4(qc), mass: props.mass, friction: 0.6, restitution: 0.15))
        var trace: [Float] = []
        var firstBelow3: Float = -1
        step(s, q, frames: 180) { f in
            let t = Self.tiltDeg(s.orientation(of: w)!)
            trace.append(t)
            if t < 3, firstBelow3 < 0 { firstBelow3 = Float(f + 1) / 60 }
            if t >= 3 { firstBelow3 = -1 }
        }
        let final = trace.last ?? 999
        let ctrlTilt = Self.tiltDeg(s.orientation(of: ctrl)!)
        let peaks = stride(from: 0, to: trace.count, by: 15).map { String(format: "%.1f", trace[$0]) }
        print("ACT_4 weebleFinalTilt=\(final)° settledBelow3At=\(firstBelow3) s trace(0.25s)=\(peaks) uniformEggTilt=\(ctrlTilt)°")
        XCTAssertLessThan(final, 3, "the ballasted Weeble self-rights from 85° within 3 s")
        XCTAssertGreaterThan(ctrlTilt, 60, "the uniform egg (COM toward the tip) lies down instead")
    }

    /// Rock one Weeble released from `tilt` at rest, sampling the signed tilt every
    /// substep (wallDt = fixedDt).
    private func rockTrace(dt: Float, iterations: Int, muR: Float, tiltDeg: Float,
                           seconds: Float) throws -> [Float] {
        let (s, q) = try makeSolver(dt: dt, iterations: iterations, rollingResistance: muR)
        let w = try XCTUnwrap(spawnWorker(s, tilt: Self.deg(tiltDeg)))
        var out: [Float] = [tiltDeg]
        step(s, q, frames: Int(seconds / dt), wallDt: dt) { _ in
            out.append(Self.signedTiltDeg(s.orientation(of: w)!))
        }
        return out
    }

    /// Largest release tilt the Weeble HOLDS (rolling resistance + the T4 dead-stop keep
    /// it propped): bisection over release angles, 0.6 s each.
    private func parkTiltDeg(dt: Float, iterations: Int, muR: Float) throws -> Float {
        func holds(_ deg: Float) throws -> Bool {
            let (s, q) = try makeSolver(dt: dt, iterations: iterations, rollingResistance: muR)
            let w = try XCTUnwrap(spawnWorker(s, tilt: Self.deg(deg)))
            var worst: Float = 0
            step(s, q, frames: 36) { _ in
                worst = max(worst, abs(Self.signedTiltDeg(s.orientation(of: w)!) - deg))
            }
            return worst < max(0.1 * deg, 0.15)
        }
        var lo: Float = 0.1, hi: Float = 25
        if try holds(hi) { return hi }
        for _ in 0..<9 {
            let mid = 0.5 * (lo + hi)
            if try holds(mid) { lo = mid } else { hi = mid }
        }
        return lo
    }

    /// #5 — rock period ≈ 2π/Ω (Ω = √(m g d / I_c)), and the park tilt that
    /// per-iteration rolling resistance produces (N1): ≤ 2° at μr = 0.024/iterations,
    /// ≥ 8° at the proposals' μr = 0.015 (8 iterations). Released from 20° so the
    /// 3.7°-per-half-cycle decay leaves ≥ 4 half-periods to time. This runs the LEGACY
    /// (default) rolling clamp on purpose — it is still what every shipping scene gets.
    /// The opt-in fix (`accumulatedRollingResistance`, VZ-0155) is proven iteration-
    /// independent in CoinDEMCorrectnessTests (1.948° at 6 AND 12 iterations, μr 0.024).
    func testWeebleRockPeriodAndParkTilt() throws {
        let p = CoinDEMSolver.ballastedEggProperties(Self.worker)
        let d = p.comBelowFatCenter, hc = Self.worker.fatRadius - d
        let ic = p.mass / p.invInertiaK.x + p.mass * hc * hc
        let omega = (p.mass * Self.g * d / ic).squareRoot()
        let predicted = 2 * Float.pi / omega

        let dt: Float = 1.0 / 180
        let tr = try rockTrace(dt: dt, iterations: 6, muR: 0.024 / 6, tiltDeg: 20, seconds: 2)
        var crossings: [Float] = []
        for i in 1..<tr.count where (tr[i - 1] > 0) != (tr[i] > 0) && tr[i - 1] != tr[i] {
            let f = tr[i - 1] / (tr[i - 1] - tr[i])
            crossings.append((Float(i - 1) + f) * dt)
        }
        var peaks: [Float] = []
        var i = 1
        while i < tr.count - 1 {
            if abs(tr[i]) >= abs(tr[i - 1]) && abs(tr[i]) > abs(tr[i + 1]) && abs(tr[i]) > 0.3 { peaks.append(abs(tr[i])) }
            i += 1
        }
        let halves = zip(crossings.dropFirst(), crossings).map { $0 - $1 }
        let measured = halves.isEmpty ? 0 : 2 * halves.prefix(4).reduce(0, +) / Float(min(4, halves.count))
        let decay = peaks.count > 1 ? (peaks.first! - peaks[min(peaks.count - 1, 4)]) / Float(min(peaks.count - 1, 4)) : 0
        print("ACT_5 Ω=\(omega) rad/s predicted T=\(predicted) s measured T=\(measured) s halfPeriods=\(halves.prefix(6)) peaks=\(peaks.prefix(8)) decay/half=\(decay)° final=\(tr.last!)°")
        XCTAssertGreaterThan(crossings.count, 2, "it rocks")
        XCTAssertEqual(measured, predicted, accuracy: predicted * 0.10, "rock period within 10% of 2π/Ω")

        let parkDesign = try parkTiltDeg(dt: dt, iterations: 6, muR: 0.024 / 6)
        let parkProposal = try parkTiltDeg(dt: dt, iterations: 8, muR: 0.015)
        let park240 = try parkTiltDeg(dt: 1.0 / 240, iterations: 8, muR: 0.024 / 8)
        let parkCorrected = try parkTiltDeg(dt: dt, iterations: 6, muR: 0.022 / 6)
        func formula(_ iterMuR: Float) -> Float { asin(iterMuR * hc / d) * 180 / .pi }
        print("ACT_5 park: design(1/180×6, μr=.024/6)=\(parkDesign)° [formula \(formula(0.024))°]  proposal(1/180×8, μr=.015)=\(parkProposal)° [formula \(formula(8 * 0.015))°]  fallback(1/240×8, μr=.024/8)=\(park240)°  corrected(1/180×6, μr=.022/6)=\(parkCorrected)° [formula \(formula(0.022))°]")
        // N1 mechanism: the park tilt follows asin(iterations·μr·h_c/d).
        XCTAssertEqual(parkDesign, formula(0.024), accuracy: formula(0.024) * 0.25)
        XCTAssertGreaterThanOrEqual(parkProposal, 8.0, "μr = 0.015 at 8 iterations parks ≥ 8° (documents N1, VZ-0155)")
        // The design's own number: measured 2.14° (> 2°). Kept as the design states it,
        // marked expected-to-fail; the corrected value is reported (see ACT_5 print).
        XCTExpectFailure("§B2/N1 claims ≤ 2° at μr = 0.024/iterations; measured \(parkDesign)°", options: {
            let o = XCTExpectedFailure.Options(); o.isStrict = false; return o }()) {
            XCTAssertLessThanOrEqual(parkDesign, 2.0, "μr = 0.024/iterations parks within 2° (N1)")
        }
    }

    /// #6 — egg contacts with yFat > 0 (fat centre ABOVE the COM): plane, static box,
    /// egg–egg and egg–hull report the true 0.1 mm overlap at the true point.
    func testPositiveFatOffsetContacts() throws {
        let boxTop: Float = 0.05
        let (s, _) = try makeSolver(colliders: [
            .plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5),
            .box(center: SIMD3(0.3, boxTop - 0.01, 0), halfExtents: SIMD3(0.03, 0.01, 0.03), friction: 0.5),
        ])
        let r1 = Self.worker.fatRadius, pen: Float = 1e-4
        let props = CoinDEMSolver.ballastedEggProperties(Self.worker)
        XCTAssertGreaterThan(props.yFat, 0, "the worker's fat centre sits above its COM")
        let e1 = try XCTUnwrap(s.spawnBallastedEgg(Self.worker, fatCenter: SIMD3(0, r1 - pen, 0)))
        let e1t = try XCTUnwrap(s.spawnBallastedEgg(Self.worker, fatCenter: SIMD3(0, r1 - pen, 0.25),
                                                    orient: simd_quatf(angle: Self.deg(30), axis: SIMD3(0, 0, 1))))
        let e2 = try XCTUnwrap(s.spawnBallastedEgg(Self.worker, fatCenter: SIMD3(0.3, boxTop + r1 - pen, 0)))
        let e3 = try XCTUnwrap(s.spawnBallastedEgg(Self.worker, fatCenter: SIMD3(-0.3 - (r1 - pen / 2), 0.1, 0)))
        let e4 = try XCTUnwrap(s.spawnBallastedEgg(Self.worker, fatCenter: SIMD3(-0.3 + (r1 - pen / 2), 0.1, 0)))
        let e5 = try XCTUnwrap(s.spawnBallastedEgg(Self.worker, fatCenter: SIMD3(0, 0.1, -0.3)))
        let hull = try XCTUnwrap(s.registerHull(vertices: Self.barHullPoints()))
        let faceX = -(r1 - pen)
        let bar = try XCTUnwrap(spawnBar(s, hull, Self.barFacingX, SIMD3(faceX - 0.00275, 0.104, -0.3)))
        s.generateContactsNow()

        func check(_ label: String, _ a: Int, _ b: Int?, axis: SIMD3<Float>, cpY: Float) {
            let cs = s.contacts(touching: [a]).filter { c in
                b == nil ? c.meta.y == 0xFFFF_FFFF : (Int(c.meta.x) == b! || Int(c.meta.y) == b!)
            }
            guard let deepest = cs.max(by: { $0.nrm.w < $1.nrm.w }) else {
                return XCTFail("\(label): no contact")
            }
            let owner = Int(deepest.meta.x)
            let cp = s.position(of: owner)! + SIMD3(deepest.rA.x, deepest.rA.y, deepest.rA.z)
            let n = SIMD3(deepest.nrm.x, deepest.nrm.y, deepest.nrm.z)
            print("ACT_6 \(label): contacts=\(cs.count) depth=\(deepest.nrm.w * 1000) mm cp.y=\(cp.y) (want \(cpY)) n=\(n)")
            XCTAssertEqual(deepest.nrm.w, pen, accuracy: 0.5e-4, "\(label): depth")
            XCTAssertEqual(cp.y, cpY, accuracy: 3e-4, "\(label): contact height (no ghost point)")
            XCTAssertGreaterThan(abs(simd_dot(n, axis)), cos(Self.deg(2)), "\(label): normal")
        }
        check("egg–plane", e1, nil, axis: SIMD3(0, 1, 0), cpY: -pen)
        check("egg–plane tilted 30°", e1t, nil, axis: SIMD3(0, 1, 0), cpY: -pen)
        check("egg–box", e2, nil, axis: SIMD3(0, 1, 0), cpY: boxTop)
        check("egg–egg", e3, e4, axis: SIMD3(1, 0, 0), cpY: 0.1)
        check("egg–hull", e5, bar, axis: SIMD3(1, 0, 0), cpY: 0.1)
    }

    /// Registration of the §A1 bar hull (not in §F.3; found while diagnosing the fence).
    /// Before VZ-0148 CoinHullMath's incremental hull used an effectively-absolute
    /// visibility tolerance: in ring order it kept 29 of 32 vertices (11 non-manifold
    /// edges, COM 1.3 mm off), and even mid-rings-first left 7 non-manifold edges and a
    /// +23% volume. The Double, scale-relative quickhull keeps all 32 in EITHER order with
    /// a closed hull and the COM at the bar's centre (the full scale-invariance proof is
    /// CoinDEMCorrectnessTests.testHullMathIsScaleInvariantOnTheClockBar).
    func testBarHullRegistrationKeepsAll32Vertices() throws {
        let (s, _) = try makeSolver()
        let ringOrder = try XCTUnwrap(s.registerHull(vertices: Self.barHullPoints(midRingsFirst: false)))
        let midFirst = try XCTUnwrap(s.registerHull(vertices: Self.barHullPoints(midRingsFirst: true)))
        print("ACT_6b ring order: \(ringOrder.vertices.count) vertices, COM offset \(ringOrder.comOffset * 1000) mm, principal \(Self.angleDeg(ringOrder.principalRotation))°; mid rings first: \(midFirst.vertices.count) vertices, COM offset \(midFirst.comOffset * 1000) mm, principal \(Self.angleDeg(midFirst.principalRotation))°")
        // Face-set audit: a closed hull has every undirected edge in exactly 2 faces.
        func audit(_ pts: [SIMD3<Float>]) -> (faces: Int, badEdges: Int, volume: Float) {
            let f = CoinHullMath.convexHullFaces(pts) ?? []
            var edges: [String: Int] = [:]
            for t in f { for (a, b) in [(t.0, t.1), (t.1, t.2), (t.2, t.0)] { edges["\(min(a, b)):\(max(a, b))", default: 0] += 1 } }
            let vol = f.reduce(Float(0)) { $0 + simd_dot(pts[$1.0], simd_cross(pts[$1.1], pts[$1.2])) / 6 }
            return (f.count, edges.values.filter { $0 != 2 }.count, vol)
        }
        let aRing = audit(Self.barHullPoints(midRingsFirst: false)), aMid = audit(Self.barHullPoints(midRingsFirst: true))
        print("ACT_6b face audit: ring order faces=\(aRing.faces) nonManifoldEdges=\(aRing.badEdges) volume=\(aRing.volume * 1e9) mm³; mid first faces=\(aMid.faces) nonManifoldEdges=\(aMid.badEdges) volume=\(aMid.volume * 1e9) mm³")
        // VZ-0148 fixed: these used to pin the bug (ring order < 32, an XCTExpectFailure on
        // the closed-hull / COM checks); they now require the correct hull in both orders.
        XCTAssertEqual(ringOrder.vertices.count, 32, "ring order keeps all 32 (scale-relative hull, VZ-0148)")
        XCTAssertEqual(midFirst.vertices.count, 32, "mid rings first keeps all 32")
        XCTAssertEqual(aRing.badEdges, 0, "closed hull (ring order)")
        XCTAssertEqual(aMid.badEdges, 0, "closed hull (mid rings first)")
        XCTAssertEqual(aRing.volume, aMid.volume, accuracy: aMid.volume * 1e-6, "same solid in either order")
        XCTAssertLessThan(simd_length(ringOrder.comOffset), 1e-6, "the symmetric bar's COM is its centre")
        XCTAssertLessThan(simd_length(midFirst.comOffset), 1e-6, "the symmetric bar's COM is its centre")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 7–9: actuation + wake
    // ══════════════════════════════════════════════════════════════════════════

    /// #7 — a worker welded to a bar (the carry), launched as ONE rigid body through the
    /// assembly COM, lands with the weld intact: drift < 0.2 mm / 0.3°.
    func testSetRigidVelocityWeldedPairFliesAsOne() throws {
        let (s, q) = try makeSolver()
        let r1 = Self.worker.fatRadius
        let w = try XCTUnwrap(spawnWorker(s))
        let hull = try XCTUnwrap(s.registerHull(vertices: Self.barHullPoints()))
        // Upright bar in front of the belly: tip 7.5 mm above the table, back face on the
        // belly at the fat equator (50 µm gap), face normal +Z.
        let bar = try XCTUnwrap(spawnBar(s, hull, Self.barVertical, SIMD3(0, 0.0075 + 0.013, r1 + 0.00275 + 5e-5)))
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 4, placeholderBody: w))
        let weld = try XCTUnwrap(pool.takeWeld())
        s.enableWeld(weld, bodyA: w, bodyB: bar, worldAnchor: SIMD3(0, r1, r1))
        step(s, q, frames: 90)                                    // settle into the load lean
        let lean = Self.tiltDeg(s.orientation(of: w)!)
        let ref = relPose(s, w, bar)
        let assembly = [w, bar]
        let y0 = s.centerOfMass(of: assembly).y
        s.setRigidVelocity(slots: assembly, linear: SIMD3(0.06, 0.485, 0.03), angular: SIMD3(0, 6, 0),
                           about: s.centerOfMass(of: assembly))
        var maxRise: Float = 0, worstFlight = (mm: Float(0), deg: Float(0))
        step(s, q, frames: 120) { _ in
            maxRise = max(maxRise, s.centerOfMass(of: assembly).y - y0)
            let d = self.drift(s, w, bar, from: ref)
            worstFlight = (max(worstFlight.mm, d.mm), max(worstFlight.deg, d.deg))
        }
        let final = drift(s, w, bar, from: ref)
        let yaw = atan2(-simd_act(s.orientation(of: w)!, SIMD3<Float>(0, 0, 1)).x,
                        simd_act(s.orientation(of: w)!, SIMD3<Float>(0, 0, 1)).z) * 180 / .pi
        print("ACT_7 loadLean=\(lean)° rise=\(maxRise * 1000) mm yaw≈\(yaw)° drift worst=\(worstFlight.mm) mm/\(worstFlight.deg)° final=\(final.mm) mm/\(final.deg)°")
        XCTAssertGreaterThan(maxRise, 0.008, "the pair actually hopped")
        XCTAssertLessThan(final.mm, 0.2, "weld position drift after the hop")
        XCTAssertLessThan(final.deg, 0.3, "weld rotation drift after the hop")
    }

    /// #8 — a yaw spin on a point contact is not damped by the engine (no torsional
    /// friction): > 70% of it remains after 1 s. Documents why the host's yaw stop exists.
    func testSetAngularVelocityYawPersistsOnPointContact() throws {
        let (s, q) = try makeSolver()
        let w = try XCTUnwrap(spawnWorker(s))
        step(s, q, frames: 40)                                    // settles, falls asleep
        XCTAssertTrue(s.isAsleep(w), "precondition: the idle worker sleeps")
        s.setAngularVelocity(ofSlot: w, to: SIMD3(0, 5, 0))
        XCTAssertFalse(s.isAsleep(w), "the setter wakes its body")
        step(s, q, frames: 60)
        let wy = s.angularVelocity(of: w)!.y
        print("ACT_8 yaw after 1 s = \(wy) rad/s of 5 (\(wy / 5 * 100)%) asleep=\(s.isAsleep(w)) tilt=\(Self.tiltDeg(s.orientation(of: w)!))°")
        XCTAssertGreaterThan(wy, 0.7 * 5)
    }

    /// #9 — `wake` clears the flag AND the timer: the woken body obeys setVelocity and
    /// stays awake; a far body stays asleep; asleepCount drops by exactly one. Control
    /// (T7): clearing only asleep[] re-freezes at the end of the same frame.
    func testWakeResetsTimerAndLeavesOthersAsleep() throws {
        let (s, q) = try makeSolver()
        let he = SIMD3<Float>(0.005, 0.005, 0.005)
        let a = try XCTUnwrap(s.spawnBox(at: SIMD3(-0.1, 0.005, 0), halfExtents: he, mass: 0.005, friction: 0))
        let b = try XCTUnwrap(s.spawnBox(at: SIMD3(0.1, 0.005, 0), halfExtents: he, mass: 0.005))
        // A spinner keeps the world stepping (so asleep bodies' timers keep counting).
        let c = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.15, 0), halfExtents: SIMD3(0.01, 0.002, 0.01), mass: 0.005))
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 1, placeholderBody: c))
        let jc = try pooledHinge(s, pool, c, nil, SIMD3(0, 0.15, 0), SIMD3(0, 1, 0))
        s.setHingeMotor(jc, targetVelocity: 3, maxTorque: 1e-3)
        step(s, q, frames: 90)
        XCTAssertTrue(s.isAsleep(a)); XCTAssertTrue(s.isAsleep(b)); XCTAssertFalse(s.isAsleep(c))
        XCTAssertEqual(s.asleepCount, 2)

        // Control: flag only.
        let flag = s.asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: s.maxCoins)
        flag[a] = 0
        let x0 = s.position(of: a)!.x
        s.setVelocity(ofSlot: a, to: SIMD3(0.05, 0, 0))
        step(s, q, frames: 1)
        let refrozen = s.isAsleep(a)
        let x1 = s.position(of: a)!.x
        step(s, q, frames: 5)
        let x2 = s.position(of: a)!.x
        print("ACT_9 flag-only wake: refrozenAfter1Frame=\(refrozen) moved=\((x1 - x0) * 1000) mm then \((x2 - x1) * 1000) mm")
        XCTAssertTrue(refrozen, "T7: clearing only asleep[] is re-frozen at the end of the same frame")

        // The real wake.
        s.wake([a])
        XCTAssertFalse(s.isAsleep(a))
        XCTAssertEqual(s.asleepCount, 1, "exactly one body woke")
        let x3 = s.position(of: a)!.x
        s.setVelocity(ofSlot: a, to: SIMD3(0.05, 0, 0))
        var everAsleep = false
        step(s, q, frames: 10) { _ in everAsleep = everAsleep || s.isAsleep(a) }
        let moved = s.position(of: a)!.x - x3
        print("ACT_9 wake(): moved=\(moved * 1000) mm in 10 frames (expect ≈8.3) everAsleep=\(everAsleep) farAsleep=\(s.isAsleep(b)) asleepCount=\(s.asleepCount)")
        XCTAssertFalse(everAsleep, "a properly woken body stays awake")
        XCTAssertGreaterThan(moved, 0.005, "and obeys setVelocity")
        XCTAssertTrue(s.isAsleep(b), "the far body stays asleep")
        XCTAssertEqual(s.asleepCount, 1)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 10–12: joint drive setters
    // ══════════════════════════════════════════════════════════════════════════

    /// Frames (at 60 Hz) until `value()` is within 5% of `target` (−1 if never).
    private func framesToReach(_ s: CoinDEMSolver, _ q: MTLCommandQueue, frames: Int, target: Float,
                               _ value: @escaping () -> Float) -> (first: Int, final: Float) {
        var first = -1
        var last: Float = 0
        step(s, q, frames: frames) { f in
            last = value()
            if first < 0, abs(last - target) <= abs(target) * 0.05 { first = f + 1 }
        }
        return (first, last)
    }

    /// A pooled slot enabled as a hinge / prismatic (the scene's rig path; see the
    /// axis-alignment caveat in CoinDEMSolver+Actuation.swift).
    private func pooledHinge(_ s: CoinDEMSolver, _ pool: CoinJointPool, _ a: Int, _ b: Int?,
                             _ anchor: SIMD3<Float>, _ axis: SIMD3<Float>) throws -> Int {
        let slot = try XCTUnwrap(pool.takeSlot())
        s.enableHinge(slot: slot, bodyA: a, bodyB: b, worldAnchor: anchor, worldAxis: axis)
        return slot
    }
    private func pooledPrismatic(_ s: CoinDEMSolver, _ pool: CoinJointPool, _ a: Int, _ b: Int?,
                                 _ anchor: SIMD3<Float>, _ axis: SIMD3<Float>) throws -> Int {
        let slot = try XCTUnwrap(pool.takeSlot())
        s.enablePrismatic(slot: slot, bodyA: a, bodyB: b, worldAnchor: anchor, worldAxis: axis)
        return slot
    }

    /// Angle (°) between a joint's axis as body A carries it and `axis` — 0 while the
    /// joint holds its axis; ~180 once it has flipped.
    private func axisDeviationDeg(_ s: CoinDEMSolver, _ joint: Int, body: Int, _ axis: SIMD3<Float>) -> Float {
        let j = s.jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: CoinDEMSolver.maxJoints)[joint]
        let aW = simd_normalize(simd_act(s.orientation(of: body)!, SIMD3(j.axisA.x, j.axisA.y, j.axisA.z)))
        return acos(min(max(simd_dot(aW, simd_normalize(axis)), -1), 1)) * 180 / .pi
    }

    /// #10 (prerequisite) — the axis-alignment sign. Until VZ-0147 the kernel's hinge /
    /// prismatic alignment bias pushed parallel axes APART, so a joint built by
    /// addHingeJoint / addPrismaticJoint (axisB = +axis) flipped its body 180° as soon as
    /// anything seeded a misalignment (measured: 180° / 176.9°), and only the pooled path
    /// (then storing axisB = −axis) held. With the kernel sign fixed and `axisBSign` = +1,
    /// BOTH paths store parallel axes and BOTH must hold under the same skew + gravity
    /// lever load. (Name kept for the log's history; the assertion now pins the fix.)
    func testAxisAlignmentConventionPooledHoldsAddFlips() throws {
        let (s, q) = try makeSolver(maxCoins: 12, maxRadius: 0.05,
                                    boundsMin: SIMD3(-0.4, -0.1, -0.4), boundsMax: SIMD3(0.4, 0.6, 0.4))
        s.setColliders([])
        let axis = simd_normalize(SIMD3<Float>(0.3, 0.2, 1))
        let perp = simd_normalize(simd_cross(axis, SIMD3(0, 1, 0)))
        let skew = simd_quatf(angle: Self.deg(1), axis: perp)
        let placeholder = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.05, 0.35), halfExtents: SIMD3(0.005, 0.005, 0.005), mass: 0.01))
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 4, placeholderBody: placeholder))
        // 10 g flap (80 × 10 × 20 mm) hinged to the world at one end, skewed 1°.
        func flap(_ z: Float) throws -> Int {
            let b = try XCTUnwrap(s.spawnBox(at: SIMD3(0.05, 0.4, z), halfExtents: SIMD3(0.04, 0.005, 0.01), mass: 0.01))
            return b
        }
        let fAdd = try flap(-0.3), fPool = try flap(-0.1)
        let jAdd = try XCTUnwrap(s.addHingeJoint(bodyA: fAdd, bodyB: nil, worldAnchor: SIMD3(0, 0.4, -0.3), worldAxis: axis))
        let jPool = try pooledHinge(s, pool, fPool, nil, SIMD3(0, 0.4, -0.1), axis)
        // 10 g slider on a skew world axis, anchored 30 mm off its COM (a lever load).
        func slider(_ z: Float) throws -> Int {
            try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.4, z), halfExtents: SIMD3(0.01, 0.01, 0.01), mass: 0.01))
        }
        let pAdd = try slider(0.1), pPool = try slider(0.3)
        let kAdd = try XCTUnwrap(s.addPrismaticJoint(bodyA: pAdd, bodyB: nil, worldAnchor: SIMD3(0.03, 0.4, 0.1), worldAxis: axis))
        let kPool = try pooledPrismatic(s, pool, pPool, nil, SIMD3(0.03, 0.4, 0.3), axis)
        for k in [kAdd, kPool] { s.setPrismaticMotor(k, targetVelocity: 0, maxForce: 1) }   // hold the slide
        // Seed: rotate every jointed body 1° off its axis (the stored local axes turn with it).
        let p = s.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: s.maxCoins)
        for b in [fAdd, fPool, pAdd, pPool] {
            let q1 = skew * s.orientation(of: b)!
            p[b].orient = Self.v4(q1); p[b].prevOrient = p[b].orient
        }
        var worst = [Float](repeating: 0, count: 4)
        step(s, q, frames: 120) { _ in
            for (i, (j, b)) in [(jAdd, fAdd), (jPool, fPool), (kAdd, pAdd), (kPool, pPool)].enumerated() {
                worst[i] = max(worst[i], self.axisDeviationDeg(s, j, body: b, axis))
            }
        }
        print("ACT_10pre worst axis deviation over 2 s: hinge add=\(worst[0])° pooled=\(worst[1])°; prismatic add=\(worst[2])° pooled=\(worst[3])°")
        // Holds = bounded Gauss–Seidel wobble under the lever load (measured 6.5° / 4.2°
        // worst while swinging, pooled, before the fix), never the 180° flip. VZ-0147
        // flipped the first two assertions (were > 90° "flips").
        XCTAssertLessThan(worst[0], 10, "addHingeJoint holds its axis (VZ-0147 fixed)")
        XCTAssertLessThan(worst[2], 10, "addPrismaticJoint holds its axis (VZ-0147 fixed)")
        XCTAssertLessThan(worst[1], 10, "pooled enableHinge holds its axis")
        XCTAssertLessThan(worst[3], 10, "pooled enablePrismatic holds its axis")
    }

    /// #10a — hinge motor setter: a WORLD hinge (the jib slew, target = A's own spin)
    /// and a TWO-BODY hinge (target = B relative to A), each retargeted mid-run, reach
    /// the target within 0.3 s; the setter wakes a sleeping joint. Two engine fixes are
    /// pinned alongside:
    ///  • the §A1 jib (mast 0.17 m off its COM) SLEEPS at 6 iterations with no residual
    ///    velocity. Until VZ-0150 the hinge's rows were solved one after another from zero
    ///    impulse every substep, so each pass removed only ~I/I_pivot of the gravity
    ///    lever's angular momentum and a ~1.8 mm/s residual survived every substep (the
    ///    sleep test needs < 2.5·sleepLinVel = 0.5 mm/s); the joint is now one warm-started
    ///    block solve;
    ///  • a jib pivoted at its COM used to be unable to slew below 0.6 rad/s: its COM
    ///    does not move, so finalize's dead-stop (|v| < sleepLinVel && |ω| < 0.6) zeroed ω
    ///    every substep (T4, VZ-0149). Fixed: a body a motor drives with a non-zero target
    ///    is never dead-stopped, so the COM-pivoted jib now follows its 0.4 rad/s command.
    func testHingeMotorSetterWorldAndTwoBodySigns() throws {
        let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.27,
                                    boundsMin: SIMD3(-0.4, -0.1, -0.4), boundsMax: SIMD3(0.4, 0.6, 0.6))
        // §A1 jib: 520 × 12 × 14 mm, 30 g, world hinge about +Y at the mast, 0.17 m off its COM.
        let jib = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.2, 0), halfExtents: SIMD3(0.26, 0.006, 0.007), mass: 0.03))
        let comJib = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.2, 0.5), halfExtents: SIMD3(0.26, 0.006, 0.007), mass: 0.03))
        let spinner = try XCTUnwrap(s.spawnBox(at: SIMD3(-0.3, 0.3, 0.3), halfExtents: SIMD3(0.01, 0.004, 0.01), mass: 0.006))
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 5, placeholderBody: jib))
        let j0 = try pooledHinge(s, pool, jib, nil, SIMD3(0.17, 0.194, 0), SIMD3(0, 1, 0))
        let jc = try pooledHinge(s, pool, comJib, nil, SIMD3(0, 0.194, 0.5), SIMD3(0, 1, 0))
        let js = try pooledHinge(s, pool, spinner, nil, SIMD3(-0.3, 0.3, 0.3), SIMD3(0, 1, 0))
        // Two-body: a 30 g plate free on a world hinge, and a 6 g block hinged to it.
        let plate = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.3, 0.3), halfExtents: SIMD3(0.05, 0.005, 0.05), mass: 0.03))
        _ = try pooledHinge(s, pool, plate, nil, SIMD3(0, 0.3, 0.3), SIMD3(0, 1, 0))
        let blk = try XCTUnwrap(s.spawnBox(at: SIMD3(0.08, 0.3, 0.3), halfExtents: SIMD3(0.01, 0.004, 0.008), mass: 0.006))
        let jab = try pooledHinge(s, pool, plate, blk, SIMD3(0.08, 0.3, 0.3), SIMD3(0, 1, 0))
        s.setHingeMotor(j0, targetVelocity: 0, maxTorque: 0.01)
        step(s, q, frames: 40)
        let jibResidual = simd_length(s.velocity(of: jib)!)
        // VZ-0150 flipped this (was XCTAssertFalse: "the §A1 off-COM jib never sleeps", 1.77 mm/s).
        XCTAssertTrue(s.isAsleep(jib), "the §A1 off-COM jib sleeps at 6 iterations (VZ-0150 fixed)")
        XCTAssertLessThan(jibResidual, 1e-4, "and carries no residual velocity (was 1.77 mm/s)")
        XCTAssertTrue(s.isAsleep(spinner), "precondition: an unloaded COM-pivoted hinge sleeps")

        // Wake-by-setter on the sleeping spinner.
        s.setHingeMotor(js, targetVelocity: 2, maxTorque: 1e-3)
        XCTAssertFalse(s.isAsleep(spinner), "setter wakes the joint's body")
        let r0 = framesToReach(s, q, frames: 18, target: 2) { s.angularVelocity(of: spinner)!.y }

        let tw0 = s.hingeTwist(j0)!
        s.setHingeMotor(j0, targetVelocity: 0.4, maxTorque: 0.01)
        s.setHingeMotor(jc, targetVelocity: 0.4, maxTorque: 0.01)
        let r1 = framesToReach(s, q, frames: 18, target: 0.4) { s.angularVelocity(of: jib)!.y }
        let comJibOmega = s.angularVelocity(of: comJib)!.y
        let tw1 = s.hingeTwist(j0)!
        s.setHingeMotor(j0, targetVelocity: -0.3, maxTorque: 0.01)
        let r2 = framesToReach(s, q, frames: 18, target: -0.3) { s.angularVelocity(of: jib)!.y }

        func rel() -> Float { s.angularVelocity(of: blk)!.y - s.angularVelocity(of: plate)!.y }
        s.setHingeMotor(jab, targetVelocity: 3, maxTorque: 1e-3)
        let r3 = framesToReach(s, q, frames: 18, target: 3) { rel() }
        s.setHingeMotor(jab, targetVelocity: -2, maxTorque: 1e-3)
        let r4 = framesToReach(s, q, frames: 18, target: -2) { rel() }
        print("ACT_10 spinner woken → +2 at frame \(r0.first) (ω=\(r0.final)); §A1 jib: +0.4 reached at frame \(r1.first) (ω=\(r1.final)), −0.3 at \(r2.first) (ω=\(r2.final)); twist \(tw0)→\(tw1) (kernel twist = −A's angle); two-body: +3 at \(r3.first) (rel=\(r3.final)), −2 at \(r4.first) (rel=\(r4.final)); §A1 jib residual |v| at rest = \(jibResidual * 1000) mm/s; COM-pivoted jib commanded 0.4 rad/s → ω=\(comJibOmega) (was 0: T4 dead-stop, VZ-0149 fixed)")
        for (r, t) in [(r0, Float(2)), (r1, 0.4), (r2, -0.3), (r3, 3), (r4, -2)] {
            XCTAssertGreaterThan(r.first, 0, "reached \(t)")
            XCTAssertLessThanOrEqual(r.first, 18, "within 0.3 s")
            XCTAssertEqual(r.final, t, accuracy: abs(t) * 0.05)
        }
        XCTAssertLessThan(tw1, tw0, "world hinge: the kernel twist runs opposite to A's own spin")
        // VZ-0149 flipped this (was |ω| < 0.05: "dead-stopped").
        XCTAssertEqual(comJibOmega, 0.4, accuracy: 0.02, "a motor-driven body turning about its own COM is not dead-stopped (VZ-0149 fixed)")
    }

    /// #10b — prismatic motor setter (trolley on the jib, both spawned at identity):
    /// retargeting mid-run reaches the target within 0.3 s; prismaticSlide tracks it.
    func testPrismaticMotorSetter() throws {
        let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.27,
                                    boundsMin: SIMD3(-0.4, -0.1, -0.4), boundsMax: SIMD3(0.4, 0.6, 0.4))
        let jib = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.2, 0), halfExtents: SIMD3(0.26, 0.006, 0.007), mass: 0.03))
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: jib))
        let j0 = try pooledHinge(s, pool, jib, nil, SIMD3(0.17, 0.194, 0), SIMD3(0, 1, 0))
        s.setHingeMotor(j0, targetVelocity: 0, maxTorque: 0.01)   // slew brake
        let trolley = try XCTUnwrap(s.spawnBox(at: SIMD3(0.10, 0.19, 0), halfExtents: SIMD3(0.01, 0.004, 0.008), mass: 0.006))
        let j1 = try pooledPrismatic(s, pool, trolley, jib, SIMD3(0.10, 0.19, 0), SIMD3(1, 0, 0))
        step(s, q, frames: 40)
        let s0 = s.prismaticSlide(j1)!
        func relV() -> Float { s.velocity(of: trolley)!.x - s.velocity(of: jib)!.x }
        s.setPrismaticMotor(j1, targetVelocity: -0.12, maxForce: 0.02)
        let r1 = framesToReach(s, q, frames: 18, target: -0.12) { relV() }
        let s1 = s.prismaticSlide(j1)!
        s.setPrismaticMotor(j1, targetVelocity: 0.08, maxForce: 0.02)
        let r2 = framesToReach(s, q, frames: 18, target: 0.08) { relV() }
        print("ACT_10 prismatic: −0.12 reached at frame \(r1.first) (v=\(r1.final)), +0.08 at \(r2.first) (v=\(r2.final)); slide \(s0)→\(s1) m over 0.3 s")
        XCTAssertGreaterThan(r1.first, 0); XCTAssertLessThanOrEqual(r1.first, 18)
        XCTAssertGreaterThan(r2.first, 0); XCTAssertLessThanOrEqual(r2.first, 18)
        XCTAssertEqual(r1.final, -0.12, accuracy: 0.006)
        XCTAssertEqual(r2.final, 0.08, accuracy: 0.004)
        XCTAssertLessThan(s1 - s0, -0.02, "prismaticSlide follows the trolley (−0.12 m/s ≈ −0.03 m)")
    }

    /// #11 — the setters take PHYSICAL units: a motor rated 1.1 × the static load holds
    /// it, 0.9 × slips — at 6, 8 AND 12 iterations. Since VZ-0155 the kernel clamps the
    /// motor impulse ACCUMULATED over the iterations, so the setter stores the rating as
    /// given (before, it divided by velocityIterations to undo a per-iteration clamp that
    /// made a 0.9× motor really 5.4× / 7.2× / 10.8×).
    ///
    /// The hinge load is a 60 mm bar pivoted 5 mm off its COM. The END-pivoted lever
    /// (I/I_pivot = 0.25) is gated too since VZ-0150: the motor is the last row of the
    /// hinge's block solve, clamped on its accumulated impulse, so a velocity servo holds
    /// a lever at 1.1× (and 2×) its static torque and slips at 0.9×. Before, the ball
    /// core, the ⟂ rows and the motor were solved one after another and each pass handed
    /// most of the lever's load back: it crept 23° at 1.1× and 16° at 2× in 1 s.
    func testPhysicalMotorBoundHoldsStaticLoad() throws {
        for iters in [6, 8, 12] {
            let (s, q) = try makeSolver(maxCoins: 12, maxRadius: 0.05, iterations: iters)
            s.setColliders([])
            let m: Float = 0.01, arm: Float = 0.005, lever: Float = 0.03
            let tauG = m * Self.g * arm, tauLever = m * Self.g * lever
            let placeholder = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.05, 0.35), halfExtents: SIMD3(0.005, 0.005, 0.005), mass: 0.01))
            let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 8, placeholderBody: placeholder))
            func flap(_ z: Float, pivotOffset: Float) throws -> (Int, Int) {
                let b = try XCTUnwrap(s.spawnBox(at: SIMD3(pivotOffset, 0.2, z), halfExtents: SIMD3(0.03, 0.004, 0.01), mass: m))
                return (b, try pooledHinge(s, pool, b, nil, SIMD3(0, 0.2, z), SIMD3(0, 0, 1)))
            }
            func slider(_ z: Float) throws -> (Int, Int) {
                let b = try XCTUnwrap(s.spawnBox(at: SIMD3(0.2, 0.2, z), halfExtents: SIMD3(0.01, 0.01, 0.01), mass: m))
                return (b, try pooledPrismatic(s, pool, b, nil, SIMD3(0.2, 0.2, z), SIMD3(0, 1, 0)))
            }
            let (hold, jh) = try flap(-0.1, pivotOffset: arm)
            let (slip, js) = try flap(0.1, pivotOffset: arm)
            let (lev11, jl11) = try flap(-0.3, pivotOffset: lever)
            let (lev20, jl20) = try flap(0.3, pivotOffset: lever)
            let (lev09, jl09) = try flap(0.2, pivotOffset: lever)
            let (pHold, jph) = try slider(-0.1)
            let (pSlip, jps) = try slider(0.1)
            s.setHingeMotor(jh, targetVelocity: 0, maxTorque: 1.1 * tauG)
            s.setHingeMotor(js, targetVelocity: 0, maxTorque: 0.9 * tauG)
            s.setHingeMotor(jl11, targetVelocity: 0, maxTorque: 1.1 * tauLever)
            s.setHingeMotor(jl20, targetVelocity: 0, maxTorque: 2.0 * tauLever)
            s.setHingeMotor(jl09, targetVelocity: 0, maxTorque: 0.9 * tauLever)
            s.setPrismaticMotor(jph, targetVelocity: 0, maxForce: 1.1 * m * Self.g)
            s.setPrismaticMotor(jps, targetVelocity: 0, maxForce: 0.9 * m * Self.g)
            func drop(_ b: Int) -> Float { asin(min(1, abs(simd_act(s.orientation(of: b)!, SIMD3<Float>(1, 0, 0)).y))) * 180 / .pi }
            step(s, q, frames: 60)
            let dHold = drop(hold), dSlip = drop(slip)
            let yHold = (0.2 - s.position(of: pHold)!.y) * 1000, ySlip = (0.2 - s.position(of: pSlip)!.y) * 1000
            print("ACT_11 iters=\(iters) hinge τ_g=\(tauG) N·m (pivot 5 mm off COM): 1.1× drop=\(dHold)°, 0.9× drop=\(dSlip)°; prismatic: 1.1× sag=\(yHold) mm, 0.9× sag=\(ySlip) mm; end-pivoted lever (block solve, 1 s): 1.1× drop=\(drop(lev11))°, 2.0× drop=\(drop(lev20))°, 0.9× drop=\(drop(lev09))°")
            XCTAssertLessThan(dHold, 1, "iters \(iters): 1.1 × τ_g holds")
            XCTAssertGreaterThan(dSlip, 10, "iters \(iters): 0.9 × τ_g slips")
            XCTAssertLessThan(yHold, 1, "iters \(iters): 1.1 × mg holds")
            XCTAssertGreaterThan(ySlip, 5, "iters \(iters): 0.9 × mg slips")
            // VZ-0150: the lever servo is exact (was 23° / 16° of creep at 1.1× / 2×).
            XCTAssertLessThan(drop(lev11), 1, "iters \(iters): end-pivoted lever, 1.1 × τ holds")
            XCTAssertLessThan(drop(lev20), 1, "iters \(iters): end-pivoted lever, 2 × τ holds")
            XCTAssertGreaterThan(drop(lev09), 10, "iters \(iters): end-pivoted lever, 0.9 × τ slips")
        }
    }

    /// #12 — the 4-fall winch: a 12 g block on 4 parallel rods hoisted 5 cm at 60 mm/s by
    /// re-issuing setDistanceRestLength every tick stays level (< 0.3°) and ends within
    /// 0.5 mm of the rest length; the SAME ramp written straight into the lane (no
    /// wake) lets the island fall asleep mid-hoist (N3).
    func testDistanceRestLengthWinchAndSleepHazard() throws {
        struct Run { var maxLag: Float = 0; var maxTilt: Float = 0; var finalErr: Float = 0
                     var rise: Float = 0; var sleptAt = -1 }
        func hoist(viaSetter: Bool) throws -> Run {
            let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.03,
                                        boundsMin: SIMD3(-0.2, -0.1, -0.2), boundsMax: SIMD3(0.2, 0.6, 0.2))
            let top: Float = 0.307, L0: Float = 0.13, L1: Float = 0.08
            let block = try XCTUnwrap(s.spawnBox(at: SIMD3(0, top - 0.007, 0), halfExtents: SIMD3(0.011, 0.007, 0.005), mass: 0.012))
            var rods: [Int] = []
            for (dx, dz) in [(Float(-0.006), Float(-0.005)), (0.006, -0.005), (-0.006, 0.005), (0.006, 0.005)] {
                rods.append(try XCTUnwrap(s.addDistanceJoint(bodyA: block, bodyB: nil,
                                                             worldAnchorA: SIMD3(dx, top, dz),
                                                             worldAnchorB: SIMD3(dx, top + L0, dz))))
            }
            step(s, q, frames: 30)
            let y0 = s.position(of: block)!.y
            var run = Run()
            let rate: Float = 0.06
            let frames = 70
            s.wake([block])
            for f in 0..<frames {
                let rest = max(L1, L0 - rate * Float(f + 1) / 60)
                for r in rods {
                    if viaSetter { s.setDistanceRestLength(r, rest) }
                    else { s.jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: CoinDEMSolver.maxJoints)[r].anchorA.w = rest }
                }
                step(s, q, frames: 1)
                if f < 50 { run.maxLag = max(run.maxLag, s.distanceJointLength(rods[0])! - rest) }
                run.maxTilt = max(run.maxTilt, Self.tiltDeg(s.orientation(of: block)!))
                if run.sleptAt < 0, s.isAsleep(block) { run.sleptAt = f }
            }
            step(s, q, frames: 30)
            run.finalErr = rods.map { abs(s.distanceJointLength($0)! - L1) }.max()!
            run.rise = s.position(of: block)!.y - y0
            return run
        }
        let a = try hoist(viaSetter: true)
        let b = try hoist(viaSetter: false)
        let predictedLag: Float = 0.06 * (1.0 / 180) / 0.2
        print("ACT_12 setter: rise=\(a.rise * 1000) mm maxLag=\(a.maxLag * 1000) mm (bias lag rate·dt/β=\(predictedLag * 1000) mm) finalErr=\(a.finalErr * 1000) mm maxTilt=\(a.maxTilt)° sleptAt=\(a.sleptAt)")
        print("ACT_12 raw lane (no wake): rise=\(b.rise * 1000) mm sleptAt frame \(b.sleptAt)")
        XCTAssertEqual(a.rise, 0.05, accuracy: 0.001, "hoisted 5 cm")
        XCTAssertLessThan(a.finalErr, 0.0005, "|len − rest| < 0.5 mm once the ramp stops")
        XCTAssertLessThan(a.maxTilt, 0.3, "the 4-fall block stays level")
        XCTAssertEqual(a.sleptAt, -1, "re-issuing the setter every tick keeps the hoist awake")
        XCTAssertGreaterThanOrEqual(b.sleptAt, 0, "N3: without a per-tick wake the island sleeps mid-hoist")
        XCTAssertLessThan(b.rise, 0.045, "…and stops short of the target")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 13–15: welds, limits from creation (T1), pooled joints
    // ══════════════════════════════════════════════════════════════════════════

    /// #13 — a pooled (native, VZ-0150) weld holds an arbitrary (37°, skew-axis) relative
    /// orientation while the welded pair spins past 2π: drift < 0.02 mm / 0.05° over 5 s
    /// (the §6 4a gate; was < 0.05 mm / 0.1° for the two-hinge weld). Gravity off to
    /// isolate the weld. What drift remains is the integrator's, not the joint's: the COMs
    /// move on straight lines within a substep while the pair turns, an anchor error of
    /// ½·ω²·|x_A − x_B|·dt² per substep that the bias removes ×(1 − β), a steady
    /// ω²·|x_A − x_B|·dt²/(2β) ≈ 0.015 mm at 3 rad/s.
    func testWeldHoldsArbitraryRelativeOrientation() throws {
        let (s, q) = try makeSolver(maxCoins: 8)
        s.gravity = 0
        let a = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.2, 0), halfExtents: SIMD3(0.01, 0.004, 0.006), mass: 0.006))
        let hull = try XCTUnwrap(s.registerHull(vertices: Self.barHullPoints()))
        let rot = simd_quatf(angle: Self.deg(37), axis: simd_normalize(SIMD3<Float>(1, 2, 0.5)))
        let bar = try XCTUnwrap(spawnBar(s, hull, rot, SIMD3(0.022, 0.206, 0.003)))
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: a))
        let w = try XCTUnwrap(pool.takeWeld())
        s.enableWeld(w, bodyA: a, bodyB: bar, worldAnchor: SIMD3(0.011, 0.203, 0.0015))
        let ref = relPose(s, a, bar)
        let pair = [a, bar]
        s.setRigidVelocity(slots: pair, linear: .zero, angular: 3 * simd_normalize(SIMD3<Float>(0.4, 1, 0.3)),
                           about: s.centerOfMass(of: pair))
        var worst = (mm: Float(0), deg: Float(0)), spun: Float = 0
        step(s, q, frames: 300) { _ in
            let d = self.drift(s, a, bar, from: ref)
            worst = (max(worst.mm, d.mm), max(worst.deg, d.deg))
            spun += simd_length(s.angularVelocity(of: a)!) / 60
        }
        print("ACT_13 spun=\(spun) rad (\(spun / (2 * .pi)) turns) weld drift worst=\(worst.mm) mm / \(worst.deg)°")
        XCTAssertGreaterThan(spun, 2 * .pi, "the pair turned past 2π")
        XCTAssertLessThan(worst.mm, 0.02)
        XCTAssertLessThan(worst.deg, 0.05)
    }

    /// #14 — T1 / VZ-0156: hinge limits are measured from the pose at CREATION. A ±0.2 rad
    /// limited hinge between bodies spawned 0.8 rad apart about its axis leaves them where
    /// they are (it used to snap them 0.60 rad: the twist read the ABSOLUTE relative
    /// orientation), exactly like the pair spawned identically; and the stops then hold
    /// ±0.2 rad about the creation pose when B is spun against A.
    func testHingeLimitIsRelativeToCreation() throws {
        let (s, q) = try makeSolver(maxCoins: 8)
        s.gravity = 0
        let he = SIMD3<Float>(0.01, 0.004, 0.006)
        let placeholder = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.05, 0.3), halfExtents: he, mass: 0.006))
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: placeholder))
        func pair(_ z: Float, _ offset: Float) throws -> (Int, Int, Int) {
            let a = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.2, z), halfExtents: he, mass: 0.006))
            let qb = simd_quatf(angle: offset, axis: SIMD3(0, 0, 1))
            let b = try XCTUnwrap(s.spawnBox(at: SIMD3(0.03, 0.2, z), halfExtents: he, orient: Self.v4(qb), mass: 0.006))
            let j = try pooledHinge(s, pool, a, b, SIMD3(0.015, 0.2, z), SIMD3(0, 0, 1))
            s.setJointLimits(j, -0.2...0.2)
            return (a, b, j)
        }
        let (a1, b1, j1) = try pair(-0.1, 0)
        let (a2, b2, j2) = try pair(0.1, 0.8)
        let r1 = relPose(s, a1, b1), r2 = relPose(s, a2, b2)
        let t2before = s.hingeTwist(j2)!
        step(s, q, frames: 30)
        let snap1 = Self.deg(drift(s, a1, b1, from: r1).deg), snap2 = Self.deg(drift(s, a2, b2, from: r2).deg)
        // Spin B against A about the axis both ways: the stops hold ±0.2 rad from creation.
        var twMax: Float = -.greatestFiniteMagnitude, twMin: Float = .greatestFiniteMagnitude
        for w in [Float(3), -3] {
            s.setAngularVelocity(ofSlot: b2, to: SIMD3(0, 0, w))
            step(s, q, frames: 30) { _ in
                let t = s.hingeTwist(j2)!
                twMax = max(twMax, t); twMin = min(twMin, t)
            }
        }
        print("ACT_14 identity pair: twist=\(s.hingeTwist(j1)!) snap=\(snap1) rad; 0.8-rad pair: twist \(t2before)→\(s.hingeTwist(j2)!) snap=\(snap2) rad; spun ±3 rad/s: twist range [\(twMin), \(twMax)] rad (stops ±0.2)")
        XCTAssertLessThan(snap1, 0.01, "identity-spawned bodies: the limit is inert at creation")
        XCTAssertEqual(t2before, 0, accuracy: 1e-5, "the twist is measured from creation")
        // VZ-0156 flipped this (was XCTAssertGreaterThan(snap2, 0.1): "the limit snaps them").
        XCTAssertLessThan(snap2, 0.01, "bodies 0.8 rad apart: the limit is inert at creation too")
        XCTAssertEqual(twMax, 0.2, accuracy: 0.02, "upper stop holds +0.2 rad from creation")
        XCTAssertEqual(twMin, -0.2, accuracy: 0.02, "lower stop holds −0.2 rad from creation")
    }

    /// #15 — toggling pooled slots (weld, ball) never wakes a sleeping bystander, and
    /// never grows the joint table; `addBallJoint` does wake it (T3, control).
    func testPooledEnableDisableDoesNotWakeAll() throws {
        let (s, q) = try makeSolver(maxCoins: 8)
        let he = SIMD3<Float>(0.005, 0.005, 0.005)
        let x = try XCTUnwrap(s.spawnBox(at: SIMD3(-0.02, 0.005, 0), halfExtents: he, mass: 0.005))
        let y = try XCTUnwrap(s.spawnBox(at: SIMD3(-0.0095, 0.005, 0), halfExtents: he, mass: 0.005))
        let z = try XCTUnwrap(s.spawnBox(at: SIMD3(0.2, 0.005, 0), halfExtents: he, mass: 0.005))
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 4, placeholderBody: x))
        let table = s.jointCount
        XCTAssertEqual(s.activeJointCount, 0, "reserved slots start disabled")
        step(s, q, frames: 40)
        XCTAssertEqual(s.asleepCount, 3, "precondition: everything sleeps")

        let w = try XCTUnwrap(pool.takeWeld())
        s.enableWeld(w, bodyA: x, bodyB: y, worldAnchor: SIMD3(-0.01475, 0.005, 0))
        XCTAssertTrue(s.isAsleep(z), "enableWeld leaves the bystander asleep")
        XCTAssertFalse(s.isAsleep(x)); XCTAssertFalse(s.isAsleep(y))
        step(s, q, frames: 3)
        XCTAssertTrue(s.isAsleep(z))
        XCTAssertEqual(s.activeJointCount, 1, "a native weld is one slot (two hinges before VZ-0150)")
        pool.give(w)
        XCTAssertTrue(s.isAsleep(z), "disable leaves the bystander asleep")
        let b = try XCTUnwrap(pool.takeSlot())
        s.enableBall(slot: b, bodyA: x, bodyB: nil, worldAnchor: SIMD3(-0.02, 0.005, 0))
        step(s, q, frames: 3)
        s.disableJoint(slot: b)
        XCTAssertTrue(s.isAsleep(z), "ball toggle leaves the bystander asleep")
        XCTAssertEqual(s.jointCount, table, "pooled toggles never grow the joint table")
        XCTAssertEqual(s.activeJointCount, 0)

        _ = s.addBallJoint(bodyA: x, bodyB: nil, worldAnchor: SIMD3(-0.02, 0.005, 0))
        print("ACT_15 bystander asleep after pooled toggles=true; after addBallJoint=\(s.isAsleep(z)) (T3 wakeAll) table=\(table)")
        XCTAssertFalse(s.isAsleep(z), "control: add*Joint wakes everything (T3)")
    }

    /// #15b — contact generation honours a POOLED slot's collideConnected bit (the grip /
    /// carry welds rely on it, §B4/§C2): two overlapping bodies emit no contact while a
    /// pooled weld with collideConnected false joins them, do once it is disabled, and do
    /// with collideConnected true.
    func testPooledWeldCollideConnectedGatesContacts() throws {
        let (s, _) = try makeSolver(maxCoins: 8)
        s.gravity = 0
        s.setColliders([])
        let he = SIMD3<Float>(0.005, 0.005, 0.005)
        let a = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.1, 0), halfExtents: he, mass: 0.005))
        let b = try XCTUnwrap(s.spawnBox(at: SIMD3(0.009, 0.1, 0), halfExtents: he, mass: 0.005))   // 1 mm overlap
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: a))
        func pairContacts() -> Int {
            s.generateContactsNow()
            return s.contacts(touching: [a]).filter { Int($0.meta.y) == b || Int($0.meta.x) == b }.count
        }
        let free = pairContacts()
        let w = try XCTUnwrap(pool.takeWeld())
        s.enableWeld(w, bodyA: a, bodyB: b, worldAnchor: SIMD3(0.0045, 0.1, 0))
        let welded = pairContacts()
        s.disableWeld(w)
        let released = pairContacts()
        s.enableWeld(w, bodyA: a, bodyB: b, worldAnchor: SIMD3(0.0045, 0.1, 0), collideConnected: true)
        let colliding = pairContacts()
        print("ACT_15b a–b contacts: free=\(free) welded(cc=false)=\(welded) released=\(released) welded(cc=true)=\(colliding)")
        XCTAssertGreaterThan(free, 0, "precondition: the overlapping pair contacts")
        XCTAssertEqual(welded, 0, "a pooled weld with collideConnected false suppresses the pair's contacts")
        XCTAssertGreaterThan(released, 0, "a disabled pooled slot no longer suppresses them")
        XCTAssertGreaterThan(colliding, 0, "collideConnected true keeps them")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 16: perf fence over a clock-like world
    // ══════════════════════════════════════════════════════════════════════════

    struct FenceResult {
        var label = ""
        var p50 = 0.0, p95 = 0.0, max = 0.0, mean = 0.0
        var listOverflow = 0, uncolored = 0, maxColorUsed = 0, beyondSweep = 0
        var bodies = 0, colliders = 0, jointSlots = 0, activeJoints = 0
        var meanAwake = 0.0, meanContacts = 0.0, meanUploaded = 0.0, latchedAsleepBefore = 0
        var hops = 0, realtime = 0.0, idleSleepSeconds: Float = -1, idleP50 = 0.0
        var stockedAsleepBefore = 0, stockedMaxTiltDeg: Float = 0
        var craneAwakeIdle = 0, craneMaxSpeedIdle: Float = 0
        var smallWorldFrames = 0
    }

    /// What the fence world contains / does (the default is the §F.3 #16 world with the
    /// §A3 zone culling the scene will run). The switches exist for cost attribution.
    struct FenceOptions {
        var cull = true
        var crane = true
        var hoppers = true
        var latches = true
        var stock = true
        var latchedBars = true
        var operate = true
        var activityFrames = 300
        var idleFrames = 300
        /// Stage B1 opt-ins (the recommended §G setting is both on).
        var manifoldSolve = false
        var warmStart = false
        /// Stage B2 opt-ins: the speculative colouring (VZ-0160), a torsion patch on the
        /// workers (plan 5a; 0 = off), Coulomb sheave friction on the crane's falls (VZ-0168;
        /// 0 = frictionless rods).
        var coloringScheme: CoinDEMSolver.ColoringScheme = .jonesPlassmann
        var workerPatchRadius: Float = 0
        var fallFrictionArm: Float = 0
        /// Stage B3: the small-world path (plan 3c — the whole frame as one threadgroup dispatch).
        /// Set explicitly either way, so VIZ_COINDEM_SMALLWORLD never changes what a config measures.
        var smallWorld = false
        /// Stage B3: the prepared contact rows (CoinDEMSolver.preparedContactSolve — the same rows,
        /// their pose-constant factors computed once per substep). Set explicitly either way, so
        /// VIZ_COINDEM_PREPARED never changes what a config measures.
        var preparedRows = false
        /// Joint Gauss–Seidel passes per velocity iteration (`CoinDEMSolver.jointInnerPasses`, the
        /// solver's default 1). The Digital Clock scene runs 6 (ClockRigidWorld.configure; VZ-0208).
        var jointInnerPasses = 1
    }

    /// The §A2 static set (134 colliders), tagged: `pads` are touched by design (seated
    /// bars rest on them), `land` must never be touched by a seated bar. `zones[i]` holds
    /// collider i's §A3 zone tags: 0 = global (always uploaded), 1–4 = display digit k
    /// (land + pad plane), 5 = frame / colon lenses, 6 = L cradle, 7 = R cradle, 8–10 =
    /// rack slots 0–3 / 4–7 / 8–11 (a ridge or the rack base/plate can carry several).
    struct ClockStatics {
        var all: [CoinStaticCollider] = []
        var land: [CoinStaticCollider] = []
        var zones: [[Int]] = []
        static let zoneCount = 11
    }

    static let tableTop: Float = 0.62
    static let digitX: [Float] = [-0.1305, -0.0875, -0.0325, 0.0105]
    static let rowD: Float = 0.6485
    static let seatZ: Float = 0.18375

    /// Segment sockets a…g as (dx, row offset from d, vertical) in mm.
    static let sockets: [(Float, Float, Bool)] = [(0, 56, false), (14, 42, true), (14, 14, true),
                                                   (0, 0, false), (-14, 14, true), (-14, 42, true), (0, 28, false)]

    static func clockStatics() -> ClockStatics {
        var out = ClockStatics()
        let mu: Float = 0.4
        var zone = [0]
        func add(_ c: CoinStaticCollider) { out.all.append(c); out.zones.append(zone) }
        func box(_ x0: Float, _ x1: Float, _ y0: Float, _ y1: Float, _ z0: Float, _ z1: Float,
                 mu m: Float = 0.4) -> CoinStaticCollider {
            .box(center: SIMD3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                 halfExtents: SIMD3((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2), friction: m, restitution: 0.1)
        }
        func obb(_ c: SIMD3<Float>, _ he: SIMD3<Float>, _ q: simd_quatf, mu m: Float = 0.4) -> CoinStaticCollider {
            .orientedBox(center: c, halfExtents: he, orientation: q, friction: m, restitution: 0.1)
        }
        let rotZ45 = simd_quatf(angle: .pi / 4, axis: SIMD3(0, 0, 1))
        let rotZm45 = simd_quatf(angle: -.pi / 4, axis: SIMD3(0, 0, 1))
        // Global.
        add(.plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.1))
        add(.box(center: SIMD3(0, 0.609, 0.2), halfExtents: SIMD3(0.28, 0.011, 0.21), friction: 0.5, restitution: 0.1))
        // Housing (16).
        add(box(-0.165, 0.045, 0.62, 0.7395, 0.100, 0.170))
        let rotY45 = simd_quatf(angle: .pi / 4, axis: SIMD3(0, 1, 0))
        for (x, z) in [(Float(-0.165), Float(0.100)), (0.045, 0.100), (-0.165, 0.1835), (0.045, 0.1835)] {
            add(obb(SIMD3(x, 0.67975, z), SIMD3(0.005, 0.05975, 0.005), rotY45))
        }
        for (z0, z1) in [(Float(0.100), Float(0.128)), (0.128, 0.156), (0.156, 0.1835)] {
            add(box(-0.165, 0.045, 0.7395, 0.7410, z0, z1))
        }
        add(box(-0.10, -0.02, 0.7410, 0.7470, 0.118, 0.142))                 // snooze bar
        add(box(-0.146, -0.134, 0.7410, 0.7440, 0.124, 0.136))               // buttons
        add(box(0.014, 0.026, 0.7410, 0.7440, 0.124, 0.136))
        add(box(-0.165, 0.045, 0.7110, 0.7395, 0.170, 0.1835))              // window frame
        add(box(-0.165, 0.045, 0.62, 0.6420, 0.170, 0.1835))
        add(box(-0.165, -0.1525, 0.6420, 0.7110, 0.170, 0.1835))
        add(box(0.0325, 0.045, 0.6420, 0.7110, 0.170, 0.1835))
        add(box(-0.15, 0.03, 0.62, 0.624, 0.09, 0.100))                      // rear foot
        // Bezel lip (4), 0.8 mm proud.
        add(box(-0.1565, 0.0365, 0.7110, 0.7150, 0.1835, 0.1843))
        add(box(-0.1565, 0.0365, 0.6380, 0.6420, 0.1835, 0.1843))
        add(box(-0.1565, -0.1525, 0.6420, 0.7110, 0.1835, 0.1843))
        add(box(0.0325, 0.0365, 0.6420, 0.7110, 0.1835, 0.1843))
        // Display land (81): 16 per digit + 7 frame/inter-digit + 6 colon/PM + 4 pad planes.
        let z0: Float = 0.176, z1: Float = 0.1835
        let zc = (z0 + z1) / 2, zh = (z1 - z0) / 2
        for xc in digitX {
            func local(_ lx0: Float, _ lx1: Float, _ ly0: Float, _ ly1: Float) -> CoinStaticCollider {
                box(xc + lx0 / 1000, xc + lx1 / 1000, rowD + ly0 / 1000, rowD + ly1 / 1000, z0, z1, mu: mu)
            }
            func web(_ lx: Float, _ ly: Float, _ q: simd_quatf) -> CoinStaticCollider {
                obb(SIMD3(xc + lx / 1000, rowD + ly / 1000, zc), SIMD3(0.0022, 0.000307, zh), q)
            }
            var d: [CoinStaticCollider] = [
                local(-10.1, 10.1, 3.9, 24.1), local(-10.1, 10.1, 31.9, 52.1),      // counters
                local(17.9, 21.0, -3.9, 59.9), local(-21.0, -17.9, -3.9, 59.9),      // outer edges
                local(14.2, 17.9, 56.2, 59.9), local(-17.9, -14.2, 56.2, 59.9),      // outer corners
                local(14.2, 17.9, -3.9, -0.2), local(-17.9, -14.2, -3.9, -0.2),
            ]
            d += [web(11.75, 53.75, rotZ45), web(-11.75, 53.75, rotZm45),            // a–b, a–f
                  web(11.75, 2.25, rotZm45), web(-11.75, 2.25, rotZ45),              // d–c, d–e
                  web(11.75, 30.25, rotZm45), web(11.75, 25.75, rotZ45),             // b–g, g–c
                  web(-11.75, 30.25, rotZ45), web(-11.75, 25.75, rotZm45)]           // f–g, g–e
            out.land += d
            zone = [digitX.firstIndex(of: xc)! + 1]
            d.forEach(add)
            add(box(xc - 0.021, xc + 0.021, 0.6420, 0.7110, 0.170, 0.181))                 // pad plane
        }
        zone = [5]
        let frame: [CoinStaticCollider] = [
            box(-0.1525, 0.0325, 0.7084, 0.7110, z0, z1), box(-0.1525, 0.0325, 0.6420, 0.6446, z0, z1),
            box(-0.1525, -0.1515, 0.6446, 0.7084, z0, z1), box(0.0315, 0.0325, 0.6446, 0.7084, z0, z1),
            box(-0.1095, -0.1085, 0.6446, 0.7084, z0, z1), box(-0.0665, -0.0535, 0.6446, 0.7084, z0, z1),
            box(-0.0115, -0.0105, 0.6446, 0.7084, z0, z1),
        ]
        out.land += frame; frame.forEach(add)
        for i in 0..<6 {
            let c = obb(SIMD3(-0.06, 0.655 + 0.01 * Float(i), zc), SIMD3(0.0015, 0.0015, zh), rotZ45)
            out.land.append(c); add(c)
        }
        zone = [0]
        // Mast (2): ballast base + column (top 0.5 mm under the jib's rail).
        add(box(0.145, 0.195, 0.62, 0.66, 0.18, 0.23))
        add(box(0.162, 0.178, 0.6595, 0.8250, 0.197, 0.213))
        // Cradles: base + V-ridges (square prisms at 45°, valley 0.6275) + back plate.
        func cradle(_ slots: [Float]) {
            let pitch: Float = 0.012
            let ridges = [slots[0] - pitch / 2] + slots.map { $0 + pitch / 2 }
            add(box(ridges.first! - 0.006, ridges.last! + 0.006, 0.62, 0.626, 0.1765, 0.1865))
            for x in ridges { add(obb(SIMD3(x, 0.6275, 0.1815), SIMD3(0.0042426, 0.0042426, 0.005), rotZ45)) }
            add(box(ridges.first! - 0.006, ridges.last! + 0.006, 0.626, 0.66, 0.1765, 0.1805))
        }
        zone = [6]; cradle([-0.2325, -0.2205, -0.2085])
        zone = [7]; cradle([0.081, 0.093, 0.105, 0.117, 0.129])
        // Rack (15): faces +X at x −0.243, slots z 0.235 + 0.01k, valleys along x.
        zone = [8, 9, 10]
        add(box(-0.2535, -0.243, 0.62, 0.626, 0.225, 0.355))
        let rotX45 = simd_quatf(angle: .pi / 4, axis: SIMD3(1, 0, 0))
        for j in 0...12 {
            zone = (j <= 4 ? [8] : []) + (j >= 4 && j <= 8 ? [9] : []) + (j >= 8 ? [10] : [])
            add(obb(SIMD3(-0.24825, 0.6275, 0.230 + 0.01 * Float(j)), SIMD3(0.00525, 0.0035355, 0.0035355), rotX45))
        }
        zone = [8, 9, 10]
        add(box(-0.2535, -0.2490, 0.626, 0.66, 0.225, 0.355))
        return out
    }

    /// Signed distance from `p` to a box / oriented-box collider (negative inside).
    static func colliderDistance(_ c: CoinStaticCollider, _ p: SIMD3<Float>) -> Float {
        let kind = c.a.w.bitPattern
        guard kind == 1 || kind == 3 else { return .greatestFiniteMagnitude }
        let q = simd_quatf(ix: c.orient.x, iy: c.orient.y, iz: c.orient.z, r: c.orient.w)
        let lp = simd_act(q.inverse, p - SIMD3(c.a.x, c.a.y, c.a.z))
        let he = SIMD3(c.b.x, c.b.y, c.b.z)
        let d = abs(lp) - he
        let outside = simd_length(simd_max(d, .zero))
        return outside > 0 ? outside : simd_reduce_max(d)
    }

    /// World-space AABB of a box / oriented-box collider (planes: nil = unbounded).
    static func colliderAABB(_ c: CoinStaticCollider) -> (lo: SIMD3<Float>, hi: SIMD3<Float>)? {
        let kind = c.a.w.bitPattern
        guard kind == 1 || kind == 3 else { return nil }
        let q = simd_quatf(ix: c.orient.x, iy: c.orient.y, iz: c.orient.z, r: c.orient.w)
        let m = simd_matrix3x3(q)
        let he = SIMD3(c.b.x, c.b.y, c.b.z)
        let ext = abs(m.columns.0) * he.x + abs(m.columns.1) * he.y + abs(m.columns.2) * he.z
        let ctr = SIMD3(c.a.x, c.a.y, c.a.z)
        return (ctr - ext, ctr + ext)
    }

    /// §A3 zone culling: the global statics plus every zone whose AABB meets an AWAKE
    /// body's AABB inflated by |v|·(frame time) + speculativeMargin + 2 mm. Exact
    /// because an asleep body is inert for the whole step (M@HEAD:3332, 3409).
    static func culledColliders(_ s: CoinDEMSolver, _ st: ClockStatics, frameTime: Float) -> [CoinStaticCollider] {
        var zoneLo = [SIMD3<Float>](repeating: SIMD3(repeating: .greatestFiniteMagnitude), count: ClockStatics.zoneCount)
        var zoneHi = [SIMD3<Float>](repeating: SIMD3(repeating: -.greatestFiniteMagnitude), count: ClockStatics.zoneCount)
        for (c, zs) in zip(st.all, st.zones) {
            guard let bb = colliderAABB(c) else { continue }
            for z in zs where z > 0 { zoneLo[z] = simd_min(zoneLo[z], bb.lo); zoneHi[z] = simd_max(zoneHi[z], bb.hi) }
        }
        var active = [Bool](repeating: false, count: ClockStatics.zoneCount)
        let bodies = s.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: s.maxCoins)
        for i in 0..<s.highWater where bodies[i].posInvMass.w != 0 && !s.isAsleep(i) {
            let b = bodies[i]
            let x = SIMD3(b.posInvMass.x, b.posInvMass.y, b.posInvMass.z)
            var ext: SIMD3<Float>
            if b.shapeExtents.w > 0.5 && b.shapeExtents.w < 1.5 {                     // box: exact OBB extent
                let m = simd_matrix3x3(simd_quatf(ix: b.orient.x, iy: b.orient.y, iz: b.orient.z, r: b.orient.w))
                let he = SIMD3(b.shapeExtents.x, b.shapeExtents.y, b.shapeExtents.z)
                ext = abs(m.columns.0) * he.x + abs(m.columns.1) * he.y + abs(m.columns.2) * he.z
            } else {
                ext = SIMD3(repeating: b.prevPos.w)                                      // bounding sphere
            }
            ext += SIMD3(repeating: simd_length(SIMD3(b.vel.x, b.vel.y, b.vel.z)) * frameTime + s.speculativeMargin + 0.002)
            for z in 1..<ClockStatics.zoneCount where !active[z] {
                if all(x + ext .>= zoneLo[z]) && all(x - ext .<= zoneHi[z]) { active[z] = true }
            }
        }
        var out: [CoinStaticCollider] = []
        for (c, zs) in zip(st.all, st.zones) where zs.contains(0) || zs.contains(where: { active[$0] }) { out.append(c) }
        return out
    }

    private func runClockFence(dt: Float, iterations: Int, label: String,
                               options o: FenceOptions = FenceOptions()) throws -> FenceResult {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: 48,
                                    coinRadius: 0.27, halfThickness: 0.27,
                                    boundsMin: SIMD3(-0.30, 0.55, 0.00), boundsMax: SIMD3(0.30, 0.90, 0.46))
        else { throw XCTSkip("solver init failed") }
        Self.applyClockConfig(s, dt: dt, iterations: iterations)
        s.manifoldSolve = o.manifoldSolve
        s.warmStart = o.warmStart
        s.coloringScheme = o.coloringScheme
        s.smallWorldPath = o.smallWorld
        s.preparedContactSolve = o.preparedRows
        s.jointInnerPasses = o.jointInnerPasses
        let statics = Self.clockStatics()
        s.setColliders(statics.all)
        var res = FenceResult(label: label)
        res.colliders = statics.all.count
        let wall: Float = 1.0 / 30
        var uploadedSum = 0, uploadedFrames = 0
        /// One 30 fps frame: cull (host, between completed frames) → encode → wait.
        func frame(_ n: Int = 1, _ perFrame: ((Int) -> Void)? = nil) -> [Double] {
            var out: [Double] = []
            for f in 0..<n {
                if o.cull {
                    let up = Self.culledColliders(s, statics, frameTime: wall)
                    s.setColliders(up)
                    uploadedSum += up.count; uploadedFrames += 1
                }
                out += step(s, queue, frames: 1, wallDt: wall)
                perFrame?(f)
            }
            return out
        }

        // ── Bars first (lowest indices: a hull pair is owned by its lower index, so the
        //    long-reach crane boxes spawned LAST are paired from 38 parallel bar threads).
        let hull = try XCTUnwrap(s.registerHull(vertices: Self.barHullPoints()))
        var latched: [Int] = []
        let reading: [[Int]] = [[1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6]]   // "10:08" = 21
        for (dIdx, segs) in reading.enumerated() where o.latchedBars {
            for seg in segs {
                let (dx, dy, vertical) = Self.sockets[seg]
                let c = SIMD3(Self.digitX[dIdx] + dx / 1000, Self.rowD + dy / 1000, Self.seatZ)
                latched.append(try XCTUnwrap(spawnBar(s, hull, vertical ? Self.barVertical : Self.barHorizontal, c)))
            }
        }
        if o.latchedBars { XCTAssertEqual(latched.count, 21) }
        let lift: Float = 0.0001
        var stocked: [Int] = []
        if o.stock {
            for x in [Float(-0.2325), -0.2205, 0.081, 0.105] {                   // 2 L + 2 R cradle
                stocked.append(try XCTUnwrap(spawnBar(s, hull, Self.barVertical, SIMD3(x, 0.6405 + lift, Self.seatZ))))
            }
            for k in 0..<12 {                                                      // 12 racked
                stocked.append(try XCTUnwrap(spawnBar(s, hull, Self.barFacingX, SIMD3(-0.24575, 0.6405 + lift, 0.235 + 0.01 * Float(k)))))
            }
        }
        let stockedPose0 = stocked.map { s.orientation(of: $0)! }
        // Seated bars must not touch the land partition (only their pad plane).
        var worstLand = Float.greatestFiniteMagnitude
        for b in latched {
            let x = s.position(of: b)!, qb = s.orientation(of: b)!
            for v in hull.vertices {
                let p = x + simd_act(qb, v)
                for c in statics.land { worstLand = min(worstLand, Self.colliderDistance(c, p)) }
            }
        }
        XCTAssertGreaterThan(worstLand, 0.00025, "seated bars clear the land partition by more than the speculative margin")

        // ── Workers.
        var workers: [Int] = []
        if o.hoppers {
            for x in [Float(-0.205), -0.175, -0.145, -0.115] {
                workers.append(try XCTUnwrap(s.spawnBallastedEgg(Self.worker, fatCenter: SIMD3(x, Self.tableTop + 0.0115, 0.30),
                                                                 friction: 0.6, restitution: 0.15)))
            }
            if o.workerPatchRadius > 0 { for w in workers { s.setPatchRadius(w, o.workerPatchRadius) } }
        }
        // ── Crane (identity-spawned chain: jib, trolley, block, rail, puck) + held bar.
        let xt: Float = 0.10, cable: Float = 0.09
        let blockTop = 0.8175 - cable
        let railY = blockTop - 0.014 - 0.0005 - 0.0025
        var crane: [Int] = []
        var j0 = -1, j1 = -1, j6 = -1, j7 = -1
        var falls: [Int] = []
        var rigPool: CoinJointPool?
        if o.crane {
            // Held bar: front face on the puck face (z 0.1967), COM on the wrist axis.
            let heldBar = try XCTUnwrap(spawnBar(s, hull, Self.barHorizontal, SIMD3(xt, railY, 0.1967 - 0.00275)))
            let jib = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.8315, 0.205), halfExtents: SIMD3(0.26, 0.006, 0.007), mass: 0.03, friction: 0.3, restitution: 0.1))
            let trolley = try XCTUnwrap(s.spawnBox(at: SIMD3(xt, 0.8215, 0.205), halfExtents: SIMD3(0.01, 0.004, 0.008), mass: 0.006, friction: 0.3, restitution: 0.1))
            let block = try XCTUnwrap(s.spawnBox(at: SIMD3(xt, blockTop - 0.007, 0.205), halfExtents: SIMD3(0.011, 0.007, 0.005), mass: 0.012, friction: 0.3, restitution: 0.1))
            let rail = try XCTUnwrap(s.spawnBox(at: SIMD3(xt, railY, 0.210), halfExtents: SIMD3(0.0025, 0.0025, 0.010), mass: 0.002, friction: 0.3, restitution: 0.1))
            let puck = try XCTUnwrap(s.spawnBox(at: SIMD3(xt, railY, 0.1982), halfExtents: SIMD3(0.0025, 0.0025, 0.0015), mass: 0.0015, friction: 0.5, restitution: 0.05))
            crane = [jib, trolley, block, rail, puck, heldBar]
            // Rig joints J0–J7 in chain order (slot order = the serial Gauss–Seidel order).
            // Hinges and prismatics are pooled slots (the stable axis convention); the falls
            // are ordinary distance joints. Pools reserve contiguous slots at creation.
            let rp = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: jib))
            rigPool = rp
            j0 = try pooledHinge(s, rp, jib, nil, SIMD3(0.17, 0.8255, 0.205), SIMD3(0, 1, 0))
            j1 = try pooledPrismatic(s, rp, trolley, jib, SIMD3(xt, 0.8215, 0.205), SIMD3(1, 0, 0))
            for (dx, dz) in [(Float(-0.006), Float(-0.005)), (0.006, -0.005), (-0.006, 0.005), (0.006, 0.005)] {
                falls.append(try XCTUnwrap(s.addDistanceJoint(bodyA: trolley, bodyB: block,
                                                              worldAnchorA: SIMD3(xt + dx, 0.8175, 0.205 + dz),
                                                              worldAnchorB: SIMD3(xt + dx, blockTop, 0.205 + dz))))
            }
            if o.fallFrictionArm > 0 { for f in falls { s.setDistanceSwingFriction(f, arm: o.fallFrictionArm) } }
            // J6 / J7 follow the falls in the GS order: the first two slots of the main pool.
        }
        res.bodies = s.activeCount
        // Main pool: J6 + J7 (crane) + the welds — 74 slots, as the design's two-slot welds
        // needed; a native weld (VZ-0150) takes one, and the ~50 left disabled cost nothing
        // on the GPU (only enabled slots are uploaded), so the table stays 80.
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 74, placeholderBody: latched.first ?? crane[0]))
        if o.crane {
            let (block, rail, puck, heldBar) = (crane[2], crane[3], crane[4], crane[5])
            j6 = try pooledPrismatic(s, pool, rail, block, SIMD3(xt, railY, 0.210), SIMD3(0, 0, -1))
            s.setJointLimits(j6, 0...0.012)
            j7 = try pooledHinge(s, pool, rail, puck, SIMD3(xt, railY, 0.1982), SIMD3(0, 0, 1))
            s.setJointLimits(j7, Self.deg(-5)...Self.deg(95))
            let grip = try XCTUnwrap(pool.takeWeld())
            s.enableWeld(grip, bodyA: puck, bodyB: heldBar, worldAnchor: SIMD3(xt, railY, 0.1967))
        }
        if o.latches {
            for b in latched {
                let w = try XCTUnwrap(pool.takeWeld())
                s.enableWeld(w, bodyA: b, bodyB: nil, worldAnchor: s.position(of: b)!)
            }
        }
        _ = rigPool
        res.jointSlots = s.jointCount
        res.activeJoints = s.activeJointCount

        func brakes() {
            guard o.crane else { return }
            s.setHingeMotor(j0, targetVelocity: 0, maxTorque: 0.01)
            s.setPrismaticMotor(j1, targetVelocity: 0, maxForce: 0.02)
            s.setPrismaticMotor(j6, targetVelocity: 0, maxForce: 0.003)
            s.setHingeMotor(j7, targetVelocity: 0, maxTorque: 2e-4)
        }
        brakes()
        _ = frame(60)                                                              // settle 2 s
        res.latchedAsleepBefore = latched.filter { s.isAsleep($0) }.count
        res.stockedAsleepBefore = stocked.filter { s.isAsleep($0) }.count

        // ── Activity: the crane works (all 5 axes + winch), 4 Weebles hop.
        let props = CoinDEMSolver.ballastedEggProperties(Self.worker)
        let restY = Self.tableTop + Self.worker.fatRadius - props.comBelowFatCenter
        let hc = Self.worker.fatRadius - props.comBelowFatCenter
        let omegaRock = (props.mass * Self.g * props.comBelowFatCenter / (props.mass / props.invInertiaK.x + props.mass * hc * hc)).squareRoot()
        struct Hopper { var slot: Int; var a: SIMD2<Float>; var b: SIMD2<Float>; var toB = true
                        var airborne = false; var grounded: Float = 0 }
        var hoppers = workers.map { w -> Hopper in
            let p = s.position(of: w)!
            return Hopper(slot: w, a: SIMD2(p.x, 0.29), b: SIMD2(p.x, 0.31))
        }
        var states: [CoinBodyState] = []
        var gpu: [Double] = []
        var awakeSum = 0, contactSum = 0, steps = 0
        for f in 0..<o.activityFrames {
            let t = Float(f) * wall
            // Crane operator (host writes happen only here, between completed frames).
            if o.crane && o.operate {
                s.wake(crane)
                s.setHingeMotor(j0, targetVelocity: 0.03 * sin(2 * .pi * t / 6), maxTorque: 0.01)
                s.setPrismaticMotor(j1, targetVelocity: -0.03 * sin(2 * .pi * t / 4), maxForce: 0.02)
                let L = cable - 0.015 * (1 - cos(2 * .pi * t / 5))
                for r in falls { s.setDistanceRestLength(r, L) }
                s.setPrismaticMotor(j6, targetVelocity: 0.012 * sin(2 * .pi * t / 2.5), maxForce: 0.003)
                s.setHingeMotor(j7, targetVelocity: 1.2 * sin(2 * .pi * t / 3), maxTorque: 2e-4)
            }
            // Hop controller.
            s.readStates(hoppers.map(\.slot), into: &states)
            for i in hoppers.indices {
                let st = states[i]
                let grounded = st.position.y < restY + 0.0015 && st.velocity.y > -0.05
                if hoppers[i].airborne {
                    if grounded {
                        hoppers[i].airborne = false; hoppers[i].grounded = 0
                        s.setAngularVelocity(ofSlot: hoppers[i].slot, to: SIMD3(st.angularVelocity.x, 0, st.angularVelocity.z))
                    }
                    continue
                }
                hoppers[i].grounded += wall
                let tilt = Self.deg(Self.tiltDeg(st.orientation))
                let rock = tilt + simd_length(st.angularVelocity) / omegaRock
                guard hoppers[i].grounded > 0.25, rock < Self.deg(6) || hoppers[i].grounded > 1.5 else { continue }
                let target = hoppers[i].toB ? hoppers[i].b : hoppers[i].a
                let delta = target - SIMD2(st.position.x, st.position.z)
                if simd_length(delta) < 0.002 { hoppers[i].toB.toggle(); continue }
                let vy = (2 * Self.g * 0.012).squareRoot(), T = 2 * vy / Self.g
                var vh = delta / T
                if simd_length(vh) > 0.1 { vh *= 0.1 / simd_length(vh) }
                s.setRigidVelocity(slots: [hoppers[i].slot], linear: SIMD3(vh.x, vy, vh.y),
                                   angular: SIMD3(0, 2, 0), about: st.position)
                hoppers[i].airborne = true
                res.hops += 1
            }
            gpu += frame()
            if s.lastFrameUsedSmallWorld { res.smallWorldFrames += 1 }
            steps += s.lastStepCount
            awakeSum += s.activeCount - s.asleepCount
            contactSum += s.contactCount
        }
        let warm = Array(gpu.dropFirst(15)).sorted()
        res.p50 = warm[warm.count / 2]
        res.p95 = warm[Int(Double(warm.count - 1) * 0.95)]
        res.max = warm.last!
        res.mean = warm.reduce(0, +) / Double(warm.count)
        let cs = s.colorStats
        res.listOverflow = cs.listOverflow; res.uncolored = cs.uncolored
        res.maxColorUsed = cs.maxColorUsed; res.beyondSweep = cs.beyondSweep
        res.meanAwake = Double(awakeSum) / Double(o.activityFrames)
        res.meanContacts = Double(contactSum) / Double(o.activityFrames)
        res.meanUploaded = o.cull ? Double(uploadedSum) / Double(max(uploadedFrames, 1)) : Double(statics.all.count)
        res.realtime = Double(Float(steps) * dt) / Double(Float(o.activityFrames) * wall)

        // ── Idle: park (brakes, no hops) and time how long until the world sleeps.
        brakes()
        var slept: Float = -1
        let idleGPU = frame(o.idleFrames) { f in
            if slept < 0, s.didSkipLastFrame { slept = Float(f) * wall }
        }
        res.idleSleepSeconds = slept
        res.idleP50 = idleGPU.suffix(60).sorted()[30]
        res.craneAwakeIdle = crane.filter { !s.isAsleep($0) }.count
        res.craneMaxSpeedIdle = crane.map { simd_length(s.velocity(of: $0)!) }.max() ?? 0
        res.stockedMaxTiltDeg = zip(stocked, stockedPose0).map { Self.angleDeg(s.orientation(of: $0.0)! * $0.1.inverse) }.max() ?? 0
        print("ACT_16 [\(label)] bodies=\(res.bodies) colliders=\(res.colliders) uploaded(mean)=\(String(format: "%.1f", res.meanUploaded)) jointSlots=\(res.jointSlots) activeJoints=\(res.activeJoints) latchedAsleepBeforeActivity=\(res.latchedAsleepBefore)/\(latched.count) stockedAsleepBefore=\(res.stockedAsleepBefore)/\(stocked.count) stockedMaxRotation=\(res.stockedMaxTiltDeg)° landClearance=\(worstLand * 1000) mm")
        print("ACT_16 [\(label)] physics GPU ms p50=\(String(format: "%.3f", res.p50)) p95=\(String(format: "%.3f", res.p95)) max=\(String(format: "%.3f", res.max)) mean=\(String(format: "%.3f", res.mean)) | colorStats overflow=\(res.listOverflow) uncolored=\(res.uncolored) maxColorUsed=\(res.maxColorUsed) beyondSweep=\(res.beyondSweep) | meanAwake=\(res.meanAwake) meanContacts=\(res.meanContacts) hops=\(res.hops) realtime=\(res.realtime) smallWorldFrames=\(res.smallWorldFrames)/\(o.activityFrames) | idle: world asleep after \(res.idleSleepSeconds) s, idle p50=\(String(format: "%.3f", res.idleP50)) ms, crane awake \(res.craneAwakeIdle)/\(crane.count) max |v| \(res.craneMaxSpeedIdle * 1000) mm/s")
        return res
    }

    /// #16 — the go/no-go perf fence: 26 bar hulls (21 latched + asleep, 1 on the crane
    /// grip, 4 cradled) over the 81-box display land, 12 racked bars, 4 hopping Weebles,
    /// the 5-body crane chain with 4 falls, 30 fps wall dt. Runs both candidate
    /// configurations and asserts the fence on the default (1/180 s × 6); then the stage-B1
    /// opt-ins, the B1 + speculative colouring (VZ-0160), every B2 opt-in the clock runs
    /// (worker torsion patch, crane sheave friction — the braked crane must sleep, VZ-0168),
    /// and — stage B3 — the SCENE configuration: those opt-ins plus the small-world path (plan
    /// 3c) and the prepared contact rows, gated the same way (p95 ≤ 3 ms, idle = the whole-world
    /// skip, 0 uncoloured, the latched bars asleep before the job); and, measured but not gated,
    /// the same at the scene's own jointInnerPasses = 6 (`g` — the pass count is decision VZ-0208).
    ///
    /// Skipped unless VIZ_COINDEM_FENCE=1: the 3 ms budget is still a TARGET, not a regression
    /// gate — timing fences flake under other sessions' GPU load, and a red fence blocked every
    /// other session's engine landing (bump-engine runs the full suite). The DEFAULT configuration's
    /// assertions (`a`, the multi-dispatch path without the opt-ins) still fail by design.
    ///
    /// Stage B3, the finished speed round (docs/coindem-constraint-solver.md), M1 Max, 3 rounds
    /// interleaved with the stage-A engine (8038c45), other sessions sharing the GPU (load average
    /// 3.5–12.2; median [range], ms): the SCENE configuration (`f`) p50 1.73 [1.72–1.74] / p95 2.72
    /// [2.71–2.72] / max 2.78 [2.77–2.78] — within the fence in 3 of 3 rounds — every activity frame
    /// on the small-world path, 0 uncoloured, 21/21 latched bars asleep before the job, 16/16 stocked
    /// bars within 0.013°, the world asleep 1.33 s into the idle (idle p50 0.000 ms, crane 0/6 awake).
    /// (Before the speed round, without the prepared rows: p50 3.41 / p95 5.99.) The DEFAULT
    /// configurations stay far above it: 1/180 s × 6 — stage A p50 11.70 [11.68–11.85] / p95 13.47
    /// [13.47–13.60] / max 14.83, B3 p50 8.83 [8.02–9.14] / p95 12.78 [10.24–13.13] / max 14.24 (the
    /// multi-dispatch path swings with the GPU's other load: a quieter set gave p50 6.43 / p95 9.21);
    /// 1/240 s × 8 — A 16.17 / 18.86, B3 8.36 / 12.47. Where the scene configuration's time goes (µs
    /// per substep, each phase timed as its own dispatch; no contact / single contacts / a polytope
    /// manifold touching): joint solve 99 / 99 / 102, contact solve 13 / 47 / 138 (VZ-0210), polytope
    /// narrowphase 53 / 53 / 106 (VZ-0211), contact generation 72–75, joint prepare 47, colouring
    /// 20–29, warm start 12–28, the prepare 11–21, ≈ 11–15 for each near-empty phase. `g`, the scene's
    /// jointInnerPasses = 6: the joint solve +62 µs per extra pass per substep, ≈ +1.85 ms per frame —
    /// p50 3.64 [3.56–4.34] / p95 4.92 [4.54–6.19] ms over 7 runs, every one with other load on the GPU
    /// (in one process against 1 pass: p50 1.93 → 3.65) — over the budget; decision VZ-0208. In the
    /// same three later sets (another session's app rendering, then the machine in interactive use;
    /// load 6.6–15.5) `f` measured p50 1.73–3.19 / p95 3.23–5.10 over 7 runs, over the fence in each:
    /// it holds on a GPU the physics has to itself.
    /// Turn the gate on unconditionally once it holds with a margin.
    ///   Run:  VIZ_COINDEM_FENCE=1 ./Scripts/test.sh --filter testClockLikeWorldPerfFence
    func testClockLikeWorldPerfFence() throws {
        guard ProcessInfo.processInfo.environment["VIZ_COINDEM_FENCE"] != nil else {
            throw XCTSkip("perf fence is a work-in-progress target; set VIZ_COINDEM_FENCE=1 to run it")
        }
        let a = try runClockFence(dt: 1.0 / 180, iterations: 6, label: "1/180×6")
        let b = try runClockFence(dt: 1.0 / 240, iterations: 8, label: "1/240×8")
        // The recommended stage-B1 setting (manifold solve + warm start), for its cost.
        let c = try runClockFence(dt: 1.0 / 180, iterations: 6, label: "1/180×6 manifold+warm",
                                  options: FenceOptions(manifoldSolve: true, warmStart: true))
        // Stage B2: the same with the speculative colouring, then with every B2 opt-in the clock
        // is meant to run (3 mm torsion patch on the workers, toy-crane sheave friction on the falls).
        let d = try runClockFence(dt: 1.0 / 180, iterations: 6, label: "1/180×6 manifold+warm+speculative",
                                  options: FenceOptions(manifoldSolve: true, warmStart: true, coloringScheme: .speculative))
        let e = try runClockFence(dt: 1.0 / 180, iterations: 6, label: "1/180×6 B2 (manifold+warm+speculative+torsion+sheaves)",
                                  options: FenceOptions(manifoldSolve: true, warmStart: true, coloringScheme: .speculative,
                                                        workerPatchRadius: 0.003,
                                                        fallFrictionArm: CoinDEMSolver.toyCraneFallFrictionArm))
        // Stage B3: the SCENE configuration — every B2 opt-in plus the small-world path (plan 3c) and
        // the prepared contact rows.
        let f = try runClockFence(dt: 1.0 / 180, iterations: 6, label: "1/180×6 scene (B2 opt-ins + small world + prepared rows)",
                                  options: FenceOptions(manifoldSolve: true, warmStart: true, coloringScheme: .speculative,
                                                        workerPatchRadius: 0.003,
                                                        fallFrictionArm: CoinDEMSolver.toyCraneFallFrictionArm,
                                                        smallWorld: true, preparedRows: true))
        // The same at the Digital Clock's jointInnerPasses = 6 (ClockRigidWorld.configure). MEASURED,
        // not gated: how many passes the crane gets against this budget is the decision VZ-0208.
        let g = try runClockFence(dt: 1.0 / 180, iterations: 6, label: "1/180×6 scene + 6 joint passes",
                                  options: FenceOptions(manifoldSolve: true, warmStart: true, coloringScheme: .speculative,
                                                        workerPatchRadius: 0.003,
                                                        fallFrictionArm: CoinDEMSolver.toyCraneFallFrictionArm,
                                                        smallWorld: true, preparedRows: true, jointInnerPasses: 6))
        print("ACT_16 SUMMARY 1/180×6 p50=\(a.p50) p95=\(a.p95) maxColor=\(a.maxColorUsed) | 1/240×8 p50=\(b.p50) p95=\(b.p95) maxColor=\(b.maxColorUsed) | 1/180×6 manifold+warm p50=\(c.p50) p95=\(c.p95) maxColor=\(c.maxColorUsed) stockedMaxRotation=\(c.stockedMaxTiltDeg)° | +speculative p50=\(d.p50) p95=\(d.p95) maxColor=\(d.maxColorUsed) | B2 all p50=\(e.p50) p95=\(e.p95) maxColor=\(e.maxColorUsed) idle asleep after \(e.idleSleepSeconds) s, idle p50=\(e.idleP50), crane awake \(e.craneAwakeIdle)/6 | scene (B2 + small world) p50=\(f.p50) p95=\(f.p95) max=\(f.max) smallWorldFrames=\(f.smallWorldFrames) idle asleep after \(f.idleSleepSeconds) s, idle p50=\(f.idleP50) | scene + 6 joint passes (not gated, VZ-0208) p50=\(g.p50) p95=\(g.p95) max=\(g.max) idle p50=\(g.idleP50)")
        XCTAssertEqual(c.uncolored, 0, "manifold + warm: every contact coloured")
        XCTAssertEqual(c.latchedAsleepBefore, 21, "manifold + warm: latched bars sleep before the job starts")
        XCTAssertEqual(d.uncolored, 0, "speculative colouring: every contact coloured")
        XCTAssertEqual(e.uncolored, 0, "B2 opt-ins: every contact coloured")
        XCTAssertEqual(e.listOverflow, 0)
        XCTAssertEqual(e.craneAwakeIdle, 0, "VZ-0168: with sheave friction the braked crane's hook stops swinging and it sleeps")
        // The scene configuration (stage B3): the same fence on the clock's engine opt-ins (at the
        // fence world's 1 joint pass — `g` measures the scene's 6).
        XCTAssertEqual(f.smallWorldFrames, 300, "scene config: every activity frame took the small-world path")
        XCTAssertLessThanOrEqual(f.p95, 3.0, "scene config fence: physics GPU p95 ≤ 3.0 ms during activity (1/180 s × 6)")
        XCTAssertEqual(f.uncolored, 0, "scene config: every contact coloured")
        XCTAssertEqual(f.listOverflow, 0)
        XCTAssertEqual(f.latchedAsleepBefore, 21, "scene config: latched bars sleep before the job starts")
        XCTAssertGreaterThanOrEqual(f.idleSleepSeconds, 0, "scene config: the parked world reaches the whole-world skip")
        XCTAssertLessThanOrEqual(f.idleP50, 0.01, "scene config: idle frames cost nothing (the whole-world skip)")
        XCTAssertEqual(f.craneAwakeIdle, 0)
        XCTAssertGreaterThan(f.hops, 20, "scene config: the Weebles actually hopped")
        XCTAssertEqual(g.smallWorldFrames, 300, "6 joint passes: measured on the small-world path")
        XCTAssertLessThanOrEqual(a.p95, 3.0, "fence: physics GPU p95 ≤ 3.0 ms during activity (1/180 s × 6)")
        XCTAssertEqual(a.uncolored, 0, "fence: every contact coloured")
        XCTAssertLessThanOrEqual(a.maxColorUsed, 12, "fence: colour sweep stays small")
        XCTAssertEqual(a.listOverflow, 0)
        XCTAssertEqual(b.uncolored, 0, "the 1/240 × 8 fallback colours every contact too")
        XCTAssertEqual(a.latchedAsleepBefore, 21, "latched bars sleep before the job starts")
        XCTAssertGreaterThan(a.hops, 20, "the Weebles actually hopped")
    }
}
