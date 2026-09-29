import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Gate for CoinDEM COMPOUND bodies (shape tag 6; Digital Clock work plan stage B1 item
/// 5c): one rigid body made of up to 128 oriented boxes (these shapes have ≤ 16 — the
/// unclustered path; CoinDEMTrussCompoundTests covers the many-child open truss) — a small
/// jib + counter-jib + cab assembly, a worker's C-shaped hand. Mass, COM and inertia by the parallel-axis
/// rule; every child box collides through the exact polytope narrowphase
/// (CoinDEMNarrowphase.h); sleeping, waking and the static bounding reject treat the union
/// as one body. Measured numbers are PRINTed with a `B1c_` prefix. Same runtime-compiled-
/// library seam and §G toy-scale configuration as CoinDEMContactsTests.
@MainActor
final class CoinDEMCompoundTests: XCTestCase {

    // ── Harness ──────────────────────────────────────────────────────────────

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

    /// §G knobs (1/180 s × 6, μ 0.5, 0.2 mm margin, 0.1 mm slop, warm start off).
    private func makeSolver(maxCoins: Int = 32, maxRadius: Float = 0.03,
                            boundsMin: SIMD3<Float> = SIMD3(-0.4, -0.1, -0.4),
                            boundsMax: SIMD3<Float> = SIMD3(0.4, 0.4, 0.4),
                            sleep: Bool = true, colliders: [CoinStaticCollider]? = nil) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: maxCoins,
                                    coinRadius: maxRadius, halfThickness: maxRadius,
                                    boundsMin: boundsMin, boundsMax: boundsMax)
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = 9.81
        s.fixedDt = 1.0 / 180
        s.maxSubsteps = 10
        s.velocityIterations = 6
        s.colorRounds = 16
        s.islandUnionRounds = 6
        s.warmStart = false
        s.linDamping = 0.99995
        s.angDamping = 0.9998
        s.frictionCoeff = 0.5
        s.rollingResistance = 0.024 / 6
        s.restitution = 0.15
        s.restThreshold = 0.14
        s.contactSlop = 1e-4
        s.baumgarteBeta = 0.2
        s.speculativeMargin = 2e-4
        s.maxSpeed = 3
        s.maxHSpeed = 3
        s.maxOmega = 60
        s.floorY = -0.5
        s.sleepEnabled = sleep
        s.sleepFrames = 20
        s.sleepLinVel = 2e-4
        s.setColliders(colliders ?? [.plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.15)])
        return (s, queue)
    }

    @discardableResult
    private func step(_ s: CoinDEMSolver, _ queue: MTLCommandQueue, frames: Int,
                      wallDt: Float = 1.0 / 60, perFrame: ((Int) -> Void)? = nil) -> [Double] {
        var out: [Double] = []
        for f in 0..<frames {
            guard let cb = queue.makeCommandBuffer() else { break }
            s.encode(to: cb, wallDt: wallDt)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            if cb.status == .error { XCTFail("command buffer error: \(String(describing: cb.error))"); break }
            out.append(max(0, cb.gpuEndTime - cb.gpuStartTime) * 1000)
            perFrame?(f)
        }
        return out
    }

    private static func angleDeg(_ q: simd_quatf) -> Float {
        let n = q.normalized
        return 2 * atan2(simd_length(n.imag), abs(n.real)) * 180 / .pi
    }
    private static func v4(_ q: simd_quatf) -> SIMD4<Float> { SIMD4(q.imag, q.real) }
    private static let identity = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

    private func contactSet(_ s: CoinDEMSolver) -> [String] {
        let p = s.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: s.maxContacts)
        return (0..<s.contactCount).map { i -> String in
            let c = p[i]
            let f: [Float] = [c.nrm.x, c.nrm.y, c.nrm.z, c.nrm.w, c.rA.x, c.rA.y, c.rA.z, c.rB.x, c.rB.y, c.rB.z]
            return "\(c.meta.x) \(c.meta.y) \(c.meta.z) \(c.meta.w) " + f.map { String($0.bitPattern, radix: 16) }.joined(separator: " ")
        }.sorted()
    }

    private func pairContacts(_ s: CoinDEMSolver, _ a: Int, _ b: Int) -> [CoinContact] {
        s.contacts(touching: [a]).filter { c in
            (Int(c.meta.x) == a && Int(c.meta.y) == b) || (Int(c.meta.x) == b && Int(c.meta.y) == a)
        }
    }

    // ── Shapes ───────────────────────────────────────────────────────────────

    /// The toy crane's jib assembly: jib, counter-jib, a denser cab hanging under the
    /// slew point, and a brace tilted 30° riding 0.01 mm above the jib — a non-convex
    /// union with an oriented child.
    static let jib: [CoinCompoundBox] = [
        CoinCompoundBox(center: SIMD3(0.012, 0, 0), halfExtents: SIMD3(0.02, 0.002, 0.002)),
        CoinCompoundBox(center: SIMD3(-0.016, 0, 0), halfExtents: SIMD3(0.008, 0.002, 0.002)),
        CoinCompoundBox(center: SIMD3(-0.002, -0.006, 0), halfExtents: SIMD3(0.004, 0.004, 0.003), density: 2),
        CoinCompoundBox(center: SIMD3(0.004, 0.0052, 0), halfExtents: SIMD3(0.005, 0.0008, 0.0012),
                        orientation: simd_quatf(angle: .pi / 6, axis: SIMD3(0, 0, 1))),
    ]

    /// An L lying flat (both boxes share the bottom plane y = −2 mm): a non-convex
    /// footprint.
    static let ell: [CoinCompoundBox] = [
        CoinCompoundBox(center: SIMD3(0.004, 0, 0), halfExtents: SIMD3(0.012, 0.002, 0.003)),
        CoinCompoundBox(center: SIMD3(-0.005, 0, 0.009), halfExtents: SIMD3(0.003, 0.002, 0.006)),
    ]

    /// A table: a 100 × 80 mm top on four legs (5 children).
    static let table: [CoinCompoundBox] = [
        CoinCompoundBox(center: SIMD3(0, 0.020, 0), halfExtents: SIMD3(0.05, 0.002, 0.04)),
        CoinCompoundBox(center: SIMD3(-0.046, 0.009, -0.036), halfExtents: SIMD3(0.003, 0.009, 0.003)),
        CoinCompoundBox(center: SIMD3( 0.046, 0.009, -0.036), halfExtents: SIMD3(0.003, 0.009, 0.003)),
        CoinCompoundBox(center: SIMD3(-0.046, 0.009,  0.036), halfExtents: SIMD3(0.003, 0.009, 0.003)),
        CoinCompoundBox(center: SIMD3( 0.046, 0.009,  0.036), halfExtents: SIMD3(0.003, 0.009, 0.003)),
    ]

    /// Independent reference: mass fractions, COM and the per-unit-mass inertia tensor of
    /// a union of solid boxes about its COM (parallel-axis rule), in Double.
    private static func referenceMassProperties(_ boxes: [CoinCompoundBox]) -> (fractions: [Double], com: SIMD3<Double>, I: simd_double3x3) {
        let m = boxes.map { Double($0.density) * 8 * Double($0.halfExtents.x) * Double($0.halfExtents.y) * Double($0.halfExtents.z) }
        let M = m.reduce(0, +)
        var com = SIMD3<Double>.zero
        for (b, mi) in zip(boxes, m) { com += SIMD3<Double>(Double(b.center.x), Double(b.center.y), Double(b.center.z)) * mi }
        com /= M
        var I = simd_double3x3(0)
        for (b, mi) in zip(boxes, m) {
            let a = Double(b.halfExtents.x), bb = Double(b.halfExtents.y), c = Double(b.halfExtents.z)
            // Solid box about its own centre, in its own axes: m/3 · (b² + c², a² + c², a² + b²).
            let own = simd_double3x3(diagonal: SIMD3(bb * bb + c * c, a * a + c * c, a * a + bb * bb) * (mi / 3))
            let q = b.orientation
            let R = simd_double3x3(simd_quatd(ix: Double(q.imag.x), iy: Double(q.imag.y), iz: Double(q.imag.z), r: Double(q.real)).normalized)
            let d = SIMD3<Double>(Double(b.center.x), Double(b.center.y), Double(b.center.z)) - com
            // Parallel axis: + m (|d|² 1 − d dᵀ).
            let steiner = (simd_double3x3(diagonal: SIMD3(repeating: simd_dot(d, d))) - simd_double3x3(columns: (d * d.x, d * d.y, d * d.z))) * mi
            I += R * own * R.transpose + steiner
        }
        return (m.map { $0 / M }, com, I * (1 / M))
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Mass properties
    // ══════════════════════════════════════════════════════════════════════════

    /// Mass fractions ∝ density × volume, COM, and the inertia tensor about the COM by the
    /// parallel-axis rule, against an independent reference — for a T, the jib (a rotated
    /// child, a denser cab), a single off-centre box (no Steiner term), and a 90°-rotated
    /// child (identical to its axis-swapped twin). The diagonal the GPU keeps (invInertiaK)
    /// in the principal frame reproduces the full tensor: R · diag(1/k) · Rᵀ = I.
    func testCompoundMassPropertiesFollowTheParallelAxisRule() throws {
        let tee: [CoinCompoundBox] = [
            CoinCompoundBox(center: SIMD3(0, 0, 0), halfExtents: SIMD3(0.002, 0.01, 0.002)),
            CoinCompoundBox(center: SIMD3(0, 0.012, 0), halfExtents: SIMD3(0.01, 0.002, 0.002)),
        ]
        let single = [CoinCompoundBox(center: SIMD3(0.003, -0.002, 0.001), halfExtents: SIMD3(0.004, 0.002, 0.001))]
        let rotated = [CoinCompoundBox(center: .zero, halfExtents: SIMD3(0.01, 0.002, 0.003),
                                       orientation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1)))]
        let swapped = [CoinCompoundBox(center: .zero, halfExtents: SIMD3(0.002, 0.01, 0.003))]
        let dense: [CoinCompoundBox] = [
            CoinCompoundBox(center: SIMD3(-0.002, 0, 0), halfExtents: SIMD3(0.002, 0.002, 0.002), density: 1),
            CoinCompoundBox(center: SIMD3( 0.002, 0, 0), halfExtents: SIMD3(0.002, 0.002, 0.002), density: 3),
        ]
        for (name, boxes) in [("T", tee), ("jib", Self.jib), ("single", single), ("rotated", rotated), ("dense", dense)] {
            let p = try XCTUnwrap(CoinCompoundMath.prepare(boxes), name)
            let ref = Self.referenceMassProperties(boxes)
            let L = 0.03, L2 = L * L
            for (a, b) in zip(p.massFractions, ref.fractions) { XCTAssertEqual(Double(a), b, accuracy: 1e-6, "\(name) mass fraction") }
            XCTAssertEqual(Double(p.comOffset.x), ref.com.x, accuracy: 1e-8, name)
            XCTAssertEqual(Double(p.comOffset.y), ref.com.y, accuracy: 1e-8, name)
            XCTAssertEqual(Double(p.comOffset.z), ref.com.z, accuracy: 1e-8, name)
            for c in 0..<3 { for r in 0..<3 {
                XCTAssertEqual(p.inertiaPerMassDesign[c][r], ref.I[c][r], accuracy: 1e-9 * L2, "\(name) I[\(c)][\(r)]")
            } }
            // The GPU's diagonal in the principal frame reproduces the full tensor.
            let q = p.principalRotation
            let R = simd_double3x3(simd_quatd(ix: Double(q.imag.x), iy: Double(q.imag.y), iz: Double(q.imag.z), r: Double(q.real)))
            let k = p.invInertiaK
            let D = simd_double3x3(diagonal: SIMD3(1 / Double(k.x), 1 / Double(k.y), 1 / Double(k.z)))
            let back = R * D * R.transpose
            for c in 0..<3 { for r in 0..<3 {
                XCTAssertEqual(back[c][r], ref.I[c][r], accuracy: 1e-6 * max(ref.I[0][0], ref.I[1][1], ref.I[2][2]), "\(name) R·diag·Rᵀ[\(c)][\(r)]")
            } }
            // The principal-frame children map back onto the design boxes exactly.
            for (b, ch) in zip(boxes, p.children) {
                let designC = simd_act(q, ch.center) + p.comOffset
                XCTAssertLessThan(simd_length(designC - b.center), 1e-8, "\(name) child centre round-trips")
                XCTAssertLessThan(Self.angleDeg((q * ch.orientation) * b.orientation.inverse), 1e-3, "\(name) child orientation round-trips")
            }
            print("B1c_mass \(name): fractions \(p.massFractions), COM \(p.comOffset * 1000) mm, principal I/m \(SIMD3(1 / k.x, 1 / k.y, 1 / k.z) * 1e6) mm², bound \(p.boundingRadius * 1000) mm")
        }
        // Hand values: the T's stem and bar weigh the same, COM halfway (y 6 mm); the
        // denser twin carries ¾ of the mass, COM 1 mm towards it; the single box keeps
        // its own inertia (no parallel-axis term); rotation by 90° = swapping the extents.
        let t = try XCTUnwrap(CoinCompoundMath.prepare(tee))
        XCTAssertEqual(t.comOffset.y, 0.006, accuracy: 1e-8)
        let Ixx = 0.5 * ((0.01 * 0.01 + 0.002 * 0.002) / 3 + (0.002 * 0.002 + 0.002 * 0.002) / 3) + 0.006 * 0.006
        XCTAssertEqual(t.inertiaPerMassDesign[0][0], Ixx, accuracy: 1e-12)
        let d = try XCTUnwrap(CoinCompoundMath.prepare(dense))
        XCTAssertEqual(d.massFractions[1], 0.75, accuracy: 1e-6)
        XCTAssertEqual(d.comOffset.x, 0.001, accuracy: 1e-8)
        let one = try XCTUnwrap(CoinCompoundMath.prepare(single))
        XCTAssertEqual(one.inertiaPerMassDesign[1][1], (0.004 * 0.004 + 0.001 * 0.001) / 3, accuracy: 1e-12)
        // (The 90° quaternion is a Float: cos 45° to ~6e-8, so the rotated tensor matches its
        // twin to ~1e-7 of its largest moment, not to Double round-off.)
        let r1 = try XCTUnwrap(CoinCompoundMath.prepare(rotated)), r2 = try XCTUnwrap(CoinCompoundMath.prepare(swapped))
        let scale = max(r2.inertiaPerMassDesign[0][0], r2.inertiaPerMassDesign[1][1], r2.inertiaPerMassDesign[2][2])
        for c in 0..<3 { for r in 0..<3 { XCTAssertEqual(r1.inertiaPerMassDesign[c][r], r2.inertiaPerMassDesign[c][r], accuracy: 1e-6 * scale) } }
        // Degenerate input is refused, and so is a 129th child (CoinDEMTrussCompoundTests covers
        // 17…128).
        XCTAssertNil(CoinCompoundMath.prepare([]))
        XCTAssertNil(CoinCompoundMath.prepare([CoinCompoundBox(center: .zero, halfExtents: SIMD3(0.01, 0, 0.01))]))
        XCTAssertNil(CoinCompoundMath.prepare(Array(repeating: single[0], count: CoinCompoundMath.maxChildren + 1)))
    }

    // ══════════════════════════════════════════════════════════════════════════
    // The collider is the drawn geometry
    // ══════════════════════════════════════════════════════════════════════════

    /// The jib, spawned at a tilted design pose: the collider the kernels see (the GPU shape
    /// table + the body pose, read back) is the drawn geometry — every child box at its
    /// drawn centre, extents and orientation — and, face by face, a 0.5 mm probe sphere
    /// held 0.1 mm off each exposed face (at its centre, or — where another child comes
    /// within the margin of that probe, e.g. the brace's lower face over the jib — at the
    /// first of four off-centre points on the face that is clear) gets exactly one contact:
    /// that face's normal, 0.1 mm apart, at the face. A probe tucked in the notch under the
    /// jib beside the cab — inside the union's bounding box and convex hull, 1 mm from
    /// every face — gets none: the collider is the boxes, not a hull or a bound.
    func testCompoundColliderIsTheDrawnGeometryFaceByFace() throws {
        let (s, _) = try makeSolver(colliders: [])
        s.gravity = 0
        let h = try XCTUnwrap(s.registerCompound(boxes: Self.jib))
        let rd = simd_quatf(angle: 0.4, axis: SIMD3(0, 1, 0)) * simd_quatf(angle: 0.2, axis: SIMD3(1, 0, 0))
        let td = SIMD3<Float>(0.01, 0.12, -0.02)
        let body = try XCTUnwrap(s.spawnCompound(design: rd, td, compound: h, mass: 0.004))
        struct Drawn { var c: SIMD3<Float>; var q: simd_quatf; var he: SIMD3<Float> }
        let drawn = Self.jib.map { Drawn(c: td + simd_act(rd, $0.center), q: (rd * $0.orientation).normalized, he: $0.halfExtents) }

        // 1. The read-back collider = the drawn boxes.
        let col = try XCTUnwrap(s.compoundColliderBoxes(of: body))
        XCTAssertEqual(col.count, drawn.count)
        var worstC: Float = 0, worstQ: Float = 0
        for (c, d) in zip(col, drawn) {
            worstC = max(worstC, simd_length(c.center - d.c))
            worstQ = max(worstQ, Self.angleDeg(c.orientation * d.q.inverse))
            XCTAssertEqual(c.halfExtents, d.he, "extents exactly as drawn")
        }
        XCTAssertLessThan(worstC, 1e-6, "every child centre where it is drawn")
        XCTAssertLessThan(worstQ, 1e-3, "every child orientation as drawn")

        // 2. Face by face.
        func inside(_ p: SIMD3<Float>, _ d: Drawn, pad: Float) -> Bool {
            let l = simd_act(d.q.inverse, p - d.c)
            return abs(l.x) <= d.he.x + pad && abs(l.y) <= d.he.y + pad && abs(l.z) <= d.he.z + pad
        }
        func distance(_ p: SIMD3<Float>, _ d: Drawn) -> Float {
            let l = simd_act(d.q.inverse, p - d.c)
            return simd_length(simd_max(abs(l) - d.he, .zero))
        }
        let r: Float = 0.0005, gap: Float = 0.0001
        var probed = 0, offCentre = 0, worstDepth: Float = 0, worstN: Float = 0, worstP: Float = 0
        for (k, d) in drawn.enumerated() {
            for axis in 0..<3 { for sgn: Float in [-1, 1] {
                var e = SIMD3<Float>.zero; e[axis] = sgn
                let n = simd_act(d.q, e)
                let a1 = (axis + 1) % 3, a2 = (axis + 2) % 3
                var e1 = SIMD3<Float>.zero; e1[a1] = 1
                var e2 = SIMD3<Float>.zero; e2[a2] = 1
                let t1 = simd_act(d.q, e1) * d.he[a1], t2 = simd_act(d.q, e2) * d.he[a2]
                let c0 = d.c + n * d.he[axis]
                // A face covered by another child (the jib / counter-jib joint, the cab's
                // top against the jib) is inside the union: nothing to probe.
                if drawn.indices.contains(where: { $0 != k && inside(c0, drawn[$0], pad: 1e-6) }) { continue }
                // The face centre, else the first of four off-centre face points whose probe
                // sphere is clear of every other child by more than the margin.
                let candidates = [c0, c0 + 0.8 * t1, c0 - 0.8 * t1, c0 + 0.8 * t2, c0 - 0.8 * t2]
                guard let p = candidates.first(where: { pt in
                    drawn.indices.allSatisfy { $0 == k || distance(pt + n * (r + gap), drawn[$0]) > r + 2e-4 + 1e-5 }
                }) else { XCTFail("child \(k) face \(axis)\(sgn): no clear probe point"); continue }
                if p != c0 { offCentre += 1 }
                let probe = try XCTUnwrap(s.spawnSphere(at: p + n * (r + gap), radius: r, mass: 1e-4))
                s.generateContactsNow()
                let cs = pairContacts(s, body, probe)
                // Normal from the compound towards the probe (nrm points B → A).
                let hits = cs.map { c -> (n: SIMD3<Float>, depth: Float, p: SIMD3<Float>) in
                    let toA = SIMD3(c.nrm.x, c.nrm.y, c.nrm.z)
                    let a = Int(c.meta.x)
                    let cp = s.position(of: a)! + SIMD3(c.rA.x, c.rA.y, c.rA.z)
                    return (a == probe ? toA : -toA, c.nrm.w, cp)
                }
                XCTAssertEqual(hits.count, 1, "child \(k) face \(sgn > 0 ? "+" : "−")\(["x", "y", "z"][axis]): one contact")
                if let h0 = hits.first {
                    worstDepth = max(worstDepth, abs(h0.depth + gap))
                    worstN = max(worstN, acos(min(1, simd_dot(h0.n, n))) * 180 / .pi)
                    worstP = max(worstP, simd_length(h0.p - (p + n * (gap / 2))))
                }
                probed += 1
                s.despawn(probe)
            } }
        }
        print("B1c_faces jib at a tilted pose: collider read-back worst centre \(worstC * 1000) mm / \(worstQ)°; \(probed) exposed faces probed (\(offCentre) off-centre): worst |depth + 0.1 mm| \(worstDepth * 1000) mm, normal \(worstN)°, point \(worstP * 1000) mm")
        XCTAssertEqual(probed, 21, "24 faces less the 3 covered (the jib / counter-jib joint, the cab's top)")
        XCTAssertLessThan(worstDepth, 2e-6, "each face exactly where it is drawn (depth = the 0.1 mm gap)")
        XCTAssertLessThan(worstN, 0.01, "normal = the drawn face normal")
        XCTAssertLessThan(worstP, 1e-5, "contact at the drawn face (midway across the gap)")

        // 3. The notch between the jib's underside and the cab: no contact.
        let jibD = drawn[0], cabD = drawn[2]
        let notchLocal = SIMD3<Float>(0.002 + 0.0015, -0.002 - 0.0015, 0)          // design frame
        let notch = td + simd_act(rd, notchLocal)
        XCTAssertFalse(inside(notch, jibD, pad: 0.001) && inside(notch, cabD, pad: 0.001))
        let probe = try XCTUnwrap(s.spawnSphere(at: notch, radius: r, mass: 1e-4))
        s.generateContactsNow()
        let notchContacts = pairContacts(s, body, probe)
        print("B1c_faces probe in the notch (1 mm from the jib and the cab): \(notchContacts.count) contact(s)")
        XCTAssertTrue(notchContacts.isEmpty, "the notch is empty space: the collider is the boxes")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Compound ≈ the same boxes welded
    // ══════════════════════════════════════════════════════════════════════════

    /// The compound against the same boxes as separate bodies held by weld joints (masses =
    /// the compound's mass fractions, welds star-connected at child 0), 20 iterations,
    /// warm start + manifold solve, undamped.
    /// (a) The jib thrown up and sideways, spinning at 2 rad/s about its major principal
    /// axis. The compound is EXACTLY rigid: it follows the analytic rigid rotation
    /// (q(t) = exp(ωt)·q₀) to the quaternion integrator's error and the discrete ballistic
    /// COM to float noise. The welded set is rigid only to its welds: each substep's linear
    /// step leaves the children ω²r·dt²/2 off their circles and the joint bias pulls them
    /// back — so it runs δ ≈ 9 µm stretched, with a proportionally larger inertia and a
    /// slower spin. The tolerance is that measured weld error: child centres within 2δ
    /// (+ 2 µm), orientations within 0.1° (2δ over the jib's 16 mm radius of gyration is
    /// 0.1 % of the 38° turned = 0.04°). At 6 rad/s the stretch is 75 µm and the welded
    /// set lags by 1.3° while the compound stays within 0.011° of analytic (printed).
    /// (b) The L dropped flat from 3 mm onto the floor: it lands, bounces (e 0.15), rests —
    /// the compound and the welded boxes agree to 5 µm and 0.02° in every frame (their welds
    /// hold to a nanometre here: no rotation, so no stretch), and it rests flat at 2 mm.
    func testCompoundMatchesWeldedBoxes() throws {
        struct Trace { var poses: [[(c: SIMD3<Float>, q: simd_quatf)]] = []; var com: [SIMD3<Float>] = []; var weldErr: Float = 0 }
        func run(_ boxes: [CoinCompoundBox], compound: Bool, rd: simd_quatf, td: SIMD3<Float>,
                 v0: SIMD3<Float>, w: SIMD3<Float>, frames: Int, floor: Bool) throws -> Trace {
            let (s, q) = try makeSolver(maxCoins: 16, sleep: false,
                                        colliders: floor ? [.plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.15)] : [])
            s.velocityIterations = 20
            s.rollingResistance = 0.024 / 20
            s.warmStart = true
            s.manifoldSolve = true
            s.linDamping = 1
            s.angDamping = 1
            let prep = try XCTUnwrap(CoinCompoundMath.prepare(boxes))
            let M: Float = 0.004
            let comW = td + simd_act(rd, prep.comOffset)
            var tr = Trace()
            if compound {
                let h = try XCTUnwrap(s.registerCompound(boxes: boxes))
                let b = try XCTUnwrap(s.spawnCompound(design: rd, td, compound: h, velocity: v0, tumble: w, mass: M))
                step(s, q, frames: frames) { _ in
                    tr.poses.append(s.compoundColliderBoxes(of: b)!.map { ($0.center, $0.orientation) })
                    tr.com.append(s.position(of: b)!)
                }
            } else {
                var ids: [Int] = []
                for (i, bx) in boxes.enumerated() {
                    let c = td + simd_act(rd, bx.center)
                    let qq = (rd * bx.orientation).normalized
                    ids.append(try XCTUnwrap(s.spawnBox(at: c, halfExtents: bx.halfExtents, velocity: v0 + simd_cross(w, c - comW),
                                                        orient: Self.v4(qq), tumble: w, mass: M * prep.massFractions[i])))
                }
                var welds: [(Int, SIMD3<Float>, SIMD3<Float>)] = []   // body, local anchor on 0, local anchor on it
                for i in 1..<ids.count {
                    let anchor = 0.5 * (s.position(of: ids[0])! + s.position(of: ids[i])!)
                    XCTAssertNotNil(s.addWeldJoint(bodyA: ids[0], bodyB: ids[i], worldAnchor: anchor))
                    let q0 = s.orientation(of: ids[0])!, qi = s.orientation(of: ids[i])!
                    welds.append((i, simd_act(q0.inverse, anchor - s.position(of: ids[0])!), simd_act(qi.inverse, anchor - s.position(of: ids[i])!)))
                }
                step(s, q, frames: frames) { _ in
                    tr.poses.append(ids.map { (s.position(of: $0)!, s.orientation(of: $0)!) })
                    tr.com.append(zip(ids, prep.massFractions).reduce(SIMD3<Float>.zero) { $0 + s.position(of: $1.0)! * $1.1 })
                    let p0 = s.position(of: ids[0])!, q0 = s.orientation(of: ids[0])!
                    for (i, a0, ai) in welds {
                        let pa = p0 + simd_act(q0, a0), pi = s.position(of: ids[i])! + simd_act(s.orientation(of: ids[i])!, ai)
                        tr.weldErr = max(tr.weldErr, simd_length(pa - pi))
                    }
                }
            }
            return tr
        }
        func compare(_ a: Trace, _ b: Trace) -> (pos: Float, deg: Float) {
            var dp: Float = 0, dq: Float = 0
            for (fa, fb) in zip(a.poses, b.poses) {
                for (x, y) in zip(fa, fb) { dp = max(dp, simd_length(x.c - y.c)); dq = max(dq, Self.angleDeg(x.q * y.q.inverse)) }
            }
            return (dp, dq)
        }

        // (a) Free flight.
        let rdA = simd_quatf(angle: 0.3, axis: simd_normalize(SIMD3<Float>(1, 2, 0.5)))
        let tdA = SIMD3<Float>(-0.05, 0.15, 0)
        let v0 = SIMD3<Float>(0.15, 1.0, 0.05)
        let prep = try XCTUnwrap(CoinCompoundMath.prepare(Self.jib))
        let k = prep.invInertiaK
        var major = SIMD3<Float>(0, 0, 0); major[k.x <= k.y && k.x <= k.z ? 0 : (k.y <= k.z ? 1 : 2)] = 1
        let axisW = simd_act(rdA * prep.principalRotation, major)
        var lines: [String] = []
        for spin: Float in [2, 6] {
            let c = try run(Self.jib, compound: true, rd: rdA, td: tdA, v0: v0, w: axisW * spin, frames: 20, floor: false)
            let w = try run(Self.jib, compound: false, rd: rdA, td: tdA, v0: v0, w: axisW * spin, frames: 20, floor: false)
            XCTAssertEqual(c.poses.count, 20)
            // Analytic: every child turns rigidly, exp(ω t) about the COM; the COM follows the
            // solver's own discrete ballistic path (semi-implicit Euler, 3 substeps a frame).
            var x = tdA + simd_act(rdA, prep.comOffset), v = v0
            var cAn: Float = 0, cCom: Float = 0, wAn: Float = 0
            for f in 0..<20 {
                for _ in 0..<3 { v.y -= 9.81 / 180; x += v / 180 }
                let qa = simd_quatf(angle: spin * Float(f + 1) / 60, axis: axisW) * (rdA * Self.jib[0].orientation)
                cAn = max(cAn, Self.angleDeg(c.poses[f][0].q * qa.inverse))
                wAn = max(wAn, Self.angleDeg(w.poses[f][0].q * qa.inverse))
                cCom = max(cCom, simd_length(c.com[f] - x))
            }
            let d = compare(c, w)
            lines.append("\(spin) rad/s: compound vs analytic \(cAn)°, COM \(cCom * 1000) mm | welded vs analytic \(wAn)° (weld error \(w.weldErr * 1000) mm) | compound vs welded \(d.pos * 1000) mm, \(d.deg)°")
            XCTAssertLessThan(cAn, 0.02 * spin / 2, "the compound turns as the rigid body it is (\(spin) rad/s)")
            XCTAssertLessThan(cCom, 1e-6, "its COM flies the ballistic path (\(spin) rad/s)")
            if spin == 2 {
                XCTAssertLessThan(d.pos, 2 * w.weldErr + 2e-6, "child boxes within the welds' own error")
                XCTAssertLessThan(d.deg, 0.1, "orientations within the weld stretch's phase lag")
            }
        }
        print("B1c_weld free flight, 1/3 s: " + lines.joined(separator: " || "))

        // (b) The L dropped flat from 3 mm onto the floor: lands, bounces, rests (2 s).
        let tdB = SIMD3<Float>(0, 0.002 + 0.003, 0)
        let cB = try run(Self.ell, compound: true, rd: Self.identity, td: tdB, v0: .zero, w: .zero, frames: 120, floor: true)
        let wB = try run(Self.ell, compound: false, rd: Self.identity, td: tdB, v0: .zero, w: .zero, frames: 120, floor: true)
        let dB = compare(cB, wB)
        let restY = cB.poses.last!.map { $0.c.y }
        print("B1c_weld L dropped flat from 3 mm, 2 s: compound vs welded worst child offset \(dB.pos * 1000) mm, \(dB.deg)° over every frame; weld error \(wB.weldErr * 1000) mm; compound child y at rest \(restY.map { $0 * 1000 }) mm (design 2.0)")
        XCTAssertLessThan(dB.pos, 5e-6, "frame by frame, the same landing")
        XCTAssertLessThan(dB.deg, 0.02)
        for y in restY { XCTAssertEqual(y, 0.002, accuracy: 1e-4 + 1e-5, "rests on its flat bottom, inside the slop") }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Rest, sleep, wake
    // ══════════════════════════════════════════════════════════════════════════

    /// The L resting flat on the floor, on a static box ledge, and on a dynamic box —
    /// sleep OFF, §G otherwise — for 10 s. A compound rests on one manifold per child (the
    /// L: two), exactly the redundant-manifold case of VZ-0157. On the floor and the ledge
    /// the manifold solve alone holds it; on a DYNAMIC box the two manifolds and the box's
    /// own four floor points are coupled through the box, and cold-started Gauss–Seidel at
    /// 6 iterations leaves a residual every substep (the cube-tower mechanism) — with the
    /// warm start the stack does not move. (The recommended §G setting is both.)
    func testCompoundRestsWithoutCreep() throws {
        func run(_ support: String, manifold: Bool, warm: Bool) throws -> (mm: Float, deg: Float) {
            let ledge = CoinStaticCollider.box(center: SIMD3(0.1, 0.01, 0), halfExtents: SIMD3(0.03, 0.01, 0.03))
            let (s, q) = try makeSolver(sleep: false,
                                        colliders: [.plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.15), ledge])
            s.manifoldSolve = manifold
            s.warmStart = warm
            let h = try XCTUnwrap(s.registerCompound(boxes: Self.ell))
            let body: Int
            switch support {
            case "floor": body = try XCTUnwrap(s.spawnCompound(design: Self.identity, SIMD3(-0.1, 0.0021, 0), compound: h, mass: 0.003))
            case "ledge": body = try XCTUnwrap(s.spawnCompound(design: Self.identity, SIMD3(0.1, 0.0221, 0), compound: h, mass: 0.003))
            default:
                _ = s.spawnBox(at: SIMD3(0, 0.005, 0), halfExtents: SIMD3(0.02, 0.005, 0.02), mass: 0.01)
                body = try XCTUnwrap(s.spawnCompound(design: Self.identity, SIMD3(0, 0.0121, 0), compound: h, mass: 0.003))
            }
            var p0 = SIMD3<Float>.zero, q0 = Self.identity
            step(s, q, frames: 630) { f in if f == 29 { p0 = s.position(of: body)!; q0 = s.orientation(of: body)! } }
            return (simd_length(s.position(of: body)! - p0) * 1000, Self.angleDeg(s.orientation(of: body)! * q0.inverse))
        }
        var lines: [String] = []
        for support in ["floor", "ledge", "box"] {
            let perPoint = try run(support, manifold: false, warm: false)
            let ms = try run(support, manifold: true, warm: false)
            let both = try run(support, manifold: true, warm: true)
            lines.append("\(support): per-point \(perPoint.mm) mm / \(perPoint.deg)°, manifold \(ms.mm) mm / \(ms.deg)°, manifold + warm \(both.mm) mm / \(both.deg)°")
            if support != "box" {
                XCTAssertLessThan(ms.mm, 1e-3, "the L on the \(support) does not slide (manifoldSolve)")
                XCTAssertLessThan(ms.deg, 1e-3, "…or turn")
            }
            XCTAssertLessThan(both.mm, 1e-3, "the L on the \(support) does not slide (manifoldSolve + warmStart)")
            XCTAssertLessThan(both.deg, 1e-3, "…or turn")
        }
        print("B1c_rest L, sleep off, §G, 0.5→10.5 s drift: " + lines.joined(separator: " | "))
    }

    /// Sleep and wake treat the compound as one body: the jib resting on its cab (§G, sleep
    /// on) falls asleep; a small box dropped onto the jib wakes it in the frame of the
    /// impact; both settle and sleep again. The static bounding reject (VZ-0151) probes
    /// a compound by its union's bounding radius: compounds scattered through every culled
    /// collider kind emit the bit-identical contact set with the reject on and off.
    func testCompoundSleepsWakesAndPassesTheColliderReject() throws {
        do {
            let (s, q) = try makeSolver()
            s.manifoldSolve = true
            let h = try XCTUnwrap(s.registerCompound(boxes: Self.jib))
            let body = try XCTUnwrap(s.spawnCompound(design: Self.identity, SIMD3(0, 0.0101, 0), compound: h, mass: 0.004))
            var asleepAt = -1
            step(s, q, frames: 120) { f in if asleepAt < 0 && s.isAsleep(body) { asleepAt = f } }
            XCTAssertGreaterThan(asleepAt, 0, "the resting jib falls asleep")
            let restP = s.position(of: body)!
            // Jib top at y = 10.1 + 2 = 12.1 mm; the 3 mm cube starts 3 mm above it, falling.
            let ball = try XCTUnwrap(s.spawnBox(at: SIMD3(0.0, 0.0121 + 0.0015 + 0.003, 0), halfExtents: SIMD3(0.0015, 0.0015, 0.0015),
                                                velocity: SIMD3(0, -0.3, 0), mass: 0.0005))
            var wokeAt = -1, contactAt = -1, sleptAgain = -1
            step(s, q, frames: 240) { f in
                if contactAt < 0 && !self.pairContacts(s, body, ball).isEmpty { contactAt = f }
                if wokeAt < 0 && !s.isAsleep(body) { wokeAt = f }
                if wokeAt >= 0 && sleptAgain < 0 && f > wokeAt + 2 && s.isAsleep(body) && s.isAsleep(ball) { sleptAgain = f }
            }
            print("B1c_sleep jib asleep @frame \(asleepAt); cube dropped on the jib: first contact @\(contactAt), jib awake @\(wokeAt), both asleep again @\(sleptAgain); jib moved \(simd_length(s.position(of: body)! - restP) * 1000) mm")
            XCTAssertGreaterThanOrEqual(wokeAt, 0, "the impact wakes the compound")
            XCTAssertLessThanOrEqual(wokeAt, contactAt + 1, "in the frame of the impact")
            XCTAssertGreaterThan(sleptAgain, wokeAt, "and it settles back to sleep")
        }

        // The collider reject: bit-identical with compounds among every culled kind.
        var seed: UInt64 = 0xB1C_0151
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(seed >> 40) / Float(1 << 24) }
        var cols: [CoinStaticCollider] = [.plane(normal: SIMD3(0, 1, 0), offset: 0),
                                          .plane(normal: simd_normalize(SIMD3(1, 0.3, 0)), offset: -0.12)]
        for _ in 0..<10 {
            cols.append(.box(center: SIMD3((rnd() - 0.5) * 0.2, rnd() * 0.08, (rnd() - 0.5) * 0.2),
                             halfExtents: SIMD3(0.004 + rnd() * 0.02, 0.004 + rnd() * 0.02, 0.004 + rnd() * 0.02)))
            cols.append(.orientedBox(center: SIMD3((rnd() - 0.5) * 0.2, rnd() * 0.08, (rnd() - 0.5) * 0.2),
                                     halfExtents: SIMD3(0.004 + rnd() * 0.02, 0.003 + rnd() * 0.01, 0.004 + rnd() * 0.02),
                                     orientation: simd_quatf(angle: rnd() * 2 * .pi, axis: simd_normalize(SIMD3(rnd(), rnd(), rnd()) + 1e-3))))
        }
        cols.append(.cylinder(center: SIMD3(0, 0.05, 0), axis: SIMD3(1, 0, 0.2), radius: 0.035,
                              up: SIMD3(0, 1, 0), halfLength: 0.09, lowerHalfOnly: false))
        cols.append(.cylinder(center: SIMD3(0, 0.03, 0.05), axis: SIMD3(0, 0, 1), radius: 0.03,
                              up: SIMD3(0, 1, 0), halfLength: 0.08, lowerHalfOnly: true))
        for i in 0..<40 { cols.append(.box(center: SIMD3(0.35, 0.35, -0.35 + 0.017 * Float(i)), halfExtents: SIMD3(0.004, 0.004, 0.004))) }
        for spec: Float in [2e-4, 0.006] {
            let (s, _) = try makeSolver(maxCoins: 64, colliders: cols)
            s.speculativeMargin = spec
            let hs = [try XCTUnwrap(s.registerCompound(boxes: Self.jib)), try XCTUnwrap(s.registerCompound(boxes: Self.ell))]
            for i in 0..<24 {
                let p = SIMD3((rnd() - 0.5) * 0.28, rnd() * 0.1, (rnd() - 0.5) * 0.28)
                let qd = simd_quatf(angle: rnd() * 2 * .pi, axis: simd_normalize(SIMD3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) + 1e-3))
                _ = s.spawnCompound(design: qd, p, compound: hs[i % 2], mass: 0.004)
            }
            s.generateContactsNow()
            let culled = contactSet(s)
            XCTAssertLessThan(s.polyPairCount, s.maxPolyPairs, "precondition: the pair list did not overflow")
            XCTAssertLessThan(s.contactCount, s.maxContacts, "precondition: the contact buffer did not overflow")
            s.colliderCullDisabledForTesting = true
            s.generateContactsNow()
            let full = contactSet(s)
            let statics = full.filter { $0.split(separator: " ")[1] == "4294967295" }.count
            print("B1c_cull pairs \(s.polyPairCount)/\(s.maxPolyPairs),  spec=\(spec): compound contacts culled=\(culled.count) unculled=\(full.count) (static \(statics)) identical=\(culled == full)")
            XCTAssertGreaterThan(statics, 30, "precondition: plenty of compound–static contacts")
            XCTAssertEqual(culled, full, "the reject never changes a compound's contact set (spec \(spec))")
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Against every shape
    // ══════════════════════════════════════════════════════════════════════════

    /// The table (5 children) on the floor with one of everything resting on its top: a
    /// sphere, a disc, a capsule lying down, a Digital Clock bar face down, a box, a small
    /// L compound, and the ballasted worker standing — each pairing runs its own compound
    /// path (swept sphere vs child, disc vs child, SAT + clipping vs box / hull / child).
    /// After 3 s everything rests at its geometric height on the top (within one contact
    /// slop per interface), the pile's deepest penetration is inside the slop, and
    /// everything is asleep.
    /// A compound against a hull WITHOUT GPU topology (a 162-vertex, 320-face icosphere —
    /// beyond the 64-face cap). cdAppendPolyPairs sent every non-polytope, non-round partner of
    /// a compound to the DISC path, so this hull was collided as a disc prism (its bounding
    /// radius × its smallest half-extent): a compound 1.3 mm clear of the 20 mm sphere got four
    /// contacts 3.4–4.0 mm "deep" (verifier, stage B1). Such a pair now runs GJK/EPA per child
    /// (CD_PP_GJK): no contact when clear, the true depth when it overlaps — in the contact set
    /// and in the penetration probe (cdPolyPairDepth) alike.
    func testCompoundAgainstATopologylessHull() throws {
        let (s, _) = try makeSolver(maxCoins: 8)
        s.gravity = 0
        let t = (1 + Float(5).squareRoot()) / 2
        var v: [SIMD3<Float>] = [SIMD3(-1, t, 0), SIMD3(1, t, 0), SIMD3(-1, -t, 0), SIMD3(1, -t, 0),
                                 SIMD3(0, -1, t), SIMD3(0, 1, t), SIMD3(0, -1, -t), SIMD3(0, 1, -t),
                                 SIMD3(t, 0, -1), SIMD3(t, 0, 1), SIMD3(-t, 0, -1), SIMD3(-t, 0, 1)].map { simd_normalize($0) }
        var f: [(Int, Int, Int)] = [(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),(1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
                                    (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),(4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)]
        for _ in 0..<2 {                                                   // subdivide twice: 162 vertices
            var mids: [Int: Int] = [:]
            func mid(_ a: Int, _ b: Int) -> Int {
                let k = min(a, b) * 100_000 + max(a, b)
                if let m = mids[k] { return m }
                v.append(simd_normalize(v[a] + v[b])); mids[k] = v.count - 1; return v.count - 1
            }
            f = f.flatMap { (a, b, c) -> [(Int, Int, Int)] in
                let ab = mid(a, b), bc = mid(b, c), ca = mid(c, a)
                return [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
            }
        }
        let R: Float = 0.02
        let hull = try XCTUnwrap(s.registerHull(vertices: v.map { $0 * R }))
        XCTAssertFalse(s.hullHasTopology(hull.index), "precondition: past the GPU topology caps")
        let cube = try XCTUnwrap(s.registerCompound(boxes: [CoinCompoundBox(center: .zero, halfExtents: SIMD3(0.003, 0.003, 0.003))]))
        let h = try XCTUnwrap(s.spawnHull(at: SIMD3(0, 0.1, 0), hull: hull))
        // Clear: the cube's nearest point (16, 14, 0) mm from the centre, 21.3 mm — 1.3 mm off.
        let clear = try XCTUnwrap(s.spawnCompound(at: SIMD3(0.019, 0.117, 0), compound: cube))
        s.generateContactsNow()
        let cs0 = pairContacts(s, clear, h)
        let pen0 = s.measurePenetration(threshold: 1e-5)
        s.despawn(clear)
        // Overlapping: its −x face at 15 mm, 5 mm inside the sphere (the icosphere's flat there
        // sits within 0.01 mm of the sphere on the ±x axis vertex).
        let deep = try XCTUnwrap(s.spawnCompound(at: SIMD3(0.018, 0.1, 0), compound: cube))
        s.generateContactsNow()
        let cs1 = pairContacts(s, deep, h)
        let pen1 = s.measurePenetration(threshold: 1e-5)
        print("B1v_gjk compound vs a topology-less hull: 1.3 mm clear → \(cs0.count) contacts (probe \(pen0.penetratingPairs) pairs) | 5 mm in → \(cs1.count) contact(s), depth \(cs1.map { $0.nrm.w * 1000 }) mm, probe \(pen1.maxPenetration * 1000) mm")
        XCTAssertTrue(cs0.isEmpty, "1.3 mm clear: no contact (was four, 3.4–4.0 mm deep)")
        XCTAssertEqual(pen0.penetratingPairs, 0, "…and the penetration probe agrees")
        XCTAssertEqual(cs1.count, 1)
        let n1 = cs1.first.map { SIMD3($0.nrm.x, $0.nrm.y, $0.nrm.z) * (Int($0.meta.x) == deep ? 1 : -1) } ?? .zero
        XCTAssertEqual(cs1.first?.nrm.w ?? 0, 0.005, accuracy: 1e-5, "the true 5 mm overlap")
        XCTAssertGreaterThan(n1.x, 0.999, "pushing the cube out along +x")
        XCTAssertEqual(pen1.maxPenetration, 0.005, accuracy: 1e-5, "the penetration probe sees the same 5 mm")
    }

    func testEveryShapeRestsOnACompound() throws {
        typealias T = CoinDEMActuationTests
        let (s, q) = try makeSolver(maxCoins: 16, maxRadius: 0.03)
        s.manifoldSolve = true
        s.warmStart = true
        let th = try XCTUnwrap(s.registerCompound(boxes: Self.table))
        let lh = try XCTUnwrap(s.registerCompound(boxes: Self.ell))
        let bar = try XCTUnwrap(s.registerHull(vertices: T.barHullPoints()))
        let table = try XCTUnwrap(s.spawnCompound(design: Self.identity, SIMD3(0, 0.00005, 0), compound: th, mass: 0.05))
        let top: Float = 0.022 + 0.00005
        let lift: Float = 0.0003
        var expect: [(String, Int, Float)] = []
        expect.append(("sphere", try XCTUnwrap(s.spawnSphere(at: SIMD3(-0.033, top + 0.004 + lift, -0.02), radius: 0.004, mass: 0.002)), 0.004))
        expect.append(("disc", try XCTUnwrap(s.spawn(at: SIMD3(-0.033, top + 0.0015 + lift, 0.02), radius: 0.006, halfThickness: 0.0015, mass: 0.001)), 0.0015))
        let alongZ = Self.v4(simd_quatf(angle: .pi / 2, axis: SIMD3(1, 0, 0)))
        expect.append(("capsule", try XCTUnwrap(s.spawnCapsule(at: SIMD3(-0.011, top + 0.003 + lift, 0), radius: 0.003, halfLength: 0.008, orient: alongZ, mass: 0.002)), 0.003))
        let rdBar = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))           // face (local z) up
        let qBar = rdBar * bar.principalRotation
        let barTd = SIMD3<Float>(0.011, top + 0.00275 + lift, 0)
        let barSlot = try XCTUnwrap(s.spawnHull(at: barTd + simd_act(rdBar, bar.comOffset), hull: bar, orient: Self.v4(qBar),
                                                mass: T.barMass, friction: 0.3, restitution: 0.2))
        expect.append(("bar", barSlot, 0.00275 + (s.position(of: barSlot)!.y - barTd.y)))
        expect.append(("box", try XCTUnwrap(s.spawnBox(at: SIMD3(0.033, top + 0.003 + lift, -0.02), halfExtents: SIMD3(0.004, 0.003, 0.005), mass: 0.003)), 0.003))
        let lSlot = try XCTUnwrap(s.spawnCompound(design: Self.identity, SIMD3(0.033, top + 0.002 + lift, 0.012), compound: lh, mass: 0.003))
        expect.append(("L", lSlot, 0.002 + (s.position(of: lSlot)!.y - (top + 0.002 + lift))))
        let wp = T.worker
        let egg = try XCTUnwrap(s.spawnBallastedEgg(wp, fatCenter: SIMD3(0, top + wp.fatRadius + lift, -0.029)))
        expect.append(("worker", egg, wp.fatRadius + (s.position(of: egg)!.y - (top + wp.fatRadius + lift))))
        step(s, q, frames: 180)
        let tableY = s.position(of: table)!.y
        let topNow = tableY - th.comOffset.y + 0.022                               // the top's surface, now
        var lines: [String] = []
        for (name, slot, h) in expect {
            let y = s.position(of: slot)! .y
            let rest = y - topNow - h                                              // gap to the top's surface
            lines.append("\(name) \(String(format: "%+.4f", rest * 1000)) mm\(s.isAsleep(slot) ? "" : " (awake)")")
            XCTAssertLessThanOrEqual(rest, 1e-5, "\(name) is not floating above the table")
            XCTAssertGreaterThanOrEqual(rest, -1e-4 - 1e-5, "\(name) rests ON the table (one interface: within the contact slop)")
            XCTAssertTrue(s.isAsleep(slot), "\(name) settles and sleeps")
        }
        let pen = s.measurePenetration(threshold: 1e-4)
        print("B1c_shapes on the table after 3 s (gap to the top): " + lines.joined(separator: ", ") + "; table legs sit \((tableY - th.comOffset.y) * 1000) mm off the floor; deepest penetration \(pen.maxPenetration * 1000) mm over \(pen.penetratingPairs) pairs")
        XCTAssertLessThanOrEqual(pen.maxPenetration, 1.2e-4, "no pair deeper than the contact slop")
        XCTAssertTrue(s.isAsleep(table))
    }
}
