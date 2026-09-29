import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Gate for MANY-CHILD compounds (CoinDEM shape tag 6 above the old 16-child cap): an OPEN
/// TRUSS — the Digital Clock's toy tower-crane jib, a 20 × 13 mm box truss of two Warren side
/// panels, every chord, diagonal and batten its own exact box (90 children) — collides as the
/// boxes it is drawn from: its collider reads back as the drawn boxes, every exposed face of
/// every child answers a probe at the drawn face with that child's own feature id, it rests on
/// supports without creep and sleeps, a bar dropped across it lands ON the top chords, and a
/// cube dropped over one of the open panels falls THROUGH the truss (top and bottom openings)
/// to the floor below. The cap is 128 (`CoinCompoundMath.maxChildren`); 129 is refused.
/// Measured numbers are PRINTed with a `B4t_` prefix. §G toy-scale configuration and the
/// runtime-compiled-library seam, as CoinDEMCompoundTests.
@MainActor
final class CoinDEMTrussCompoundTests: XCTestCase {

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

    /// §G knobs (1/180 s × 6, μ 0.5, 0.2 mm margin, 0.1 mm slop), manifold solve + warm start.
    /// The broadphase cell is 0.1 m: its ±4-cell neighbourhood (0.4 m) reaches the jib's tip from
    /// its COM (0.31 m) — at the clock's 0.06 m cells the multi-dispatch path would not see a
    /// partner past 0.24 m (the scene runs the one-cell small-world path, which sees every pair).
    private func makeSolver(maxCoins: Int = 16, colliders: [CoinStaticCollider], sleep: Bool = true,
                            smallWorld: Bool = false) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: maxCoins,
                                    coinRadius: 0.05, halfThickness: 0.05,
                                    boundsMin: SIMD3(-0.1, -0.05, -0.1), boundsMax: SIMD3(0.55, 0.2, 0.1))
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = 9.81
        s.fixedDt = 1.0 / 180
        s.maxSubsteps = 10
        s.velocityIterations = 6
        s.colorRounds = 16
        s.islandUnionRounds = 6
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
        s.manifoldSolve = true
        s.warmStart = true
        s.smallWorldPath = smallWorld
        s.setColliders(colliders)
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

    private static let identity = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private static func angleDeg(_ q: simd_quatf) -> Float {
        let n = q.normalized
        return 2 * atan2(simd_length(n.imag), abs(n.real)) * 180 / .pi
    }

    private func pairContacts(_ s: CoinDEMSolver, _ a: Int, _ b: Int) -> [CoinContact] {
        s.contacts(touching: [a]).filter { c in
            (Int(c.meta.x) == a && Int(c.meta.y) == b) || (Int(c.meta.x) == b && Int(c.meta.y) == a)
        }
    }

    /// The compound child a contact's feature id names (cdPieceFeat): low 4 bits at 26 / 22,
    /// high 3 bits at 9 / 12, on the side of the body that listed the pair (the lower index).
    private static func piece(_ feature: UInt32, threadSide: Bool) -> Int {
        threadSide ? Int(((feature >> 26) & 15) | (((feature >> 9) & 7) << 4))
                   : Int(((feature >> 22) & 15) | (((feature >> 12) & 7) << 4))
    }

    // ── The truss ────────────────────────────────────────────────────────────

    /// One child as drawn: the box and what it is.
    struct Member { var box: CoinCompoundBox; var role: String }

    /// The toy crane's jib as the Digital Clock builds it (`ClockCraneTruss`): along +x from
    /// its root (x = 0) to x = `length`, bottom chords' underside at y = 0, +z across. Two
    /// Warren side panels (`diagonals` each, ends buried 0.5 mm past the chords' centre lines,
    /// faces 0.2 mm inside the chords'), a batten across the panels at every panel point
    /// (bottom at even k, top at odd k). Children in the order a caller might give them —
    /// diagonals, battens, then the four chords LAST, so the chords a bar rests on are children
    /// 85…88 (feature high bits in use) — and a 20 mm ballast block under the root.
    static func warrenJib(length: Double = 0.44, diagonals n: Int = 28) -> [Member] {
        let chord = 0.003, web = 0.0024, inset = 0.0002, bury = 0.0005
        let webThick = chord - 2 * inset
        let depth = 0.020, width = 0.013
        let yB = chord / 2, yT = depth - chord / 2, zP = width / 2 - chord / 2
        let x0 = 0.004, x1 = length - 0.004
        func px(_ k: Int) -> Double { x0 + (x1 - x0) * Double(k) / Double(n) }
        func f(_ v: SIMD3<Double>) -> SIMD3<Float> { SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z)) }
        /// A member from a to b (local +x along it), `w` across in the plane with normal `nrm`
        /// (local +y), `t` along the normal (local +z).
        func strut(_ a: SIMD3<Double>, _ b: SIMD3<Double>, w: Double, t: Double, nrm: SIMD3<Double>) -> CoinCompoundBox {
            let d = b - a, len = simd_length(d), ex = d / len
            let ez = simd_normalize(nrm - ex * simd_dot(nrm, ex))
            let ey = simd_cross(ez, ex)
            let q = simd_quatd(simd_double3x3(ex, ey, ez)).normalized
            return CoinCompoundBox(center: f((a + b) / 2), halfExtents: SIMD3<Float>(Float(len / 2), Float(w / 2), Float(t / 2)),
                                   orientation: simd_quatf(ix: Float(q.imag.x), iy: Float(q.imag.y), iz: Float(q.imag.z), r: Float(q.real)).normalized)
        }
        var out: [Member] = []
        for (side, z) in [("front", zP), ("back", -zP)] {
            for k in 0..<n {
                let lowFirst = k % 2 == 0
                let a = SIMD3(px(k), lowFirst ? yB - bury : yT + bury, z)
                let b = SIMD3(px(k + 1), lowFirst ? yT + bury : yB - bury, z)
                out.append(Member(box: strut(a, b, w: web, t: webThick, nrm: SIMD3(0, 0, 1)), role: "diag.\(side).\(k)"))
            }
        }
        for k in 0...n {
            let y = k % 2 == 0 ? yB : yT
            out.append(Member(box: strut(SIMD3(px(k), y, -zP), SIMD3(px(k), y, zP), w: web, t: webThick, nrm: SIMD3(0, 1, 0)),
                              role: "batten.\(k % 2 == 0 ? "bottom" : "top").\(k)"))
        }
        for (name, y) in [("bottom", yB), ("top", yT)] {
            for (side, z) in [("front", zP), ("back", -zP)] {
                out.append(Member(box: CoinCompoundBox(center: f(SIMD3(length / 2, y, z)),
                                                       halfExtents: SIMD3<Float>(Float(length / 2), Float(chord / 2), Float(chord / 2))),
                                  role: "chord.\(name).\(side)"))
            }
        }
        // Ballast under the root (a denser block, 20 × 20 × 20 mm, its underside level with the chords').
        out.append(Member(box: CoinCompoundBox(center: SIMD3(-0.010, 0.010, 0), halfExtents: SIMD3(0.010, 0.010, 0.010), density: 2),
                          role: "ballast"))
        return out
    }

    // ══════════════════════════════════════════════════════════════════════════
    // The cap
    // ══════════════════════════════════════════════════════════════════════════

    /// 128 children register (a 128-box compound), 129 do not; the 90-child jib does.
    func testCompoundCapIs128() throws {
        let (s, _) = try makeSolver(colliders: [])
        func grid(_ count: Int) -> [CoinCompoundBox] {
            (0..<count).map { i in
                CoinCompoundBox(center: SIMD3(Float(i % 16) * 0.004, Float(i / 16) * 0.004, 0), halfExtents: SIMD3(0.0015, 0.0015, 0.0015))
            }
        }
        XCTAssertEqual(CoinCompoundMath.maxChildren, 128)
        XCTAssertNotNil(s.registerCompound(boxes: grid(128)), "128 children register")
        XCTAssertNil(s.registerCompound(boxes: grid(129)), "129 are refused")
        let jib = try XCTUnwrap(s.registerCompound(boxes: Self.warrenJib().map(\.box)))
        XCTAssertEqual(jib.children.count, 90)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Collider = drawn
    // ══════════════════════════════════════════════════════════════════════════

    /// The 90-child jib at a tilted pose: (1) its collider read back from the GPU's own shape
    /// table is the drawn boxes (centres ≤ 1 µm, orientations ≤ 0.001°, extents exact);
    /// (2) every exposed face of every child — a probe sphere 0.1 mm off the face, at its centre
    /// or the first off-centre point clear of every other child — gets ONE contact, at the drawn
    /// face (depth = −0.1 mm ± 2 µm; the drawn normal ± 0.1° — float32 world coordinates out to
    /// 0.45 m resolve a 0.1 mm probe gap's direction to ~0.03°), and that contact's feature id
    /// names THE child it was probed on (children 0…89: the high piece bits decode); (3) a probe
    /// in the middle of an open panel — 1.5 mm clear of every strut — gets none.
    func testTrussColliderIsTheDrawnGeometryFaceByFace() throws {
        let (s, _) = try makeSolver(maxCoins: 8, colliders: [])
        s.gravity = 0
        let members = Self.warrenJib()
        let h = try XCTUnwrap(s.registerCompound(boxes: members.map(\.box)))
        let rd = simd_quatf(angle: 0.35, axis: SIMD3(0, 1, 0)) * simd_quatf(angle: 0.15, axis: SIMD3(1, 0, 0))
        let td = SIMD3<Float>(0.01, 0.05, 0.0)
        let body = try XCTUnwrap(s.spawnCompound(design: rd, td, compound: h, mass: 0.03))
        struct Drawn { var c: SIMD3<Float>; var q: simd_quatf; var he: SIMD3<Float> }
        let drawn = members.map { Drawn(c: td + simd_act(rd, $0.box.center), q: (rd * $0.box.orientation).normalized, he: $0.box.halfExtents) }

        // 1. Read-back.
        let col = try XCTUnwrap(s.compoundColliderBoxes(of: body))
        XCTAssertEqual(col.count, drawn.count, "every drawn member is a child")
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
        var probed = 0, worstDepth: Float = 0, worstN: Float = 0, worstP: Float = 0
        var probedPerChild = [Int](repeating: 0, count: drawn.count)
        var wrongPiece: [String] = []
        for (k, d) in drawn.enumerated() {
            for axis in 0..<3 { for sgn: Float in [-1, 1] {
                var e = SIMD3<Float>.zero; e[axis] = sgn
                let n = simd_act(d.q, e)
                let a1 = (axis + 1) % 3, a2 = (axis + 2) % 3
                var e1 = SIMD3<Float>.zero; e1[a1] = 1
                var e2 = SIMD3<Float>.zero; e2[a2] = 1
                let t1 = simd_act(d.q, e1) * d.he[a1], t2 = simd_act(d.q, e2) * d.he[a2]
                let c0 = d.c + n * d.he[axis]
                var candidates = [c0]
                for f: Float in [0.8, 0.5, 0.3] { candidates += [c0 + f * t1, c0 - f * t1, c0 + f * t2, c0 - f * t2] }
                guard let p = candidates.first(where: { pt in
                    !drawn.indices.contains(where: { $0 != k && inside(pt, drawn[$0], pad: 1e-6) })
                        && drawn.indices.allSatisfy { $0 == k || distance(pt + n * (r + gap), drawn[$0]) > r + 2e-4 + 1e-5 }
                }) else { continue }                                   // a face covered where it meets the others
                let probe = try XCTUnwrap(s.spawnSphere(at: p + n * (r + gap), radius: r, mass: 1e-4))
                s.generateContactsNow()
                let cs = pairContacts(s, body, probe)
                XCTAssertEqual(cs.count, 1, "\(members[k].role) face \(sgn > 0 ? "+" : "−")\(["x", "y", "z"][axis]): one contact")
                if let c = cs.first {
                    let toA = SIMD3(c.nrm.x, c.nrm.y, c.nrm.z)
                    let a = Int(c.meta.x)
                    let cn = a == probe ? toA : -toA
                    let cp = s.position(of: a)! + SIMD3(c.rA.x, c.rA.y, c.rA.z)
                    worstDepth = max(worstDepth, abs(c.nrm.w + gap))
                    worstN = max(worstN, acos(min(1, simd_dot(cn, n))) * 180 / .pi)
                    worstP = max(worstP, simd_length(cp - (p + n * (gap / 2))))
                    let named = Self.piece(c.meta.z, threadSide: a == body)
                    if named != k { wrongPiece.append("\(members[k].role) (child \(k)) → feature names child \(named)") }
                }
                probed += 1
                probedPerChild[k] += 1
                s.despawn(probe)
            } }
        }
        let unprobed = probedPerChild.enumerated().filter { $0.element == 0 }.map { members[$0.offset].role }
        print("B4t_faces 90-child jib, tilted: read-back worst centre \(worstC * 1e6) µm / \(worstQ)°; \(probed) exposed faces probed on \(drawn.count - unprobed.count) children: worst |depth + 0.1 mm| \(worstDepth * 1e6) µm, normal \(worstN)°, point \(worstP * 1e6) µm; feature ids naming the wrong child: \(wrongPiece.count)")
        XCTAssertTrue(unprobed.isEmpty, "every child shows at least one face: \(unprobed)")
        XCTAssertGreaterThan(probed, 300)
        XCTAssertTrue(wrongPiece.isEmpty, "each contact's feature id names its own child: \(wrongPiece.prefix(5))")
        XCTAssertLessThan(worstDepth, 2e-6, "each face exactly where it is drawn")
        XCTAssertLessThan(worstN, 0.1, "normal = the drawn face normal (to float32 at 0.45 m over a 0.1 mm gap)")
        XCTAssertLessThan(worstP, 1e-5, "contact at the drawn face")

        // 3. The middle of an open side panel (a Warren triangle) and of the top between two
        //    battens: empty space.
        let jibDesign: [(String, SIMD3<Float>)] = [
            ("side-panel triangle", SIMD3(Float(0.004 + 0.432 * 3.0 / 28), 0.0135, 0.005)),   // over the bottom chord, between diagonals 2 and 3
            ("top opening", SIMD3(Float(0.004 + 0.432 * 6.0 / 28), 0.0185, 0)),
        ]
        for (name, pd) in jibDesign {
            let p = td + simd_act(rd, pd)
            let clear = drawn.map { distance(p, $0) }.min()!
            let probe = try XCTUnwrap(s.spawnSphere(at: p, radius: r, mass: 1e-4))
            s.generateContactsNow()
            let hits = pairContacts(s, body, probe)
            print("B4t_faces probe in the \(name) (\(clear * 1000) mm clear of every strut): \(hits.count) contact(s)")
            XCTAssertGreaterThan(clear, r + 3e-4, "precondition: the probe point is open space")
            XCTAssertTrue(hits.isEmpty, "the \(name) is empty space: the collider is the struts")
            s.despawn(probe)
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Culling clusters
    // ══════════════════════════════════════════════════════════════════════════

    /// The 90-child jib's culling clusters (CoinCompound.swift step 4): every child in exactly
    /// one cluster of ≤ 16, each cluster's AABB holding its children's boxes, the four chords
    /// (each spanning the whole jib) in a cluster of their own; a 16-child compound has none.
    /// Then the contact set: the jib and a second jib crossing it, among static boxes, oriented
    /// boxes, a plane and a tube cutting through them, with boxes, bar hulls, spheres, capsules,
    /// eggs, discs, a topology-less hull and a small compound scattered through the struts —
    /// generated with the clusters and with every child walked (the old loop), at the §G 0.2 mm
    /// margin and the clock's 2.5 mm: the SAME contacts, bit for bit, from fewer listed pairs.
    func testTrussClustersAreSoundAndLeaveTheContactSetUnchanged() throws {
        let members = Self.warrenJib()
        let prep = try XCTUnwrap(CoinCompoundMath.prepare(members.map(\.box)))
        XCTAssertEqual(prep.clusters.flatMap(\.members).sorted(), Array(0..<members.count), "every child in exactly one cluster")
        XCTAssertTrue(prep.clusters.allSatisfy { $0.members.count <= CoinCompoundMath.flatChildren })
        for cl in prep.clusters {
            for m in cl.members {
                let c = prep.children[m]
                let R = simd_float3x3(c.orientation)
                let e = abs(R.columns.0) * c.halfExtents.x + abs(R.columns.1) * c.halfExtents.y + abs(R.columns.2) * c.halfExtents.z
                XCTAssertTrue(all(abs(c.center - cl.center) + e .<= cl.halfExtents + 1e-6), "cluster box holds child \(m)")
            }
        }
        let chordCluster = prep.clusters.first { $0.members.contains(85) }!
        XCTAssertEqual(Set(chordCluster.members), [85, 86, 87, 88], "the four full-length chords cluster alone")
        let crate = (0..<16).map { i in CoinCompoundBox(center: SIMD3(Float(i) * 0.01, 0, 0), halfExtents: SIMD3(0.004, 0.004, 0.004)) }
        XCTAssertTrue(try XCTUnwrap(CoinCompoundMath.prepare(crate)).clusters.isEmpty, "≤ 16 children: no clusters")
        print("B4t_clusters 90-child jib: \(prep.clusters.count) clusters of \(prep.clusters.map(\.members.count)); extents (mm) \(prep.clusters.map { Int(2000 * max($0.halfExtents.x, max($0.halfExtents.y, $0.halfExtents.z))) })")

        // The contact set, clustered vs walked.
        let t = (1 + Float(5).squareRoot()) / 2
        var ico: [SIMD3<Float>] = [SIMD3(-1, t, 0), SIMD3(1, t, 0), SIMD3(-1, -t, 0), SIMD3(1, -t, 0),
                                   SIMD3(0, -1, t), SIMD3(0, 1, t), SIMD3(0, -1, -t), SIMD3(0, 1, -t),
                                   SIMD3(t, 0, -1), SIMD3(t, 0, 1), SIMD3(-t, 0, -1), SIMD3(-t, 0, 1)].map { simd_normalize($0) }
        var f: [(Int, Int, Int)] = [(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),(1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
                                    (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),(4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)]
        for _ in 0..<2 {
            var mids: [Int: Int] = [:]
            func mid(_ a: Int, _ b: Int) -> Int {
                let k = min(a, b) * 100_000 + max(a, b)
                if let m = mids[k] { return m }
                ico.append(simd_normalize(ico[a] + ico[b])); mids[k] = ico.count - 1; return ico.count - 1
            }
            f = f.flatMap { (a, b, c) -> [(Int, Int, Int)] in
                let ab = mid(a, b), bc = mid(b, c), ca = mid(c, a)
                return [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
            }
        }
        for spec: Float in [2e-4, 2.5e-3] {
            var sets: [[String]] = [], pairs: [Int] = []
            for clustered in [true, false] {
                var seed: UInt64 = 0x7A55_0C1C
                func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(seed >> 40) / Float(1 << 24) }
                func rq() -> simd_quatf { simd_quatf(angle: rnd() * 2 * .pi, axis: simd_normalize(SIMD3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) + 1e-3)) }
                // Points along the first jib (x 0…0.44, y 0.05…0.07, z ±7 mm), a little outside it too.
                func nearJib() -> SIMD3<Float> { SIMD3(rnd() * 0.46 - 0.01, 0.047 + rnd() * 0.026, (rnd() - 0.5) * 0.02) }
                var cols: [CoinStaticCollider] = [.plane(normal: SIMD3(0, 1, 0), offset: 0.05 - 0.0005 * rnd())]
                for _ in 0..<12 {
                    cols.append(.box(center: nearJib(), halfExtents: SIMD3(0.002 + rnd() * 0.01, 0.002 + rnd() * 0.004, 0.002 + rnd() * 0.01)))
                    cols.append(.orientedBox(center: nearJib(), halfExtents: SIMD3(0.002 + rnd() * 0.01, 0.001 + rnd() * 0.003, 0.002 + rnd() * 0.01),
                                             orientation: rq()))
                }
                cols.append(.cylinder(center: SIMD3(0.2, 0.06, 0), axis: SIMD3(1, 0, 0), radius: 0.012, up: SIMD3(0, 1, 0),
                                      halfLength: 0.05, lowerHalfOnly: false))
                let (s, _) = try makeSolver(maxCoins: 64, colliders: cols)
                s.speculativeMargin = spec
                s.compoundClustersForTesting = clustered
                let jibH = try XCTUnwrap(s.registerCompound(boxes: members.map(\.box)))
                let small = try XCTUnwrap(s.registerCompound(boxes: [
                    CoinCompoundBox(center: SIMD3(0.004, 0, 0), halfExtents: SIMD3(0.006, 0.0015, 0.002)),
                    CoinCompoundBox(center: SIMD3(-0.003, 0.003, 0), halfExtents: SIMD3(0.0015, 0.004, 0.002))]))
                let bar = try XCTUnwrap(s.registerHull(vertices: CoinDEMActuationTests.barHullPoints()))
                let ball = try XCTUnwrap(s.registerHull(vertices: ico.map { $0 * 0.006 }))
                _ = try XCTUnwrap(s.spawnCompound(design: Self.identity, SIMD3(0, 0.05, 0), compound: jibH, mass: 0.05))
                // A second jib crossing the first at 30°, through its middle.
                _ = try XCTUnwrap(s.spawnCompound(design: simd_quatf(angle: .pi / 6, axis: SIMD3(0, 1, 0)), SIMD3(0.12, 0.058, 0.09),
                                                  compound: jibH, mass: 0.05))
                for i in 0..<40 {
                    let p = nearJib(), q = rq(), q4 = SIMD4(q.imag, q.real)
                    switch i % 8 {
                    case 0: _ = s.spawnBox(at: p, halfExtents: SIMD3(0.003, 0.0015, 0.006), orient: q4, mass: 0.002)
                    case 1: _ = s.spawnHull(at: p, hull: bar, orient: q4, mass: CoinDEMActuationTests.barMass)
                    case 2: _ = s.spawnSphere(at: p, radius: 0.002 + rnd() * 0.003, mass: 0.001)
                    case 3: _ = s.spawnCapsule(at: p, radius: 0.0015, halfLength: 0.004, orient: q4, mass: 0.001)
                    case 4: _ = s.spawnEgg(at: p, fatRadius: 0.004, tipRadius: 0.0025, centerDistance: 0.004, orient: q4, mass: 0.001)
                    case 5: _ = s.spawn(at: p, orient: q4, radius: 0.005, halfThickness: 0.001, mass: 0.001)
                    case 6: _ = s.spawnHull(at: p, hull: ball, orient: q4, mass: 0.002)
                    default: _ = s.spawnCompound(design: q, p, compound: small, mass: 0.002)
                    }
                }
                s.generateContactsNow()
                XCTAssertLessThan(s.polyPairCount, s.maxPolyPairs, "precondition: the pair list did not overflow")
                XCTAssertLessThan(s.contactCount, s.maxContacts, "precondition: the contact buffer did not overflow")
                let p = s.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: s.maxContacts)
                sets.append((0..<s.contactCount).map { i -> String in
                    let c = p[i]
                    let fl: [Float] = [c.nrm.x, c.nrm.y, c.nrm.z, c.nrm.w, c.rA.x, c.rA.y, c.rA.z, c.rB.x, c.rB.y, c.rB.z]
                    return "\(c.meta.x) \(c.meta.y) \(c.meta.z) \(c.meta.w) " + fl.map { String($0.bitPattern, radix: 16) }.joined(separator: " ")
                }.sorted())
                pairs.append(s.polyPairCount)
            }
            let statics = sets[0].filter { $0.split(separator: " ")[1] == "4294967295" }.count
            print("B4t_clusters spec \(spec * 1000) mm: \(sets[0].count) contacts (\(statics) static) clustered vs \(sets[1].count) walked, identical \(sets[0] == sets[1]); listed pairs \(pairs[0]) clustered vs \(pairs[1]) walked")
            XCTAssertGreaterThan(sets[0].count, 200, "precondition: plenty of contacts on the jibs")
            XCTAssertGreaterThan(statics, 40, "precondition: plenty against statics")
            XCTAssertEqual(sets[0], sets[1], "clusters never change the contact set (spec \(spec))")
            XCTAssertLessThan(pairs[0], pairs[1], "…they only skip candidate pairs that could not touch")
        }
    }

    /// The clusters cull with the WHOLE speculative margin (verifier regression): a 32-cube row —
    /// principal frame = design frame, so each cluster's box is exactly its 16 cubes' union — with a
    /// static box, an oriented box, a plane and a sphere each 0.8 × the margin off a cluster's face.
    /// Every one is a speculative contact with every child walked, and must stay one with the
    /// clusters (a cull margin of half the speculative margin passed the jib test above — its
    /// clusters' boxes are loose — and lost the box, the oriented box and the sphere here; the plane has
    /// its own cull, which the jib test above pins).
    func testClusterCullKeepsContactsAtTheFullMargin() throws {
        let row = (0..<32).map { i in
            CoinCompoundBox(center: SIMD3(Float(i) * 0.005 - 0.0775, 0, 0), halfExtents: SIMD3(0.002, 0.0015, 0.001))
        }
        XCTAssertEqual(try XCTUnwrap(CoinCompoundMath.prepare(row)).clusters.count, 2, "precondition: two clusters of 16")
        for spec: Float in [2e-4, 2.5e-3] {
            let gap = 0.8 * spec, top: Float = 0.05 + 0.0015
            var counts: [[Int]] = []
            for clustered in [true, false] {
                let cols: [CoinStaticCollider] = [
                    .box(center: SIMD3(-0.06, top + gap + 0.002, 0), halfExtents: SIMD3(0.006, 0.002, 0.003)),
                    .orientedBox(center: SIMD3(0.05, top + gap + 0.002, 0), halfExtents: SIMD3(0.006, 0.002, 0.003),
                                 orientation: simd_quatf(angle: 0.3, axis: SIMD3(0, 1, 0))),
                    .plane(normal: SIMD3(0, 1, 0), offset: 0.05 - 0.0015 - gap),
                ]
                let (s, _) = try makeSolver(maxCoins: 8, colliders: cols)
                s.speculativeMargin = spec
                s.compoundClustersForTesting = clustered
                let h = try XCTUnwrap(s.registerCompound(boxes: row))
                let body = try XCTUnwrap(s.spawnCompound(design: Self.identity, SIMD3(0, 0.05, 0), compound: h, mass: 0.01))
                let ball = try XCTUnwrap(s.spawnSphere(at: SIMD3(0.0275, top + gap + 0.002, 0), radius: 0.002, mass: 0.001))
                s.generateContactsNow()
                let p = s.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: s.maxContacts)
                var c = [0, 0, 0, 0]
                for i in 0..<s.contactCount {
                    let m = p[i].meta
                    if Int(m.x) == body && m.y == 0xFFFF_FFFF, Int(m.z & 0xFFFF) < 3 { c[Int(m.z & 0xFFFF)] += 1 }
                    if (Int(m.x) == body && Int(m.y) == ball) || (Int(m.x) == ball && Int(m.y) == body) { c[3] += 1 }
                }
                counts.append(c)
            }
            print("B4t_clusterMargin spec \(spec * 1000) mm: contacts [box, oriented box, plane, sphere] clustered \(counts[0]) walked \(counts[1])")
            XCTAssertTrue(counts[1].allSatisfy { $0 > 0 }, "precondition: each is a speculative contact (spec \(spec))")
            XCTAssertEqual(counts[0], counts[1], "the clusters keep every contact within the margin (spec \(spec))")
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Rest, sleep, a bar on the chords, a cube through the panels
    // ══════════════════════════════════════════════════════════════════════════

    /// The jib on two pillars 50 mm up (its bottom chords on the pillar tops, its ballast
    /// block hanging clear), §G: (a) it settles and, kept awake, does not creep (≤ 1 µm, ≤ 0.001°
    /// from 0.5 s to 4 s), then sleeps once allowed to; (b) a 24 × 3 × 5.6 mm bar dropped across it from 3 mm lands
    /// on the TOP CHORDS — every one of its contacts is with a top-chord child (87 or 88, as
    /// the contacts' feature ids name them), it rests at the chords' top (within one contact
    /// slop) and everything sleeps again; (c) a 3 mm cube dropped over the middle of an open top
    /// panel falls through the truss's top and bottom openings without touching a strut and
    /// lands on the floor 50 mm below.
    func testTrussRestsCarriesABarAndLetsACubeThrough() throws {
        for smallWorld in [false, true] { try restCarryPass(smallWorld: smallWorld) }
    }

    private func restCarryPass(smallWorld: Bool) throws {
        let path = smallWorld ? "small-world" : "multi-dispatch"
        let pillarTop: Float = 0.05
        let cols: [CoinStaticCollider] = [
            .plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.15),
            .box(center: SIMD3(0.02, pillarTop / 2, 0), halfExtents: SIMD3(0.01, pillarTop / 2, 0.012)),
            .box(center: SIMD3(0.40, pillarTop / 2, 0), halfExtents: SIMD3(0.01, pillarTop / 2, 0.012)),
        ]
        let (s, q) = try makeSolver(colliders: cols, smallWorld: smallWorld)
        let members = Self.warrenJib()
        let h = try XCTUnwrap(s.registerCompound(boxes: members.map(\.box)))
        let lift: Float = 0.00005
        // The ballast block (x −20…0 mm) hangs off the root pillar's end (x 10…30 mm): clear of it.
        let body = try XCTUnwrap(s.spawnCompound(design: Self.identity, SIMD3(0, pillarTop + lift, 0), compound: h, mass: 0.05))
        func designY(_ slot: Int) -> Float { s.position(of: slot)!.y - h.comOffset.y }   // the jib's y = 0 plane, now

        // (a) Rest (sleep OFF, so the solver keeps stepping it: no creep), then sleep.
        s.sleepEnabled = false
        var p0 = SIMD3<Float>.zero, q0 = Self.identity
        step(s, q, frames: 240) { f in
            if f == 29 { p0 = s.position(of: body)!; q0 = s.orientation(of: body)! }
        }
        let drift = simd_length(s.position(of: body)! - p0), turn = Self.angleDeg(s.orientation(of: body)! * q0.inverse)
        let seat = designY(body) - pillarTop
        s.sleepEnabled = true
        var asleepAt = -1
        step(s, q, frames: 60) { f in if asleepAt < 0 && s.isAsleep(body) { asleepAt = f } }
        print("B4t_rest [\(path)] jib on two pillars: chords' underside \(seat * 1e6) µm off the pillar tops; awake drift 0.5→4 s \(drift * 1e6) µm / \(turn)°; asleep \(asleepAt) frames after sleep is allowed")
        XCTAssertLessThan(drift, 1e-6, "no creep")
        XCTAssertLessThan(turn, 1e-3, "no turn")
        XCTAssertGreaterThanOrEqual(seat, -1e-4 - 1e-5, "rests ON the pillars (within the contact slop)")
        XCTAssertLessThanOrEqual(seat, 1e-5, "not floating")
        XCTAssertGreaterThanOrEqual(asleepAt, 0, "the resting truss sleeps")

        // (b) A bar across the top chords (along z, over panel 9–10, away from any top batten).
        let chordTop = pillarTop + 0.020
        let barHalf = SIMD3<Float>(0.0028, 0.0015, 0.012)
        let xBar = Float(0.004 + 0.432 * 9.5 / 28)
        let bar = try XCTUnwrap(s.spawnBox(at: SIMD3(xBar, chordTop + barHalf.y + 0.003, 0), halfExtents: barHalf,
                                           mass: 0.002, friction: 0.3, restitution: 0.2))
        var barPieces = Set<Int>(), wokeAt = -1
        step(s, q, frames: 240) { f in
            if wokeAt < 0 && !s.isAsleep(body) { wokeAt = f }
            for c in self.pairContacts(s, body, bar) {
                barPieces.insert(Self.piece(c.meta.z, threadSide: Int(c.meta.x) == body))
            }
        }
        let barGap = s.position(of: bar)!.y - barHalf.y - (designY(body) + 0.020)
        let roles = barPieces.sorted().map { "\($0) \(members[$0].role)" }
        print("B4t_bar [\(path)] bar across the jib: rests \(barGap * 1e6) µm above the chords' top; touched children \(roles); jib woke @\(wokeAt); asleep again: jib \(s.isAsleep(body)), bar \(s.isAsleep(bar))")
        XCTAssertFalse(barPieces.isEmpty, "the bar touched the truss")
        XCTAssertTrue(barPieces.allSatisfy { members[$0].role.hasPrefix("chord.top") }, "…only its top chords: \(roles)")
        XCTAssertGreaterThanOrEqual(barGap, -1e-4 - 1e-5, "rests ON the chords (within the contact slop)")
        XCTAssertLessThanOrEqual(barGap, 1e-5, "not floating")
        XCTAssertTrue(s.isAsleep(bar) && s.isAsleep(body), "bar and truss settle and sleep")

        // (c) A cube over the middle of the open top panel between the top battens at k = 3 and
        //     k = 5 (x 51…81 mm) — it drops through the top opening, the interior and the bottom
        //     opening between the bottom battens at k = 4 and 6 onto the floor.
        let cubeHalf: Float = 0.0015
        let xCube = Float(0.004 + 0.432 * 4.5 / 28)
        let cube = try XCTUnwrap(s.spawnBox(at: SIMD3(xCube, chordTop + 0.01, 0), halfExtents: SIMD3(repeating: cubeHalf),
                                            mass: 0.0005, friction: 0.3, restitution: 0.1))
        var cubeTouchedTruss = 0
        step(s, q, frames: 180) { _ in cubeTouchedTruss += self.pairContacts(s, body, cube).count }
        let cubeY = s.position(of: cube)!.y
        print("B4t_cube [\(path)] 3 mm cube dropped over an open top panel: \(cubeTouchedTruss) contacts with the truss on the way; rests at y \(cubeY * 1000) mm (floor + \((cubeY - cubeHalf) * 1e6) µm)")
        XCTAssertEqual(cubeTouchedTruss, 0, "the cube passes through the openings without touching a strut")
        XCTAssertLessThan(abs(cubeY - cubeHalf), 1.2e-4, "…and lands on the floor below")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Cost (measured, printed)
    // ══════════════════════════════════════════════════════════════════════════

    /// The jib awake on its pillars with eight bars raining onto it, against the same jib as
    /// two envelope boxes (solid where the truss is open): physics GPU ms per 1/60 s frame.
    /// A measurement, not a gate (GPU time moves with whatever else the machine renders):
    ///   Run:  VIZ_COINDEM_TRUSS_COST=1 ./Scripts/test.sh --filter testTrussCompoundCost
    func testTrussCompoundCost() throws {
        guard ProcessInfo.processInfo.environment["VIZ_COINDEM_TRUSS_COST"] != nil else {
            throw XCTSkip("a measurement; set VIZ_COINDEM_TRUSS_COST=1 to run it")
        }
        func run(_ boxes: [CoinCompoundBox], label: String) throws -> (p50: Double, p95: Double) {
            let cols: [CoinStaticCollider] = [
                .plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.15),
                .box(center: SIMD3(0.02, 0.025, 0), halfExtents: SIMD3(0.01, 0.025, 0.012)),
                .box(center: SIMD3(0.40, 0.025, 0), halfExtents: SIMD3(0.01, 0.025, 0.012)),
            ]
            let (s, q) = try makeSolver(maxCoins: 16, colliders: cols, sleep: false)
            let h = try XCTUnwrap(s.registerCompound(boxes: boxes))
            _ = try XCTUnwrap(s.spawnCompound(design: Self.identity, SIMD3(0, 0.05005, 0), compound: h, mass: 0.05))
            var ms: [Double] = []
            for i in 0..<8 {
                _ = s.spawnBox(at: SIMD3(0.05 + 0.045 * Float(i), 0.08, Float(i % 3 - 1) * 0.002), halfExtents: SIMD3(0.0028, 0.0015, 0.012),
                               mass: 0.002, friction: 0.3, restitution: 0.2)
                ms += step(s, q, frames: 15)
            }
            ms += step(s, q, frames: 120)
            let w = Array(ms.dropFirst(10)).sorted()
            let r = (w[w.count / 2], w[min(w.count - 1, w.count * 95 / 100)])
            print("B4t_cost \(label): physics GPU per 1/60 s p50 \(String(format: "%.3f", r.0)) ms, p95 \(String(format: "%.3f", r.1)) ms")
            return r
        }
        let exact = Self.warrenJib().map(\.box)
        let envelope: [CoinCompoundBox] = [
            CoinCompoundBox(center: SIMD3(0.22, 0.010, 0), halfExtents: SIMD3(0.22, 0.010, 0.0065), density: 0.3),
            CoinCompoundBox(center: SIMD3(-0.010, 0.010, 0), halfExtents: SIMD3(0.010, 0.010, 0.010), density: 2),
        ]
        let e = try run(envelope, label: "envelope (2 boxes)")
        let x = try run(exact, label: "open truss (90 boxes)")
        print("B4t_cost SUMMARY envelope p50 \(e.p50) p95 \(e.p95) | truss p50 \(x.p50) p95 \(x.p95)")
    }
}
