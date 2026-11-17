import Foundation
import simd

// plausibility: real — panel details are authored solids placed on the real shell surface (ray-cast onto the finished
// `a.body`), not textures: door cuts / shut lines are 4–5 mm dark strips proud of the paint, handles and mirrors are
// bevelled solids, sills are conforming patches.

/// Which mesh a detail piece belongs to.
enum DTarget { case body, trim, chrome }

/// One named detail solid (right-hand side pieces are followed by their mirrored left twin).
struct DPiece { var name: String; var mesh: Mesh3; var target: DTarget }

/// Where the doors are on the flank (z of the cuts, bottom line).
struct DoorLayout {
    var zFront: Double
    var zShut: Double?       // B-pillar shut (4-door) or the sliding-door shut (van)
    var zRear: Double
    var yBot: Double
    var shutLean: Double
}

/// Everything a detail step needs: the context, the greenhouse model and the surface prober.
struct DetailEnv {
    let c: CarContext
    let m: GHModel
    let probe: BodyProbe
    var kind: GHKind { m.kind }
    var era: CarEra { c.era }
    var classic: Bool { c.era == .classic1950s }
    /// Class sizing for handles/mirrors: SUV, pickup, van are chunkier.
    var big: Bool { [.suv, .pickup, .van].contains(kind) }

    /// Inner edge (|x|) of the corner tail lamps, if any (the tailgate shut runs just inboard of them).
    let lampInner: Double?

    init(_ c: CarContext, body: Mesh3, tailLights: Mesh3 = Mesh3()) {
        self.c = c; m = GHModel(c); probe = BodyProbe(body)
        let xs = tailLights.positions.filter { $0.x > 0.3 && $0.y < c.shellDeck(c.rearZ) }.map(\.x)
        lampInner = xs.min()
    }

    // ── surface queries ───────────────────────────────────────────────
    /// Outermost flank point at (z, y) on the right side.
    func flank(_ z: Double, _ y: Double) -> DSurf? { probe.cast(Vec3(c.W, y, z), Vec3(-1, 0, 0)) }
    /// Top surface at (x, z).
    func down(_ x: Double, _ z: Double) -> DSurf? { probe.cast(Vec3(x, 4, z), Vec3(0, -1, 0)) }
    /// Rear-face surface at (x, y).
    func rear(_ x: Double, _ y: Double) -> DSurf? { probe.cast(Vec3(x, y, c.rearZ - 2), Vec3(0, 0, 1)) }
    /// Shoulder radius point at `z` for angle `phi` (0 = flank, π/2 = top), cast outward from the arc centre.
    func shoulder(_ z: Double, _ phi: Double) -> DSurf? {
        let (pv, _) = pivot(z)
        return probe.cast(pv, Vec3(cos(phi), sin(phi), 0), fromInside: true)
    }
    func pivot(_ z: Double) -> (Vec3, Double) {
        let h = c.shellHalf(z)
        let cT = h[7].x - h[4].x
        return (Vec3(h[4].x, h[4].y - cT, z), cT)
    }
    /// Height where the flank's vertical part starts (top of the door skin) at z.
    func flankTop(_ z: Double) -> Double { pivot(z).0.y }
    /// Height of the character line at z (handle height).
    func creaseY(_ z: Double) -> Double { c.shellHalf(z)[9].y }

    // ── layout ────────────────────────────────────────────────────────
    var openingR: Double { c.shellOpeningR + c.shellToe }
    var archFrontRear: Double { c.axles[0] - openingR }    // rear edge of the front wheel opening (+toe)
    var archRearFront: Double { c.axles[1] + openingR }    // front edge of the rear wheel opening (+toe)

    var doorLayout: DoorLayout? {
        guard kind != .bus else { return nil }
        let yBot = c.bp.sill + (c.shellStyle.cladding ? 0.19 : 0.16)
        let zF = min(m.zF - 0.02, archFrontRear - 0.06)
        var zS: Double? = nil
        var zR: Double
        switch kind {
        case .sedan, .hatch, .suv:
            zS = m.zB
            zR = m.hasQuarterWanted ? m.zRD - m.wDiv / 2 : m.lineCf.z(m.yB) - 0.02
        case .pickup:
            zS = m.zB
            zR = m.lineCf.z(m.yB) - 0.02
        case .coupe:
            zR = m.zDoorEndCoupe - m.wDiv / 2
        case .van:
            zS = zF - 0.98
            zR = max(archRearFront + 0.05, zS! - 1.15)
        case .bus: return nil
        }
        zR = max(zR, c.axles[1] + 0.12)
        return DoorLayout(zFront: zF, zShut: zS, zRear: zR, yBot: yBot, shutLean: -0.05)
    }

    // ── path helpers ──────────────────────────────────────────────────
    /// Samples a vertical-ish cut at z0 (leaning `lean` dz/dy) from `yFrom` down to `yTo` (outside-in flank casts).
    func flankLine(z0: Double, lean: Double, yFrom: Double, yTo: Double, step: Double = 0.05) -> [DSurf] {
        var out: [DSurf] = []
        let n = max(1, Int(((yFrom - yTo) / step).rounded(.up)))
        for i in 0...n {
            let y = yFrom + (yTo - yFrom) * Double(i) / Double(n)
            if let s = flank(z0 + lean * (y - yFrom), y) { out.append(s) }
        }
        return out
    }
    func shoulderArc(_ z: Double, from: Double, to: Double, n: Int = 4) -> [DSurf] {
        (0...n).compactMap { shoulder(z, from + (to - from) * Double($0) / Double(n)) }
    }
}

// MARK: - build

extension CarContext {

    /// Panel-level details on the finished shell (runs after Body, Greenhouse and Lamps; reads their meshes in `a`).
    func buildBodyDetails(_ a: inout CarAccumulator) {
        for p in bodyDetailPieces(a) {
            switch p.target {
            case .body: a.body.append(p.mesh)
            case .trim: a.trim.append(p.mesh)
            case .chrome: a.chrome.append(p.mesh)
            }
        }
    }

    /// All detail solids (named; used by the tests).
    func bodyDetailPieces(_ a: CarAccumulator) -> [DPiece] {
        let e = DetailEnv(self, body: a.body, tailLights: a.tailLights)
        var out: [DPiece] = []
        func both(_ name: String, _ m: Mesh3, _ t: DTarget) {
            guard !m.isEmpty else { return }
            out.append(DPiece(name: name + ".R", mesh: m, target: t))
            out.append(DPiece(name: name + ".L", mesh: m.mirrored(.x), target: t))
        }
        func right(_ name: String, _ m: Mesh3, _ t: DTarget) { if !m.isEmpty { out.append(DPiece(name: name, mesh: m, target: t)) } }
        func centre(_ name: String, _ m: Mesh3, _ t: DTarget) { right(name, m, t) }

        // 11 / 15 / 16 — doors, sills, handles (not for the bus: the class cues own its doors)
        if let lay = e.doorLayout {
            if e.kind == .van {   // one sliding door only (right-hand): the left side keeps just the front-door U
                right("doorcut.R", e.doorCuts(lay), .trim)
                right("doorcut.L", e.doorCuts(lay, includeShut: false).mirrored(.x), .trim)
            } else { both("doorcut", e.doorCuts(lay), .trim) }
            for (n, m) in e.sillPieces(lay) { both(n, m.mesh, m.target) }
            for (n, m) in e.handlePieces(lay) { both(n, m.mesh, m.target) }
        }
        // 12 — hood / trunk / tailgate shut lines, 1950s ornament and side trim
        centre("shut.hood", e.hoodShutLine(), .trim)
        centre("shut.trunk", e.trunkShutLine(), .trim)
        centre("shut.tailgate", e.tailgateShutLine(), .trim)
        if e.classic {
            centre("ornament.hood", e.hoodOrnament(), .chrome)
            if let lay = e.doorLayout { both("trim.spear", e.sideSpear(lay), .chrome) }
        }
        // 13 — fuel flap, left (−X) rear quarter only
        let flap = e.fuelFlap()
        if !flap.groove.isEmpty {
            out.append(DPiece(name: "flap.groove.L", mesh: flap.groove.mirrored(.x), target: .trim))
            out.append(DPiece(name: "flap.hinge.L", mesh: flap.hinge.mirrored(.x), target: .trim))
            out.append(DPiece(name: "flap.cap.L", mesh: flap.cap.mirrored(.x), target: .trim))
        }
        // 17 — mirrors
        for (n, mm) in e.mirrorPieces() { both(n, mm.mesh, mm.target) }
        // 19 — wipers + antenna
        for (n, mm) in e.wiperPieces() { centre(n, mm.mesh, mm.target) }
        for (n, mm) in e.antennaPieces() { centre(n, mm.mesh, mm.target) }
        return out
    }
}

// MARK: - 11 door cuts

extension DetailEnv {
    static let cutWidth = 0.0045

    /// Door cut lines: a U (front cut → lower sill line → rear cut) plus the B / slider shut, on the right side.
    func doorCuts(_ lay: DoorLayout, includeShut: Bool = true) -> Mesh3 {
        var mesh = Mesh3()
        let r = 0.04
        // U path in (z, y) below the shoulder.
        var uz: [(Double, Double)] = []
        let yTopF = flankTop(lay.zFront) - 0.005, yTopR = flankTop(lay.zRear) - 0.005
        func line(_ z0: Double, _ y0: Double, _ y1: Double) { let n = max(1, Int((abs(y0 - y1) / 0.05).rounded(.up))); for i in 0...n { uz.append((z0, y0 + (y1 - y0) * Double(i) / Double(n))) } }
        line(lay.zFront, yTopF, lay.yBot + r)
        for k in 1...5 { let a = Double(k) / 5 * .pi / 2; uz.append((lay.zFront - r + r * cos(a), lay.yBot + r - r * sin(a))) }
        // the lower line follows the sill, then climbs the rear wheel opening (door skin wraps over the arch)
        func yAt(_ z: Double) -> Double { max(lay.yBot, c.shellArchEdge(z) + 0.05) }
        let yEnd = yAt(lay.zRear)
        let nh = max(2, Int(((lay.zFront - lay.zRear - 2 * r) / 0.05).rounded(.up)))
        for i in 1..<nh { let z = lay.zFront - r - (lay.zFront - lay.zRear - 2 * r) * Double(i) / Double(nh); uz.append((z, min(yAt(z), yEnd))) }
        for k in 0...5 { let a = Double(k) / 5 * .pi / 2; uz.append((lay.zRear + r * cos(a), yEnd + r * sin(a))) }
        line(lay.zRear, yEnd + r, yTopR)
        var path = shoulderArc(lay.zFront, from: 1.0, to: 0.12, n: 4)
        path += uz.compactMap { flank($0.0, $0.1) }
        path += shoulderArc(lay.zRear, from: 0.12, to: 1.0, n: 4)
        mesh.append(DKit.strip(path, width: Self.cutWidth, embed: 0.002, proud: 0.0011))

        if includeShut, let zs = lay.zShut {
            // The shut runs up to the shoulder and down to the sill line, leaning with the B-pillar.
            let top = flankTop(zs) - 0.005
            var p = shoulderArc(zs, from: 1.0, to: 0.12, n: 4)
            p += flankLine(z0: zs, lean: lay.shutLean, yFrom: top, yTo: lay.yBot)
            mesh.append(DKit.strip(p, width: Self.cutWidth, embed: 0.002, proud: 0.0015))
        }
        return mesh
    }
}
