import Foundation
import simd

// plausibility: real — glass panes, pillars, frames, roof (cambered) and drip rails are authored closed solids
// extruded off the tumbled cabin surface; every pane is a thin closed shell (no textures, no CSG).

/// Body class as the greenhouse sees it. `CarContext` does not carry the `CarStyle`, so it is recovered from the
/// blueprint (length + cabin width fraction are unique per class — see `CarBlueprint.make`).
enum GHKind: Equatable { case sedan, hatch, suv, pickup, van, coupe, bus }

/// A line in the side view, `z = z0 + k*(y - y0)` (k = dz/dy).
struct GHLine {
    var z0: Double, y0: Double, k: Double
    func z(_ y: Double) -> Double { z0 + k * (y - y0) }
    func shifted(_ dz: Double) -> GHLine { GHLine(z0: z0 + dz, y0: y0, k: k) }
}

/// One named glass pane (right-hand side meshes are followed by their mirrored left twin).
struct GHNamed { var name: String; var mesh: Mesh3 }

/// Everything the greenhouse step produces, kept apart so tests (and the interior) can inspect each piece.
///
/// **`a.glass` append order** (each entry is its own closed shell; none are merged):
/// `windscreen`, then (classic only) `wrap` corner panes R,L, `door-front` R,L, `door-rear` R,L (4-door),
/// `quarter` R,L, `rear` (rear / tailgate glass), `sunroof`. Front = clear, side/rear = privacy group, so a host
/// that wants a darker rear group can split `a.glass` by the shell index with no geometry change.
struct GHPieces {
    var panes: [GHNamed] = []         // → a.glass
    var frames: [GHNamed] = []        // → a.trim  (weatherstrip ring per pane, same names)
    var chromeFrames: [GHNamed] = []  // → a.chrome (classic window surrounds)
    var pillarsBody: [GHNamed] = []   // → a.body
    var pillarsTrim: [GHNamed] = []   // → a.trim  (B-pillar black-out)
    var pillarCaps: [GHNamed] = []    // → a.chrome (classic chrome-capped pillars)
    var roof = Mesh3()                // → a.body
    var dripRail = Mesh3()            // → a.trim
    var sill = Mesh3()                // → a.chrome
    var sunroofGlass = Mesh3()        // → a.glass
    var sunroofFrame = Mesh3()        // → a.trim
}

struct GHModel {
    let c: CarContext
    let kind: GHKind
    let yB: Double, roofY: Double
    let zR: Double, z2: Double, z1: Double, zF: Double
    let yR: Double                       // y of the rear base of the cabin
    let rearVertical: Bool
    let roofFrac: Double
    let crown: Double

    init(_ c: CarContext) {
        self.c = c
        let cp = c.cabinPoly.count >= 4 ? c.cabinPoly : [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        zR = cp[0].0; z2 = cp[1].0; z1 = cp[2].0; zF = cp[3].0
        yR = cp[0].1; yB = cp[3].1
        roofY = cp.map(\.1).max() ?? c.H
        rearVertical = (roofY - cp[0].1) < 0.15
        let frac = c.bp.cabinWidthFrac + (c.era == .classic1950s ? 0.04 : 0)
        let len = c.bp.length
        if c.bp.boxy { kind = .bus }
        else if len > 5.5 { kind = .pickup }
        else if frac > 0.91 { kind = .van }
        else if frac > 0.89 { kind = .suv }
        else if abs(frac - 0.84) < 0.005 { kind = .coupe }
        else if len < 4.5 { kind = .hatch }
        else { kind = .sedan }
        var rf: Double
        switch kind {
        case .sedan: rf = 0.80
        case .hatch: rf = 0.81
        case .suv, .pickup: rf = 0.84
        case .van: rf = 0.86
        case .coupe: rf = 0.78
        case .bus: rf = 1.0
        }
        if c.era == .classic1950s { rf -= 0.045 }
        roofFrac = min(rf, c.bp.cabinWidthFrac - 0.05)
        crown = c.era == .classic1950s ? 0.030 : (c.era == .nineties ? 0.025 : 0.022)
    }

    // MARK: era / class dimensions
    var era: CarEra { c.era }
    var wA: Double { (era == .classic1950s ? 0.055 : era == .nineties ? 0.10 : 0.115) * ([.suv, .pickup, .van].contains(kind) ? 1.1 : 1.0) }
    var wB: Double { era == .classic1950s ? 0.05 : era == .nineties ? 0.085 : 0.075 }
    var wC: Double {
        switch kind {
        case .pickup: return 0.10
        case .suv: return era == .classic1950s ? 0.12 : 0.20
        case .coupe: return era == .classic1950s ? 0.16 : 0.22
        default: return era == .classic1950s ? 0.12 : 0.17
        }
    }
    var wDiv: Double { era == .classic1950s ? 0.06 : 0.09 }
    var frameW: Double { kind == .coupe && era == .modern ? 0.008 : (era == .classic1950s ? 0.012 : era == .nineties ? 0.014 : 0.016) }
    var sillH: Double { era == .classic1950s ? 0.040 : era == .nineties ? 0.018 : 0.014 }
    var tumble: Double { 0 }

    // MARK: cabin surface
    func plan(_ z: Double) -> Double {
        guard z < z2, !rearVertical else { return 1 }
        let u = min(1, (z2 - z) / max(z2 - zR, 1e-6))
        return 1 - max(0.04, (1 - c.bp.tailTaper) * 0.9) * pow(u, 1.5)
    }
    func hwBase(_ z: Double) -> Double { c.W / 2 * c.bp.cabinWidthFrac * plan(z) }
    /// Half-width of the roof panel at z.
    func hwRoofPanel(_ z: Double) -> Double { c.W / 2 * roofFrac * plan(z) }
    /// Half-width of the (tumbled) side surface at height y.
    func sideHW(_ z: Double, _ y: Double) -> Double {
        let top = hwRoofPanel(z) - 0.012, yTopRef = roofY - 0.04
        let t = (y - yB) / max(yTopRef - yB, 1e-6)
        return hwBase(z) + (top - hwBase(z)) * t
    }
    /// Outer cabin line height at z (the profile polygon).
    func yTop(_ z: Double) -> Double { c.interp(c.cabinPoly, z) }
    /// The three profile segments as (z0, y0, slope dy/dz).
    var segs: [(z0: Double, y0: Double, m: Double, flat: Bool)] {
        let cp = c.cabinPoly
        return (0..<3).map { i in
            let a = cp[i], b = cp[i + 1]
            let m = (b.1 - a.1) / max(b.0 - a.0, 1e-9)
            return (a.0, a.1, m, abs(m) < 1e-6)
        }
    }

    var lineA: GHLine { GHLine(z0: zF, y0: yF, k: (z1 - zF) / max(roofY - yF, 1e-6)) }
    var yF: Double { yB }
    var lineC: GHLine {
        rearVertical ? GHLine(z0: zR, y0: yR, k: 0) : GHLine(z0: zR, y0: yR, k: (z2 - zR) / max(roofY - yR, 1e-6))
    }

    /// Convex side-view region between a front and a rear line, above `yLo`, under the cabin line
    /// (lowered by `drip`; the flat roof by `drip + crown` if `underCamber`).
    func region(front: GHLine?, rear: GHLine?, yLo: Double, drip: Double, underCamber: Bool = false, round r: Double = 0.02) -> [Vec2] {
        var p: [Vec2] = [Vec2(zR - 1, yLo), Vec2(zF + 1, yLo), Vec2(zF + 1, roofY + 1), Vec2(zR - 1, roofY + 1)]
        if let f = front { p = GHKit.clipLine(p, x0: f.z0, k: f.k, y0: f.y0, keepLess: true) }
        if let b = rear { p = GHKit.clipLine(p, x0: b.z0, k: b.k, y0: b.y0, keepLess: false) }
        for s in segs {
            let lower = drip + (s.flat && underCamber ? crown + 0.004 : 0)
            // y <= y0 + m (z - z0) - lower
            p = GHKit.clip(p, a: -s.m, b: 1, c: s.y0 - s.m * s.z0 - lower)
        }
        return r > 0 ? GHKit.round(p, r) : p
    }

    func polyArea(_ p: [Vec2]) -> Double { abs(GHKit.area(p)) }

    var sideSurface: GHSurface {
        GHSurface { q in   // q = (z, y)
            func P(_ z: Double, _ y: Double) -> Vec3 { Vec3(sideHW(z, y), y, z) }
            let e = 1e-3
            let tz = P(q.x + e, q.y) - P(q.x - e, q.y), ty = P(q.x, q.y + e) - P(q.x, q.y - e)
            return (P(q.x, q.y), normalize3(cross3(ty, tz)))
        }
    }

    /// Windscreen plane: (x, s) with s metres up the rake from the base.
    var wsLen: Double { hypot(zF - z1, roofY - yB) }
    var windscreenSurface: GHSurface {
        let L = wsLen, dz = (z1 - zF) / L, dy = (roofY - yB) / L
        let n = normalize3(Vec3(0, -dz, dy))
        return GHSurface { q in (Vec3(q.x, yB + dy * q.y, zF + dz * q.y), n) }
    }
    var rearLen: Double { hypot(z2 - zR, roofY - yR) }
    var rearSurface: GHSurface {
        let L = rearLen, dz = (z2 - zR) / L, dy = (roofY - yR) / L
        let n = normalize3(Vec3(0, dz, -dy))
        return GHSurface { q in (Vec3(q.x, yR + dy * q.y, zR + dz * q.y), n) }
    }

    // MARK: roof
    func roofTop(_ x: Double, _ z: Double) -> Double {
        let hw = max(hwRoofPanel(z), 0.1)
        var y = yTop(z)
        var camber = 1.0
        if z > z1 { let t = min(1, (z - z1) / 0.03); y += 0.010 * min(1, (z - z1) / 0.02); camber = 1 - t }
        else if z < z2 && !rearVertical { let t = min(1, (z2 - z) / 0.03); y += 0.010 * min(1, (z2 - z) / 0.02); camber = 1 - t }
        let u = min(1, abs(x) / hw)
        return y - crown * camber * u * u      // the header lips run flat across, above the glass plane
    }
    var roofRange: (Double, Double) { (rearVertical ? zR : z2 - 0.06, z1 + 0.06) }

    // MARK: layout
    var fourDoor: Bool { [.sedan, .hatch, .suv, .pickup].contains(kind) }
    var hasQuarterWanted: Bool { [.sedan, .hatch, .suv, .coupe].contains(kind) }
    var hasRearGlass: Bool { kind != .van }
    var hasSunroof: Bool { era == .modern && (kind == .sedan || kind == .suv) && !c.bp.roofSign }   // no sunroof under the taxi roof sign
    var zB: Double { z1 - 0.47 * (z1 - z2) }
    var zRD: Double { z2 + 0.05 * (z1 - z2) }
    var zDoorEndCoupe: Double { z2 + 0.30 * (z1 - z2) }
    var lineBf: GHLine { GHLine(z0: zB + wB / 2, y0: yB, k: -0.05) }
    var lineBr: GHLine { GHLine(z0: zB - wB / 2, y0: yB, k: -0.05) }
    func vline(_ z: Double) -> GHLine { GHLine(z0: z, y0: yB, k: 0) }
    var lineCf: GHLine { rearVertical ? vline(zR + 0.06) : lineC.shifted(wC) }
}

/// Tint group of a glass shell: the windscreen and front-door glass are clear, the rest is the rear / privacy group.
public enum GlassGroup: Sendable, Equatable { case windscreen, front, rear, sunroof }

/// Records, per appended glass shell, its name / group / triangle range inside `a.glass`.
struct GHGlassRecorder {
    var shells: [(name: String, group: GlassGroup, triangleRange: Range<Int>)] = []
    mutating func add(_ a: inout CarAccumulator, _ name: String, _ group: GlassGroup, _ mesh: Mesh3) {
        guard !mesh.isEmpty else { return }
        let s = a.glass.triangleCount
        a.glass.append(mesh)
        shells.append((name, group, s..<(s + mesh.triangleCount)))
    }
    static func group(for name: String) -> GlassGroup {
        if name.hasPrefix("windscreen") { return .windscreen }
        if name.hasPrefix("door-front") { return .front }
        if name.hasPrefix("sunroof") { return .sunroof }
        return .rear
    }
}

public extension CarFactory {
    /// The glass shells of `spec` in `CarParts.glass` order: which closed shell (by triangle range) is the windscreen,
    /// front-door glass, rear / privacy glass (rear doors, quarter, rear / tailgate window) or sunroof, so a host can
    /// tint the rear group darker. Ranges are exact, contiguous and disjoint, and cover the whole `glass` mesh.
    /// Bus: windscreen halves = `.windscreen`, the front half of the side windows = `.front`, the rest + rear window = `.rear`.
    static func glassShellMap(_ spec: CarSpec) -> [(name: String, group: GlassGroup, triangleRange: Range<Int>)] {
        let c = CarContext.lampContext(spec)
        var a = CarAccumulator(), rec = GHGlassRecorder()
        c.buildGreenhouse(&a, &rec)
        return rec.shells
    }
}

extension CarContext {
    /// Floor height of the interior: the underside of the seat cushions. Everything in `a.interior` is at or above
    /// this y; a cabin tub in the body shell must have its floor at (or just below) this value to show the seats.
    func interiorFloorY(_ m: GHModel) -> Double { GHInteriorLayout.floorY(m) }

    func buildGreenhouse(_ a: inout CarAccumulator) { var r = GHGlassRecorder(); buildGreenhouse(&a, &r) }

    /// Glass panes, frames, roof panel, pillars, interior. Appends to `a.glass`, `a.body`, `a.interior`, `a.trim`, `a.chrome`.
    func buildGreenhouse(_ a: inout CarAccumulator, _ rec: inout GHGlassRecorder) {
        if bp.boxy { buildBusGreenhouse(&a, &rec); return }
        let m = GHModel(self)
        let g = greenhousePieces(m)
        // glass — documented order (see GHPieces)
        for p in g.panes { rec.add(&a, p.name, GHGlassRecorder.group(for: p.name), p.mesh) }
        rec.add(&a, "sunroof", .sunroof, g.sunroofGlass)
        for f in g.frames { a.trim.append(f.mesh) }
        for f in g.chromeFrames { a.chrome.append(f.mesh) }
        for p in g.pillarsBody { a.body.append(p.mesh) }
        for p in g.pillarsTrim { a.trim.append(p.mesh) }
        for p in g.pillarCaps { a.chrome.append(p.mesh) }
        a.body.append(g.roof)
        a.trim.append(g.dripRail)
        a.trim.append(g.sunroofFrame)
        a.chrome.append(g.sill)
        buildInterior(&a, m)
    }
}
