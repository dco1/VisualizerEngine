import Foundation
import simd

// plausibility: real — lamp/bumper solids are lofted along the body's own plan outline (planFactor) at the
// real end-of-car stations, so they sit on / are embedded into the shell; no textures stand in for shape.

// MARK: - Lamp-shell bookkeeping (public: lets a host drive each lamp function independently)

/// What a lamp-pass shell *is*. See `CarContext.buildLamps` for the mesh → function mapping.
public enum CarLampFunction: String, Sendable, CaseIterable {
    case headlampLens, projectorLens, headlampHousing, headlampReflector, drl, frontIndicator, sideRepeater
    case fogLamp, fogHousing
    case brakeLamp, rearIndicator, reverseLamp, tailHousing, thirdBrake
    case grille, badge, bumper, bumperLip, overrider, exhaustTip, plateFrame, plate, destinationSign
}

/// Which `CarParts` mesh a shell lives in.
public enum CarLampMesh: String, Sendable {
    case headlights, tailLights, signals, trim, chrome, plates, body
}

/// One closed shell emitted by the lamp pass. `triangles` indexes the shell's triangles inside its mesh, relative to
/// the lamp pass's own appends: it is exact for `headlights`, `tailLights`, `signals` and `plates` (only the lamp pass
/// writes them) and an ordering aid only for the shared `trim` / `chrome` / `body`.
public struct CarLampShell: Sendable, Equatable {
    public var mesh: CarLampMesh
    public var function: CarLampFunction
    /// +1 right, -1 left, 0 centred.
    public var side: Int
    public var triangles: Range<Int>
}

struct LampRecorder {
    var shells: [CarLampShell] = []
}

extension CarLampMesh {
    var keyPath: WritableKeyPath<CarAccumulator, Mesh3> {
        switch self {
        case .headlights: return \.headlights
        case .tailLights: return \.tailLights
        case .signals: return \.signals
        case .trim: return \.trim
        case .chrome: return \.chrome
        case .plates: return \.plates
        case .body: return \.body
        }
    }
}

// MARK: - End-of-car frame

/// The front or rear end of the body, measured as a distance `d` inward from the end plane.
struct LampEnd {
    let front: Bool
    let z0: Double
    /// +1 pointing out of the car along z at this end (front) / -1 (rear).
    let dir: Double
}

struct LStation {
    var x: Double, z: Double
    var n: Vec2           // outward plan normal (x, z)
    var m: Vec2 = .zero   // mitre vector: offset curve at distance t is `P - m * t` with no folds at corners
    var u: Double         // arc length from the path start
    var onFace: Bool
}

struct LampPath {
    var s: [LStation]
    /// Mirror about x = 0 into one continuous left-to-right path. Requires the first station at x = 0.
    func full() -> LampPath {
        var left: [LStation] = []
        for st in s.dropFirst().reversed() { var m = st; m.x = -m.x; m.n = Vec2(-m.n.x, m.n.y); m.m = Vec2(-m.m.x, m.m.y); left.append(m) }
        return LampPath(s: left + s)
    }
}

struct LampSection { var y0: Double, y1: Double, out: Double, inn: Double }

extension CarContext {
    func lampEnd(_ front: Bool) -> LampEnd { LampEnd(front: front, z0: front ? frontZ : rearZ, dir: front ? 1 : -1) }
    func zAt(_ e: LampEnd, _ d: Double) -> Double { e.z0 - e.dir * d }
    func hwAt(_ e: LampEnd, _ d: Double) -> Double { W / 2 * planFactor(zAt(e, d)) }
    func topAt(_ e: LampEnd, _ d: Double) -> Double { interp(poly(bp.top), zAt(e, d)) }
    func botAt(_ e: LampEnd, _ d: Double) -> Double { min(bottom(zAt(e, d)), topAt(e, d) - 0.12) }
    /// Corner radius of the body cross-section at distance `d` (mirrors `CarContext.profile`).
    func cornerAt(_ e: LampEnd, _ d: Double) -> Double {
        let hh = (topAt(e, d) - botAt(e, d)) / 2
        return max(0.004, min(bp.cornerRadius, hh * 0.9, hwAt(e, d) * 0.5))
    }
    /// Straight (un-rounded) wall band of the shell at distance `d` from the end.
    func wallBand(_ e: LampEnd, _ d: Double) -> (lo: Double, hi: Double) {
        let c = cornerAt(e, d)
        return (botAt(e, d) + c, topAt(e, d) - c)
    }

    /// Plan path hugging the body's outline on the +x side. `faceX0` starts on the end face (0 = centreline);
    /// the face runs to `faceX1` (default: the face corner); a flank run `flankD0…flankD1` (metres inward) follows.
    /// The corner between face and flank is swept through intermediate normals (a rounded wrap).
    func lampPath(_ e: LampEnd, faceX0: Double?, faceX1: Double? = nil, flankD0: Double = 0, flankD1: Double? = nil) -> LampPath {
        var st: [LStation] = []
        let hwN = hwAt(e, 0)
        func rot(_ t: Vec2) -> Vec2 { e.dir > 0 ? Vec2(-t.y, t.x) : Vec2(t.y, -t.x) }
        let nFace = Vec2(0, e.dir)
        var cornerIn = false
        if let x0 = faceX0 {
            let x1 = min(faceX1 ?? hwN, hwN)
            let n = max(1, Int(((x1 - x0) / 0.09).rounded(.up)))
            for i in 0...n { st.append(LStation(x: x0 + (x1 - x0) * Double(i) / Double(n), z: e.z0, n: nFace, m: nFace, u: 0, onFace: true)) }
            cornerIn = (faceX1 ?? hwN) >= hwN - 1e-9 && flankD1 != nil
        }
        var flank: [LStation] = []
        if let d1 = flankD1, d1 > flankD0 + 1e-6 {
            let n = max(1, Int(((d1 - flankD0) / 0.09).rounded(.up)))
            for j in 0...n {
                let d = flankD0 + (d1 - flankD0) * Double(j) / Double(n)
                flank.append(LStation(x: hwAt(e, d), z: zAt(e, d), n: .zero, m: .zero, u: 0, onFace: false))
            }
            func segN(_ i: Int) -> Vec2 {
                let t = Vec2(flank[i + 1].x - flank[i].x, flank[i + 1].z - flank[i].z)
                return rot(simd_normalize(t))
            }
            for j in 0..<flank.count {
                let a = j > 0 ? segN(j - 1) : segN(0), b = j < flank.count - 1 ? segN(j) : segN(flank.count - 2)
                flank[j].n = simd_normalize(a + b)
                flank[j].m = (a + b) / (1 + simd_dot(a, b))
            }
            flank[0].m = flank[0].n; flank[flank.count - 1].m = flank[flank.count - 1].n
        }
        if cornerIn, let f0 = flank.first {
            // Sweep the normal from the face's to the flank's about the corner vertex.
            let a0 = atan2(nFace.y, nFace.x), a1 = atan2(f0.n.y, f0.n.x)
            var da = a1 - a0
            while da > .pi { da -= 2 * .pi }
            while da < -.pi { da += 2 * .pi }
            let mit = (nFace + f0.n) / (1 + simd_dot(nFace, f0.n))
            st[st.count - 1].m = mit
            for k in 1...2 {
                let ang = a0 + da * Double(k) / 3
                st.append(LStation(x: st.last!.x, z: e.z0, n: Vec2(cos(ang), sin(ang)), m: mit, u: 0, onFace: true))
            }
            flank[0].m = mit
        }
        st += flank
        var u = 0.0
        for i in 0..<st.count {
            if i > 0 { u += simd_length(Vec2(st[i].x - st[i - 1].x, st[i].z - st[i - 1].z)) }
            st[i].u = u
        }
        return LampPath(s: st)
    }

    /// A closed solid lofted along `path`: each station is a chamfered slab from `inn` behind the body surface to `out`
    /// proud of it, between heights `y0…y1`.
    func lampWrap(_ path: LampPath, chamfer: Double = 0.012, smooth: Double? = 55,
                  _ sec: (LStation) -> LampSection) -> Mesh3 {
        var rings: [[Vec3]] = []
        for st in path.s {
            let s = sec(st)
            let ch = max(0.0004, min(chamfer, (s.y1 - s.y0) * 0.45, (s.out + s.inn) * 0.45))
            let pts: [(Double, Double)] = [(-s.inn, s.y0), (s.out - ch, s.y0), (s.out, s.y0 + ch),
                                           (s.out, s.y1 - ch), (s.out - ch, s.y1), (-s.inn, s.y1)]
            let inner = s.inn >= 0 ? st.m : st.n
            rings.append(pts.enumerated().map { (i, q) in
                // i 0 and 5 are the back (inn) points: offset along the mitre so a convex corner cannot fold.
                let v = (i == 0 || i == 5) ? inner : st.n
                return Vec3(st.x + v.x * q.0, q.1, st.z + v.y * q.0)
            })
        }
        var m = Mesh3()
        guard m.loft(rings: rings) else { return Mesh3() }
        return smooth.map { m.smoothed(creaseDegrees: $0) } ?? m
    }
}

// MARK: - Primitive solids

/// Lathe about +z (`dirZ = +1`) or -z (`-1`): `profile` is `(r, axial)` read exactly like `Mesh3.revolve`'s `(r, y)`
/// (solid on the left of travel; both ends on the axis). `base` is where axial = 0 sits.
func lampLathe(_ profile: [(Double, Double)], at base: Vec3, dirZ: Double = 1, segments: Int = 20) -> Mesh3 {
    var m = Mesh3()
    m.revolve(profile: profile.map { Mesh3.ProfilePoint(r: $0.0, y: $0.1) }, segments: segments)
    return m.smoothed(creaseDegrees: 50).rotatedX(dirZ > 0 ? .pi / 2 : -.pi / 2).translated(by: base)
}

/// Torus about the z axis through `centre` (major radius `major`, tube radius `minor`).
func lampTorus(at centre: Vec3, major: Double, minor: Double, segsMajor: Int = 24, segsMinor: Int = 8) -> Mesh3 {
    var m = Mesh3()
    func p(_ i: Int, _ j: Int) -> (pos: Vec3, c: Vec3) {
        let a = 2 * Double.pi * Double(i % segsMajor) / Double(segsMajor)
        let b = 2 * Double.pi * Double(j % segsMinor) / Double(segsMinor)
        let cc = Vec3(major * cos(a), major * sin(a), 0)
        let dir = Vec3(cos(a), sin(a), 0)
        return (centre + cc + dir * (minor * cos(b)) + Vec3(0, 0, minor * sin(b)), centre + cc)
    }
    for i in 0..<segsMajor { for j in 0..<segsMinor {
        let a = p(i, j), b = p(i + 1, j), c = p(i + 1, j + 1), d = p(i, j + 1)
        let mid = (a.pos + b.pos + c.pos + d.pos) / 4
        m.addQuad(a.pos, b.pos, c.pos, d.pos, outward: mid - (a.c + b.c) / 2)
    } }
    return m.smoothed(creaseDegrees: 60)
}

/// Honeycomb of hex studs standing `height` proud of `z` (towards `dir`), packed into a `2hw × 2hh` rectangle.
/// Stud count is bounded by `maxCells`.
func lampHoneycomb(cx: Double, cy: Double, hw: Double, hh: Double, z: Double, dir: Double,
                   cell: Double, height: Double, maxCells: Int = 400) -> Mesh3 {
    var m = Mesh3()
    let pitchX = 1.5 * cell, rowH = 3.0.squareRoot() * cell
    var count = 0
    var col = 0
    var x = -hw + cell
    while x <= hw - cell * 0.9 {
        let off = col % 2 == 0 ? 0.0 : rowH / 2
        var y = -hh + rowH / 2 + off
        while y <= hh - rowH * 0.45 {
            if count < maxCells {
                m.append(Mesh3.sweep(profile: Mesh3.circleProfile(radius: cell * 0.80, segments: 6),
                                     along: [Vec3(cx + x, cy + y, z), Vec3(cx + x, cy + y, z + dir * height)]))
                count += 1
            }
            y += rowH
        }
        x += pitchX; col += 1
    }
    return m
}

/// Plate-sized flat panel: `w × h`, centred at `(x, y)`, front face at `zFront`, `t` thick behind it.
func lampPlatePanel(x: Double, y: Double, w: Double, h: Double, zFront: Double, dir: Double, t: Double) -> Mesh3 {
    carBox(Vec3(x - w / 2, y - h / 2, min(zFront, zFront - dir * t)), Vec3(x + w / 2, y + h / 2, max(zFront, zFront - dir * t)))
}

extension CarContext {
    /// Records `m` into `a[mesh]` and logs its triangle range.
    func emit(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ mesh: CarLampMesh, _ fn: CarLampFunction,
              _ m: Mesh3, side: Int = 0) {
        guard !m.isEmpty else { return }
        let start = a[keyPath: mesh.keyPath].triangleCount
        a[keyPath: mesh.keyPath].append(m)
        r.shells.append(CarLampShell(mesh: mesh, function: fn, side: side, triangles: start..<(start + m.triangleCount)))
    }
    /// Emits `m` on the right and its mirror on the left.
    func emitPair(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ mesh: CarLampMesh, _ fn: CarLampFunction, _ m: Mesh3) {
        emit(&a, &r, mesh, fn, m, side: 1)
        emit(&a, &r, mesh, fn, m.mirrored(.x), side: -1)
    }
}
