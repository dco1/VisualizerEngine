import Foundation
import simd

// plausibility: real — body shells are authored lofts whose underside follows real wheel arches
// (no cylinders poking through a box), greenhouse/glass/roof/pillars are separate parts, tyres and
// rims are lathes, trim and lamps are bevelled solids. No CSG, no textures standing in for shape.

/// **One engine-neutral car builder** (proportions from real reference models — see `CarBlueprint.make`) — replaces the four separate car implementations
/// (FlyingCats, CityStreet, CityStreetUltra, VistaIntersection). Pure `Mesh3`: no SceneKit,
/// AppKit or Metal, so it builds for macOS, iOS, tvOS and Daydream Home alike.
///
/// ```swift
/// let car = CarFactory.build(CarSpec(style: .suv))              // modern SUV, nominal size
/// let old = CarFactory.build(CarSpec(style: .coupe, era: .classic1950s))
/// let big = CarFactory.build(CarSpec(style: .sedan, length: 5.0)) // stretch to 5 m
/// ```
///
/// **Local frame** (metres): +Z forward, +X right (driver's-right when facing forward), +Y up;
/// origin on the ground midway between the axles. Convention conversion: for a scene whose
/// cars drive along +X call `parts.forwardAlongX()` (= `rotatedY(+π/2)`, i.e. (x,y,z)→(z,y,−x)).
///
/// **Paint is not baked in** — tint `body` per instance; `glass`, `tyres`/`rims`, `headlights`,
/// `tailLights`, `trim` are separate meshes so each can take its own material. Wheels are placed
/// at `wheelCentres` (right-hand wheel axle runs along ±X); to spin them, re-use one
/// `wheelMesh`/`rimMesh` per centre (`wheelUnit`) and rotate about X.
///
/// Every mesh is a closed, outward-wound, smooth/bevelled solid (`CarFactoryTests` asserts closure,
/// winding-vs-normals, dimensions, wheel clearance and determinism). Output is deterministic.
public struct CarSpec: Sendable, Equatable {
    public var style: CarStyle
    public var era: CarEra
    /// Overrides (metres). `nil` keeps the style's nominal value. Length/width/height rescale the
    /// body's key positions; wheel radius resizes tyres AND their arches.
    public var length: Double?, width: Double?, height: Double?, wheelRadius: Double?

    public init(style: CarStyle, era: CarEra = .modern, length: Double? = nil, width: Double? = nil,
                height: Double? = nil, wheelRadius: Double? = nil) {
        self.style = style; self.era = era
        self.length = length; self.width = width; self.height = height; self.wheelRadius = wheelRadius
    }
}

public struct CarDimensions: Sendable, Equatable {
    public var length: Double, width: Double, height: Double
    public var wheelbase: Double
    public var wheelRadius: Double, wheelWidth: Double
    /// Distance between the centres of the two tyres on an axle.
    public var track: Double
    /// Z of the front and rear faces of the body (rear is negative).
    public var frontZ: Double, rearZ: Double
}

public struct CarParts: Sendable, Equatable {
    /// Paintwork: shell with real wheel arches, roof panel, pillars.
    public var body: Mesh3
    /// Windscreen / side / rear glass (a separate dark mesh).
    public var glass: Mesh3
    /// All tyres in one mesh (closed solids), already placed.
    public var tyres: Mesh3
    /// Rim dishes proud of the tyre faces, already placed.
    public var rims: Mesh3
    public var headlights: Mesh3
    public var tailLights: Mesh3
    /// Bumpers, grille, mirrors (chrome/black plastic).
    public var trim: Mesh3
    /// Taxi roof sign (empty for other styles) — usually emissive.
    public var roofSign: Mesh3
    /// One right-hand tyre / rim about the origin, axle along +X (outer face +X). Left wheels are
    /// `.mirrored(.x)`. Use with `wheelCentres` to instance and spin wheels.
    public var wheelUnit: Mesh3
    public var rimUnit: Mesh3
    /// Axle centres, ordered front-right, front-left, rear-right, rear-left (…per axle).
    public var wheelCentres: [Vec3]
    public var dimensions: CarDimensions

    /// Chrome / bright-metal strips, door handles, badges, wiper arms, exhaust tips.
    public var chrome: Mesh3
    /// Amber indicators, side repeaters, DRL strip, reverse lamps (emissive in a host).
    public var signals: Mesh3
    /// Licence plates (front and rear), flat panels with a recess.
    public var plates: Mesh3
    /// Dashboard, steering wheel, seats — seen through the glass.
    public var interior: Mesh3
    /// Dark under-body parts: wheel-well liners, undertray, tow hook, exhaust.
    public var underbody: Mesh3
    /// Brake discs and callipers, placed in world position (do not spin with the wheel).
    public var brakes: Mesh3

    /// Alias for `tyres` (the spec's "wheels").
    public var wheels: Mesh3 { tyres }

    /// Every part's meshes in one list (for bounds / auditing).
    public var allMeshes: [Mesh3] { [body, glass, tyres, rims, headlights, tailLights, trim, roofSign, chrome, signals, plates, interior, underbody, brakes] }

    /// Axis-aligned bounds of everything.
    public var bounds: (min: Vec3, max: Vec3) {
        var lo = Vec3(repeating: .infinity), hi = Vec3(repeating: -.infinity)
        for m in allMeshes { if let b = m.bounds { lo = simd_min(lo, b.min); hi = simd_max(hi, b.max) } }
        return (lo, hi)
    }

    /// The same car for scenes whose vehicles drive along +X: forward +Z → +X (right +X → −Z).
    public func forwardAlongX() -> CarParts {
        func r(_ m: Mesh3) -> Mesh3 { m.rotatedY(.pi / 2) }
        func rv(_ v: Vec3) -> Vec3 { Vec3(v.z, v.y, -v.x) }
        var d = dimensions
        d.frontZ = dimensions.frontZ; d.rearZ = dimensions.rearZ    // still measured along the travel axis
        return CarParts(body: r(body), glass: r(glass), tyres: r(tyres), rims: r(rims),
                        headlights: r(headlights), tailLights: r(tailLights), trim: r(trim),
                        roofSign: r(roofSign), wheelUnit: r(wheelUnit), rimUnit: r(rimUnit),
                        wheelCentres: wheelCentres.map(rv), dimensions: d,
                        chrome: r(chrome), signals: r(signals), plates: r(plates),
                        interior: r(interior), underbody: r(underbody), brakes: r(brakes))
    }
}

/// Everything a build step contributes to. Each `buildX` step (one file each) appends to its own meshes.
struct CarAccumulator {
    var body = Mesh3(), glass = Mesh3(), trim = Mesh3(), chrome = Mesh3()
    var headlights = Mesh3(), tailLights = Mesh3(), signals = Mesh3(), plates = Mesh3()
    var interior = Mesh3(), underbody = Mesh3(), brakes = Mesh3(), roofSign = Mesh3()
    var tyres = Mesh3(), rims = Mesh3(), tyreUnit = Mesh3(), rimUnit = Mesh3()
    var wheelCentres: [Vec3] = []
}

public enum CarFactory {

    public static func build(_ spec: CarSpec) -> CarParts {
        var b = CarBlueprint.make(spec.style, era: spec.era)
        switch spec.era {
        case .classic1950s: b.cornerRadius *= 1.6; b.cabinWidthFrac -= 0.04
        case .nineties:     break
        case .modern:       b.cornerRadius *= 0.85
        }
        let L = spec.length ?? b.length, W = spec.width ?? b.width, H = spec.height ?? b.height
        let R = spec.wheelRadius ?? b.wheelRadius
        let c = CarContext(bp: b, era: spec.era, L: L, W: W, H: H, R: R, rL: L / b.length, rH: H / b.height)

        // One file per concern; order matters only in that later steps may read earlier meshes.
        var a = CarAccumulator()
        c.buildBody(&a)         // CarFactory+Body.swift        shell, stance, arches, panels, mirrors, handles…
        c.buildGreenhouse(&a)   // CarFactory+Greenhouse.swift  glass panes, frames, pillars, roof, interior
        c.buildLamps(&a)        // CarFactory+Lamps.swift       face, lamps, grille, bumpers, plates
        c.buildBodyDetails(&a)  // CarFactory+BodyDetails.swift  door cuts, shut lines, handles, flap, mirrors, sills, wiper/antenna
        c.buildWheels(&a)       // CarFactory+Wheels.swift      tyres, rims, brakes, wells, undertray, exhaust
        c.buildClassCues(&a)    // CarFactory+ClassCues.swift   pickup bed, rails, taxi/bus/van/hatch cues

        let dims = CarDimensions(length: L, width: W, height: H, wheelbase: b.wheelbase * c.rL,
                                 wheelRadius: R, wheelWidth: c.wheelW, track: a.wheelCentres.count >= 2 ? abs(a.wheelCentres[0].x - a.wheelCentres[1].x) : c.trackHalf * 2,
                                 frontZ: c.frontZ, rearZ: c.rearZ)
        return CarParts(body: a.body, glass: a.glass, tyres: a.tyres, rims: a.rims, headlights: a.headlights,
                        tailLights: a.tailLights, trim: a.trim, roofSign: a.roofSign, wheelUnit: a.tyreUnit, rimUnit: a.rimUnit,
                        wheelCentres: a.wheelCentres, dimensions: dims, chrome: a.chrome,
                        signals: a.signals, plates: a.plates, interior: a.interior, underbody: a.underbody,
                        brakes: a.brakes)
    }
}

// MARK: - Construction context (shared helpers; edited only by the integrator)

struct CarContext {
    let bp: CarBlueprint, era: CarEra
    let L: Double, W: Double, H: Double, R: Double, rL: Double, rH: Double

    var frontZ: Double { bp.frontZ * rL }
    var rearZ: Double { frontZ - L }
    var axles: [Double] { [bp.wheelbase / 2 * rL, -bp.wheelbase / 2 * rL] }
    var wheelW: Double { bp.wheelWidth * R / bp.wheelRadius }
    var trackHalf: Double { W / 2 - wheelW / 2 - 0.02 }   // tyre face ~2 cm inside the fender line
    var archR: Double { R + 0.06 }
    var classic: Bool { era == .classic1950s }

    /// Piecewise-linear `(d, y)` from the front → ascending-z `(z, y)`.
    func poly(_ p: [(Double, Double)]) -> [(Double, Double)] {
        p.map { (frontZ - $0.0 * rL, $0.1 * rH) }.reversed()
    }
    func interp(_ p: [(Double, Double)], _ x: Double) -> Double {
        guard let f = p.first, let l = p.last else { return 0 }
        if x <= f.0 { return f.1 }
        if x >= l.0 { return l.1 }
        for i in 1..<p.count where x <= p[i].0 {
            let (a, b) = (p[i - 1], p[i]); let t = (x - a.0) / max(b.0 - a.0, 1e-9)
            return a.1 + (b.1 - a.1) * t
        }
        return l.1
    }

    // MARK: body-shell helpers

    func planFactor(_ z: Double) -> Double {
        let zone = bp.boxy ? 0.6 : 0.9
        let fn = bp.noseTaper + (1 - bp.noseTaper) * sin(min((frontZ - z) / zone, 1) * .pi / 2)
        let fr = bp.tailTaper + (1 - bp.tailTaper) * sin(min((z - rearZ) / zone, 1) * .pi / 2)
        return min(fn, fr)
    }

    func bottom(_ z: Double) -> Double {
        let t = min(min(frontZ - z, z - rearZ) / 0.5, 1)
        var y = bp.endBottom + (bp.sill - bp.endBottom) * sin(t * .pi / 2)
        for a in axles {
            let dz = z - a
            if abs(dz) < archR { y = max(y, R + (archR * archR - dz * dz).squareRoot()) }
        }
        return y
    }

    struct Ring { var hw: Double, bottom: Double, top: Double, corner: Double }

    func profile(_ r: Ring) -> [Vec2] {
        let hh = (r.top - r.bottom) / 2, cy = (r.top + r.bottom) / 2
        let cr = max(0.004, min(r.corner, hh * 0.9, r.hw * 0.5))
        // Mesh3.loft stands the section up with (x, y) → world (−x, −y): symmetric, so only the Y offset flips.
        return Mesh3.roundedRectProfile(halfWidth: r.hw, halfHeight: hh, cornerRadius: cr, corner: 4)
            .map { Vec2($0.x, $0.y - cy) }
    }

    func stations(_ lo: Double, _ hi: Double, _ extra: [Double]) -> [Double] {
        var zs = [lo, hi] + extra.filter { $0 > lo && $0 < hi }
        var s = lo + 0.25
        while s < hi - 0.01 { zs.append(s); s += 0.25 }
        zs.sort()
        var out: [Double] = []
        for z in zs where out.isEmpty || z - out.last! >= 0.004 { out.append(z) }
        if hi - out.last! < 0.004, out.count > 1 { out[out.count - 1] = hi } else if out.last! < hi { out.append(hi) }
        return out
    }

    func bodyRing(_ z: Double, _ top: [(Double, Double)]) -> Ring {
        let t = interp(top, z)
        let b = min(bottom(z), t - 0.12)
        return Ring(hw: W / 2 * planFactor(z), bottom: b, top: t, corner: bp.cornerRadius)
    }

    // Greenhouse geometry
    var cabinPoly: [(Double, Double)] { poly(bp.cabin) }
    var cabinSpan: (Double, Double) { (cabinPoly.first?.0 ?? 0, cabinPoly.last?.0 ?? 0) }
    func cabinRing(_ z: Double) -> Ring {
        let (z0, z1) = cabinSpan
        let t = min(max((z - z0) / max(z1 - z0, 1e-6), 0), 1)
        let hw = W / 2 * bp.cabinWidthFrac * (0.80 + 0.20 * pow(sin(t * .pi), 0.5))
        let b = bp.belt * rH - 0.08
        let top = max(interp(cabinPoly, z) - 0.03, b + 0.05)
        return Ring(hw: hw, bottom: b, top: top, corner: 0.12)
    }

    func loftRings(_ zs: [Double], _ ring: (Double) -> Ring) -> Mesh3 {
        Mesh3.loft(sections: zs.map { profile(ring($0)) }, along: zs.map { Vec3(0, 0, $0) })
    }
    func a_(_ v: Double) -> Double { v < 0 ? rearZ - v : frontZ - v }
}

// MARK: - Box helpers

// MARK: - Box helpers

func carBox(_ lo: Vec3, _ hi: Vec3) -> Mesh3 {
    var m = Mesh3()
    let (x0, y0, z0) = (lo.x, lo.y, lo.z), (x1, y1, z1) = (hi.x, hi.y, hi.z)
    m.addQuad(Vec3(x0, y1, z0), Vec3(x1, y1, z0), Vec3(x1, y1, z1), Vec3(x0, y1, z1), outward: Vec3(0, 1, 0))
    m.addQuad(Vec3(x0, y0, z0), Vec3(x1, y0, z0), Vec3(x1, y0, z1), Vec3(x0, y0, z1), outward: Vec3(0, -1, 0))
    m.addQuad(Vec3(x1, y0, z0), Vec3(x1, y1, z0), Vec3(x1, y1, z1), Vec3(x1, y0, z1), outward: Vec3(1, 0, 0))
    m.addQuad(Vec3(x0, y0, z0), Vec3(x0, y1, z0), Vec3(x0, y1, z1), Vec3(x0, y0, z1), outward: Vec3(-1, 0, 0))
    m.addQuad(Vec3(x0, y0, z1), Vec3(x1, y0, z1), Vec3(x1, y1, z1), Vec3(x0, y1, z1), outward: Vec3(0, 0, 1))
    m.addQuad(Vec3(x0, y0, z0), Vec3(x1, y0, z0), Vec3(x1, y1, z0), Vec3(x0, y1, z0), outward: Vec3(0, 0, -1))
    return m
}

func carBevelBox(_ lo: Vec3, _ hi: Vec3, _ radius: Double) -> Mesh3 {
    let d = hi - lo
    let r = min(radius, min(d.x, min(d.y, d.z)) * 0.45)
    return carBox(lo, hi).beveled(radius: r, creaseDegrees: 30, segments: 2)
}

// ── Car kinematics: smooth turning, built in ────────────────────────────────────────────────
//
// A real car doesn't pivot about its centre on the path — its axles do the tracking. `CarPose`
// puts the REAR and FRONT axle on the path (one wheelbase apart, centred on `s`), takes the body
// heading from the axle-to-axle chord and the body centre from their midpoint. That alone gives
// the correct swing-in at a corner (rear cuts inside, nose leads) and turns a polyline's per-vertex
// heading snap into a change spread over a wheelbase. `CarSteering` then low-passes the heading
// with a yaw-rate limit so any remaining kink (OSM-sampled bends) can't read as a twitch.
// plausibility: real — kinematic bicycle model; the path is the only thing that's authored.
public struct CarPose: Sendable {
    public var position: SIMD2<Float>      // body centre (mid-axle)
    public var heading: SIMD2<Float>       // unit, axle chord
    public var steerAngle: Float           // front-wheel angle, radians (+ = counter-clockwise in the x/y plane)

    /// - Parameter sample: path position at arc length `s` (clamped by the caller's path).
    public static func track(s: Float, wheelbase wb: Float, sample: (Float) -> SIMD2<Float>) -> CarPose {
        let rear = sample(s - wb / 2), front = sample(s + wb / 2)
        let d = front - rear
        let l = simd_length(d)
        let h = l > 1e-4 ? d / l : SIMD2<Float>(1, 0)
        // steer angle from the heading change across the wheelbase (look a wheelbase either side)
        let r0 = sample(s - wb), f1 = sample(s + wb)
        let hb = simd_normalize_safe(rear - r0, fallback: h), hf = simd_normalize_safe(f1 - front, fallback: h)
        let cross = hb.x * hf.y - hb.y * hf.x
        let dot = simd_dot(hb, hf)
        let dTheta = atan2(cross, dot)
        let steer = atan(dTheta / 3)           // curvature = dTheta / 3wb, steer = atan(wb·curvature)
        return CarPose(position: (rear + front) / 2, heading: h, steerAngle: steer)
    }

    @inline(__always) private static func simd_normalize_safe(_ v: SIMD2<Float>, fallback: SIMD2<Float>) -> SIMD2<Float> {
        let l = simd_length(v); return l > 1e-4 ? v / l : fallback
    }
}

public struct CarSteering: Sendable {
    public var heading: SIMD2<Float>
    public var yawRate: Float = 0
    public init(heading: SIMD2<Float>) { self.heading = heading }

    /// Critically-damped follow of `target` with a yaw-rate cap (rad/s). Returns the smoothed heading.
    public mutating func update(toward target: SIMD2<Float>, dt: Float, maxYawRate: Float = 1.6, response: Float = 14) -> SIMD2<Float> {
        let cur = atan2(heading.y, heading.x), tgt = atan2(target.y, target.x)
        var err = tgt - cur
        while err > .pi { err -= 2 * .pi }
        while err < -.pi { err += 2 * .pi }
        // spring-damper on the yaw error, then clamp the rate
        let accel = response * response * err - 2 * response * yawRate
        yawRate = max(-maxYawRate, min(maxYawRate, yawRate + accel * dt))
        let a = cur + yawRate * dt
        heading = SIMD2(cos(a), sin(a))
        return heading
    }
}
