import Foundation
import simd

/// Where the sun, the moon and the stars are, for a night sky that turns and phases like
/// the real one — the host half of the physical night sky (IlluminatoramaNightSky.h).
///
/// A deliberately SIMPLE ephemeris: a circular solar orbit (ecliptic longitude linear in the
/// day of year, the March equinox on day 80), the moon on the ecliptic at an elongation set
/// by its age in a mean synodic month (no orbital inclination, no libration, no parallax),
/// and apparent solar time. It is accurate to a degree or two — plenty for a scene whose
/// clock is a slider — and it keeps the three things a viewer actually reads CONSISTENT
/// with each other: the moon's phase matches its distance from the sun, the lit limb faces
/// the sun, and the stars turn about the celestial pole with the hour.
///
/// Everything is computed in local ENU (east, north, up); `world(east:north:up:)` maps it
/// into a host's axes and builds the world → equatorial quaternion the shaders take.
public struct NightSkyEphemeris: Sendable {

    // ── Constants ────────────────────────────────────────────────────────────

    /// The moon's true mean angular RADIUS (0.259°).
    public static let moonAngularRadius: Float = 0.00452
    /// The legacy dome moon's radius (≈3.2°) — what `.legacy` hosts were tuned around.
    public static let legacyDomeMoonAngularRadius: Float = 0.0559
    /// Mean synodic month (days, new moon to new moon).
    public static let synodicMonth = 29.530589
    /// Obliquity of the ecliptic (J2000).
    static let obliquity = 23.439 * Double.pi / 180

    // ── Inputs ───────────────────────────────────────────────────────────────

    public let latitudeDeg: Double
    public let dayOfYear: Double
    public let localSolarHour: Double
    /// Days since new moon (0 new, ~7.4 first quarter, ~14.8 full, ~22.1 last quarter).
    public let moonAgeDays: Double

    // ── Outputs (local ENU: x = east, y = north, z = up) ─────────────────────

    /// Unit vector toward the sun.
    public let sunENU: SIMD3<Double>
    /// Unit vector toward the moon.
    public let moonENU: SIMD3<Double>
    /// Rotation taking a J2000-equatorial unit vector (x → RA 0h, z → north celestial
    /// pole) into ENU at this instant.
    public let equatorialToENU: simd_double3x3
    /// Moon phase angle (radians) — the sun–moon–Earth angle; 0 = full, π = new.
    public let moonPhaseAngle: Double

    /// Fraction of the moon's disk that is lit: (1 + cos α) / 2.
    public var moonIlluminatedFraction: Double { 0.5 * (1 + cos(moonPhaseAngle)) }
    /// The moon's flux relative to a full moon (Allen's phase law) — scale moonlight on the
    /// scene by this, as the shader scales the moonlit sky.
    public var moonFluxFraction: Float { Self.moonPhaseFlux(cosPhaseAngle: Float(cos(moonPhaseAngle))) }

    public init(latitudeDeg: Double, dayOfYear: Double, localSolarHour: Double, moonAgeDays: Double) {
        self.latitudeDeg = latitudeDeg
        self.dayOfYear = dayOfYear
        self.localSolarHour = localSolarHour
        self.moonAgeDays = moonAgeDays

        let phi = latitudeDeg * .pi / 180
        let lambdaSun = 2 * Double.pi * (dayOfYear - 80) / 365.2422
        let elongation = 2 * Double.pi * moonAgeDays.truncatingRemainder(dividingBy: Self.synodicMonth)
            / Self.synodicMonth
        let (raSun, _) = Self.equatorialFromEcliptic(longitude: lambdaSun)
        // Apparent solar time: the sun's hour angle is 15°/h from local noon, so the local
        // sidereal angle (the hour angle of RA 0h) is RA☉ + H☉.
        let lst = raSun + (localSolarHour - 12) * 15 * .pi / 180

        // Columns: where RA 0h / RA 6h on the equator and the north pole sit in ENU.
        func enu(hourAngle h: Double, dec d: Double) -> SIMD3<Double> {
            SIMD3(-cos(d) * sin(h),
                  cos(phi) * sin(d) - sin(phi) * cos(d) * cos(h),
                  sin(phi) * sin(d) + cos(phi) * cos(d) * cos(h))
        }
        let m = simd_double3x3(columns: (enu(hourAngle: lst, dec: 0),
                                         enu(hourAngle: lst - .pi / 2, dec: 0),
                                         enu(hourAngle: 0, dec: .pi / 2)))
        self.equatorialToENU = m
        self.sunENU = simd_normalize(m * Self.equatorialVector(eclipticLongitude: lambdaSun))
        self.moonENU = simd_normalize(m * Self.equatorialVector(eclipticLongitude: lambdaSun + elongation))
        // Sun at infinity: the angle at the moon between the sun and the Earth is π − elongation.
        self.moonPhaseAngle = .pi - acos(max(-1, min(1, cos(elongation))))
    }

    /// The same sky in a host's world axes.
    public struct World: Sendable {
        /// Unit vectors toward the sun / moon (the sun is below the horizon at night — the
        /// true direction, which is what lights the moon's phase).
        public let sunToward: SIMD3<Float>
        public let moonToward: SIMD3<Float>
        /// World → equatorial quaternion (ix, iy, iz, r) — `Params.celestialOrientation`,
        /// `IlluminatoramaRenderer.nightSkyCelestialOrientation`.
        public let celestialOrientation: SIMD4<Float>
        /// The north celestial pole in world space.
        public let celestialPole: SIMD3<Float>
    }

    /// Map into a host's world, given the world directions of local east, north and up
    /// (an orthonormal, right-handed basis: east × north = up).
    public func world(east: SIMD3<Float>, north: SIMD3<Float>, up: SIMD3<Float>) -> World {
        let basis = simd_double3x3(columns: (SIMD3<Double>(simd_normalize(east)),
                                             SIMD3<Double>(simd_normalize(north)),
                                             SIMD3<Double>(simd_normalize(up))))
        let eqToWorld = basis * equatorialToENU
        let q = simd_quatd(eqToWorld).inverse.normalized
        return World(sunToward: SIMD3<Float>(simd_normalize(basis * sunENU)),
                     moonToward: SIMD3<Float>(simd_normalize(basis * moonENU)),
                     celestialOrientation: SIMD4<Float>(Float(q.imag.x), Float(q.imag.y), Float(q.imag.z), Float(q.real)),
                     celestialPole: SIMD3<Float>(simd_normalize(eqToWorld * SIMD3<Double>(0, 0, 1))))
    }

    // ── Shared helpers (also used by the renderer's uniform packing) ─────────

    /// A direction to light the moon from so it shows `illuminatedFraction` (0 new … 1 full)
    /// while keeping its lit limb toward the true sun: the phase angle α has cos α = 2k − 1,
    /// and the light sits α away from the moon→Earth direction, rotated toward the sun.
    /// This is how a host that places the moon BY HAND (a fixed offset from the sun) keeps
    /// the phase it chose with a real sphere-lit terminator.
    public static func effectiveSun(moonDir: SIMD3<Float>, trueToSun: SIMD3<Float>,
                                    illuminatedFraction k: Float) -> SIMD3<Float> {
        let m = simd_normalize(moonDir)
        var t = trueToSun - simd_dot(trueToSun, m) * m
        if simd_length_squared(t) < 1e-8 { t = SIMD3<Float>(0, 1, 0) - m.y * m }
        if simd_length_squared(t) < 1e-8 { t = SIMD3<Float>(1, 0, 0) - m.x * m }
        t = simd_normalize(t)
        let cosA = max(-1, min(1, 2 * k - 1))
        let sinA = (1 - cosA * cosA).squareRoot()
        return simd_normalize(-cosA * m + sinA * t)
    }

    /// Allen's lunar phase law: flux relative to full, m(α) = 0.026|α| + 4·10⁻⁹α⁴ (α°).
    /// Mirror of `nightMoonPhaseFlux` in IlluminatoramaNightSky.h.
    public static func moonPhaseFlux(cosPhaseAngle: Float) -> Float {
        let a = acos(max(-1, min(1, cosPhaseAngle))) * 180 / .pi
        let mag = 0.026 * a + 4.0e-9 * a * a * a * a
        return pow(10, -0.4 * mag)
    }

    /// Kasten–Young relative air mass. Mirror of `nightAirmass`.
    public static func airmass(sinElevation: Float) -> Float {
        let el = max(asin(max(-1, min(1, sinElevation))) * 180 / .pi, 0)
        return 1 / (sin(el * .pi / 180) + 0.50572 * pow(el + 6.07995, -1.6364))
    }

    /// Point-source transmittance through the air (R, G, B). Mirror of `nightExtinction`.
    public static func extinction(sinElevation: Float) -> SIMD3<Float> {
        let x = airmass(sinElevation: sinElevation)
        let k = SIMD3<Float>(0.11, 0.19, 0.32)
        return SIMD3<Float>(pow(10, -0.4 * k.x * x), pow(10, -0.4 * k.y * x), pow(10, -0.4 * k.z * x))
    }

    // ── Internals ────────────────────────────────────────────────────────────

    static func equatorialFromEcliptic(longitude l: Double) -> (ra: Double, dec: Double) {
        (atan2(cos(obliquity) * sin(l), cos(l)), asin(sin(obliquity) * sin(l)))
    }

    static func equatorialVector(eclipticLongitude l: Double) -> SIMD3<Double> {
        SIMD3(cos(l), cos(obliquity) * sin(l), sin(obliquity) * sin(l))
    }
}
