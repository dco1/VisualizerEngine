import Foundation
import simd

// ── AEROSOLS FOR THE NISHITA SKY (`VolumetricCloudRenderer.Params.atmosphereAerosol`) ─────────
//
// The engine's Nishita atmosphere carried one fixed aerosol: Bruneton's (2008) β_M = 21e-6 m⁻¹,
// grey, ×1.1 for extinction, a 1.2 km scale height and a Cornette–Shanks g of 0.76. In the terms
// atmospheric science uses for aerosols that is an optical depth at 550 nm of 0.0277 — cleaner
// than any continental air OPAC tabulates (Hess, Koepke & Schult 1998, BAMS 79:831, Table 3:
// continental clean 0.064, average 0.151, polluted 0.327, urban 0.643). With multiple scattering
// on, that air is "too clean on the sun side" (Daydream Home, DH-0957): the golden band over a
// low sun is thin, and at −3° the orange twilight arch a real sky shows is gone — the vivid sunward
// twilight colours need tropospheric haze (Lee & Mollner 2017, "Tropospheric haze and colors of the
// clear twilight sky", Appl. Opt. 56:G179). `NishitaAerosol` makes the aerosol a physical, host-set
// quantity that reaches EVERY consumer of the sky from one set of numbers.
//
// Parameterisation (the primary quantity is τ550, the one sun photometers — AERONET — and
// satellites report):
//
//   τa(λ) = τ550 · (λ / 550 nm)^−α            Ångström's law; α ≈ 1.3 for continental aerosol
//   βe(λ, h) = τa(λ) / H_M · e^(−h / H_M)     extinction: exponential profile, scale height H_M
//   βs(λ, h) = ω₀ · βe(λ, h)                  scattering; ω₀ = single-scattering albedo (1 − ω₀
//                                              of the extinction is absorption — soot)
//   phase: Cornette–Shanks with parameter g   (its mean cosine is 3g(4 + g²) / (5(2 + g²)))
//
// evaluated at the sky's channel wavelengths — 680 / 550 / 440 nm, where kBetaR (the Rayleigh
// coefficients) is quoted.
//
// Mappings: Ångström's turbidity β (τa at 1 µm) gives τ550 = β · 0.55^−α. Preetham, Shirley &
// Smits (1999, SIGGRAPH, App. A.1)'s turbidity T gives β = 0.04608 T − 0.04586 with α = 1.3, so
// τ550 = 2.1755 β: T = 2 → 0.10, T = 3 → 0.20, T = 3.5 → 0.25, T = 4 → 0.30 (`preetham(turbidity:)`).

/// The aerosol (Mie) component of the engine's Nishita atmosphere. `.builtin` is exactly the
/// constants the sky always had (every host that never sets `atmosphereAerosol` is bit-identical).
public struct NishitaAerosol: Sendable, Equatable {
    /// Vertical aerosol optical depth at 550 nm, sea level to space (τ550 = βe(550) · H_M).
    public var opticalDepth: Float
    /// Ångström exponent α: τa(λ) = τ550 (λ / 550 nm)^−α. 0 = grey (the built-in aerosol);
    /// ~1.3 continental and urban (OPAC: 1.1–1.45), ~0.1–0.4 maritime and desert dust.
    public var angstromExponent: Float
    /// Scale height of the exponential aerosol profile, m — an EFFECTIVE height: one exponential
    /// standing in for the mixing layer plus the aerosol above it. It decides the twilight arch:
    /// the arch and afterglow are sunlit aerosol ABOVE the Earth's shadow (2–6 km at −3°), seen
    /// through the shadowed air near the ground, so a low dense haze dims the sunward horizon
    /// after sunset while aerosol spread higher lights it. OPAC's continental types put
    /// τ / σe(surface) at 1.8–2.0 km, lidar boundary layers at 1–2 km; Koomen et al. (1952)'s
    /// measured twilight skies (a hazy low site, Maryland) are best reproduced with 3–4 km
    /// (NishitaAerosolTests). Maritime ~1.1 km; the built-in aerosol is 1.2 km.
    public var scaleHeight: Float
    /// Single-scattering albedo ω₀ = scattering / extinction (OPAC at 80 % RH: continental clean
    /// 0.97, average 0.93, polluted 0.89, urban 0.82; maritime ≥ 0.98). Built in: 1/1.1 = 0.909.
    public var singleScatteringAlbedo: Float
    /// Cornette–Shanks g (the phase function's parameter, not its mean cosine — see
    /// `cornetteShanksG(asymmetryParameter:)`). Built in: 0.76 (mean cosine 0.81).
    public var phaseG: Float

    public init(opticalDepth: Float, angstromExponent: Float, scaleHeight: Float,
                singleScatteringAlbedo: Float, phaseG: Float) {
        self.opticalDepth = opticalDepth
        self.angstromExponent = angstromExponent
        self.scaleHeight = scaleHeight
        self.singleScatteringAlbedo = singleScatteringAlbedo
        self.phaseG = phaseG
    }

    // ── Presets ───────────────────────────────────────────────────────────────────────────

    /// The engine's built-in aerosol, exactly: Bruneton's β_M = 21e-6 m⁻¹ scattering, grey,
    /// extinction ×1.1, H_M 1200 m, g 0.76 (VolumetricSky.metal kBetaM / kMieH / kMieG) —
    /// τ550 = 21e-6 · 1.1 · 1200 = 0.0277. The default of `Params.atmosphereAerosol`.
    public static let builtin = NishitaAerosol(
        opticalDepth: NishitaAtmosphere.builtinMieScattering * NishitaAtmosphere.builtinMieExtinctionRatio
            * NishitaAtmosphere.builtinMieScaleHeight,
        angstromExponent: 0,
        scaleHeight: NishitaAtmosphere.builtinMieScaleHeight,
        singleScatteringAlbedo: 1 / NishitaAtmosphere.builtinMieExtinctionRatio,
        phaseG: NishitaAtmosphere.builtinMieG)

    /// Remote continental air (OPAC "continental clean": τ550 0.064, ω₀ 0.97, mean cosine 0.71).
    public static let cleanContinental = NishitaAerosol(
        opticalDepth: 0.064, angstromExponent: 1.3, scaleHeight: 3000,
        singleScatteringAlbedo: 0.97, phaseG: cornetteShanksG(asymmetryParameter: 0.709))

    /// Rural / average continental air (OPAC "continental average": τ550 0.151, visibility ~35 km,
    /// ω₀ 0.925, mean cosine 0.70; Preetham T ≈ 2.5), H_M 3 km. The closest of these presets to
    /// Koomen et al. (1952)'s measured twilight skies (0.42 stop rms over their sunward and
    /// anti-sunward verticals at the hazy Maryland site, sun +5° … −6°, against 0.57 for the
    /// built-in air).
    public static let rural = NishitaAerosol(
        opticalDepth: 0.151, angstromExponent: 1.3, scaleHeight: 3000,
        singleScatteringAlbedo: 0.925, phaseG: cornetteShanksG(asymmetryParameter: 0.703))

    /// **Hazy suburban air, turbidity ≈ 3–4** — the preset for a golden low sun and a twilight
    /// arch. τ550 0.25 (Preetham T = 3.5; between OPAC's continental average 0.151 and polluted
    /// 0.327), α 1.3, H_M 4 km (the haze reaching the air that is still sunlit after sunset — at
    /// 2 km the same τ dims the −3° sunward horizon to ~1× the zenith, far below the ~8× measured),
    /// ω₀ 0.92, mean cosine 0.70.
    public static let hazySuburban = NishitaAerosol(
        opticalDepth: 0.25, angstromExponent: 1.3, scaleHeight: 4000,
        singleScatteringAlbedo: 0.92, phaseG: cornetteShanksG(asymmetryParameter: 0.70))

    /// Polluted urban haze (OPAC "urban": τ550 0.643, visibility ~8 km, ω₀ 0.82 — soot), H_M 2 km.
    public static let urban = NishitaAerosol(
        opticalDepth: 0.643, angstromExponent: 1.3, scaleHeight: 2000,
        singleScatteringAlbedo: 0.817, phaseG: cornetteShanksG(asymmetryParameter: 0.689))

    /// Clean maritime air (OPAC "maritime clean": τ550 0.096, α ≈ 0.1, ω₀ 0.997, mean cosine 0.77).
    public static let cleanMaritime = NishitaAerosol(
        opticalDepth: 0.096, angstromExponent: 0.1, scaleHeight: 1100,
        singleScatteringAlbedo: 0.997, phaseG: cornetteShanksG(asymmetryParameter: 0.772))

    /// Preetham et al. (1999)'s turbidity T → this model: Ångström β = 0.04608 T − 0.04586,
    /// α = 1.3 (their App. A.1), so τ550 = β · 0.55^−1.3; the particle properties are the hazy
    /// suburban preset's (continental aerosol). T ≤ 1 is air without aerosol.
    public static func preetham(turbidity: Float) -> NishitaAerosol {
        var a = hazySuburban
        let beta = max(0, 0.04608 * turbidity - 0.04586)
        a.opticalDepth = beta * pow(0.55, -1.3)
        return a
    }

    /// Cornette–Shanks g whose mean scattering cosine ⟨cos θ⟩ — the asymmetry parameter measured
    /// and tabulated for aerosols (OPAC, AERONET) — is `asymmetryParameter`:
    /// solves 3g(4 + g²) / (5(2 + g²)) = ⟨cos θ⟩ (Newton; exact to float precision).
    public static func cornetteShanksG(asymmetryParameter a: Float) -> Float {
        let target = Double(min(max(a, -0.9), 0.9))
        var g = target
        for _ in 0..<20 {
            let f = 3 * g * (4 + g * g) / (5 * (2 + g * g)) - target
            let df = (3 * (4 + 3 * g * g) * (2 + g * g) - 3 * g * (4 + g * g) * 2 * g) / (5 * (2 + g * g) * (2 + g * g))
            g -= f / df
        }
        return Float(g)
    }

    // ── What the shader gets ──────────────────────────────────────────────────────────────

    /// The sky's channel wavelengths (nm): the 680 / 550 / 440 nm samples its Rayleigh
    /// coefficients (and so every channel of the march) are evaluated at.
    public static let channelWavelengths = SIMD3<Float>(680, 550, 440)

    /// The same values with host input sanitised (non-finite → built in; clamped to the physical
    /// range) — what the kernels actually integrate.
    public var sanitized: NishitaAerosol {
        func fin(_ v: Float, _ fallback: Float) -> Float { v.isFinite ? v : fallback }
        let b = NishitaAerosol.builtin
        return NishitaAerosol(
            opticalDepth: min(max(fin(opticalDepth, b.opticalDepth), 0), 5),
            angstromExponent: min(max(fin(angstromExponent, 0), -1), 4),
            scaleHeight: min(max(fin(scaleHeight, b.scaleHeight), 10), 20_000),
            singleScatteringAlbedo: min(max(fin(singleScatteringAlbedo, b.singleScatteringAlbedo), 0), 1),
            phaseG: min(max(fin(phaseG, b.phaseG), -0.95), 0.95))
    }

    /// Sea-level aerosol optical depth per channel: τ550 (λ / 550)^−α at 680 / 550 / 440 nm.
    public var channelOpticalDepth: SIMD3<Double> {
        let s = sanitized
        let l = SIMD3<Double>(NishitaAerosol.channelWavelengths)
        let a = Double(s.angstromExponent)
        return Double(s.opticalDepth) * SIMD3(pow(l.x / 550, -a), pow(l.y / 550, -a), pow(l.z / 550, -a))
    }

    /// Sea-level extinction coefficient per channel (m⁻¹).
    public var seaLevelExtinction: SIMD3<Double> { channelOpticalDepth / Double(sanitized.scaleHeight) }
    /// Sea-level scattering coefficient per channel (m⁻¹).
    public var seaLevelScattering: SIMD3<Double> { seaLevelExtinction * Double(sanitized.singleScatteringAlbedo) }

    /// `SkyUniforms.aerosolA / aerosolB`: (τ550, α, ω₀, g) and (0, 0, 0, H_M) — the parameters
    /// themselves, which the shader evaluates at its channel wavelengths (`NishitaMieAerosol`); zeros
    /// for `.builtin`, which the unchanged kernels keep as their compile-time constants.
    var uniforms: (a: SIMD4<Float>, b: SIMD4<Float>) {
        guard self != .builtin else { return (.zero, .zero) }
        let s = sanitized
        return (SIMD4(s.opticalDepth, s.angstromExponent, s.singleScatteringAlbedo, s.phaseG),
                SIMD4(0, 0, 0, s.scaleHeight))
    }
}

// ── The atmosphere's constants, and the sun through it ──────────────────────────────────────────

/// The engine's Nishita atmosphere on the CPU: the Swift mirror of VolumetricSky.metal's
/// constants (`NishitaAerosolTests` reads the shader's back from the GPU so the two cannot drift),
/// and the direct sun through that air — the ONE transmittance a host should colour its sun disc
/// (`Params.sunColor`) and its directional sun light with, so both redden exactly as the sky does,
/// whatever the aerosol.
public enum NishitaAtmosphere {
    /// Planet radius (m) — kEarthRadius.
    public static let earthRadius: Double = 6360e3
    /// Atmosphere top of the single-scatter march (m) — kAtmosRadius − kEarthRadius.
    public static let singleScatterTop: Double = 60e3
    /// Atmosphere top of the multiple-scattering LUTs (m) — kMSAtmosRadius − kEarthRadius.
    public static let multipleScatteringTop: Double = 100e3
    /// Rayleigh scale height (m) — kRayleighH.
    public static let rayleighScaleHeight: Double = 7994
    /// Rayleigh scattering at sea level (m⁻¹) at 680 / 550 / 440 nm — kBetaR.
    public static let rayleighScattering = SIMD3<Double>(5.8e-6, 13.5e-6, 33.1e-6)
    /// Ozone absorption at the layer peak (m⁻¹) — kBetaO; a tent of ±`ozoneHalfWidth` around
    /// `ozoneCenter`.
    public static let ozoneAbsorption = SIMD3<Double>(0.650e-6, 1.881e-6, 0.085e-6)
    public static let ozoneCenter: Double = 25000
    public static let ozoneHalfWidth: Double = 15000
    /// The built-in aerosol — kBetaM (scattering, grey), its ×1.1 extinction, kMieH, kMieG.
    public static let builtinMieScattering: Float = 21e-6
    public static let builtinMieExtinctionRatio: Float = 1.1
    public static let builtinMieScaleHeight: Float = 1200
    public static let builtinMieG: Float = 0.76

    /// Direct-beam transmittance of sunlight (or moonlight) from space to a point `altitude` m above
    /// the ground, per channel (680 / 550 / 440 nm), for a light toward `toLight` (unit vector,
    /// y up) — Rayleigh + aerosol + ozone, double precision. Exactly 0 when the ray meets the planet
    /// (the light is below the local horizon, the Earth's shadow). `top` = the atmosphere top of the
    /// sky it should match (60 km single scatter, 100 km multiple scattering; immaterial by day).
    public static func transmittance(toLight: SIMD3<Float>, altitude: Float = 1,
                                     aerosol: NishitaAerosol = .builtin,
                                     top: Double = multipleScatteringTop) -> SIMD3<Float> {
        let l = simd_normalize(SIMD3<Double>(toLight))
        let o = SIMD3<Double>(0, earthRadius + max(Double(altitude), 0), 0)
        let rt = earthRadius + top
        // Planet shadow: the ray toward the light meets the ground.
        let b = simd_dot(o, l)
        let cG = simd_dot(o, o) - earthRadius * earthRadius
        if b < 0 && b * b - cG >= 0 { return .zero }
        let c = simd_dot(o, o) - rt * rt
        guard c < 0 else { return SIMD3(repeating: 1) }       // at or above the top
        let tl = -b + (b * b - c).squareRoot()
        let s = aerosol.sanitized
        let betaE = aerosol.seaLevelExtinction
        // Samples crowd toward the observer (t ∝ u²): a grazing path is ~1100 km long but its
        // optical depth sits in the first ~100 km.
        let n = 1024
        var od = SIMD3<Double>(repeating: 0)
        var tPrev = 0.0
        for i in 1...n {
            let u = Double(i) / Double(n)
            let t = tl * u * u
            let h = simd_length(o + l * (0.5 * (tPrev + t))) - earthRadius
            let seg = t - tPrev
            let ozone = max(0, 1 - abs(h - ozoneCenter) / ozoneHalfWidth)
            od += (rayleighScattering * exp(-h / rayleighScaleHeight)
                   + betaE * exp(-h / Double(s.scaleHeight))
                   + ozoneAbsorption * ozone) * seg
            tPrev = t
        }
        return SIMD3<Float>(Float(exp(-od.x)), Float(exp(-od.y)), Float(exp(-od.z)))
    }
}

extension VolumetricCloudRenderer.Params {
    /// The direct sun through THIS sky's air (its aerosol, its atmosphere top), per channel, at
    /// `altitude` m — the one number the sun disc (`sunColor`, given as the colour above the
    /// atmosphere × this) and a host's directional sun light should both be multiplied by, so they
    /// redden with the sky. 0 once the sun is below the horizon.
    public func sunTransmittance(altitude: Float = 1) -> SIMD3<Float> {
        NishitaAtmosphere.transmittance(toLight: -sunDir, altitude: altitude,
                                        aerosol: atmosphere == .nishita ? atmosphereAerosol : .builtin,
                                        top: physicalMultipleScattering ? NishitaAtmosphere.multipleScatteringTop
                                                                        : NishitaAtmosphere.singleScatterTop)
    }
}
