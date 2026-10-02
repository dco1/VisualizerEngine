import Foundation

/// **Physically based glare for the bloom pyramid (Daydream DH-1001).**
///
/// A bloom threshold and an intensity dial are artistic devices. What a bright source actually
/// does is scatter a FIXED FRACTION of its light into a halo whose surface brightness falls with the
/// ANGLE from the source — every source, at every level; only the very bright ones read as glare
/// because of the dynamic range. This file is that model:
///
/// **The glare spread function (GSF)** is the CIE standard one — CIE 146:2002 / CIE 147:2002, "CIE
/// equations for disability glare", the general disability-glare equation of Vos & van den Berg
/// (CIE Collection 135/1, 1999), valid 0.1° ≤ θ ≤ 100°:
///
///     L_veil / E_glare = 10/θ³ + (5/θ² + 0.1·p/θ)·(1 + (A/62.5)⁴) + 0.0025·p      [sr⁻¹, θ in degrees]
///
/// with A the observer's age and p the ocular pigmentation (0 very dark … 0.5 brown … 1 blue-green).
/// It is the light a point source of illuminance E_glare spreads per steradian at angle θ — i.e. the
/// PSF's scattered part, normalised so its integral over the sphere is the SCATTERED FRACTION (the
/// remainder is the direct image). The case for the eye: a display cannot reproduce a lamp's
/// luminance, so the glare a viewer's own eye would have produced looking at the real room never
/// happens in front of the picture; adding the eye's GSF restores it (Spencer, Shirley, Zimmerman &
/// Greenberg, "Physically-Based Glare Effects for Digital Images", SIGGRAPH 1995).
///
/// **A camera lens (Daydream DH-1009 — what Export Quality ▸ Maximum ships).** Danny chose the
/// photograph over the observer: a photo's glare is the LENS's — light scattered by the roughness
/// and contamination of its polished, coated surfaces. `Model.lens(LensScatter)` is that PSF — the
/// 3-parameter Harvey surface-scatter BSDF summed over the lens's scattering surfaces, with the
/// generic parameters Ritt, Schwarz & Eberle MEASURED for double-Gauss camera lenses (Sensors
/// 20(21):6308, 2020; see `LensScatter.doubleGauss`): θ^−1.86 past a 0.12° shoulder, η ≈ 0.025 —
/// ISO 9358 "good" veiling glare, ~12× less than the eye. Same fit, same far tail, same energy
/// closure; only the target changes (`fit(texelRadians:levels:model:)`).
///
/// **Realisation.** The convex dual-filter pyramid is a LINEAR filter once its threshold and Karis
/// weighting are off: its output is Σ aᵢ·Kᵢ ⊛ hdr, with Kᵢ the fixed impulse response of level i's
/// branch (prefilter + i box downsamples + i tent upsamples) and aᵢ the convex weights the per-level
/// scatters set. The Kᵢ were MEASURED on the GPU through the real kernels
/// (`IlluminatoramaRenderer.measureBloomImpulseResponse`, phase-averaged, tabulated below — and
/// re-measured against this table by the host's gate), so the weights are solved by non-negative
/// least squares against the GSF in the frame's own angular units (`pixelAngle` from the camera's
/// projection), not assumed from the nominal 2^(i+1) reach (that assumption is why DH-0992's α 2.7
/// realised only r^−1.43). The solved Σwᵢ IS the pyramid's scattered energy η; the tonemap then
/// forms `hdr·(1 − η − M) + η·bloom + tail` — energy conserving, no threshold, no dial.
///
/// **The far field (Daydream DH-0998).** The pyramid's compact kernels can only follow the GSF out
/// to ≈ 1.5·2^levels mip0 texels (`reachTexels`) — a few degrees — and past that its realised glare
/// fell to 0.4× then 0.02× of the standard: the halo just STOPPED, while the 5/θ² + 0.1p/θ + 0.0025p
/// skirt carries real energy out to tens of degrees (≈ 0.04 of a source's light past 8°). So the
/// GSF is split in two by a smooth ramp S(θ) (`tailRamp`, cubic in log θ from 0.25 to 0.8 of the
/// reach): the pyramid is fitted to GSF·(1 − S), and the rest, GSF·S, is the FAR-FIELD TAIL — an
/// exact, direct convolution in true view angles over a coarse level of the down chain
/// (`illumi_bloom_glare_tail`; level `tailGridLevel`, ≤ ~10⁴ source cells), so it reaches every pixel of
/// the frame, the corner included, with nothing to run out of. The tail is ZERO-PADDED (only the
/// frame's own pixels scatter into it) and energy-closed: the share of a pixel's light that would
/// land outside the frame stays in its direct image, so a uniform field stays exactly uniform
/// (see the shader). Measured (BloomProfile CIE gate, real frame geometries): a centred source's
/// glare is the standard's to the frame corner — every bin past the pyramid's reach within 2.5 %,
/// in-frame energy within 5 %, energy closed to 0.02 %. **Limitation:** the near field (inside
/// ~0.8 of the reach) is the screen-space pyramid, calibrated at the frame centre; off axis a
/// pixel subtends less angle (cos²φ radially, cosφ tangentially), so near a 24 mm lens's corner
/// that part of the halo reads 0.65–0.95× the GSF (the far field there stays exact).
public enum IlluminatoramaGlareSpread {

    // ── The standard ─────────────────────────────────────────────────────────────

    /// CIE 146:2002 general disability-glare equation (see the type note), sr⁻¹ at `thetaDegrees`.
    /// Clamped below at the equation's 0.1° validity limit.
    public static func cieGSF(thetaDegrees: Double, age: Double = defaultAge,
                              pigment: Double = defaultPigment) -> Double {
        let t = max(0.1, thetaDegrees)
        let ageTerm = 1 + pow(age / 62.5, 4)
        return 10 / (t * t * t) + (5 / (t * t) + 0.1 * pigment / t) * ageTerm + 0.0025 * pigment
    }

    /// The CIE standard observer the shipped glare is computed for: a 25-year-old with brown eyes.
    public static let defaultAge: Double = 25
    public static let defaultPigment: Double = 0.5

    // ── The camera lens (Daydream DH-1009) ───────────────────────────────────────

    /// Surface-scatter parameters of a camera lens, in the 3-parameter Harvey model the stray-light
    /// literature fits lenses with. Per scattering SURFACE (the BSDF of one polished, coated glass
    /// face, roughness + contamination), and the lens has `surfaces` of them:
    ///
    ///     BSDF(θ) = b₀ · [1 + (sin θ / l)²]^(s/2),       b₀ = b · (100·l)^s          [sr⁻¹]
    ///
    /// (Harvey & Shack; the form and the b/b₀ relation are Eqs. (9), (13) of Ritt, Schwarz &
    /// Eberle, "Estimation of Lens Stray Light with Regard to the Incapacitation of Imaging
    /// Sensors", Sensors 20(21):6308, 2020 — doi:10.3390/s20216308.) `b` is the BSDF at
    /// sin θ = 0.01, `l` the shoulder (rad) where the flat core turns into the power law of slope `s`.
    public struct LensScatter: Hashable, Sendable {
        public var s: Double
        public var b: Double
        public var l: Double
        public var surfaces: Int
        public init(s: Double, b: Double, l: Double, surfaces: Int) {
            self.s = s; self.b = b; self.l = l; self.surfaces = surfaces
        }
        public var b0: Double { b * pow(100 * l, s) }
        /// Total integrated scatter of ONE surface over the hemisphere — Ritt et al. Eq. (6):
        /// TIS = 2π b 100^s / (s+2) · [(1+l²)^((s+2)/2) − (l²)^((s+2)/2)]   (s ≠ −2).
        public var tisPerSurface: Double {
            if abs(s + 2) < 1e-9 { return 2 * Double.pi * b * pow(100 * l, s) * l * l * log(1 + 1 / (l * l)) }
            let e = (s + 2) / 2
            return 2 * Double.pi * b * pow(100, s) / (s + 2) * (pow(1 + l * l, e) - pow(l * l, e))
        }
        /// The whole lens's scattered fraction over the hemisphere: Σ surfaces' TIS (single scatter).
        public var totalScatter: Double { Double(surfaces) * tisPerSurface }

        /// **The shipped lens.** Ritt, Schwarz & Eberle (2020) MEASURED the focal-plane stray-light
        /// profile of seven COTS camera lenses (25–100 mm, 5–7 elements, MgF₂ / BBAR coated) over
        /// ~6 decades of irradiance, fitted this model per lens, and balanced the fits into ONE
        /// generic parameter set for the class — S = −1.86, B = 0.36 sr⁻¹, B₀ = 6.92, L = 2.04 mrad
        /// (their Eq. (24) ff.) — "valid for double Gauss lenses", the design of every 35 mm SLR
        /// normal lens. Seven elements (their FRED model Fr1, a 50 mm f/1.8 double Gauss) ⇒ 14
        /// scattering surfaces (N_ss = 2·N_oe, their Table 1).
        ///
        /// That gives a total scattered fraction η = 14 × 1.785e-3 = **0.0250** over the hemisphere,
        /// and ≈ 0.020 inside a 25° half-field — which is where ISO 9358's veiling glare index sits
        /// for a lens graded "good" (1.5 – 3 %; ≤ 1.5 % "very good", > 3 % "poor"): the VGI is the
        /// irradiance in the image of a black spot on a uniform field ÷ the field's, i.e. by
        /// reciprocity exactly ∫ PSF_scatter dΩ over the field. A camera lens therefore scatters
        /// ~12× less than the CIE eye (η ≈ 0.30), and with a single, shallower power law
        /// (θ^−1.86 everywhere past a 0.12° shoulder, where the eye has a θ^−3 core): at 1° the
        /// eye's halo is 8.6× the lens's, at 0.2° 51×, at 20° 2.5×.
        public static let doubleGauss = LensScatter(s: -1.86, b: 0.36, l: 2.04e-3, surfaces: 14)
    }

    /// Whose glare the physical bloom is (DH-1009): the human eye's (CIE 146:2002) or a camera
    /// lens's surface scatter (Harvey). Every lane that shipped before DH-1009 used `.eye`;
    /// Export Quality ▸ Maximum ships `.lens(.doubleGauss)` — a photograph, not an observer.
    public enum Model: Hashable, Sendable {
        case eye(age: Double, pigment: Double)
        case lens(LensScatter)

        public static let standardObserver = Model.eye(age: defaultAge, pigment: defaultPigment)
        public static let cameraLens = Model.lens(.doubleGauss)

        /// The scatter PSF, sr⁻¹ — the fraction of a point source's light landing per steradian
        /// at field angle θ (its integral over the sphere is the scattered fraction η).
        ///  • eye: `cieGSF` (clamped below at the equation's 0.1° validity limit).
        ///  • lens: Σ surfaces · BSDF(θ) · cos θ — the BSDF is per PROJECTED solid angle, so the
        ///    light per solid angle carries the obliquity cos θ (that is what makes its hemisphere
        ///    integral the TIS of Ritt Eq. (6)). In the small-angle limit this is their Eq. (11)
        ///    (E_s·f²/P_in, for an aperture-filling source, ν* = 1) written in field angle θ = r/f.
        @inline(__always)
        public func psf(thetaDegrees t: Double) -> Double {
            switch self {
            case let .eye(age, pigment):
                return cieGSF(thetaDegrees: t, age: age, pigment: pigment)
            case let .lens(L):
                guard t < 90 else { return 0 }
                let r = t * .pi / 180, sn = sin(r) / L.l
                return Double(L.surfaces) * L.b0 * pow(1 + sn * sn, L.s / 2) * cos(r)
            }
        }
        /// Below this angle the model is not defined (the eye equation's 0.1°); the lens model is
        /// defined to 0 (its flat Harvey core).
        public var minDegrees: Double {
            if case .eye = self { return 0.1 }
            return 0
        }
        /// Where the far-field tail stops: the CIE equation's 100° validity limit; the lens's
        /// hemisphere (cos θ reaches 0).
        public var maxDegrees: Double {
            if case .eye = self { return 100 }
            return 90
        }
    }

    /// ∫ PSF dΩ between two angles (degrees) — the fraction of a source's light the model
    /// scatters into that annulus. Simpson on log θ, dΩ = 2π sin θ dθ.
    public static func energy(_ model: Model, fromDegrees a: Double, toDegrees b: Double) -> Double {
        guard b > a, a > 0 else { return 0 }
        let n = 1024                                   // even
        let la = log(a), lb = log(b), h = (lb - la) / Double(n)
        var s = 0.0
        for k in 0...n {
            let θ = exp(la + Double(k) * h)
            let f = model.psf(thetaDegrees: θ) * 2 * Double.pi * sin(θ * .pi / 180) * (θ * .pi / 180)
            s += f * ((k == 0 || k == n) ? 1 : (k % 2 == 1 ? 4 : 2))
        }
        return s * h / 3
    }

    /// Energy of the GSF between two angles (degrees), i.e. the fraction of a source's light the eye
    /// scatters into that annulus: ∫ GSF(θ) dΩ, dΩ = 2π sin θ dθ. Simpson on log θ.
    public static func cieEnergy(fromDegrees a: Double, toDegrees b: Double,
                                 age: Double = defaultAge, pigment: Double = defaultPigment) -> Double {
        guard b > a, a > 0 else { return 0 }
        let n = 512                                    // even
        let la = log(a), lb = log(b), h = (lb - la) / Double(n)
        var s = 0.0
        for k in 0...n {
            let θ = exp(la + Double(k) * h)
            let f = cieGSF(thetaDegrees: θ, age: age, pigment: pigment)
                * 2 * Double.pi * sin(θ * .pi / 180) * (θ * .pi / 180)   // dΩ/d(ln θ)
            s += f * ((k == 0 || k == n) ? 1 : (k % 2 == 1 ? 4 : 2))
        }
        return s * h / 3
    }

    // ── Radial bins (shared by the table, the fit and the gates) ─────────────────

    /// Bin edges in mip0 texels: [0, 2^(−2/4)) then quarter octaves up to 2048 (a hero still's
    /// mip0 diagonal is ~2260 texels; its half-diagonal — a centred source's corner — ~1130).
    public static let binEdges: [Double] = [0] + (-2...44).map { pow(2, Double($0) / 4) }
    public static var binCount: Int { binEdges.count - 1 }

    /// Bin index of a radius (texels), nil past the last edge.
    @inline(__always)
    public static func bin(of r: Double) -> Int? {
        if r < binEdges[1] { return 0 }
        let k = Int(floor(4 * log2(r))) + 3                  // edge index of 2^(j/4) is j + 3
        return k >= 1 && k < binCount ? k : nil
    }

    /// Radial profile of an image about `center` (texel coordinates, texel centres at i + 0.5):
    /// per bin the MEAN value, the mean radius and the texel count. A kernel normalised to sum 1
    /// therefore reads as surface brightness per texel².
    public static func radialProfile(_ img: [Float], width w: Int, height h: Int,
                                     center c: SIMD2<Double>, maxRadius: Double = 2048)
        -> (mean: [Double], meanR: [Double], count: [Int]) {
        var sum = [Double](repeating: 0, count: binCount)
        var rs = [Double](repeating: 0, count: binCount)
        var n = [Int](repeating: 0, count: binCount)
        let x0 = max(0, Int(c.x - maxRadius)), x1 = min(w - 1, Int(c.x + maxRadius))
        let y0 = max(0, Int(c.y - maxRadius)), y1 = min(h - 1, Int(c.y + maxRadius))
        guard x1 >= x0, y1 >= y0 else { return (sum, rs, n) }
        for y in y0...y1 {
            let dy = Double(y) + 0.5 - c.y
            let row = y * w
            for x in x0...x1 {
                let dx = Double(x) + 0.5 - c.x
                let r = (dx * dx + dy * dy).squareRoot()
                guard let b = bin(of: r) else { continue }
                sum[b] += Double(img[row + x]); rs[b] += r; n[b] += 1
            }
        }
        return ((0..<binCount).map { n[$0] > 0 ? sum[$0] / Double(n[$0]) : 0 },
                (0..<binCount).map { n[$0] > 0 ? rs[$0] / Double(n[$0]) : 0 }, n)
    }

    // ── The measured level kernels ───────────────────────────────────────────────

    /// Kᵢ's radial profile (mean value per mip0 texel², each kernel summing to 1), per bin of
    /// `binEdges`, for pyramid levels 0…6 at the shipping tent radius 1.0. MEASURED on the GPU —
    /// `HouseRenderBridgeGPUTests_BloomProfile.testGlareLevelKernelTableMatchesTheGPU` re-measures
    /// them and fails if this table drifts from the kernels (it prints a fresh table to paste).
    public static let levelKernels: [[Double]] = IlluminatoramaGlareSpreadTable.levelKernels.map {
        $0.count >= binCount ? $0 : $0 + [Double](repeating: 0, count: binCount - $0.count)
    }
    /// Area-weighted mean radius of each bin, texels (from the same measurement; a bin the table
    /// predates takes the annulus's analytic area-weighted mean ⅔(b³−a³)/(b²−a²)).
    public static let binMeanRadius: [Double] = (0..<binCount).map { b in
        let t = IlluminatoramaGlareSpreadTable.binMeanRadius
        if b < t.count { return t[b] }
        let a = binEdges[b], e = binEdges[b + 1]
        return 2 * (e * e * e - a * a * a) / (3 * (e * e - a * a))
    }

    // ── The fit ──────────────────────────────────────────────────────────────────

    public struct Fit: Equatable {
        /// Absolute weight of each level's kernel (Σ = `energy`).
        public var weights: [Double]
        /// η of the PYRAMID — the fraction of every pixel's light the near-field glare scatters
        /// (Σ weights). The far-field tail scatters up to `tailEnergy` more (see the type note).
        public var energy: Double
        /// The up-chain's per-level convex scatters that realise `weights / energy`.
        public var scatters: [Float]
        /// One mip0 texel's angular size, degrees.
        public var texelDegrees: Double
        /// The radii (mip0 texels) the GSF was fitted over.
        public var fitRange: ClosedRange<Double>
        /// The model's energy scattered INTO the fitted range (Σ over its bins of w·K·annulus area) —
        /// the number to hold against `targetEnergyInRange`. `energy` − this is what the finest
        /// levels leave inside the range's lower bound (within ~0.1° of the source, where the eye's
        /// own optical core is far brighter than the CIE scatter term anyway) or past its reach.
        public var modelEnergyInRange: Double
        /// CIE energy inside / beyond the fitted range.
        public var targetEnergyInRange: Double
        public var targetEnergyBeyondReach: Double
        /// Worst |model/target − 1| over the fitted bins (pyramid + the tail's ramp share).
        public var worstRelativeError: Double
        /// DH-0998 — the far-field tail: the ramp S(θ) runs from `tailStartDegrees` (0) to
        /// `tailEndDegrees` (1), cubic in log θ; the tail carries GSF·S out to `tailMaxDegrees`.
        public var tailStartDegrees: Double
        public var tailEndDegrees: Double
        public var tailMaxDegrees: Double
        /// ∫ GSF·S dΩ to `tailMaxDegrees` — what a source in an INFINITE frame scatters into the tail
        /// (a real frame keeps the part landing outside it in the direct image).
        public var tailEnergy: Double
        /// The down-chain level the tail convolves over (its source grid; it is evaluated on a grid
        /// half that in each axis and bilinearly upsampled — the tail starts ≥ 3 source cells out and
        /// is smooth on that scale: measured far bins 1.1–1.6 % → 1.9–3.3 % off the GSF for 4× less work).
        public var tailGridLevel: Int
        /// η of the whole glare, pyramid + tail (in an infinite frame).
        public var totalEnergy: Double { energy + tailEnergy }
        /// DH-1009 — the scatter model the weights were solved against (and the tail evaluates).
        public var model: Model = .standardObserver
    }

    /// The far-field ramp: 0 at `a`, 1 at `b` (degrees), smoothstep in log θ.
    @inline(__always)
    public static func tailRamp(thetaDegrees t: Double, start a: Double, end b: Double) -> Double {
        if t <= a { return 0 }
        if t >= b { return 1 }
        let x = log(t / a) / log(b / a)
        return x * x * (3 - 2 * x)
    }
    /// Where the ramp sits, as fractions of the pyramid's reach. Swept on the three gate geometries
    /// (prototype NNLS, rows to 2× reach): [0.5, 1] left a 0.79 / 1.25 ripple through the hand-off,
    /// [0.25, 1] ±5 %, [0.25, 0.8] ±6 % with every bin past it exactly the standard — the coarsest
    /// level's soft edge is what the ramp hands over to, so it must END before that edge does.
    public static let tailRampStartFraction: Double = 0.25
    public static let tailRampEndFraction: Double = 0.8
    /// The CIE equation's own upper validity limit; the tail stops here.
    public static let tailMaxDegrees: Double = 100

    /// ∫ GSF(θ)·S(θ) dΩ from `a` to `maxDegrees` — the tail's energy in an infinite frame.
    public static func tailEnergy(startDegrees a: Double, endDegrees b: Double, maxDegrees m: Double = tailMaxDegrees,
                                  age: Double = defaultAge, pigment: Double = defaultPigment) -> Double {
        tailEnergy(.eye(age: age, pigment: pigment), startDegrees: a, endDegrees: b, maxDegrees: m)
    }

    /// ∫ PSF(θ)·S(θ) dΩ from `a` to `maxDegrees` for any model.
    public static func tailEnergy(_ model: Model, startDegrees a: Double, endDegrees b: Double,
                                  maxDegrees m: Double) -> Double {
        guard m > a, a > 0 else { return 0 }
        let n = 1024
        let la = log(a), lb = log(m), h = (lb - la) / Double(n)
        var s = 0.0
        for k in 0...n {
            let θ = exp(la + Double(k) * h)
            let f = model.psf(thetaDegrees: θ) * tailRamp(thetaDegrees: θ, start: a, end: b)
                * 2 * Double.pi * sin(θ * .pi / 180) * (θ * .pi / 180)
            s += f * ((k == 0 || k == n) ? 1 : (k % 2 == 1 ? 4 : 2))
        }
        return s * h / 3
    }

    /// Fitted range for a pyramid of `levels`: from the GSF's own 0.1° validity limit (never under
    /// 2 texels — below that the half-res pyramid has no resolution and the direct image carries the
    /// core) out to where the coarsest level's kernel still has the support to follow the target.
    public static func fitRange(texelDegrees: Double, levels: Int,
                                minDegrees: Double = Model.standardObserver.minDegrees) -> ClosedRange<Double> {
        let lo = max(2.0, minDegrees / max(texelDegrees, 1e-9))
        let hi = max(lo * 2, reachTexels(levels: levels))
        return lo...hi
    }

    /// The radius (mip0 texels) out to which a `levels`-deep pyramid can follow a power-law tail:
    /// 1.5 × the coarsest level's nominal 2^levels reach (measured: its kernel holds ≥ ~10 % of its
    /// peak surface brightness that far out; past it the pyramid alone fell off the DH-0998 cliff —
    /// 0.4× then 0.02× of the GSF — which is why the far-field tail takes over before it).
    public static func reachTexels(levels: Int) -> Double { 1.5 * pow(2, Double(levels)) }

    /// How much a sub-range core row counts against an in-range one (see `fit`). Swept 1 → 0.15 on
    /// the three gate geometries: 0.15 keeps the core from piling up (η 0.30–0.35) while letting
    /// the first octave past 0.1° follow the r^−3 term (worst bin 0.36 → 0.19, octave slope
    /// 0.92 → 0.46 at 45 mm); the far field is unchanged (≤ 0.12 / 0.25).
    public static let coreRowWeight: Double = 0.15

    /// Solve the level weights so Σ wᵢ·Kᵢ + the far-field tail follows the CIE GSF in this frame's
    /// angular units: the pyramid is fitted to GSF·(1 − S), the tail IS GSF·S (DH-0998).
    public static func fit(texelRadians: Double, levels requested: Int,
                           age: Double = defaultAge, pigment: Double = defaultPigment) -> Fit {
        fit(texelRadians: texelRadians, levels: requested, model: .eye(age: age, pigment: pigment))
    }

    /// The same solve for any scatter model (DH-1009: the eye's GSF or a camera lens's PSF).
    public static func fit(texelRadians: Double, levels requested: Int, model: Model) -> Fit {
        let levels = max(1, min(requested, levelKernels.count))
        let deg = texelRadians * 180 / .pi
        let range = fitRange(texelDegrees: deg, levels: levels, minDegrees: model.minDegrees)
        let reach = reachTexels(levels: levels)
        let ta = tailRampStartFraction * reach * deg, tb = tailRampEndFraction * reach * deg
        // Rows: each quarter-octave bin out to TWICE the reach (where the coarsest kernel is gone),
        // residual relative to the GSF so every octave counts the same (a shape fit, not a fit
        // dominated by the bright core). Past the ramp the pyramid's share is 0 — rows there hold
        // the coarsest level's soft edge from spilling over the tail.
        var rows: [[Double]] = [], rhs: [Double] = [], share: [Double] = []
        for b in 0..<binCount {
            let r = binMeanRadius[b]
            // Inside the range's lower bound (sub-0.1° / sub-2-texel) the rows still count, against
            // the GSF held at its 0.1° value: without them NNLS is free to pile the finest level's
            // energy into the core — up to 19 % of every pixel smeared over ~2 px at 45 mm, a
            // whole-image softening no GSF asks for. The clamp is an UNDER-estimate of the eye's
            // true core (the optics, not the scatter), so it never adds what the standard omits.
            guard r >= 1, r <= 2 * reach else { continue }
            let target = model.psf(thetaDegrees: r * deg) * texelRadians * texelRadians
            let k = r < range.lowerBound ? coreRowWeight : 1
            let s = tailRamp(thetaDegrees: r * deg, start: ta, end: tb)
            rows.append((0..<levels).map { k * levelKernels[$0][b] / target })
            rhs.append(k * (1 - s)); share.append(k < 1 ? -1 : s)
        }
        let w = nnls(rows, rhs: rhs)
        var worst = 0.0
        for (row, s) in zip(rows, share) where s >= 0 {
            worst = max(worst, abs(zip(row, w).map(*).reduce(0, +) + s - 1))
        }
        let eta = w.reduce(0, +)
        // Energy into the gated span [lower bound, 2·reach]: the pyramid's plus the tail's (GSF·S).
        var modelIn = 0.0
        for b in 0..<binCount where binMeanRadius[b] >= range.lowerBound && binMeanRadius[b] <= 2 * reach {
            let area = Double.pi * (binEdges[b + 1] * binEdges[b + 1] - binEdges[b] * binEdges[b])
            let r = binMeanRadius[b]
            let tail = model.psf(thetaDegrees: r * deg) * texelRadians * texelRadians
                * tailRamp(thetaDegrees: r * deg, start: ta, end: tb)
            modelIn += area * ((0..<levels).map { w[$0] * levelKernels[$0][b] }.reduce(0, +) + tail)
        }
        let inRange = energy(model, fromDegrees: range.lowerBound * deg, toDegrees: 2 * reach * deg)
        let beyond = energy(model, fromDegrees: range.upperBound * deg, toDegrees: 90)
        // The tail's grid: the down level whose cells are 1/12 of the reach (ramp start = 3 cells),
        // so a hero still convolves 120 × 75 cells and the BloomProfile lamp 75 × 56.
        let grid = max(0, levels - 3)
        return Fit(weights: w, energy: eta,
                   scatters: eta > 0 ? convexScatters(w.map { $0 / eta }) : [Float](repeating: 0, count: levels),
                   texelDegrees: deg, fitRange: range.lowerBound...(2 * reach), modelEnergyInRange: modelIn,
                   targetEnergyInRange: inRange, targetEnergyBeyondReach: beyond,
                   worstRelativeError: worst,
                   tailStartDegrees: ta, tailEndDegrees: tb, tailMaxDegrees: model.maxDegrees,
                   tailEnergy: tailEnergy(model, startDegrees: ta, endDegrees: tb, maxDegrees: model.maxDegrees),
                   tailGridLevel: grid, model: model)
    }

    /// The per-level convex scatters that give level i the weight aᵢ (Σ aᵢ = 1): level i's down
    /// texture ends up weighted (1−sᵢ)·Π_{k<i} s_k, so sᵢ = 1 − aᵢ / (1 − Σ_{k<i} a_k).
    public static func convexScatters(_ a: [Double]) -> [Float] {
        var remaining = 1.0
        var s: [Float] = []
        for ai in a {
            s.append(remaining > 1e-9 ? Float(min(1, max(0, 1 - ai / remaining))) : 1)
            remaining -= ai
        }
        return s
    }

    /// Lawson–Hanson non-negative least squares: argmin ‖A·x − b‖, x ≥ 0. A is rows × n (n small).
    public static func nnls(_ A: [[Double]], rhs b: [Double]) -> [Double] {
        guard let n = A.first?.count, n > 0 else { return [] }
        var x = [Double](repeating: 0, count: n)
        var passive = [Bool](repeating: false, count: n)
        func gradient(_ x: [Double]) -> [Double] {
            var g = [Double](repeating: 0, count: n)
            for (row, bi) in zip(A, b) {
                let res = bi - zip(row, x).map(*).reduce(0, +)
                for j in 0..<n { g[j] += row[j] * res }
            }
            return g
        }
        func solvePassive() -> [Double] {
            let idx = (0..<n).filter { passive[$0] }
            let m = idx.count
            var M = [[Double]](repeating: [Double](repeating: 0, count: m + 1), count: m)
            for (row, bi) in zip(A, b) {
                for (p, j) in idx.enumerated() {
                    for (q, k) in idx.enumerated() { M[p][q] += row[j] * row[k] }
                    M[p][m] += row[j] * bi
                }
            }
            // Gaussian elimination with partial pivoting.
            for c in 0..<m {
                var piv = c
                for r in c..<m where abs(M[r][c]) > abs(M[piv][c]) { piv = r }
                M.swapAt(c, piv)
                let d = M[c][c]
                guard abs(d) > 1e-300 else { continue }
                for r in 0..<m where r != c {
                    let f = M[r][c] / d
                    if f != 0 { for k in c...m { M[r][k] -= f * M[c][k] } }
                }
            }
            var z = [Double](repeating: 0, count: n)
            for (p, j) in idx.enumerated() { z[j] = abs(M[p][p]) > 1e-300 ? M[p][m] / M[p][p] : 0 }
            return z
        }
        for _ in 0..<(3 * n + 10) {
            let g = gradient(x)
            guard let j = (0..<n).filter({ !passive[$0] && g[$0] > 1e-12 }).max(by: { g[$0] < g[$1] }) else { break }
            passive[j] = true
            for _ in 0..<(3 * n + 10) {
                let z = solvePassive()
                if (0..<n).allSatisfy({ !passive[$0] || z[$0] > 0 }) { x = z; break }
                var alpha = 1.0
                for k in 0..<n where passive[k] && z[k] <= 0 {
                    alpha = min(alpha, x[k] / max(x[k] - z[k], 1e-300))
                }
                for k in 0..<n { x[k] += alpha * (z[k] - x[k]) }
                for k in 0..<n where passive[k] && x[k] <= 1e-15 { passive[k] = false; x[k] = 0 }
            }
        }
        return x
    }
}
