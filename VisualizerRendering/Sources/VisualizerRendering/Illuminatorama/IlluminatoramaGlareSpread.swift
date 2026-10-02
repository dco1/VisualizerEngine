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
/// remainder is the direct image). Why the observer's eye and not a camera lens: a display cannot
/// reproduce a lamp's luminance, so the glare a viewer's own eye would have produced looking at the
/// real room never happens in front of the picture; adding the eye's GSF to the scene-referred image
/// restores it (Spencer, Shirley, Zimmerman & Greenberg, "Physically-Based Glare Effects for Digital
/// Images", SIGGRAPH 1995 — the same rationale). A good camera lens scatters roughly an order of
/// magnitude less (ISO 9358 veiling-glare indices of a few percent) with a similar r⁻²…⁻³ shape.
///
/// **Realisation.** The convex dual-filter pyramid is a LINEAR filter once its threshold and Karis
/// weighting are off: its output is Σ aᵢ·Kᵢ ⊛ hdr, with Kᵢ the fixed impulse response of level i's
/// branch (prefilter + i box downsamples + i tent upsamples) and aᵢ the convex weights the per-level
/// scatters set. The Kᵢ were MEASURED on the GPU through the real kernels
/// (`IlluminatoramaRenderer.measureBloomImpulseResponse`, phase-averaged, tabulated below — and
/// re-measured against this table by the host's gate), so the weights are solved by non-negative
/// least squares against the GSF in the frame's own angular units (`pixelAngle` from the camera's
/// projection), not assumed from the nominal 2^(i+1) reach (that assumption is why DH-0992's α 2.7
/// realised only r^−1.43). The solved Σwᵢ IS the scattered energy η; the tonemap then forms
/// `hdr·(1 − η) + η·bloom` — energy conserving, no threshold, no dial.
///
/// **What it cannot do yet.** The pyramid stops at its coarsest level, so the GSF beyond that reach
/// (several degrees at a normal lens; the 5/θ² skirt carries real energy out to tens of degrees) is
/// left in the direct image. `Fit.targetEnergyBeyondReach` reports how much (DH-0998 extends the reach).
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

    /// Bin edges in mip0 texels: [0, 2^(−2/4)) then quarter octaves up to 1024.
    public static let binEdges: [Double] = [0] + (-2...40).map { pow(2, Double($0) / 4) }
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
                                     center c: SIMD2<Double>, maxRadius: Double = 1024)
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
    public static let levelKernels: [[Double]] = IlluminatoramaGlareSpreadTable.levelKernels
    /// Area-weighted mean radius of each bin, texels (from the same measurement).
    public static let binMeanRadius: [Double] = IlluminatoramaGlareSpreadTable.binMeanRadius

    // ── The fit ──────────────────────────────────────────────────────────────────

    public struct Fit: Equatable {
        /// Absolute weight of each level's kernel (Σ = `energy`).
        public var weights: [Double]
        /// η — the fraction of every pixel's light the glare scatters (Σ weights).
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
        /// Worst |model/target − 1| over the fitted bins.
        public var worstRelativeError: Double
    }

    /// Fitted range for a pyramid of `levels`: from the GSF's own 0.1° validity limit (never under
    /// 2 texels — below that the half-res pyramid has no resolution and the direct image carries the
    /// core) out to where the coarsest level's kernel still has the support to follow the target.
    public static func fitRange(texelDegrees: Double, levels: Int) -> ClosedRange<Double> {
        let lo = max(2.0, 0.1 / max(texelDegrees, 1e-9))
        let hi = max(lo * 2, reachTexels(levels: levels))
        return lo...hi
    }

    /// The radius (mip0 texels) out to which a `levels`-deep pyramid can follow a power-law tail:
    /// 1.5 × the coarsest level's nominal 2^levels reach (measured: its kernel holds ≥ ~10 % of its
    /// peak surface brightness that far out; past it the profile falls off the DH-0998 cliff).
    public static func reachTexels(levels: Int) -> Double { 1.5 * pow(2, Double(levels)) }

    /// How much a sub-range core row counts against an in-range one (see `fit`). Swept 1 → 0.15 on
    /// the three gate geometries: 0.15 keeps the core from piling up (η 0.30–0.35) while letting
    /// the first octave past 0.1° follow the r^−3 term (worst bin 0.36 → 0.19, octave slope
    /// 0.92 → 0.46 at 45 mm); the far field is unchanged (≤ 0.12 / 0.25).
    public static let coreRowWeight: Double = 0.15

    /// Solve the level weights so Σ wᵢ·Kᵢ follows the CIE GSF in this frame's angular units.
    public static func fit(texelRadians: Double, levels requested: Int,
                           age: Double = defaultAge, pigment: Double = defaultPigment) -> Fit {
        let levels = max(1, min(requested, levelKernels.count))
        let deg = texelRadians * 180 / .pi
        let range = fitRange(texelDegrees: deg, levels: levels)
        // Rows: each quarter-octave bin inside the range, residual relative to the target so every
        // octave counts the same (a shape fit, not a fit dominated by the bright core).
        var rows: [[Double]] = [], rhs: [Double] = []
        for b in 0..<binCount {
            let r = binMeanRadius[b]
            // Inside the range's lower bound (sub-0.1° / sub-2-texel) the rows still count, against
            // the GSF held at its 0.1° value: without them NNLS is free to pile the finest level's
            // energy into the core — up to 19 % of every pixel smeared over ~2 px at 45 mm, a
            // whole-image softening no GSF asks for. The clamp is an UNDER-estimate of the eye's
            // true core (the optics, not the scatter), so it never adds what the standard omits.
            guard r >= 1, r <= range.upperBound else { continue }
            let target = cieGSF(thetaDegrees: r * deg, age: age, pigment: pigment) * texelRadians * texelRadians
            let k = r < range.lowerBound ? coreRowWeight : 1
            rows.append((0..<levels).map { k * levelKernels[$0][b] / target })
            rhs.append(k)
        }
        let w = nnls(rows, rhs: rhs)
        var worst = 0.0
        for (row, k) in zip(rows, rhs) where k == 1 { worst = max(worst, abs(zip(row, w).map(*).reduce(0, +) - 1)) }
        let eta = w.reduce(0, +)
        var modelIn = 0.0
        for b in 0..<binCount where binMeanRadius[b] >= range.lowerBound && binMeanRadius[b] <= range.upperBound {
            let area = Double.pi * (binEdges[b + 1] * binEdges[b + 1] - binEdges[b] * binEdges[b])
            modelIn += area * (0..<levels).map { w[$0] * levelKernels[$0][b] }.reduce(0, +)
        }
        let inRange = cieEnergy(fromDegrees: range.lowerBound * deg, toDegrees: range.upperBound * deg,
                                age: age, pigment: pigment)
        let beyond = cieEnergy(fromDegrees: range.upperBound * deg, toDegrees: 90, age: age, pigment: pigment)
        return Fit(weights: w, energy: eta,
                   scatters: eta > 0 ? convexScatters(w.map { $0 / eta }) : [Float](repeating: 0, count: levels),
                   texelDegrees: deg, fitRange: range, modelEnergyInRange: modelIn,
                   targetEnergyInRange: inRange, targetEnergyBeyondReach: beyond,
                   worstRelativeError: worst)
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
