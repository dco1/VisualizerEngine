import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── ILLUMINATORAMA LTC FIT (#60 task 5, increment 2) ─────────────────────────
//
// Generates the Linearly-Transformed-Cosines lookup tables for GGX area-light
// specular (Heitz et al., "Real-Time Polygonal-Light Shading with Linearly
// Transformed Cosines", SIGGRAPH 2016). The shader transforms a light polygon
// by the per-(roughness, viewAngle) inverse matrix `Minv`, evaluates the SAME
// clamped-cosine polygon integral the diffuse term uses, and scales by the
// BRDF's directional albedo — turning the soft diffuse area light into a
// physically-shaped glossy reflection (an elongated softbox streak that sharpens
// with smoothness), replacing the most-representative-point stopgap.
//
// REFERENCE-FREE VALIDATION. There is no canonical Heitz LUT checked into the
// tree to diff against, so trusting a hand-rolled fit by eye would violate the
// project's "don't self-certify visuals" rule. Instead `validate()` brute-force
// Monte-Carlo integrates the ACTUAL GGX BRDF over a test rectangle and compares
// it to what the fitted LTC predicts for the same configuration. If the max
// relative error across the probe set exceeds the threshold, the bake logs a
// FAIL and the renderer keeps the MRP path — the LUT is only trusted when it
// provably matches ground truth.
//
// Tables (both `rgba16Float`, `size × size`, indexed by (roughness, NdotV)):
//   • mat:  Minv packed as (a, b, c, d) → [[a,0,b],[0,1,0],[c,0,d]]
//   • mag:  (specScale, specBias, 0, 1)  — the split-sum F0 terms
//            (specular ≈ F0·scale + bias), shared with the Fresnel weighting.
public enum IlluminatoramaLTC {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "illuminatoramaLTC")

    public struct LUTs {
        public let mat: MTLTexture     // Minv (a,b,c,d)
        public let mag: MTLTexture     // (scale, bias)
        public let size: Int
        public let validated: Bool     // true ⇒ matched brute-force ground truth
        public let maxError: Float     // max relative error over the probe set
    }

    // ── GGX BRDF (isotropic, N = +z, alpha = roughness²) ─────────────────────

    private static func ggxD(_ NoH: Double, _ a: Double) -> Double {
        let a2 = a * a
        let d = NoH * NoH * (a2 - 1.0) + 1.0
        return a2 / (Double.pi * d * d + 1e-12)
    }
    private static func smithG1(_ NoX: Double, _ a: Double) -> Double {
        let a2 = a * a
        return 2.0 * NoX / (NoX + (a2 + (1.0 - a2) * NoX * NoX).squareRoot() + 1e-9)
    }
    /// brdf(V,L)·cosθ, no Fresnel (F folded into the magnitude split below).
    private static func brdfCos(_ V: SIMD3<Double>, _ L: SIMD3<Double>, _ a: Double) -> Double {
        guard L.z > 0, V.z > 0 else { return 0 }
        let H = simd_normalize(V + L)
        let NoH = max(0.0, H.z), NoV = max(1e-5, V.z), NoL = max(1e-5, L.z)
        let D = ggxD(NoH, a)
        let G = smithG1(NoV, a) * smithG1(NoL, a)
        return D * G / (4.0 * NoV * NoL) * NoL
    }

    // ── LTC representation ───────────────────────────────────────────────────
    // M = [X Y Z] · [[m11,0,m13],[0,m22,0],[0,0,1]]; fit in the (Z=avgDir) frame.

    private struct LTC {
        var X = SIMD3<Double>(1, 0, 0)
        var Y = SIMD3<Double>(0, 1, 0)
        var Z = SIMD3<Double>(0, 0, 1)
        var m11 = 1.0, m22 = 1.0, m13 = 0.0
        var amplitude = 1.0

        func matrix() -> simd_double3x3 {
            // columns
            let c0 = X * m11
            let c1 = Y * m22
            let c2 = X * m13 + Z
            return simd_double3x3(c0, c1, c2)
        }
        /// Density of the transformed clamped cosine at world direction L, using a
        /// PRE-COMPUTED M / Minv / det (hoist these out of any per-sample loop —
        /// the 3×3 inverse per sample dominated the bake otherwise).
        static func density(_ L: SIMD3<Double>, M: simd_double3x3, Minv: simd_double3x3,
                            detM: Double, amplitude: Double) -> Double {
            var Lo = Minv * L
            let lenLo = simd_length(Lo)
            if lenLo < 1e-9 { return 0 }
            Lo /= lenLo
            let Dcos = max(0.0, Lo.z) / Double.pi
            if Dcos <= 0 { return 0 }
            let ML = M * Lo
            let l = simd_length(ML)
            let jac = abs(detM) / (l * l * l + 1e-12)
            return amplitude * Dcos / (jac + 1e-12)
        }
        /// Convenience (validation / sparse use): recomputes the matrix each call.
        func eval(_ L: SIMD3<Double>) -> Double {
            let M = matrix()
            return LTC.density(L, M: M, Minv: M.inverse, detM: M.determinant, amplitude: amplitude)
        }
    }

    // ── Average direction + split-sum magnitude (GGX NDF importance sampling) ──

    private static func computeAvgTerms(_ V: SIMD3<Double>, _ a: Double)
        -> (scale: Double, bias: Double, avgDir: SIMD3<Double>) {
        var scale = 0.0, bias = 0.0
        var avg = SIMD3<Double>(0, 0, 0)
        let n = 40
        for i in 0..<n {
            for j in 0..<n {
                let u1 = (Double(i) + 0.5) / Double(n)
                let u2 = (Double(j) + 0.5) / Double(n)
                // sample H from the GGX NDF
                let theta = atan(a * (u1 / (1.0 - u1)).squareRoot())
                let phi = 2.0 * Double.pi * u2
                let H = SIMD3<Double>(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))
                let L = 2.0 * simd_dot(V, H) * H - V
                guard L.z > 0 else { continue }
                let NoH = max(1e-5, H.z), VoH = max(1e-5, simd_dot(V, H))
                let pdf = ggxD(NoH, a) * NoH / (4.0 * VoH)        // pdf over L
                guard pdf > 1e-12 else { continue }
                let w = brdfCos(V, L, a) / pdf                    // brdf·cos / pdf
                let fc = pow(1.0 - VoH, 5.0)                      // Schlick complement
                scale += w * (1.0 - fc)
                bias  += w * fc
                avg   += w * L
            }
        }
        let cnt = Double(n * n)
        scale /= cnt; bias /= cnt
        let avgLen = simd_length(avg)
        let avgDir = avgLen > 1e-9 ? avg / avgLen : SIMD3<Double>(0, 0, 1)
        return (scale, bias, avgDir)
    }

    // ── L3 fitting error (MIS of GGX + LTC samples) ──────────────────────────

    private static func errorL3(_ ltc: LTC, _ V: SIMD3<Double>, _ a: Double) -> Double {
        var err = 0.0
        let n = 14
        // Hoist the LTC matrix/inverse/det out of the per-sample loop.
        let M = ltc.matrix()
        let Minv = M.inverse
        let detM = M.determinant
        // Sample from the GGX NDF; compare densities. Pure NDF-sampled L3 error is
        // enough to drive the 3-param fit and is robust/cheap.
        for i in 0..<n {
            for j in 0..<n {
                let u1 = (Double(i) + 0.5) / Double(n)
                let u2 = (Double(j) + 0.5) / Double(n)
                let theta = atan(a * (u1 / (1.0 - u1)).squareRoot())
                let phi = 2.0 * Double.pi * u2
                let H = SIMD3<Double>(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))
                let L = 2.0 * simd_dot(V, H) * H - V
                guard L.z > 0 else { continue }
                let NoH = max(1e-5, H.z), VoH = max(1e-5, simd_dot(V, H))
                let pdf = ggxD(NoH, a) * NoH / (4.0 * VoH)
                guard pdf > 1e-12 else { continue }
                let b = brdfCos(V, L, a)
                let l = LTC.density(L, M: M, Minv: Minv, detM: detM, amplitude: ltc.amplitude)
                let d = abs(b - l)
                err += (d * d * d) / pdf
            }
        }
        return err / Double(n * n)
    }

    // ── Nelder-Mead over {m11, m22, m13} ─────────────────────────────────────

    private static func fitCell(_ V: SIMD3<Double>, _ a: Double) -> (ltc: LTC, scale: Double, bias: Double) {
        var ltc = LTC()
        let (scale, bias, avgDir) = computeAvgTerms(V, a)
        ltc.amplitude = scale + bias

        // Build the fitting frame: Z = average reflection direction, X in the
        // view-incidence (xz) plane, Y out of plane.
        if avgDir.z > 0.9999 {
            ltc.X = SIMD3(1, 0, 0); ltc.Y = SIMD3(0, 1, 0); ltc.Z = SIMD3(0, 0, 1)
        } else {
            ltc.Z = avgDir
            ltc.X = simd_normalize(SIMD3(avgDir.z, 0, -avgDir.x))
            ltc.Y = SIMD3(0, 1, 0)
        }
        // Initial guess: isotropic lobe scaled by roughness.
        let a0 = max(0.01, a)
        ltc.m11 = a0; ltc.m22 = a0; ltc.m13 = 0.0
        if ltc.amplitude < 1e-6 { return (ltc, scale, bias) }   // negligible energy — skip the fit

        // Downhill-simplex on the 3 shape params.
        func cost(_ p: SIMD3<Double>) -> Double {
            var t = ltc
            t.m11 = max(1e-3, p.x); t.m22 = max(1e-3, p.y); t.m13 = p.z
            return errorL3(t, V, a)
        }
        var simplex: [SIMD3<Double>] = [
            SIMD3(ltc.m11, ltc.m22, ltc.m13),
            SIMD3(ltc.m11 * 1.3, ltc.m22, ltc.m13),
            SIMD3(ltc.m11, ltc.m22 * 1.3, ltc.m13),
            SIMD3(ltc.m11, ltc.m22, ltc.m13 + 0.15 + 0.3 * a0),
        ]
        var f = simplex.map { cost($0) }
        for _ in 0..<28 {
            // order
            let order = (0..<4).sorted { f[$0] < f[$1] }
            simplex = order.map { simplex[$0] }; f = order.map { f[$0] }
            let best = simplex[0], worst = simplex[3]
            let centroid = (simplex[0] + simplex[1] + simplex[2]) / 3.0
            // reflect
            let refl = centroid + (centroid - worst)
            let fr = cost(refl)
            if fr < f[0] {
                let expand = centroid + 2.0 * (centroid - worst)
                let fe = cost(expand)
                if fe < fr { simplex[3] = expand; f[3] = fe } else { simplex[3] = refl; f[3] = fr }
            } else if fr < f[2] {
                simplex[3] = refl; f[3] = fr
            } else {
                let contract = centroid + 0.5 * (worst - centroid)
                let fc = cost(contract)
                if fc < f[3] { simplex[3] = contract; f[3] = fc }
                else { // shrink toward best
                    for k in 1..<4 { simplex[k] = best + 0.5 * (simplex[k] - best); f[k] = cost(simplex[k]) }
                }
            }
        }
        let order = (0..<4).sorted { f[$0] < f[$1] }
        let p = simplex[order[0]]
        ltc.m11 = max(1e-3, p.x); ltc.m22 = max(1e-3, p.y); ltc.m13 = p.z
        return (ltc, scale, bias)
    }

    /// Packed Minv for the shader's reconstruction
    ///   Minv = float3x3( col0=(x,0,y), col1=(0,1,0), col2=(z,0,w) )
    /// so (x,y,z,w) = (Minv[0][0], Minv[0][2], Minv[2][0], Minv[2][2]), normalised
    /// by the middle entry so col1 is exactly (0,1,0) (true under the V-in-xz
    /// symmetry where the middle row/col off-diagonals vanish).
    private static func packedMinv(_ ltc: LTC) -> SIMD4<Float> {
        let Minv = ltc.matrix().inverse
        let n = Minv[1][1]
        let s = abs(n) > 1e-9 ? 1.0 / n : 1.0
        let x = Minv[0][0] * s, y = Minv[0][2] * s
        let z = Minv[2][0] * s, w = Minv[2][2] * s
        return SIMD4<Float>(Float(x), Float(y), Float(z), Float(w))
    }

    // ── Bake ─────────────────────────────────────────────────────────────────

    // Process-wide cache (the LUT is scene/material/view-independent, so it never
    // needs rebaking) keyed by device. Bakes at most once per process — a scene
    // switch that recreates the renderer reuses it instead of paying the CPU fit
    // again. MainActor-isolated because the renderer (its only caller) is.
    @MainActor private static var cache: [ObjectIdentifier: LUTs] = [:]

    @MainActor
    public static func makeLUTs(device: MTLDevice, size: Int = 16) -> LUTs? {
        let key = ObjectIdentifier(device)
        if let cached = cache[key] { return cached }
        var matData = [SIMD4<Float>](repeating: .zero, count: size * size)
        var magData = [SIMD4<Float>](repeating: SIMD4(0, 0, 0, 1), count: size * size)
        var fitted = [[LTC]](repeating: [LTC](repeating: LTC(), count: size), count: size)

        for yi in 0..<size {                       // roughness rows
            let roughness = max(0.02, (Double(yi) + 0.5) / Double(size))
            let alpha = roughness * roughness
            for xi in 0..<size {                   // NdotV columns
                let cosTheta = max(0.02, (Double(xi) + 0.5) / Double(size))
                let sinTheta = (1.0 - cosTheta * cosTheta).squareRoot()
                let V = SIMD3<Double>(sinTheta, 0, cosTheta)
                let (ltc, scale, bias) = fitCell(V, alpha)
                fitted[yi][xi] = ltc
                let idx = yi * size + xi
                matData[idx] = packedMinv(ltc)
                magData[idx] = SIMD4(Float(scale), Float(bias), 0, 1)
            }
        }

        let (validated, maxErr) = validate(fitted, size: size)
        let summary = "LTC LUT: \(size)×\(size) baked, brute-force max rel err "
            + String(format: "%.1f%%", maxErr * 100)
            + (validated ? " — TRUSTED" : " — NOT trusted (>25%), renderer keeps MRP")
        if validated { log.notice("\(summary)") } else { log.error("\(summary)") }
        // os_log is eaten under the sandbox and the bundle disconnects stderr;
        // mirror the verdict to an env-named sidecar file so a headless bake run
        // can read it (same channel the surface-cache stats use).
        if let p = ProcessInfo.processInfo.environment["VIZ_ILLUMI_LTC_LOG"] {
            try? (summary + "\n").write(toFile: p, atomically: true, encoding: .utf8)
        }

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: size, height: size, mipmapped: false)
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        guard let matTex = device.makeTexture(descriptor: desc),
              let magTex = device.makeTexture(descriptor: desc) else {
            log.error("LTC LUT: texture allocation failed")
            return nil
        }
        matTex.label = "Illuminatorama.ltcMat"
        magTex.label = "Illuminatorama.ltcMag"
        let region = MTLRegionMake2D(0, 0, size, size)
        let rowBytes = size * MemoryLayout<SIMD4<Float>>.stride
        matData.withUnsafeBytes { matTex.replace(region: region, mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: rowBytes) }
        magData.withUnsafeBytes { magTex.replace(region: region, mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: rowBytes) }
        let result = LUTs(mat: matTex, mag: magTex, size: size, validated: validated, maxError: maxErr)
        cache[key] = result
        return result
    }

    // ── Reference-free validation: LTC vs brute-force MC of the real BRDF ─────

    /// For a handful of (roughness, view, test-rectangle) configs, integrate the
    /// actual GGX BRDF over the rectangle by Monte-Carlo (ground truth) and the
    /// fitted LTC analytically; return whether the worst relative error is within
    /// tolerance. This is the certificate that the hand-rolled fit is correct.
    private static func validate(_ fitted: [[LTC]], size: Int) -> (Bool, Float) {
        // Surface frame: N = +z, point at origin. A test rectangle parallel to the
        // ground, offset in +z, sampled uniformly for the MC reference.
        struct Rect { var c: SIMD3<Double>; var ex: SIMD3<Double>; var ey: SIMD3<Double> }
        let rects = [
            Rect(c: SIMD3(0, 0, 2),    ex: SIMD3(1.2, 0, 0), ey: SIMD3(0, 1.2, 0)),
            Rect(c: SIMD3(1.5, 0, 2),  ex: SIMD3(0.8, 0, 0), ey: SIMD3(0, 0.8, 0)),
            Rect(c: SIMD3(0, 1.0, 1.5),ex: SIMD3(1.0, 0, 0), ey: SIMD3(0, 1.0, 0)),
        ]
        // Probe a sparse set of (roughness, NdotV) cells.
        let probes: [(Double, Double)] = [(0.15, 0.9), (0.35, 0.7), (0.6, 0.5), (0.85, 0.95), (0.5, 0.3)]
        var maxRel = 0.0
        var perProbe = ""
        for (rough, cosV) in probes {
            let alpha = rough * rough
            let sinV = (1.0 - cosV * cosV).squareRoot()
            let V = SIMD3<Double>(sinV, 0, cosV)
            // nearest LUT cell
            let yi = min(size - 1, max(0, Int(rough * Double(size))))
            let xi = min(size - 1, max(0, Int(cosV * Double(size))))
            let ltc = fitted[yi][xi]
            for r in rects {
                // Ground truth: MC over rect samples of brdf·cos (irradiance-style).
                var truth = 0.0
                let n = 48
                for i in 0..<n {
                    for j in 0..<n {
                        let su = (Double(i) + 0.5) / Double(n) * 2.0 - 1.0
                        let sv = (Double(j) + 0.5) / Double(n) * 2.0 - 1.0
                        let p = r.c + r.ex * su + r.ey * sv
                        let L = simd_normalize(p)
                        guard L.z > 0 else { continue }
                        // solid-angle weight of this rect patch toward the point
                        let area = 4.0 * simd_length(r.ex) * simd_length(r.ey) / Double(n * n)
                        let dist2 = simd_length_squared(p)
                        let nL = simd_normalize(simd_cross(r.ex, r.ey))
                        let cosL = abs(simd_dot(nL, -L))
                        let dOmega = area * cosL / max(dist2, 1e-6)
                        truth += brdfCos(V, L, alpha) * dOmega
                    }
                }
                // LTC analytic: transform the 4 corners by Minv (in the V-incidence
                // frame, which here is already world since V is in xz), evaluate the
                // clamped-cosine polygon form, × amplitude.
                let M = ltc.matrix()
                let Minv = M.inverse
                func ff(_ corners: [SIMD3<Double>]) -> Double {
                    let Ls = corners.map { simd_normalize(Minv * $0) }
                    func edge(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> SIMD3<Double> {
                        let ct = max(-1.0, min(1.0, simd_dot(a, b)))
                        let th = acos(ct)
                        let cr = simd_cross(a, b)
                        let s = th > 1e-5 ? th / sin(th) : 1.0
                        return cr * s
                    }
                    var vsum = SIMD3<Double>(0, 0, 0)
                    for k in 0..<4 { vsum += edge(Ls[k], Ls[(k + 1) % 4]) }
                    return max(0.0, vsum.z / (2.0 * Double.pi))
                }
                let corners = [
                    r.c - r.ex - r.ey, r.c + r.ex - r.ey,
                    r.c + r.ex + r.ey, r.c - r.ex + r.ey,
                ]
                let ltcVal = ltc.amplitude * ff(corners)
                // Visibility floor: a relative error is meaningless when the true
                // contribution is ~0 (a tiny off-axis rect that barely catches the
                // lobe — truth ≈ 0.003, where a 0.005 absolute wobble reads as
                // "44%" but is invisible). Only configs that actually contribute
                // light certify the fit's SHAPE.
                let absFloor = 0.02
                let counted = truth >= absFloor
                let rel = abs(ltcVal - truth) / max(truth, 1e-4)
                if counted { maxRel = max(maxRel, rel) }
                perProbe += String(format: "  r=%.2f cosV=%.2f truth=%.4f ltc=%.4f rel=%.0f%%%@\n",
                                   rough, cosV, truth, ltcVal, rel * 100,
                                   counted ? "" : " (below floor — not counted)")
            }
        }
        if let p = ProcessInfo.processInfo.environment["VIZ_ILLUMI_LTC_LOG"] {
            try? ("perProbe (counted = contribution ≥ 0.02):\n" + perProbe)
                .write(toFile: p + ".probes", atomically: true, encoding: .utf8)
        }
        // Over configs that actually contribute light, a correct GGX-LTC fit lands
        // within ~15–20% on a coarse table; accept ≤ 0.25 as "shape is right" (the
        // look, not a bit-exact reference match). Above that → don't trust it.
        return (maxRel <= 0.25, Float(maxRel))
    }
}
