import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// **Daydream DH-1308 — the path lane's glossy vertex BSDF.**
///
/// With `kPathFlagGlossyVertices` a path vertex past the primary samples the layered stack — coat,
/// GGX specular (VNDF), Lambert — one lobe picked ∝ its albedo (IlluminatoramaBSDF.h, "The path
/// lane's VERTEX BSDF"). Two halves:
///
/// • **CPU reference (GPU-free, runs anywhere).** The shader's lobe math transcribed to Double —
///   GGX D, exact Smith G1, the LUT's IBL-remapped G, the Dupuy–Benyoub VNDF sampler — with the
///   DFG LUT's (A, B) integrated on the CPU exactly as `illumi_dfg_bake` does. It proves the
///   DESIGN: the sampler and its pdf agree, the one-sample lobe pick conserves energy (white
///   furnace = 1 for a white dielectric, a white metal, with and without a coat, at every
///   roughness), the lobe is reciprocal, a coloured metal reflects its F0, the MIS weights the
///   kernel uses against a window (area light) and the sun (a cone) integrate to the reference,
///   and the material lane round-trips.
/// • **GPU probe (the SHIPPED shader code).** `IlluminatoramaBSDF.h` compiled from source with the
///   engine's own LUT bake, the same furnace run on the GPU — the transcription cannot drift.
final class IlluminatoramaPathGlossyVertexTests: XCTestCase {

    // ── CPU transcription of the shader (Double) ─────────────────────────────────────────────
    typealias V3 = SIMD3<Double>

    static func ggxD(_ alpha: Double, _ nh: Double) -> Double {
        let a2 = alpha * alpha, d = nh * nh * (a2 - 1) + 1
        return a2 / max(.pi * d * d, 1e-30)
    }
    static func smithG1(_ alpha: Double, _ c0: Double) -> Double {
        let c = max(c0, 1e-6), a2 = alpha * alpha
        return 2 * c / (c + (a2 + (1 - a2) * c * c).squareRoot())
    }
    static func gIBL(_ nv: Double, _ nl: Double, _ r: Double) -> Double {
        let k = r * r * 0.5
        return (nv / (nv * (1 - k) + k)) * (nl / (nl * (1 - k) + k))
    }
    static func radInv(_ i: UInt32) -> Double {
        var b = i
        b = (b << 16) | (b >> 16)
        b = ((b & 0x55555555) << 1) | ((b & 0xAAAAAAAA) >> 1)
        b = ((b & 0x33333333) << 2) | ((b & 0xCCCCCCCC) >> 2)
        b = ((b & 0x0F0F0F0F) << 4) | ((b & 0xF0F0F0F0) >> 4)
        b = ((b & 0x00FF00FF) << 8) | ((b & 0xFF00FF00) >> 8)
        return Double(b) * 2.3283064365386963e-10
    }
    /// `illumi_dfg_bake`'s (A, B) at (N·V, roughness): GGX-NDF importance sampling, IBL G.
    static func lutAB(_ mu: Double, _ r: Double, samples: Int = 4096) -> (Double, Double) {
        let v = V3((1 - mu * mu).squareRoot(), 0, mu)
        let a = r * r
        var sA = 0.0, sB = 0.0
        for i in 0..<samples {
            let x1 = Double(i) / Double(samples), x2 = radInv(UInt32(i))
            let phi = 2 * .pi * x1
            let ct = ((1 - x2) / (1 + (a * a - 1) * x2)).squareRoot(), st = max(0, 1 - ct * ct).squareRoot()
            let h = V3(cos(phi) * st, sin(phi) * st, ct)
            let l = simd_normalize(2 * simd_dot(v, h) * h - v)
            let nl = max(l.z, 0), nh = max(h.z, 0), vh = max(simd_dot(v, h), 0)
            if nl > 0 {
                let gv = gIBL(mu, nl, r) * vh / max(nh * mu, 1e-6)
                let fc = pow(1 - vh, 5)
                sA += (1 - fc) * gv; sB += fc * gv
            }
        }
        return (sA / Double(samples), sB / Double(samples))
    }
    /// `layeredEnvSpecular` (Fdez-Agüera multiple scattering).
    static func envSpec(_ f0: V3, _ ab: (Double, Double)) -> V3 {
        let fss = f0 * ab.0 + V3(repeating: ab.1)
        let ems = min(max(1 - (ab.0 + ab.1), 0), 1)
        let favg = f0 + (V3(repeating: 1) - f0) / 21
        return fss + (ems * fss * favg) / simd_max(V3(repeating: 1) - favg * ems, V3(repeating: 1e-4))
    }

    struct Lobe { var N: V3; var wo: V3; var f0: V3; var scale: V3; var r: Double; var alpha: Double; var nv: Double; var g1v: Double }
    static func lobe(N: V3, wo: V3, f0: V3, r: Double, layer: V3, ab: (Double, Double)) -> Lobe {
        let a = r * r, nv = max(simd_dot(N, wo), 1e-4)
        let ss = simd_max(f0 * ab.0 + V3(repeating: ab.1), V3(repeating: 1e-4))
        return Lobe(N: N, wo: wo, f0: f0, scale: layer * envSpec(f0, ab) / ss, r: r, alpha: a, nv: nv,
                    g1v: smithG1(a, nv))
    }
    static func fresnel(_ f0: V3, _ vh: Double) -> V3 { f0 + (V3(repeating: 1) - f0) * pow(1 - vh, 5) }
    /// `pathGlossyEval`: (f, pdf).
    static func eval(_ b: Lobe, _ wi: V3) -> (V3, Double) {
        let nl = simd_dot(b.N, wi)
        if nl <= 0 { return (.zero, 0) }
        let h = simd_normalize(wi + b.wo)
        let nh = min(max(simd_dot(b.N, h), 0), 1), vh = min(max(simd_dot(b.wo, h), 0), 1)
        let d = ggxD(b.alpha, nh)
        let pdf = d * b.g1v / (4 * b.nv)
        let f = fresnel(b.f0, vh) * b.scale * (d * gIBL(b.nv, nl, b.r) / (4 * b.nv * nl))
        return (f, pdf)
    }
    static func basis(_ n: V3) -> (V3, V3) {
        let s: Double = n.z >= 0 ? 1 : -1, a = -1 / (s + n.z), d = n.x * n.y * a
        return (V3(1 + s * n.x * n.x * a, s * d, -s * n.x), V3(d, s + n.y * n.y * a, -n.y))
    }
    /// `pathGlossySample`: (weight f·cos/pdf, wi, pdf).
    static func sample(_ b: Lobe, _ u: SIMD2<Double>) -> (V3, V3, Double) {
        let (t, bt) = basis(b.N)
        let vl = V3(simd_dot(b.wo, t), simd_dot(b.wo, bt), b.nv)
        let vh = simd_normalize(V3(b.alpha * vl.x, b.alpha * vl.y, vl.z))
        let phi = 2 * .pi * u.x
        let z = (1 - u.y) * (1 + vh.z) - vh.z
        let sinT = max(0, 1 - z * z).squareRoot()
        let hh = V3(sinT * cos(phi), sinT * sin(phi), z) + vh
        let hl = simd_normalize(V3(b.alpha * hh.x, b.alpha * hh.y, max(hh.z, 0)))
        let h = simd_normalize(t * hl.x + bt * hl.y + b.N * hl.z)
        let wi = simd_reflect(-b.wo, h)
        let nl = simd_dot(b.N, wi)
        if nl <= 0 { return (.zero, wi, 0) }
        let vdh = min(max(simd_dot(b.wo, h), 0), 1)
        let pdf = ggxD(b.alpha, min(max(hl.z, 0), 1)) * b.g1v / (4 * b.nv)
        return (fresnel(b.f0, vdh) * b.scale * (gIBL(b.nv, nl, b.r) / b.g1v), wi, pdf)
    }
    static func luma(_ c: V3) -> Double { simd_dot(c, V3(0.2126, 0.7152, 0.0722)) }

    /// A stratified (jittered-grid-free) 2-D point set: Hammersley.
    static func points(_ n: Int) -> [SIMD2<Double>] {
        (0..<n).map { SIMD2((Double($0) + 0.5) / Double(n), radInv(UInt32($0))) }
    }

    private func say(_ s: String) { print(s); fflush(stdout) }

    static let roughnesses: [Double] = [0.045, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
    static let views: [Double] = [1.0, 0.7, 0.4, 0.15]

    // ── CPU: the sampler and its density agree ───────────────────────────────────────────────
    /// ∫ f·cos dω two ways — by VNDF sampling (Σ weight / N) and by quadrature over the hemisphere
    /// with the lobe's own `eval` — must agree; and Σ pdf over the hemisphere (quadrature) must be
    /// ≤ 1 and equal the share of VNDF samples that land above the horizon.
    func testVNDFSamplerMatchesItsDensityAndTheLobe() {
        let N = V3(0, 0, 1)
        let pts = Self.points(1 << 16)
        for r in [0.2, 0.4, 0.7, 1.0] { for mu in [0.9, 0.4, 0.15] {
            let wo = V3((1 - mu * mu).squareRoot(), 0, mu)
            let b = Self.lobe(N: N, wo: wo, f0: V3(repeating: 1), r: r, layer: V3(repeating: 1),
                              ab: (1, 0))   // scale 1/(A+B)·envSpec(1,(1,0)) = 1: the bare single-scatter lobe
            var mc = 0.0, above = 0
            for u in pts { let (w, _, pdf) = Self.sample(b, u); if pdf > 0 { mc += w.x; above += 1 } }
            mc /= Double(pts.count)
            // Quadrature in (cosθ, φ), fine near the specular peak via a cosθ power warp.
            var q = 0.0, qp = 0.0
            let nt = 1024, np = 4096
            for i in 0..<nt { for j in 0..<np {
                let s = (Double(i) + 0.5) / Double(nt)
                let ct = 1 - s * s, dct = 2 * s / Double(nt)        // cosθ = 1 − s², dense near the pole…
                let st = max(0, 1 - ct * ct).squareRoot(), ph = (Double(j) + 0.5) / Double(np) * 2 * .pi
                let wi = V3(st * cos(ph), st * sin(ph), ct)
                let (f, pdf) = Self.eval(b, wi)
                let dw = dct * 2 * .pi / Double(np)
                q += f.x * ct * dw; qp += pdf * dw
            } }
            let shareAbove = Double(above) / Double(pts.count)
            say(String(format: "[dh1308] r %.2f μ %.2f: ∫f·cos VNDF %.4f vs quadrature %.4f | ∫pdf %.4f vs share above horizon %.4f",
                       r, mu, mc, q, qp, shareAbove))
            // (A sharper lobe off-normal is narrower than this grid's φ step — the window and sun
            // tests below cover the near-mirror lobes, with quadrature over the light instead.)
            XCTAssertEqual(mc, q, accuracy: max(0.006, 0.01 * q), "sampler vs lobe r \(r) μ \(mu)")
            XCTAssertEqual(qp, shareAbove, accuracy: 0.01, "pdf vs sampler r \(r) μ \(mu)")
            XCTAssertLessThanOrEqual(qp, 1.0 + 1e-3)
        } }
    }

    // ── CPU: the one-sample lobe pick conserves energy (white furnace) ───────────────────────
    /// The vertex as the kernel runs it: pick coat / specular / diffuse ∝ (Ec, (1−Ec)·envSpec,
    /// luma(Wd·A)), weight by 1/p, then the picked lobe's own estimate (Lambert in a furnace = A;
    /// a GGX lobe = its VNDF weight). Its EXPECTATION — Σ_i p_i · (1/p_i) · E[lobe i], each lobe's
    /// E taken over the whole point set (a pick that reuses one low-discrepancy set for both the
    /// pick and the lobe correlates them and reads as bias that is the harness's, not the kernel's,
    /// which uses independent Sobol dimensions) — must be 1 for a white surface, metal or
    /// dielectric, coated or not, at every roughness and view.
    func testOneSampleLobePickWhiteFurnace() {
        let N = V3(0, 0, 1)
        let pts = Self.points(1 << 14)
        var rows: [String] = []
        for (label, metal, coat) in [("dielectric", 0.0, 0.0), ("metal", 1.0, 0.0),
                                     ("coated dielectric (cc 0.5, r_c 0.08)", 0.0, 0.5),
                                     ("coated metal (cc 1, r_c 0.08)", 1.0, 1.0)] {
            var line = "  \(label):"
            for r in Self.roughnesses {
                var worst = 0.0, mean = 0.0
                for mu in Self.views {
                    let A = V3(repeating: 1)
                    let wo = V3((1 - mu * mu).squareRoot(), 0, mu)
                    let abB = Self.lutAB(mu, r)
                    let f0 = simd_mix(V3(repeating: 0.04), A, V3(repeating: metal))
                    let es = simd_clamp(Self.envSpec(f0, abB), V3(repeating: 0), V3(repeating: 1))
                    let rc = 0.08, abC = Self.lutAB(mu, rc)
                    let ec = coat > 0 ? min(1, coat * Self.envSpec(V3(repeating: 0.04), abC).x) : 0
                    let under = 1 - ec
                    let wd = under * (1 - metal) * (V3(repeating: 1) - es)
                    let wD = Self.luma(wd * A), wS = Self.luma(under * es), wC = ec, wT = wD + wS + wC
                    let base = Self.lobe(N: N, wo: wo, f0: f0, r: r, layer: V3(repeating: under), ab: abB)
                    let coatL = Self.lobe(N: N, wo: wo, f0: V3(repeating: 0.04), r: rc, layer: V3(repeating: coat), ab: abC)
                    var eB = V3.zero, eC = V3.zero
                    for u in pts { eB += Self.sample(base, u).0; if coat > 0 { eC += Self.sample(coatL, u).0 } }
                    eB /= Double(pts.count); eC /= Double(pts.count)
                    // Σ_i p_i · (estimate | pick i), p_i = w_i / wT — exactly the kernel's weights.
                    var acc = (wS / wT) * (wT / wS) * eB                     // specular pick: lobeW = wT/wS
                    if wC > 0 { acc += (wC / wT) * (wT / wC) * eC }          // coat pick:     lobeW = wT/wC
                    if wD > 0 { acc += (wD / wT) * (wd * (wT / wD)) * A }    // diffuse pick:  lobeW·A, Lambert ∫A/π·cos = A
                    let e = acc.y
                    mean += e / Double(Self.views.count)
                    worst = max(worst, abs(e - 1))
                    XCTAssertEqual(e, 1.0, accuracy: 0.02, "furnace \(label) r \(r) μ \(mu)")
                }
                line += String(format: " r%.2f %.3f(±%.3f)", r, mean, worst)
            }
            rows.append(line)
        }
        say("── DH-1308 path-vertex white furnace (CPU reference; mean over N·V 1/.7/.4/.15, worst |Δ|) ──")
        rows.forEach(say)
    }

    /// A coloured metal reflects its F0: at normal incidence a smooth gold vertex returns ≈ F0
    /// (all through the specular pick — its diffuse weight is 0), and the colour survives the pick.
    func testColouredMetalBouncesItsF0() {
        let N = V3(0, 0, 1), wo = N
        let gold = V3(1.0, 0.78, 0.34)
        let r = 0.1
        let ab = Self.lutAB(1, r)
        let b = Self.lobe(N: N, wo: wo, f0: gold, r: r, layer: V3(repeating: 1), ab: ab)
        let pts = Self.points(1 << 14)
        var acc = V3.zero
        for u in pts { acc += Self.sample(b, u).0 }
        acc /= Double(pts.count)
        let expect = Self.envSpec(gold, ab)
        say(String(format: "[dh1308] gold r 0.1 at normal incidence: returns (%.3f %.3f %.3f), envSpec (%.3f %.3f %.3f), F0 (1 .78 .34)",
                   acc.x, acc.y, acc.z, expect.x, expect.y, expect.z))
        for c in 0..<3 {
            XCTAssertEqual(acc[c], expect[c], accuracy: 0.01)
            XCTAssertEqual(acc[c], gold[c], accuracy: 0.03, "a smooth metal at normal incidence reflects F0")
        }
    }

    /// The single-scatter lobe is reciprocal: f(ωi, ωo) = f(ωo, ωi) (scale 1 — the multiscatter
    /// scale is a function of ωo by construction, as in every albedo-scaled layered model).
    func testLobeIsReciprocal() {
        let N = V3(0, 0, 1)
        var worst = 0.0
        for r in [0.1, 0.4, 0.9] {
            for (a, b) in [(0.9, 0.3), (0.5, 0.7), (0.2, 0.95)] {
                let wo = simd_normalize(V3((1 - a * a).squareRoot(), 0.1, a))
                let wi = simd_normalize(V3(-(1 - b * b).squareRoot(), 0.3, b))
                func f(_ o: V3, _ i: V3) -> Double {
                    var l = Self.lobe(N: N, wo: o, f0: V3(repeating: 0.5), r: r, layer: V3(repeating: 1), ab: (1, 0))
                    l.scale = V3(repeating: 1)
                    return Self.eval(l, i).0.x
                }
                let x = f(wo, wi), y = f(wi, wo)
                worst = max(worst, abs(x - y) / max(x, 1e-9))
                XCTAssertEqual(x, y, accuracy: 1e-9 + 1e-6 * x, "reciprocity r \(r)")
            }
        }
        say(String(format: "[dh1308] reciprocity: worst relative |f(i,o) − f(o,i)| = %.2e", worst))
    }

    // ── CPU: the MIS the kernel uses converges to the reference ──────────────────────────────
    /// A window (a rectangle of radiance 1 above a glossy vertex) seen by the lobe: the kernel's two
    /// strategies — a uniform point on the rectangle weighted `f·cosθ / ((1 + (p_B/p_L)²)·p_L)`, and
    /// a VNDF continuation that crosses the rectangle weighted `w·p_B²/(p_B² + p_L²)` — summed, must
    /// equal a brute-force quadrature of ∫_rect f·cosθ dω. Rough AND near-mirror lobes; one strategy
    /// alone would be the noisy one, the MIS sum is never biased.
    func testWindowMISMatchesQuadrature() {
        let N = V3(0, 0, 1)
        // 1 × 0.8 m, horizontal (a skylight), centred on the lobe's mirror direction (0.8, 0, 0.6).
        let center = V3(1.2, 0.0, 0.9), ex = V3(0.5, 0, 0), ey = V3(0, 0.4, 0)
        let nL = simd_normalize(simd_cross(ex, ey)), area4 = 4 * simd_length(simd_cross(ex, ey))
        let pts = Self.points(1 << 16)
        for r in [0.05, 0.2, 0.6] {
            let mu = 0.6
            let wo = V3(-(1 - mu * mu).squareRoot(), 0, mu)       // mirror direction points at the window
            let b = Self.lobe(N: N, wo: wo, f0: V3(repeating: 0.04), r: r, layer: V3(repeating: 1), ab: Self.lutAB(mu, r))
            func crossing(_ d: V3) -> Double? {               // `pathPortalCrossing` (two-sided here)
                let side = simd_dot(nL, -center), den = simd_dot(d, nL)
                if abs(den) < 1e-9 { return nil }
                let t = -side / den
                if t <= 1e-6 { return nil }
                let q = d * t - center
                let u = simd_dot(q, ex) / simd_dot(ex, ex), v = simd_dot(q, ey) / simd_dot(ey, ey)
                return (abs(u) <= 1 && abs(v) <= 1) ? t : nil
            }
            var light = 0.0, bsdf = 0.0
            for u in pts {
                // light strategy
                let X = center + ex * (u.x * 2 - 1) + ey * (u.y * 2 - 1)
                let dist = simd_length(X), d = X / dist
                let cosR = simd_dot(N, d), cosL = abs(simd_dot(nL, d))
                if cosR > 0 && cosL > 1e-4 {
                    let pdfW = dist * dist / (area4 * cosL)
                    let (f, pdfB) = Self.eval(b, d)
                    let rB = pdfB / pdfW
                    light += f.x * cosR / ((1 + rB * rB) * pdfW)
                }
                // BSDF strategy
                let (w, wi, pdfB) = Self.sample(b, u)
                if pdfB > 0, let t = crossing(wi) {
                    let cosL2 = abs(simd_dot(nL, wi))
                    let pL = t * t / (area4 * cosL2)
                    bsdf += w.x * (pdfB * pdfB) / (pdfB * pdfB + pL * pL)
                }
            }
            light /= Double(pts.count); bsdf /= Double(pts.count)
            // Reference: dense quadrature over the rectangle's AREA.
            var ref = 0.0
            let n = 1200
            for i in 0..<n { for j in 0..<n {
                let X = center + ex * ((Double(i) + 0.5) / Double(n) * 2 - 1) + ey * ((Double(j) + 0.5) / Double(n) * 2 - 1)
                let dist = simd_length(X), d = X / dist
                let cosR = simd_dot(N, d), cosL = abs(simd_dot(nL, d))
                if cosR <= 0 { continue }
                ref += Self.eval(b, d).0.x * cosR * cosL / (dist * dist) * (area4 / Double(n * n))
            } }
            say(String(format: "[dh1308] window MIS r %.2f: light %.5f + lobe %.5f = %.5f vs quadrature %.5f",
                       r, light, bsdf, light + bsdf, ref))
            XCTAssertEqual(light + bsdf, ref, accuracy: max(1e-4, 0.02 * ref), "window MIS r \(r)")
        }
    }

    /// The sun as the kernel treats it at a glossy vertex: a cone of half-angle θ, NEE weighted
    /// `E·f·cosθ / (1 + (p_B·Ω)²)` and a continuation inside the cone weighted
    /// `(E/Ω)·w·(p_B·Ω)²/(1 + (p_B·Ω)²)`. Their sum must be E·∫_cone f·cosθ dω/Ω.
    func testSunConeMISMatchesQuadrature() {
        let N = V3(0, 0, 1)
        let theta = 0.02                                         // a soft sun, so the quadrature resolves it
        let half = sin(0.5 * theta), omega = 4 * .pi * half * half, chord = 2 * half
        let pts = Self.points(1 << 17)
        for r in [0.05, 0.15, 0.5] {
            let mu = 0.7
            let wo = V3(-(1 - mu * mu).squareRoot(), 0, mu)
            let Ld = V3((1 - mu * mu).squareRoot(), 0, mu)        // the sun sits on the mirror direction
            let b = Self.lobe(N: N, wo: wo, f0: V3(repeating: 0.04), r: r, layer: V3(repeating: 1), ab: Self.lutAB(mu, r))
            let (t, bt) = Self.basis(Ld)
            var nee = 0.0, hit = 0.0
            for u in pts {
                let cosT = 1 - u.x * (1 - cos(theta)), sinT = max(0, 1 - cosT * cosT).squareRoot(), ph = 2 * .pi * u.y
                let d = simd_normalize(t * (sinT * cos(ph)) + bt * (sinT * sin(ph)) + Ld * cosT)
                let (f, pdfB) = Self.eval(b, d)
                let rB = pdfB * omega
                nee += f.x * max(simd_dot(N, d), 0) / (1 + rB * rB)
                let (w, wi, pdfS) = Self.sample(b, u)
                if pdfS > 0, simd_length(wi - Ld) <= chord {
                    let rr = pdfS * omega
                    hit += w.x / omega * (rr * rr / (1 + rr * rr))
                }
            }
            nee /= Double(pts.count); hit /= Double(pts.count)
            var ref = 0.0
            let n = 600
            for i in 0..<n { for j in 0..<n {
                let cosT = 1 - (Double(i) + 0.5) / Double(n) * (1 - cos(theta))
                let sinT = max(0, 1 - cosT * cosT).squareRoot(), ph = (Double(j) + 0.5) / Double(n) * 2 * .pi
                let d = simd_normalize(t * (sinT * cos(ph)) + bt * (sinT * sin(ph)) + Ld * cosT)
                ref += Self.eval(b, d).0.x * max(simd_dot(N, d), 0) / Double(n * n)
            } }
            say(String(format: "[dh1308] sun-cone MIS r %.2f: NEE %.5f + lobe %.5f = %.5f vs quadrature %.5f",
                       r, nee, hit, nee + hit, ref))
            XCTAssertEqual(nee + hit, ref, accuracy: max(1e-4, 0.02 * ref), "sun MIS r \(r)")
        }
    }

    /// The material lane: `rtPackMaterialLane` and the shader's decode (transcribed) round-trip,
    /// and the lane is always negative (a glass row's ≥ 0 and an unpacked 0 decode to "absent").
    func testMaterialLaneRoundTrips() {
        func decode(_ w: Float) -> (Float, Float)? {
            if w > -0.5 { return nil }
            let x = -w - 1, aq = (x * 0.5).rounded(.down)
            return (aq * 0.001, min(max(x - 2 * aq, 0), 1))
        }
        for m in stride(from: Float(0), through: 1, by: 0.05) { for r in stride(from: Float(0), through: 1, by: 0.037) {
            let w = IlluminatoramaRenderer.rtPackMaterialLane(m, r)
            XCTAssertLessThan(w, -0.9)
            guard let (dm, dr) = decode(w) else { return XCTFail("decoded as absent") }
            XCTAssertEqual(dm, m, accuracy: 6e-4); XCTAssertEqual(dr, min(r, 0.999), accuracy: 3e-4)
        } }
        XCTAssertNil(decode(0)); XCTAssertNil(decode(17))
    }

    // ── GPU: the shipped shader code ─────────────────────────────────────────────────────────
    private static let probeKernel = """

    // ── DH-1308 probe (appended by IlluminatoramaPathGlossyVertexTests) ──
    static inline float dh1308RadInv(uint bits) {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return float(bits) * 2.3283064365386963e-10;
    }
    // in: [r, NdotV, metallic, coat]   out: [furnace (white), baseSpecShare, 0, 0]
    // The kernel's vertex pick, verbatim in structure: coat (r 0.08) / base GGX / Lambert ∝ albedo.
    kernel void dh1308Furnace(device const float4* inp [[buffer(0)]],
                              device float4* outp [[buffer(1)]],
                              texture2d<half, access::sample> dfg [[texture(0)]],
                              uint gid [[thread_position_in_grid]]) {
        float4 a = inp[gid];
        float r = a.x, mu = a.y, m = a.z, ccS = a.w;
        float3 A = float3(1.0);
        float3 N = float3(0, 0, 1);
        float3 wo = float3(sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu);
        constexpr sampler ls(filter::linear, address::clamp_to_edge);
        float4 lutB = float4(dfg.sample(ls, float2(mu, r)));
        float3 F0 = mix(float3(0.04), A, m);
        float3 Es = saturate(layeredEnvSpecular(F0, lutB.xy));
        float rC = 0.08;
        float4 lutC = float4(dfg.sample(ls, float2(mu, rC)));
        float Ec = ccS > 0.0 ? saturate(ccS * layeredEnvSpecular(float3(0.04), lutC.xy).x) : 0.0;
        float under = 1.0 - Ec;
        float3 Wd = under * (1.0 - m) * (1.0 - Es);
        PathGlossyLobe base = pathGlossyLobe(N, wo, F0, r, float3(under), lutB);
        PathGlossyLobe coat = pathGlossyLobe(N, wo, float3(0.04), rC, float3(ccS), lutC);
        const uint NS = 8192u;
        float3 spec = float3(0.0), coatE = float3(0.0);
        for (uint i = 0u; i < NS; ++i) {
            float2 u = float2((float(i) + 0.5) / float(NS), dh1308RadInv(i));
            float3 wi; float pdf;
            spec += pathGlossySample(base, u, wi, pdf);
            if (ccS > 0.0) coatE += pathGlossySample(coat, u, wi, pdf);
        }
        spec /= float(NS); coatE /= float(NS);
        // The same base lobe integrated through `pathGlossyEval` with cosine samples (π·f per sample):
        // eval and sample must describe one lobe.
        float3 specEval = float3(0.0);
        const uint NC = 65536u;
        for (uint i = 0u; i < NC; ++i) {
            float2 u = float2((float(i) + 0.5) / float(NC), dh1308RadInv(i));
            float rr = sqrt(u.x), ph = 2.0 * M_PI_F * u.y;
            float3 wi = float3(rr * cos(ph), rr * sin(ph), sqrt(max(0.0, 1.0 - u.x)));
            float pdf;
            specEval += M_PI_F * pathGlossyEval(base, wi, pdf);
        }
        specEval /= float(NC);
        float3 total = Wd * A + spec + coatE;
        outp[gid] = float4(total.g, spec.g, coatE.g, specEval.g);
    }
    """

    private static func shaderURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/\(name)")
    }

    /// The same furnace on the GPU, through the shipped `IlluminatoramaBSDF.h` and the engine's
    /// own LUT bake (half-precision, 128², bilinear) — the transcription above cannot drift from it.
    @MainActor
    func testGPUWhiteFurnaceOnShippedShaderCode() throws {
        guard ProcessInfo.processInfo.environment["DH1308_GPU"] == "1" else {
            throw XCTSkip("GPU probe: set DH1308_GPU=1 (kept off while another session owns the GPU)")
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("no Metal") }  // gpu-ok: test harness
        let bakeLib = try MetalSourceLoader.makeLibrary(device: device, contentsOf: Self.shaderURL("IlluminatoramaIBLBake.metal"))
        guard let bakeFn = bakeLib.makeFunction(name: "illumi_dfg_bake") else { throw XCTSkip("no bake") }
        let bake = try device.makeComputePipelineState(function: bakeFn)  // gpu-ok: test harness
        let n = 128
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: n, height: n, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]; desc.storageMode = .private
        guard let lut = device.makeTexture(descriptor: desc), let cb0 = queue.makeCommandBuffer(),
              let e0 = cb0.makeComputeCommandEncoder() else { throw XCTSkip("no LUT") }
        e0.setComputePipelineState(bake); e0.setTexture(lut, index: 0)
        var remap: UInt32 = IlluminatoramaRenderer.dfgIBLGeometryRemapDefault ? 1 : 0
        e0.setBytes(&remap, length: 4, index: 0)
        e0.dispatchThreads(MTLSize(width: n, height: n, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        e0.endEncoding(); cb0.commit(); cb0.waitUntilCompleted()  // gpu-ok: test harness
        let src = "#include <metal_stdlib>\nusing namespace metal;\n"
            + (try MetalSourceLoader.source(contentsOf: Self.shaderURL("IlluminatoramaBSDF.h"))) + Self.probeKernel
        let lib = try device.makeLibrary(source: src, options: nil)
        guard let fn = lib.makeFunction(name: "dh1308Furnace") else { throw XCTSkip("probe missing") }
        let probe = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        var rows: [SIMD4<Float>] = []
        let cases: [(String, Float, Float)] = [("dielectric", 0, 0), ("metal", 1, 0), ("coated dielectric", 0, 0.5), ("coated metal", 1, 1)]
        for c in cases { for r in Self.roughnesses { for mu in Self.views { rows.append(SIMD4(Float(r), Float(mu), c.1, c.2)) } } }
        guard let ib = device.makeBuffer(bytes: rows, length: rows.count * 16, options: .storageModeShared),
              let ob = device.makeBuffer(length: rows.count * 16, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no buffers") }
        enc.setComputePipelineState(probe); enc.setBuffer(ib, offset: 0, index: 0); enc.setBuffer(ob, offset: 0, index: 1)
        enc.setTexture(lut, index: 0)
        enc.dispatchThreads(MTLSize(width: rows.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: rows.count)
        say("── DH-1308 path-vertex white furnace (GPU, shipped IlluminatoramaBSDF.h + engine LUT; per N·V 1/.7/.4/.15) ──")
        var k = 0
        for c in cases {
            var line = "  \(c.0):"
            for r in Self.roughnesses {
                var vals: [Float] = []
                for _ in Self.views {
                    vals.append(o[k].x)
                    if Double(r) >= 0.2 {
                        XCTAssertEqual(Double(o[k].w), Double(o[k].y), accuracy: max(0.003, 0.03 * Double(o[k].y)),
                                       "eval vs sample \(c.0) r \(r): eval \(o[k].w) sample \(o[k].y)")
                    }
                    k += 1
                }
                line += String(format: " r%.2f ", r) + vals.map { String(format: "%.3f", $0) }.joined(separator: "/")
                for v in vals { XCTAssertEqual(Double(v), 1.0, accuracy: 0.02, "GPU furnace \(c.0) r \(r)") }
            }
            say(line)
        }
    }
}
