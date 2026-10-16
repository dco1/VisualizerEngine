import Foundation
@testable import VisualizerRendering

/// CPU model of the still's whole reconstruction chain (Daydream DH-1365), for
/// `ReconstructionFilterTests`. GPU-free: it reproduces what the photo lane does to a scene between
/// the camera and the HDR texture the tonemap reads —
///
///  1. **jitter accumulation** — the still's sub-pixel jitter spans one INTERNAL pixel
///     (`PhotoQuality.jitterPixels` = 1.0) and the TAA still mode is a straight running mean, so a
///     converged internal texel is the scene AVERAGED OVER ITS OWN SQUARE (a box of 1/scale output
///     pixels). Modelled exactly in 1-D (closed-form box integral) and by n × n stratified
///     quadrature in 2-D (n² jittered frames).
///  2. **the TAA still sharpen** (optional) — `illumi_taa_resolve`'s still branch: luma +
///     0.25 × (luma − 3×3 mean), clamped to the 3×3 range. Applied per internal texel.
///  3. **the reduction** — `IlluminatoramaReconstructionFilter.gather1D`, the shader's arithmetic.
///
/// Positions are in OUTPUT pixels; output pixel o covers [o, o+1), internal texel j covers
/// [j/s, (j+1)/s).
enum ReconstructionModel {

    struct Chain {
        var filter: IlluminatoramaReconstructionFilter
        var scale: Double
        var sharpen: Bool = false
    }

    // MARK: 1-D

    /// Reduce a 1-D row of internal texel values to `outputCount` output pixels.
    static func reduce1D(_ texels: [Double], chain: Chain, outputCount: Int) -> [Double] {
        let v = chain.sharpen ? sharpen1D(texels) : texels
        return (0..<outputCount).map { o in
            chain.filter.gather1D(outputPixel: o, scale: chain.scale, internalCount: v.count)
                .reduce(0) { $0 + $1.weight * v[$1.texel] }
        }
    }

    /// The TAA still sharpen, 1-D (a 3-tap mean and range — the 1-D section of the 3 × 3).
    static func sharpen1D(_ v: [Double]) -> [Double] {
        let n = v.count
        return (0..<n).map { j in
            let a = v[max(j - 1, 0)], b = v[j], c = v[min(j + 1, n - 1)]
            let mean = (a + b + c) / 3
            return min(max(b + 0.25 * (b - mean), min(a, b, c)), max(a, b, c))
        }
    }

    /// Internal texels of `A·cos(2π f x + φ) + dc`, each the exact average over its own square.
    static func sinusoidTexels(f: Double, phase: Double, amplitude: Double = 1, dc: Double = 0,
                               scale: Double, count: Int) -> [Double] {
        (0..<count).map { j in
            let a = Double(j) / scale, b = Double(j + 1) / scale
            let w = 2 * Double.pi * f
            guard w > 1e-12 else { return dc + amplitude * cos(phase) }
            return dc + amplitude * (sin(w * b + phase) - sin(w * a + phase)) / (w * (b - a))
        }
    }

    /// Internal texels of a step from `low` to `high` at `edge` (output px), box-integrated.
    static func stepTexels(edge: Double, low: Double, high: Double, scale: Double, count: Int) -> [Double] {
        (0..<count).map { j in
            let a = Double(j) / scale, b = Double(j + 1) / scale
            let covered = min(max(b - edge, 0), b - a) / (b - a)
            return low + (high - low) * covered
        }
    }

    struct Response {
        /// Modulation transfer at the frequency (fit amplitude / input amplitude), below Nyquist.
        var mtf: Double
        /// RMS of everything that is NOT the fitted sinusoid, relative to the input's RMS — the
        /// fixed-pattern error a phase-varying kernel adds (0 for any kernel that is the same at
        /// every output pixel). Above Nyquist the whole output is alias and this is the alias RMS.
        var residual: Double
    }

    /// Push a unit sinusoid through the chain and measure what comes out, over the central output
    /// pixels (away from the clamped edges). Averaged over several input phases.
    static func response(f: Double, chain: Chain, outputCount: Int = 256, phases: Int = 4) -> Response {
        let internalCount = Int((Double(outputCount) * chain.scale).rounded(.up))
        let margin = 12
        var mtfAcc = 0.0, resAcc = 0.0
        for p in 0..<phases {
            let phase = 2 * Double.pi * Double(p) / Double(phases) + 0.37
            let texels = sinusoidTexels(f: f, phase: phase, scale: chain.scale, count: internalCount)
            let out = reduce1D(texels, chain: chain, outputCount: outputCount)
            let idx = Array(margin..<(outputCount - margin))
            let rmsIn = 1 / 2.0.squareRoot()
            if f < 0.5 {
                // Least-squares fit of a·cos + b·sin + c at the output pixel centres.
                let fit = fitSinusoid(out, at: idx, f: f)
                mtfAcc += (fit.a * fit.a + fit.b * fit.b).squareRoot()
                resAcc += fit.residualRMS / rmsIn
            } else {
                let m = idx.reduce(0) { $0 + out[$1] } / Double(idx.count)
                let rms = (idx.reduce(0) { $0 + (out[$1] - m) * (out[$1] - m) } / Double(idx.count)).squareRoot()
                resAcc += rms / rmsIn
            }
        }
        return Response(mtf: mtfAcc / Double(phases), residual: resAcc / Double(phases))
    }

    static func fitSinusoid(_ y: [Double], at idx: [Int], f: Double) -> (a: Double, b: Double, residualRMS: Double) {
        // Normal equations for [cos, sin, 1].
        var m = [[Double]](repeating: [0, 0, 0], count: 3)
        var r = [0.0, 0.0, 0.0]
        for o in idx {
            let x = Double(o) + 0.5
            let row = [cos(2 * .pi * f * x), sin(2 * .pi * f * x), 1]
            for i in 0..<3 { for k in 0..<3 { m[i][k] += row[i] * row[k] }; r[i] += row[i] * y[o] }
        }
        let s = solve3(m, r)
        var res = 0.0
        for o in idx {
            let x = Double(o) + 0.5
            let e = y[o] - (s[0] * cos(2 * .pi * f * x) + s[1] * sin(2 * .pi * f * x) + s[2])
            res += e * e
        }
        return (s[0], s[1], (res / Double(idx.count)).squareRoot())
    }

    static func solve3(_ a: [[Double]], _ b: [Double]) -> [Double] {
        var m = a, v = b
        for c in 0..<3 {
            var p = c
            for r in c..<3 where abs(m[r][c]) > abs(m[p][c]) { p = r }
            m.swapAt(c, p); v.swapAt(c, p)
            for r in 0..<3 where r != c {
                let k = m[r][c] / m[c][c]
                for q in 0..<3 { m[r][q] -= k * m[c][q] }
                v[r] -= k * v[c]
            }
        }
        return (0..<3).map { v[$0] / m[$0][$0] }
    }

    struct Summary {
        var mtfQuarter: Double      // MTF at 0.25 cycles / output px
        var mtfNyquistish: Double   // MTF at 0.45 cycles / output px
        var mtf50: Double           // frequency (cycles / px) where MTF first falls to 0.5
        var aliasRMS: Double        // RMS output over 0.5…2.0 cycles / px, relative to the input
        var patternRMS: Double      // worst below-Nyquist residual (phase-varying kernel)
    }

    static func summary(_ chain: Chain) -> Summary {
        var mtf50 = 0.5
        var prev = (f: 0.0, m: 1.0)
        var pattern = 0.0
        var f = 0.02
        while f < 0.5 {
            let r = response(f: f, chain: chain)
            pattern = max(pattern, r.residual)
            if r.mtf < 0.5, mtf50 == 0.5 {
                mtf50 = prev.f + (f - prev.f) * (prev.m - 0.5) / max(prev.m - r.mtf, 1e-9)
            }
            prev = (f, r.mtf)
            f += 0.02
        }
        var aliasSq = 0.0, n = 0.0
        f = 0.51
        while f <= 2.0 {
            let r = response(f: f, chain: chain)
            aliasSq += r.residual * r.residual; n += 1
            f += 0.03
        }
        return Summary(mtfQuarter: response(f: 0.25, chain: chain).mtf,
                       mtfNyquistish: response(f: 0.45, chain: chain).mtf,
                       mtf50: mtf50,
                       aliasRMS: (aliasSq / n).squareRoot(),
                       patternRMS: pattern)
    }

    /// The edge's worst undershoot below `low` and overshoot above `high`, each as a FRACTION OF
    /// THE STEP, over many sub-pixel edge positions. Undershoot × (contrast − 1) is the dark halo a
    /// bright window would print on the wall beside it, as a fraction of the wall.
    static func stepExcursions(_ chain: Chain, low: Double = 0, high: Double = 1) -> (under: Double, over: Double) {
        let outputCount = 48
        let internalCount = Int((Double(outputCount) * chain.scale).rounded(.up))
        var under = 0.0, over = 0.0
        for k in 0..<16 {
            let edge = 24 + Double(k) / 16
            let out = reduce1D(stepTexels(edge: edge, low: low, high: high, scale: chain.scale, count: internalCount),
                               chain: chain, outputCount: outputCount)
            for o in 8..<(outputCount - 8) {
                under = max(under, (low - out[o]) / (high - low))
                over = max(over, (out[o] - high) / (high - low))
            }
        }
        return (under, over)
    }

    // MARK: 2-D

    /// A grey image's internal texels: each the mean of `quadrature²` stratified samples of
    /// `scene(x, y)` over its square — `quadrature²` converged jittered frames.
    static func texels2D(width: Int, height: Int, scale: Double, quadrature: Int,
                         scene: (Double, Double) -> Double) -> [[Double]] {
        let q = Double(quadrature)
        return (0..<height).map { j in
            (0..<width).map { i in
                var acc = 0.0
                for b in 0..<quadrature { for a in 0..<quadrature {
                    let x = (Double(i) + (Double(a) + 0.5) / q) / scale
                    let y = (Double(j) + (Double(b) + 0.5) / q) / scale
                    acc += scene(x, y)
                } }
                return acc / (q * q)
            }
        }
    }

    static func sharpen2D(_ v: [[Double]]) -> [[Double]] {
        let h = v.count, w = v.first?.count ?? 0
        return (0..<h).map { j in (0..<w).map { i in
            var mean = 0.0, lo = Double.infinity, hi = -Double.infinity
            for dj in -1...1 { for di in -1...1 {
                let x = v[min(max(j + dj, 0), h - 1)][min(max(i + di, 0), w - 1)]
                mean += x; lo = min(lo, x); hi = max(hi, x)
            } }
            mean /= 9
            return min(max(v[j][i] + 0.25 * (v[j][i] - mean), lo), hi)
        } }
    }

    /// Separable 2-D reduction — the shader's gather (both branches are an outer product).
    static func reduce2D(_ texels: [[Double]], chain: Chain, outWidth: Int, outHeight: Int) -> [[Double]] {
        let v = chain.sharpen ? sharpen2D(texels) : texels
        let ih = v.count, iw = v.first?.count ?? 0
        let gx = (0..<outWidth).map { chain.filter.gather1D(outputPixel: $0, scale: chain.scale, internalCount: iw) }
        let gy = (0..<outHeight).map { chain.filter.gather1D(outputPixel: $0, scale: chain.scale, internalCount: ih) }
        return (0..<outHeight).map { o in (0..<outWidth).map { p in
            var acc = 0.0
            for ty in gy[o] { var row = 0.0; for tx in gx[p] { row += tx.weight * v[ty.texel][tx.texel] }; acc += ty.weight * row }
            return acc
        } }
    }

    /// **Zone plate.** cos(π k r²) about the image centre: local frequency k·r cycles / px, from 0
    /// at the centre to ~1.6 cycles / px in the corners. Returns the output's RMS modulation in two
    /// annuli, relative to the input's: the PASS band (0.10–0.20 c/px, should survive) and the ALIAS
    /// band (0.65–1.35 c/px — past Nyquist, so every bit of modulation there is a false pattern).
    static func zonePlate(_ chain: Chain, size: Int = 160, quadrature: Int = 4) -> (pass: Double, alias: Double) {
        let c = Double(size) / 2
        let rMax = c * 2.0.squareRoot()
        let k = 1.6 / rMax
        let scene: (Double, Double) -> Double = { x, y in
            let dx = x - c, dy = y - c
            return 0.5 + 0.5 * cos(Double.pi * k * (dx * dx + dy * dy))
        }
        let iw = Int((Double(size) * chain.scale).rounded(.up))
        let tex = texels2D(width: iw, height: iw, scale: chain.scale, quadrature: quadrature, scene: scene)
        let out = reduce2D(tex, chain: chain, outWidth: size, outHeight: size)
        func bandRMS(_ lo: Double, _ hi: Double) -> Double {
            var vals: [Double] = []
            for o in 2..<(size - 2) { for p in 2..<(size - 2) {
                let dx = Double(p) + 0.5 - c, dy = Double(o) + 0.5 - c
                let f = k * (dx * dx + dy * dy).squareRoot()
                if f >= lo && f <= hi { vals.append(out[o][p]) }
            } }
            let m = vals.reduce(0, +) / Double(vals.count)
            return (vals.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(vals.count)).squareRoot()
        }
        let rmsIn = 0.5 / 2.0.squareRoot()
        return (bandRMS(0.10, 0.20) / rmsIn, bandRMS(0.65, 1.35) / rmsIn)
    }

    /// **Slanted edge** (the ISO 12233 method, simplified): a straight dark→light edge at `degrees`
    /// from vertical; project every output pixel onto the edge normal, bin at ¼ px into an
    /// oversampled edge-spread function, differentiate to the line-spread function, and read the
    /// MTF off its DFT. Returns MTF50 (cycles / px) and the ESF's under/overshoot (fraction of the step).
    static func slantedEdge(_ chain: Chain, degrees: Double = 5, size: Int = 96, quadrature: Int = 4,
                            low: Double = 0.2, high: Double = 0.8) -> (mtf50: Double, under: Double, over: Double) {
        let theta = degrees * Double.pi / 180
        let n = (cos(theta), -sin(theta))           // edge normal (x right, y down)
        let c = Double(size) / 2
        let scene: (Double, Double) -> Double = { x, y in
            ((x - c) * n.0 + (y - c) * n.1) >= 0 ? high : low
        }
        let iw = Int((Double(size) * chain.scale).rounded(.up))
        let tex = texels2D(width: iw, height: iw, scale: chain.scale, quadrature: quadrature, scene: scene)
        let out = reduce2D(tex, chain: chain, outWidth: size, outHeight: size)
        let binsPerPx = 4, halfSpan = 8
        let nb = 2 * halfSpan * binsPerPx
        var sum = [Double](repeating: 0, count: nb), cnt = [Double](repeating: 0, count: nb)
        for o in 8..<(size - 8) { for p in 8..<(size - 8) {
            let d = (Double(p) + 0.5 - c) * n.0 + (Double(o) + 0.5 - c) * n.1
            let b = Int(floor((d + Double(halfSpan)) * Double(binsPerPx)))
            if b >= 0 && b < nb { sum[b] += out[o][p]; cnt[b] += 1 }
        } }
        let esf = (0..<nb).map { cnt[$0] > 0 ? sum[$0] / cnt[$0] : .nan }
        let valid = esf.filter { !$0.isNaN }
        let under = max(0, (low - (valid.min() ?? low)) / (high - low))
        let over = max(0, ((valid.max() ?? high) - high) / (high - low))
        // Fill empty bins by linear interpolation, then LSF = derivative, Hamming-windowed.
        var e = esf
        for i in 0..<nb where e[i].isNaN {
            let l = (0..<i).last { !esf[$0].isNaN }, r = ((i + 1)..<nb).first { !esf[$0].isNaN }
            switch (l, r) {
            case let (l?, r?): e[i] = esf[l] + (esf[r] - esf[l]) * Double(i - l) / Double(r - l)
            case let (l?, nil): e[i] = esf[l]
            case let (nil, r?): e[i] = esf[r]
            default: e[i] = 0
            }
        }
        var lsf = (0..<nb).map { i in i == 0 ? 0 : e[i] - e[i - 1] }
        for i in 0..<nb { lsf[i] *= 0.54 - 0.46 * cos(2 * Double.pi * Double(i) / Double(nb - 1)) }
        let dt = 1.0 / Double(binsPerPx)
        func mtf(_ f: Double) -> Double {
            var re = 0.0, im = 0.0, dc = 0.0
            for i in 0..<nb { re += lsf[i] * cos(2 * .pi * f * Double(i) * dt); im += lsf[i] * sin(2 * .pi * f * Double(i) * dt); dc += lsf[i] }
            return (re * re + im * im).squareRoot() / max(dc, 1e-12)
        }
        // ¼-px binning integrates each bin like a 0.25 px box: divide that sinc back out.
        func corrected(_ f: Double) -> Double {
            let x = Double.pi * f * dt
            return mtf(f) / (x < 1e-9 ? 1 : sin(x) / x)
        }
        var prev = (f: 0.0, m: 1.0), m50 = 0.5
        var f = 0.01
        while f <= 0.5 {
            let m = corrected(f)
            if m < 0.5 { m50 = prev.f + (f - prev.f) * (prev.m - 0.5) / max(prev.m - m, 1e-9); break }
            prev = (f, m); f += 0.01
        }
        return (m50, under, over)
    }
}
