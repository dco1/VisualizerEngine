import Foundation

/// **A shared fibre / thread layer for cloth** (DH-1339).
///
/// The fabric audit (2026-10-07) shot each cloth alone at 1 m / 10 cm / 2 cm and found the same thing in all
/// of them: structure ends at roughly the size of the baked noise (a few mm). Real thread is 0.3–0.8 mm and
/// real pile fibre 0.1–0.2 mm. The detail band is sampled at 8× the macro UV, so on a 0.30 m upholstery tile one
/// 512² texel is ~0.07 mm — fine enough to carry a thread — but the detail band was only ever filled with
/// low-octave fBm, which has no *strands* in it.
///
/// This writes strands: each is a short rounded ridge laid at its own angle (a dominant direction ± a spread) and
/// splatted into the detail height field, wrapping at the tile edge so it tiles. The same field then derives BOTH
/// the detail normal and the diffuse micro-occlusion through `setDetailRelief` — one relief, one writer.
///
/// Sizes are in TEXELS of the detail slice (texel = tile / (8 × size)); the call sites state the real-world size
/// they are aiming at next to the number.
extension MaterialGenerator {

    /// One population of strands.
    struct FibreSpec {
        /// How many strands to lay down on the slice.
        var count: Int
        /// Strand length, texels (±40 %).
        var length: Double
        /// Strand width, texels (±30 %; `widthJitter` widens that for slub).
        var width: Double
        /// Dominant direction, radians (0 = along +U).
        var angle: Double
        /// Half-range of the scatter around `angle`, radians (π = every direction).
        var spread: Double
        /// 1 → strands overlap with `max` (woven thread laying over thread); 0 → they ADD (fuzz piling up).
        var layered: Bool = true
        /// Extra random width multiplier span: width × (1 ± widthJitter). 0.3 for even yarn, ~0.9 for slub.
        var widthJitter: Double = 0.3
        /// Relative height of this population against the others (0…1).
        var lift: Double = 1
    }

    /// A regular plain-weave lattice for the detail band: `pitch` texels per thread (must divide the slice so it tiles),
    /// warp over weft alternating every thread, each thread with its own width and a slow slub along its length.
    struct WeaveSpec {
        var pitch: Int
        /// Relief depth of the lattice, 0…1 against the fibres.
        var depth: Double = 0.7
        /// Per-thread width variation (0 = machine-even, 0.5 = hand-spun).
        var threadJitter: Double = 0.35
        /// How much a thread swells and thins along its length (the slub), 0…1.
        var slub: Double = 0.5
    }

    /// Replaces `addMicroDetail` for a cloth: the same low-octave fBm (so the existing tooth survives, weighted down)
    /// PLUS fibre strands. No-op if the material already set its own detail normal.
    static func addFibreDetail(_ ch: inout MaterialChannels, seed: UInt64, fibres: [FibreSpec],
                               baseCells: Int = 96, baseWeight: Double = 0.35, weave: WeaveSpec? = nil,
                               strength: Double, occlusionStrength: Double? = nil) {
        guard ch.detailNormal == nil else { return }
        let n = ch.size
        var h = [Double](repeating: 0, count: n * n)
        for y in 0..<n {
            for x in 0..<n {
                let u = Double(x) / Double(n), v = Double(y) / Double(n)
                h[ch.idx(x, y)] = baseWeight * Noise.fbmTiled(u, v, baseCells: baseCells, octaves: 2, seed: seed)
            }
        }
        var rng = FibreRNG(seed: seed ^ 0xF1B2E5)
        let nd = Double(n)
        if let w = weave, w.pitch > 1, n % w.pitch == 0 {
            let threads = n / w.pitch
            // per-thread width factor and a slub phase, for warp (columns) and weft (rows)
            let warpW = (0..<threads).map { _ in 1 + (rng.unit() * 2 - 1) * w.threadJitter }
            let weftW = (0..<threads).map { _ in 1 + (rng.unit() * 2 - 1) * w.threadJitter }
            let warpPh = (0..<threads).map { _ in rng.unit() * 6.2832 }
            let weftPh = (0..<threads).map { _ in rng.unit() * 6.2832 }
            let p = Double(w.pitch)
            for y in 0..<n {
                for x in 0..<n {
                    let ti = x / w.pitch, tj = y / w.pitch
                    let fx = (Double(x % w.pitch) + 0.5) / p, fy = (Double(y % w.pitch) + 0.5) / p
                    // thread cross-section: a rounded ridge across the thread, width scaled per thread
                    func ridge(_ f: Double, _ width: Double) -> Double { let d = abs(f - 0.5) / (0.5 * min(width * 1.25, 1.45)); return d >= 1 ? 0 : (1 - d * d) }   // wide + soft: neighbours touch, so threads read as threads, not a grid of holes
                    // slub: thickness swells slowly ALONG the thread (warp runs in y, weft in x)
                    let warpSlub = 1 + w.slub * 0.5 * sin(Double(y) * 0.035 + warpPh[ti]) * sin(Double(y) * 0.011 + warpPh[ti] * 1.7)
                    let weftSlub = 1 + w.slub * 0.5 * sin(Double(x) * 0.035 + weftPh[tj]) * sin(Double(x) * 0.011 + weftPh[tj] * 1.7)
                    let warp = ridge(fx, warpW[ti] * warpSlub) * (0.75 + 0.25 * warpSlub)
                    let weft = ridge(fy, weftW[tj] * weftSlub) * (0.75 + 0.25 * weftSlub)
                    let warpOver = (ti + tj) & 1 == 0          // plain weave: alternates every thread
                    let top = warpOver ? max(warp, 0.7 * weft) : max(weft, 0.7 * warp)
                    h[ch.idx(x, y)] += w.depth * top
                }
            }
        }
        for (pi, spec) in fibres.enumerated() {
            var field = [Double](repeating: 0, count: n * n)
            for _ in 0..<spec.count {
                let cx = rng.unit() * nd, cy = rng.unit() * nd
                let a = spec.angle + (rng.unit() * 2 - 1) * spec.spread
                let len = spec.length * (0.6 + 0.8 * rng.unit())
                let wid = max(1.2, spec.width * (1 + (rng.unit() * 2 - 1) * spec.widthJitter))
                let lift = spec.lift * (0.6 + 0.4 * rng.unit())
                let dx = cos(a), dy = sin(a)
                let r = wid * 0.5
                let steps = max(2, Int(len / 0.7))
                for s in 0..<steps {
                    let t = (Double(s) / Double(steps - 1) - 0.5) * len
                    // Taper the last 20 % at each end so a strand is a rounded fibre, not a capsule stub.
                    let e = min(1, (1 - abs(2 * Double(s) / Double(steps - 1) - 1)) / 0.2)
                    let px = cx + dx * t, py = cy + dy * t
                    let ri = r * (0.55 + 0.45 * e)
                    let x0 = Int(floor(px - ri)), x1 = Int(ceil(px + ri))
                    let y0 = Int(floor(py - ri)), y1 = Int(ceil(py + ri))
                    for yy in y0...y1 {
                        for xx in x0...x1 {
                            let ddx = Double(xx) + 0.5 - px, ddy = Double(yy) + 0.5 - py
                            let d2 = (ddx * ddx + ddy * ddy) / (ri * ri)
                            if d2 >= 1 { continue }
                            let prof = (1 - d2) * (1 - d2) * lift      // rounded cross-section
                            let wx = ((xx % n) + n) % n, wy = ((yy % n) + n) % n
                            let k = wy * n + wx
                            if spec.layered { field[k] = max(field[k], prof) } else { field[k] += prof * 0.35 }
                        }
                    }
                }
            }
            let weight = fibres.isEmpty ? 0 : 1.0 / Double(fibres.count) * (weave == nil ? 1.0 : 0.45)
            _ = pi
            for k in 0..<(n * n) { h[k] += min(field[k], 1.0) * weight }
        }
        setDetailRelief(&ch, height: h, strength: strength, occlusionStrength: occlusionStrength)
        ch.fibreDetail = true
    }

    /// SplitMix64 — tiny, deterministic, stable across launches (never `hashValue`).
    private struct FibreRNG {
        var state: UInt64
        init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func unit() -> Double { Double(next() >> 11) / Double(1 << 53) }
    }
}
