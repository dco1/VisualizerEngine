import simd
import Foundation

/// **Sourdough crumb — the raw, torn dough surface and exposed crumb inside a loaf's score.**
///
/// What reads as "inside bread" is an open, irregular PORE structure (elliptical air pockets,
/// 0.5–4 mm with a few up to 8 mm, real depressions with darker warmer interiors and thin bright
/// walls between them), fibrous gluten streaks stretched along `u`, a ragged torn micro-surface and
/// a soft large-scale undulation where the dough pulled apart. A faint darker, toasted toning
/// toward one `v` direction (a periodic falloff, so the tile stays seamless) suggests the crust
/// next to it. Height-first: every channel reads the same pore list; `normal` is derived at the TRUE
/// physical slope from `sourdoughCrumbHeightRangeMeters` over the pixel pitch.
///
/// Hidden from every picker — a loaf's slit/ear resolves it by id.
extension MaterialGenerator {

    /// Physical edge of one tile, metres. 0.12 m at the default 1024 px is 0.117 mm per texel.
    public static let sourdoughCrumbTileMeters: Double = 0.12
    public static let sourdoughCrumbBakeSize = 1024
    /// Height range the 0…1 height channel spans, metres (pores are ~1–2 mm deep ≈ 0.25–0.4).
    public static let sourdoughCrumbHeightRangeMeters: Double = 0.005

    public static func sourdoughCrumb(size: Int = MaterialGenerator.sourdoughCrumbBakeSize,
                                      seed: UInt64 = 1201) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .ceramic)
        let sh = seed
        let tile = sourdoughCrumbTileMeters
        let pxM = tile / Double(size)
        let range = sourdoughCrumbHeightRangeMeters
        let wrap = Noise.wrap

        let ivory = Vec3(0.80, 0.66, 0.44)
        let tanDeep = Vec3(0.62, 0.45, 0.26)
        let poreCol = Vec3(0.30, 0.165, 0.07)       // darker, warmer interior
        let wallCol = Vec3(0.86, 0.73, 0.52)       // thin bright wall
        let toastCol = Vec3(0.50, 0.30, 0.13)
        let flour = Vec3(0.85, 0.78, 0.62)

        // Pore layers: cells per tile, radius range (fraction of cell), depth (m), presence.
        struct Spec { let n: Int; let rLo: Double; let rHi: Double; let depth: Double; let p: Double }
        let specs = [
            Spec(n: 12,  rLo: 0.22, rHi: 0.50, depth: 0.0020, p: 0.34),   // 10 mm cells: 2–5 mm radius (few to ~8 mm wide)
            Spec(n: 32,  rLo: 0.22, rHi: 0.50, depth: 0.0016, p: 0.80),   // 3.75 mm cells: 0.8–1.9 mm
            Spec(n: 80,  rLo: 0.22, rHi: 0.50, depth: 0.0011, p: 0.85),   // 1.5 mm cells: 0.3–0.75 mm
        ]
        struct Layer {
            let n: Int
            var present: [Bool], px: [Double], py: [Double], r: [Double]
            var stretch: [Double], angle: [Double], depth: [Double], tone: [Double]
        }
        var layers: [Layer] = []
        for (li, s) in specs.enumerated() {
            let c = s.n * s.n
            var L = Layer(n: s.n, present: [Bool](repeating: false, count: c),
                          px: [Double](repeating: 0, count: c), py: [Double](repeating: 0, count: c),
                          r: [Double](repeating: 0, count: c), stretch: [Double](repeating: 1, count: c),
                          angle: [Double](repeating: 0, count: c), depth: [Double](repeating: 0, count: c),
                          tone: [Double](repeating: 0, count: c))
            let ls = sh ^ (0x2000 &* UInt64(li + 1))
            for cy in 0..<s.n { for cx in 0..<s.n {
                let h = Noise.hash2(cx, cy, ls)
                let k = cy * s.n + cx
                L.px[k] = Double(cx) + 0.15 + 0.70 * Noise.unit(h)
                L.py[k] = Double(cy) + 0.15 + 0.70 * Noise.unit(Noise.mix(h))
                // open and closed regions of crumb
                let dn = Noise.fbmTiled(L.px[k] / Double(s.n), L.py[k] / Double(s.n), baseCells: 3, octaves: 2, seed: sh ^ 0xC1)
                let prob = s.p * (0.35 + 1.1 * smoothstep(0.30, 0.65, dn))
                guard Noise.unit(Noise.mix(h ^ 0xAA55)) < prob else { continue }
                L.present[k] = true
                let q = Noise.unit(Noise.mix(h ^ 0x5151))
                L.r[k] = s.rLo + (s.rHi - s.rLo) * (0.2 * q + 0.8 * q * q)
                L.stretch[k] = 1 + 1.3 * Noise.unit(Noise.mix(h ^ 0x1357))      // elongated…
                L.angle[k] = 0.35 * (Noise.unit(Noise.mix(h ^ 0x9999)) - 0.5)   // …roughly along u (stretch)
                L.depth[k] = s.depth * (0.6 + 0.4 * Noise.unit(Noise.mix(h ^ 0x7777)))
                L.tone[k] = Noise.unit(Noise.mix(h ^ 0x7E7E))
            } }
            layers.append(L)
        }

        struct Hit { var depth = 0.0, d = 2.0, wall = 0.0, tone = 0.5 }
        func scan(_ L: Layer, _ u: Double, _ v: Double, _ warp: Double, _ hit: inout Hit) {
            let n = L.n
            let x = u * Double(n), y = v * Double(n)
            let xi = Int(floor(x)), yi = Int(floor(y))
            for dj in -1...1 { for di in -1...1 {
                let cx = xi + di, cy = yi + dj
                let k = wrap(cy, n) * n + wrap(cx, n)
                guard L.present[k] else { continue }
                let dx = x - (L.px[k] - Double(wrap(cx, n)) + Double(cx))
                let dy = y - (L.py[k] - Double(wrap(cy, n)) + Double(cy))
                let ca = cos(L.angle[k]), sa = sin(L.angle[k])
                let ex = (dx * ca + dy * sa) / L.stretch[k], ey = -dx * sa + dy * ca
                let ang = atan2(ey, ex)
                let lobes = 1 + 0.12 * sin(3 * ang + 7 * L.tone[k]) + 0.06 * sin(5 * ang + 11 * L.tone[k])
                let dd = (ex * ex + ey * ey).squareRoot() / (L.r[k] * lobes * warp)
                if dd < 1.25 { hit.wall = max(hit.wall, smoothstep(0.85, 1.0, dd) * (1 - smoothstep(1.0, 1.25, dd))) }
                if dd < 1 {
                    let dep = L.depth[k] * pow(1 - dd * dd, 0.6)
                    if dep > hit.depth { hit.depth = dep; hit.d = dd; hit.tone = L.tone[k] }
                }
            } }
        }

        struct Px { var a: Vec3; var r: Double; var h: Double }
        let tau = 2 * Double.pi
        let rows = parallelMap(Array(0..<size)) { y -> [Px] in
            var out = [Px](); out.reserveCapacity(size)
            let v = (Double(y) + 0.5) / Double(size)
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                // soft large-scale undulation (dough pulled apart) + ragged torn micro-surface
                let wave = Noise.fbmTiled(u, v, baseCells: 3, octaves: 3, gain: 0.5, seed: sh ^ 0x11)
                let torn = Noise.fbmTiled(u, v, baseCells: 150, octaves: 3, gain: 0.55, seed: sh ^ 0x12) - 0.5
                // gluten streaks along u: few cells across u, very many across v
                let streak = Noise.fbmTiledAniso(u, v, cellsX: 4, cellsY: 70, octaves: 2, gain: 0.5, seed: sh ^ 0x13) - 0.5
                let fine = Noise.fbmTiledAniso(u, v, cellsX: 14, cellsY: 420, octaves: 1, seed: sh ^ 0x14) - 0.5

                var hit = Hit()
                let warp = 0.85 + 0.30 * Noise.fbmTiled(u, v, baseCells: 24, octaves: 2, seed: sh ^ 0x15)
                for L in layers { scan(L, u, v, warp, &hit) }
                let inPore = hit.depth > 0
                let dnorm = clamp01(hit.depth / 0.0020)

                // base tone: ivory → deeper tan in low/undulating zones, streak + torn mottle
                var col = mix(ivory, tanDeep, smoothstep(0.25, 0.75, 1 - wave) * 0.95)
                col = col * (1 + 0.22 * streak + 0.05 * fine + 0.30 * torn)
                // toasted toning toward v≈0 (periodic, so the tile still wraps)
                let toast = pow(0.5 + 0.5 * cos(tau * v), 3)
                col = mix(col, toastCol, toast * 0.55 * (0.6 + 0.8 * Noise.fbmTiled(u, v, baseCells: 6, octaves: 2, seed: sh ^ 0x16)))
                // bright thin walls between pores
                col = mix(col, wallCol, hit.wall * 0.55 * (inPore ? 0 : 1))
                var rough = 0.82 + 0.28 * torn + 0.14 * streak
                var h01 = 0.62 + 0.30 * (wave - 0.5) + 0.07 * streak + 0.12 * torn + 0.015 * fine
                if inPore {
                    col = mix(col, poreCol, (0.78 + 0.2 * dnorm) * smoothstep(0.0, 0.35, 1 - hit.d + 0.12))
                    col = col * (0.85 + 0.3 * hit.tone)
                    rough += 0.12 * dnorm
                    h01 -= hit.depth / range
                }
                // flour dust, light
                let fn = 0.6 * Noise.valueTiled(u, v, cells: 300, seed: sh ^ 0x71) + 0.4 * Noise.valueTiled(u, v, cells: 150, seed: sh ^ 0x72)
                let speck = smoothstep(0.80 - 0.08 * (inPore ? 0 : 1), 0.90, fn)
                col = mix(col, flour, speck * 0.55)
                rough = mix(rough, 0.95, speck * 0.7)
                out.append(Px(a: clampBand(col), r: min(0.95, max(0.70, rough)), h: clamp01(h01)))
            }
            return out
        }
        for y in 0..<size { for x in 0..<size {
            let p = rows[y][x], i = ch.idx(x, y)
            ch.albedo[i] = p.a; ch.roughness[i] = p.r; ch.height[i] = p.h
        } }
        ch.clearcoat = 0
        ch.deriveNormals(strength: range / (2 * pxM))
        addMicroDetail(&ch, seed: seed ^ 0xBB, baseCells: 300, strength: 0.5)
        return ch
    }
}
