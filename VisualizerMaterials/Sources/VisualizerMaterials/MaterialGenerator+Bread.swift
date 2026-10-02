import simd
import Foundation

/// **Sourdough crust — the baked skin of an artisan bâtard**, blisters and all.
///
/// Reference: a macro of a long-fermented, steam-baked loaf. What makes it read as sourdough
/// rather than "orange bread" is one structure — a **blister field**: raised, glassy domes from
/// pinhead (~1 mm) to ~9 mm across, CLUSTERED in patches (the dough's gas pockets bunch) with
/// bare, smoother, deep-copper crust between the patches. Each dome's crown is a thin pale-gold
/// glaze (low roughness, light albedo) ringed by a darker, thicker rim; the flat crust between is
/// matte and mottled mahogany → copper → gold on a 2–6 cm scale. Fine flour dust settles in the
/// hollows, and hairline cracks craze the bare stretches.
///
/// **Height-first.** Every dome is a smooth bump in `height`, the albedo/roughness are read off
/// the same dome parameters, and `normal` is derived from `height` at the TRUE physical slope
/// (`heightRangeMeters` over the tile's real pixel pitch), so a blister's lit and shadowed flanks
/// are the real ~35–50° and nothing in the colour can disagree with the relief
/// ([[feedback-single-source-of-truth]]).
///
/// **Not a pickable finish.** Registered `hiddenFromPicker` — a loaf mesh resolves it by id.
extension MaterialGenerator {

    /// **The physical size of one tile, in metres** — the loaf's UVs span this much crust per
    /// 0→1. 0.16 m at `sourdoughCrustBakeSize` px is 0.156 mm per texel, so a 1 mm pinhead
    /// blister is ~6 px across. The mesh owns the UV mapping; this is the number it reads.
    public static let sourdoughCrustTileMeters: Double = 0.16

    /// Default bake side. 1024 px over 0.16 m ⇒ 0.156 mm / px (the shared 512 would be 0.31 mm).
    public static let sourdoughCrustBakeSize = 1024

    /// Height range the 0…1 `height` channel spans, metres (the tallest 25 mm torn blister is ~7 mm; raised for DH macro relief).
    public static let sourdoughCrustHeightRangeMeters: Double = 0.011

    /// One blister layer's dome list — a jittered lattice thinned by the clustering field.
    private struct DomeLayer {
        let n: Int                 // cells per tile edge
        var present: [Bool]
        var px: [Double], py: [Double]   // centre, in cell units (unwrapped by the caller)
        var r: [Double]            // radius, cell units
        var tone: [Double]         // 0…1 per-dome crown lottery
        var stretch: [Double]      // elongation (≥ 1) along `angle`
        var angle: [Double]
        var phase: [Double]        // lobe phase for the irregular outline
        var big: Bool = false      // 15–25 mm torn / compound blister layer
        var lobeAmp: Double = 1    // outline irregularity multiplier
        var domeHeight: [Double]   // metres at the apex
    }

    /// A pixel's best dome so far.
    private struct DomeHit {
        var h = 0.0, d = 2.0, tone = 0.5, halo = 0.0, big = false
    }

    public static func sourdoughCrust(size: Int = MaterialGenerator.sourdoughCrustBakeSize,
                                      seed: UInt64 = 1101) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .ceramic)
        let sh = seed
        let tile = sourdoughCrustTileMeters
        let pxM = tile / Double(size)
        let range = sourdoughCrustHeightRangeMeters

        // ── Linear albedo palette (sRGB mahogany / copper / gold, then the glassy crown).
        let mahogany = Vec3(0.15, 0.030, 0.008)
        let toasted  = Vec3(0.10, 0.030, 0.010)    // near-black toasted brown
        let tan      = Vec3(0.80, 0.42, 0.11)       // light orange-gold
        let paleCrust = Vec3(0.74, 0.50, 0.27)      // flour-bleached, partly bare
        let cream    = Vec3(0.86, 0.70, 0.44)       // torn-skin edge
        let chestnut = Vec3(0.30, 0.070, 0.014)
        let copper   = Vec3(0.55, 0.150, 0.025)
        let golden   = Vec3(0.72, 0.300, 0.050)
        let crown    = Vec3(0.80, 0.58, 0.24)        // thin pale-gold glaze
        let crownLo  = Vec3(0.66, 0.38, 0.12)        // a less-blown crown
        let flour    = Vec3(0.85, 0.78, 0.62)
        let wrap = Noise.wrap

        // Clustering field: low-frequency, thresholded so patches are dense and between is bare.
        func density(_ u: Double, _ v: Double) -> Double {
            let n = Noise.fbmTiled(u, v, baseCells: 3, octaves: 3, gain: 0.55, seed: sh ^ 0xD1)
            return smoothstep(0.37, 0.57, n)
        }

        // ── Build the four blister layers: (cells over the tile, radius as a fraction of the
        // cell [lo, hi], apex height ÷ radius, presence ceiling).
        struct Spec { let n: Int; let rLo: Double; let rHi: Double; let hk: Double; let p: Double; let bare: Double; var big = false }
        let specs = [
            Spec(n: 5,   rLo: 0.23, rHi: 0.40, hk: 0.34, p: 0.34, bare: 0.04, big: true),   // 15–25 mm torn / compound blisters
            Spec(n: 10,  rLo: 0.20, rHi: 0.54, hk: 0.62, p: 0.58, bare: 0.02),  // 3–9 mm hero domes
            Spec(n: 24,  rLo: 0.22, rHi: 0.50, hk: 0.62, p: 0.48, bare: 0.03),  // 1.5–3.3 mm
            Spec(n: 56,  rLo: 0.25, rHi: 0.50, hk: 0.62, p: 0.32, bare: 0.04),  // 0.7–1.4 mm
            Spec(n: 130, rLo: 0.22, rHi: 0.46, hk: 0.62, p: 0.22, bare: 0.08),  // pinheads
        ]
        var layers: [DomeLayer] = []
        for (li, s) in specs.enumerated() {
            let cellM = tile / Double(s.n)
            var L = DomeLayer(n: s.n, present: [Bool](repeating: false, count: s.n * s.n),
                              px: [Double](repeating: 0, count: s.n * s.n),
                              py: [Double](repeating: 0, count: s.n * s.n),
                              r: [Double](repeating: 0, count: s.n * s.n),
                              tone: [Double](repeating: 0, count: s.n * s.n),
                              stretch: [Double](repeating: 1, count: s.n * s.n),
                              angle: [Double](repeating: 0, count: s.n * s.n),
                              phase: [Double](repeating: 0, count: s.n * s.n),
                              domeHeight: [Double](repeating: 0, count: s.n * s.n))
            let ls = sh ^ (0x1000 &* UInt64(li + 1))
            for cy in 0..<s.n {
                for cx in 0..<s.n {
                    let h = Noise.hash2(cx, cy, ls)
                    let jx = 0.15 + 0.70 * Noise.unit(h)
                    let jy = 0.15 + 0.70 * Noise.unit(Noise.mix(h))
                    let k = cy * s.n + cx
                    L.px[k] = Double(cx) + jx; L.py[k] = Double(cy) + jy
                    let dens = density((Double(cx) + jx) / Double(s.n), (Double(cy) + jy) / Double(s.n))
                    let prob = (s.bare + (s.p - s.bare) * dens) 
                    guard Noise.unit(Noise.mix(h ^ 0xAA55)) < prob else { continue }
                    L.present[k] = true
                    // Size lottery biased small: squared draw ⇒ many mid, few at the ceiling.
                    let q = Noise.unit(Noise.mix(h ^ 0x5151))
                    L.r[k] = s.rLo + (s.rHi - s.rLo) * (0.25 * q + 0.75 * q * q)
                    L.tone[k] = Noise.unit(Noise.mix(h ^ 0x7E7E))
                    L.stretch[k] = 1 + (s.big ? 0.8 : 0.55) * Noise.unit(Noise.mix(h ^ 0x1357)) * Noise.unit(Noise.mix(h ^ 0x2468))
                    L.angle[k] = Double.pi * Noise.unit(Noise.mix(h ^ 0x9999))
                    L.phase[k] = 6.2832 * Noise.unit(Noise.mix(h ^ 0x4242))
                    L.domeHeight[k] = s.hk * L.r[k] * cellM
                }
            }
            L.big = s.big; L.lobeAmp = s.big ? 2.6 : 1
            layers.append(L)
        }

        // Scan one layer's 3×3 neighbourhood; keep the TALLEST dome at the pixel (blisters that
        // merge keep a crease where they cross, as real ones do).
        func scan(_ L: DomeLayer, _ u: Double, _ v: Double, _ warp: Double, _ hit: inout DomeHit) {
            let n = L.n
            let x = u * Double(n), y = v * Double(n)
            let xi = Int(floor(x)), yi = Int(floor(y))
            for dj in -1...1 {
                for di in -1...1 {
                    let cx = xi + di, cy = yi + dj
                    let k = wrap(cy, n) * n + wrap(cx, n)
                    guard L.present[k] else { continue }
                    // toroidal: shift the stored centre to the unwrapped cell
                    let pcx = L.px[k] - Double(wrap(cx, n)) + Double(cx)
                    let pcy = L.py[k] - Double(wrap(cy, n)) + Double(cy)
                    let dx = x - pcx, dy = y - pcy
                    // Lumpy, elongated outline: rotate into the dome's frame, squash one axis, and
                    // modulate the radius with a few low harmonics + a shared warp noise so no
                    // two blisters are circles.
                    let ca = cos(L.angle[k]), sa = sin(L.angle[k])
                    let ex = (dx * ca + dy * sa) / L.stretch[k], ey = -dx * sa + dy * ca
                    let ang = atan2(ey, ex)
                    let lobes = 1 + L.lobeAmp * (0.07 * sin(3 * ang + L.phase[k]) + 0.04 * sin(5 * ang + 2 * L.phase[k])) + (L.big ? 0.08 * sin(2 * ang + 3 * L.phase[k]) : 0)
                    let dd = (ex * ex + ey * ey).squareRoot() / (L.r[k] * lobes * warp)
                    if dd < 1.7 {
                        hit.halo = max(hit.halo, 1 - smoothstep(1.0, 1.7, dd))
                    }
                    if dd < 1 {
                        let hh = L.domeHeight[k] * pow(1 - dd * dd, 0.55)
                        // smooth union: overlapping blisters fuse into one lumpy clump
                        let kk = 0.00035
                        let a = hit.h, diff = abs(hh - a)
                        let fused = max(hh, a) + (a > 0 ? max(kk - diff, 0) * max(kk - diff, 0) / (4 * kk) : 0)
                        if hh > a { hit.d = dd; hit.tone = L.tone[k]; hit.big = L.big }
                        hit.h = fused
                    }
                }
            }
        }

        struct Px { var a: Vec3; var r: Double; var h: Double }
        let rows = parallelMap(Array(0..<size)) { y -> [Px] in
            var out = [Px](); out.reserveCapacity(size)
            let v = (Double(y) + 0.5) / Double(size)
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)

                // ── Base crust: broad copper/gold mottle (2–6 cm) with toasted dark patches.
                let m1 = Noise.fbmTiled(u, v, baseCells: 4, octaves: 3, gain: 0.55, seed: sh ^ 0x31)
                let m2 = Noise.fbmTiled(u, v, baseCells: 6, octaves: 3, gain: 0.5, seed: sh ^ 0x32)
                let toast = smoothstep(0.50, 0.68, Noise.fbmTiled(u, v, baseCells: 3, octaves: 3, seed: sh ^ 0x33))
                // Large-scale (3–6 cm) zones, wide range: near-black toasted → mahogany → copper →
                // golden tan, plus flour-bleached pale patches.
                let zone = Noise.fbmTiled(u, v, baseCells: 3, octaves: 2, gain: 0.5, seed: sh ^ 0x35)
                let zn = clamp01((zone - 0.25) / 0.5)
                var base = mix(toasted, mahogany, smoothstep(0.0, 0.25, zn))
                base = mix(base, chestnut, smoothstep(0.18, 0.40, zn))
                base = mix(base, copper, smoothstep(0.38, 0.62, zn))
                base = mix(base, tan, smoothstep(0.62, 0.90, zn))
                base = mix(base, golden, smoothstep(0.58, 0.78, m2) * 0.45 * smoothstep(0.3, 0.6, zn))
                base = mix(base, mahogany, toast * 0.40)
                let bleach = smoothstep(0.64, 0.76, Noise.fbmTiled(u, v, baseCells: 4, octaves: 3, seed: sh ^ 0x36))
                base = mix(base, paleCrust, bleach * 0.62)
                // fine pixel-scale tone grain so bare crust is never a flat colour
                let grain = Noise.fbmTiled(u, v, baseCells: 96, octaves: 2, seed: sh ^ 0x34) - 0.5
                base = base * (1 + 0.28 * grain)

                // ── Blisters.
                var hit = DomeHit()
                let warp = 0.80 + 0.40 * Noise.fbmTiled(u, v, baseCells: 40, octaves: 2, seed: sh ^ 0x51)
                for L in layers { scan(L, u, v, warp, &hit) }
                let onDome = hit.h > 0
                let d = hit.d
                var col = base
                var rough = 0.78 + 0.06 * (Noise.fbmTiled(u, v, baseCells: 40, octaves: 2, seed: sh ^ 0x41) - 0.5)
                var h01 = 0.06 + 0.02 * m1

                // cavity darkening / flour trap around domes
                let cavity = hit.halo * (onDome ? 0 : 1)
                col = col * (1 - 0.42 * cavity * cavity - 0.12 * cavity)

                if onDome {
                    // Crown: pale glassy gold, per-dome lottery; thicker darker rim ring.
                    let crownAmt = 1 - smoothstep(0.55, 1.0, d)
                    let cc = mix(crownLo, crown, hit.tone)
                    // faint lit-from-above brightening toward the apex
                    let apex = cc * (1 + 0.10 * (1 - d))
                    col = mix(base * 1.05, apex, crownAmt * 0.97)
                    let ring = smoothstep(0.58, 0.86, d) * (1 - smoothstep(0.93, 1.0, d))
                    col = col * (1 - 0.40 * ring)
                    rough = mix(0.82, 0.25 + 0.05 * (1 - hit.tone), crownAmt)
                    rough += 0.20 * ring
                    if hit.big {
                        // torn blister skin: paler cream crown, cream edge, matte-ish skin at the lip
                        let edge = smoothstep(0.70, 0.95, d)
                        col = mix(col, cream, 0.55 * crownAmt + 0.45 * edge)
                        rough = mix(rough, 0.55, edge * 0.5)
                    }
                    h01 += hit.h / range
                }

                // ── Hairline cracks on the bare stretches — slightly paler grooves.
                let web = Noise.voronoiTiled(u, v, cells: 22, jitter: 1.0, seed: sh ^ 0x61)
                let crackZone = smoothstep(0.55, 0.70, Noise.fbmTiled(u, v, baseCells: 5, octaves: 2, seed: sh ^ 0x62))
                let crack = (1 - smoothstep(0.018, 0.045, web.f2 - web.f1)) * crackZone * (onDome ? 0.15 : 1)
                col = mix(col, golden * 1.15, crack * 0.55)
                h01 -= 0.03 * crack
                rough = mix(rough, 0.55, crack * 0.6)

                // ── Flour dust — pale matte speckle, thickest in the hollows beside domes.
                let fn = 0.55 * Noise.valueTiled(u, v, cells: 330, seed: sh ^ 0x71) + 0.45 * Noise.valueTiled(u, v, cells: 170, seed: sh ^ 0x73)
                let fm = Noise.valueTiled(u, v, cells: 150, seed: sh ^ 0x72)
                let hollow = max(cavity, onDome ? 0.0 : 0.25) + (onDome ? 0.12 * smoothstep(0.7, 1.0, d) : 0)
                let thr = 0.74 - 0.16 * hollow - 0.06 * fm - 0.22 * bleach
                let speck = smoothstep(thr, thr + 0.10, fn)
                col = mix(col, flour, speck * 0.78)
                rough = mix(rough, 0.95, speck * 0.85)
                h01 += 0.012 * speck

                out.append(Px(a: clampBand(col), r: clamp01(rough), h: clamp01(h01)))
            }
            return out
        }

        for y in 0..<size {
            for x in 0..<size {
                let p = rows[y][x]
                let i = ch.idx(x, y)
                ch.albedo[i] = p.a; ch.roughness[i] = p.r; ch.height[i] = p.h
            }
        }
        ch.clearcoat = 0.10
        ch.clearcoatRoughness = 0.25
        // Physically exact: the derivation turns Δh01 into tanθ = Δh01·strength over 2 px, so
        // strength = range ÷ (2 · pixel pitch) makes the baked slope the real slope.
        ch.deriveNormals(strength: range / (2 * pxM))
        // Sub-0.3 mm crumb/crackle tooth for close range (the matte-porous census requires it).
        addMicroDetail(&ch, seed: seed ^ 0xBB, baseCells: 260, strength: 0.45)
        return ch
    }
}
