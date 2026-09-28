import Foundation

/// The brute-force reference behind `NishitaMultipleScatteringTests.reference` (VZ-0159): the
/// engine's Nishita atmosphere (VolumetricSky.metal constants — Rayleigh / Mie / ozone, scale
/// heights, a 100 km top, Lambertian ground) solved WITHOUT the LUT method's approximations.
///
/// Radiance per unit solar irradiance, order by order: backward path tracing from the viewer
/// (1 m up), each vertex a volume scatter (true Rayleigh / Cornette–Shanks phase) or a Lambertian
/// ground bounce, with the single-scatter radiance arriving at every vertex integrated
/// DETERMINISTICALLY along the sampled direction (adaptive quadrature, planet shadow tested at
/// every point, sun transmittance from a fine brute-force optical-depth table). Order 1 is that
/// quadrature along the view ray itself. No isotropy assumption, no geometric series. Double
/// precision. `NishitaMultipleScatteringTests.testReferenceIntegratorReproducesTheStoredTable`
/// re-runs it (opt-in: `VIZ_NISHITA_REFERENCE_PATHS=<paths>`); the stored table is 400 000 paths
/// per configuration, orders 1…6, built with the release-compiled twin of this file.
enum NishitaReference {
    /// A pointer handed to `concurrentPerform` iterations that each write their own slot.
    struct Slots<T>: @unchecked Sendable { let p: UnsafeMutablePointer<T> }

    // Double-precision model of the engine's Nishita atmosphere (VolumetricSky.metal constants).
    // Units: metres. Radiance is per unit solar irradiance (E = 1 in every channel).

    typealias V3 = SIMD3<Double>

    @inline(__always) static func vexp(_ v: V3) -> V3 { V3(exp(v.x), exp(v.y), exp(v.z)) }
    @inline(__always) static func dot3(_ a: V3, _ b: V3) -> Double { a.x * b.x + a.y * b.y + a.z * b.z }
    @inline(__always) static func len3(_ a: V3) -> Double { sqrt(dot3(a, a)) }
    @inline(__always) static func norm3(_ a: V3) -> V3 { a / len3(a) }
    @inline(__always) static func cross3(_ a: V3, _ b: V3) -> V3 {
        V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }
    @inline(__always) static func luma(_ c: V3) -> Double { 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }

    struct Atmosphere {
        var Rg = 6360e3
        var Rt = 6460e3
        var HR = 7994.0
        var HM = 1200.0
        var betaR = V3(5.8e-6, 13.5e-6, 33.1e-6)
        var betaM = 21e-6
        var mieExt = 1.1
        var g = 0.76
        var betaO = V3(0.650e-6, 1.881e-6, 0.085e-6)
        var ozoneC = 25000.0
        var ozoneW = 15000.0
        var albedo = 0.0

        @inline(__always) func ozone(_ h: Double) -> Double { max(0, 1 - abs(h - ozoneC) / ozoneW) }

        /// (sigmaS Rayleigh per channel, sigmaS Mie, sigmaT per channel) at radius r.
        @inline(__always) func coeffs(_ r: Double) -> (sR: V3, sM: Double, t: V3) {
            let h = r - Rg
            if h > Rt - Rg { return (.zero, 0, .zero) }
            let dr = exp(-h / HR), dm = exp(-h / HM), dO = ozone(h)
            let sR = betaR * dr
            let sM = betaM * dm
            return (sR, sM, sR + V3(repeating: betaM * mieExt * dm) + betaO * dO)
        }
    }

    @inline(__always) static func phaseR(_ mu: Double) -> Double { 3.0 / (16.0 * Double.pi) * (1 + mu * mu) }
    @inline(__always) static func phaseM(_ mu: Double, _ g: Double) -> Double {
        let g2 = g * g
        return 3.0 / (8.0 * Double.pi) * ((1 - g2) * (1 + mu * mu))
            / ((2 + g2) * pow(max(1 + g2 - 2 * g * mu, 1e-4), 1.5))
    }
    @inline(__always) static func phaseHG(_ mu: Double, _ g: Double) -> Double {
        let g2 = g * g
        return (1 - g2) / (4 * Double.pi * pow(1 + g2 - 2 * g * mu, 1.5))
    }

    /// Far root of |o + t d|² = R² (d unit), −1 if missed.
    @inline(__always) static func raySphereFar(_ o: V3, _ d: V3, _ R: Double) -> Double {
        let b = dot3(o, d), c = dot3(o, o) - R * R
        let disc = b * b - c
        if disc < 0 { return -1 }
        return -b + sqrt(disc)
    }
    /// Nearest positive root, −1 if none.
    @inline(__always) static func raySphereNear(_ o: V3, _ d: V3, _ R: Double) -> Double {
        let b = dot3(o, d), c = dot3(o, o) - R * R
        let disc = b * b - c
        if disc < 0 { return -1 }
        let s = sqrt(disc)
        let t0 = -b - s
        if t0 > 0 { return t0 }
        let t1 = -b + s
        return t1 > 0 ? t1 : -1
    }

    /// Fine optical-depth table τ(r, μ) to the top of the atmosphere, Bruneton's mapping,
    /// node-aligned. Only valid for rays that do NOT hit the ground (callers test the shadow first).
    final class OpticalDepthTable: @unchecked Sendable {   // immutable after init
        let atm: Atmosphere
        let nr: Int, nmu: Int
        let H: Double
        let tau: [V3]

        init(atm: Atmosphere, nr: Int = 256, nmu: Int = 2048) {
            self.atm = atm
            self.nr = nr
            self.nmu = nmu
            self.H = sqrt(atm.Rt * atm.Rt - atm.Rg * atm.Rg)
            var table = [V3](repeating: .zero, count: nr * nmu)
            let H = self.H
            table.withUnsafeMutableBufferPointer { buf in
                let base = Slots(p: buf.baseAddress!)
                DispatchQueue.concurrentPerform(iterations: nr) { ir in
                    for imu in 0..<nmu {
                        let xr = Double(ir) / Double(nr - 1)
                        let xmu = Double(imu) / Double(nmu - 1)
                        let rho = H * xr
                        let r = sqrt(rho * rho + atm.Rg * atm.Rg)
                        let dmin = atm.Rt - r, dmax = rho + H
                        let d = dmin + xmu * (dmax - dmin)
                        var mu = d == 0 ? 1.0 : (H * H - rho * rho - d * d) / (2 * r * d)
                        mu = min(1, max(-1, mu))
                        base.p[ir * nmu + imu] = OpticalDepthTable.integrate(atm: atm, r: r, mu: mu)
                    }
                }
            }
            tau = table
        }

        /// Brute-force optical depth from radius r along zenith-cosine mu to the top boundary.
        static func integrate(atm: Atmosphere, r: Double, mu: Double) -> V3 {
            let o = V3(0, r, 0)
            let d = V3(sqrt(max(0, 1 - mu * mu)), mu, 0)
            let tEnd = raySphereFar(o, d, atm.Rt)
            if tEnd <= 0 { return .zero }
            var t = 0.0
            var od = V3.zero
            while t < tEnd {
                let p = o + d * t
                let rr = len3(p)
                let h = rr - atm.Rg
                let dhdt = abs(dot3(p, d) / rr)
                let Hs = h < 8000 ? atm.HM : atm.HR
                var dt = 0.01 * Hs / max(dhdt, 1e-5)
                dt = min(max(dt, 5.0), 1000.0)
                dt = min(dt, tEnd - t)
                let pm = o + d * (t + 0.5 * dt)
                od += atm.coeffs(len3(pm)).t * dt
                t += dt
            }
            return od
        }

        /// τ(r, μ) by bilinear interpolation (ray must not hit the ground).
        @inline(__always) func lookup(r: Double, mu: Double) -> V3 {
            let rho = sqrt(max(0, r * r - atm.Rg * atm.Rg))
            let xr = min(1, rho / H)
            let disc = r * r * (mu * mu - 1) + atm.Rt * atm.Rt
            let d = max(0, -r * mu + sqrt(max(0, disc)))
            let dmin = atm.Rt - r, dmax = rho + H
            let xmu = dmax > dmin ? min(1, max(0, (d - dmin) / (dmax - dmin))) : 0
            let fr = xr * Double(nr - 1), fm = xmu * Double(nmu - 1)
            let ir = min(Int(fr), nr - 2), im = min(Int(fm), nmu - 2)
            let ar = fr - Double(ir), am = fm - Double(im)
            let t00 = tau[ir * nmu + im], t01 = tau[ir * nmu + im + 1]
            let t10 = tau[(ir + 1) * nmu + im], t11 = tau[(ir + 1) * nmu + im + 1]
            return (t00 * (1 - am) + t01 * am) * (1 - ar) + (t10 * (1 - am) + t11 * am) * ar
        }

        /// Transmittance to the sun from point p (planet shadow tested analytically).
        @inline(__always) func sunTrans(_ p: V3, _ s: V3) -> V3 {
            let r = len3(p)
            let mu = dot3(p, s) / r
            if mu < 0 && r * r * (1 - mu * mu) < atm.Rg * atm.Rg { return .zero }
            if r >= atm.Rt { return V3(1, 1, 1) }
            return vexp(-lookup(r: r, mu: mu))
        }
    }


    // Brute-force multiple-scattering reference: order-resolved backward path tracing through the
    // spherical-shell atmosphere, with the single-scatter radiance arriving at every path vertex
    // integrated DETERMINISTICALLY along the sampled direction (fine adaptive quadrature, planet
    // shadow tested at every quadrature point, sun transmittance from a fine brute-force table).
    // No isotropy assumption, no geometric series: order n = n scattering events (volume, or a
    // Lambertian ground bounce when albedo > 0), each with the true Rayleigh / Cornette–Shanks
    // phase. Double precision throughout.

    struct Seg {
        var t0: Double
        var dt: Double
        var sT: V3
        var sR: V3
        var sM: Double
        var T0: V3
    }

    struct RayResult {
        var L: V3          // single-scatter radiance arriving at the ray origin (+ direct sun off the ground end)
        var Tend: V3
        var tEnd: Double
        var hitGround: Bool
    }

    @inline(__always) static func marchL1(_ atm: Atmosphere, _ tab: OpticalDepthTable, _ x: V3, _ w: V3, _ s: V3,
                                   _ segs: inout [Seg], record: Bool, stepScale: Double = 1) -> RayResult {
        let tTop = raySphereFar(x, w, atm.Rt)
        let tG = raySphereNear(x, w, atm.Rg)
        let hitGround = tG > 0
        let tEnd = hitGround ? tG : max(tTop, 0)
        let mu = dot3(w, s)
        let pR = phaseR(mu), pM = phaseM(mu, atm.g)
        var t = 0.0
        var T = V3(1, 1, 1)
        var L = V3.zero
        if record { segs.removeAll(keepingCapacity: true) }
        while t < tEnd {
            let p = x + w * t
            let r = len3(p)
            let h = r - atm.Rg
            let dhdt = abs(dot3(p, w) / r)
            let Hs = h < 6000 ? atm.HM : atm.HR
            var dt = stepScale * 0.05 * Hs / max(dhdt, 1e-4)
            dt = min(max(dt, 10.0), 4000.0 * stepScale)
            dt = min(dt, tEnd - t)
            let pm = x + w * (t + 0.5 * dt)
            let c = atm.coeffs(len3(pm))
            let Ts = tab.sunTrans(pm, s)
            let S = (c.sR * pR + V3(repeating: c.sM * pM)) * Ts
            let Tseg = vexp(-c.t * dt)
            L += T * S * (V3(1, 1, 1) - Tseg) / c.t
            if record { segs.append(Seg(t0: t, dt: dt, sT: c.t, sR: c.sR, sM: c.sM, T0: T)) }
            T *= Tseg
            t += dt
        }
        if hitGround && atm.albedo > 0 {
            let pg = x + w * tEnd
            let n = norm3(pg)
            let cs = dot3(n, s)
            if cs > 0 { L += T * (atm.albedo / Double.pi) * cs * tab.sunTrans(pg * (1 + 1e-9), s) }
        }
        return RayResult(L: L, Tend: T, tEnd: tEnd, hitGround: hitGround)
    }

    struct RNG {
        var s0: UInt64, s1: UInt64, s2: UInt64, s3: UInt64
        init(seed: UInt64) {
            var z = seed &+ 0x9E3779B97F4A7C15
            func mix(_ zz: inout UInt64) -> UInt64 {
                zz = zz &+ 0x9E3779B97F4A7C15
                var x = zz
                x = (x ^ (x >> 30)) &* 0xBF58476D1CE4E5B9
                x = (x ^ (x >> 27)) &* 0x94D049BB133111EB
                return x ^ (x >> 31)
            }
            s0 = mix(&z); s1 = mix(&z); s2 = mix(&z); s3 = mix(&z)
        }
        @inline(__always) mutating func next() -> UInt64 {
            let result = ((s0 &+ s3) << 23 | (s0 &+ s3) >> 41) &+ s0
            let t = s1 << 17
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t
            s3 = (s3 << 45) | (s3 >> 19)
            return result
        }
        @inline(__always) mutating func uniform() -> Double { Double(next() >> 11) * (1.0 / 9007199254740992.0) }
    }

    struct DirSampler {
        let g: Double
        // Sample a new direction around the current ray direction w at vertex x. Returns (dir, pdf).
        @inline(__always) static func basis(_ n: V3) -> (V3, V3) {
            let a = abs(n.x) > 0.9 ? V3(0, 1, 0) : V3(1, 0, 0)
            let t = norm3(cross3(a, n))
            return (t, cross3(n, t))
        }
        @inline(__always) static func uniformSphere(_ u1: Double, _ u2: Double) -> V3 {
            let z = 1 - 2 * u1
            let r = sqrt(max(0, 1 - z * z))
            let phi = 2 * Double.pi * u2
            return V3(r * cos(phi), r * sin(phi), z)
        }
    }

    /// Sunward band around the local horizon toward the sun (used when the sun is low / below).
    struct Band {
        let up: V3, sh: V3, b: V3
        let phiMax: Double, sMin: Double, sMax: Double
        var pdf: Double { 1.0 / (2 * phiMax * (sMax - sMin)) }
        init(x: V3, s: V3) {
            up = norm3(x)
            var h = s - up * dot3(s, up)
            if len3(h) < 1e-9 { h = DirSampler.basis(up).0 }
            sh = norm3(h)
            b = cross3(up, sh)
            phiMax = Double.pi * 0.5
            sMin = sin(-8.0 * Double.pi / 180)
            sMax = sin(25.0 * Double.pi / 180)
        }
        func sample(_ u1: Double, _ u2: Double) -> V3 {
            let phi = (2 * u1 - 1) * phiMax
            let se = sMin + u2 * (sMax - sMin)
            let ce = sqrt(max(0, 1 - se * se))
            return (sh * cos(phi) + b * sin(phi)) * ce + up * se
        }
        func contains(_ d: V3) -> Bool {
            let se = dot3(d, up)
            if se < sMin || se > sMax { return false }
            let x = dot3(d, sh), y = dot3(d, b)
            let phi = atan2(y, x)
            return abs(phi) <= phiMax
        }
    }

    struct MCResult {
        var order: [V3]      // index 1…K: order n contribution (index 0 unused)
        var stderrMS: V3     // standard error of the orders ≥ 2 sum
        var paths: Int
    }

    /// Order-resolved radiance toward the camera at `x0` looking along `w0`, sun toward `s`.
    static func mcRadiance(atm: Atmosphere, tab: OpticalDepthTable, x0: V3, w0: V3, s: V3,
                    paths: Int, maxOrder: Int, seed: UInt64) -> MCResult {
        var camRecord: [Seg] = []
        let cam = marchL1(atm, tab, x0, w0, s, &camRecord, record: true)
        let camSegs = camRecord
        let chunks = 64
        let perChunk = (paths + chunks - 1) / chunks
        var partial = [[V3]](repeating: [V3](repeating: .zero, count: maxOrder + 1), count: chunks)
        var partialSq = [V3](repeating: .zero, count: chunks)
        partial.withUnsafeMutableBufferPointer { pb in
            partialSq.withUnsafeMutableBufferPointer { sb in
                let pbuf = Slots(p: pb.baseAddress!), sqbuf = Slots(p: sb.baseAddress!)
                DispatchQueue.concurrentPerform(iterations: chunks) { ci in
                    var rng = RNG(seed: seed &+ UInt64(ci) &* 7919)
                    var acc = [V3](repeating: .zero, count: maxOrder + 1)
                    var accSq = V3.zero
                    var segs: [Seg] = []
                    segs.reserveCapacity(2048)
                    var cur = camSegs
                    cur.reserveCapacity(2048)
                    for _ in 0..<perChunk {
                        var pathSum = V3.zero
                        var beta = V3(1, 1, 1)
                        var x = x0, w = w0
                        var ray = cam
                        cur.removeAll(keepingCapacity: true); cur.append(contentsOf: camSegs)
                        for k in 1..<maxOrder {
                            // ── vertex k along (x, w)
                            let TgEnd = ray.Tend.y
                            var groundVertex = false
                            let u = rng.uniform()
                            var xn = x
                            var sR = V3.zero, sM = 0.0
                            // Analog free flight when the ray ends on the ground: τ* ≥ τ_end ⇒ the ground
                            // (probability T_end); a ray that escapes to space forces a volume collision.
                            if ray.hitGround && u >= 1 - TgEnd { groundVertex = true }
                            if groundVertex {
                                if atm.albedo <= 0 { break }
                                beta *= ray.Tend / TgEnd
                                xn = x + w * ray.tEnd
                                xn = norm3(xn) * (atm.Rg + 0.01)
                                beta *= atm.albedo
                                let n = norm3(xn)
                                let (t1, t2) = DirSampler.basis(n)
                                let u1 = rng.uniform(), u2 = rng.uniform()
                                let r = sqrt(u1), phi = 2 * Double.pi * u2
                                let wn = t1 * (r * cos(phi)) + t2 * (r * sin(phi)) + n * sqrt(max(0, 1 - u1))
                                x = xn; w = wn
                            } else {
                                // target optical depth in green
                                let tauStar: Double
                                var forcedW = 1.0
                                if ray.hitGround {
                                    tauStar = -log(1 - u)  // analog: u < 1 − T_end ⇒ τ* < τ_end
                                } else {
                                    let q = 1 - TgEnd
                                    if q <= 0 { break }
                                    tauStar = -log(1 - u * q)
                                    forcedW = q
                                }
                                // locate
                                var tauAcc = 0.0
                                var found = -1
                                var tIn = 0.0
                                for (i, sg) in cur.enumerated() {
                                    let dtau = sg.sT.y * sg.dt
                                    if tauAcc + dtau >= tauStar {
                                        found = i
                                        tIn = (tauStar - tauAcc) / sg.sT.y
                                        break
                                    }
                                    tauAcc += dtau
                                }
                                if found < 0 { found = cur.count - 1; tIn = cur[found].dt * 0.999 }
                                let sg = cur[found]
                                let Tc = sg.T0 * vexp(-sg.sT * tIn)
                                beta *= Tc * forcedW / (Tc.y * sg.sT.y)
                                xn = x + w * (sg.t0 + tIn)
                                sR = sg.sR; sM = sg.sM
                                // direction sampling (mixture)
                                let band = Band(x: xn, s: s)
                                let sunLow = dot3(norm3(xn), s) < 0.1
                                let wU = sunLow ? 0.45 : 0.8, wH = 0.2, wB = sunLow ? 0.35 : 0.0
                                let pick = rng.uniform()
                                var wn: V3
                                if pick < wU {
                                    wn = DirSampler.uniformSphere(rng.uniform(), rng.uniform())
                                } else if pick < wU + wH {
                                    let g = atm.g
                                    let uu = rng.uniform()
                                    let sq = (1 - g * g) / (1 - g + 2 * g * uu)
                                    let ct = min(1, max(-1, (1 + g * g - sq * sq) / (2 * g)))
                                    let st = sqrt(max(0, 1 - ct * ct))
                                    let phi = 2 * Double.pi * rng.uniform()
                                    let (t1, t2) = DirSampler.basis(w)
                                    wn = w * ct + t1 * (st * cos(phi)) + t2 * (st * sin(phi))
                                } else {
                                    wn = band.sample(rng.uniform(), rng.uniform())
                                }
                                wn = norm3(wn)
                                let ct = dot3(w, wn)
                                var pdf = wU / (4 * Double.pi) + wH * phaseHG(ct, atm.g)
                                if wB > 0 && band.contains(wn) { pdf += wB * band.pdf }
                                let f = sR * phaseR(ct) + V3(repeating: sM * phaseM(ct, atm.g))
                                beta *= f / pdf
                                x = xn; w = wn
                            }
                            // ── single-scatter radiance arriving at the vertex from direction w
                            ray = marchL1(atm, tab, x, w, s, &segs, record: true)
                            swap(&cur, &segs)
                            let c = beta * ray.L
                            acc[k + 1] += c
                            pathSum += c
                        }
                        accSq += pathSum * pathSum
                    }
                    pbuf.p[ci] = acc
                    sqbuf.p[ci] = accSq
                }
            }
        }
        let n = Double(perChunk * chunks)
        var order = [V3](repeating: .zero, count: maxOrder + 1)
        for c in partial { for k in 0...maxOrder { order[k] += c[k] } }
        for k in 0...maxOrder { order[k] /= n }
        order[1] = cam.L
        var sq = V3.zero
        for c in partialSq { sq += c }
        let msMean = order[2...].reduce(V3.zero, +)
        let variance = sq / n - msMean * msMean
        let se = V3(sqrt(max(0, variance.x) / n), sqrt(max(0, variance.y) / n), sqrt(max(0, variance.z) / n))
        return MCResult(order: order, stderrMS: se, paths: Int(n))
    }
}
