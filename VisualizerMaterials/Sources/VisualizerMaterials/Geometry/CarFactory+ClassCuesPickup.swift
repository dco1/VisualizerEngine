import Foundation
import simd

// plausibility: real — the bed is a true recess carved (localised BSP) out of the shell, with wheel-arch humps, bulkhead,
// tailgate, rail caps and tie-down rings as separate solids; no texture or fake slab.

/// The numbers of the carved bed, shared with the tests.
struct PickupBedLayout {
    var zFront: Double, zRear: Double        // cavity front (against the cab) and rear (inside the tailgate)
    var floorY: Double, rimY: Double         // floor level, top of the side walls
    var hw: [(z: Double, hw: Double)]        // inner half-width by z
    var humpWidth = 0.235, humpHalfZ = 0.52, humpHeight = 0.33     // wheel-arch hump blocks (inboard width, half length, height over the floor)
    var humpAxles: [Double] = []
    var hwMin: Double { hw.map(\.hw).min() ?? 0 }
    func hwAt(_ z: Double) -> Double {
        guard let f = hw.first, let l = hw.last else { return 0 }
        if z >= f.z { return f.hw }
        if z <= l.z { return l.hw }
        for i in 1..<hw.count where z >= hw[i].z {
            let a = hw[i - 1], b = hw[i]
            let t = (a.z - z) / max(a.z - b.z, 1e-9)
            return a.hw + (b.hw - a.hw) * t
        }
        return l.hw
    }
}

extension CarContext {

    /// Bed layout read off the shell (descending z stations).
    func pickupBedLayout(_ body: CCProbe) -> PickupBedLayout? {
        // The step from the cab roof line down to the bed rim: first drop > 5 cm in the blueprint top line.
        var zStep: Double? = nil
        for i in 1..<bp.top.count where bp.top[i].1 < bp.top[i - 1].1 - 0.05 { zStep = frontZ - bp.top[i].0 * rL; break }
        guard let zs = zStep else { return nil }
        let zFront = zs - 0.02, zRear = rearZ + 0.075
        guard zFront - zRear > 1.0 else { return nil }
        // few stations (the walls are planar between them, which keeps the boolean cheap); linear chords of the
        // concave tail taper sit inside the real flank, so the walls are never thinner than the 7.5 cm design
        let zsT = [zFront, zRear + 1.0, zRear + 0.5, zRear].filter { $0 <= zFront }
        let st: [(Double, Double)] = zsT.map { z in
            let fl = min(body.flank(0.94, z)?.x ?? 0.9, body.flank(0.90, z)?.x ?? 0.9)
            return (z, fl - 0.075)
        }
        let hw = st.map { $0.1 }
        let rim = body.top(0.0, zFront - 0.6) ?? 1.05
        var lay = PickupBedLayout(zFront: zFront, zRear: zRear, floorY: 0.58, rimY: rim, hw: zip(st, hw).map { (z: $0.0.0, hw: $0.1) })
        lay.humpAxles = axles.filter { $0 < zFront - 0.3 && $0 > zRear + 0.3 }
        return lay
    }

    func buildPickupBed(_ a: inout CarAccumulator) {
        let probe = CCProbe(a.body)
        guard let L = pickupBedLayout(probe) else { return }

        // ── 1. carve the recess out of the main shell component (the lips/cladding are welded into it) ──
        let comps = ccComponents(a.body)
        guard let mainIdx = comps.indices.max(by: { comps[$0].triangleCount < comps[$1].triangleCount }) else { return }
        let cavity = ccBedCavity(L)
        // symmetrised: the boolean's triangulation is not mirror-symmetric, the bed is
        let carved = ccMirrorSymmetrized(comps[mainIdx].subtracting(cavity))
        // T-junctions left by the boolean are tolerated for this one part (CarFactoryTests.testEveryPartIsSound documents it)
        var rest = Mesh3()
        for (i, c) in comps.enumerated() where i != mainIdx { rest.append(c) }
        a.body = carved
        a.body.append(rest)

        // ── 2. liner (dark): floor, wall panels, bulkhead and tailgate-inside, ribs ──
        var liner = Mesh3()
        let hmin = L.hwMin
        let topFront = (probe.top(0, L.zFront + 0.015) ?? L.rimY) + 0.004
        let wallTop = max(L.rimY, probe.top(L.hwMin + 0.03, L.zRear + 0.02) ?? 0, probe.top(L.hwAt(L.zFront) + 0.03, L.zFront) ?? 0) + 0.006
        // floor liner: full width between the hump blocks, trimmed to the central strip alongside them
        var slabZ: Set<Double> = [L.zRear - 0.003, L.zFront + 0.003]
        for q in L.hw { slabZ.insert(q.z) }
        var z0s = L.zRear + 0.20
        while z0s < L.zFront - 0.10 { slabZ.insert(z0s); z0s += 0.20 }
        for ax in L.humpAxles { for d in [-0.56, -0.53, -0.51, 0.51, 0.53, 0.56] { slabZ.insert(ax + d) } }
        let slabRows = slabZ.sorted().filter { $0 > L.zRear - 0.01 || $0 == L.zRear - 0.003 }
        func slabHW(_ z: Double) -> Double {
            for ax in L.humpAxles {
                let d = abs(z - ax)
                let inner = L.hwAt(z) - L.humpWidth - 0.006
                if d < 0.52 { return inner }
                if d < 0.55 { return inner + (L.hwAt(z) - 0.003 - inner) * (d - 0.52) / 0.03 }
            }
            return L.hwAt(z) - 0.003
        }
        liner.append(GHKit.slab(zs: slabRows, hw: slabHW, nx: 8, thickness: 0.015) { _, _ in L.floorY + 0.011 })
        for i in 0..<6 {
            let x = -0.50 + 0.20 * Double(i)
            liner.append(carBevelBox(Vec3(x - 0.020, L.floorY + 0.009, L.zRear + 0.04), Vec3(x + 0.020, L.floorY + 0.022, L.zFront - 0.04), 0.004))
        }
        // wall panels, in the stretches between the wheel humps (the hump liners cover the rest)
        var spans: [(Double, Double)] = [(L.zFront, L.zRear)]
        for ax in L.humpAxles {
            var out: [(Double, Double)] = []
            for (a, b) in spans {
                let hi = ax + L.humpHalfZ + 0.02, lo = ax - L.humpHalfZ - 0.02
                if b >= hi || a <= lo { out.append((a, b)); continue }
                if a > hi { out.append((a, hi)) }
                if b < lo { out.append((lo, b)) }
            }
            spans = out
        }
        for sx in [1.0, -1.0] {
            for (a0, b0) in spans where a0 - b0 > 0.1 {
                var zsP: Set<Double> = [a0 >= L.zFront - 1e-9 ? a0 + 0.015 : a0, b0 <= L.zRear + 1e-9 ? b0 - 0.015 : b0]
                for q in L.hw where q.z < a0 && q.z > b0 { zsP.insert(q.z) }
                var zz = b0 + 0.25
                while zz < a0 - 0.05 { zsP.insert(zz); zz += 0.25 }
                let path = zsP.sorted(by: >).map { Vec3(sx * (L.hwAt($0) - 0.012), (L.floorY + wallTop) / 2, $0) }
                liner.append(ccBar(path, w: 0.024, h: wallTop - L.floorY, corner: 0.004))
            }
        }
        // wall above each hump block (the block's own liner stops at its top)
        for ax in L.humpAxles {
            for sx in [1.0, -1.0] {
                let yb = L.floorY + L.humpHeight + 0.004
                let path = [Vec3(sx * (L.hwAt(ax + L.humpHalfZ) - 0.012), (yb + wallTop) / 2, ax + L.humpHalfZ + 0.02),
                            Vec3(sx * (L.hwAt(ax - L.humpHalfZ) - 0.012), (yb + wallTop) / 2, ax - L.humpHalfZ - 0.02)]
                liner.append(ccBar(path, w: 0.024, h: wallTop - yb, corner: 0.004))
            }
        }
        // bulkhead (front) and inside of the tailgate (rear), each with two horizontal ribs
        for (zc, dirz) in [(L.zFront, -1.0), (L.zRear, 1.0)] {
            let hz = L.hwAt(zc) + 0.002
            let yTop = dirz < 0 ? topFront : wallTop
            liner.append(carBevelBox(Vec3(-hz, L.floorY + 0.008, zc - 0.003 - (dirz < 0 ? 0.012 : -0.0)),
                                     Vec3(hz, yTop, zc + 0.003 + (dirz < 0 ? 0 : 0.012)), 0.003))
            for k in 1...2 {
                let y = L.floorY + (yTop - L.floorY) * Double(k) / 3
                let zr = dirz < 0 ? zc - 0.020 : zc + 0.020
                liner.append(carBevelBox(Vec3(-(hz - 0.03), y - 0.014, min(zc, zr) - 0.003), Vec3(hz - 0.03, y + 0.014, max(zc, zr) + 0.003), 0.004))
            }
        }
        // wheel-arch humps: the body keeps solid blocks round the rear wheels (the cavity is cut around them); the
        // black liner wraps each block 8 mm proud on its top and three visible faces
        for ax in axles where ax < L.zFront && ax > L.zRear {
            for sx in [1.0, -1.0] {
                let hwz = L.hwAt(ax)
                let xi = hwz - L.humpWidth - 0.012, xo = hwz + 0.012
                let m = carBevelBox(Vec3(min(sx * xi, sx * xo), L.floorY - 0.005, ax - L.humpHalfZ - 0.014),
                                    Vec3(max(sx * xi, sx * xo), L.floorY + L.humpHeight + 0.020, ax + L.humpHalfZ + 0.014), 0.012)
                liner.append(m)
            }
        }
        a.trim.append(liner)   // black plastic drop-in bedliner

        // ── 3. rail caps (sides + across the tailgate top) and 4 tie-down rings ──
        var caps = Mesh3()
        for sx in [1.0, -1.0] {
            var path: [Vec3] = []
            let nd = Int(((L.zFront - L.zRear) / 0.10).rounded(.up))
            for i in stride(from: nd, through: 0, by: -1) {
                let z = L.zRear + (L.zFront - L.zRear) * Double(i) / Double(nd)
                let xc = L.hwAt(z) + 0.036
                path.append(Vec3(sx * xc, (probe.top(xc, z) ?? L.rimY) + 0.003, z))
            }
            // extend over the front of the wall up to the cab
            caps.append(ccBar(path, w: 0.074, h: 0.020, corner: 0.008))
        }
        var tg: [Vec3] = []
        let zt = rearZ + 0.040
        for i in 0...14 {
            let x = -(L.hwAt(L.zRear) + 0.07) + 2 * (L.hwAt(L.zRear) + 0.07) * Double(i) / 14
            tg.append(Vec3(x, (probe.top(x, zt) ?? L.rimY) + 0.003, zt))
        }
        caps.append(ccBar(tg, w: 0.074, h: 0.020, corner: 0.008))
        if era == .modern { a.trim.append(caps) } else { a.chrome.append(caps) }

        var hooks = Mesh3()
        for zf in [0.22, 0.84] {
            let z = L.zFront - (L.zFront - L.zRear) * zf
            for sx in [1.0, -1.0] {
                let hw = L.hwAt(z), y = L.rimY - 0.15
                hooks.append(carBevelBox(Vec3(min(sx * (hw - 0.010), sx * (hw + 0.004)), y - 0.026, z - 0.032),
                                         Vec3(max(sx * (hw - 0.010), sx * (hw + 0.004)), y + 0.026, z + 0.032), 0.004))
                hooks.append(GHKit.torus(centre: Vec3(sx * (hw - 0.020), y, z), axis: Vec3(1, 0, 0), big: 0.020, small: 0.0045, around: 20, tube: 8))
            }
        }
        a.chrome.append(hooks)
    }

    /// The bed cavity as one closed solid: flat lid, side walls following the (linear) inner half-width, a floor at
    /// `floorY` that rises to the hump height inside the wheel-arch blocks. Few planes on purpose (boolean cutter).
    func ccBedCavity(_ L: PickupBedLayout) -> Mesh3 {
        var zr: Set<Double> = [L.zFront, L.zRear]
        for p in L.hw { zr.insert(p.z) }
        for ax in L.humpAxles { zr.insert(ax + L.humpHalfZ); zr.insert(ax - L.humpHalfZ) }
        let rows = zr.sorted(by: >)                         // descending z (front → rear)
        let top = L.rimY + 0.30
        func inHump(_ zMid: Double) -> Bool { L.humpAxles.contains { abs(zMid - $0) < L.humpHalfZ - 1e-9 } }
        var m = Mesh3()
        func v(_ x: Double, _ y: Double, _ z: Double) -> Vec3 { Vec3(x, y, z) }
        for r in 0..<(rows.count - 1) {
            let z0 = rows[r], z1 = rows[r + 1]
            let h0 = L.hwAt(z0), h1 = L.hwAt(z1)
            let hump = inHump((z0 + z1) / 2)
            let yf = hump ? L.floorY + L.humpHeight : L.floorY
            let xb0 = h0 - L.humpWidth, xb1 = h1 - L.humpWidth
            // lid
            m.addQuad(v(-h0, top, z0), v(h0, top, z0), v(h1, top, z1), v(-h1, top, z1), outward: Vec3(0, 1, 0))
            // floor: central strip (+ notch strips raised in hump rows)
            m.addQuad(v(-xb0, L.floorY, z0), v(xb0, L.floorY, z0), v(xb1, L.floorY, z1), v(-xb1, L.floorY, z1), outward: Vec3(0, -1, 0))
            m.addQuad(v(xb0, yf, z0), v(h0, yf, z0), v(h1, yf, z1), v(xb1, yf, z1), outward: Vec3(0, -1, 0))
            m.addQuad(v(-h0, yf, z0), v(-xb0, yf, z0), v(-xb1, yf, z1), v(-h1, yf, z1), outward: Vec3(0, -1, 0))
            // side walls
            m.addQuad(v(h0, yf, z0), v(h1, yf, z1), v(h1, top, z1), v(h0, top, z0), outward: Vec3(1, 0, 0))
            m.addQuad(v(-h0, yf, z0), v(-h1, yf, z1), v(-h1, top, z1), v(-h0, top, z0), outward: Vec3(-1, 0, 0))
            // hump inboard faces
            if hump {
                m.addQuad(v(xb0, L.floorY, z0), v(xb1, L.floorY, z1), v(xb1, yf, z1), v(xb0, yf, z0), outward: Vec3(1, 0, 0))
                m.addQuad(v(-xb0, L.floorY, z0), v(-xb1, L.floorY, z1), v(-xb1, yf, z1), v(-xb0, yf, z0), outward: Vec3(-1, 0, 0))
            }
        }
        // hump front/rear faces (between a hump row and a flat row)
        for r in 0..<(rows.count - 2) {
            let zb = rows[r + 1]
            let a = inHump((rows[r] + rows[r + 1]) / 2), b = inHump((rows[r + 1] + rows[r + 2]) / 2)
            if a == b { continue }
            let hb = L.hwAt(zb), xb = hb - L.humpWidth
            let yt = L.floorY + L.humpHeight
            let outward = Vec3(0, 0, a ? 1 : -1)             // the lower-floor (flat) side's solid faces the raised block
            m.addQuad(v(xb, L.floorY, zb), v(hb, L.floorY, zb), v(hb, yt, zb), v(xb, yt, zb), outward: outward)
            m.addQuad(v(-hb, L.floorY, zb), v(-xb, L.floorY, zb), v(-xb, yt, zb), v(-hb, yt, zb), outward: outward)
        }
        // end walls
        let hf = L.hwAt(L.zFront), hr = L.hwAt(L.zRear)
        m.addQuad(v(-hf, L.floorY, L.zFront), v(hf, L.floorY, L.zFront), v(hf, top, L.zFront), v(-hf, top, L.zFront), outward: Vec3(0, 0, 1))
        m.addQuad(v(-hr, L.floorY, L.zRear), v(hr, L.floorY, L.zRear), v(hr, top, L.zRear), v(-hr, top, L.zRear), outward: Vec3(0, 0, -1))
        return m
    }

    /// Connected components (by 0.1 mm-quantised shared vertices) of a triangle soup, as separate meshes.
    func ccComponents(_ m: Mesh3) -> [Mesh3] {
        var parent: [SIMD3<Int64>: SIMD3<Int64>] = [:]
        func find(_ x: SIMD3<Int64>) -> SIMD3<Int64> {
            var r = x
            while let p = parent[r], p != r { r = p }
            var c = x
            while let p = parent[c], p != r { parent[c] = r; c = p }
            return r
        }
        var i = 0
        while i + 2 < m.indices.count {
            let k = (0..<3).map { Mesh3.qpos(m.positions[Int(m.indices[i + $0])]) }
            for q in k where parent[q] == nil { parent[q] = q }
            let r0 = find(k[0])
            for q in k.dropFirst() { let r = find(q); if r != r0 { parent[r] = r0 } }
            i += 3
        }
        var order: [SIMD3<Int64>] = [], out: [SIMD3<Int64>: Mesh3] = [:]
        i = 0
        while i + 2 < m.indices.count {
            let r = find(Mesh3.qpos(m.positions[Int(m.indices[i])]))
            if out[r] == nil { order.append(r); out[r] = Mesh3() }
            let base = UInt32(out[r]!.positions.count)
            for k in 0..<3 {
                let v = Int(m.indices[i + k])
                out[r]!.positions.append(m.positions[v]); out[r]!.normals.append(m.normals[v]); out[r]!.uvs.append(m.uvs[v])
            }
            out[r]!.indices.append(contentsOf: [base, base + 1, base + 2])
            i += 3
        }
        return order.map { out[$0]! }
    }
}
