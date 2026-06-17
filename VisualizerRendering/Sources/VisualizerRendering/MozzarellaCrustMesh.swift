import Foundation
import simd

// ── MozzarellaCrustMesh ──────────────────────────────────────────────────────
//
// The breaded SHELL of a fried mozzarella stick — one HALF at a time. A stick is
// two of these placed nose-to-nose at a shared "break plane"; the two halves pull
// apart and the molten cheese (the MLS-MPM core, NOT this mesh) strings between
// the freshly-torn faces.
//
// Geometry (one half, local space)
// ────────────────────────────────
//   • Long axis is local +X. The TORN FACE sits at x = 0 (a jagged, near-flat cap
//     the cheese erupts from); the body runs +X to a ROUNDED OUTER TIP.
//   • Cross-section is a SUPERELLIPSE in the YZ plane (exponent ~2.6) so the stick
//     reads as a rounded-rectangular log, not a perfect cylinder — matching the
//     reference photos where the breading holds a squarish-round profile.
//   • The crumb is real geometry: every surface vertex is pushed out along its
//     normal by layered value-noise so the silhouette is lumpy panko, not a smooth
//     capsule. A flat normal map alone is the "CG plastic" giveaway this avoids.
//
// The mesh is a CLOSED, opaque solid (torn-cap fan + body rings + tip pole), so —
// like FrankMesh — it is emitted DOUBLE-SIDED: back faces are always depth-occluded
// by front faces, which sidesteps the inside-out-winding invisibility trap while
// keeping the outward normals shading correctly. See
// docs/known-issues/scn_geometry_winding.md.
//
// plausibility: real — the crust is honest swept-superellipse geometry with
// value-noise crumb displacement and a fan-capped torn face; no decals, no
// billboard shortcuts. The cheese it pairs with is the real MLS-MPM core.

public enum MozzarellaCrustMesh {

    /// Build one breaded half-stick. Returns a `CoinMesh.Mesh` (positions /
    /// normals / uvs / indices) ready to wrap in an `SCNGeometry`.
    ///
    /// - Parameters:
    ///   - halfLength: tip-to-break length of THIS half along +X.
    ///   - radiusY/radiusZ: superellipse half-extents of the body cross-section.
    ///   - superExponent: superellipse exponent (2 = ellipse, larger = squarer).
    ///   - crumb: panko bump amplitude (fraction of mean radius).
    ///   - axialSegments/radialSegments: tessellation.
    ///   - tipRings: rings used to round the outer tip.
    ///   - tornJag: in/out jitter amplitude of the torn-face rim (fraction of radius).
    ///   - seed: per-instance variation seed (crumb pattern + torn rim + tip jitter).
    public static func makeHalf(halfLength: Float = 0.55,
                                radiusY: Float = 0.16,
                                radiusZ: Float = 0.17,
                                superExponent: Float = 2.6,
                                crumb: Float = 0.10,
                                axialSegments: Int = 36,
                                radialSegments: Int = 40,
                                tipRings: Int = 7,
                                tornJag: Float = 0.16,
                                seed: UInt64 = 0xC0FFEE) -> CoinMesh.Mesh {

        let radial = max(12, radialSegments)
        let axial  = max(8, axialSegments)
        let caps   = max(3, tipRings)
        let meanR  = (radiusY + radiusZ) * 0.5

        var pos: [SIMD3<Float>] = []
        var uv:  [SIMD2<Float>] = []
        var idx: [Int32] = []

        var rng = Hash(seed)

        // Superellipse unit point for angle θ (signed-power keeps it convex).
        @inline(__always) func superUnit(_ theta: Float) -> SIMD2<Float> {
            let c = cos(theta), s = sin(theta)
            let n = 2.0 / superExponent
            let x = sign(c) * pow(abs(c), n)
            let y = sign(s) * pow(abs(s), n)
            return SIMD2(x, y)   // (.x → Y axis, .y → Z axis)
        }

        // Panko crumb: layered value-noise sampled in the stick's own space so the
        // bump pattern travels with the geometry. Returns a small +/- displacement.
        @inline(__always) func crumbDisp(_ p: SIMD3<Float>) -> Float {
            let f1 = valueNoise(p * 11.0 + SIMD3(repeating: Float(seed & 0xFF) * 0.013))
            let f2 = valueNoise(p * 26.0 + SIMD3(7.3, 2.1, 5.7))
            let f3 = valueNoise(p * 53.0 + SIMD3(1.7, 9.4, 3.2))
            // Bias toward bumps OUT (panko sits proud), with finer speckle on top.
            let n = f1 * 0.62 + f2 * 0.28 + f3 * 0.10
            return (n - 0.46) * 2.0
        }

        // Axial profile: x position and a radial scale that rounds the OUTER tip.
        // Body runs 0 → bodyLen at scale 1; the last `caps` rings sweep a quarter
        // circle from scale 1 down to ~0 at the pole, advancing x by the tip bulge.
        let tipBulge = meanR * 0.92
        let bodyLen  = max(0.02, halfLength - tipBulge)

        struct Ring { var x: Float; var scale: Float; var v: Float; var slopeX: Float }
        var ringSpecs: [Ring] = []

        // Body rings (scale 1). A subtle waist near the torn face so the break
        // doesn't read as a clean machined cut.
        for i in 0...axial {
            let t = Float(i) / Float(axial)               // 0 at torn face → 1 at tip start
            let x = bodyLen * t
            // gentle barrelling: fattest in the middle of the body
            let barrel = 1.0 + 0.05 * sin(t * .pi)
            ringSpecs.append(Ring(x: x, scale: barrel, v: t * 0.5, slopeX: 0))
        }
        // Tip rings (quarter circle).
        for k in 1...caps {
            let a = (Float.pi / 2) * Float(k) / Float(caps)   // 0 → π/2
            let scale = cos(a)
            let x = bodyLen + tipBulge * sin(a)
            ringSpecs.append(Ring(x: x, scale: scale, v: 0.5 + 0.5 * Float(k) / Float(caps),
                                  slopeX: sin(a)))
        }

        // Emit rings.
        var rings: [[Int32]] = []
        for (ri, spec) in ringSpecs.enumerated() {
            // Per-ring radius wobble so no two cross-sections are identical.
            let wob = 1.0 + (rng.unitF() - 0.5) * 0.05
            var ringIdx: [Int32] = []
            ringIdx.reserveCapacity(radial)
            let isTorn = (ri == 0)
            for j in 0..<radial {
                let theta = Float(j) / Float(radial) * 2 * .pi
                let su = superUnit(theta)
                var ry = radiusY * spec.scale * wob
                var rz = radiusZ * spec.scale * wob
                // Torn rim: rough, ragged edge — pull random spots in/out.
                if isTorn {
                    let jag = 1.0 + (valueNoise(SIMD3(theta * 2.0, 0, Float(seed & 0x3F))) - 0.5) * 2.0 * tornJag
                    ry *= jag; rz *= jag
                }
                let baseY = ry * su.x
                let baseZ = rz * su.y
                var p = SIMD3<Float>(spec.x, baseY, baseZ)
                // Outward radial direction in the YZ plane for crumb push.
                let radialDir = simd_length(SIMD2(baseY, baseZ)) > 1e-5
                    ? simd_normalize(SIMD3<Float>(0, baseY, baseZ))
                    : SIMD3<Float>(0, 1, 0)
                let bump = crumbDisp(p) * crumb * meanR * (isTorn ? 0.4 : 1.0)
                p += radialDir * bump
                pos.append(p)
                uv.append(SIMD2(Float(j) / Float(radial), spec.v))
                ringIdx.append(Int32(pos.count - 1))
            }
            rings.append(ringIdx)
        }

        // Outer tip pole.
        let tipPole = Int32(pos.count)
        pos.append(SIMD3(halfLength + tipBulge * 0.02, 0, 0))
        uv.append(SIMD2(0.5, 1.0))

        // Torn-face centre (slightly recessed + jittered → the cheese cavity).
        let tornCentre = Int32(pos.count)
        pos.append(SIMD3(meanR * 0.10 * rng.unitF(), 0, 0))
        uv.append(SIMD2(0.5, 0.0))

        // ── Triangles ────────────────────────────────────────────────────────
        // Body / tip ring strips (outward winding seen from +radius).
        for k in 0..<(rings.count - 1) {
            let a = rings[k], b = rings[k + 1]
            for j in 0..<radial {
                let jn = (j + 1) % radial
                idx += [a[j], b[j], a[jn]]
                idx += [a[jn], b[j], b[jn]]
            }
        }
        // Tip pole fan.
        let last = rings[rings.count - 1]
        for j in 0..<radial {
            let jn = (j + 1) % radial
            idx += [tipPole, last[jn], last[j]]
        }
        // Torn-face fan (faces toward -X).
        let first = rings[0]
        for j in 0..<radial {
            let jn = (j + 1) % radial
            idx += [tornCentre, first[j], first[jn]]
        }

        // Smooth per-vertex normals by face-normal accumulation (handles the
        // noise displacement correctly — analytic normals would ignore the crumb).
        var nrm = [SIMD3<Float>](repeating: .zero, count: pos.count)
        var t = 0
        while t < idx.count {
            let i0 = Int(idx[t]), i1 = Int(idx[t + 1]), i2 = Int(idx[t + 2])
            let e1 = pos[i1] - pos[i0]
            let e2 = pos[i2] - pos[i0]
            let fn = simd_cross(e1, e2)
            nrm[i0] += fn; nrm[i1] += fn; nrm[i2] += fn
            t += 3
        }
        for i in 0..<nrm.count {
            nrm[i] = simd_length(nrm[i]) > 1e-6 ? simd_normalize(nrm[i]) : SIMD3(1, 0, 0)
        }

        // Double-side: closed opaque solid → back faces never seen; emitting both
        // windings makes the mesh winding-agnostic. winding-ok: closed solid,
        // outward smooth normals keep visible front faces correct (cf. FrankMesh).
        let single = idx
        for s in stride(from: 0, to: single.count, by: 3) {
            idx += [single[s], single[s + 2], single[s + 1]]
        }

        return CoinMesh.Mesh(positions: pos, normals: nrm, uvs: uv, indices: idx)
    }

    // ── Value noise ────────────────────────────────────────────────────────
    // Trilinearly-interpolated hash lattice noise in [0,1]. Cheap, deterministic,
    // tileable enough for crumb bumps. Not Perlin — we only need lumpiness.
    private static func valueNoise(_ p: SIMD3<Float>) -> Float {
        let pi = floor(p)
        let pf = p - pi
        let w = pf * pf * (3 - 2 * pf)   // smoothstep
        @inline(__always) func h(_ c: SIMD3<Float>) -> Float {
            var n = c.x * 127.1 + c.y * 311.7 + c.z * 74.7
            n = sin(n) * 43758.5453
            return n - floor(n)
        }
        let c000 = h(pi + SIMD3(0,0,0)), c100 = h(pi + SIMD3(1,0,0))
        let c010 = h(pi + SIMD3(0,1,0)), c110 = h(pi + SIMD3(1,1,0))
        let c001 = h(pi + SIMD3(0,0,1)), c101 = h(pi + SIMD3(1,0,1))
        let c011 = h(pi + SIMD3(0,1,1)), c111 = h(pi + SIMD3(1,1,1))
        let x00 = mix(c000, c100, w.x), x10 = mix(c010, c110, w.x)
        let x01 = mix(c001, c101, w.x), x11 = mix(c011, c111, w.x)
        let y0 = mix(x00, x10, w.y), y1 = mix(x01, x11, w.y)
        return mix(y0, y1, w.z)
    }

    @inline(__always) private static func mix(_ a: Float, _ b: Float, _ t: Float) -> Float {
        a + (b - a) * t
    }

    // Tiny deterministic hash RNG for per-instance jitter.
    private struct Hash {
        var s: UInt64
        init(_ seed: UInt64) { s = seed == 0 ? 0x9E3779B97F4A7C15 : seed }
        mutating func next() -> UInt64 {
            s &+= 0x9E3779B97F4A7C15
            var z = s
            z = (z ^ (z &>> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z &>> 27)) &* 0x94D049BB133111EB
            return z ^ (z &>> 31)
        }
        mutating func unitF() -> Float { Float(next() >> 40) / Float(1 << 24) }
    }
}
