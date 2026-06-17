import Foundation
import simd

// ── FrankMesh ─────────────────────────────────────────────────────────────────
//
// Unit frankfurter (cocktail-sausage) geometry for the instanced renderer — a
// smooth capsule whose LONG axis is local +Y, so it maps onto the same oriented-
// cylinder collider the solver uses for coins/jewels: radius = sausage radius,
// halfThickness = capsule half-length. The Coin Pusher's "Frankfurter" drop asset
// is this mesh fed into the shared CoinDEMSolver alongside the coins + jewels.
//
// SMOOTH normals (sphere/cylinder revolution) — a cooked sausage reads as a
// glossy rounded tube, the opposite of the jewel's flat facets. Built to the
// `CoinMesh.Mesh` contract so it drops straight into `CoinInstancedRenderer`.
//
// Sized small (cocktail-frank scale) so it piles WITH the coins and jewels instead
// of dwarfing them; the reddish-brown cooked-frankfurter colour is set by the
// scene's instance material (matching Hot Dog+'s sausage base).

public enum FrankMesh {

    /// Build a unit frankfurter capsule.
    /// - `radius`: the sausage radius (the collider radius).
    /// - `halfLength`: half the TOTAL length tip-to-centre (the collider
    ///   halfThickness); the cylindrical mid-section is `halfLength − radius` long.
    /// - `bend`: small banana-style curve (fraction of length the mid bows in +Z),
    ///   so the franks don't read as dead-straight rods. 0 = straight.
    public static func make(radius: Float = 0.06,
                            halfLength: Float = 0.20,
                            radialSegments: Int = 24,
                            capRings: Int = 5,
                            bend: Float = 0.12) -> CoinMesh.Mesh {
        let seg = max(8, radialSegments)
        let caps = max(2, capRings)
        let R = radius
        let cylHalf = max(0.0, halfLength - R)   // half-length of the straight middle

        var pos: [SIMD3<Float>] = []
        var nrm: [SIMD3<Float>] = []
        var uv:  [SIMD2<Float>] = []
        var idx: [Int32] = []

        // Gentle bow: offset +Z by a parabola in y/halfLength so the sausage curves.
        func bowZ(_ y: Float) -> Float {
            let t = y / max(1e-4, halfLength)        // -1 … 1
            return bend * R * (1.0 - t * t)          // max at centre, 0 at tips
        }

        // One latitude ring at centre-height `yc` with profile radius `rr` and a
        // latitude `lat` (for the cap normals); `straight` marks the two cylinder
        // seam rings whose normal is purely radial.
        var rings: [[Int32]] = []
        func ring(yc: Float, rr: Float, lat: Float, vCoord: Float) {
            var r: [Int32] = []
            r.reserveCapacity(seg)
            let dz = bowZ(yc)
            for j in 0..<seg {
                let a = Float(j) / Float(seg) * 2 * .pi
                let ca = cos(a), sa = sin(a)
                let p = SIMD3<Float>(rr * ca, yc, dz + rr * sa)
                // Normal: sphere normal on the caps (lat≠0), radial on the cylinder.
                let n = simd_normalize(SIMD3<Float>(cos(lat) * ca, sin(lat), cos(lat) * sa))
                let i = Int32(pos.count)
                pos.append(p); nrm.append(n); uv.append(SIMD2(Float(j) / Float(seg), vCoord))
                r.append(i)
            }
            rings.append(r)
        }

        // Bottom pole.
        let botPole = Int32(pos.count)
        pos.append(SIMD3(0, -halfLength, bowZ(-halfLength)))
        nrm.append(SIMD3(0, -1, 0)); uv.append(SIMD2(0.5, 0))

        // Bottom cap: lat from −π/2 (pole) up to 0 (seam), excluding the pole.
        for k in 1...caps {
            let lat = -Float.pi / 2 + (Float.pi / 2) * Float(k) / Float(caps)
            let yc = -cylHalf + R * sin(lat)
            let rr = R * cos(lat)
            ring(yc: yc, rr: rr, lat: lat, vCoord: 0.08 * Float(k) / Float(caps))
        }
        // Top cap: lat from 0 (seam) to +π/2 (pole), excluding the pole.
        for k in 0..<caps {
            let lat = (Float.pi / 2) * Float(k) / Float(caps)
            let yc = cylHalf + R * sin(lat)
            let rr = R * cos(lat)
            ring(yc: yc, rr: rr, lat: lat, vCoord: 0.5 + 0.42 * Float(k) / Float(caps))
        }
        // Top pole.
        let topPole = Int32(pos.count)
        pos.append(SIMD3(0, halfLength, bowZ(halfLength)))
        nrm.append(SIMD3(0, 1, 0)); uv.append(SIMD2(0.5, 1))

        // Bottom pole fan → first ring. Winding: CW seen from below (outward).
        let first = rings[0]
        for j in 0..<seg {
            let jn = (j + 1) % seg
            idx += [botPole, first[j], first[jn]]
        }
        // Ring strips between consecutive rings.
        for k in 0..<(rings.count - 1) {
            let a = rings[k], b = rings[k + 1]
            for j in 0..<seg {
                let jn = (j + 1) % seg
                idx += [a[j], b[j], a[jn]]
                idx += [a[jn], b[j], b[jn]]
            }
        }
        // Top ring → top pole fan.
        let last = rings[rings.count - 1]
        for j in 0..<seg {
            let jn = (j + 1) % seg
            idx += [topPole, last[jn], last[j]]
        }

        // Double-side: the capsule is a closed convex solid, so back faces are always
        // depth-occluded by front faces and never seen — emitting both windings makes
        // the mesh winding-agnostic and sidesteps the inside-out invisibility trap
        // (the smooth outward normals keep the visible front faces shading correctly).
        let single = idx
        for t in stride(from: 0, to: single.count, by: 3) {
            idx += [single[t], single[t + 2], single[t + 1]]
        }

        return CoinMesh.Mesh(positions: pos, normals: nrm, uvs: uv, indices: idx)
    }
}
