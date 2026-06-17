import Foundation
import simd

// ── CoinMesh ──────────────────────────────────────────────────────────────────
//
// Unit coin geometry for the instanced renderer: a flat cylinder whose axis is
// local +Y (identity-oriented coins lie flat). Authored at TRUE coin dimensions
// (radius R, half-thickness h) because the solver's CoinTransform basis is a pure
// rotation — the mesh carries the size.
//
// The "TOKEN" stamp is baked as GEOMETRIC relief: each face is a displaced
// polar grid sampling the same height field CoinMaterial uses, so the emboss
// survives native Illuminatorama (which shades from geometry, not normal maps)
// and catches real highlights under RT. Normals come from the height gradient.

public enum CoinMesh {

    public struct Mesh {
        public var positions: [SIMD3<Float>]
        public var normals:   [SIMD3<Float>]
        public var uvs:       [SIMD2<Float>]
        public var indices:   [Int32]
    }

    /// Build a unit coin with the embossed faces.
    /// - `rings`/`segments`: face grid resolution (drives vertex count × instances).
    /// - `reliefDepth`: world-space height of the struck relief.
    /// - `text`: the struck legend (default "TOKEN").
    public static func make(radius R: Float,
                            halfThickness h: Float,
                            radialSegments segments: Int = 36,
                            rings: Int = 9,
                            reliefDepth: Float = 0.0,
                            text: String = "TOKEN",
                            reliefRes: Int = 128) -> Mesh {
        let seg = max(8, segments)
        let ringsN = max(2, rings)
        // Auto relief depth: ~35% of half-thickness reads as a struck coin.
        let depth = reliefDepth > 0 ? reliefDepth : h * 0.35

        // Height field (0…1) for the stamp; world R maps to UV radius 0.45 so the
        // bead ring sits near the rim and a flat margin meets the rim wall.
        let S = max(32, reliefRes)
        let hf = CoinMaterial.heightField(text: text, size: S)
        let uvR: Float = 0.45
        func sampleH(_ x: Float, _ z: Float) -> Float {
            // world (x,z) within the disk → uv → bilinear height-field sample.
            let u = 0.5 + uvR * (x / R)
            let v = 0.5 + uvR * (z / R)
            let fx = min(Float(S - 1), max(0, u * Float(S - 1)))
            let fz = min(Float(S - 1), max(0, v * Float(S - 1)))
            let x0 = Int(fx), z0 = Int(fz)
            let x1 = min(S - 1, x0 + 1), z1 = min(S - 1, z0 + 1)
            let tx = fx - Float(x0), tz = fz - Float(z0)
            let h00 = hf[z0 * S + x0], h10 = hf[z0 * S + x1]
            let h01 = hf[z1 * S + x0], h11 = hf[z1 * S + x1]
            return (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz
        }

        var pos: [SIMD3<Float>] = []
        var nrm: [SIMD3<Float>] = []
        var uv:  [SIMD2<Float>] = []
        var idx: [Int32] = []

        func push(_ p: SIMD3<Float>, _ n: SIMD3<Float>, _ u: SIMD2<Float>) -> Int32 {
            let i = Int32(pos.count); pos.append(p); nrm.append(n); uv.append(u); return i
        }

        let e = R * 0.02   // gradient sample step (world)

        // Build one displaced face. faceSign +1 = top (+Y), −1 = bottom (−Y).
        // Returns the outer ring's vertex indices (for the rim wall).
        func buildFace(faceSign: Float) -> [Int32] {
            let yBase = faceSign > 0 ? h : -h
            func vertex(_ x: Float, _ z: Float) -> Int32 {
                let hC = depth * sampleH(x, z)
                let y = yBase + faceSign * hC
                // Gradient (central diff) for the relief normal.
                let hxp = depth * sampleH(x + e, z), hxm = depth * sampleH(x - e, z)
                let hzp = depth * sampleH(x, z + e), hzm = depth * sampleH(x, z - e)
                let n = simd_normalize(SIMD3(-(hxp - hxm) / (2 * e),
                                             faceSign,
                                             -(hzp - hzm) / (2 * e)))
                let u = SIMD2<Float>(0.5 + uvR * (x / R), 0.5 + uvR * (z / R))
                return push(SIMD3(x, y, z), n, u)
            }
            // Centre + concentric rings.
            let center = vertex(0, 0)
            var rings2: [[Int32]] = []
            for k in 1...ringsN {
                let r = R * Float(k) / Float(ringsN)
                var ring: [Int32] = []
                ring.reserveCapacity(seg)
                for j in 0..<seg {
                    let a = Float(j) / Float(seg) * 2 * .pi
                    ring.append(vertex(r * cos(a), r * sin(a)))
                }
                rings2.append(ring)
            }
            // Centre fan (winding by face side: top is CW seen from +Y).
            let inner = rings2[0]
            for j in 0..<seg {
                let jn = (j + 1) % seg
                if faceSign > 0 { idx += [center, inner[jn], inner[j]] }
                else            { idx += [center, inner[j], inner[jn]] }
            }
            // Ring strips.
            for k in 0..<(ringsN - 1) {
                let a = rings2[k], b = rings2[k + 1]
                for j in 0..<seg {
                    let jn = (j + 1) % seg
                    if faceSign > 0 {
                        idx += [a[j], a[jn], b[j]]
                        idx += [a[jn], b[jn], b[j]]
                    } else {
                        idx += [a[j], b[j], a[jn]]
                        idx += [a[jn], b[j], b[jn]]
                    }
                }
            }
            return rings2[ringsN - 1]
        }

        let topOuter = buildFace(faceSign: 1)
        let botOuter = buildFace(faceSign: -1)

        // Rim wall — WELDED to the face edges. It must use the SAME radius (R) and the
        // SAME segment count (`seg`) as the faces' outer rings so its vertices are
        // COINCIDENT with `topOuter` / `botOuter`: that closes the surface into one
        // watertight solid. (The previous rim used a different radius — a reeded
        // R ± reedAmp — and 3× the segments, so nothing met the face edge: an open
        // ring-gap ran all the way around each face and the coin read as see-through /
        // not welded shut. The reeds were sub-0.001 units — invisible at coin scale;
        // a milled edge belongs in a normal map, not as a hole in the mesh.) The rim
        // verts are duplicates of the face-edge positions with OUTWARD normals — the
        // intended hard edge — so there is no gap, only a crease.
        var rimTop: [Int32] = [], rimBot: [Int32] = []
        for j in 0..<seg {
            let a = Float(j) / Float(seg) * 2 * .pi
            let nOut = SIMD3<Float>(cos(a), 0, sin(a))               // outward radial
            rimTop.append(push(SIMD3(R * cos(a),  h, R * sin(a)), nOut, SIMD2(Float(j)/Float(seg), 0.02)))
            rimBot.append(push(SIMD3(R * cos(a), -h, R * sin(a)), nOut, SIMD2(Float(j)/Float(seg), 0.02)))
        }
        for j in 0..<seg {
            let jn = (j + 1) % seg
            idx += [rimTop[j], rimTop[jn], rimBot[j]]
            idx += [rimTop[jn], rimBot[jn], rimBot[j]]
        }
        _ = topOuter; _ = botOuter   // face edges; rim verts are coincident with these

        return Mesh(positions: pos, normals: nrm, uvs: uv, indices: idx)
    }
}
