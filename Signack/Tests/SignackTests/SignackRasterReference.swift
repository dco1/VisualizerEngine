import simd
import SignackGPU

/// A CPU reference of `SignackRaster.metal`'s coverage rule, written from the rule's statement
/// (1/256 snap, pixel-centre samples, exact edge functions, top-left ties, CCW-in-clip = front) —
/// not from the shader — so the GPU kernel is checked against an independent reading of it.
enum SignackRasterReference {
    struct Sample: Equatable { var triangle: Int; var front: Bool; var depth: Float }

    /// Nearest covering triangle per pixel (row-major), nil where none covers the centre.
    static func rasterize(_ tris: [SignackRasterTriangle], width: Int, height: Int) -> [Sample?] {
        var out = [Sample?](repeating: nil, count: width * height)
        for (ti, t) in tris.enumerated() {
            func snap(_ x: Float, _ y: Float) -> SIMD2<Int64> { SIMD2(Int64((x * 256).rounded(.toNearestOrEven)), Int64((y * 256).rounded(.toNearestOrEven))) }
            var v = [snap(t.v0x, t.v0y), snap(t.v1x, t.v1y), snap(t.v2x, t.v2y)]
            var z = [t.v0z, t.v1z, t.v2z]
            func edge(_ a: SIMD2<Int64>, _ b: SIMD2<Int64>, _ p: SIMD2<Int64>) -> Int64 {
                (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
            }
            var area = edge(v[0], v[1], v[2])
            if area == 0 { continue }
            let front = area < 0
            if t.idX & SignackRasterTriangle.cullBackFaces != 0 && !front { continue }
            if area < 0 { v.swapAt(1, 2); z.swapAt(1, 2); area = -area }
            // A sample on an edge belongs to the triangle only across a top or a left edge.
            func owns(_ a: SIMD2<Int64>, _ b: SIMD2<Int64>) -> Bool { let d = b &- a; return (d.y == 0 && d.x > 0) || d.y < 0 }
            let xs = v.map(\.x), ys = v.map(\.y)
            let pxLo = max(0, Int(((Double(xs.min()!) - 128) / 256).rounded(.up))), pxHi = min(width - 1, Int(((Double(xs.max()!) - 128) / 256).rounded(.down)))
            let pyLo = max(0, Int(((Double(ys.min()!) - 128) / 256).rounded(.up))), pyHi = min(height - 1, Int(((Double(ys.max()!) - 128) / 256).rounded(.down)))
            guard pxLo <= pxHi, pyLo <= pyHi else { continue }
            for py in pyLo...pyHi {
                for px in pxLo...pxHi {
                    let p = SIMD2<Int64>(Int64(px) * 256 + 128, Int64(py) * 256 + 128)
                    let w = [edge(v[1], v[2], p), edge(v[2], v[0], p), edge(v[0], v[1], p)]
                    let own = [owns(v[1], v[2]), owns(v[2], v[0]), owns(v[0], v[1])]
                    guard (0..<3).allSatisfy({ w[$0] > 0 || (w[$0] == 0 && own[$0]) }) else { continue }
                    let inv = 1 / Float(area)
                    let l1 = Float(w[1]) * inv, l2 = Float(w[2]) * inv
                    let d = z[0].addingProduct(l1, z[1] - z[0]).addingProduct(l2, z[2] - z[0])   // the shader's two fma
                    guard d >= 0, d <= 1 else { continue }
                    let i = py * width + px
                    if let o = out[i], o.depth <= d { continue }
                    out[i] = Sample(triangle: ti, front: front, depth: d)
                }
            }
        }
        return out
    }
}
