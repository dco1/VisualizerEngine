//  SculptKernel.swift — the per-stamp brush maths on the CPU (ticket item I): the oracle and fallback for the Metal kernels
//  in `Sculpt+GPU.swift`, which compute the same thing line for line (Float, same operation order; tests hold them within 1e-4).
//
//  One stamp is a `SculptParams`: a brush centre and radius, a kind, and the sample-index box it may touch. The box is
//  first copied to a scratch (so neighbours read the field as it was before the stamp), then each voxel of the box is
//  recomputed from the scratch (or, outside the box, the untouched field). Nothing outside the box is read for writing,
//  so a stamp costs O(radius³). Every kind is a resampling of the old field, v'(p) = v(q(p)), or a blend of it:
//    draw     q = p − n·a·w            the surface moves a·w along the stamp normal (add), or the opposite (subtract)
//    grab     q = p − d·w              the surface follows the drag d
//    smooth   v' = v + (mean of the 6 neighbours − v)·w
//    inflate  v' = v − a·w             offset along each local normal (a distance field offsets by subtraction)
//    crease   v' = v + a·w(t/0.45)     a narrow groove (subtract) or ridge (add)
//    flatten  v' = v + (plane − v)·w·band   toward the plane through the centre, only near the surface
//    pinch    q = p + (p − c)⊥·f·w     features slide toward the stroke axis (the normal)
//  with w = falloff(|p − c|/r) · (1 − mask). The amount a = strength · pressure · radius (metres).

import Foundation
import simd

/// Layout shared with `SculptParams` in the MSL: seven float4 / uint4.
struct SculptParams {
    var centerRadius = SIMD4<Float>(repeating: 0)      // centre xyz, radius
    var normalAmount = SIMD4<Float>(repeating: 0)      // unit normal xyz, signed amount a (metres)
    var drag = SIMD4<Float>(repeating: 0)              // drag xyz, factor (strength · pressure)
    var gridOrigin = SIMD4<Float>(repeating: 0)        // origin xyz, cell
    var lo = SIMD4<UInt32>(repeating: 0)               // box first sample xyz, kind
    var size = SIMD4<UInt32>(repeating: 0)             // box sample counts xyz, falloff kind
    var dims = SIMD4<UInt32>(repeating: 0)             // grid samples xyz, hasMask
}

public enum SculptFalloff: UInt32, Sendable {
    case smooth = 0, linear = 1, sharp = 2, constant = 3, spherical = 4, cubic = 5
}

enum SculptKernel {
    static let draw: UInt32 = 0, grab: UInt32 = 1, smooth: UInt32 = 2, inflate: UInt32 = 3
    static let crease: UInt32 = 4, flatten: UInt32 = 5, pinch: UInt32 = 6

    static func falloff(_ kind: UInt32, _ t: Float) -> Float {
        if t >= 1 { return 0 }
        switch kind {
        case 0: let s = 1 - t * t; return s * s
        case 1: return 1 - t
        case 2: let s = 1 - t; return s * s
        case 3: return 1
        case 4: return (1 - t * t).squareRoot()
        default: return 1 - (3 * t * t - 2 * t * t * t)
        }
    }

    @inline(__always)
    static func trilinear(_ g: SIMD3<Float>, _ n: SIMD3<Int>, _ read: (Int, Int, Int) -> Float) -> Float {
        let c = simd_clamp(g, SIMD3<Float>(repeating: 0), SIMD3(Float(n.x - 1), Float(n.y - 1), Float(n.z - 1)))
        let i = min(Int(c.x), n.x - 2), j = min(Int(c.y), n.y - 2), k = min(Int(c.z), n.z - 2)
        let f = c - SIMD3(Float(i), Float(j), Float(k))
        let x00 = read(i, j, k) * (1 - f.x) + read(i + 1, j, k) * f.x
        let x10 = read(i, j + 1, k) * (1 - f.x) + read(i + 1, j + 1, k) * f.x
        let x01 = read(i, j, k + 1) * (1 - f.x) + read(i + 1, j, k + 1) * f.x
        let x11 = read(i, j + 1, k + 1) * (1 - f.x) + read(i + 1, j + 1, k + 1) * f.x
        return (x00 * (1 - f.y) + x10 * f.y) * (1 - f.z) + (x01 * (1 - f.y) + x11 * f.y) * f.z
    }

    static func apply(_ P: SculptParams, field: UnsafeMutablePointer<Float>, mask: UnsafePointer<Float>?) {
        let nx = Int(P.dims.x), ny = Int(P.dims.y), nz = Int(P.dims.z)
        let n = SIMD3(nx, ny, nz)
        let lo = SIMD3(Int(P.lo.x), Int(P.lo.y), Int(P.lo.z)), sz = SIMD3(Int(P.size.x), Int(P.size.y), Int(P.size.z))
        var scratch = [Float](repeating: 0, count: sz.x * sz.y * sz.z)
        for k in 0..<sz.z { for j in 0..<sz.y { for i in 0..<sz.x {
            scratch[(k * sz.y + j) * sz.x + i] = field[((lo.z + k) * ny + lo.y + j) * nx + lo.x + i]
        } } }
        let read: (Int, Int, Int) -> Float = { i, j, k in
            let l = SIMD3(i - lo.x, j - lo.y, k - lo.z)
            if l.x >= 0 && l.y >= 0 && l.z >= 0 && l.x < sz.x && l.y < sz.y && l.z < sz.z {
                return scratch[(l.z * sz.y + l.y) * sz.x + l.x]
            }
            return field[(k * ny + j) * nx + i]
        }
        let origin = SIMD3(P.gridOrigin.x, P.gridOrigin.y, P.gridOrigin.z), cell = P.gridOrigin.w
        let c = SIMD3(P.centerRadius.x, P.centerRadius.y, P.centerRadius.z), r = P.centerRadius.w
        let nrm = SIMD3(P.normalAmount.x, P.normalAmount.y, P.normalAmount.z), amt = P.normalAmount.w
        let drag = SIMD3(P.drag.x, P.drag.y, P.drag.z), factor = P.drag.w
        let kind = P.lo.w, fk = P.size.w
        for k in 0..<sz.z { for j in 0..<sz.y { for i in 0..<sz.x {
            let gi = SIMD3(lo.x + i, lo.y + j, lo.z + k)
            let idx = (gi.z * ny + gi.y) * nx + gi.x
            let p = origin + SIMD3(Float(gi.x), Float(gi.y), Float(gi.z)) * cell
            let d = p - c
            let t = (d.x * d.x + d.y * d.y + d.z * d.z).squareRoot() / r
            var w = falloff(fk, t)
            if w <= 0 { continue }
            if P.dims.w != 0, let mask { w *= 1 - mask[idx] }
            if w <= 0 { continue }
            let v = scratch[(k * sz.y + j) * sz.x + i]
            let tri: (SIMD3<Float>) -> Float = { q in trilinear((q - origin) / cell, n, read) }
            var out = v
            switch kind {
            case draw: out = tri(p - nrm * (amt * w))
            case grab: out = tri(p - drag * (w * factor))
            case smooth:
                let avg = (read(max(gi.x - 1, 0), gi.y, gi.z) + read(min(gi.x + 1, nx - 1), gi.y, gi.z)
                           + read(gi.x, max(gi.y - 1, 0), gi.z) + read(gi.x, min(gi.y + 1, ny - 1), gi.z)
                           + read(gi.x, gi.y, max(gi.z - 1, 0)) + read(gi.x, gi.y, min(gi.z + 1, nz - 1))) / 6
                out = v + (avg - v) * min(max(w * factor, 0), 1)
            case inflate: out = v - amt * w
            case crease:
                var wn = falloff(fk, min(t / 0.45, 1))
                if P.dims.w != 0, let mask { wn *= 1 - mask[idx] }
                out = v + amt * wn
            case flatten:
                let plane = d.x * nrm.x + d.y * nrm.y + d.z * nrm.z
                let band = min(max(1 - abs(v) / r, 0), 1)
                out = v + (plane - v) * min(max(w * factor, 0), 1) * band
            default:
                let along = d.x * nrm.x + d.y * nrm.y + d.z * nrm.z
                let perp = d - nrm * along
                out = tri(p + perp * (min(factor, 0.9) * w))
            }
            field[idx] = out
        } } }
    }

    /// 2× trilinear upsample: new sample (i, j, k) sits at old coordinate (i/2, j/2, k/2).
    static func upsample(from src: UnsafePointer<Float>, _ n: SIMD3<Int>, to dst: UnsafeMutablePointer<Float>, _ m: SIMD3<Int>) {
        for k in 0..<m.z { for j in 0..<m.y { for i in 0..<m.x {
            let g = SIMD3(Float(i), Float(j), Float(k)) * 0.5
            dst[(k * m.y + j) * m.x + i] = trilinear(g, n) { src[($2 * n.y + $1) * n.x + $0] }
        } } }
    }

    /// Project a mesh vertex onto the field and take its gradient normal (the twin of `sculpt_refine`).
    static func refine(position: SIMD3<Float>, cellMin lo: SIMD3<Float>, field: UnsafePointer<Float>, n: SIMD3<Int>,
                       origin: SIMD3<Float>, cell: Float, iterations: Int) -> (SIMD3<Float>, SIMD3<Float>) {
        let read: (Int, Int, Int) -> Float = { field[($2 * n.y + $1) * n.x + $0] }
        func f(_ p: SIMD3<Float>) -> Float { trilinear((p - origin) / cell, n, read) }
        func grad(_ p: SIMD3<Float>, _ h: Float) -> SIMD3<Float> {
            SIMD3(f(p + SIMD3(h, 0, 0)) - f(p - SIMD3(h, 0, 0)), f(p + SIMD3(0, h, 0)) - f(p - SIMD3(0, h, 0)),
                  f(p + SIMD3(0, 0, h)) - f(p - SIMD3(0, 0, h))) / (2 * h)
        }
        var p = position
        let hi = lo + SIMD3<Float>(repeating: cell), h = cell * 0.25
        for _ in 0..<iterations {
            let d = f(p), g = grad(p, h), g2 = simd_dot(g, g)
            if g2 > 1e-12 { p = simd_clamp(p - g * (d / g2), lo, hi) }
        }
        let g = grad(p, h), l = simd_length(g)
        return (p, l > 1e-6 ? g / l : SIMD3(0, 1, 0))
    }
}
