// plausibility: real — a CPU mirror of the tonemap's AgX display transform
// (IlluminatoramaTonemap.metal `agx` / `agxWithoutLook`) and its inverse on a grey, so a host can
// ask "what exposed level prints at code N" instead of guessing through the toe.
import Foundation
import simd

/// The display transform's inverse, tabulated by bisection on a grey input.
///
/// Why a host needs it: AgX's 'punchy' look is a ~2.3-power toe (4 stops under mid-grey prints at
/// sRGB 4, 5 stops at 1). An operator that places a dark region's level BEFORE the display
/// transform (local tone mapping) has to aim at a level the display still prints — a target set
/// in exposed units without the inverse lands inside the toe and prints black whatever its slope.
/// `IlluminatoramaLocalToneMapping.printFloorLevel` is meant to come from here.
public enum IlluminatoramaDisplayInverse {

    private static let inset = simd_float3x3(rows: [
        SIMD3(0.8424790622530940, 0.0784335999999992, 0.0792237451477643),
        SIMD3(0.0423282422610123, 0.8784686364697720, 0.0791661274605434),
        SIMD3(0.0423756549057051, 0.0784336000000000, 0.8791429737931040)])
    private static let outset = simd_float3x3(rows: [
        SIMD3(1.1968790051201738, -0.0980208811401368, -0.0990297440797205),
        SIMD3(-0.0528968517574562, 1.1519031299041727, -0.0989611768448433),
        SIMD3(-0.0529716355144438, -0.0980434501171241, 1.1510736726411610)])
    private static let luma = SIMD3<Float>(0.2126, 0.7152, 0.0722)

    /// AgX on linear exposed RGB → linear display RGB (the shader's `agx` with `look`, else
    /// `agxWithoutLook`).
    public static func agx(_ x: SIMD3<Float>, look: Bool = true) -> SIMD3<Float> {
        let minEv: Float = -12.47393, maxEv: Float = 4.026069
        var v = inset * simd_max(x, .zero)
        v = SIMD3(log2(max(v.x, 1e-10)), log2(max(v.y, 1e-10)), log2(max(v.z, 1e-10)))
        v = simd_clamp(v, SIMD3(repeating: minEv), SIMD3(repeating: maxEv))
        v = simd_clamp((v - minEv) / (maxEv - minEv), .zero, SIMD3(repeating: 1))
        let x2 = v * v, x4 = x2 * x2
        v = 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v + 0.4298 * x2 + 0.1191 * v
            - SIMD3(repeating: 0.00232)
        if look {
            v = SIMD3(pow(max(v.x, 0), 1.35), pow(max(v.y, 0), 1.35), pow(max(v.z, 0), 1.35))
            let l = simd_dot(v, luma)
            v = SIMD3(repeating: l) + 1.4 * (v - SIMD3(repeating: l))
        }
        v = outset * v
        let o = SIMD3(pow(max(v.x, 0), 2.2), pow(max(v.y, 0), 2.2), pow(max(v.z, 0), 2.2))
        return simd_clamp(o, .zero, SIMD3(repeating: 1))
    }

    /// The sRGB OETF (0…1 → 0…1).
    public static func srgbEncode(_ c: Float) -> Float {
        let v = min(max(c, 0), 1)
        return v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1 / 2.4) - 0.055
    }

    /// The 8-bit code (0…255, fractional) a GREY at exposed brightness `x` prints at through AgX
    /// (`look`: the punchy look, else the bare sigmoid) and the tone curve's shadow lift
    /// (`shadows`, the renderer's `shadows`: × mix(1, shadows, 1 − smoothstep(0, 0.5, luma))).
    public static func printedCode(exposed x: Float, look: Bool = true, shadows: Float = 1) -> Float {
        let m = agx(SIMD3(repeating: x), look: look)
        let l = simd_dot(m, luma)
        let t = min(max(l / 0.5, 0), 1)
        let sw = 1 - t * t * (3 - 2 * t)
        let lifted = min(1, m.y * (1 + (shadows - 1) * sw))
        return srgbEncode(lifted) * 255
    }

    /// The exposed brightness a grey needs to print at `code` (bisection in log2 over
    /// [2^−20, 2^6]; the print is monotone in exposure).
    public static func exposedLevel(printingAt code: Float, look: Bool = true, shadows: Float = 1) -> Float {
        var lo: Float = -20, hi: Float = 6
        for _ in 0..<48 {
            let mid = 0.5 * (lo + hi)
            if printedCode(exposed: exp2(mid), look: look, shadows: shadows) < code { lo = mid } else { hi = mid }
        }
        return exp2(0.5 * (lo + hi))
    }
}
