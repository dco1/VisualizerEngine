import simd

// Float / SIMD counterparts of the Double scalar helpers in MaterialChannels.swift
// (`clamp01`, `smoothstep`, `mix`). Float-based code (the vegetation baker, the app's
// render bridge) used to re-declare these privately; one copy each now lives here.
// Same semantics as the Double versions: `smoothstep` is the Hermite step and tolerates a
// degenerate edge pair; `mix` does NOT clamp `t` (use `mixClamped` when `t` may leave 0…1).

@inlinable public func clamp01(_ v: Float) -> Float { max(0, min(1, v)) }

/// Hermite step from 0→1 as `t` crosses `[edge0, edge1]`. A reversed edge pair gives 1→0.
@inlinable public func smoothstep(_ edge0: Float, _ edge1: Float, _ t: Float) -> Float {
    if edge1 == edge0 { return t < edge0 ? 0 : 1 }
    let x = clamp01((t - edge0) / (edge1 - edge0))
    return x * x * (3 - 2 * x)
}

@inlinable public func mix(_ a: Float, _ b: Float, _ t: Float) -> Float { a + (b - a) * t }
@inlinable public func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> { a + (b - a) * t }

/// `mix` with `t` pinned to 0…1 first.
@inlinable public func mixClamped(_ a: Float, _ b: Float, _ t: Float) -> Float { mix(a, b, clamp01(t)) }
@inlinable public func mixClamped(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> { mix(a, b, clamp01(t)) }
@inlinable public func mixClamped(_ a: Double, _ b: Double, _ t: Double) -> Double { mix(a, b, clamp01(t)) }
@inlinable public func mixClamped(_ a: Vec3, _ b: Vec3, _ t: Double) -> Vec3 { mix(a, b, clamp01(t)) }
