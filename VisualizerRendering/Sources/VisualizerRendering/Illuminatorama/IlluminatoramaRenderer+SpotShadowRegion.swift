import Metal
import simd

// SPOT/AREA-SLICE REGION RE-RENDERS (Daydream DH-1674)
// ====================================================
//
// DH-1590 lets a spot or window-portal slice whose light frustum no changed caster reaches keep its
// map. A slice a change DOES reach used to re-rasterise whole — on a wall drag in 4000 Sunset that
// was every window portal, every frame (~5 ms of GPU, ~2 ms of encode), because the dragged walls
// stand inside every portal's frustum.
//
// The sun cascades already answer this (DH-1668, `+SunCascadeReuse.swift`): re-rasterise only the
// texel rectangle the change projects to. The argument carries over unchanged. The slice's depth is
// the minimum over the triangles covering each texel; the recorded change holds the world bounds of
// every changed caster's OLD and NEW pose (or, for a swapped mesh, of the triangles that differ); so
// outside the rectangle those bounds project to no texel can have changed. The pass loads the slice,
// clears the rectangle to the far plane under a scissor, and redraws every caster whose bounds reach
// the rectangle's own frustum (`scissorFrustum`). A box that crosses the light's eye plane cannot be
// projected and re-renders the slice whole, as does a rectangle past `cascadeRegionMaxFraction`.
//
// Only a slice that passed every other reuse test (its matrix held, no live GPU geometry or sway can
// reach it) takes this path; everything else re-renders whole, as before.
//
// A/B lever: `spotShadowRegionEnabled` (env `VIZ_ILLUMI_NO_SPOT_REGION=1`).
extension IlluminatoramaRenderer {

    /// What one slice a bounded change (or a bounded swing) reaches does this frame: the rectangle to
    /// re-rasterise (nil = the whole slice) and a verdict for `spotShadowLastVerdicts`. `swaying` = the
    /// swing spheres reaching the slice, which move every frame and are dirty every frame — exactly
    /// how the sun cascades treat them (`liveShadowCasterBounds`).
    func spotSliceRegion(_ vp: simd_float4x4,
                         swaying: [IlluminatoramaCullBounds]) -> (rect: MTLScissorRect?, verdict: String) {
        guard spotShadowRegionEnabled else { return (nil, "whole: lever") }
        guard shd.shadowRegionClearPipeline != nil, !shd.pendingCasterChangeUnbounded,
              let atlas = shd.spotShadowAtlas else { return (nil, "whole: caster-unbounded") }
        let clip = IlluminatoramaClipVolume(vp)
        let dirty = shd.pendingCasterChange.filter { !clip.excludes($0) } + swaying
        guard !dirty.isEmpty else { return (nil, "whole: caster") }   // the caller saw a touch; be safe
        guard let rect = Self.nearClippedTexelRect(of: dirty, vp: vp, width: atlas.width, height: atlas.height) else {
            return (nil, "whole: unprojectable")
        }
        let fraction = Double(rect.width * rect.height) / Double(max(1, atlas.width * atlas.height))
        guard fraction <= Self.cascadeRegionMaxFraction else {
            return (nil, String(format: "whole: wide %.0f%%", fraction * 100))
        }
        return (rect, String(format: "region %.1f%%", fraction * 100))
    }

    /// `texelRect` for a PERSPECTIVE slice whose boxes may straddle the light's eye plane. A window
    /// portal's eye sits in its own window, so the wall being dragged — the one the window is in —
    /// always crosses it, and the plain projection gives up. The depth clip mode is `.clip`, so only
    /// the part of a box on the far side of the near plane (clip z ≥ 0) can write a texel: each box is
    /// cut there first (its corners in front, plus every edge's crossing point), and only that part
    /// is projected. A box wholly behind the near plane writes nothing and adds nothing. nil = no
    /// box reaches the slice, or a point the projection cannot place.
    static func nearClippedTexelRect(of boxes: [IlluminatoramaCullBounds], vp: simd_float4x4,
                                     width: Int, height: Int) -> MTLScissorRect? {
        guard width > 0, height > 0 else { return nil }
        var lo = SIMD2<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        var any = false
        for b in boxes {
            var q = [SIMD4<Float>](repeating: .zero, count: 8)
            for i in 0..<8 {
                let p = SIMD3<Float>((i & 1) == 0 ? b.lo.x : b.hi.x, (i & 2) == 0 ? b.lo.y : b.hi.y,
                                     (i & 4) == 0 ? b.lo.z : b.hi.z)
                q[i] = vp * SIMD4<Float>(p, 1)
            }
            var pts: [SIMD4<Float>] = q.filter { $0.z >= 0 }
            for i in 0..<8 {
                for axis in [1, 2, 4] where i & axis == 0 {
                    let a = q[i], c = q[i | axis]
                    guard (a.z >= 0) != (c.z >= 0) else { continue }
                    let t = a.z / (a.z - c.z)
                    pts.append(a + (c - a) * t)
                }
            }
            for p in pts {
                guard p.w > 1e-6 else { return nil }
                let ndc = SIMD2<Float>(p.x, p.y) / p.w
                lo = simd_min(lo, ndc); hi = simd_max(hi, ndc); any = true
            }
        }
        guard any else { return nil }
        lo = simd_clamp(lo, SIMD2(repeating: -1), SIMD2(repeating: 1))
        hi = simd_clamp(hi, SIMD2(repeating: -1), SIMD2(repeating: 1))
        guard lo.x < hi.x, lo.y < hi.y else { return nil }
        let w = Float(width), h = Float(height)
        let x0 = max(0, Int(((lo.x + 1) * 0.5 * w).rounded(.down)) - 1)
        let x1 = min(width, Int(((hi.x + 1) * 0.5 * w).rounded(.up)) + 1)
        let y0 = max(0, Int(((1 - hi.y) * 0.5 * h).rounded(.down)) - 1)
        let y1 = min(height, Int(((1 - lo.y) * 0.5 * h).rounded(.up)) + 1)
        guard x1 > x0, y1 > y0 else { return nil }
        return MTLScissorRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    }

    /// World bounds of every swing sphere whose box reaches the clip volume of `vp`, or nil when a
    /// swing cannot be bounded (an infinite sphere).
    func swayBounds(reaching vp: simd_float4x4) -> [IlluminatoramaCullBounds]? {
        guard !shd.swayingSpheres.isEmpty else { return [] }
        let clip = IlluminatoramaClipVolume(vp)
        var out: [IlluminatoramaCullBounds] = []
        for s in shd.swayingSpheres {
            if s.radius == .infinity { return nil }
            var b = IlluminatoramaCullBounds.empty
            b.formUnion(center: s.center, radius: s.radius)
            if !clip.excludes(b) { out.append(b) }
        }
        return out
    }

    /// The LOAD-action pass descriptor for `slice` (cached; rebuilt with the atlas).
    func spotShadowRegionPassDesc(_ slice: Int) -> MTLRenderPassDescriptor {
        while shd.spotShadowRegionPassDescs.count <= slice {
            let pass = MTLRenderPassDescriptor()
            pass.depthAttachment.texture = shd.spotShadowAtlas
            pass.depthAttachment.slice = shd.spotShadowRegionPassDescs.count
            pass.depthAttachment.loadAction = .load
            pass.depthAttachment.storeAction = .store
            shd.spotShadowRegionPassDescs.append(pass)
        }
        return shd.spotShadowRegionPassDescs[slice]
    }

    /// Per atlas slice: what the last spot-shadow encode did with it — `""` reused, `"region N%"`
    /// re-rasterised only the rectangle a change reached, otherwise `"whole: <why>"` (matrix, caster,
    /// wide, unprojectable, gpu, sway, lever). Diagnostic.
    public var spotShadowLastVerdicts: [String] { shd.lastSpotVerdicts }
}
