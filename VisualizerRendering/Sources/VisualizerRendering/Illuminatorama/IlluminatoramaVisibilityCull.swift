import simd

// ── VISIBILITY CULLING (DH-0534, phase 1: CPU) ───────────────────────────────
//
// Every raster pass used to draw every mesh group — a 512² spot map over a kitchen counter
// rasterised the yard's trees, and a worst-case frame ran up to 36 full-scene geometry passes.
// Each pass now skips a group whose world bounds lie wholly outside the clip volume of the
// EXACT matrix that pass rasterises with (the jittered camera VP, a cascade's ortho VP with its
// `casterSlack` already folded in, a spot/area slice's perspective, a cube face's 90° frustum).
//
// It is a PURE SKIP, and that is the whole correctness argument: the clip volume is six linear
// half-spaces in world space, a group's bounds are convex, and every vertex the vertex stage can
// emit lies inside those bounds — so a group outside one half-space cannot produce a fragment,
// and removing its draw changes no pixel. The bounds therefore have to cover everything the
// vertex stage does to a position, not just the mesh at rest:
//
//   · `applyTreeWind` — a world-space offset bounded by the gust formula and the mesh's largest
//     packed sway / flutter weights (`IlluminatoramaMesh.windAttributeMax`).
//   · `applySway`     — a rigid rotation about a pivot by an arbitrary angle plus a vertical hop,
//     so the rotated sphere stays inside the sphere about the pivot that reaches its far side.
//   · GPU-written geometry — vertices a compute kernel rewrites (`gpuRepackTasks`, any mesh built
//     on caller-owned buffers, `IlluminatoramaMesh.cpuAuthoredVertices == false`) or instance
//     slots a host kernel overwrites (`onEncodeGPUInstances`) are invisible to the CPU, so those
//     groups are UNBOUNDED and always draw.

/// World-space axis-aligned bounds of one draw group, or "unbounded" (always draws).
struct IlluminatoramaCullBounds: Equatable {
    var lo: SIMD3<Float>
    var hi: SIMD3<Float>

    static let unbounded = IlluminatoramaCullBounds(lo: SIMD3(repeating: -.infinity),
                                                    hi: SIMD3(repeating: .infinity))
    static let empty = IlluminatoramaCullBounds(lo: SIMD3(repeating: .greatestFiniteMagnitude),
                                                hi: SIMD3(repeating: -.greatestFiniteMagnitude))

    var isBounded: Bool { lo.x.isFinite && lo.y.isFinite && lo.z.isFinite
                          && hi.x.isFinite && hi.y.isFinite && hi.z.isFinite }

    mutating func formUnion(center c: SIMD3<Float>, radius r: Float) {
        lo = simd_min(lo, c - SIMD3(repeating: r))
        hi = simd_max(hi, c + SIMD3(repeating: r))
    }

    /// World bounding sphere of ONE instance of a mesh whose object-space bounding sphere is
    /// `local`, grown by everything `illumi_vs` / `illumi_shadow_vs` can do to a vertex. nil when
    /// the result is not finite (a degenerate matrix) — the caller then treats the group as
    /// unbounded. `windMax` is asked for only when this instance actually receives wind.
    static func worldSphere(local: (center: SIMD3<Float>, radius: Float),
                            instance inst: IlluminatoramaInstance,
                            treeWindStrength: Float,
                            windMax: () -> (sway: Float, flutter: Float)?) -> (center: SIMD3<Float>, radius: Float)? {
        let m = inst.modelMatrix
        let c4 = m * SIMD4<Float>(local.center, 1)
        var center = SIMD3<Float>(c4.x, c4.y, c4.z)
        let scale = max(simd_length(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z)),
                        simd_length(SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z)),
                        simd_length(SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z)))
        var radius = local.radius * scale

        // applyTreeWind: no-op unless strength × windScale > 0. Horizontal bend ≤ sway·s·(0.5 + gust)
        // with gust ≤ 1, droop ≤ 0.18·sway·s, flutter ≤ 0.05 + 0.05 + 0.035 per unit weight.
        let windStrength = treeWindStrength * inst.windScale
        if windStrength > 0 {
            guard let w = windMax() else { return nil }
            radius += windStrength * (w.sway * (1.5 + 0.18) + w.flutter * (0.05 + 0.05 + 0.035))
        }

        // applySway: rotation about a pivot (box base for mode 1, model origin for mode 2) by any
        // angle, then a vertical jostle. Applied after the wind, so it carries the wind's reach.
        if inst.swayMode != 0 {
            let pivotY: Float = inst.swayMode == 2 ? 0 : -0.5
            let p4 = m * SIMD4<Float>(0, pivotY, 0, 1)
            let pivot = SIMD3<Float>(p4.x, p4.y, p4.z)
            radius = simd_length(center - pivot) + radius + abs(inst.swayJostle)
            center = pivot
        }

        guard center.x.isFinite, center.y.isFinite, center.z.isFinite, radius.isFinite else { return nil }
        // A margin far above float error in the matrix product (millimetres at 100 m) — the test
        // must never be the thing that clips a sliver of a real silhouette.
        return (center, radius * 1.001 + 0.002)
    }
}

/// The clip volume of one view-projection, as six world-space half-spaces (Metal clip space:
/// −w ≤ x ≤ w, −w ≤ y ≤ w, 0 ≤ z ≤ w). Valid for perspective AND orthographic matrices, and for
/// points behind a perspective eye: each inequality is linear in the world position, so a convex
/// set wholly outside one of them produces no fragment. The renderer never sets depth CLAMPING
/// (`depthClipMode` is left at `.clip`), so the near/far planes clip like the sides do.
struct IlluminatoramaClipVolume {
    private let planes: (SIMD4<Double>, SIMD4<Double>, SIMD4<Double>,
                         SIMD4<Double>, SIMD4<Double>, SIMD4<Double>)

    init(_ m: simd_float4x4) {
        func row(_ i: Int) -> SIMD4<Double> {
            SIMD4(Double(m.columns.0[i]), Double(m.columns.1[i]), Double(m.columns.2[i]), Double(m.columns.3[i]))
        }
        let r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3)
        planes = (r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2)
    }

    /// True when `b` lies entirely on the outside of any one clip plane. Unbounded never does.
    func excludes(_ b: IlluminatoramaCullBounds) -> Bool {
        guard b.isBounded else { return false }
        let lo = SIMD3<Double>(Double(b.lo.x), Double(b.lo.y), Double(b.lo.z))
        let hi = SIMD3<Double>(Double(b.hi.x), Double(b.hi.y), Double(b.hi.z))
        @inline(__always) func outside(_ p: SIMD4<Double>) -> Bool {
            // The box corner furthest along the plane normal — if even it is outside, all are.
            let v = SIMD3<Double>(p.x >= 0 ? hi.x : lo.x, p.y >= 0 ? hi.y : lo.y, p.z >= 0 ? hi.z : lo.z)
            return p.x * v.x + p.y * v.y + p.z * v.z + p.w < 0
        }
        return outside(planes.0) || outside(planes.1) || outside(planes.2)
            || outside(planes.3) || outside(planes.4) || outside(planes.5)
    }
}

/// Draw calls each raster pass issued and skipped, cumulative since the renderer was made (take
/// deltas). Groups, not instances: one group is one instanced draw. `trianglesCulled` is the
/// instance-expanded triangle count those skipped draws would have rasterised, over all passes.
public struct IlluminatoramaVisibilityCullStats: Equatable, Sendable {
    public var gbufferDrawn = 0, gbufferCulled = 0
    public var cascadeDrawn = 0, cascadeCulled = 0
    public var spotDrawn = 0, spotCulled = 0
    public var pointDrawn = 0, pointCulled = 0
    public var trianglesCulled = 0
    public init() {}
}
