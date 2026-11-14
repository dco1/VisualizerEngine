import Metal
import simd

// SUN-CASCADE REUSE (Daydream DH-1668)
// ===================================
//
// Every sun cascade used to re-rasterise the whole scene every frame. A cascade's depth depends on
// exactly three things: its light matrix, the raster state it is drawn under, and the casters inside
// that matrix's clip volume. With the camera and sun still (a wall drag, a property edit, a held
// view) the matrix holds; and DH-1590's caster-change record already says WHERE the shape changed
// (the old and new world bounds of every instance whose shadow shape differs). So a cascade is
// reused when:
//
//   1. its matrix is bit-identical to the one its slice was last rasterised with (the camera, sun
//      and DH-0858 sub-texel jitter all held — any of them moving re-renders, as before);
//   2. the raster state matches (`CascadeRasterKey`: bias, cull convention, leaf cutout, excluded
//      kinds, field sink, the texture itself);
//   3. no recorded caster change reaches its volume (the same `IlluminatoramaClipVolume.excludes`
//      argument DH-1590 makes for spot slices: a convex set outside one clip half-space writes —
//      or un-writes — no texel; the depth clip mode is `.clip`, so near/far clip like the sides);
//   4. nothing the CPU cannot see moves inside it: no GPU repack task's vertices, no host instance
//      kernel at all; a swaying instance's swing-sphere counts as a change EVERY frame.
//
// A cascade a bounded change DOES reach re-rasterises only the texel rectangle the change projects
// to (`sunCascadeRegionEnabled`): the pass loads the slice, clears the rectangle to the far plane
// and redraws every caster whose bounds reach the rectangle's own frustum, under a scissor. Depth
// is the minimum over the triangles covering a texel, so outside the rectangle — where no changed
// caster's old or new triangles can land — last frame's depth is still exact. This is the
// virtual-shadow-map page idea at one-rectangle granularity.
//
// Anything that cannot be bounded marks the change unbounded and every cascade re-renders — the
// pre-DH-1668 behaviour. A frame on which the cascades are not drawn (shadows off) forgets every
// slice, so a later frame cannot reuse a map of an older scene.
//
// A/B lever: `sunCascadeReuseEnabled` (env `VIZ_ILLUMI_NO_CASCADE_REUSE=1`); the global
// `VIZ_ILLUMI_NO_STATIC_SKIP=1` turns it off with every other skip.

/// World bounds of casters whose shadow shape changed since a consumer last rendered, or
/// `unbounded` when one could not be bounded (or there were too many).
struct IlluminatoramaCasterChangeLog {
    var bounds: [IlluminatoramaCullBounds] = []
    var unbounded = false
    var isEmpty: Bool { bounds.isEmpty && !unbounded }

    mutating func markUnbounded() { bounds.removeAll(keepingCapacity: true); unbounded = true }

    mutating func add(_ b: IlluminatoramaCullBounds, cap: Int) {
        guard !unbounded else { return }
        guard bounds.count < cap else { markUnbounded(); return }
        bounds.append(b)
    }

    func touches(_ clip: IlluminatoramaClipVolume) -> Bool {
        unbounded || bounds.contains { !clip.excludes($0) }
    }

    mutating func reset() { bounds.removeAll(keepingCapacity: true); unbounded = false }
}

extension IlluminatoramaRenderer {

    /// Every input to a cascade's depth raster besides its matrix and its casters.
    struct CascadeRasterKey: Equatable {
        var bias: ShadowRasterBias
        var storesFrontFaces: Bool
        var cullFollowsDoubleSided: Bool
        var leafCutout: Bool
        var leafCutoutCascades: Int
        var excludedKinds: Set<MeshKind>
        var hiddenKinds: Set<MeshKind>
        var fieldSink: Float
        var map: ObjectIdentifier?
    }

    var cascadeRasterKey: CascadeRasterKey {
        CascadeRasterKey(bias: effectiveShadowRasterBias,
                         storesFrontFaces: shadowStoresFrontFaces,
                         cullFollowsDoubleSided: shadowCullFollowsDoubleSidedForTest,
                         leafCutout: leafCutoutEnabled,
                         leafCutoutCascades: leafCutoutShadowCascades,
                         excludedKinds: directionalShadowExcludedKinds,
                         hiddenKinds: cameraHiddenKinds,
                         fieldSink: fieldSinkProgress,
                         map: shd.shadowMap.map { ObjectIdentifier($0) })
    }

    /// Master A/B lever for the DH-1668 paths (sun-cascade reuse and region re-renders, the swap
    /// diff, the TLAS swap rebuild): reads true only when all are on; setting it sets all of them.
    public var dirtyRegionReuseEnabled: Bool {
        get { sunCascadeReuseEnabled && sunCascadeRegionEnabled && swapDiffEnabled && tlasSwapRebuildEnabled }
        set {
            sunCascadeReuseEnabled = newValue; sunCascadeRegionEnabled = newValue
            swapDiffEnabled = newValue; tlasSwapRebuildEnabled = newValue
        }
    }

    /// Per cascade: what the last frame did with it — `""` reused whole, `"region N%"` re-rasterised
    /// only the rectangle a change reached (N = its share of the slice), otherwise why it re-rendered
    /// whole: "matrix" (camera/sun/jitter moved), "caster-unbounded" (a change nobody could bound),
    /// "caster-wide" (the changed rectangle is most of the slice), "gpu" / "sway" (live geometry
    /// that cannot be bounded), "state" (raster state / first frame), "off".
    public var sunCascadeLastVerdicts: [String] { shd.lastCascadeVerdicts }

    /// What one cascade does this frame.
    enum CascadePlan: Equatable {
        case reuse
        /// Re-rasterise only this texel rectangle (cleared to the far plane first).
        case region(MTLScissorRect)
        case full

        static func == (a: CascadePlan, b: CascadePlan) -> Bool {
            switch (a, b) {
            case (.reuse, .reuse), (.full, .full): return true
            case (.region(let x), .region(let y)):
                return x.x == y.x && x.y == y.y && x.width == y.width && x.height == y.height
            default: return false
            }
        }
    }

    /// Past this share of the slice a region re-render is not worth its clear; draw the whole.
    static let cascadeRegionMaxFraction: Double = 0.5

    /// Decide, per cascade, what its slice needs this frame. Consumes the cascade caster-change
    /// record: after this every slice is reused (no change reaches it), re-rasterised in the
    /// rectangle the changes project to, or re-rendered whole.
    func sunCascadePlans() -> [CascadePlan] {
        let count = Self.cascadeCount
        defer { shd.cascadeChange.reset() }
        let key = cascadeRasterKey
        let keyHeld = key == shd.lastCascadeRasterKey
        shd.lastCascadeRasterKey = key
        var verdicts = Array(repeating: "", count: count)
        var plans = Array(repeating: CascadePlan.full, count: count)
        defer { shd.lastCascadeVerdicts = verdicts }
        guard sunCascadeReuseEnabled, !Self.staticSkipDisabled, shadowsEnabled else {
            for c in 0..<count { verdicts[c] = "off" }
            return plans
        }
        guard keyHeld, onEncodeGPUInstances == nil else {
            for c in 0..<count { verdicts[c] = "state" }
            return plans
        }
        let live = liveShadowCasterBounds()   // nil = live geometry no bound covers
        let size = shd.shadowMap.map { ($0.width, $0.height) } ?? (0, 0)
        for c in 0..<count {
            let vp = shd.cascadeVPs[c]
            guard let last = shd.lastCascadeVPs[c] else { verdicts[c] = "state"; continue }
            guard last == vp else { verdicts[c] = "matrix"; continue }
            let clip = IlluminatoramaClipVolume(vp)
            if shd.cascadeChange.unbounded { verdicts[c] = "caster-unbounded"; continue }
            guard let live else { verdicts[c] = "gpu"; continue }
            let dirty = (shd.cascadeChange.bounds + live).filter { !clip.excludes($0) }
            if dirty.isEmpty { plans[c] = .reuse; continue }
            guard sunCascadeRegionEnabled, shd.shadowRegionClearPipeline != nil,
                  let rect = Self.texelRect(of: dirty, vp: vp, width: size.0, height: size.1) else {
                verdicts[c] = "caster"; continue
            }
            let fraction = Double(rect.width * rect.height) / Double(max(1, size.0 * size.1))
            guard fraction <= Self.cascadeRegionMaxFraction else { verdicts[c] = "caster-wide"; continue }
            plans[c] = .region(rect)
            verdicts[c] = String(format: "region %.1f%%", fraction * 100)
        }
        return plans
    }

    /// World bounds of the casters that move EVERY frame without any instance changing — every
    /// swaying instance's whole swing — or nil when live geometry cannot be bounded: a GPU repack
    /// task rewrites its vertices each frame, and a swing with no CPU-readable mesh is unbounded.
    ///
    /// GPU-CUT indirect draws and GPU-compacted instance records are NOT live here, by the same
    /// contract the spot atlas has reused under since DH-1064: their per-view cut is a function of
    /// the view, and a cascade is only ever reused under the matrix it was drawn with — the camera,
    /// sun and jitter all held. (Counting them live re-rendered the house's region of every cascade
    /// on every still frame of 4000 Sunset, whose Signack-cut indirect draws sit inside the house.)
    private func liveShadowCasterBounds() -> [IlluminatoramaCullBounds]? {
        let repacked = Set(fxs.gpuRepackTasks.map(\.kind))
        if !repacked.isEmpty, gbs.meshGroups.contains(where: {
            repacked.contains($0.kind) && !directionalShadowExcludedKinds.contains($0.kind)
        }) { return nil }
        guard gbs.contentHasSway else { return [] }
        var out: [IlluminatoramaCullBounds] = []
        for s in shd.swayingSpheres {
            if s.radius == .infinity { return nil }
            var b = IlluminatoramaCullBounds.empty
            b.formUnion(center: s.center, radius: s.radius)
            out.append(b)
        }
        return out
    }

    /// The texel rectangle (top-left origin, as `setScissorRect` takes it) that every box in
    /// `boxes` projects into under `vp`, grown a texel each way so the raster rule's edge texels are
    /// inside. nil when a box crosses a perspective eye plane (not projectable) or covers nothing.
    static func texelRect(of boxes: [IlluminatoramaCullBounds], vp: simd_float4x4,
                          width: Int, height: Int) -> MTLScissorRect? {
        guard width > 0, height > 0 else { return nil }
        var lo = SIMD2<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        for b in boxes {
            for i in 0..<8 {
                let p = SIMD3<Float>((i & 1) == 0 ? b.lo.x : b.hi.x, (i & 2) == 0 ? b.lo.y : b.hi.y,
                                     (i & 4) == 0 ? b.lo.z : b.hi.z)
                let q = vp * SIMD4<Float>(p, 1)
                guard q.w > 1e-6 else { return nil }
                let ndc = SIMD2<Float>(q.x, q.y) / q.w
                lo = simd_min(lo, ndc); hi = simd_max(hi, ndc)
            }
        }
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

    /// `vp` narrowed to the texel rectangle `r` of a square `size`² slice: its clip volume is the
    /// part of `vp`'s that rasterises into `r`, so the visibility cull can skip every group that
    /// cannot write a texel of the region.
    static func scissorFrustum(_ vp: simd_float4x4, _ r: MTLScissorRect, size: Int) -> simd_float4x4 {
        let s = Float(max(1, size))
        let ax = Float(r.x) / s * 2 - 1, bx = Float(r.x + r.width) / s * 2 - 1
        let by = 1 - Float(r.y) / s * 2, ay = 1 - Float(r.y + r.height) / s * 2
        let cx = (ax + bx) / 2, hx = max(1e-6, (bx - ax) / 2)
        let cy = (ay + by) / 2, hy = max(1e-6, (by - ay) / 2)
        var m = matrix_identity_float4x4
        m.columns.0 = SIMD4(1 / hx, 0, 0, 0)
        m.columns.1 = SIMD4(0, 1 / hy, 0, 0)
        m.columns.3 = SIMD4(-cx / hx, -cy / hy, 0, 1)
        return m * vp
    }

    /// Clear `r` of the bound slice to the far plane: the region pass loads the slice, so only the
    /// rectangle is redrawn and everything outside it keeps last frame's depth.
    func encodeCascadeRegionClear(_ enc: MTLRenderCommandEncoder, _ r: MTLScissorRect) {
        encodeShadowRegionClear(enc, r)
        staticSkipStats.sunCascadePassesPartial += 1
    }

    /// Clear `r` of the bound depth slice to the far plane and leave the scissor on `r` — the start
    /// of every region re-render (sun cascades, DH-1668; spot/area slices, DH-1674).
    func encodeShadowRegionClear(_ enc: MTLRenderCommandEncoder, _ r: MTLScissorRect) {
        guard let pipe = shd.shadowRegionClearPipeline, let ds = shd.shadowRegionClearDepth else { return }
        enc.setScissorRect(r)
        enc.setRenderPipelineState(pipe)
        enc.setDepthStencilState(ds)
        enc.setCullMode(.none)
        enc.setDepthBias(0, slopeScale: 0, clamp: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    }

    /// Record that cascade `c` was just rasterised with its current matrix (or, when `vp` is nil,
    /// that its slice no longer holds a reusable map).
    func noteCascadeRendered(_ c: Int, _ vp: simd_float4x4?) {
        if shd.lastCascadeVPs.count != Self.cascadeCount {
            shd.lastCascadeVPs = Array(repeating: nil, count: Self.cascadeCount)
        }
        shd.lastCascadeVPs[c] = vp
    }
}

extension IlluminatoramaRenderer {
    /// The draw groups' cull bounds this frame: how many are unbounded (always drawn) and the
    /// diagonals (m) of the bounded ones, largest first. Diagnostic — explains a region pass that
    /// culls nothing.
    public var visibilityCullBoundsCensus: (unbounded: Int, bounded: Int, largestDiagonals: [Float]) {
        var unb = 0
        var diags: [Float] = []
        for b in gbs.meshGroupCullBounds {
            if b.isBounded { diags.append(simd_length(b.hi - b.lo)) } else { unb += 1 }
        }
        return (unb, diags.count, Array(diags.sorted(by: >).prefix(12)))
    }
}
