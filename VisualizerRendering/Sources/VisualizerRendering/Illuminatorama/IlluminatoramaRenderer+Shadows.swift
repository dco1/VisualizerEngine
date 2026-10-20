import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {

    // ── Phase 2.5: Cascaded shadow maps ──────────────────────────────────────
    //
    // Per frame we fit three orthographic light frustums to camera-frustum
    // sub-slices (PSSM, λ=0.5 between log and uniform splits) and rasterise
    // scene depth into a depth2d_array. Disabled cascades still get a depth
    // pass clear, but no draws are issued — keeps the array consistent so
    // the lighting kernel can read from any slice without UB.

    func encodeShadowPasses(_ cb: MTLCommandBuffer) {
        // Even with the sun cascades OFF we still open (and therefore `.clear`) every
        // depth slice to 1.0 and issue no draws — a fully-lit map. Returning early
        // instead (`guard shadowsEnabled else { return }`) left the last ENABLED frame's
        // depths resident in the array, and the lighting kernel gates shadowing on the
        // `kLightingShadowEnabled` FUNCTION CONSTANT alone (`frame.shadowEnabled` is a
        // dead uniform). The init-time uber lighting variant — the one that runs for the
        // frames the shadow-OFF specialization is still compiling in the background — is
        // built with that constant true, so it kept sampling the stale cascade map and
        // printed the previous frame's shadows with the toggle already off. A cleared
        // slice (depth == 1.0) compares as "lit" everywhere, so no consumer can resurrect
        // a shadow regardless of which lighting variant is live. (DH-0619)
        for cascade in 0..<Self.cascadeCount {
            let pass = MTLRenderPassDescriptor()
            pass.depthAttachment.texture = shd.shadowMap
            pass.depthAttachment.slice = cascade
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .store
            pass.depthAttachment.clearDepth = 1.0

            guard let enc = timedRenderEncoder(cb, pass, "shadow.sunCascades") else { continue }
            enc.label = "Illuminatorama.shadow.c\(cascade)"
            // Disabled → the clear above already made this a fully-lit slice; no draws.
            guard shadowsEnabled else { enc.endEncoding(); continue }
            enc.setRenderPipelineState(shd.shadowPipeline)
            enc.setDepthStencilState(gbs.depthState)
            // Front-face cull keeps the shadow projection biased on the
            // back faces of geometry, which combats acne on lit surfaces
            // without the peter-panning a constant bias alone would cause.
            // Set per group below (`shadowCullMode`) — an OPEN shell has no
            // back face to store, so it opts out and casts two-sided.
            var cull: MTLCullMode = shadowInitialCullMode
            enc.setCullMode(cull)
            enc.setFrontFacing(.counterClockwise)
            applyShadowRasterBias(enc)

            var lightVP = shd.cascadeVPs[cascade]
            enc.setVertexBytes(&lightVP,
                               length: MemoryLayout<simd_float4x4>.stride,
                               index: 3)
            // Shader-animated sway (swayMode 2 pendant, 3 wind flap) reads (time, wind, heading) so it casts
            // its swung shadow in phase with the visible mesh; no-op for modes 0/1.
            var shadowClock = SIMD4<Float>(time, treeWindStrength, treeWindHeading, fieldSinkProgress)   // (time, wind, heading, sink)
            enc.setVertexBytes(&shadowClock, length: MemoryLayout<SIMD4<Float>>.stride, index: 4)

            // Phase 4.12 — instanced draw per mesh kind via the recipe
            // built in `uploadInstances`. Same win as the G-buffer pass:
            // one draw call replaces N per-instance draws per cascade /
            // per spot.
            let instStride = MemoryLayout<IlluminatoramaInstance>.stride
            let cullClip = cullVolume(lightVP)   // DH-0534
            for (gi, group) in gbs.meshGroups.enumerated() {
                if directionalShadowExcludedKinds.contains(group.kind) { continue }  // e.g. lamps — no sun shadow
                guard let mesh = meshes[group.kind] else { continue }
                if shouldCull(gi, group, mesh, cullClip, drawn: \.cascadeDrawn, culled: \.cascadeCulled) { continue }
                let want = shadowCullMode(mesh)
                if want != cull { cull = want; enc.setCullMode(cull) }
                let off = instStride * group.start
                enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
                enc.setVertexBuffer(currentInstanceBuffer, offset: off, index: 2)
                enc.drawIndexedPrimitives(
                    type: .triangle,
                    indexCount: mesh.indexCount,
                    indexType: mesh.indexType,
                    indexBuffer: mesh.indexBuffer,
                    indexBufferOffset: 0,
                    instanceCount: group.count,
                    baseVertex: 0,
                    baseInstance: 0
                )
            }
            enc.endEncoding()
        }
    }

    /// Refit the three cascade VPs to the current camera + sun. Result lands
    /// in `cascadeVPs` and `cascadeSplitsView`; `uploadFrameUniforms` then
    /// stuffs them into the frame uniform buffer.
    func updateCascades() {
        let near = max(camera.zNear, 0.05)
        let far = min(shadowMaxDistance, camera.zFar)
        let count = Self.cascadeCount
        let lambda: Float = 0.5

        // PSSM split distances (in view-space positive distance) — log + uniform.
        // Fixed SIMD storage (count + 1 ≤ 8): this ran every frame and used to
        // heap-allocate three small arrays (splits + the two corner lists below).
        var splitsPos = SIMD8<Float>()
        splitsPos[0] = near
        for i in 1...count {
            let f = Float(i) / Float(count)
            let logSplit = near * pow(far / near, f)
            let uniSplit = near + (far - near) * f
            splitsPos[i] = lambda * logSplit + (1 - lambda) * uniSplit
        }
        // Expose far-Z of each cascade to the lighting kernel.
        shd.cascadeSplitsView = SIMD4(splitsPos[1], splitsPos[2], splitsPos[3], 0)

        let invView = camera.viewMatrix.inverse

        let toLight = simd_normalize(directionalLightDirection)

        // DH-0858 sub-texel jitter (see below): the wobble must be ONE shared physical
        // light-position shift across all cascades, not scaled per cascade — captured from
        // cascade 0 (the finest) on its iteration, reused by the coarser cascades after it.
        var jitterTexelWorld0: Float = 0

        for c in 0..<count {
            let n = splitsPos[c]
            let f = splitsPos[c + 1]
            // Frustum extents at each split plane. Asking the camera keeps this correct
            // for BOTH projections: under perspective the extents open out with
            // distance, under orthographic they are constant (a box, not a pyramid), and
            // a hardcoded `d · tan(fov/2)` here would silently fit the cascades to a cone
            // the ortho camera does not have — shadows would land in the wrong places
            // rather than fail loudly.
            let (hHN, hWN) = camera.halfExtents(atViewDistance: n)
            let (hHF, hWF) = camera.halfExtents(atViewDistance: f)
            // View-space corners (Metal/right-handed: looking down -Z), generated
            // procedurally from the index bits (b0 = ±x, b1 = ±y, b2 = near/far)
            // — same 8 corners the old literal arrays held, no allocation.
            func worldCorner(_ i: Int) -> SIMD3<Float> {
                let isFar = (i & 4) != 0
                let d: Float = isFar ? f : n
                let hH = isFar ? hHF : hHN, hW = isFar ? hWF : hWN
                let v = SIMD4<Float>((i & 1) == 0 ? -hW : hW,
                                     (i & 2) == 0 ? -hH : hH, -d, 1)
                let w = invView * v
                return SIMD3(w.x, w.y, w.z)
            }

            // Bounding sphere of the sub-frustum — rotation-invariant, so
            // the cascade extents don't pulse as the camera spins. Two passes
            // (centre, then radius) recompute the corners — 16 mat-vec mults,
            // cheaper than materialising two heap arrays per cascade per frame.
            var center = SIMD3<Float>.zero
            for i in 0..<8 { center += worldCorner(i) }
            center /= 8
            var radius: Float = 0
            for i in 0..<8 {
                radius = max(radius, simd_length(worldCorner(i) - center))
            }
            // Round the radius up slightly so PCF taps near the sphere edge
            // still find geometry.
            radius = ceil(radius * 16) / 16

            // Build a stable light-space frame. The "up" pick avoids the
            // Y-aligned degeneracy where cross(up, forward) would collapse.
            let lightUpHint: SIMD3<Float> = abs(toLight.y) < 0.95 ? SIMD3(0, 1, 0)
                                                                  : SIMD3(0, 0, 1)
            // Push the eye back beyond the sphere's near side and stretch the
            // far plane to match, so occluders sitting between the light and
            // the visible sub-frustum are still included in the shadow map.
            // A `2 * radius` slack catches buildings / trees / etc. directly
            // above the camera frustum without exploding depth precision.
            let casterSlack: Float = radius * 2
            let lightEye = center + toLight * (radius + casterSlack)
            let lightView = Self.lookAtRH(eye: lightEye, target: center, up: lightUpHint)
            let lightProj = Self.orthoRH(
                left: -radius, right: radius,
                bottom: -radius, top: radius,
                near: 0, far: 2 * radius + casterSlack
            )
            // DH-0858 — sub-texel shadow-map jitter. The window/occluder edge rasterizes into
            // this cascade at whole-texel precision (confirmed by pulling the raw depth texture:
            // texelWorld exactly halves when resolution doubles — a real rasterization limit, not
            // a filtering artifact), and no amount of downstream PCF/denoising can recover detail
            // that was never captured. Jittering the light's own ortho projection by a sub-texel
            // offset each accumulated frame — same Halton(2,3) sequence and taaFrameIndex the
            // camera's own TAA jitter already uses (`uploadFrameUniforms`, further down this file)
            // — re-rasterizes the occluder edge at a different sub-texel position every frame, so
            // the photo lane's existing 32-frame accumulation supersamples the OCCLUDER, not just
            // the receiver. Measured on the "Primary Bedroom, Facing North" repro: sub-pixel edge
            // residual std 1.67 -> 0.87, matching the ~49% gain from doubling shadow-map
            // resolution (1.66 -> 0.85) at zero extra VRAM/GPU cost, since this reuses accumulation
            // that already runs every photo-lane frame. Gated on `taaEnabled` alone (not a
            // separate flag) — harmless no-op magnitude (≤0.5 texel, ~1.6-19.6mm depending on
            // cascade) when TAA is off, and this is the same the-photo-lane-IS-real-TAA-now fact
            // DH-0858's history already re-derived once (a stale "TAA off everywhere" assumption
            // misled two earlier sessions in this same investigation — see DH-0856/0858).
            //
            // The wobble is ONE shared physical light-position shift, not scaled per cascade. A
            // first cut jittered each cascade by a fraction of ITS OWN texel (fine near, coarse
            // far), which regressed `testCascadeSplitDoesNotKinkAShadowEdgeCrossingIt`: cascades
            // 0 and 1 then wobbled by different REAL-WORLD amounts every frame, breaking the
            // agreement the cascade-split blend depends on at their shared boundary. Fix: derive
            // the jitter's world-space magnitude from cascade 0's (finest) texel once, then convert
            // that SAME world shift into each cascade's own NDC units — every cascade moves
            // together, so the blend stays seamless.
            var cascadeVP = lightProj * lightView
            if taaEnabled {
                if c == 0 { jitterTexelWorld0 = 2 * radius / Float(shd.shadowMap.width) }
                let jh2 = Self.halton(pfx.taaFrameIndex &+ 1, base: 2) - 0.5
                let jh3 = Self.halton(pfx.taaFrameIndex &+ 1, base: 3) - 0.5
                let jx = jh2 * jitterTexelWorld0 / radius
                let jy = jh3 * jitterTexelWorld0 / radius
                var shadowJitter = matrix_identity_float4x4
                shadowJitter.columns.3 = SIMD4(jx, jy, 0, 1)
                cascadeVP = shadowJitter * cascadeVP
            }
            shd.cascadeVPs[c] = cascadeVP
        }
    }

    /// Phase 4.10 — per-spot shadow setup.
    ///
    /// Two things happen here, both on the CPU:
    ///   1. Assign each spot a slice index into `spotShadowAtlas` (first
    ///      `spotShadowAtlasCapacity` spots get slots 0..N-1; the rest
    ///      get -1 = no shadow data).
    ///   2. Compute each shadowed spot's light-space view-projection
    ///      matrix from `(position, direction, outerCone, radius)` and
    ///      stuff it into the spot struct so the uploaded GPU buffer
    ///      carries it through to the lighting kernel.
    ///
    /// The matrix is a standard perspective with FOV = full outer cone
    /// (so the cone fully fits into NDC), near = small fixed bias, far =
    /// the spot's `radius`. Aspect 1.0 (square shadow map).
    func updateSpotShadows() {
        let capacity = spotShadowCapacityRequest
        let enabled = spotShadowsEnabled
        // Slices go to the shadow-CASTING spots, compacted — not to array indices. A spot with
        // `castsShadow == 0` is skipped and, crucially, does not consume the slice it would
        // have sat on, so a later caster can still have one. Before this the slice WAS the
        // index, which made `castsShadow`'s host-side equivalent (sorting a cone to the back
        // of the array) a no-op in any scene under capacity: the cone landed under the cap and
        // was shadowed anyway. See `IlluminatoramaSpotLight.castsShadow`.
        //
        // Every spot starts unshadowed so a light that loses (or never wanted) a slice cannot
        // sample a stale one — the same reset `updatePointShadows` does.
        shd.spotShadowCasters.removeAll(keepingCapacity: true)
        for i in 0..<spotLights.count { spotLights[i].shadowSliceIndex = -1 }
        for i in 0..<areaLights.count { areaLights[i].shadowSliceIndex = -1 }
        guard enabled else { return }
        // DH-0601 — window portals (area lights) claim slices FIRST. Daylight is the room's
        // dominant daytime source, so it outranks a fixture cone when the atlas is tight; at
        // night there are no portals (dayFrac 0 ⇒ none emitted), so every slice still goes to
        // fixtures. `windowDaylightMode != .portals` scenes never set `castsShadow`, so this
        // loop is a no-op for them (and for Visualizer softboxes).
        for i in 0..<areaLights.count {
            guard areaLights[i].castsShadow != 0 else { continue }
            guard shd.spotShadowCasters.count < capacity else { break }
            let slice = shd.spotShadowCasters.count
            areaLights[i].shadowSliceIndex = Int32(slice)
            var m = areaShadowMatrix(areaLights[i])
            // DH-0889 — sub-texel jitter of the portal's projection, the DH-0858 cascade fix
            // applied to the portal slices. A portal map is ONE 150° perspective, so on a wall
            // its light grazes a texel smears across ~10 screen px and a thin occluder's shadow
            // edge prints as a sawtooth, one tooth per texel. Shifting the projection by a
            // Halton(2,3) sub-texel offset each accumulated frame re-rasterises the occluder at a
            // new sub-texel position, so the still's accumulation supersamples it. The SAME
            // matrix renders the slice and samples it, and a changed matrix already defeats the
            // static-slice reuse (`mats == lastSpotShadowMats`), so nothing reads a stale map.
            // Off by default: the live canvas keeps its reuse; a host enables it for stills.
            if areaShadowJitterEnabled && taaEnabled {
                let texel = 2 / Float(max(1, shd.spotShadowAtlas.width))
                var j = matrix_identity_float4x4
                j.columns.3 = SIMD4((Self.halton(pfx.taaFrameIndex &+ 1, base: 2) - 0.5) * texel,
                                    (Self.halton(pfx.taaFrameIndex &+ 1, base: 3) - 0.5) * texel, 0, 1)
                m = j * m
            }
            areaLights[i].shadowMatrix = m
            // A portal faces INTO the room and its frustum is capped at the portal range —
            // the exterior lawn (the only GPU-fed geometry) can never enter it, so the slice
            // is reusable while the CPU-visible scene holds (same argument as an interior cone,
            // DH-0631).
            shd.spotShadowCasters.append(ShadowCaster(matrix: m, reuseBit: true, owner: .area(i)))
        }
        for i in 0..<spotLights.count {
            guard spotLights[i].castsShadow != 0 else { continue }
            guard shd.spotShadowCasters.count < capacity else { break }
            let slice = shd.spotShadowCasters.count
            spotLights[i].shadowSliceIndex = Int32(slice)
            // Light view: apex at spot.position, looking down the spot's
            // direction (which already points the way light travels).
            let position = spotLights[i].position
            let dir = simd_normalize(spotLights[i].direction)
            // Pick a stable up that isn't degenerate with the cone axis.
            let upHint: SIMD3<Float> = abs(dir.y) < 0.95 ? SIMD3(0, 1, 0)
                                                          : SIMD3(0, 0, 1)
            let lightView = Self.lookAtRH(eye: position,
                                           target: position + dir,
                                           up: upHint)
            // Cone half-angle from cosine. `outerCone` is `cos(half-angle)`
            // and the perspective takes the FULL field-of-view (= 2 ×
            // half-angle). Add a small margin (10%) so the cone edge
            // doesn't graze the edge of the shadow map.
            let halfAngle = acos(min(0.9999, max(0.01, spotLights[i].outerCone)))
            let fovY = min(2.95, halfAngle * 2 * 1.1) // clamp <170° to avoid divergence
            let far  = max(0.5, spotLights[i].radius)
            // DERIVED, not fixed. A hardcoded 0.1 m near plane clips the occluder nearest the
            // bulb, and for a fixture standing against a wall that occluder IS the wall. A ray
            // from the bulb to a floor point at horizontal distance d crosses a wall face at
            // lateral offset o at view depth h·o/d, so the near plane silently stops shadowing
            // beyond d = h·o/near. With a floor lamp 43 mm off a wall face (bulb 1.19 m up,
            // 1.79 m pool) that cutoff is 0.51 m — the lamp's own foot — and its light leaks
            // across the room. It only became reachable when `shadowStoresFrontFaces` flipped
            // true (2026-08-19): storing the NEAR face moved the occluder from 157 mm off the
            // light to 43 mm, inside the clip. Measured on
            // `testWallOcclusionIsTheShadowMapNotJustConeDirection`: far-room delta 0.6 luma
            // shipped, 71.8 with the old convention.
            //
            // The atlas is 512² `.depth32Float`, so far/near ≈ 500 costs nothing in precision.
            let near = max(0.005, far * 0.002)
            let proj = Self.perspectiveRH(fovY: fovY, aspect: 1.0, near: near, far: far)
            spotLights[i].shadowMatrix = proj * lightView
            let reuse = spotLights[i].castsShadow & IlluminatoramaSpotLight.castsShadowIgnoresGPUGeometry != 0
            shd.spotShadowCasters.append(ShadowCaster(matrix: spotLights[i].shadowMatrix,
                                                  reuseBit: reuse, owner: .spot(i)))
        }
        ensureSpotShadowAtlas(casting: shd.spotShadowCasters.count)
        // If the reallocation could not happen, no slice index may point past the atlas —
        // trim the surplus back to unshadowed (the same state a caster past the cap is in).
        while shd.spotShadowCasters.count > shd.spotShadowAtlasCapacity {
            switch shd.spotShadowCasters.removeLast().owner {
            case .spot(let i): spotLights[i].shadowSliceIndex = -1
            case .area(let i): areaLights[i].shadowSliceIndex = -1
            }
        }
    }

    /// DH-0601 — the light-space view-projection for a window portal's VISIBILITY shadow map.
    /// A single perspective from the portal centre looking along its inward emitting normal
    /// (`normalize(cross(ex, ey))`), wide enough to frame the room the portal lights. It cannot
    /// cover the full emitting hemisphere — the LTC eval still lights the grazing edges — but a
    /// receiver outside the frustum reads as fully visible (the same fallback the spot path
    /// takes), and the near/mid room where objects actually stand is well inside it. Near/far
    /// follow the spot convention against the same 512² depth atlas.
    private func areaShadowMatrix(_ al: IlluminatoramaAreaLight) -> simd_float4x4 {
        var nL = simd_cross(al.ex, al.ey)
        let len = simd_length(nL)
        guard len > 1e-6 else { return matrix_identity_float4x4 }
        nL /= len
        let upHint: SIMD3<Float> = abs(nL.y) < 0.95 ? SIMD3(0, 1, 0) : SIMD3(0, 0, 1)
        let lightView = Self.lookAtRH(eye: al.center, target: al.center + nL, up: upHint)
        let far  = max(0.5, al.radius)
        let near = max(0.005, far * 0.002)
        // ~150° — wide enough to reach the near-window floor and flanking walls without the
        // grazing-edge precision loss of a full 170°.
        let fovY: Float = 2.62
        let proj = Self.perspectiveRH(fovY: fovY, aspect: 1.0, near: near, far: far)
        return proj * lightView
    }

    /// DH-0134 — size the spot-shadow atlas to the host's request, lazily.
    ///
    /// Grows only when a CASTER needs a slice the atlas does not have (never on the request
    /// alone, so a host that asks for 64 and has three lamps still pays for 8), in whole
    /// 8-slice steps so a scene gaining lights one at a time does not reallocate per light,
    /// and never past the request. Shrinks back to the shipped 8-slice footprint when the
    /// request drops below the allocation — the host left its photo lane. A failed
    /// reallocation keeps the old atlas; the caller trims slice indices to what exists.
    private func ensureSpotShadowAtlas(casting: Int) {
        let request = spotShadowCapacityRequest
        let floor = Self.spotShadowCapacityDefault
        var want = shd.spotShadowAtlasCapacity
        if casting > want { want = min(request, ((casting + 7) / 8) * 8) }
        if request < want && want > floor { want = max(floor, request) }
        guard want != shd.spotShadowAtlasCapacity else { return }
        reallocateSpotShadowAtlas(resolution: spotShadowMapResolution, capacity: want)
    }

    /// Rebuild the spot/area shadow atlas at `resolution` × `capacity`, and drop every cache that
    /// points at the old texture. The ONE place the atlas is replaced, so a resolution change and
    /// a capacity change cannot forget different halves of the invalidation.
    func reallocateSpotShadowAtlas(resolution: Int, capacity: Int) {
        let fresh: MTLTexture
        do {
            fresh = try Self.makeSpotShadowAtlas(device: device,
                                                 resolution: resolution,
                                                 capacity: capacity)
        } catch {
            Self.log.error("spot-shadow atlas reallocation to \(capacity) slices at \(resolution)² failed: \(error.localizedDescription); keeping \(self.shd.spotShadowAtlasCapacity)")
            return
        }
        shd.spotShadowAtlas = fresh
        shd.spotShadowAtlasCapacity = capacity
        // The cached pass descriptors point at the OLD texture, and the static-scene skip
        // would otherwise believe the (empty) new atlas already holds this frame's maps.
        shd.spotShadowPassDescs.removeAll()
        shd.spotShadowPassLabels.removeAll()
        shd.lastSpotShadowMats.removeAll()
    }

    /// Render the depth-only pass for each shadowed spot into its slice
    /// of the spot shadow atlas. Skips when the toggle is off, when no
    /// spots are present, or when the host hasn't pushed any instances
    /// (no occluders → no shadow data needed).
    func encodeSpotShadowPasses(_ cb: MTLCommandBuffer) {
        guard spotShadowsEnabled, !shd.spotShadowCasters.isEmpty, !instances.isEmpty else { return }
        let count = shd.spotShadowCasters.count
        // PERF (static-scene skip): spot shadow maps are LIGHT-space — a camera
        // orbit re-rendered every slice (up to 8 full scene depth passes) into
        // byte-identical maps every frame. When the scene content held (see
        // `sceneStaticForShadows`) and every slice's light matrix is unchanged,
        // the atlas already contains exactly these maps — reuse them.
        let mats = shd.spotShadowCasters.map { $0.matrix }
        if sceneStaticForShadows && mats == shd.lastSpotShadowMats {
            staticSkipStats.spotShadowPassesSkipped += count
            return
        }
        // DH-0631 / DH-0637 — per-slice reuse when the only things keeping the scene from "static"
        // are live GPU-fed geometry (the lawn) and/or swaying pendants. A slice still holds a valid
        // map from the last time it was rendered with this same matrix if (a) every CPU-visible
        // instance held, (b) either no GPU geometry is live or the owner asserts none can enter its
        // cone, and (c) no swaying instance's swing-sphere intersects its cone. Slices that fail
        // (b) or (c) — exterior cones, the can above a swinging pendant — re-render as before.
        let gpuGeometryLive = !fxs.gpuRepackTasks.isEmpty || onEncodeGPUInstances != nil
        let perSliceBase = spotShadowGPUGeometryOptOutEnabled
            && gbs.instanceShapeStableFrames >= 1 && mats == shd.lastSpotShadowMats
        shd.lastSpotShadowMats = mats
        // Pass descriptors + labels are cached per slice (they never change after
        // the atlas exists) — this loop used to allocate a fresh descriptor and
        // interpolate a fresh label string per slice per frame.
        while shd.spotShadowPassDescs.count < count {
            let slice = shd.spotShadowPassDescs.count
            let pass = MTLRenderPassDescriptor()
            pass.depthAttachment.texture = shd.spotShadowAtlas
            pass.depthAttachment.slice = slice
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .store
            pass.depthAttachment.clearDepth = 1.0
            shd.spotShadowPassDescs.append(pass)
            shd.spotShadowPassLabels.append("Illuminatorama.spotShadow.s\(slice)")
        }
        for slice in 0..<count {
            if perSliceBase {
                // (b) GPU geometry: none live, or the owner asserted none can enter its frustum.
                // (c) Sway (DH-0637): no swaying instance's swing-sphere intersects a spot's cone; an
                //     area-light slice has no cone to test, so it re-renders while anything sways.
                let caster = shd.spotShadowCasters[slice]
                let gpuOK = !gpuGeometryLive || caster.reuseBit
                let swayOK: Bool
                switch caster.owner {
                case .spot(let li): swayOK = !swayingGeometryCanEnter(spotLights[li])
                case .area:         swayOK = shd.swayingSpheres.isEmpty
                }
                if gpuOK && swayOK {
                    staticSkipStats.spotShadowPassesSkipped += 1
                    continue
                }
            }
            guard let enc = timedRenderEncoder(cb, shd.spotShadowPassDescs[slice], "shadow.spot") else { continue }
            enc.label = shd.spotShadowPassLabels[slice]
            enc.setRenderPipelineState(shd.shadowPipeline)
            enc.setDepthStencilState(gbs.depthState)
            // Same back-face-cast trick as the cascaded path, and the same per-group
            // opt-out for open shells (`shadowCullMode`).
            var cull: MTLCullMode = shadowInitialCullMode
            enc.setCullMode(cull)
            enc.setFrontFacing(.counterClockwise)
            applyShadowRasterBias(enc)

            var lightVP = shd.spotShadowCasters[slice].matrix
            enc.setVertexBytes(&lightVP,
                               length: MemoryLayout<simd_float4x4>.stride,
                               index: 3)
            // Shader-animated sway (swayMode 2 pendant, 3 wind flap) reads (time, wind, heading) so it casts
            // its swung shadow in phase with the visible mesh; no-op for modes 0/1.
            var shadowClock = SIMD4<Float>(time, treeWindStrength, treeWindHeading, fieldSinkProgress)   // (time, wind, heading, sink)
            enc.setVertexBytes(&shadowClock, length: MemoryLayout<SIMD4<Float>>.stride, index: 4)

            // Phase 4.12 — instanced draw per mesh kind via the recipe
            // built in `uploadInstances`. Same win as the G-buffer pass:
            // one draw call replaces N per-instance draws per cascade /
            // per spot.
            let instStride = MemoryLayout<IlluminatoramaInstance>.stride
            let cullClip = cullVolume(lightVP)   // DH-0534
            for (gi, group) in gbs.meshGroups.enumerated() {
                guard let mesh = meshes[group.kind] else { continue }
                if shouldCull(gi, group, mesh, cullClip, drawn: \.spotDrawn, culled: \.spotCulled) { continue }
                let want = shadowCullMode(mesh)
                if want != cull { cull = want; enc.setCullMode(cull) }
                let off = instStride * group.start
                enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
                enc.setVertexBuffer(currentInstanceBuffer, offset: off, index: 2)
                enc.drawIndexedPrimitives(
                    type: .triangle,
                    indexCount: mesh.indexCount,
                    indexType: mesh.indexType,
                    indexBuffer: mesh.indexBuffer,
                    indexBufferOffset: 0,
                    instanceCount: group.count,
                    baseVertex: 0,
                    baseInstance: 0
                )
            }
            enc.endEncoding()
        }
    }

    // ── Point-light cubemap shadows ───────────────────────────────────────────
    //
    // The point-light analogue of `updateSpotShadows` + `encodeSpotShadowPasses`.
    // A point light is omnidirectional, so one perspective shadow map can't cover
    // it — instead we render SIX 90°-FOV depth maps, one per cube face (+X −X +Y
    // −Y +Z −Z), into 6 consecutive slices of a shared depth atlas. The lighting
    // kernel then picks the dominant-axis face from the light→fragment direction,
    // projects the fragment through that face's VP, and PCF-compares — the same
    // matrix-projection + hardware-compare the spot path uses, just with a face
    // pick in front. Result: the bulb is occluded by real geometry (walls block,
    // furniture casts shadows) yet light still POURS through an open doorway,
    // because the doorway is a genuine gap the depth map records as "far".

    /// The 6 cube-face look directions and their stable up vectors (RH). The up
    /// for ±Y avoids the look-at degeneracy where forward ∥ up.
    private static let cubeFaceDirs: [(fwd: SIMD3<Float>, up: SIMD3<Float>)] = [
        (SIMD3( 1, 0, 0), SIMD3(0, 1, 0)),   // +X
        (SIMD3(-1, 0, 0), SIMD3(0, 1, 0)),   // -X
        (SIMD3( 0, 1, 0), SIMD3(0, 0, 1)),   // +Y
        (SIMD3( 0,-1, 0), SIMD3(0, 0,-1)),   // -Y
        (SIMD3( 0, 0, 1), SIMD3(0, 1, 0)),   // +Z
        (SIMD3( 0, 0,-1), SIMD3(0, 1, 0)),   // -Z
    ]

    /// CPU step (mirrors `updateSpotShadows`): choose which point lights get a cube
    /// (the N nearest the camera, capped at `pointShadowCubeCapacity`), stamp each
    /// chosen light's `shadowCubeIndex`, and compute its 6 face view-projection
    /// matrices into `pointShadowFaceBuffer`. Lights not chosen get `shadowCubeIndex
    /// = -1` (kernel treats them as fully visible). A no-op that clears every index
    /// to -1 when the feature is off, so a stale buffer is never sampled.
    func updatePointShadows() {
        // Feature off (or no lazy resources yet) → force every light unshadowed.
        guard pointShadowsEnabled else {
            for i in pointLights.indices { pointLights[i].shadowCubeIndex = -1 }
            return
        }
        ensurePointShadowResources()
        guard let faceBuffer = pointShadowFaceBuffer else {
            for i in pointLights.indices { pointLights[i].shadowCubeIndex = -1 }
            return
        }

        // Rank the shadow-casting lights by camera distance and keep the nearest N.
        let cam = camera.position
        let casters = pointLights.indices
            .filter { pointLights[$0].castsShadow != 0 }
            .sorted { simd_distance(pointLights[$0].position, cam)
                    < simd_distance(pointLights[$1].position, cam) }
        let chosen = Array(casters.prefix(shd.pointShadowCubeCapacity))

        // Everyone starts unshadowed; chosen lights claim a cube page below.
        for i in pointLights.indices { pointLights[i].shadowCubeIndex = -1 }

        let faces = faceBuffer.contents().bindMemory(
            to: simd_float4x4.self, capacity: shd.pointShadowCubeCapacity * 6)
        let near: Float = 0.05
        for (cube, lightIdx) in chosen.enumerated() {
            pointLights[lightIdx].shadowCubeIndex = Int32(cube)
            let pos = pointLights[lightIdx].position
            let far = max(0.5, pointLights[lightIdx].radius)
            // 90° per face so the six frusta tile the full sphere with no gaps.
            let proj = Self.perspectiveRH(fovY: .pi / 2, aspect: 1.0, near: near, far: far)
            for f in 0..<6 {
                let face = Self.cubeFaceDirs[f]
                let view = Self.lookAtRH(eye: pos, target: pos + face.fwd, up: face.up)
                faces[cube * 6 + f] = proj * view
            }
        }
    }

    /// Depth-only pass: render the scene from each chosen point light into its 6
    /// cube-face slices. Reuses `shadowPipeline` + the exact instanced-draw loop
    /// the spot path uses. Skips entirely when the feature is off, when nothing
    /// claimed a cube, or when there are no occluders.
    func encodePointShadowPasses(_ cb: MTLCommandBuffer) {
        guard pointShadowsEnabled, !instances.isEmpty,
              let atlas = shd.pointShadowAtlas, let faceBuffer = pointShadowFaceBuffer
        else { return }
        let cubeCount = pointLights.reduce(0) { $0 + (($1.shadowCubeIndex >= 0) ? 1 : 0) }
        guard cubeCount > 0 else { return }

        // PERF (static-scene skip): a point cube is 6 full scene depth passes,
        // and its face matrices depend only on the light's position + radius
        // (the ranking that ASSIGNS cubes is camera-driven, so orbit can still
        // reshuffle pages — that breaks the assignment vector and re-renders).
        // With stable content, identical (pos, radius) per cube-holder, and an
        // identical page assignment, the atlas already holds these cubes.
        var sig: [SIMD4<Float>] = []
        var assigns: [Int32] = []
        for pl in pointLights where pl.shadowCubeIndex >= 0 {
            sig.append(SIMD4(pl.position, pl.radius))
            assigns.append(pl.shadowCubeIndex)
        }
        if sceneStaticForShadows && sig == shd.lastPointShadowSig && assigns == shd.lastPointCubeAssign {
            staticSkipStats.pointShadowPassesSkipped += cubeCount * 6
            return
        }
        shd.lastPointShadowSig = sig
        shd.lastPointCubeAssign = assigns

        let faces = faceBuffer.contents().bindMemory(
            to: simd_float4x4.self, capacity: shd.pointShadowCubeCapacity * 6)
        let instStride = MemoryLayout<IlluminatoramaInstance>.stride
        // Pass descriptors + labels cached per cube-face slice (fixed atlas).
        while shd.pointShadowPassDescs.count < shd.pointShadowCubeCapacity * 6 {
            let slice = shd.pointShadowPassDescs.count
            let pass = MTLRenderPassDescriptor()
            pass.depthAttachment.texture = atlas
            pass.depthAttachment.slice = slice
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .store
            pass.depthAttachment.clearDepth = 1.0
            shd.pointShadowPassDescs.append(pass)
            shd.pointShadowPassLabels.append("Illuminatorama.pointShadow.c\(slice / 6).f\(slice % 6)")
        }
        for pl in pointLights where pl.shadowCubeIndex >= 0 {
            let cube = Int(pl.shadowCubeIndex)
            for f in 0..<6 {
                let slice = cube * 6 + f
                guard let enc = timedRenderEncoder(cb, shd.pointShadowPassDescs[slice], "shadow.point") else { continue }
                enc.label = shd.pointShadowPassLabels[slice]
                enc.setRenderPipelineState(shd.shadowPipeline)
                enc.setDepthStencilState(gbs.depthState)
                // Back-face cast, same acne mitigation as the sun / spot maps, and the
                // same per-group opt-out for open shells (`shadowCullMode`).
                var cull: MTLCullMode = shadowInitialCullMode
                enc.setCullMode(cull)
                enc.setFrontFacing(.counterClockwise)
                applyShadowRasterBias(enc)

                var lightVP = faces[slice]
                enc.setVertexBytes(&lightVP, length: MemoryLayout<simd_float4x4>.stride, index: 3)
                var shadowClock = SIMD4<Float>(time, treeWindStrength, treeWindHeading, fieldSinkProgress)   // (time, wind, heading, sink)
                enc.setVertexBytes(&shadowClock, length: MemoryLayout<SIMD4<Float>>.stride, index: 4)

                let cullClip = cullVolume(lightVP)   // DH-0534
                for (gi, group) in gbs.meshGroups.enumerated() {
                    guard let mesh = meshes[group.kind] else { continue }
                    if shouldCull(gi, group, mesh, cullClip, drawn: \.pointDrawn, culled: \.pointCulled) { continue }
                    let want = shadowCullMode(mesh)
                    if want != cull { cull = want; enc.setCullMode(cull) }
                    let off = instStride * group.start
                    enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
                    enc.setVertexBuffer(currentInstanceBuffer, offset: off, index: 2)
                    enc.drawIndexedPrimitives(
                        type: .triangle,
                        indexCount: mesh.indexCount,
                        indexType: mesh.indexType,
                        indexBuffer: mesh.indexBuffer,
                        indexBufferOffset: 0,
                        instanceCount: group.count,
                        baseVertex: 0,
                        baseInstance: 0
                    )
                }
                enc.endEncoding()
            }
        }
    }

    /// Lazily allocate the point-shadow depth atlas (6 slices per cube) + the
    /// per-face matrix buffer the first time the feature is enabled. A no-op once
    /// allocated. Never called on a Visualizer frame (feature stays off), so the
    /// memory is only paid where the app opts in.
    private func ensurePointShadowResources() {
        if shd.pointShadowAtlas == nil {
            let d = MTLTextureDescriptor()
            d.textureType = .type2DArray
            d.pixelFormat = .depth32Float
            d.width = shd.pointShadowMapResolution
            d.height = shd.pointShadowMapResolution
            d.arrayLength = shd.pointShadowCubeCapacity * 6
            d.usage = [.renderTarget, .shaderRead]
            d.storageMode = .private
            if let t = device.makeTexture(descriptor: d) {
                t.label = "Illuminatorama.pointShadowAtlas"
                shd.pointShadowAtlas = t
            }
        }
        if shd.pointShadowFaceBufferA == nil || shd.pointShadowFaceBufferB == nil {
            let len = MemoryLayout<simd_float4x4>.stride * shd.pointShadowCubeCapacity * 6
            if let a = device.makeBuffer(length: len, options: .storageModeShared),
               let b = device.makeBuffer(length: len, options: .storageModeShared) {
                a.label = "Illuminatorama.pointShadowFacesA"
                b.label = "Illuminatorama.pointShadowFacesB"
                shd.pointShadowFaceBufferA = a
                shd.pointShadowFaceBufferB = b
            }
        }
    }
}

// MARK: - ShadowState

/// Shadow maps: sun cascades, spot atlas, point cubes, static-scene reuse bookkeeping.
/// Owned by the renderer as `shd`; every member keeps the name it had on the renderer.
@MainActor
final class ShadowState {
    /// Spot-shadow skip signature: the atlas slices currently hold depth maps
    /// rendered with exactly these light matrices (and the stable content).
    var lastSpotShadowMats: [simd_float4x4] = []

    var spotShadowCasters: [IlluminatoramaRenderer.ShadowCaster] = []
    /// Point-shadow skip signature: (position, radius) per cube-holding light in
    /// enumeration order + the cube-page assignment vector. Any ranking change,
    /// light move, or radius change breaks equality → re-render.
    var lastPointShadowSig: [SIMD4<Float>] = []
    var lastPointCubeAssign: [Int32] = []
    /// Cached per-slice shadow pass descriptors + labels (spot atlas slices and
    /// point cube-face slices) — built once, reused every re-render; the encode
    /// loops used to allocate ~35 descriptors + label strings per frame.
    var spotShadowPassDescs: [MTLRenderPassDescriptor] = []
    var spotShadowPassLabels: [String] = []
    var pointShadowPassDescs: [MTLRenderPassDescriptor] = []
    var pointShadowPassLabels: [String] = []

    /// World-space spheres that bound every SWAYING instance's whole swing (DH-0637): the mesh's
    /// bounding sphere under the instance's matrix, grown by the pendulum's reach (amplitude ×
    /// diameter, an upper bound on how far a top-pivot pendant's foot travels) and its jostle.
    /// Rebuilt with `contentHasSway`, i.e. only when the instance SHAPE changed — never per frame.
    /// A swaying instance whose mesh has no CPU-readable vertices contributes an unbounded entry
    /// (`radius = .infinity`), which blocks every cone, the old behaviour.
    var swayingSpheres: [(center: SIMD3<Float>, radius: Float)] = []

    // Phase 2.5 — cascaded shadow maps. One depth array (3 slices) shared
    // across cascades; per-slice views aren't required because we attach a
    // single slice at a time via `pass.depthAttachment.slice`. The cascade
    // VP matrices are recomputed per frame and stuffed into FrameUniforms.
    var shadowMap: MTLTexture!
    var cascadeVPs: [simd_float4x4] = Array(repeating: matrix_identity_float4x4, count: 3)

    // Phase 4.10 — depth atlas for spot light shadow maps. One slice per
    // shadowed spot. 8 slices × 512² × 4-byte depth = 8 MB at rest. NOT fixed any more
    // (DH-0134): `ensureSpotShadowAtlas` reallocates toward the host's
    // `spotShadowCapacityRequest` when a caster needs a slice it does not have, and shrinks
    // back when the request drops. Every binding reads the var at encode time, and the
    // cached per-slice pass descriptors are dropped on reallocation.
    var spotShadowAtlas: MTLTexture!
    var spotShadowAtlasCapacity: Int = IlluminatoramaRenderer.spotShadowCapacityDefault

    var cascadeSplitsView: SIMD4<Float> = .zero

    // ── Point-light cubemap shadows (lazy — allocated on first enable) ─────────
    // A single depth 2D-array texture holds every shadowed point light's cube as
    // 6 consecutive slices (light c, face f → slice c*6 + f). Reusing a 2D-array
    // (not a native cube) lets the depth PASS reuse the exact spot-shadow slice
    // mechanism, and the SAMPLE reuse the exact matrix-projection + PCF path
    // (per-face perspective VP, dominant-axis face select). Nil until the first
    // frame `pointShadowsEnabled` is true → Visualizer never allocates it.
    var pointShadowAtlas: MTLTexture?
    /// The 6 face view-projection matrices per shadow-casting light, uploaded to
    /// the GPU so the kernel can project a fragment into the chosen face. Layout:
    /// `[cube0.face0 … cube0.face5, cube1.face0 …]`. Grown with the atlas.
    // A/B ring pair (see the MFIF ring block) — `updatePointShadows` rewrites the
    // face matrices on the CPU every frame, so the slot must alternate with the
    // frame parity or the in-flight frame's kernel reads mid-write matrices.
    var pointShadowFaceBufferA: MTLBuffer?
    var pointShadowFaceBufferB: MTLBuffer?

    let pointShadowCubeCapacity: Int = 4      // N nearest shadow-casting lights
    let pointShadowMapResolution: Int = 512

    var shadowPipeline: MTLRenderPipelineState!

    // ── Per-frame input buffers — ring-buffered (maxFramesInFlight) ─────────────
    // These hold the frame's uniforms + lights + impostor params. They are
    // `memcpy`'d in place at the top of every `render()` and bound (read-only)
    // by that frame's GPU passes. At `maxFramesInFlight == 1` an in-place
    // single buffer was safe. At 2 the CPU encodes frame N+1 (overwriting the
    // buffer) while the GPU still reads it for frame N → torn lighting (the
    // blocky floor-reflection band on the eggs) and uneven cadence (POV
    // jitter, because the camera matrices in the frame uniforms get half-stomped
    // mid-read). Fix: cycle one slot per accepted frame via `frameRingIndex`,
    // the same scheme the instance buffer already uses (`useBufferA`) and the
    // LDR present pool uses (`acquireWriteTarget`). Each property below resolves
    // to the current ring slot so the bind sites are unchanged.
    var pointLightRing: [MTLBuffer]!

    var pointLightCapacity: Int!
    var spotLightRing: [MTLBuffer]!

    /// Daydream DH-0540 — the host's spot-profile table: `spotLightProfileRowCount` rows of
    /// `illuminatoramaSpotProfileSampleCount` floats each, a luminaire's measured distribution over 0..90° off
    /// axis, which a spot names with `profileIndex`. Allocated as a NEW buffer on every `setSpotLightProfiles`
    /// (never mutated in place — an in-flight frame may still read the old one); a 1-float placeholder when the
    /// host never calls it, so every kernel can bind it unconditionally.
    var spotProfileBuffer: MTLBuffer!

    var spotLightCapacity: Int!

    /// Side length (px) of each directional (sun) cascade shadow map, set once at
    /// init (see `IlluminatoramaRenderer.init`). Default 2048 — a larger value
    /// halves the outer-cascade ground texel per doubling (finer, less-aliased sun
    /// shadow edges) at 4× shadow-map memory. The lighting shader derives its PCF
    /// texel size from `shadowMap.get_width()`, so this scales safely with no
    /// shader change.
    var shadowMapResolution: Int!
}

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    /// Who owns each shadow-atlas slice, in slice order (DH-0601 — the atlas is shared by
    /// spot cones AND window-portal area lights). Set by `updateSpotShadows`;
    /// `encodeSpotShadowPasses` renders through it. A slice is no longer a light's array
    /// index — see `IlluminatoramaSpotLight.castsShadow`.
    enum ShadowOwner { case spot(Int); case area(Int) }

    /// One caster per atlas slice, in slice order. `matrix` is the light-space VP the depth
    /// pass rasterises with; `reuseBit` = the DH-0631 "no GPU-fed geometry can enter this
    /// frustum" assertion; `owner` maps back to the light so a trimmed slice can be reset.
    struct ShadowCaster { var matrix: simd_float4x4; var reuseBit: Bool; var owner: ShadowOwner }

    func rebuildSwayingSpheres() {
        shd.swayingSpheres.removeAll(keepingCapacity: true)
        guard gbs.contentHasSway else { return }
        for ref in instances where ref.data.swayMode >= 2 {
            let m = ref.data.modelMatrix
            guard let mesh = meshes[ref.meshKind], let local = mesh.boundingSphere else {
                shd.swayingSpheres.append((SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z), .infinity))
                continue
            }
            let c4 = m * SIMD4<Float>(local.center, 1)
            let scale = max(simd_length(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z)),
                            simd_length(SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z)),
                            simd_length(SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z)))
            let r = local.radius * scale
            // Mode 3's flap scales with the frame's wind, which has no upper clamp.
            let reach = ref.data.swayMode == 3 ? max(1, treeWindStrength) : 1
            let swing = abs(ref.data.swayLean) * reach * 2 * r + abs(ref.data.swayJostle)
            shd.swayingSpheres.append((SIMD3(c4.x, c4.y, c4.z), r + swing))
        }
    }

    /// Does any swaying instance's swing-sphere intersect this spot's shadow frustum (the cone
    /// `updateSpotShadows` fits the map to: outer half-angle × 1.1, out to `radius`)? A slice whose
    /// cone none of them can enter still holds a valid map while everything CPU-visible holds.
    func swayingGeometryCanEnter(_ spot: IlluminatoramaSpotLight) -> Bool {
        guard !shd.swayingSpheres.isEmpty else { return false }
        let apex = spot.position
        let axis = simd_normalize(spot.direction)
        let half = min(1.475, acos(min(0.9999, max(0.01, spot.outerCone))) * 1.1)
        let cosT = cos(half), sinT = sin(half)
        let range = max(0.5, spot.radius)
        for s in shd.swayingSpheres {
            if s.radius == .infinity { return true }
            let v = s.center - apex
            let along = simd_dot(v, axis)
            if along > range + s.radius || along < -s.radius { continue }
            let perp = simd_length(v - axis * along)
            // Signed distance from the cone's surface (negative inside the cone).
            if perp * cosT - along * sinT <= s.radius { return true }
        }
        return false
    }

    /// Cull mode for `mesh` in a shadow pass. `.front` (second-depth casting) is the
    /// renderer's primary acne defence and the default for everything; a mesh that is an
    /// OPEN shell has no second depth to store and opts into `.none` — see
    /// `IlluminatoramaMesh.shadowCastsBothFaces`.
    func shadowCullMode(_ mesh: IlluminatoramaMesh) -> MTLCullMode {
        let bothFaces = shadowCullFollowsDoubleSidedForTest ? mesh.doubleSided
                                                            : mesh.shadowCastsBothFaces
        if bothFaces { return .none }
        return shadowStoresFrontFaces ? .back : .front
    }

    /// **Raster-time depth bias for the shadow passes — the acne defence that PAIRS with
    /// front-face storage.**
    ///
    /// Second-depth needs none: the stored depth is a whole solid behind any lit surface, so
    /// `shadowBias` is a hair of float slop and `shadowSlopeBias` is 0. Storing FRONT faces
    /// removes that margin, and a flat surface then self-shadows wherever the depth ramp ACROSS
    /// one shadow texel exceeds the bias — the ripple that reads as sand-bar banding on a plain
    /// wall (Danny, 2026-08-19: "banding all over … looks like a beach-ocean floor").
    ///
    /// This is *not* the receiver-side `shadowSlopeBias` that was tried and reverted. That one
    /// scales with `(1 - NdotL)`, so it dumps its full value onto grazing WALLS and detaches
    /// their shadows from the contact line. This scales with the occluder primitive's own depth
    /// gradient in the shadow map — dz/dtexel, exactly the quantity that produces acne — and is
    /// applied while the caster is rasterised, so a receiver's own orientation never enters into
    /// it. `clamp` caps the push so a face seen edge-on by the light (infinite slope) cannot
    /// bias itself out of shadowing altogether, which is what would re-open the wall-foot leak.
    public struct ShadowRasterBias: Equatable, Sendable {
        /// Constant offset in depth-buffer units (see `MTLRenderCommandEncoder.setDepthBias`).
        public var constant: Float
        /// Multiplier on the primitive's maximum depth slope.
        public var slopeScale: Float
        /// Upper bound on the total bias. 0 = unbounded.
        public var clamp: Float
        public init(constant: Float, slopeScale: Float, clamp: Float) {
            self.constant = constant; self.slopeScale = slopeScale; self.clamp = clamp
        }
        /// What second-depth needs: nothing. Named `off` rather than `none` on purpose — this
        /// type is nearly always handled as `ShadowRasterBias?`, where `.none` is Optional's own
        /// case and silently means "derive from the convention", i.e. the exact opposite of
        /// switching the bias off.
        public static let off = ShadowRasterBias(constant: 0, slopeScale: 0, clamp: 0)
        /// **The front-face profile, and the window it sits in the middle of.** Swept on
        /// Daydream's `4000 Sunset` (2026-08-19) against two measurements at once — wall ripple
        /// on a patch of plain wall, and sunlit pixels on a ceilinged pantry floor, the leak
        /// that front-face storage exists to close:
        ///
        /// | slopeScale | wall ripple | wall-foot leak |
        /// |---|---|---|
        /// | second-depth (reference) | 1.53 | 2 565 px |
        /// | 0 | **3.52** | 0 px |
        /// | 1 | 1.57 | — |
        /// | **2** | **1.51** | **0 px** |
        /// | 4 | 1.53 | 528 px |
        /// | 8 | 1.51 | 2 675 px |
        ///
        /// The two failure modes bound it from both sides: under ~1 the ripple survives, at 4
        /// and up the bias lifts the wall's stored depth clear of the floor and the leak comes
        /// back. 2 is the middle of that window, and it lands the wall at the second-depth
        /// reference — flat, with the contact still sealed.
        ///
        /// `constant` is 0 because it bought nothing (100 depth units read 1.50, inside the
        /// run-to-run spread), and `clamp` is 0 because 0.02 was measured NON-binding: the
        /// slopeScale-4 arm is identical to three decimal places with and without it. A guard
        /// that never fires would only suggest a safety this profile does not have — the guard
        /// here is slopeScale itself, and the gate that pins it.
        public static let frontFace = ShadowRasterBias(constant: 0, slopeScale: 2, clamp: 0)
    }

    /// Apply `effectiveShadowRasterBias` to a shadow-pass encoder.
    func applyShadowRasterBias(_ enc: MTLRenderCommandEncoder) {
        let b = effectiveShadowRasterBias
        enc.setDepthBias(b.constant, slopeScale: b.slopeScale, clamp: b.clamp)
    }

    /// The live-lane allocation every host gets without asking. Mirrored by hosts that ration
    /// slices on the CPU (Daydream Home's `ShadowSlotBudget.engineSpotShadowCapacity`).
    public static let spotShadowCapacityDefault = 8

    /// The most slices the atlas will ever hold. Metal allows 2048 array slices, but each one
    /// is a full-scene depth pass per frame (~0.5 ms) plus 1 MiB, so "no limit" still needs an
    /// implementation cap; 64 cones is far past any house's fixture count.
    public static let spotShadowCapacityHardMax = 64

    // ── Phase 2.5 cascade constants ───────────────────────────────────────────
    // Three cascades is the sweet spot — four is more "correct" but the extra
    // pass cost (one render pass per cascade) isn't visible at typical scene
    // scale. 2048² per cascade matches what UE / Frostbite use for their inner
    // cascade and gives clean PCF at 3×3.
    static let cascadeCount: Int = 3

    /// Allocate the cascaded shadow depth array. depth32Float gives us plenty
    /// of precision for the comparison test; `.shaderRead` is required so the
    /// lighting kernel can sample it, and `.renderTarget` lets us write into
    /// each slice via a render pass.
    static func makeShadowMap(device: MTLDevice, resolution: Int) throws -> MTLTexture {
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.pixelFormat = .depth32Float
        d.width = resolution
        d.height = resolution
        d.arrayLength = cascadeCount
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        guard let t = device.makeTexture(descriptor: d) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.shadowMap")
        }
        t.label = "Illuminatorama.shadowMap"
        return t
    }

    /// Phase 4.10 — depth atlas for spot light shadows. Same format /
    /// usage as the cascaded shadow map; smaller per-slice resolution
    /// and N slices (one per shadowed spot).
    static func makeSpotShadowAtlas(device: MTLDevice,
                                             resolution: Int,
                                             capacity: Int) throws -> MTLTexture {
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.pixelFormat = .depth32Float
        d.width = resolution
        d.height = resolution
        d.arrayLength = capacity
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        guard let t = device.makeTexture(descriptor: d) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.spotShadowAtlas")
        }
        t.label = "Illuminatorama.spotShadowAtlas"
        return t
    }

    // ── Cascade math helpers ──────────────────────────────────────────────────

    /// Right-handed look-at, depth conventions matching the rest of
    /// IlluminatoramaCamera's matrices.
    static func lookAtRH(eye: SIMD3<Float>,
                                 target: SIMD3<Float>,
                                 up: SIMD3<Float>) -> simd_float4x4 {
        let f = simd_normalize(target - eye)
        let s = simd_normalize(simd_cross(f, up))
        let u = simd_cross(s, f)
        return simd_float4x4(
            SIMD4( s.x,  u.x, -f.x, 0),
            SIMD4( s.y,  u.y, -f.y, 0),
            SIMD4( s.z,  u.z, -f.z, 0),
            SIMD4(-simd_dot(s, eye),
                  -simd_dot(u, eye),
                   simd_dot(f, eye), 1)
        )
    }

    /// Right-handed orthographic projection with depth range [0, 1] —
    /// matches `IlluminatoramaCamera.projectionMatrix`'s convention.
    static func orthoRH(left l: Float, right r: Float,
                                bottom b: Float, top t: Float,
                                near n: Float, far f: Float) -> simd_float4x4 {
        let zRange = f - n
        return simd_float4x4(
            SIMD4(2 / (r - l), 0, 0, 0),
            SIMD4(0, 2 / (t - b), 0, 0),
            SIMD4(0, 0, -1 / zRange, 0),
            SIMD4(-(r + l) / (r - l),
                  -(t + b) / (t - b),
                  -n / zRange, 1)
        )
    }

    /// Right-handed perspective projection with depth range [0, 1] —
    /// matches `IlluminatoramaCamera.projectionMatrix`'s convention. Used
    /// by `updateSpotShadows` to build per-spot light-space VPs.
    static func perspectiveRH(fovY: Float, aspect: Float,
                                       near: Float, far: Float) -> simd_float4x4 {
        let yScale = 1 / tan(fovY * 0.5)
        let xScale = yScale / aspect
        let zRange = far - near
        return simd_float4x4(
            SIMD4(xScale, 0, 0, 0),
            SIMD4(0, yScale, 0, 0),
            SIMD4(0, 0, -far / zRange, -1),
            SIMD4(0, 0, -(far * near) / zRange, 0)
        )
    }
}

// MARK: - Pipeline / resource construction (called from `init`, in the order it lists)

extension IlluminatoramaRenderer {
    /// Depth-only sun/spot shadow pipeline (init order step 1).
    @MainActor static func buildShadowPipeline(into s: ShadowState, device: MTLDevice, library: MTLLibrary) throws {
        // Phase 2.5 — depth-only shadow pipeline. Same vertex layout as the
        // G-buffer pass but no fragment shader and no colour attachments;
        // depth is written into one slice of the cascade array per pass.
        let sdesc = MTLRenderPipelineDescriptor()
        sdesc.label = "Illuminatorama.shadow"
        sdesc.vertexFunction = library.makeFunction(name: "illumi_shadow_vs")
        sdesc.fragmentFunction = nil
        sdesc.depthAttachmentPixelFormat = .depth32Float
        s.shadowPipeline = try device.makeRenderPipelineState(descriptor: sdesc)
    }

    /// Sun cascade shadow map array + spot shadow atlas.
    @MainActor static func buildShadowMaps(into s: ShadowState, device: MTLDevice, sunResolution: Int, spotResolution: Int) throws {
        s.shadowMap = try Self.makeShadowMap(device: device, resolution: sunResolution)
        s.spotShadowAtlas = try Self.makeSpotShadowAtlas(
            device: device,
            resolution: spotResolution,
            capacity: s.spotShadowAtlasCapacity
        )
    }
}
