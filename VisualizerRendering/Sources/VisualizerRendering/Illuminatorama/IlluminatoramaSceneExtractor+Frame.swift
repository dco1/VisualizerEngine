import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

extension IlluminatoramaSceneExtractor {

    public func extractFrame(into renderer: IlluminatoramaRenderer) {
        guard let scene = scene else { return }
        glassEntriesThisFrame.removeAll(keepingCapacity: true)
        // Open this frame's cold-mesh-build budget window (see the property
        // docs above). `registerMesh` consumes it for new CPU mesh builds.
        meshBuildDeadline = DispatchTime.now().uptimeNanoseconds
            &+ UInt64(Self.meshBuildBudgetSeconds * 1_000_000_000)
        meshBuiltThisFrame = 0
        let buildRT = renderer.buildRTFromExtractedScene && renderer.rtSupported
        if buildRT { rtFrameRefs.removeAll(keepingCapacity: true); rtSceneHash = 0 }
        // Apply any deferred atlas reset queued by `resetCache()` —
        // dropping the per-image slice mappings so this scene's textures
        // get fresh slots starting at 0. Both per-material atlases share
        // the same reset trigger.
        if atlasNeedsReset {
            renderer.albedoAtlas.reset()
            renderer.nonColorAtlas.reset()
            atlasNeedsReset = false
        }

        var instances: [IlluminatoramaRenderer.InstanceRef] = []
        var pointLights: [IlluminatoramaPointLight] = []
        var spotLights: [IlluminatoramaSpotLight] = []
        var areaLights: [IlluminatoramaAreaLight] = []
        var directionalDir: SIMD3<Float>? = nil
        var directionalColor: SIMD3<Float>? = nil
        // The first directional becomes the shadow-casting primary (cascade
        // rig). SCN scenes frequently ship key + fill + back as three
        // directionals (Eggs is the canonical example). #60 task 5 retires the
        // 4.20 "ambient fold" that collapsed the secondaries (past the first)
        // into a flat hemispheric ambient supplement — which threw away their
        // direction (no NdotL gradient) and specular. They now flow into
        // `extraDirectionals` and shade with a real BRDF (no shadow; SCN fill/
        // back lights are `castsShadow = false`). `ambientFromAmbientLights`
        // now holds ONLY genuine SCN `.ambient` lights (uniform, no NdotL).
        var ambientFromAmbientLights: SIMD3<Float> = .zero
        var extraDirectionals: [IlluminatoramaDirectionalLight] = []

        walk(node: scene.rootNode,
             parentHidden: false,
             renderer: renderer,
             instances: &instances,
             pointLights: &pointLights,
             spotLights: &spotLights,
             areaLights: &areaLights,
             directionalDir: &directionalDir,
             directionalColor: &directionalColor,
             extraDirectionals: &extraDirectionals,
             ambientFromAmbientLights: &ambientFromAmbientLights)

        renderer.instances = instances

        // AAA glass (#60): route the collected glass nodes into the renderer's
        // unified TLAS glass path. Group by registered mesh kind so one draw per
        // mesh covers all its instances. Setting `rtGlassEnabled` makes the
        // renderer add the glass to the TLAS and trace it — so an extracted scene
        // gets real refraction with zero per-scene plumbing (just the node flag).
        if !glassEntriesThisFrame.isEmpty {
            var byKind: [IlluminatoramaRenderer.MeshKind: [IlluminatoramaGlassInstance]] = [:]
            for e in glassEntriesThisFrame { byKind[e.kind, default: []].append(e.inst) }
            renderer.glassMeshGroups = byKind.map { .init(kind: $0.key, instances: $0.value) }
            renderer.rtGlassEnabled = true
        } else {
            renderer.glassMeshGroups = []
        }

        // Phase 4.27 — synthesise point lights from strongly-emissive instances
        // so glowing surfaces illuminate the scene (see the tunables above).
        if emissiveLightsEnabled {
            synthesiseEmissiveLights(from: instances, into: &pointLights)
        }
        renderer.pointLights = pointLights
        renderer.spotLights = spotLights
        renderer.areaLights = areaLights
        renderer.directionalLightDirection = directionalDir ?? fallbackDirectionalLightDirection
        renderer.directionalLightColor    = directionalColor ?? fallbackDirectionalLightColor
        renderer.ambientColor = ambientFromAmbientLights
        renderer.extraDirectionals = extraDirectionals

        // RT-from-extracted-scene: point the RT sun at the scene's key light and
        // hand the sun to RT (zero the deferred directional + cascaded shadows
        // so RT owns it — no double-count). The acceleration structure itself is
        // built by the renderer's TLAS path (instanced, refit per frame) on RT
        // hardware; only when TLAS is unavailable do we fall back to baking the
        // CPU world-space soup here (dirty-rebuilt).
        if buildRT && renderer.rtEnabled {
            // NEGATED on purpose: the RT lighting kernel shades the sun as
            // dot(N, sunDir) (sunDir = to-light), but directionalLightDirection
            // arrives pointing the opposite way for a sun above, so an un-negated
            // value leaves up-facing surfaces with NdotL < 0 — no direct sun, no
            // shadows (proven with an overhead eulerAngles sun on shadowLab).
            // RT-only; the deferred path keeps using directionalLightDirection
            // as-is. See docs/known-issues/illuminatorama-extracted-directional-sun-inverted.md.
            renderer.rtSunDirection = -simd_normalize(renderer.directionalLightDirection)
            renderer.rtSunColor = renderer.directionalLightColor * 2.0
            renderer.directionalLightColor = .zero
            renderer.shadowsEnabled = false
            // When TLAS is available the renderer bakes the AS (and, for the
            // surface cache, the grouped soup + cards) itself in `rebuildRTAccel`
            // — including the cache case as of P1c. Only fall back to the CPU
            // world-space soup here when TLAS is unsupported.
            if !renderer.rtTLASSupported
                && !rtFrameRefs.isEmpty && rtSceneHash != rtLastBuiltHash {
                finalizeRTGeometry(into: renderer)
                rtLastBuiltHash = rtSceneHash
            }
        }

        // Host-owned GPU particle fields (Foam spray, Firework bursts) for the
        // active scene only. Producers register into the process-wide
        // `SimEngine.particleFields`; filtering by this scene's identity keeps
        // an inactive (cached-controller) scene's still-registered fields out.
        renderer.setParticleFields(
            SimEngine.particleFields.sources(forScene: ObjectIdentifier(scene)))

        // Phase 4.2 — pass the SCN scene's tuned IBL through to the renderer
        // so Plus-tier scenes inherit their own warm/cool/HDR environment
        // instead of falling through to the generic procedural sky. When
        // the scene has no `lightingEnvironment`, leave `equirectSky` nil
        // so the renderer's `dummySkyTexture` (the procedural gradient)
        // still kicks in. Any change to the resolved texture identity
        // signals the bake.
        let newSky = extractEquirectSky(scene: scene)
        if newSky !== renderer.equirectSky {
            renderer.equirectSky = newSky
            renderer.markIBLDirty()
        }
        // Phase 4.19 — pass `lightingEnvironment.intensity` through as
        // a multiplier on `renderer.iblIntensity`. SCN treats this as
        // a scalar over the entire IBL contribution; before this,
        // Illuminatorama silently used a fixed `1.0` regardless.
        //
        // CLAMP — many SCN scenes set `intensity` significantly below
        // 1 (Eggs uses 0.24) because SCN's IBL is bright by default
        // and the scene has its own carefully tuned direct lights.
        // Illuminatorama's IBL is more conservative; using the raw
        // value pushes good scenes near-black. Apply a floor at 0.75
        // so the SCN's "I want a quiet IBL" tuning still reads as
        // a meaningful ambient through the deferred pipeline. Scenes
        // tuned for very-bright IBL (intensity > 1) still ride the
        // signal up.
        let envIntensity = Float(scene.lightingEnvironment.intensity)
        if envIntensity > 0 {
            renderer.iblIntensity = max(envIntensity, 0.75)
        }
        updateCamera(on: renderer)
    }

    /// Bake the world-space triangle soup from this frame's draw refs and hand
    /// it to the renderer's RT acceleration structure. Object-space triangles
    /// are cached per (geometry, element); each ref applies its node's world
    /// matrix. Per-triangle geometric normal is the world-space face normal;
    /// albedo is the instance's. Triangles past `rtMaxTriangles` are dropped.
    func finalizeRTGeometry(into renderer: IlluminatoramaRenderer) {
        var positions: [SIMD3<Float>] = []
        var indices: [UInt32] = []
        var triAlbedo: [SIMD3<Float>] = []
        var triNormal: [SIMD3<Float>] = []
        for ref in rtFrameRefs {
            let tris: (positions: [SIMD3<Float>], indices: [UInt32])
            if let cached = rtCpuTriCache[ref.key] {
                tris = cached
            } else if let read = IlluminatoramaMesh.objectTriangles(
                scnGeometry: ref.geom, elementIndex: ref.elem) {
                rtCpuTriCache[ref.key] = read; tris = read
            } else { continue }
            if indices.count / 3 + tris.indices.count / 3 > Self.rtMaxTriangles { break }
            let base = UInt32(positions.count)
            for p in tris.positions {
                let w = ref.model * SIMD4<Float>(p, 1)
                positions.append(SIMD3(w.x, w.y, w.z))
            }
            var t = 0
            while t + 2 < tris.indices.count {
                let i0 = base + tris.indices[t], i1 = base + tris.indices[t + 1], i2 = base + tris.indices[t + 2]
                indices.append(i0); indices.append(i1); indices.append(i2)
                let w0 = positions[Int(i0)], w1 = positions[Int(i1)], w2 = positions[Int(i2)]
                var n = simd_cross(w1 - w0, w2 - w0)
                let len = simd_length(n)
                n = len > 1e-8 ? n / len : SIMD3<Float>(0, 1, 0)
                triNormal.append(n)
                triAlbedo.append(ref.albedo)
                t += 3
            }
        }
        guard !indices.isEmpty else { return }
        renderer.setRTGeometry(positions: positions, indices: indices,
                               triangleAlbedo: triAlbedo, triangleNormal: triNormal)
        Self.log.notice("RT soup baked: \(indices.count / 3) tris from \(self.rtFrameRefs.count) refs")

        // Phase 4.38 (#17 #3, P1b) — surface-cache generalisation. Synthesise one
        // per-triangle micro-card per soup triangle so an ARBITRARY extracted
        // scene gets the multi-bounce radiance cache — not just hand-authored
        // planar rooms. Reuses the exact same atlas/update/sample machinery; the
        // only host work is turning each world-space triangle into a `SurfCard`.
        // Bounded by a triangle cap (the atlas grows per card) and a small tile.
        if renderer.surfaceCacheEnabled {
            let triCount = indices.count / 3
            if triCount <= Self.surfaceCacheMaxTriangles {
                let g = IlluminatoramaRenderer.makePerTriangleSurfaceCards(
                    positions: positions, indices: indices,
                    triangleAlbedo: triAlbedo, triangleNormal: triNormal)
                renderer.setSurfaceCacheCards(cards: g.cards, triCard: g.triCard,
                                              triUVa: g.triUVa, triUVc: g.triUVc,
                                              tileSize: Self.surfaceCachePerTriTileSize)
            } else {
                Self.log.notice("Surface cache skipped: \(triCount) tris > cap \(Self.surfaceCacheMaxTriangles)")
            }
        }
    }

    /// Pull SCNCamera state (position, target, FOV, near/far) into the
    /// renderer's `IlluminatoramaCamera`. Aspect is left alone — the host
    /// owns the output size and so the aspect ratio.
    /// Return the world transform to use for `node`. Respects `useModelTransform`:
    /// when true reads the authored `worldTransform`; otherwise reads the
    /// render-server `presentation.worldTransform`.
    @inline(__always)
    func effectiveWorldTransform(of node: SCNNode) -> SCNMatrix4 {
        useModelTransform ? node.worldTransform : node.presentation.worldTransform
    }

    private func updateCamera(on renderer: IlluminatoramaRenderer) {
        guard let camNode = cameraNode else { return }
        let world = effectiveWorldTransform(of: camNode)
        let pos = SIMD3<Float>(Float(world.m41), Float(world.m42), Float(world.m43))
        // SCNCamera looks down -Z in its own space. World-space forward
        // is the third column negated.
        let forward = simd_normalize(SIMD3<Float>(-Float(world.m31),
                                                  -Float(world.m32),
                                                  -Float(world.m33)))
        let upHint  = simd_normalize(SIMD3<Float>(Float(world.m21),
                                                  Float(world.m22),
                                                  Float(world.m23)))
        let fovYDeg: Float
        let zNear: Float
        let zFar: Float
        if let cam = camNode.camera {
            // SCNCamera.fieldOfView is in degrees on the axis specified by
            // `projectionDirection`. Default is `.vertical`, but a lot of
            // app scenes override to `.horizontal` — and reading a
            // horizontal-axis FOV as if it were vertical squashes the
            // image, producing the "squat" rendering vs the SCN baseline.
            // Convert horizontal → vertical via the current camera aspect
            // when the scene asks for it.
            let scnFovDeg = Float(cam.fieldOfView)
            switch cam.projectionDirection {
            case .vertical:
                fovYDeg = scnFovDeg
            case .horizontal:
                let hFovRad = scnFovDeg * .pi / 180
                let aspect = max(0.0001, renderer.camera.aspect)
                let vFovRad = 2 * atan(tan(hFovRad * 0.5) / aspect)
                fovYDeg = vFovRad * 180 / .pi
            @unknown default:
                fovYDeg = scnFovDeg
            }
            zNear   = Float(cam.zNear)
            zFar    = Float(cam.zFar)
        } else {
            fovYDeg = fallbackCameraFovYDegrees
            zNear   = 0.1
            zFar    = 200.0
        }
        renderer.camera.position = pos
        renderer.camera.target = pos + forward
        renderer.camera.up = upHint
        renderer.camera.fovYRadians = fovYDeg * .pi / 180.0
        renderer.camera.zNear = zNear
        renderer.camera.zFar  = zFar
    }
}
