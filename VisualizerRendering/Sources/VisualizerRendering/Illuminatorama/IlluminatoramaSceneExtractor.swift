import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

// ── ILLUMINATORAMA SCENE EXTRACTOR (Phase 2.6) ───────────────────────────────
//
// Walks an `SCNScene`'s node graph each frame and produces the per-frame
// inputs Illuminatorama needs:
//
//   • `[IlluminatoramaRenderer.InstanceRef]` for every visible mesh node,
//     including world-space `modelMatrix`, extracted PBR material, and a
//     stable `MeshKind` keyed by `SCNGeometry` identity.
//   • The directional sun + a list of `IlluminatoramaPointLight` from
//     `SCNLight` nodes (`.directional` + `.omni` only — `.spot` is on the
//     deferred list, see below).
//
// Owns a mesh cache so SCNGeometry → IlluminatoramaMesh conversion happens
// exactly once per unique geometry, even when the host re-extracts every
// frame. The cache is keyed by `ObjectIdentifier(geometry)` so two nodes
// sharing the same SCNGeometry pointer get a single mesh registration.
//
// ── What works (this PR)
//
//   • `.physicallyBased` materials → PBR (metallic / roughness / emission).
//     `.blinn` / `.lambert` get a reasonable fallback (low metallic, derived
//     roughness from shininess).
//   • `SCNGeometry` whose first element is `.triangles` with 16-bit or 32-bit
//     indices — covers `SCNBox`, `SCNSphere`, `SCNFloor`, `SCNCapsule`,
//     `SCNCylinder`, `SCNCone`, `SCNPlane`, and most hand-built geometries.
//   • Per-node hidden / visible state via `SCNNode.isHidden` walks.
//   • Transform animation (`SCNAction`, `SCNNode.transform` mutations)
//     because we read `presentation.worldTransform` each frame.
//
// ── What doesn't (yet)
//
//   • `SCNParticleSystem`: SceneKit-only; particles don't appear in the
//     Illuminatorama image.
//   • `SCNMaterial.shaderModifiers`: SceneKit-only; the deferred lighting
//     pass uses its own fixed BRDF.
//   • Spot lights (`SCNLight.type == .spot`): cube shadow maps aren't
//     implemented; spots are silently dropped.
//   • Skinned / morphed geometry: extractor reads the static geometry
//     buffers, so animation that's baked into bone weights or morphers
//     won't show. Plain transform animation works.
//   • Triangle strips / fans / lines / points: converted to no-op (zero
//     indices) with a log warning. Add the strip-to-list helper if needed.
//
// ── Lifecycle
//
//   let extractor = IlluminatoramaSceneExtractor(device: device, scene: scene)
//   extractor.cameraNode = controller.cameraNode
//   // each tick:
//   extractor.extractFrame(into: renderer)
//
// `extractFrame(into:)` replaces the renderer's `instances`, `pointLights`,
// `directionalLightDirection`, `directionalLightColor`, and `camera` to
// match the current state of the SCNScene. Properties the host wants to
// override (e.g. exposure, IBL knobs) should be set on the renderer
// *after* `extractFrame(into:)` returns.

@MainActor
public final class IlluminatoramaSceneExtractor {

    private static let log = Logger(subsystem: AppLog.subsystem,
                                     category: "illuminatoramaExtractor")

    public let device: MTLDevice
    public weak var scene: SCNScene?
    public weak var cameraNode: SCNNode?

    /// Fallback when the SCNScene has no `.directional` light. Daylight
    /// from above-east, warm-tinted, 3× intensity matches the lab's
    /// default sun.
    public var fallbackDirectionalLightDirection: SIMD3<Float> =
        simd_normalize(SIMD3(0.5, 1.0, 0.3))
    public var fallbackDirectionalLightColor: SIMD3<Float> =
        SIMD3(1.0, 0.97, 0.92) * 3.0
    /// Camera FOV used when the scene's camera node has no `SCNCamera`
    /// attached (or has one in non-perspective mode). Most app scenes set
    /// `fieldOfView = 60`, so this matches.
    public var fallbackCameraFovYDegrees: Float = 60.0

    /// Walked nodes whose `isHidden == true` skip both their own geometry
    /// AND their descendants' geometry. SceneKit's runtime honours this for
    /// rendering, so Illuminatorama should too — otherwise hidden helper
    /// nodes leak into the deferred image.
    public var respectHidden: Bool = true

    /// When `true`, read `node.worldTransform` (the authored model transform)
    /// instead of `node.presentation.worldTransform` (the render-server
    /// transform). Use this when the host hides geometry nodes from SceneKit
    /// so its render server doesn't double-draw them — SceneKit skips
    /// `presentation` updates for hidden subtrees, so `presentation.worldTransform`
    /// can be stale or identity. Authored writes made inside
    /// `SCNTransaction(animationDuration: 0, disableActions: true)` are
    /// reflected in `worldTransform` immediately and are always safe to read.
    public var useModelTransform: Bool = false

    /// Multiplier on extracted SCNLight intensity (lumens → linear HDR is
    /// a rough conversion; this knob lets the host re-balance after
    /// switching to Illuminatorama's PBR pipeline). 1.0 ≈ SceneKit-ish.
    public var lightIntensityScale: Float = 1.0 / 1000.0

    // ── Emissive-as-light (Phase 4.27) ─────────────────────────────────────────
    //
    // The deferred direct-lighting pass has no "emissive surface emits light"
    // term — emissive geometry is drawn bright but illuminates nothing. SCN
    // scenes lean on this heavily (Pizza+'s oven coils, neon signs, glowing
    // rails ARE the key light), so they read flat through the overlay. As an
    // approximation, after the walk we synthesise a point light at the centroid
    // of every strongly-emissive instance, coloured by its emission. Crude
    // (a point at the node origin, no occlusion, no area shape) but turns
    // "glowing thing lights nothing" into "glowing thing lights its
    // surroundings", which is the dominant gap for emissive-lit scenes.
    // DDGI-with-mesh-proxies is the correct, occluded follow-on.

    /// Emit a synthesised light only when the instance's peak emission channel
    /// exceeds this. Keeps faintly-emissive surfaces (subtle rim glow) from
    /// spawning lights.
    public var emissiveLightThreshold: Float = 0.6
    /// Scalar on emission → point-light colour (pre-multiplied intensity).
    public var emissiveLightGain: Float = 1.2
    /// Falloff radius (m) of a synthesised emissive light.
    public var emissiveLightRadius: Float = 4.0
    /// Upper clamp on a synthesised light's peak colour channel — guards
    /// against a runaway emissive material producing an absurd light.
    public var emissiveLightMaxColor: Float = 8.0
    /// Cap on synthesised emissive lights per frame (brightest kept). Bounds
    /// the per-pixel point-light loop on scenes with many emissive instances
    /// (a coil/neon strip built from dozens of segments).
    public var maxEmissiveLights: Int = 24
    /// Master switch for emissive-as-light synthesis.
    public var emissiveLightsEnabled: Bool = true
    /// Peak-channel clamp on rendered emission after folding in
    /// `emission.intensity` — caps runaway neon so it doesn't white out the
    /// frame via bloom. Moderate emissive stays under it.
    public var emissiveRenderClamp: Float = 8.0

    // Mesh cache keyed by `(SCNGeometry, elementIndex)`. Phase 4.3
    // extends the original geometry-only key to include the element so
    // multi-element geometries (a pizza with crust/sauce/cheese/pepperoni
    // submeshes, each with its own material) produce one cache entry per
    // submesh. Same MeshKind / mesh table entry is reused across two nodes
    // pointing at the same SCNGeometry.
    private struct MeshCacheKey: Hashable {
        let geometryID: ObjectIdentifier
        let elementIndex: Int
    }
    private var meshCache: [MeshCacheKey: (kind: IlluminatoramaRenderer.MeshKind,
                                            mesh: IlluminatoramaMesh)] = [:]

    // SCNGeometries the converter has already rejected (GPU-backed meshes,
    // non-triangle primitives, etc). We remember them so the per-frame extract
    // doesn't re-attempt conversion and re-log the same warning every tick.
    // Tracked per-element since one element may be supported (triangles) and
    // another may not (lines/points).
    private var unsupportedMeshes: Set<MeshCacheKey> = []

    // ── Cold-cache mesh-build budget ────────────────────────────────────────
    // The first frame after attaching to a scene, EVERY SCNGeometry is a cache
    // miss, so `registerMesh` would convert + synthesise normals + synthesise
    // tangents for the whole scene graph inside one synchronous `extractFrame`
    // call on the main thread. For a heavy scene that single tick blocked the
    // run loop for ~9 s (observed in a hang report: main thread parked in
    // `IlluminatoramaMesh.accumulateTangents` via the 60 Hz `tick`).
    //
    // Fix: spend at most `meshBuildBudgetSeconds` of wall-clock on *new* CPU
    // mesh builds per frame. Once the budget is spent we stop building unseen
    // meshes this tick (the node skips a frame and registers on a later one) —
    // but we always build at least one so a giant scene still fully registers
    // within `meshCount` frames. Net effect: cold-start cost amortises over a
    // few ticks (brief pop-in at scene load) instead of one multi-second hang.
    // Cheap paths (pre-registered GPU handles / descriptors — no CPU synth) are
    // intentionally NOT budgeted.
    private static let meshBuildBudgetSeconds: Double = 0.006
    private var meshBuildDeadline: UInt64 = 0
    private var meshBuiltThisFrame: Int = 0

    // Phase 4.2 — cache of SCNScene.lightingEnvironment.contents →
    // MTLTexture so we only upload each unique equirect sky once. Set on
    // every extractFrame; cleared with resetCache(). Sentinel `.some(nil)`
    // marks contents the uploader rejected so we don't reattempt.
    private var equirectSkyCache: [ObjectIdentifier: MTLTexture?] = [:]

    public init(device: MTLDevice, scene: SCNScene? = nil) {
        self.device = device
        self.scene = scene
    }

    /// Drop the mesh cache. Call when the scene's geometries have been
    /// replaced wholesale (e.g. scene-reload). Per-node deletions are fine
    /// without flushing because the cache only holds weak structural data
    /// — stale entries cost VRAM but don't break correctness.
    public func resetCache() {
        meshCache.removeAll(keepingCapacity: true)
        unsupportedMeshes.removeAll(keepingCapacity: true)
        equirectSkyCache.removeAll(keepingCapacity: true)
        emissionAvgCache.removeAll(keepingCapacity: true)
        atlasNeedsReset = true
    }

    /// Phase 4.0 — set on `resetCache()`, consumed at the start of the
    /// next `extractFrame(into:)`. We can't reset the atlas inside
    /// `resetCache()` directly because the atlas lives on the renderer
    /// and `resetCache` doesn't have a renderer reference. Per-extract
    /// reset keeps the API surface unchanged.
    private var atlasNeedsReset: Bool = false

    /// One-shot per-frame extraction. Reads the scene graph and writes
    /// directly into the renderer.
    ///
    /// Mesh registration is push-driven from inside `registerMesh` — the
    /// first frame a previously-unseen SCNGeometry is encountered, the
    /// extractor converts it AND calls `renderer.setMesh` on the same
    /// frame, so the G-buffer pass picks it up immediately instead of
    /// missing one frame.
    // ── RT geometry from extracted scene (generalises the room-only path) ──
    /// Hard cap on baked scene triangles so a pathological mesh can't blow the
    /// AS build time / memory. Scenes past this skip the tail (RT degrades to
    /// partial coverage rather than hanging).
    private static let rtMaxTriangles = 300_000
    /// Phase 4.38 — surface cache uses one micro-card (atlas tile) per triangle,
    /// so the atlas grows with triangle count. Cap it so the cache can't blow
    /// VRAM: at `surfaceCachePerTriTileSize`² per card, 60k tris ≈ a 1500²
    /// RGBA16F atlas (~18 MB). Past the cap the RT soup still works; only the
    /// cache is skipped. The tile is small because a triangle doesn't need 64².
    private static let surfaceCacheMaxTriangles = 60_000
    private static let surfaceCachePerTriTileSize = 6
    /// Object-space CPU triangle soup per (geometry, element), read once.
    private struct RTSoupKey: Hashable { let geom: ObjectIdentifier; let elem: Int }
    private var rtCpuTriCache: [RTSoupKey: (positions: [SIMD3<Float>], indices: [UInt32])] = [:]
    /// Lightweight per-frame draw refs gathered during the walk; the expensive
    /// world-space bake only runs when the scene hash changes.
    private var rtFrameRefs: [(geom: SCNGeometry, elem: Int,
                               model: simd_float4x4, albedo: SIMD3<Float>,
                               key: RTSoupKey)] = []
    private var rtSceneHash: Int = 0
    private var rtLastBuiltHash: Int = .min

    /// AAA glass (#60): nodes flagged with `illuminatoramaGlass` collected this
    /// frame, grouped by registered mesh kind for `renderer.glassMeshGroups`.
    /// Instance property (not threaded through `walk`) to keep the walk signature
    /// stable; reset at the top of every `extractFrame`.
    private var glassEntriesThisFrame: [(kind: IlluminatoramaRenderer.MeshKind, inst: IlluminatoramaGlassInstance)] = []

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
    private func finalizeRTGeometry(into renderer: IlluminatoramaRenderer) {
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
    private func effectiveWorldTransform(of node: SCNNode) -> SCNMatrix4 {
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

    // ── Emissive-as-light synthesis (Phase 4.27) ───────────────────────────────

    /// Append a synthesised point light for each strongly-emissive instance.
    /// Position is the instance's world-space origin (model matrix translation
    /// — a fair proxy for a glowing object's centre); colour is its emission ×
    /// `emissiveLightGain`. When more than `maxEmissiveLights` qualify, only the
    /// brightest are kept so the per-pixel point-light loop stays bounded.
    private func synthesiseEmissiveLights(
        from instances: [IlluminatoramaRenderer.InstanceRef],
        into pointLights: inout [IlluminatoramaPointLight]
    ) {
        var lights: [IlluminatoramaPointLight] = []
        for inst in instances {
            // `lightEmission` is the effective radiance for lighting — solid
            // emission colour OR an emission texture's average, × intensity.
            // (Phase 4.27a; texture-driven glow lights the scene through this.)
            let e = inst.lightEmission
            // Guard against non-finite emission (a material that resolved to
            // NaN/Inf) — a bad light position/colour can abort the render.
            guard e.x.isFinite, e.y.isFinite, e.z.isFinite else { continue }
            let peak = max(e.x, max(e.y, e.z))
            guard peak >= emissiveLightThreshold else { continue }
            let m = inst.data.modelMatrix
            let pos = SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
            guard pos.x.isFinite, pos.y.isFinite, pos.z.isFinite else { continue }
            // Clamp the pre-multiplied colour so a runaway emissive value
            // can't produce an absurd light that blows out the composite.
            let scaled = e * emissiveLightGain
            let cMax = max(scaled.x, max(scaled.y, scaled.z))
            let color = cMax > emissiveLightMaxColor
                ? scaled * (emissiveLightMaxColor / cMax)
                : scaled
            lights.append(IlluminatoramaPointLight(
                position: pos,
                radius: emissiveLightRadius,
                color: color))
        }
        if lights.count > maxEmissiveLights {
            // Luminance-sort descending; keep the brightest N.
            lights.sort { a, b in
                let la = 0.2126 * a.color.x + 0.7152 * a.color.y + 0.0722 * a.color.z
                let lb = 0.2126 * b.color.x + 0.7152 * b.color.y + 0.0722 * b.color.z
                return la > lb
            }
            lights.removeLast(lights.count - maxEmissiveLights)
        }
        pointLights.append(contentsOf: lights)
    }

    // ── Tree walk ─────────────────────────────────────────────────────────────

    private func walk(node: SCNNode,
                      parentHidden: Bool,
                      renderer: IlluminatoramaRenderer,
                      instances: inout [IlluminatoramaRenderer.InstanceRef],
                      pointLights: inout [IlluminatoramaPointLight],
                      spotLights: inout [IlluminatoramaSpotLight],
                      areaLights: inout [IlluminatoramaAreaLight],
                      directionalDir: inout SIMD3<Float>?,
                      directionalColor: inout SIMD3<Float>?,
                      extraDirectionals: inout [IlluminatoramaDirectionalLight],
                      ambientFromAmbientLights: inout SIMD3<Float>) {
        let hidden = parentHidden || (respectHidden && node.isHidden)
        if !hidden {
            extractGeometry(node: node, renderer: renderer, into: &instances)
            extractLight(node: node,
                         pointLights: &pointLights,
                         spotLights: &spotLights,
                         areaLights: &areaLights,
                         directionalDir: &directionalDir,
                         directionalColor: &directionalColor,
                         extraDirectionals: &extraDirectionals,
                         ambientFromAmbientLights: &ambientFromAmbientLights)
        }
        // Continue walking even hidden subtrees? No — SceneKit treats a
        // hidden ancestor as hiding the whole subtree, and walking through
        // a hidden helper rig (which often contains thousands of cosmetic
        // children) would balloon the instance count for no visible gain.
        if hidden { return }
        for child in node.childNodes {
            walk(node: child,
                 parentHidden: hidden,
                 renderer: renderer,
                 instances: &instances,
                 pointLights: &pointLights,
                 spotLights: &spotLights,
                 areaLights: &areaLights,
                 directionalDir: &directionalDir,
                 directionalColor: &directionalColor,
                 extraDirectionals: &extraDirectionals,
                 ambientFromAmbientLights: &ambientFromAmbientLights)
        }
    }

    private func extractGeometry(node: SCNNode,
                                  renderer: IlluminatoramaRenderer,
                                  into instances: inout [IlluminatoramaRenderer.InstanceRef]) {
        guard let geometry = node.geometry else { return }
        let world = effectiveWorldTransform(of: node)
        let modelMatrix = simdFromSCN(world)

        // AAA glass (#60): a node flagged `illuminatoramaGlass` is routed into the
        // renderer's TLAS glass path (true ray-traced refraction) instead of the
        // deferred opaque pipeline — which can't render transparency honestly (the
        // "transparent PBR pane renders black" gotcha). Register each element's
        // mesh and emit a glass instance; the node never reaches the opaque loop.
        if let glassMat = node.illuminatoramaGlass {
            for elementIndex in 0..<max(1, geometry.elements.count) {
                guard let entry = registerMesh(geometry, elementIndex: elementIndex,
                                               into: renderer) else { continue }
                glassEntriesThisFrame.append((entry.kind,
                    IlluminatoramaGlassInstance(modelMatrix: modelMatrix, material: glassMat)))
            }
            return
        }

        // Phase 4.3 — walk every element / submesh. SCNGeometry.material(at:)
        // wraps SCN's materials array indexing AND its 1-material fallback
        // rule: when `materials.count == 1` it's reused across every
        // element, which is the common case for hand-built meshes; when
        // `materials.count >= elements.count` each element gets its
        // matching material; otherwise we get nil for the out-of-range
        // tail and fall through to the default-material branch.
        // Host-owned GPU particle buffers (Foam spray, Firework bursts) are
        // NOT discovered here — they're published into `SimEngine.particleFields`
        // by their renderer-shim and fed to the renderer in `extractFrame`
        // (filtered to the active scene). The deferred opaque pipeline can't
        // draw `.point` + additive geometry anyway, so such geometry that
        // reaches the walk just falls through the unsupported-mesh path.
        let elementCount = max(1, geometry.elements.count)
        let materials = geometry.materials
        for elementIndex in 0..<elementCount {
            let scnMaterial: SCNMaterial?
            if elementIndex < materials.count {
                scnMaterial = materials[elementIndex]
            } else if materials.count == 1 {
                scnMaterial = materials[0]
            } else {
                scnMaterial = nil
            }
            // Phase 4.18 — skip materials that the deferred opaque
            // pipeline can't reproduce honestly. Three buckets, all of
            // which read as solid-block artefacts when forced through
            // the G-buffer pass:
            //
            //   • `blendMode == .add` — additive emissive volumes
            //     (CityStreet's manhole light beam, every glow shell
            //     scene authors layer over an emissive light source).
            //     These rely on SCN's transparent forward pass adding
            //     to the framebuffer; the deferred pipeline writes
            //     opaque to the G-buffer and the cone shows up as a
            //     solid red wall blocking the rest of the scene.
            //   • `transparency < 1` AND `writesToDepthBuffer == false`
            //     — translucent volumes (water shells, fog cards,
            //     stylised glow domes) that depend on alpha compositing
            //     SCN does for free. Deferred opaque can't fake it.
            //
            // Skipping is the honest fix until Illuminatorama gains a
            // proper transparent/forward pass. Scene authors can opt
            // a node back in via `illuminatoramaOverride` if they
            // want the deferred pass to render it as solid PBR
            // regardless (e.g. by overriding emission to the intended
            // tint).
            if let m = scnMaterial {
                let isAdditive = (m.blendMode == .add)
                let isAlphaVolume = (m.transparency < 0.999
                                     && !m.writesToDepthBuffer)
                if isAdditive || isAlphaVolume {
                    if node.illuminatoramaOverride == nil { continue }
                }
            }
            guard let entry = registerMesh(geometry, elementIndex: elementIndex,
                                            into: renderer) else { continue }
            var material = extractMaterial(scnMaterial, renderer: renderer)
            // Phase 4.14 — per-node material overrides. Lets a scene
            // share one SCNGeometry archetype across many SCNNodes (huge
            // perf win via Phase 4.12 instancing) while keeping per-
            // instance colour / metallic / roughness / emission variety
            // via lightweight `SCNNode.userData` keys. The host stashes
            // overrides on each node; extractor applies them here on
            // top of the geometry's base material.
            applyNodeOverrides(node: node, to: &material)
            var inst = IlluminatoramaInstance(
                modelMatrix: modelMatrix,
                albedo: material.albedo,
                metallic: material.metallic,
                roughness: material.roughness,
                emission: material.emission,
                albedoTextureSlice: material.albedoTextureSlice,
                metallicTextureSlice: material.metallicTextureSlice,
                roughnessTextureSlice: material.roughnessTextureSlice,
                normalTextureSlice: material.normalTextureSlice,
                emissionTextureSlice: material.emissionTextureSlice,
                emissionIntensity: material.emissionIntensity
            )
            // Raster-only node (#60 item 7): keeps its G-buffer draw, gets
            // TLAS mask 0 at AS rebuild (its RT twin is supplied elsewhere,
            // e.g. a registered curve set).
            if node.illuminatoramaOverride?.rtExclude == true { inst.rtExclude = 1 }
            instances.append(.init(meshKind: entry.kind, data: inst,
                                   lightEmission: material.lightEmission))

            // RT soup: gather a lightweight ref + fold a cheap scene hash. The
            // world-space vertex bake is deferred to `finalizeRTGeometry` and
            // only runs when this hash changes (static scenes build the AS once).
            // Normally skipped when the renderer can build a TLAS (RT hardware) —
            // the renderer instances its own meshes, no CPU soup needed. BUT the
            // surface cache (Phase 4.38) rides the soup primitive AS, so when it's
            // enabled we DO gather the soup even on TLAS hardware.
            if renderer.buildRTFromExtractedScene && renderer.rtSupported
                && (!renderer.rtTLASSupported || renderer.surfaceCacheEnabled) {
                let key = RTSoupKey(geom: ObjectIdentifier(geometry), elem: elementIndex)
                rtFrameRefs.append((geom: geometry, elem: elementIndex,
                                    model: modelMatrix, albedo: material.albedo, key: key))
                // Fold translation AND the rotation/scale columns so an object
                // spinning or scaling in place (origin unchanged) still dirties
                // the hash and triggers a rebuild — not just one that moves.
                let c0 = modelMatrix.columns.0, c1 = modelMatrix.columns.1
                let t = modelMatrix.columns.3
                rtSceneHash = rtSceneHash &* 31 &+ key.hashValue
                rtSceneHash = rtSceneHash &+ Int((t.x * 97 + t.y * 89 + t.z * 83
                    + c0.x * 53 + c0.y * 47 + c0.z * 43
                    + c1.x * 41 + c1.y * 37 + c1.z * 31) * 64)
            }
        }
    }

    private func extractLight(node: SCNNode,
                              pointLights: inout [IlluminatoramaPointLight],
                              spotLights: inout [IlluminatoramaSpotLight],
                              areaLights: inout [IlluminatoramaAreaLight],
                              directionalDir: inout SIMD3<Float>?,
                              directionalColor: inout SIMD3<Float>?,
                              extraDirectionals: inout [IlluminatoramaDirectionalLight],
                              ambientFromAmbientLights: inout SIMD3<Float>) {
        guard let light = node.light else { return }
        let intensity = Float(light.intensity) * lightIntensityScale
        let color = (scnContentsToRGB(light.color) ?? SIMD3(repeating: 1)) * intensity
        switch light.type {
        case .directional:
            // SCNLight directional points down -Z in its own space (same as
            // SCNCamera). World direction TO the light = +Z of the node.
            let world = effectiveWorldTransform(of: node)
            let dirToLight = simd_normalize(SIMD3<Float>(-Float(world.m31),
                                                         -Float(world.m32),
                                                         -Float(world.m33)))
            // First directional becomes the shadow-casting primary —
            // the deferred pipeline has one cascade rig and we hand it
            // to the first one we see (SCN convention is "key first").
            // #60 task 5: subsequent directionals (fill / back) are carried
            // as REAL directional lights in `extraDirectionals` and shaded
            // with the same NdotL + GGX-specular BRDF the primary uses (full
            // colour, not the old 0.5× ambient fold). They don't get an
            // individual cascade rig — SCN fill/back lights ship
            // `castsShadow = false`, so this is the complete best-version for
            // real 3-point rigs. (A scene with >1 *shadow-casting* directional
            // would need an N-cascade rig — declared out of scope; no current
            // scene has one.)
            if directionalDir == nil {
                directionalDir = dirToLight
                directionalColor = color
            } else {
                extraDirectionals.append(
                    IlluminatoramaDirectionalLight(dir: dirToLight, color: color))
            }
        case .omni:
            let world = effectiveWorldTransform(of: node)
            let pos = SIMD3<Float>(Float(world.m41), Float(world.m42), Float(world.m43))
            // SceneKit's `attenuationEndDistance` is the cutoff; map it to
            // the renderer's `radius`. 0 (the SCNLight default) is "no
            // attenuation, infinite radius" — clamp to something finite so
            // the renderer's window function still works.
            let radius = Float(light.attenuationEndDistance > 0
                                 ? light.attenuationEndDistance
                                 : 25.0)
            pointLights.append(.init(position: pos, radius: radius, color: color))
        case .spot:
            // SCNLight `.spot` cones extend along the node's local -Z axis
            // (same convention as `.directional` and `SCNCamera`). Apex
            // sits at the node's world position; the cone opens in the
            // -Z direction. `SCNMatrix4` is column-major, so the third
            // column (m31/m32/m33) is the local +Z axis in world space;
            // light travels along the NEGATIVE of that — pass the
            // negated vector so the lighting kernel's `dot(direction, -L)`
            // gives the cosine relative to the actual cone axis.
            let world = effectiveWorldTransform(of: node)
            let pos = SIMD3<Float>(Float(world.m41), Float(world.m42), Float(world.m43))
            let dirLightTravels = simd_normalize(SIMD3<Float>(-Float(world.m31),
                                                               -Float(world.m32),
                                                               -Float(world.m33)))
            // `spotInnerAngle` / `spotOuterAngle` are SceneKit FULL cone
            // angles in degrees. The lighting kernel compares the dot
            // product of the cone axis against the light-to-fragment
            // direction, so we store the cosines of the HALF-angles.
            let deg2rad = Float.pi / 180.0
            let innerCos = cos(Float(light.spotInnerAngle) * 0.5 * deg2rad)
            let outerCos = cos(Float(light.spotOuterAngle) * 0.5 * deg2rad)
            // Same fallback for `attenuationEndDistance == 0` ("infinite")
            // as omni — clamp to something finite for the windowing math.
            let radius = Float(light.attenuationEndDistance > 0
                                 ? light.attenuationEndDistance
                                 : 25.0)
            spotLights.append(.init(position: pos,
                                     direction: dirLightTravels,
                                     innerCone: innerCos,
                                     outerCone: outerCos,
                                     color: color,
                                     radius: radius))
        case .area:
            // #60 task 5 — closed-form rectangular area light, retiring the 4.24
            // five-spot approximation (one centre + four corner spots, which read
            // as five discrete speculars on a glossy surface instead of one
            // elongated area reflection). The node's local +X / +Y axes span the
            // emitting plane; `areaExtents.xy` are the half-width / half-height in
            // metres (centre-relative). We pass the half-EDGE vectors `ex` / `ey`
            // (unit axis × half-extent) so the kernel's corners are simply
            // `centre ± ex ± ey`; the diffuse term is then the exact polygon
            // clamped-cosine integral (LTC with M = identity) and the specular a
            // most-representative-point sample.
            let world = effectiveWorldTransform(of: node)
            let localX = simd_normalize(SIMD3<Float>(Float(world.m11),
                                                       Float(world.m12),
                                                       Float(world.m13)))
            let localY = simd_normalize(SIMD3<Float>(Float(world.m21),
                                                       Float(world.m22),
                                                       Float(world.m23)))
            let extents = light.areaExtents
            let halfX: Float = max(0.02, extents.x)
            let halfY: Float = max(0.02, extents.y)
            let centre = SIMD3<Float>(Float(world.m41),
                                        Float(world.m42),
                                        Float(world.m43))
            let radius = Float(light.attenuationEndDistance > 0
                                 ? light.attenuationEndDistance
                                 : 25.0)
            // SCN `.area` lights emit from one face (the local −Z side); the
            // kernel's one-sided gate uses the +normal = cross(ex, ey) face, so
            // order the edges to make that normal point the way the light travels.
            areaLights.append(.init(center: centre,
                                     ex: localX * halfX,
                                     ey: localY * halfY,
                                     color: color,
                                     radius: radius,
                                     twoSided: false))
        case .ambient:
            // Genuine SCN ambient lights → the scene-wide ambient supplement.
            // They illuminate all surfaces equally (no NdotL). The 0.5× scale
            // keeps the unmodulated `mix(0.4·amb, amb, upness)·albedo` term in
            // the lighting kernel from overshooting. (As of #60 task 5 this term
            // is fed ONLY by `.ambient` lights — secondary directionals now
            // shade with a real BRDF via `extraDirectionals`.)
            ambientFromAmbientLights += color * 0.5
        default:
            // IES, probe — not extracted yet. Skip silently.
            break
        }
    }

    // ── Material extraction ───────────────────────────────────────────────────

    private struct Material {
        var albedo: SIMD3<Float>
        var metallic: Float
        var roughness: Float
        var emission: SIMD3<Float>
        /// Effective emissive radiance for the Phase 4.27 light synthesis —
        /// solid emission colour OR an emission texture's average colour,
        /// each × `emission.intensity`, clamped. Kept separate from
        /// `emission` (the G-buffer term) so texture-driven glow can light
        /// the scene without washing the rendered surface toward white.
        var lightEmission: SIMD3<Float> = .zero
        /// Multiplier on the emission TEXTURE sample (Phase 4.27b) so a
        /// texture-driven glow renders at its tuned `emission.intensity`.
        var emissionIntensity: Float = 1.0
        var albedoTextureSlice: Int32 = -1
        var metallicTextureSlice: Int32 = -1
        var roughnessTextureSlice: Int32 = -1
        var normalTextureSlice: Int32 = -1
        var emissionTextureSlice: Int32 = -1
    }

    /// Cache of `emission.contents` (image) identity → average linear RGB,
    /// so the per-frame walk doesn't re-rasterise the same emission texture.
    /// Sentinel `.some(nil)` marks contents we couldn't average (non-image).
    private var emissionAvgCache: [ObjectIdentifier: SIMD3<Float>?] = [:]

    /// Average colour of an emission texture (NSImage / CGImage), used as a
    /// coarse radiance estimate so texture-driven glow (Pizza's rainbow oven
    /// coils) can synthesise a light. Rasterises the image to 1×1 and reads
    /// the pixel; cached by contents identity. Returns nil for non-image
    /// contents (the caller falls back to the solid-colour path).
    private func emissionTextureAverage(_ contents: Any?) -> SIMD3<Float>? {
        guard let contents = contents else { return nil }
        let key = ObjectIdentifier(contents as AnyObject)
        if let cached = emissionAvgCache[key] { return cached }
        let cg: CGImage?
        if let img = contents as? NSImage {
            cg = Self.cgImage(from: img)
        } else if CFGetTypeID(contents as CFTypeRef) == CGImage.typeID {
            cg = (contents as! CGImage)
        } else {
            cg = nil
        }
        guard let cgImage = cg else {
            emissionAvgCache[key] = .some(nil)
            return nil
        }
        var px: [UInt8] = [0, 0, 0, 0]
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(data: &px, width: 1, height: 1,
                                   bitsPerComponent: 8, bytesPerRow: 4,
                                   space: cs, bitmapInfo: info) else {
            emissionAvgCache[key] = .some(nil)
            return nil
        }
        ctx.interpolationQuality = .medium
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        // Device-RGB bytes are ~sRGB; rough-decode to linear (gamma 2.2) so
        // the radiance estimate is in the same space as the lighting maths.
        let avg = SIMD3<Float>(pow(Float(px[0]) / 255.0, 2.2),
                               pow(Float(px[1]) / 255.0, 2.2),
                               pow(Float(px[2]) / 255.0, 2.2))
        emissionAvgCache[key] = .some(avg)
        return avg
    }

    private func extractMaterial(_ material: SCNMaterial?,
                                  renderer: IlluminatoramaRenderer) -> Material {
        guard let material = material else {
            return Material(albedo: SIMD3(0.8, 0.8, 0.8),
                            metallic: 0.0,
                            roughness: 0.5,
                            emission: .zero)
        }

        // Phase 4.0/4.1 — register image-based material properties as
        // texture-atlas slices. A diffuse image goes into the sRGB albedo
        // atlas; metalness / roughness images go into the linear
        // non-colour atlas. Each registration returns nil when the input
        // isn't an image or the atlas is full — the existing scalar
        // fallback then drives that property.
        //
        // Diagnostic env var `VIZ_ILLUMINATORAMA_NO_TEXTURES=1` bypasses
        // registration so you can A/B the textured vs scalar paths
        // without rebuilding. (Also keyed off the older
        // `VIZ_DISABLE_TEXTURE_ATLAS` name for back-compat with shell
        // scripts.)
        let disableTex =
            ProcessInfo.processInfo.environment["VIZ_ILLUMINATORAMA_NO_TEXTURES"] == "1"
            || ProcessInfo.processInfo.environment["VIZ_DISABLE_TEXTURE_ATLAS"] == "1"
        let albedoSlice: Int32
        let metallicSlice: Int32
        let roughnessSlice: Int32
        let normalSlice: Int32
        let emissionSlice: Int32
        if disableTex {
            albedoSlice = -1
            metallicSlice = -1
            roughnessSlice = -1
            normalSlice = -1
            emissionSlice = -1
        } else {
            albedoSlice = renderer.albedoAtlas.register(contents: material.diffuse.contents) ?? -1
            metallicSlice = renderer.nonColorAtlas.register(contents: material.metalness.contents) ?? -1
            roughnessSlice = renderer.nonColorAtlas.register(contents: material.roughness.contents) ?? -1
            normalSlice = renderer.nonColorAtlas.register(contents: material.normal.contents) ?? -1
            emissionSlice = renderer.albedoAtlas.register(contents: material.emission.contents) ?? -1
        }
        let albedo = scnContentsToRGB(material.diffuse.contents)
                    ?? SIMD3(0.8, 0.8, 0.8)

        // Honour `emission.intensity` (the SCN HDR-emissive knob). The
        // canonical neon-tube/glow recipe is a moderate emission COLOUR ×
        // a high `emission.intensity` (often 2–8); without folding the
        // intensity in, every emissive surface rendered at intensity 1, so
        // glowing things read flat AND the Phase 4.27 emissive-light
        // synthesis (which keys off this scalar) never fired on them.
        //
        // Texture-driven emission (`emission.contents` is an image, e.g.
        // Pizza's rainbow coils) is left to the shader's emission-texture
        // sample — we do NOT inject a white scalar here, which would wash
        // the texture toward white in the G-buffer. Lighting from
        // texture-emissive surfaces needs a per-texture average-radiance
        // estimate; that's a follow-on.
        let emissionIntensity = Float(material.emission.intensity)
        var emission = (scnContentsToRGB(material.emission.contents) ?? .zero)
                     * emissionIntensity
        // Clamp peak emission so an extreme `emission.intensity` (neon signs
        // run 4–8×) can't blow the whole frame to white through bloom +
        // tonemap. Moderate emissive (CityStreet windows ≈ 1–3×) stays under
        // the cap unaffected; only runaway neon is compressed.
        let ePeak = max(emission.x, max(emission.y, emission.z))
        if ePeak > emissiveRenderClamp {
            emission *= (emissiveRenderClamp / ePeak)
        }

        // Phase 4.27a — effective emissive radiance for light synthesis. Use
        // the solid emission colour when there is one; otherwise (texture-
        // driven glow, e.g. Pizza's rainbow coils) fall back to the emission
        // texture's average colour. × intensity, clamped like the rendered
        // term. Kept separate from `emission` so the texture average never
        // washes the rendered G-buffer surface.
        let lightBase = scnContentsToRGB(material.emission.contents)
                     ?? emissionTextureAverage(material.emission.contents)
                     ?? .zero
        var lightEmission = lightBase * emissionIntensity
        let lPeak = max(lightEmission.x, max(lightEmission.y, lightEmission.z))
        if lPeak > emissiveRenderClamp {
            lightEmission *= (emissiveRenderClamp / lPeak)
        }

        // Metallic / roughness: PBR materials carry these as
        // `SCNMaterialProperty.contents` (a number 0–1) or as a texture.
        // Lambert / Blinn carry only shininess; derive a roughness.
        let metallic: Float
        let roughness: Float
        if material.lightingModel == .physicallyBased {
            metallic = scnContentsToScalar(material.metalness.contents) ?? 0.0
            roughness = scnContentsToScalar(material.roughness.contents) ?? 0.5
        } else {
            metallic = 0.0
            // Phong shininess (0 = matte ... ~128 = mirror-ish). Map
            // log-spaced shininess to perceptual roughness, clamping.
            let shininess = Float(material.shininess)
            // shininess=0 → roughness 1; shininess≥128 → roughness 0.05
            let r = 1.0 - min(1.0, shininess / 128.0)
            roughness = max(0.05, r)
        }
        return Material(albedo: albedo,
                        metallic: metallic,
                        roughness: roughness,
                        emission: emission,
                        lightEmission: lightEmission,
                        emissionIntensity: emissionIntensity,
                        albedoTextureSlice: albedoSlice,
                        metallicTextureSlice: metallicSlice,
                        roughnessTextureSlice: roughnessSlice,
                        normalTextureSlice: normalSlice,
                        emissionTextureSlice: emissionSlice)
    }

    // ── Equirect-sky extraction ─────────────────────────────────────────────

    /// Surface the host SCNScene's `lightingEnvironment.contents` as an
    /// `MTLTexture` for `IlluminatoramaRenderer.equirectSky`. Cached per
    /// content identity so we upload each unique image once. Returns nil
    /// when no environment is set or the contents are a type we can't
    /// convert (MTLTexture passed directly is uncommon in app-level
    /// scenes; can be added later).
    ///
    /// Falls back to `scene.background.contents` when `lightingEnvironment`
    /// is unset — many scenes (FloatingFlowers, NapaValley, etc.) only
    /// set the background image and rely on SCN to auto-derive the IBL
    /// from it. Picking up the background as the equirectSky also makes
    /// the rendered sky pixels (depth=1.0 in `illumi_lighting`) match
    /// what the SCN renderer shows, which is the main visible difference
    /// for those scenes.
    // ── Phase 4.14 — per-node material overrides ───────────────────

    private func applyNodeOverrides(node: SCNNode, to material: inout Material) {
        guard let ov = node.illuminatoramaOverride else { return }
        if let v = ov.albedo    { material.albedo    = v }
        if let v = ov.metallic  { material.metallic  = v }
        if let v = ov.roughness { material.roughness = v }
        if let v = ov.emission  {
            material.emission = v
            // Keep the light-synthesis radiance in step with an overridden
            // emission so an override-driven glow also lights the scene.
            material.lightEmission = v
        }
    }

    private func extractEquirectSky(scene: SCNScene) -> MTLTexture? {
        let contents = scene.lightingEnvironment.contents
                    ?? scene.background.contents
        guard let contents = contents else { return nil }
        // Fast-path: already an MTLTexture (some scenes do this for
        // VolumetricCloudRenderer's HDR output). Hand it straight through.
        if let direct = contents as? MTLTexture { return direct }
        // NSImage / CGImage path — same identity-cache trick as the
        // material atlas.
        let key: ObjectIdentifier
        let cg: CGImage
        if let img = contents as? NSImage {
            key = ObjectIdentifier(img)
            if let cached = equirectSkyCache[key] { return cached }
            guard let extracted = Self.cgImage(from: img) else {
                equirectSkyCache[key] = .some(nil)
                return nil
            }
            cg = extracted
        } else if CFGetTypeID(contents as CFTypeRef) == CGImage.typeID {
            let img = contents as! CGImage
            key = ObjectIdentifier(img as AnyObject)
            if let cached = equirectSkyCache[key] { return cached }
            cg = img
        } else {
            return nil
        }
        let texture = Self.uploadEquirect(cgImage: cg, device: device)
        equirectSkyCache[key] = .some(texture)
        return texture
    }

    /// Upload a CGImage as an `rgba16Float` equirect texture, keeping its
    /// source resolution (capped at 4096×2048 to avoid pathological VRAM
    /// hits from arbitrary asset sizes). LDR source values go to half-
    /// floats unchanged — proper HDR (.hdr / .exr loaded via ImageIO with
    /// the float-data flag) would preserve >1 luminance; LDR sources
    /// stay clamped at 1.0. Sufficient for the Plus-scene environments
    /// shipped as 8-bit sRGB equirect PNGs today.
    private static func uploadEquirect(cgImage: CGImage,
                                        device: MTLDevice) -> MTLTexture? {
        let w = min(4096, cgImage.width)
        let h = min(2048, cgImage.height)
        guard w > 0, h > 0 else { return nil }
        let bytesPerRow8 = w * 4
        var buf8 = [UInt8](repeating: 0, count: bytesPerRow8 * h)
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
                       | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = buf8.withUnsafeMutableBytes({ raw -> CGContext? in
            guard let base = raw.baseAddress else { return nil }
            return CGContext(data: base, width: w, height: h,
                             bitsPerComponent: 8, bytesPerRow: bytesPerRow8,
                             space: cs, bitmapInfo: bitmapInfo)
        }) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        // Convert BGRA8 → RGBA32Float. Float16 is unavailable on x86_64 so
        // the buffer uses Float; rgba32Float textures are read identically
        // to rgba16Float in Metal shaders. sRGB → linear via gamma 2.2.
        var linear = [Float](repeating: 0, count: w * h * 4)
        for i in 0..<(w * h) {
            let b = Float(buf8[i * 4 + 0]) / 255.0
            let g = Float(buf8[i * 4 + 1]) / 255.0
            let r = Float(buf8[i * 4 + 2]) / 255.0
            linear[i * 4 + 0] = pow(r, 2.2)
            linear[i * 4 + 1] = pow(g, 2.2)
            linear[i * 4 + 2] = pow(b, 2.2)
            linear[i * 4 + 3] = 1.0
        }
        let d = MTLTextureDescriptor()
        d.textureType = .type2D
        d.pixelFormat = .rgba32Float
        d.width = w
        d.height = h
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: d) else { return nil }
        tex.label = "Illuminatorama.sceneEquirect"
        linear.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            tex.replace(region: MTLRegionMake2D(0, 0, w, h),
                        mipmapLevel: 0,
                        withBytes: base,
                        bytesPerRow: w * 4 * MemoryLayout<Float>.stride)
        }
        return tex
    }

    private static func cgImage(from nsImage: NSImage) -> CGImage? {
        var rect = CGRect(origin: .zero, size: nsImage.size)
        return nsImage.cgImage(forProposedRect: &rect, context: nil, hints: nil)
    }

    // ── Mesh registration ─────────────────────────────────────────────────────

    /// Convert (and cache) the SCNGeometry, and push it into the renderer's
    /// mesh table on the same frame the first instance referencing it
    /// shows up. Returns nil for unsupported geometry shapes — caller
    /// should skip the node rather than render with a placeholder.
    private func registerMesh(_ geometry: SCNGeometry,
                               elementIndex: Int,
                               into renderer: IlluminatoramaRenderer)
        -> (kind: IlluminatoramaRenderer.MeshKind, mesh: IlluminatoramaMesh)? {
        let cacheKey = MeshCacheKey(geometryID: ObjectIdentifier(geometry),
                                     elementIndex: elementIndex)
        if let cached = meshCache[cacheKey] { return cached }
        if unsupportedMeshes.contains(cacheKey) { return nil }

        // Phase 4.8 — GPU-direct path. If the caller pre-registered an
        // IlluminatoramaMeshHandle on the SCNGeometry, use the handle's
        // mesh directly (skipping the CPU-readback path that would
        // silently fail for compute-fed geometry). Single-element only:
        // the handle is per-geometry, so we only attach to element 0.
        // Multi-element compute-fed geometry is rare in this project; the
        // existing CPU path catches all other elements via the loop in
        // extractGeometry, and Phase 4.8 doesn't ship a per-element
        // handle yet.
        if elementIndex == 0, let handle = geometry.illuminatoramaMeshHandle {
            let entry = (kind: handle.kind, mesh: handle.mesh)
            meshCache[cacheKey] = entry
            return entry
        }

        // Phase 4.13a — GPU mesh descriptor path. `DynamicMesh` (and any
        // future compute-fed geometry that publishes one) drops an
        // `IlluminatoramaGPUMeshDescriptor` on its SCNGeometry. The
        // extractor sees it on first encounter, asks the renderer to
        // build an interleaved vertex buffer + register a repack task,
        // and stashes the returned handle so subsequent frames just hit
        // the cache. This is single-element by construction — the
        // descriptor's body + caps are merged into one index buffer at
        // registration.
        if elementIndex == 0, let desc = geometry.illuminatoramaGPUMesh {
            if let handle = renderer.registerGPUMesh(desc) {
                geometry.illuminatoramaMeshHandle = handle
                let entry = (kind: handle.kind, mesh: handle.mesh)
                meshCache[cacheKey] = entry
                return entry
            } else {
                // Allocation failed (out of VRAM, etc.) — fall through
                // and let the CPU readback path try, which will also
                // fail for compute-fed geometry but at least logs once.
                Self.log.warning("registerGPUMesh failed for \(type(of: geometry))")
            }
        }

        // Cold-cache budget gate. This is the expensive CPU path (vertex read
        // + normal/tangent synthesis). If we've already built ≥1 new mesh this
        // frame AND blown the time budget, defer this one to a later tick so a
        // cold-start scene can't park the main thread for seconds. Returning
        // nil here (without inserting into `unsupportedMeshes`) makes the caller
        // skip the node for this frame only; it retries next frame.
        if meshBuiltThisFrame > 0
            && DispatchTime.now().uptimeNanoseconds > meshBuildDeadline {
            return nil
        }

        guard let mesh = IlluminatoramaMesh.from(scnGeometry: geometry,
                                                  elementIndex: elementIndex,
                                                  device: device) else {
            // Log once per (geometry, element); record so the next tick skips silently.
            Self.log.debug("Skipping unsupported SCNGeometry element: \(type(of: geometry))[\(elementIndex)]")
            unsupportedMeshes.insert(cacheKey)
            return nil
        }

        // The MeshKind string is `<TypeName>#<hexPointer>#elem<N>` so it
        // stays stable for a given geometry+element pair, and different
        // elements of the same geometry resolve to different MeshKinds.
        let addr = String(unsafeBitCast(cacheKey.geometryID, to: Int.self), radix: 16)
        let key = "\(type(of: geometry))#\(addr)#elem\(elementIndex)"
        let kind = IlluminatoramaRenderer.MeshKind.custom(key)
        // Phase 4.21 — run normal/tangent synthesis on the GPU (one-shot) for
        // geometry whose source shipped neither. Replaces the CPU passes that
        // used to run on the main thread inside `IlluminatoramaMesh.from` and
        // hung the run loop on a cold-cache scene. No-op when both channels
        // were present on the source.
        renderer.synthesiseMeshGeometry(mesh)
        meshCache[cacheKey] = (kind: kind, mesh: mesh)
        renderer.setMesh(kind, mesh)
        meshBuiltThisFrame += 1
        return (kind, mesh)
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private func simdFromSCN(_ m: SCNMatrix4) -> simd_float4x4 {
        // SCNMatrix4 fields are CGFloat / Double on macOS but the
        // semantics match simd_float4x4 column-major.
        return simd_float4x4(
            SIMD4(Float(m.m11), Float(m.m12), Float(m.m13), Float(m.m14)),
            SIMD4(Float(m.m21), Float(m.m22), Float(m.m23), Float(m.m24)),
            SIMD4(Float(m.m31), Float(m.m32), Float(m.m33), Float(m.m34)),
            SIMD4(Float(m.m41), Float(m.m42), Float(m.m43), Float(m.m44))
        )
    }
}

// MARK: - Color/contents helpers (file-private, free functions so they
//        can be reused by IlluminatoramaMesh.from(scnGeometry:) too).

@MainActor
func scnColorToRGB(_ color: NSColor) -> SIMD3<Float> {
    // Convert to a sRGB-ish device colourspace so the channel reads are
    // consistent. `usingColorSpace(.sRGB)` returns nil for some non-RGB
    // input colours (named, calibrated white, etc.); fall back to mid-grey
    // rather than letting a force-unwrap crash on a content edge case.
    let rgb = color.usingColorSpace(.sRGB) ?? .gray
    // sRGB-encoded components → LINEAR. Illuminatorama shades in linear and
    // tonemaps to sRGB on store; the texture-atlas, emission-average, and
    // environment paths all decode with `pow(x, 2.2)` (see
    // emissionTextureAverage / uploadEnvironment). Solid-colour albedo/
    // emission/light-colour used to skip this decode, so a SceneKit material's
    // sRGB diffuse was fed in as if linear — translated scenes washed out
    // (green felt → khaki). Decode here so colour matches the textured paths.
    return SIMD3(pow(Float(rgb.redComponent),   2.2),
                 pow(Float(rgb.greenComponent), 2.2),
                 pow(Float(rgb.blueComponent),  2.2))
}

@MainActor
func scnContentsToRGB(_ contents: Any?) -> SIMD3<Float>? {
    if let color = contents as? NSColor {
        return scnColorToRGB(color)
    }
    if let cf = contents, CFGetTypeID(cf as CFTypeRef) == CGColor.typeID {
        let cgColor = cf as! CGColor
        if let ns = NSColor(cgColor: cgColor) {
            return scnColorToRGB(ns)
        }
    }
    if let num = contents as? NSNumber {
        let v = Float(truncating: num)
        return SIMD3(repeating: v)
    }
    // Image / texture contents — Phase 2.6 leaves these unhandled
    // (sampling them on the GPU needs the renderer to take a texture per
    // instance, which is Phase 4-level work). Returning nil signals
    // "fall back to default" at the call site.
    return nil
}

func scnContentsToScalar(_ contents: Any?) -> Float? {
    if let num = contents as? NSNumber {
        return Float(truncating: num)
    }
    if let color = contents as? NSColor {
        // Some scenes pack a single-channel value into a colour. Take the
        // red channel as the value.
        let rgb = color.usingColorSpace(.sRGB) ?? .gray
        return Float(rgb.redComponent)
    }
    return nil
}
