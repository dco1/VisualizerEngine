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

    static let log = Logger(subsystem: AppLog.subsystem,
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
    struct MeshCacheKey: Hashable {
        let geometryID: ObjectIdentifier
        let elementIndex: Int
    }
    var meshCache: [MeshCacheKey: (kind: IlluminatoramaRenderer.MeshKind,
                                            mesh: IlluminatoramaMesh)] = [:]

    // SCNGeometries the converter has already rejected (GPU-backed meshes,
    // non-triangle primitives, etc). We remember them so the per-frame extract
    // doesn't re-attempt conversion and re-log the same warning every tick.
    // Tracked per-element since one element may be supported (triangles) and
    // another may not (lines/points).
    var unsupportedMeshes: Set<MeshCacheKey> = []

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
    static let meshBuildBudgetSeconds: Double = 0.006
    var meshBuildDeadline: UInt64 = 0
    var meshBuiltThisFrame: Int = 0

    // Phase 4.2 — cache of SCNScene.lightingEnvironment.contents →
    // MTLTexture so we only upload each unique equirect sky once. Set on
    // every extractFrame; cleared with resetCache(). Sentinel `.some(nil)`
    // marks contents the uploader rejected so we don't reattempt.
    var equirectSkyCache: [ObjectIdentifier: MTLTexture?] = [:]

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
    var atlasNeedsReset: Bool = false

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
    static let rtMaxTriangles = 300_000
    /// Phase 4.38 — surface cache uses one micro-card (atlas tile) per triangle,
    /// so the atlas grows with triangle count. Cap it so the cache can't blow
    /// VRAM: at `surfaceCachePerTriTileSize`² per card, 60k tris ≈ a 1500²
    /// RGBA16F atlas (~18 MB). Past the cap the RT soup still works; only the
    /// cache is skipped. The tile is small because a triangle doesn't need 64².
    static let surfaceCacheMaxTriangles = 60_000
    static let surfaceCachePerTriTileSize = 6
    /// Object-space CPU triangle soup per (geometry, element), read once.
    struct RTSoupKey: Hashable { let geom: ObjectIdentifier; let elem: Int }
    var rtCpuTriCache: [RTSoupKey: (positions: [SIMD3<Float>], indices: [UInt32])] = [:]
    /// Lightweight per-frame draw refs gathered during the walk; the expensive
    /// world-space bake only runs when the scene hash changes.
    var rtFrameRefs: [(geom: SCNGeometry, elem: Int,
                               model: simd_float4x4, albedo: SIMD3<Float>,
                               key: RTSoupKey)] = []
    var rtSceneHash: Int = 0
    var rtLastBuiltHash: Int = .min

    /// AAA glass (#60): nodes flagged with `illuminatoramaGlass` collected this
    /// frame, grouped by registered mesh kind for `renderer.glassMeshGroups`.
    /// Instance property (not threaded through `walk`) to keep the walk signature
    /// stable; reset at the top of every `extractFrame`.
    var glassEntriesThisFrame: [(kind: IlluminatoramaRenderer.MeshKind, inst: IlluminatoramaGlassInstance)] = []


    /// Cache of `emission.contents` (image) identity → average linear RGB,
    /// so the per-frame walk doesn't re-rasterise the same emission texture.
    /// Sentinel `.some(nil)` marks contents we couldn't average (non-image).
    var emissionAvgCache: [ObjectIdentifier: SIMD3<Float>?] = [:]
}
