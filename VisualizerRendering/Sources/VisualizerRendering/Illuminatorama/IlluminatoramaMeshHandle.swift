import Foundation
import Metal
import ObjectiveC
import OSLog
import SceneKit
import VisualizerCore

// ── ILLUMINATORAMA MESH HANDLE (Phase 4.8) ──────────────────────────────────
//
// The Phase 2.6 scene extractor reads CPU-side `SCNGeometrySource` data
// out of an `SCNGeometry` and builds an `IlluminatoramaMesh` from it. That
// path works for static, asset-imported geometry but silently fails for the
// project's compute-fed meshes — `DynamicMesh` (XPBD chains, MLS-MPM
// surfaces), `BarkRenderer`, anything wired through `SimEngine`. Their
// vertex bytes never make a CPU round trip; the SCNGeometry wrapping them
// reports `vectorCount > 0` with an empty `data` blob.
//
// The handle below gives those callers a direct line into Illuminatorama:
//
//   let mesh   = IlluminatoramaMesh(vertexBuffer: vb, indexBuffer: ib, …)
//   let handle = renderer.registerMesh(mesh)
//   scnGeometry.illuminatoramaMeshHandle = handle
//
// From that point on the scene extractor finds the handle via the
// associated-object hook below and binds the caller's MTLBuffers directly
// at draw time — no copy, no readback. When the caller drops its strong
// reference to the handle, deinit removes the entry from the renderer's
// mesh table.
//
// The caller is responsible for the vertex bytes matching
// `IlluminatoramaVertex` layout (position[12] pad[4] normal[12] pad[4]
// uv[8] pad[8] tangent[16] — stride 96). Future work could let callers
// register a custom vertex descriptor; for now the layout is fixed so the
// existing G-buffer pipeline binds without per-mesh state swaps.

@MainActor
public final class IlluminatoramaMeshHandle {

    private static let log = Logger(subsystem: AppLog.subsystem,
                                     category: "illuminatoramaMeshHandle")

    /// Renderer-side identity. Stored in `IlluminatoramaInstance` via the
    /// host-side `InstanceRef`, looked up in the renderer's mesh table.
    public let kind: IlluminatoramaRenderer.MeshKind

    /// The mesh entry the renderer holds. Public so power users can
    /// introspect or repoint buffers; in normal use you just attach the
    /// handle to your `SCNGeometry` and let the extractor pick it up.
    public let mesh: IlluminatoramaMesh

    private weak var renderer: IlluminatoramaRenderer?

    /// Created by `IlluminatoramaRenderer.registerMesh(_:)` only — the
    /// renderer wires up the back-reference so `deinit` can clean up.
    internal init(kind: IlluminatoramaRenderer.MeshKind,
                  mesh: IlluminatoramaMesh,
                  renderer: IlluminatoramaRenderer) {
        self.kind = kind
        self.mesh = mesh
        self.renderer = renderer
    }

    deinit {
        // Strong references to the handle keep the renderer-side entry
        // alive; once they drop, evict the entry so the table doesn't
        // leak across scene reloads. Capture `kind` because `self` is
        // gone by the time the closure body runs.
        let evictKind = kind
        // deinit can be called off the main actor (e.g. from a frame's
        // ARC release); shuttle to MainActor before touching the renderer.
        Task { @MainActor [weak renderer] in
            renderer?.removeMesh(evictKind)
        }
    }
}

// MARK: - GPU mesh descriptor (Phase 4.13a — DynamicMesh bridge)

/// Describes a compute-fed `SCNGeometry`'s GPU vertex buffers so
/// Illuminatorama can repack them into its own interleaved vertex layout
/// without a CPU round-trip. Set on the geometry via
/// `SCNGeometry.illuminatoramaGPUMesh` at construction time (e.g. inside
/// `DynamicMesh.init`); the scene extractor picks it up and asks the
/// renderer to register an Illuminatorama-side mesh + repack task.
///
/// Today supports the `DynamicMesh` pattern: separate position + normal
/// buffers in `packed_float3` (12-byte stride), one or two triangle-list
/// index buffers. UV / tangent are synthesised by the repack kernel
/// (UV = (0,0), tangent = first stable perpendicular to the normal —
/// good enough for hot dogs and any non-tangent-mapped surface).
public struct IlluminatoramaGPUMeshDescriptor {
    public let positionBuffer: MTLBuffer
    public let normalBuffer: MTLBuffer
    /// Bytes per position entry. 12 for `packed_float3`.
    public let positionStride: Int
    /// Bytes per normal entry. 12 for `packed_float3`.
    public let normalStride: Int
    public let vertexCount: Int
    /// Body element (the cylindrical mid-section of a tube, or the only
    /// element of a single-element mesh).
    public let bodyIndexBuffer: MTLBuffer
    public let bodyIndexCount: Int
    public let bodyIndexType: MTLIndexType
    /// Optional second element. For hot-dog-style meshes this is the
    /// hemispherical end-cap fan. `nil` for single-element meshes.
    public let capIndexBuffer: MTLBuffer?
    public let capIndexCount: Int
    public let capIndexType: MTLIndexType
    /// Phase 4.17 — optional per-vertex UV buffer. `nil` falls back to
    /// the synthetic `(0,0)` UV the repack kernel wrote previously,
    /// which is fine for non-textured surfaces (hot dogs, mustard pool)
    /// but kills any texture-mapped asset bridged through this path
    /// (LeafField sprite cards, future textured tubes). Format is
    /// packed `float2`, stride 8 by default. The repack kernel reads
    /// per-vertex into `Vertex.uv` when bound, leaves the synthetic
    /// default when not.
    public let uvBuffer: MTLBuffer?
    public let uvStride: Int
    /// Optional per-vertex RGBA color buffer (stride-16 `float4`/`SIMD4<Float>`).
    /// `nil` → the repack writes identity white (albedo untouched). Non-nil →
    /// the color is multiplied into albedo at shading time. The coin soup uses
    /// this for per-coin DEBUG tints; any instanced GPU mesh can supply it.
    public let colorBuffer: MTLBuffer?
    public let colorStride: Int
    /// Render two-sided (cull `.none` + back-face normal flip). For open /
    /// dynamic surfaces (a marching-cubes fluid that tilts or pours) so the
    /// back side renders instead of going hollow. Default false.
    public let doubleSided: Bool

    public init(positionBuffer: MTLBuffer,
                normalBuffer: MTLBuffer,
                positionStride: Int = 12,
                normalStride: Int = 12,
                vertexCount: Int,
                bodyIndexBuffer: MTLBuffer,
                bodyIndexCount: Int,
                bodyIndexType: MTLIndexType = .uint16,
                capIndexBuffer: MTLBuffer? = nil,
                capIndexCount: Int = 0,
                capIndexType: MTLIndexType = .uint16,
                uvBuffer: MTLBuffer? = nil,
                uvStride: Int = 8,
                colorBuffer: MTLBuffer? = nil,
                colorStride: Int = 16,
                doubleSided: Bool = false) {
        self.colorBuffer = colorBuffer
        self.colorStride = colorStride
        self.doubleSided = doubleSided
        self.positionBuffer = positionBuffer
        self.normalBuffer = normalBuffer
        self.positionStride = positionStride
        self.normalStride = normalStride
        self.vertexCount = vertexCount
        self.bodyIndexBuffer = bodyIndexBuffer
        self.bodyIndexCount = bodyIndexCount
        self.bodyIndexType = bodyIndexType
        self.capIndexBuffer = capIndexBuffer
        self.capIndexCount = capIndexCount
        self.capIndexType = capIndexType
        self.uvBuffer = uvBuffer
        self.uvStride = uvStride
    }
}

// MARK: - SCNGeometry associated-object hook

// The associated-object API only cares about the *address* of this byte
// as a unique key; the value is never read or written. `nonisolated(unsafe)`
// is the documented escape hatch for Swift 6 strict concurrency when the
// global itself is read-only after init — no mutation, no data race.
nonisolated(unsafe) private var illuminatoramaMeshHandleKey: UInt8 = 0
nonisolated(unsafe) private var illuminatoramaGPUMeshKey: UInt8 = 0

public extension SCNGeometry {

    /// Attach an `IlluminatoramaMeshHandle` to this geometry so the scene
    /// extractor uses the caller's pre-built GPU buffers instead of
    /// trying to read SCN's (empty) vertex source. The handle is retained
    /// for the lifetime of the SCNGeometry by Objective-C associated-object
    /// semantics — drop the strong reference on your side and it goes
    /// when the geometry does.
    ///
    /// Setting `nil` clears the association and Illuminatorama reverts to
    /// the CPU-extraction path for this geometry.
    var illuminatoramaMeshHandle: IlluminatoramaMeshHandle? {
        get {
            objc_getAssociatedObject(self, &illuminatoramaMeshHandleKey)
                as? IlluminatoramaMeshHandle
        }
        set {
            objc_setAssociatedObject(self, &illuminatoramaMeshHandleKey,
                                      newValue,
                                      .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        }
    }

    /// Phase 4.13a — declare that this geometry is compute-fed by GPU
    /// buffers in the described layout. The scene extractor uses this on
    /// first encounter to lazily build an Illuminatorama-side
    /// `IlluminatoramaVertex` buffer + repack task, then attaches a
    /// regular `illuminatoramaMeshHandle` so subsequent frames just
    /// dispatch the repack kernel and draw.
    ///
    /// Set on construction (e.g. inside `DynamicMesh.init`); never
    /// changed after. Returning a fresh struct each frame triggers
    /// re-registration, which is wasteful — cache it.
    var illuminatoramaGPUMesh: IlluminatoramaGPUMeshDescriptor? {
        get {
            (objc_getAssociatedObject(self, &illuminatoramaGPUMeshKey)
                as? GPUMeshBox)?.descriptor
        }
        set {
            let box = newValue.map { GPUMeshBox(descriptor: $0) }
            objc_setAssociatedObject(self, &illuminatoramaGPUMeshKey,
                                      box,
                                      .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        }
    }
}

/// Box for the struct descriptor — associated objects only accept class
/// types, so we wrap.
final private class GPUMeshBox {
    let descriptor: IlluminatoramaGPUMeshDescriptor
    init(descriptor: IlluminatoramaGPUMeshDescriptor) { self.descriptor = descriptor }
}

// Host-owned GPU particle buffers (Foam spray, Firework bursts) are no longer
// bridged through an SCNGeometry associated object. They publish a
// `ParticleFieldSource` into the shared `SimEngine.particleFields` registry,
// which the Illuminatorama extractor reads each frame (filtered to the active
// scene). See `SimEngine.swift` and `ParticleFieldRegistry`.

// MARK: - SCNNode material overrides (Phase 4.14)

/// Per-SCNNode material override values. Scenes set one of these on
/// individual SCNNodes when sharing an archetype SCNGeometry across
/// many nodes — the extractor reads the override on top of the
/// geometry's material. Lets Phase 4.12 instancing actually fire on
/// scenes whose visual identity was per-flower / per-egg colour
/// variation: same shape, different per-instance look.
public struct IlluminatoramaNodeOverride {
    public var albedo:    SIMD3<Float>?
    public var metallic:  Float?
    public var roughness: Float?
    public var emission:  SIMD3<Float>?
    /// Raster-only node (#60 item 7): the node draws into the G-buffer as
    /// usual but its TLAS instance gets mask 0 — RT rays never intersect it.
    /// For geometry whose RT representation is supplied another way (e.g. a
    /// registered `IlluminatoramaCurveSet` twin: visible triangle tubes,
    /// curve-primitive shadows/GI/reflections). Read at AS-rebuild time.
    public var rtExclude: Bool

    public init(albedo: SIMD3<Float>? = nil,
                metallic: Float? = nil,
                roughness: Float? = nil,
                emission: SIMD3<Float>? = nil,
                rtExclude: Bool = false) {
        self.albedo = albedo
        self.metallic = metallic
        self.roughness = roughness
        self.emission = emission
        self.rtExclude = rtExclude
    }
}

nonisolated(unsafe) private var illuminatoramaNodeOverrideKey: UInt8 = 0

public extension SCNNode {
    /// Per-node material override. Set in the controller / asset
    /// generator after creating the node; the extractor picks it up
    /// each frame on top of the SCNGeometry's base material.
    /// `nil` means "use the geometry's material as-is."
    var illuminatoramaOverride: IlluminatoramaNodeOverride? {
        get {
            (objc_getAssociatedObject(self, &illuminatoramaNodeOverrideKey)
                as? NodeOverrideBox)?.value
        }
        set {
            let box = newValue.map { NodeOverrideBox(value: $0) }
            objc_setAssociatedObject(self, &illuminatoramaNodeOverrideKey,
                                      box,
                                      .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        }
    }
}

final private class NodeOverrideBox {
    let value: IlluminatoramaNodeOverride
    init(value: IlluminatoramaNodeOverride) { self.value = value }
}

// MARK: - SCNNode glass flag (#60 AAA glass)

nonisolated(unsafe) private var illuminatoramaGlassKey: UInt8 = 0

public extension SCNNode {
    /// Flag this node's geometry as AAA ray-traced glass. Set an
    /// `IlluminatoramaGlassMaterial` (IOR, tint, roughness, density, dispersion)
    /// and the extractor routes the node into the renderer's TLAS glass path
    /// (`illumi_glass_rt_fs` — true entry+exit refraction of the scene behind it)
    /// instead of the deferred opaque pipeline. The one-liner a scene author
    /// reaches for: `node.illuminatoramaGlass = .clearGlass` (or
    /// `.init(ior:tint:roughness:density:dispersion:)`). `nil` = not glass.
    ///
    /// Setting this makes the extractor turn on `renderer.rtGlassEnabled`, so the
    /// glass refracts/reflects the rest of the extracted scene automatically — no
    /// per-scene plumbing.
    var illuminatoramaGlass: IlluminatoramaGlassMaterial? {
        get {
            (objc_getAssociatedObject(self, &illuminatoramaGlassKey)
                as? GlassMaterialBox)?.value
        }
        set {
            let box = newValue.map { GlassMaterialBox(value: $0) }
            objc_setAssociatedObject(self, &illuminatoramaGlassKey, box,
                                      .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        }
    }
}

final private class GlassMaterialBox {
    let value: IlluminatoramaGlassMaterial
    init(value: IlluminatoramaGlassMaterial) { self.value = value }
}
