import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

// ── ILLUMINATORAMA MESH ──────────────────────────────────────────────────────
//
// Phase-1 mesh storage. The three procedural primitives below (unitBox,
// unitSphere, unitGround) exercise the G-buffer + deferred lighting path with
// no host plumbing required. Phase 2.6 adds `from(scnGeometry:device:)` so the
// IlluminatoramaSceneExtractor can register any SCNGeometry as a custom mesh.

@MainActor
public final class IlluminatoramaMesh {

    static let log = Logger(subsystem: AppLog.subsystem,
                                     category: "illuminatoramaMesh")

    public let vertexBuffer: MTLBuffer
    public let indexBuffer: MTLBuffer
    public let indexCount: Int
    public let indexType: MTLIndexType
    /// Number of `IlluminatoramaVertex` records in `vertexBuffer`. Needed to
    /// size + dispatch the one-shot GPU normal/tangent synthesis pass
    /// (`IlluminatoramaRenderer.synthesiseMeshGeometry`).
    public let vertexCount: Int

    /// Phase 4.21 — set by `from(scnGeometry:)` when the source geometry
    /// shipped no normals / no tangents. The extractor hands a flagged mesh to
    /// `IlluminatoramaRenderer.synthesiseMeshGeometry`, which runs the synthesis
    /// on the GPU (one-shot) and clears the flags. Replaces the old CPU
    /// `synthesiseNormals` / `synthesiseTangents` passes that ran on the main
    /// thread inside `from` and parked the run loop on a cold-cache scene.
    public var pendingNormalSynth: Bool = false
    public var pendingTangentSynth: Bool = false

    /// Render this mesh two-sided in the G-buffer pass (cull `.none` + flip the
    /// surface normal for back-facing fragments). Default false (closed opaque
    /// meshes cull back-faces for free). Set true for open / dynamic surfaces —
    /// e.g. a marching-cubes fluid surface that, when it tilts or pours, presents
    /// its back side to the camera and would otherwise render HOLLOW.
    public var doubleSided: Bool = false

    /// When set, the mesh is drawn with an INDIRECT indexed draw whose arguments
    /// (`MTLDrawIndexedPrimitivesIndirectArguments`) a compute kernel wrote this frame — the GPU
    /// decides how many indices are drawn (view-culled, compacted geometry). Used by
    /// `GrassTileField`; `instanceCount` in the arguments must match the single instance.
    public var indirectDrawBuffer: MTLBuffer?
    /// Byte offset of the camera (G-buffer) pass's record in `indirectDrawBuffer`.
    public var indirectDrawOffset: Int = 0
    /// Byte offset of a SEPARATE record the shadow passes draw with, or nil to use the camera's.
    /// Signack (DH-1461) culls a cut to the camera frustum for the G-buffer but must keep every
    /// off-screen caster for the sun/spot/point maps — see `IlluminatoramaMesh+IndirectDraw`.
    public var shadowIndirectDrawOffset: Int?

    /// Cast shadows from BOTH faces (shadow passes use cull `.none` instead of the
    /// default `.front`). Opt-in, default false — an unset mesh keeps the renderer's
    /// historical behaviour byte-for-byte.
    ///
    /// **Why this is not just `doubleSided`.** The shadow passes cull FRONT faces, which
    /// stores the *second* depth (the back of the caster) and is the renderer's PRIMARY
    /// acne defence — `shadowBias` (0.0003), `shadowSlopeBias` (0.0) and the normal-offset
    /// push in `cascadeVisibility` are all deliberately set to near-nothing *because* this
    /// second-depth trick already separates a lit receiver from its own occluder. That
    /// trick is only valid for a **closed solid**, where a back face exists behind every
    /// front face. For an **open shell** — a leaf card, a grass ribbon, a zero-thickness
    /// sheet — the face that intercepts the light IS the only face, so culling it means
    /// the mesh writes no shadow depth where it should and its shadow is displaced (or
    /// absent) by the depth of whatever shell lies beyond.
    ///
    /// `doubleSided` is a *shading* property ("don't go hollow when the camera sees the
    /// back"), and closed solids legitimately set it — Daydream's whole structural shell,
    /// millwork and stairs are `doubleSided` so the cutaway shows interior faces. Driving
    /// the shadow cull off `doubleSided` would therefore hand two-sided shadow casting to
    /// every wall and floor in the scene, i.e. remove the primary acne defence from exactly
    /// the flat, sun-facing geometry it was calibrated for. Hence a separate, narrow,
    /// default-off flag that only genuinely open geometry opts into.
    public var shadowCastsBothFaces: Bool = false

    /// Whether this mesh's instances get SURFACE-CACHE cards (DH-1314). Default true — an unset
    /// mesh is carded exactly as before, so a scene that never clears it is byte-identical.
    ///
    /// The cache cards every soup triangle (~18 atlas texels each across five half-float
    /// atlases) and is skipped outright past `surfaceCacheMaxTriangles`. Dense non-planar
    /// geometry — a tree's leaf cards, draped cloth — is most of a furnished house's triangles
    /// and almost none of its bounce light, and chart merging cannot shrink it (it is not
    /// coplanar). Clearing this leaves such a mesh OUT of the cache only: it still rasterises,
    /// casts shadows, sits in the TLAS (so it occludes and is hit by every ray), and is lit by
    /// everything — including the cache's light, through the ordinary gather at its own pixels.
    /// What it loses is a card of its own: an RT GI / reflection hit on it is shaded through the
    /// uncached secondary path (as Standard shades every hit), and the cache update re-shades a
    /// bounce ray that lands on it once (its own sun + the gathering texel's indirect) instead of
    /// reading its card. It does not count against `surfaceCacheMaxTriangles`.
    ///
    /// Read when the cache bakes (a TLAS topology change); set it before `registerMesh`.
    public var surfaceCacheCards: Bool = true

    /// A RAY-TRACING PROXY (Signack, Daydream DH-1461): the mesh lives in the TLAS — rays hit it,
    /// it shadows/bounces/reflects/refracts — but NO raster pass draws it (G-buffer, sun cascades,
    /// spot atlas, point cubes all skip it in `drawMesh`). Default false.
    ///
    /// It is the other half of a raster-only (`rtExcludeFully`) instance of the same object: the
    /// raster side can then change its level of detail every few frames (a new index list, no new
    /// mesh kind) while the traced side keeps one fixed cut — so a camera-distance LOD change never
    /// touches TLAS topology (a 75–200 ms rebuild on a real house).
    public var tracedOnly: Bool = false

    /// Draw this mesh's CAMERA pass through the visibility buffer (Signack S3, Daydream DH-1461):
    /// it skips the conventional G-buffer draw, rasterises only a triangle id against the
    /// G-buffer's depth, and is shaded once per covered pixel by the resolve, which writes the same
    /// G-buffer targets (`IlluminatoramaRenderer+VisBuffer`). Shadow passes are unchanged. Needs
    /// 32-bit indices; ignored (drawn conventionally) otherwise or when the host lever is off.
    public var visBuffer: Bool = false
    /// Optional per-triangle cluster id buffer parallel to `indexBuffer` (Signack's GPU cut writes
    /// it) — only the visibility buffer's cluster debug view reads it.
    public var visBufferClusterIDs: MTLBuffer?

    /// Object-space AABB, computed once from the vertex buffer and cached.
    ///
    /// Diagnostic-path only (the geometry auditors) — nothing per-frame reads
    /// this, so the one-time CPU walk over shared vertex memory is not the
    /// `snapshot() → CPU array` antipattern; there is no GPU round-trip and no
    /// rebuild, it just measures memory the CPU already has mapped.
    private var cachedBounds: (min: SIMD3<Float>, max: SIMD3<Float>)??
    public func localBounds() -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        if let c = cachedBounds { return c }
        // A private-storage buffer has no CPU mapping to read; report "unknown"
        // rather than reading garbage.
        guard vertexBuffer.storageMode != .private, vertexCount > 0 else {
            cachedBounds = .some(nil); return nil
        }
        let p = vertexBuffer.contents().bindMemory(to: IlluminatoramaVertex.self,
                                                   capacity: vertexCount)
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for i in 0..<vertexCount {
            let v = p[i].position
            guard v.x.isFinite, v.y.isFinite, v.z.isFinite else { continue }
            lo = simd_min(lo, v); hi = simd_max(hi, v)
        }
        guard lo.x <= hi.x else { cachedBounds = .some(nil); return nil }
        let r = (min: lo, max: hi)
        cachedBounds = .some(r)
        return r
    }

    /// Object-space bounding sphere (centre, radius), computed ONCE from the vertex buffer on first
    /// ask (DH-0637). nil when the buffer is `.private` (GPU-only vertices) — callers must then
    /// treat the mesh as unbounded. Used by the spot-shadow reuse to ask "can this swaying
    /// instance swing through that cone?", and by visibility culling (DH-0534) to bound every
    /// draw group — so a mesh whose CPU-visible positions are rewritten in place after the first
    /// ask would be culled against stale bounds. GPU-rewritten meshes are built with
    /// `init(vertexBuffer:…)` (`cpuAuthoredVertices == false`), which the culler treats as unbounded.
    public var boundingSphere: (center: SIMD3<Float>, radius: Float)? {
        if let cached = cachedBoundingSphere { return cached }
        guard vertexBuffer.storageMode != .private, vertexCount > 0 else { return nil }
        let stride = MemoryLayout<IlluminatoramaVertex>.stride
        let base = vertexBuffer.contents()
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for v in 0..<vertexCount {
            let p = base.load(fromByteOffset: v * stride, as: SIMD3<Float>.self)
            lo = simd_min(lo, p); hi = simd_max(hi, p)
        }
        let c = (lo + hi) * 0.5
        let sphere = (center: c, radius: simd_length(hi - c))
        cachedBoundingSphere = sphere
        return sphere
    }
    private var cachedBoundingSphere: (center: SIMD3<Float>, radius: Float)?

    /// The largest positive vegetation-wind weights packed in this mesh's tangents — `sway` is
    /// `tangent.x`, `flutter` is `tangent.z`, exactly what `applyTreeWind` reads — so visibility
    /// culling (DH-0534) can bound how far the gust moves a vertex. Floored at 1: a tangent the
    /// GPU synthesis pass writes later is a unit vector, and a scan taken before it lands must not
    /// under-bound it. nil for GPU-only vertices (the caller then treats the draw as unbounded).
    public var windAttributeMax: (sway: Float, flutter: Float)? {
        if let cached = cachedWindAttributeMax { return cached }
        guard vertexBuffer.storageMode != .private, vertexCount > 0 else { return nil }
        let stride = MemoryLayout<IlluminatoramaVertex>.stride
        let offset = MemoryLayout<IlluminatoramaVertex>.offset(of: \.tangent) ?? 48
        let base = vertexBuffer.contents()
        var sway: Float = 1, flutter: Float = 1
        for v in 0..<vertexCount {
            let t = base.load(fromByteOffset: v * stride + offset, as: SIMD4<Float>.self)
            sway = max(sway, t.x); flutter = max(flutter, t.z)
        }
        let result = (sway: sway, flutter: flutter)
        cachedWindAttributeMax = result
        return result
    }
    private var cachedWindAttributeMax: (sway: Float, flutter: Float)?

    /// True when the engine COPIED this mesh's vertices out of CPU arrays (every `vertices:` init
    /// and the primitives built on them) — the CPU-side bounds are then the truth. False for the
    /// GPU-direct `init(vertexBuffer:…)`, whose buffer a compute kernel may rewrite at any time;
    /// visibility culling (DH-0534) never trusts a bound of such a mesh. Keyed on provenance, not
    /// on `registerMesh`: Daydream Home registers its CPU-built house, furniture and trees through
    /// that API, and a registration-based rule left every one of them unculled.
    public private(set) var cpuAuthoredVertices: Bool = true

    public init(device: MTLDevice, vertices: [IlluminatoramaVertex], indices: [UInt16]) {
        guard let vb = device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<IlluminatoramaVertex>.stride * vertices.count,
            options: .storageModeShared
        ) else { preconditionFailure("MTLDevice.makeBuffer(vertices) returned nil") }
        guard let ib = device.makeBuffer(
            bytes: indices,
            length: MemoryLayout<UInt16>.stride * indices.count,
            options: .storageModeShared
        ) else { preconditionFailure("MTLDevice.makeBuffer(indices) returned nil") }
        vb.label = "Illuminatorama.vertices"
        ib.label = "Illuminatorama.indices"
        self.vertexBuffer = vb
        self.indexBuffer = ib
        self.indexCount = indices.count
        self.indexType = .uint16
        self.vertexCount = vertices.count
    }

    public init(device: MTLDevice, vertices: [IlluminatoramaVertex], indices: [UInt32]) {
        guard let vb = device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<IlluminatoramaVertex>.stride * vertices.count,
            options: .storageModeShared
        ) else { preconditionFailure("MTLDevice.makeBuffer(vertices) returned nil") }
        guard let ib = device.makeBuffer(
            bytes: indices,
            length: MemoryLayout<UInt32>.stride * indices.count,
            options: .storageModeShared
        ) else { preconditionFailure("MTLDevice.makeBuffer(indices) returned nil") }
        vb.label = "Illuminatorama.vertices"
        ib.label = "Illuminatorama.indices"
        self.vertexBuffer = vb
        self.indexBuffer = ib
        self.indexCount = indices.count
        self.indexType = .uint32
        self.vertexCount = vertices.count
    }

    /// Phase 2.6 — variable-index-width init for meshes converted from
    /// `SCNGeometry`. The G-buffer pass already binds `indexType` per draw,
    /// so a 32-bit-index mesh works the same as a 16-bit-index mesh from
    /// the renderer's side. We promote 8-bit SceneKit indices to 16-bit
    /// at conversion time (8-bit doesn't have a MTLIndexType counterpart).
    public init(device: MTLDevice,
                vertices: [IlluminatoramaVertex],
                indices: Data,
                indexCount: Int,
                indexType: MTLIndexType,
                label: String = "Illuminatorama.mesh") {
        // Metal's debug-layer asserts on length==0, so reject empty inputs
        // up here with a useful message rather than at the Obj-C boundary.
        precondition(!vertices.isEmpty,
                     "IlluminatoramaMesh init given 0 vertices (\(label))")
        precondition(indices.count > 0 && indexCount > 0,
                     "IlluminatoramaMesh init given empty index data (\(label))")
        guard let vb = device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<IlluminatoramaVertex>.stride * vertices.count,
            options: .storageModeShared
        ) else { preconditionFailure("MTLDevice.makeBuffer(vertices) returned nil") }
        let ib: MTLBuffer = indices.withUnsafeBytes { raw -> MTLBuffer in
            guard let base = raw.baseAddress else {
                preconditionFailure("Empty index data passed to IlluminatoramaMesh init")
            }
            guard let buf = device.makeBuffer(
                bytes: base, length: indices.count, options: .storageModeShared
            ) else { preconditionFailure("MTLDevice.makeBuffer(indices) returned nil") }
            return buf
        }
        vb.label = "\(label).vertices"
        ib.label = "\(label).indices"
        self.vertexBuffer = vb
        self.indexBuffer = ib
        self.indexCount = indexCount
        self.indexType = indexType
        self.vertexCount = vertices.count
    }

    /// Phase 4.8 — GPU-direct init for compute-fed geometry. The buffers
    /// are NOT copied: the caller retains ownership and is responsible for
    /// the contents being valid `IlluminatoramaVertex` (pos+normal+uv+
    /// tangent, stride 96) at draw time. Lets a compute kernel write
    /// vertices once and have Illuminatorama read them straight back —
    /// no round-trip through CPU memory.
    ///
    /// Use cases: `DynamicMesh`, `BarkRenderer`, MLS-MPM marching-cubes
    /// output, any solver whose vertices live on the GPU. The caller
    /// pairs this with `IlluminatoramaRenderer.registerMesh(...)` to land
    /// in the renderer's draw table and gets an `IlluminatoramaMeshHandle`
    /// it attaches to its `SCNGeometry` via the associated-object hook in
    /// `SCNGeometry.illuminatoramaMeshHandle`.
    public init(vertexBuffer: MTLBuffer,
                indexBuffer: MTLBuffer,
                indexCount: Int,
                indexType: MTLIndexType,
                label: String = "Illuminatorama.gpuMesh") {
        precondition(indexCount > 0,
                     "IlluminatoramaMesh GPU-direct init given indexCount=0 (\(label))")
        vertexBuffer.label = "\(label).vertices"
        indexBuffer.label  = "\(label).indices"
        self.vertexBuffer = vertexBuffer
        self.indexBuffer  = indexBuffer
        self.indexCount   = indexCount
        self.indexType    = indexType
        self.vertexCount  = vertexBuffer.length / MemoryLayout<IlluminatoramaVertex>.stride
        self.cpuAuthoredVertices = false
    }

    /// A second mesh over `source`'s vertex buffer with its OWN index buffer (Signack, DH-1461):
    /// every level of detail of an object shares one set of vertices and differs only in which
    /// triangles it indexes. The vertex buffer is not copied; provenance (and with it whether
    /// visibility culling may trust the CPU-side bounds) is inherited from `source`, so a
    /// CPU-authored object stays culled. `indexCount` may be less than the buffer holds (a
    /// capacity buffer drawn through `indirectDrawBuffer`).
    public init(sharingVerticesOf source: IlluminatoramaMesh, indexBuffer: MTLBuffer,
                indexCount: Int, indexType: MTLIndexType = .uint32,
                label: String = "Illuminatorama.sharedVertexMesh") {
        precondition(indexCount > 0, "IlluminatoramaMesh sharing init given indexCount=0 (\(label))")
        indexBuffer.label = "\(label).indices"
        self.vertexBuffer = source.vertexBuffer
        self.indexBuffer = indexBuffer
        self.indexCount = indexCount
        self.indexType = indexType
        self.vertexCount = source.vertexCount
        self.cpuAuthoredVertices = source.cpuAuthoredVertices
    }

    // ── Procedural primitives ────────────────────────────────────────────────

    /// Unit cube centred at the origin, side length 1.
    public static func unitBox(device: MTLDevice) -> IlluminatoramaMesh {
        let h: Float = 0.5
        // 6 faces × 4 verts. Each face has its own normals so flats stay sharp.
        let faces: [(SIMD3<Float>, [SIMD3<Float>])] = [
            (SIMD3( 1, 0, 0), [SIMD3( h,-h, h), SIMD3( h,-h,-h), SIMD3( h, h,-h), SIMD3( h, h, h)]),
            (SIMD3(-1, 0, 0), [SIMD3(-h,-h,-h), SIMD3(-h,-h, h), SIMD3(-h, h, h), SIMD3(-h, h,-h)]),
            (SIMD3( 0, 1, 0), [SIMD3(-h, h, h), SIMD3( h, h, h), SIMD3( h, h,-h), SIMD3(-h, h,-h)]),
            (SIMD3( 0,-1, 0), [SIMD3(-h,-h,-h), SIMD3( h,-h,-h), SIMD3( h,-h, h), SIMD3(-h,-h, h)]),
            (SIMD3( 0, 0, 1), [SIMD3(-h,-h, h), SIMD3( h,-h, h), SIMD3( h, h, h), SIMD3(-h, h, h)]),
            (SIMD3( 0, 0,-1), [SIMD3( h,-h,-h), SIMD3(-h,-h,-h), SIMD3(-h, h,-h), SIMD3( h, h,-h)]),
        ]
        var verts: [IlluminatoramaVertex] = []
        var indices: [UInt16] = []
        for (normal, quad) in faces {
            let base = UInt16(verts.count)
            for (i, p) in quad.enumerated() {
                let uv = SIMD2<Float>(Float(i & 1), Float((i >> 1) & 1))
                verts.append(IlluminatoramaVertex(position: p, normal: normal, uv: uv))
            }
            indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
        }
        return IlluminatoramaMesh(device: device, vertices: verts, indices: indices)
    }

    /// UV sphere with `meridians` longitudinal divisions and `parallels`
    /// latitudinal divisions. 32×16 gives a smooth specular highlight without
    /// being expensive.
    public static func unitSphere(
        device: MTLDevice,
        meridians: Int = 32,
        parallels: Int = 16
    ) -> IlluminatoramaMesh {
        var verts: [IlluminatoramaVertex] = []
        verts.reserveCapacity((parallels + 1) * (meridians + 1))
        for j in 0...parallels {
            let v = Float(j) / Float(parallels)
            let phi = v * .pi
            for i in 0...meridians {
                let u = Float(i) / Float(meridians)
                let theta = u * .pi * 2
                let n = SIMD3<Float>(
                    sin(phi) * cos(theta),
                    cos(phi),
                    sin(phi) * sin(theta)
                )
                verts.append(IlluminatoramaVertex(position: n, normal: n, uv: SIMD2(u, v)))
            }
        }
        var indices: [UInt16] = []
        let stride = meridians + 1
        for j in 0..<parallels {
            for i in 0..<meridians {
                let a = UInt16(j * stride + i)
                let b = UInt16(j * stride + i + 1)
                let c = UInt16((j + 1) * stride + i)
                let d = UInt16((j + 1) * stride + i + 1)
                // CCW viewed from outside: a → b (east) → c (south)
                indices.append(contentsOf: [a, b, c, b, d, c])
            }
        }
        return IlluminatoramaMesh(device: device, vertices: verts, indices: indices)
    }

    /// Ground plane in the XZ plane, side length 1, normal +Y.
    public static func unitGround(device: MTLDevice) -> IlluminatoramaMesh {
        let h: Float = 0.5
        let n = SIMD3<Float>(0, 1, 0)
        let verts: [IlluminatoramaVertex] = [
            IlluminatoramaVertex(position: SIMD3(-h, 0, -h), normal: n, uv: SIMD2(0, 0)),
            IlluminatoramaVertex(position: SIMD3( h, 0, -h), normal: n, uv: SIMD2(1, 0)),
            IlluminatoramaVertex(position: SIMD3( h, 0,  h), normal: n, uv: SIMD2(1, 1)),
            IlluminatoramaVertex(position: SIMD3(-h, 0,  h), normal: n, uv: SIMD2(0, 1)),
        ]
        let indices: [UInt16] = [0, 2, 1, 0, 3, 2]
        return IlluminatoramaMesh(device: device, vertices: verts, indices: indices)
    }
}

