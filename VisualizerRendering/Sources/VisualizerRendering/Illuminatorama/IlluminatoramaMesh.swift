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

    private static let log = Logger(subsystem: AppLog.subsystem,
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

    // ── Phase 2.6 — SCNGeometry conversion ────────────────────────────────────
    //
    // Builds an `IlluminatoramaMesh` from a SceneKit geometry by reading its
    // position / normal / texture-coordinate sources and the first element's
    // index data. Returns nil for geometries the converter can't handle yet
    // (non-triangle primitives, missing position source, exotic component
    // layouts) — the extractor logs and skips when this happens.
    //
    // Conventions:
    //   • Position source is required.
    //   • If a normal source is missing, normals are computed by averaging
    //     per-triangle face normals around each vertex (cheap; gives smooth
    //     shading on welded meshes, faceted shading on unwelded ones).
    //   • Texture-coordinate source is optional — `(0, 0)` is used as a
    //     stand-in. UVs only show up in the renderer today as a passthrough
    //     to the fragment shader (which doesn't sample textures yet), so a
    //     missing UV is harmless.
    //   • Multi-element geometries collapse to their FIRST element. This
    //     means a multi-material `SCNGeometry` only renders its first
    //     submesh's material — workable for the common case where a node
    //     uses a single material, awkward for the rare case where it
    //     doesn't. Track in `docs/known-issues/` if it bites a real scene.

    /// Back-compat shim — defaults to the first element. Preserved so
    /// any external caller still works; the extractor now prefers the
    /// indexed variant below to support per-element materials.
    public static func from(scnGeometry geometry: SCNGeometry,
                             device: MTLDevice) -> IlluminatoramaMesh? {
        return from(scnGeometry: geometry, elementIndex: 0, device: device)
    }

    /// Phase 4.3 — element-indexed conversion. A multi-element SCNGeometry
    /// (e.g. a pizza with separate submeshes for crust / sauce / cheese
    /// / pepperoni, each with its own material) emits ONE
    /// `IlluminatoramaMesh` per element here; the extractor pairs each
    /// with the geometry's matching `material(at:)`. Vertices are shared
    /// conceptually but each mesh gets its own copy of the buffer — the
    /// renderer's draw path is per-mesh, so a shared-vertex / multiple-
    /// index-buffer split would require a deeper API change. Memory
    /// overhead is bounded: vertex payloads are small relative to the
    /// shadow / G-buffer / history texture footprint.
    public static func from(scnGeometry geometry: SCNGeometry,
                             elementIndex: Int,
                             device: MTLDevice) -> IlluminatoramaMesh? {
        // SCNFloor is special-cased: it has no readable CPU geometry
        // (SceneKit synthesises an infinite plane at render time based on
        // the camera). For Illuminatorama's purposes a large XZ-plane
        // quad is an exact stand-in — it has the same surface normal,
        // position, and material, and the renderer's far plane clips
        // beyond what the camera can see anyway. Without this branch the
        // floor disappears under Illuminatorama for every scene that
        // uses one (Forest, Fireworks, HotdogWaterslide,
        // GiantGummyBearsPlus, HotdogDropPlus, …).
        if let floor = geometry as? SCNFloor {
            // SCNFloor has a single element; ignore the element index.
            return groundQuad(for: floor, device: device)
        }

        guard elementIndex < geometry.elements.count else {
            return nil
        }
        let element = geometry.elements[elementIndex]
        // Only triangle lists supported. Strips/fans/lines/points return nil
        // and the caller logs+skips. SceneKit's built-in primitives all use
        // `.triangles`, so this covers SCNBox, SCNSphere, SCNFloor, SCNPlane,
        // SCNCapsule, SCNCylinder, SCNCone, and hand-built meshes.
        guard element.primitiveType == .triangles else {
            log.debug("SCNGeometry \(type(of: geometry)) uses non-triangle primitive (\(element.primitiveType.rawValue)); skipping")
            return nil
        }

        let positions = floatTriples(from: geometry.sources(for: .vertex).first)
        guard let positions = positions else {
            log.debug("SCNGeometry \(type(of: geometry)) has no readable vertex source")
            return nil
        }
        // Empty meshes (placeholder SCNGeometry, particle stand-ins, etc.) would
        // otherwise reach MTLDevice.makeBuffer(length: 0) and trip the Metal
        // debug-layer assert. Drop them at conversion time.
        guard !positions.isEmpty else {
            log.debug("SCNGeometry \(type(of: geometry)) has 0 vertices; skipping")
            return nil
        }
        guard element.primitiveCount > 0 else {
            log.debug("SCNGeometry \(type(of: geometry)) has 0 primitives; skipping")
            return nil
        }
        // MTLBuffer-backed / compute-fed SCNGeometry can report primitiveCount > 0
        // but expose `element.data` as empty Data — SceneKit hides the GPU buffer.
        // We can't convert that to a CPU-side IlluminatoramaMesh; skip it.
        guard !element.data.isEmpty else {
            log.debug("SCNGeometry \(type(of: geometry)) has primitiveCount=\(element.primitiveCount) but empty element.data (GPU-backed); skipping")
            return nil
        }
        let normals = floatTriples(from: geometry.sources(for: .normal).first)
        let uvs     = floatPairs(from: geometry.sources(for: .texcoord).first)
        // Phase 4.17 — optional per-vertex RGBA colour stored on a
        // `SCNGeometrySource(semantic: .color)`. SceneKit allows 3-
        // or 4-component, float or normalised-byte; we read whichever
        // the source happens to ship and normalise to float4. Missing
        // → default white (no-op multiply at shading time).
        let colors  = floatColors(from: geometry.sources(for: .color).first)

        // Build per-vertex IlluminatoramaVertex. If we synthesise normals
        // we'll fill them in after we have the index buffer.
        var verts: [IlluminatoramaVertex] = []
        verts.reserveCapacity(positions.count)
        let needsSynthNormals = (normals == nil)
        let normalArray = normals ?? Array(repeating: SIMD3<Float>(0, 1, 0),
                                            count: positions.count)
        let uvArray     = uvs     ?? Array(repeating: SIMD2<Float>(0, 0),
                                            count: positions.count)
        let colorArray  = colors  ?? Array(repeating: SIMD4<Float>(1, 1, 1, 1),
                                            count: positions.count)
        let uvCount = uvArray.count
        let normalCount = normalArray.count
        let colorCount = colorArray.count
        for i in 0..<positions.count {
            let n = i < normalCount ? normalArray[i] : SIMD3<Float>(0, 1, 0)
            let uv = i < uvCount ? uvArray[i] : SIMD2<Float>(0, 0)
            let c  = i < colorCount ? colorArray[i] : SIMD4<Float>(1, 1, 1, 1)
            verts.append(IlluminatoramaVertex(position: positions[i],
                                               normal: n,
                                               uv: uv,
                                               color: c))
        }

        // Read indices into a `Data` blob in whatever format the renderer
        // will dispatch. SceneKit elements can be 1/2/4 bytes per index;
        // promote 1-byte → 2-byte (no MTLIndexType.uint8) and keep 2/4
        // verbatim.
        guard let raw = readIndexData(element: element) else {
            log.debug("SCNGeometry \(type(of: geometry)) has unsupported index width (\(element.bytesPerIndex) bytes)")
            return nil
        }
        // Drop any triangle that references a vertex outside the readable vertex
        // source. `vertexCount` (from `sources(for:.vertex).first.vectorCount`)
        // and the element's index buffer are independent SceneKit fields; for
        // built-in primitives they always agree, but a merged / imported / hand-
        // built geometry can desync them. Registering that mismatch lets every
        // downstream consumer that does `positions[index]` (BLAS build, face
        // normals, the surface-cache soup) read out of bounds and SIGTRAP — the
        // crash this method now forecloses. Sanitise once, here, so no consumer
        // has to. See docs/known-issues/illuminatorama-index-exceeds-vertexcount.md.
        guard let (indexData, indexCount, indexType) =
            sanitiseIndices(raw, vertexCount: verts.count,
                            geometryLabel: "\(type(of: geometry))") else {
            return nil
        }

        // Phase 4.21 — normal + tangent synthesis runs on the GPU now, NOT
        // here. The old CPU `synthesiseNormals` / `synthesiseTangents` passes
        // (O(triangles) scatter-add) ran on the main thread inside this method;
        // on a cold-cache scene the extractor converts every geometry in one
        // synchronous `extractFrame` tick, which parked the run loop for
        // seconds (observed: ~9 s main-thread hang in `accumulateTangents`).
        //
        // Instead we flag what the source was missing and hand the flagged
        // mesh to `IlluminatoramaRenderer.synthesiseMeshGeometry`, which runs
        // the synthesis once on the GPU before the first draw. `verts` is
        // uploaded as-is (placeholder normals where the source had none, zero
        // tangents); the GPU pass overwrites the flagged channels in place.
        // The CPU `synthesiseNormals` / `synthesiseTangents` below are retained
        // as the reference implementation of the same math.
        let hasShippedTangents = geometry.sources(for: .tangent).first != nil

        let mesh = IlluminatoramaMesh(
            device: device,
            vertices: verts,
            indices: indexData,
            indexCount: indexCount,
            indexType: indexType,
            label: "Illuminatorama.scn.\(type(of: geometry))"
        )
        mesh.pendingNormalSynth = needsSynthNormals
        mesh.pendingTangentSynth = !hasShippedTangents
        return mesh
    }

    // ── SCNFloor stand-in ─────────────────────────────────────────────────────

    /// Large quad in the XZ plane standing in for an `SCNFloor`. SCNFloor's
    /// `width` / `length` default to 0 which SceneKit interprets as
    /// "infinite"; we substitute a 2000 m side in that case so the camera's
    /// far plane clips before the edge becomes visible. UVs map 1:1 over
    /// the quad's extent so any tiled material texture lines up reasonably
    /// (today the renderer doesn't sample diffuse textures anyway, but the
    /// math is straightforward to keep correct).
    private static func groundQuad(for floor: SCNFloor,
                                    device: MTLDevice) -> IlluminatoramaMesh {
        let defaultSide: Float = 2000
        let w = Float(floor.width  > 0 ? floor.width  : CGFloat(defaultSide)) * 0.5
        let l = Float(floor.length > 0 ? floor.length : CGFloat(defaultSide)) * 0.5
        let n = SIMD3<Float>(0, 1, 0)
        // Winding: CCW viewed from above (+Y looking down -Y) — SceneKit's
        // standard front-face convention. The two indexed triangles cover
        // the quad with the same orientation as `unitGround`.
        let verts: [IlluminatoramaVertex] = [
            IlluminatoramaVertex(position: SIMD3(-w, 0, -l), normal: n, uv: SIMD2(0, 0)),
            IlluminatoramaVertex(position: SIMD3( w, 0, -l), normal: n, uv: SIMD2(1, 0)),
            IlluminatoramaVertex(position: SIMD3( w, 0,  l), normal: n, uv: SIMD2(1, 1)),
            IlluminatoramaVertex(position: SIMD3(-w, 0,  l), normal: n, uv: SIMD2(0, 1)),
        ]
        let indices: [UInt16] = [0, 2, 1, 0, 3, 2]
        return IlluminatoramaMesh(device: device, vertices: verts, indices: indices)
    }

    // ── SCNGeometrySource → [SIMD3<Float>] / [SIMD2<Float>] ──────────────────

    /// Reads a vertex source with 3 floats per element. Tolerates double
    /// components (rare but legal — SceneKit can emit Doubles on Intel)
    /// and any positive `dataOffset` / `dataStride`. Returns nil for
    /// non-float components or unsupported widths.
    private static func floatTriples(from source: SCNGeometrySource?) -> [SIMD3<Float>]? {
        guard let s = source else { return nil }
        guard s.componentsPerVector >= 3, s.usesFloatComponents else { return nil }
        let bpc = s.bytesPerComponent
        let stride = s.dataStride
        let offset = s.dataOffset
        let count = s.vectorCount
        guard count > 0, bpc == 4 || bpc == 8 else { return nil }
        // MTLBuffer-backed sources surface a Data that doesn't reflect the
        // GPU contents — usually empty or too short for vectorCount*stride.
        // Bail rather than walk an unsafe pointer off the end.
        let neededBytes = offset + (count - 1) * stride + 3 * bpc
        guard s.data.count >= neededBytes else { return nil }
        var out: [SIMD3<Float>] = []
        out.reserveCapacity(count)
        s.data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            for i in 0..<count {
                let p = base.advanced(by: offset + i * stride)
                if bpc == 4 {
                    let f = p.assumingMemoryBound(to: Float.self)
                    out.append(SIMD3<Float>(f[0], f[1], f[2]))
                } else {
                    let d = p.assumingMemoryBound(to: Double.self)
                    out.append(SIMD3<Float>(Float(d[0]), Float(d[1]), Float(d[2])))
                }
            }
        }
        return out.count == count ? out : nil
    }

    /// Phase 4.17 — pull a per-vertex `SCNGeometrySource(semantic: .color)`
    /// into a flat `[SIMD4<Float>]`. Handles the two layouts SceneKit
    /// actually emits: float RGBA (`bpc==4`, `componentsPerVector==4`)
    /// — the path `SCNGeometrySource(data:semantic:.color, …,
    /// usesFloatComponents:true, componentsPerVector:4, bytesPerComponent:4)`
    /// takes — and the legacy `usesFloatComponents:false` byte path some
    /// imported assets ship. Float RGB (`componentsPerVector==3`) is also
    /// accepted, alpha defaulting to 1. Returns nil for unsupported widths
    /// rather than guessing — callers fall through to the default-white
    /// path so a mesh without colours stays a no-op multiply at shading
    /// time.
    private static func floatColors(from source: SCNGeometrySource?) -> [SIMD4<Float>]? {
        guard let s = source else { return nil }
        let comps = s.componentsPerVector
        let bpc = s.bytesPerComponent
        let stride = s.dataStride
        let offset = s.dataOffset
        let count = s.vectorCount
        guard count > 0, comps >= 3, comps <= 4 else { return nil }
        let bytesPerVector = comps * bpc
        let neededBytes = offset + (count - 1) * stride + bytesPerVector
        guard s.data.count >= neededBytes else { return nil }
        var out: [SIMD4<Float>] = []
        out.reserveCapacity(count)
        s.data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            for i in 0..<count {
                let p = base.advanced(by: offset + i * stride)
                if s.usesFloatComponents {
                    if bpc == 4 {
                        let f = p.assumingMemoryBound(to: Float.self)
                        let a = comps >= 4 ? f[3] : 1
                        out.append(SIMD4<Float>(f[0], f[1], f[2], a))
                    } else if bpc == 8 {
                        let d = p.assumingMemoryBound(to: Double.self)
                        let a = comps >= 4 ? Float(d[3]) : 1
                        out.append(SIMD4<Float>(Float(d[0]), Float(d[1]),
                                                 Float(d[2]), a))
                    }
                } else if bpc == 1 {
                    // Unsigned-byte RGBA (the path SCNGeometrySource(colors:)
                    // takes when fed an [SCNVector4]-of-bytes asset import).
                    // 0..255 → 0..1 linearly.
                    let b = p.assumingMemoryBound(to: UInt8.self)
                    let a = comps >= 4 ? Float(b[3]) / 255.0 : 1
                    out.append(SIMD4<Float>(Float(b[0]) / 255.0,
                                             Float(b[1]) / 255.0,
                                             Float(b[2]) / 255.0,
                                             a))
                }
            }
        }
        return out.count == count ? out : nil
    }

    private static func floatPairs(from source: SCNGeometrySource?) -> [SIMD2<Float>]? {
        guard let s = source else { return nil }
        guard s.componentsPerVector >= 2, s.usesFloatComponents else { return nil }
        let bpc = s.bytesPerComponent
        let stride = s.dataStride
        let offset = s.dataOffset
        let count = s.vectorCount
        guard count > 0, bpc == 4 || bpc == 8 else { return nil }
        // Same bound check as floatTriples — GPU-backed sources can ship a
        // CPU `data` that's shorter than `vectorCount * stride`.
        let neededBytes = offset + (count - 1) * stride + 2 * bpc
        guard s.data.count >= neededBytes else { return nil }
        var out: [SIMD2<Float>] = []
        out.reserveCapacity(count)
        s.data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            for i in 0..<count {
                let p = base.advanced(by: offset + i * stride)
                if bpc == 4 {
                    let f = p.assumingMemoryBound(to: Float.self)
                    out.append(SIMD2<Float>(f[0], f[1]))
                } else {
                    let d = p.assumingMemoryBound(to: Double.self)
                    out.append(SIMD2<Float>(Float(d[0]), Float(d[1])))
                }
            }
        }
        return out.count == count ? out : nil
    }

    // ── Index conversion ──────────────────────────────────────────────────────

    /// Pull the raw index `Data` blob out of an `SCNGeometryElement`, promote
    /// 1-byte indices to 2-byte (Metal has no `MTLIndexType.uint8`), and
    /// return a tuple ready to feed `init(device:vertices:indices:indexCount:indexType:)`.
    private static func readIndexData(element: SCNGeometryElement)
        -> (data: Data, count: Int, type: MTLIndexType)? {
        let count = element.primitiveCount * 3
        let bpi = element.bytesPerIndex
        let raw = element.data
        // GPU-backed elements report a nonzero primitiveCount but ship empty
        // Data — there's no CPU buffer to copy. Refuse them here so callers
        // see a clean nil rather than a zero-length Metal buffer.
        guard count > 0, !raw.isEmpty else { return nil }
        switch bpi {
        case 1:
            // Promote: walk each byte and write a UInt16 for it.
            var promoted = Data(count: count * MemoryLayout<UInt16>.size)
            promoted.withUnsafeMutableBytes { outRaw in
                let out = outRaw.bindMemory(to: UInt16.self)
                raw.withUnsafeBytes { inRaw in
                    let bytes = inRaw.bindMemory(to: UInt8.self)
                    for i in 0..<count {
                        out[i] = UInt16(bytes[i])
                    }
                }
            }
            return (promoted, count, .uint16)
        case 2:
            return (raw, count, .uint16)
        case 4:
            return (raw, count, .uint32)
        default:
            return nil
        }
    }

    /// Validate an index buffer against the mesh's vertex count and, if any
    /// triangle references a vertex `>= vertexCount`, return a filtered copy
    /// with those triangles dropped. Returns the input unchanged (cheap, no
    /// copy) when every index is in range — the overwhelmingly common case.
    /// Returns `nil` only if every triangle was out of range (nothing left to
    /// draw), so the caller skips the mesh rather than build an empty one.
    ///
    /// `from(scnGeometry:)` derives `vertexCount` from the `.vertex` source's
    /// `vectorCount` and the indices from the element independently; merged /
    /// imported / hand-built geometry can desync the two. Every downstream
    /// consumer indexes `positions[index]` trusting this invariant — enforce it
    /// once at the door so an out-of-range index can't SIGTRAP the render loop.
    private static func sanitiseIndices(
        _ input: (data: Data, count: Int, type: MTLIndexType),
        vertexCount: Int,
        geometryLabel: String
    ) -> (data: Data, count: Int, type: MTLIndexType)? {
        let (data, count, type) = input
        // Decode to UInt32 so the bounds test is uniform across widths.
        var idx = [UInt32](); idx.reserveCapacity(count)
        data.withUnsafeBytes { raw in
            switch type {
            case .uint16:
                let p = raw.bindMemory(to: UInt16.self)
                for i in 0..<count { idx.append(UInt32(p[i])) }
            default:   // .uint32
                let p = raw.bindMemory(to: UInt32.self)
                for i in 0..<count { idx.append(p[i]) }
            }
        }
        let limit = UInt32(vertexCount)
        // Fast path: scan for any out-of-range index; if none, hand the
        // original buffer straight back (no allocation, no rebuild).
        guard idx.contains(where: { $0 >= limit }) else { return input }

        var kept = [UInt32](); kept.reserveCapacity(count)
        var t = 0
        while t + 2 < count {
            let a = idx[t], b = idx[t + 1], c = idx[t + 2]
            if a < limit && b < limit && c < limit { kept.append(a); kept.append(b); kept.append(c) }
            t += 3
        }
        let dropped = (count - kept.count) / 3
        log.notice("Illuminatorama mesh \(geometryLabel): \(dropped) of \(count / 3) triangles reference a vertex >= vertexCount \(vertexCount) (maxIndex \(idx.max() ?? 0)); dropped.")
        guard !kept.isEmpty else { return nil }
        // Re-emit as uint32 (always valid for the kept indices) so we don't
        // have to worry about whether the survivors still fit in uint16.
        let outData = kept.withUnsafeBytes { Data($0) }
        return (outData, kept.count, .uint32)
    }

    // ── CPU object-space triangle soup (for RT acceleration structure) ─────────

    /// Object-space triangle soup for one element: positions + a flat `UInt32`
    /// index list. The RT path uses this to bake a world-space triangle soup
    /// from extracted scene geometry (the caller applies the node's world
    /// transform). Returns nil for GPU-backed / non-triangle / empty elements —
    /// the same elements `from(...)` already refuses. `SCNFloor` is emitted as a
    /// large XZ quad, matching `groundQuad`'s deferred-pass stand-in.
    public static func objectTriangles(scnGeometry geometry: SCNGeometry,
                                        elementIndex: Int)
        -> (positions: [SIMD3<Float>], indices: [UInt32])? {
        if geometry is SCNFloor {
            let s: Float = 400
            return ([SIMD3(-s, 0, -s), SIMD3(s, 0, -s), SIMD3(s, 0, s), SIMD3(-s, 0, s)],
                    [0, 1, 2, 0, 2, 3])
        }
        guard elementIndex < geometry.elements.count else { return nil }
        let element = geometry.elements[elementIndex]
        guard element.primitiveType == .triangles,
              let positions = floatTriples(from: geometry.sources(for: .vertex).first),
              !positions.isEmpty, element.primitiveCount > 0,
              let idx = readIndexData(element: element) else { return nil }
        var indices = [UInt32](); indices.reserveCapacity(idx.count)
        idx.data.withUnsafeBytes { raw in
            switch idx.type {
            case .uint16:
                let p = raw.bindMemory(to: UInt16.self)
                for i in 0..<idx.count { indices.append(UInt32(p[i])) }
            case .uint32:
                let p = raw.bindMemory(to: UInt32.self)
                for i in 0..<idx.count { indices.append(p[i]) }
            @unknown default: break
            }
        }
        guard !indices.isEmpty else { return nil }
        return (positions, indices)
    }

    /// Per-triangle OBJECT-space geometric normals (`float4`, w = 0), computed
    /// from this mesh's own (shared-storage) vertex + index buffers. Built once
    /// when the RT BLAS is created; the instanced RT kernel transforms each by
    /// the per-instance normal matrix to get the world normal at a hit. Returns
    /// `[]` if the buffers aren't CPU-readable.
    public func objectFaceNormals() -> [SIMD4<Float>] {
        let triCount = indexCount / 3
        guard triCount > 0 else { return [] }
        // GPU-direct meshes (.storageModePrivate) have no CPU-readable backing;
        // contents() would abort under the Metal debug layer.
        guard vertexBuffer.storageMode != .private else { return [] }
        let stride = MemoryLayout<IlluminatoramaVertex>.stride
        let vcount = vertexCount
        let vbase = vertexBuffer.contents()
        let ibase = indexBuffer.contents()
        func pos(_ i: Int) -> SIMD3<Float> {
            vbase.load(fromByteOffset: i * stride, as: SIMD3<Float>.self)
        }
        var out = [SIMD4<Float>](); out.reserveCapacity(triCount)
        for t in 0..<triCount {
            let i0: Int, i1: Int, i2: Int
            if indexType == .uint16 {
                let o = t * 3 * 2
                i0 = Int(ibase.load(fromByteOffset: o,     as: UInt16.self))
                i1 = Int(ibase.load(fromByteOffset: o + 2, as: UInt16.self))
                i2 = Int(ibase.load(fromByteOffset: o + 4, as: UInt16.self))
            } else {
                let o = t * 3 * 4
                i0 = Int(ibase.load(fromByteOffset: o,     as: UInt32.self))
                i1 = Int(ibase.load(fromByteOffset: o + 4, as: UInt32.self))
                i2 = Int(ibase.load(fromByteOffset: o + 8, as: UInt32.self))
            }
            // Guard against an index that reaches past the readable vertex block.
            // from()-registered meshes are sanitised (sanitiseIndices), but
            // descriptor / setRTGeometry meshes set indexBuffer + vertexCount
            // independently and bypass that path; an out-of-range index here is a
            // raw out-of-bounds pointer load — worse than an array trap (UB /
            // garbage normals), and the load() can't be caught. Emit a placeholder
            // normal so `out` stays length == triCount and primitive_id alignment
            // with the BLAS is preserved. See
            // docs/known-issues/illuminatorama-index-exceeds-vertexcount.md.
            guard i0 < vcount, i1 < vcount, i2 < vcount else {
                out.append(SIMD4<Float>(0, 1, 0, 0)); continue
            }
            let a = pos(i0), b = pos(i1), c = pos(i2)
            var n = simd_cross(b - a, c - a)
            let len = simd_length(n)
            n = len > 1e-8 ? n / len : SIMD3<Float>(0, 1, 0)
            out.append(SIMD4<Float>(n, 0))
        }
        return out
    }

    /// Object-space positions + flat `UInt32` index list read straight from this
    /// mesh's CPU-readable shared-storage buffers (same access pattern as
    /// `objectFaceNormals`). The surface-cache TLAS path (P1c) uses this to bake
    /// a per-instance world-space soup in the SAME grouped order the TLAS
    /// enumerates instances, so a hit's `(instance_id, primitive_id)` maps to a
    /// global soup triangle via `soupTriBase[instance_id] + primitive_id`.
    public func objectTriangleSoup() -> (positions: [SIMD3<Float>], indices: [UInt32]) {
        guard vertexBuffer.storageMode != .private else { return ([], []) }
        let stride = MemoryLayout<IlluminatoramaVertex>.stride
        let vbase = vertexBuffer.contents()
        var positions = [SIMD3<Float>](); positions.reserveCapacity(vertexCount)
        for v in 0..<vertexCount {
            positions.append(vbase.load(fromByteOffset: v * stride, as: SIMD3<Float>.self))
        }
        let ibase = indexBuffer.contents()
        var indices = [UInt32](); indices.reserveCapacity(indexCount)
        if indexType == .uint16 {
            for i in 0..<indexCount {
                indices.append(UInt32(ibase.load(fromByteOffset: i * 2, as: UInt16.self)))
            }
        } else {
            for i in 0..<indexCount {
                indices.append(ibase.load(fromByteOffset: i * 4, as: UInt32.self))
            }
        }
        return (positions, indices)
    }

    // ── Synthesised normals ───────────────────────────────────────────────────

    /// When the source mesh ships positions but no normals (common for
    /// hand-built `SCNGeometrySource` content), compute smooth vertex
    /// normals by averaging the face normals of every triangle each vertex
    /// participates in. Cheap one-pass over the index buffer.
    private static func synthesiseNormals(into verts: [IlluminatoramaVertex],
                                           indexData: Data,
                                           indexCount: Int,
                                           indexType: MTLIndexType) -> [IlluminatoramaVertex] {
        var accum = Array(repeating: SIMD3<Float>.zero, count: verts.count)
        indexData.withUnsafeBytes { raw in
            switch indexType {
            case .uint16:
                let idx = raw.bindMemory(to: UInt16.self)
                accumulate(triangleCount: indexCount / 3, getIndex: { i in Int(idx[i]) },
                           verts: verts, into: &accum)
            case .uint32:
                let idx = raw.bindMemory(to: UInt32.self)
                accumulate(triangleCount: indexCount / 3, getIndex: { i in Int(idx[i]) },
                           verts: verts, into: &accum)
            @unknown default:
                break
            }
        }
        var out = verts
        for i in 0..<out.count {
            let n = accum[i]
            let len = simd_length(n)
            out[i] = IlluminatoramaVertex(
                position: verts[i].position,
                normal: len > 1e-6 ? n / len : SIMD3<Float>(0, 1, 0),
                uv: verts[i].uv
            )
        }
        return out
    }

    // ── Synthesised tangents ──────────────────────────────────────────────────

    /// Standard derivative-of-UV tangent synthesis. For each triangle with
    /// positions (p0, p1, p2) and UVs (uv0, uv1, uv2), the partial
    /// derivative of position w.r.t. u (in tangent space) is:
    ///
    ///     T = (duv2.y * edge1 - duv1.y * edge2) / det
    ///
    /// Accumulate T per vertex and orthonormalise against the vertex
    /// normal — produces a stable TBN frame for shading normal maps.
    /// Handedness is left at +1; mirrored-UV islands would need a per-
    /// triangle sign computation, which would matter for those islands
    /// but is out of scope here. Most non-mirrored meshes look correct
    /// with the +1 default.
    private static func synthesiseTangents(into verts: [IlluminatoramaVertex],
                                            indexData: Data,
                                            indexCount: Int,
                                            indexType: MTLIndexType) -> [IlluminatoramaVertex] {
        var accum = Array(repeating: SIMD3<Float>.zero, count: verts.count)
        indexData.withUnsafeBytes { raw in
            switch indexType {
            case .uint16:
                let idx = raw.bindMemory(to: UInt16.self)
                accumulateTangents(triangleCount: indexCount / 3,
                                    getIndex: { i in Int(idx[i]) },
                                    verts: verts, into: &accum)
            case .uint32:
                let idx = raw.bindMemory(to: UInt32.self)
                accumulateTangents(triangleCount: indexCount / 3,
                                    getIndex: { i in Int(idx[i]) },
                                    verts: verts, into: &accum)
            @unknown default:
                break
            }
        }
        var out = verts
        for i in 0..<out.count {
            let n = out[i].normal
            let raw = accum[i]
            // Gram-Schmidt: project T onto the plane orthogonal to N so
            // the TBN frame is orthogonal. Falls back to a stable
            // arbitrary tangent when the accumulator is degenerate (no
            // triangle contributed a derivative — typical for unwelded
            // single-vertex islands or zero-UV-area triangles).
            var t = raw - n * simd_dot(raw, n)
            let len = simd_length(t)
            if len < 1e-5 {
                // Arbitrary tangent perpendicular to N — pick whichever
                // axis is least aligned.
                t = abs(n.y) < 0.99 ? simd_cross(SIMD3(0, 1, 0), n)
                                     : simd_cross(SIMD3(1, 0, 0), n)
                t = simd_normalize(t)
            } else {
                t /= len
            }
            out[i].tangent = SIMD4(t, 1.0)
        }
        return out
    }

    private static func accumulateTangents(triangleCount: Int,
                                            getIndex: (Int) -> Int,
                                            verts: [IlluminatoramaVertex],
                                            into accum: inout [SIMD3<Float>]) {
        for t in 0..<triangleCount {
            let i0 = getIndex(t * 3)
            let i1 = getIndex(t * 3 + 1)
            let i2 = getIndex(t * 3 + 2)
            guard i0 < verts.count, i1 < verts.count, i2 < verts.count else { continue }
            let p0 = verts[i0].position
            let p1 = verts[i1].position
            let p2 = verts[i2].position
            let uv0 = verts[i0].uv
            let uv1 = verts[i1].uv
            let uv2 = verts[i2].uv
            let edge1 = p1 - p0
            let edge2 = p2 - p0
            let duv1 = uv1 - uv0
            let duv2 = uv2 - uv0
            let det = duv1.x * duv2.y - duv1.y * duv2.x
            if abs(det) < 1e-8 { continue }
            let inv = 1.0 / det
            let T = (edge1 * duv2.y - edge2 * duv1.y) * inv
            accum[i0] += T
            accum[i1] += T
            accum[i2] += T
        }
    }

    private static func accumulate(triangleCount: Int,
                                    getIndex: (Int) -> Int,
                                    verts: [IlluminatoramaVertex],
                                    into accum: inout [SIMD3<Float>]) {
        for t in 0..<triangleCount {
            let i0 = getIndex(t * 3)
            let i1 = getIndex(t * 3 + 1)
            let i2 = getIndex(t * 3 + 2)
            guard i0 < verts.count, i1 < verts.count, i2 < verts.count else { continue }
            let p0 = verts[i0].position
            let p1 = verts[i1].position
            let p2 = verts[i2].position
            let n = simd_cross(p1 - p0, p2 - p0)
            // Don't normalise here: a triangle's contribution scales with
            // its area, which is the standard area-weighted-average
            // normals scheme. Normalisation happens in the caller.
            accum[i0] += n
            accum[i1] += n
            accum[i2] += n
        }
    }
}
