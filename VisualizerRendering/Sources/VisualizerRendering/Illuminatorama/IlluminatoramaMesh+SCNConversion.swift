import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

extension IlluminatoramaMesh {

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
    //   • Texture-coordinate source is optional. A missing texcoord source
    //     used to fall back to a bare `(0, 0)` for every vertex on the
    //     reasoning that "the renderer doesn't sample textures yet" — no
    //     longer true (the material system samples real albedo/roughness/
    //     normal textures at `in.uv`), and a single shared UV meant a
    //     textured material sampled ONE texel for the whole mesh: a wood
    //     grain, a tile pattern, anything with real texture variation
    //     rendered as a flat, uniform colour (Danny, 2026-09-10, an
    //     imported custom object with a wood finish showing no grain at
    //     all). `triplanarUV(position:normal:)` synthesises a box-projected
    //     UV instead — one texture repeat per metre, the SAME "prebaked at
    //     1 m" convention every instanced placeable's own procedurally-
    //     generated mesh already bakes (`ElementUVScale.furnitureSlice` /
    //     `SurfaceTiling.prebaked(1.0)`), so a custom object tiles a wood/
    //     tile/stone material at the same real-world scale a piece of
    //     built-in furniture does. It is a hard per-vertex box projection,
    //     not blended triplanar sampling — visible seams at the projection's
    //     axis boundaries are the honest cost of a real per-pixel texture
    //     where the source file authored none, and a strict improvement
    //     over one flat colour.
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

        let rawPositions = floatTriples(from: geometry.sources(for: .vertex).first)
        guard let rawPositions = rawPositions else {
            log.debug("SCNGeometry \(type(of: geometry)) has no readable vertex source")
            return nil
        }
        // Empty meshes (placeholder SCNGeometry, particle stand-ins, etc.) would
        // otherwise reach MTLDevice.makeBuffer(length: 0) and trip the Metal
        // debug-layer assert. Drop them at conversion time.
        guard !rawPositions.isEmpty else {
            log.debug("SCNGeometry \(type(of: geometry)) has 0 vertices; skipping")
            return nil
        }
        // Recover authored primitive size/orientation. SceneKit's parametric
        // primitives (SCNBox, SCNCylinder, SCNCone, SCNSphere, …) store their
        // vertex SOURCE in a canonical UNIT object space (box = ±0.5,
        // cylinder = ±0.5) and fold the authored width/height/length/radius
        // into geometry-intrinsic scaling that `sources(for:.vertex)` never
        // exposes — only `geometry.boundingBox` reflects it. Reading the unit
        // source verbatim is what collapsed large SCNBoxes toward a unit cube
        // and sheared rotated/flat SCNCylinders in the gasStation scene.
        let positions = normaliseToBoundingBox(rawPositions, geometry: geometry)
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

        // DH-0742 — detect SceneKit's "split-vertex" element layout. When a
        // geometry's sources have DIFFERENT `vectorCount`s (the common USD/
        // Blender-import case: position is vertex-varying/shared, while normal
        // is authored FACEVARYING — one value per TRIANGLE CORNER, for a hard
        // edge), SceneKit widens each corner's index entry to one index PER
        // DISTINCT vectorCount group, interleaved per corner — [posIndex,
        // cornerIndex, posIndex, cornerIndex, …] — instead of the single shared
        // index every hand-built/procedural mesh in this engine has always had.
        // Reading that as one flat index stream (this code did, before this
        // fix) pairs arbitrary position indices with what are actually
        // per-corner counters, wiring triangles to essentially random vertices.
        // A real `.usdz` side table (Blender export, DH-0742) rendered as a
        // shredded, torn mess because of exactly this — verified against the
        // raw USD stage (`usdcat`) and SceneKit's own reported index buffer:
        // `elementData.count / bytesPerIndex` was 2× `primitiveCount * 3`, and
        // de-interleaving it by 2 recovered the real per-face-vertex indices
        // (matching the USD file's own `faceVertexIndices` exactly) alongside a
        // trivial 0…N corner counter (matching the facevarying normal count).
        // Detected purely by entry-count arithmetic — no SceneKit SPI needed.
        let cornerCount = element.primitiveCount * 3
        let indexEntryWidth = element.bytesPerIndex
        let totalIndexEntries = indexEntryWidth > 0 ? element.data.count / indexEntryWidth : 0
        let streamsPerCorner = (cornerCount > 0 && totalIndexEntries > 0
                                 && totalIndexEntries % cornerCount == 0)
            ? totalIndexEntries / cornerCount : 1

        let verts: [IlluminatoramaVertex]
        let needsSynthNormals: Bool
        let rawIndex: (data: Data, count: Int, type: MTLIndexType)?

        if streamsPerCorner > 1,
           let expanded = expandSplitVertexElement(
               element: element, cornerCount: cornerCount, streamsPerCorner: streamsPerCorner,
               positions: positions, normals: normals, uvs: uvs, colors: colors,
               geometryLabel: "\(type(of: geometry))") {
            verts = expanded.verts
            needsSynthNormals = expanded.needsSynthNormals
            rawIndex = (expanded.indexData, expanded.indexCount, .uint32)
        } else {
            // The overwhelmingly common case: one shared index per corner.
            // Build per-vertex IlluminatoramaVertex. If we synthesise normals
            // we'll fill them in after we have the index buffer.
            var v: [IlluminatoramaVertex] = []
            v.reserveCapacity(positions.count)
            needsSynthNormals = (normals == nil)
            let normalArray = normals ?? Array(repeating: SIMD3<Float>(0, 1, 0),
                                                count: positions.count)
            // No texcoord source at all: synthesise a box-projected UV from each vertex's own
            // position + normal rather than stamping the same (0,0) on every vertex — see the
            // header comment above this method. A *short* per-vertex source (uvs.count <
            // positions.count) is left to fall back to (0,0) below as before; that is a rarer,
            // differently-shaped defect (a malformed source) this fix isn't aimed at.
            let uvArray     = uvs
            let colorArray  = colors  ?? Array(repeating: SIMD4<Float>(1, 1, 1, 1),
                                                count: positions.count)
            let uvCount = uvArray?.count ?? 0
            let normalCount = normalArray.count
            let colorCount = colorArray.count
            for i in 0..<positions.count {
                let n = i < normalCount ? normalArray[i] : SIMD3<Float>(0, 1, 0)
                let uv = uvArray == nil ? triplanarUV(position: positions[i], normal: n)
                    : (i < uvCount ? uvArray![i] : SIMD2<Float>(0, 0))
                let c  = i < colorCount ? colorArray[i] : SIMD4<Float>(1, 1, 1, 1)
                v.append(IlluminatoramaVertex(position: positions[i],
                                               normal: n,
                                               uv: uv,
                                               color: c))
            }
            verts = v
            // Read indices into a `Data` blob in whatever format the renderer
            // will dispatch. SceneKit elements can be 1/2/4 bytes per index;
            // promote 1-byte → 2-byte (no MTLIndexType.uint8) and keep 2/4
            // verbatim.
            rawIndex = readIndexData(element: element)
        }
        guard let raw = rawIndex else {
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
    static func groundQuad(for floor: SCNFloor,
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

    // ── Unit-primitive size recovery ──────────────────────────────────────────

    /// Map a geometry's raw object-space vertex positions onto the coordinate
    /// frame SceneKit itself reports via `geometry.boundingBox`.
    ///
    /// SceneKit's parametric primitives (`SCNBox`, `SCNCylinder`, `SCNCone`,
    /// `SCNSphere`, `SCNCapsule`, `SCNTube`, `SCNPlane`, …) keep their vertex
    /// SOURCE in a canonical UNIT object space and fold the authored
    /// width/height/length/radius into geometry-intrinsic scaling that
    /// `sources(for:.vertex)` does NOT bake into the returned positions — only
    /// `boundingBox` reflects the real extent. Reading the unit source verbatim
    /// is what shrank large SCNBoxes to a unit cube and sheared rotated/flat
    /// SCNCylinders (the rotation in the node's model matrix then acts on the
    /// wrong proportions).
    ///
    /// For a hand-built `SCNGeometry` the auto-computed `boundingBox` equals the
    /// vertex AABB exactly, so the mapping is the identity and the original
    /// positions are returned untouched (no allocation, no float drift) — the
    /// early-out below covers that common case. The correction only fires when
    /// the reported bounds actually disagree with the vertex bounds, i.e. for a
    /// SceneKit parametric primitive whose source is unit-normalised.
    static func normaliseToBoundingBox(_ positions: [SIMD3<Float>],
                                                geometry: SCNGeometry) -> [SIMD3<Float>] {
        let bb = geometry.boundingBox
        let bbMin = SIMD3<Float>(Float(bb.min.x), Float(bb.min.y), Float(bb.min.z))
        let bbMax = SIMD3<Float>(Float(bb.max.x), Float(bb.max.y), Float(bb.max.z))
        let bbSize = bbMax - bbMin
        // A zero / inverted reported box (e.g. a primitive that hasn't realised
        // its layout) is not a usable correction target — keep the source verbatim.
        guard bbSize.x > 0 || bbSize.y > 0 || bbSize.z > 0 else { return positions }

        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let vertSize = hi - lo

        // Per-axis scale that maps the vertex AABB onto the reported box. A
        // degenerate axis (flat plane → vertSize ≈ 0, e.g. SCNPlane in Z) keeps
        // scale 1 so we never divide by ~0; its extent is 0 either way.
        let eps: Float = 1e-6
        var scale = SIMD3<Float>(1, 1, 1)
        for a in 0..<3 {
            if vertSize[a] > eps && bbSize[a] > eps { scale[a] = bbSize[a] / vertSize[a] }
        }
        let vertCenter = (lo + hi) * 0.5
        let bbCenter = (bbMin + bbMax) * 0.5

        // Identity early-out: scale ≈ 1 on every axis AND centres aligned →
        // hand-built mesh, return the source untouched (preserves exact
        // positions / winding precision; the overwhelmingly common path).
        let scaleId = abs(scale.x - 1) < 1e-5 && abs(scale.y - 1) < 1e-5 && abs(scale.z - 1) < 1e-5
        let centreId = simd_length(bbCenter - vertCenter) < 1e-5 * max(1, simd_length(bbSize))
        if scaleId && centreId { return positions }

        return positions.map { bbCenter + ($0 - vertCenter) * scale }
    }
}
