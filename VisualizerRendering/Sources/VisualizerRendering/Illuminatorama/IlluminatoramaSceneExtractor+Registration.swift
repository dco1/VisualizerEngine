import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

extension IlluminatoramaSceneExtractor {

    // ── Mesh registration ─────────────────────────────────────────────────────

    /// Convert (and cache) the SCNGeometry, and push it into the renderer's
    /// mesh table on the same frame the first instance referencing it
    /// shows up. Returns nil for unsupported geometry shapes — caller
    /// should skip the node rather than render with a placeholder.
    func registerMesh(_ geometry: SCNGeometry,
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

    func simdFromSCN(_ m: SCNMatrix4) -> simd_float4x4 {
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
