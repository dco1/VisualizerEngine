import Metal

/// Which kind of raster pass is drawing — the one fact an indirectly-drawn mesh needs to pick its
/// argument record (`IlluminatoramaMesh.shadowIndirectDrawOffset`).
enum IlluminatoramaDrawPass {
    /// The camera's G-buffer pass.
    case primary
    /// Sun cascades, spot/area atlas slices, point cube faces.
    case shadow
}

extension MTLRenderCommandEncoder {
    /// Draw `mesh` for `instanceCount` instances — or, when the mesh carries a GPU-written
    /// indirect argument buffer, with those arguments (the shadow record for a shadow pass when the
    /// mesh has one). A `tracedOnly` mesh is never rasterised — this is the one place every raster
    /// pass draws through, so the skip lives here.
    ///
    /// **Why two records (Signack S2, Daydream DH-1461).** A GPU-selected cut is frustum-culled to
    /// the CAMERA for the G-buffer, but a caster off screen still throws a shadow on screen, so the
    /// shadow passes draw the whole LOD cut. Signack compacts off-screen clusters first and
    /// on-screen ones after them, so both records index ONE buffer: shadow = [0, S + P), primary =
    /// [S, S + P). An indirect mesh draws as a single instance (the record's `instanceCount`).
    @MainActor func drawMesh(_ mesh: IlluminatoramaMesh, instanceCount: Int,
                             pass: IlluminatoramaDrawPass = .primary) {
        guard !mesh.tracedOnly else { return }
        if let args = mesh.indirectDrawBuffer {
            let offset = pass == .shadow ? (mesh.shadowIndirectDrawOffset ?? mesh.indirectDrawOffset)
                                         : mesh.indirectDrawOffset
            drawIndexedPrimitives(type: .triangle, indexType: mesh.indexType,
                                  indexBuffer: mesh.indexBuffer, indexBufferOffset: 0,
                                  indirectBuffer: args, indirectBufferOffset: offset)
        } else {
            drawIndexedPrimitives(type: .triangle, indexCount: mesh.indexCount,
                                  indexType: mesh.indexType, indexBuffer: mesh.indexBuffer,
                                  indexBufferOffset: 0, instanceCount: instanceCount,
                                  baseVertex: 0, baseInstance: 0)
        }
    }
}

extension IlluminatoramaMesh {
    /// Triangles one instance of this mesh draws in the camera pass, for the draw census. For an
    /// indirect mesh it is the GPU's own count from the last completed frame (read from shared
    /// memory without waiting, so it may lag by the frames in flight) — `indexCount` there is only
    /// the buffer's capacity. A private argument buffer cannot be read: capacity is reported.
    public var drawnTriangleEstimate: Int {
        guard let args = indirectDrawBuffer, args.storageMode != .private,
              args.length >= indirectDrawOffset + 4 else { return indexCount / 3 }
        return Int(args.contents().load(fromByteOffset: indirectDrawOffset, as: UInt32.self)) / 3
    }

    /// How far (world units) `applyTreeWind` can move a vertex of a mesh whose packed wind weights
    /// peak at `max`, at gust strength `strength` (= `treeWindStrength × instance.windScale`):
    /// horizontal bend ≤ sway·s·(0.5 + gust) with gust ≤ 1, droop ≤ 0.18·sway·s, flutter ≤
    /// 0.05 + 0.05 + 0.035 per unit weight. The ONE copy of that bound — instance culling (DH-0534)
    /// and Signack's per-cluster culling both pad by it.
    nonisolated public static func windReach(strength: Float, max w: (sway: Float, flutter: Float)) -> Float {
        guard strength > 0 else { return 0 }
        return strength * (w.sway * (1.5 + 0.18) + w.flutter * (0.05 + 0.05 + 0.035))
    }
}
