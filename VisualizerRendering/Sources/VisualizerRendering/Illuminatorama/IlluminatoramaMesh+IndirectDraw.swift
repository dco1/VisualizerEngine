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
    ///
    /// A mesh with a software-raster record (`softRasterDrawOffset`, S4) draws that record too — in
    /// every shadow pass, and in a camera pass unless the caller rasterises it in software this frame
    /// (`withSoft: false`).
    ///
    /// A shadow pass names its `view` (DH-1465): a mesh with a record for that view draws only the
    /// part of its cut that can land in that light's frustum.
    @MainActor func drawMesh(_ mesh: IlluminatoramaMesh, instanceCount: Int,
                             pass: IlluminatoramaDrawPass = .primary, view: IlluminatoramaShadowView? = nil,
                             withSoft: Bool = true) {
        guard !mesh.tracedOnly else { return }
        if let args = mesh.indirectDrawBuffer {
            let perView = view.flatMap { mesh.shadowViewDrawOffsets.isEmpty ? nil : mesh.shadowViewDrawOffsets[$0] }
            let offset = pass == .shadow ? (perView ?? mesh.shadowIndirectDrawOffset ?? mesh.indirectDrawOffset)
                                         : mesh.indirectDrawOffset
            drawIndexedPrimitives(type: .triangle, indexType: mesh.indexType,
                                  indexBuffer: mesh.indexBuffer, indexBufferOffset: 0,
                                  indirectBuffer: args, indirectBufferOffset: offset)
            for rec in mesh.additionalIndirectRecords {   // DH-1470 — a packed batch's other buckets
                drawIndexedPrimitives(type: .triangle, indexType: mesh.indexType,
                                      indexBuffer: mesh.indexBuffer, indexBufferOffset: 0,
                                      indirectBuffer: args, indirectBufferOffset: rec.offset(pass: pass, view: view))
            }
            if withSoft, let soft = mesh.softRasterDrawOffset {
                drawIndexedPrimitives(type: .triangle, indexType: mesh.indexType,
                                      indexBuffer: mesh.indexBuffer, indexBufferOffset: 0,
                                      indirectBuffer: args, indirectBufferOffset: soft)
            }
        } else {
            drawIndexedPrimitives(type: .triangle, indexCount: mesh.indexCount,
                                  indexType: mesh.indexType, indexBuffer: mesh.indexBuffer,
                                  indexBufferOffset: 0, instanceCount: instanceCount,
                                  baseVertex: 0, baseInstance: 0)
        }
    }
}

extension IlluminatoramaMesh {
    /// The instance buffer (and byte offset) a raster pass binds for this mesh's draw group: the
    /// frame's grouped ring at the group's offset, or — Signack S6 — the GPU-compacted records at 0.
    func instanceBinding(ring: MTLBuffer, groupOffset: Int) -> (buffer: MTLBuffer, offset: Int) {
        gpuInstanceBuffer.map { ($0, 0) } ?? (ring, groupOffset)
    }

    /// Triangles one instance of this mesh draws in the camera pass, for the draw census. For an
    /// indirect mesh it is the GPU's own count from the last completed frame (read from shared
    /// memory without waiting, so it may lag by the frames in flight) — `indexCount` there is only
    /// the buffer's capacity. A private argument buffer cannot be read: capacity is reported.
    public var drawnTriangleEstimate: Int {
        guard let args = indirectDrawBuffer, args.storageMode != .private,
              args.length >= indirectDrawOffset + 4 else { return indexCount / 3 }
        func record(_ o: Int) -> Int {
            guard args.length >= o + 8 else { return 0 }
            var n = Int(args.contents().load(fromByteOffset: o, as: UInt32.self))
            if gpuInstanceBuffer != nil {   // S6: a batched record draws many instances
                n *= Int(args.contents().load(fromByteOffset: o + 4, as: UInt32.self))
            }
            return n
        }
        var n = record(indirectDrawOffset)
        for rec in additionalIndirectRecords { n += record(rec.primary) }   // DH-1470
        if let soft = softRasterDrawOffset, args.length >= soft + 4 {   // S4: the soft-routed part
            n += Int(args.contents().load(fromByteOffset: soft, as: UInt32.self))
        }
        return n / 3
    }

    /// How far (world units) `applyTreeWind` can move a vertex of a mesh whose packed wind weights
    /// peak at `max`, at gust strength `strength` (= `treeWindStrength × instance.windScale`):
    /// horizontal bend ≤ sway·s·(0.5 + gust) with gust ≤ 1, droop ≤ 0.18·sway·s, flutter ≤
    /// 0.05 + 0.05 + 0.035 per unit weight. The ONE copy of that bound — instance culling (DH-0534)
    /// and Signack's per-cluster culling both pad by it.
    nonisolated public static func windReach(strength: Float, max w: (sway: Float, flutter: Float)) -> Float {
        guard strength > 0 else { return 0 }
        return strength * (w.sway * (1.5 + 0.18) + w.flutter * (0.05 + 0.05 + 0.035) + branchWindReach)
    }

    /// Signack S5 — the limb layer's reach per unit strength (`applyBranchWind`,
    /// IlluminatoramaBranchWind.h): `kBranchWindAmp` 0.55 × (1 along the wind + 0.45 across + 0.25 up),
    /// limb weight ≤ 1. Added for every mesh (it cannot see which vertices carry a limb record): a
    /// few centimetres of extra padding at a yard breeze, a valid bound always.
    nonisolated public static let branchWindReach: Float = 0.55 * (1 + 0.45 + 0.25)
}

/// One more indirect record an `IlluminatoramaMesh` draws (DH-1470 — a packed batch's bucket): its
/// camera-pass record, the shadow passes' record (nil = the camera's), and per-light shadow views'.
public struct IlluminatoramaIndirectRecord: Sendable, Equatable {
    public var primary: Int
    public var shadow: Int?
    public var shadowViews: [IlluminatoramaShadowView: Int]

    public init(primary: Int, shadow: Int? = nil, shadowViews: [IlluminatoramaShadowView: Int] = [:]) {
        self.primary = primary; self.shadow = shadow; self.shadowViews = shadowViews
    }

    /// The byte offset a pass draws this record from — the same choice the mesh makes for its own.
    func offset(pass: IlluminatoramaDrawPass, view: IlluminatoramaShadowView?) -> Int {
        guard pass == .shadow else { return primary }
        return view.flatMap { shadowViews[$0] } ?? shadow ?? primary
    }
}
