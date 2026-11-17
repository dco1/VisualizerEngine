import simd

/// One shadow map a frame rasterises — the key a mesh's per-view record is filed under
/// (`IlluminatoramaMesh.shadowViewDrawOffsets`, Daydream DH-1465).
public enum IlluminatoramaShadowView: Hashable, Sendable {
    /// Sun cascade slice.
    case cascade(Int)
    /// Spot / area-light atlas slice.
    case spot(Int)
    /// Point-light cube face slice (cube × 6 + face).
    case point(Int)
}

extension IlluminatoramaRenderer {
    /// The shadow views THIS frame will rasterise, each with the view-projection its pass draws with
    /// — sun cascades first, then spot slices, then point faces. Valid from `onEncodeFrameCompute` on
    /// (the cascade, spot and point matrices are refit before it runs). A view whose pass the frame
    /// then skips (static reuse) costs nothing; a view not listed draws a mesh's whole shadow record.
    public func shadowViewsThisFrame() -> [(view: IlluminatoramaShadowView, viewProjection: simd_float4x4)] {
        var out: [(view: IlluminatoramaShadowView, viewProjection: simd_float4x4)] = []
        if shadowsEnabled {
            for c in 0..<sunShadowCascadeCount { out.append((.cascade(c), shd.cascadeVPs[c])) }
        }
        if spotShadowsEnabled {
            for (s, caster) in shd.spotShadowCasters.enumerated() { out.append((.spot(s), caster.matrix)) }
        }
        if pointShadowsEnabled, let faceBuffer = pointShadowFaceBuffer {
            let faces = faceBuffer.contents().bindMemory(to: simd_float4x4.self, capacity: shd.pointShadowCubeCapacity * 6)
            for pl in pointLights where pl.shadowCubeIndex >= 0 {
                for f in 0..<6 {
                    let slice = Int(pl.shadowCubeIndex) * 6 + f
                    out.append((.point(slice), faces[slice]))
                }
            }
        }
        return out
    }
}
