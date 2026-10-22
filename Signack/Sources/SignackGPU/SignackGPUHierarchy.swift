import Metal
import Signack

/// A cluster LOD DAG made RESIDENT on the GPU — once per baked object TYPE (a tree spec), shared by
/// every placed instance of it.
///
/// Holds the cluster records and the WHOLE hierarchy's index buffer (every level, ≈ 2× level 0).
/// An instance owns only its selection output (`SignackGPUCut`); its transform lives in the host's
/// draw instance. The vertex buffer is the host's (the hierarchy indexes it; Signack never copies
/// vertices).
public final class SignackGPUHierarchy: @unchecked Sendable {
    public let clusterBuffer: MTLBuffer
    public let indexBuffer: MTLBuffer
    public let clusterCount: Int
    /// Indices in level 0 — the largest cut there can be, so the size of every instance's output.
    public let maxCutIndices: Int
    /// Bytes resident for this type (clusters + all-level indices).
    public var residentBytes: Int { clusterBuffer.length + indexBuffer.length }

    public init?(device: MTLDevice, hierarchy h: SignackHierarchy, label: String = "Signack") {
        guard !h.clusters.isEmpty, !h.indices.isEmpty else { return nil }
        let packed = h.clusters.map(SignackGPUCluster.init)
        guard let cb = packed.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                                                 options: .storageModeShared) }),
              let ib = h.indices.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                                                    options: .storageModeShared) })
        else { return nil }
        cb.label = "\(label).clusters"
        ib.label = "\(label).dagIndices"
        clusterBuffer = cb
        indexBuffer = ib
        clusterCount = packed.count
        maxCutIndices = 3 * h.sourceTriangleCount
    }
}
