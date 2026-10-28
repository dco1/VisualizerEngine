import Metal

/// The shared GPU broadphase grid: counting-sort of particle ids into a hashed uniform grid (`SimSpatialHash.metal`).
/// Cloth, soft bodies and ropes all build theirs through this type, so a speed-up or fix lands in every solver, and
/// their self-collide kernels walk the grid with the one `simHashCell` in `SimSpatialHash.h`.
///
/// After `encodeBuild`, `counts[b]` is bucket b's size and `sorted[offsets[b] ..< offsets[b] + counts[b]]` its particle ids.
/// The particle record only has to hold a float4 position at `posOffset` (PBDParticle, SMParticle: stride 32, offset 0).
@MainActor
public final class SimSpatialHash {
    public let tableSize: Int
    public let counts: MTLBuffer      // atomic_uint[tableSize] (count → scatter cursor → count)
    public let offsets: MTLBuffer     // uint[tableSize]
    public let sorted: MTLBuffer      // uint[maxParticles]
    private let clear, count, scan, scatter: MTLComputePipelineState

    public init?(device: MTLDevice, maxParticles: Int, tableSize: Int = 1 << 14, label: String) {
        let cache = SimPipelineCache.shared
        guard let ccBuf = device.makeBuffer(length: tableSize * 4, options: .storageModePrivate),
              let coBuf = device.makeBuffer(length: tableSize * 4, options: .storageModePrivate),
              let srBuf = device.makeBuffer(length: max(1, maxParticles) * 4, options: .storageModePrivate),
              let c0 = cache.pipelineState(name: "simHashClear", device: device),
              let c1 = cache.pipelineState(name: "simHashCount", device: device),
              let c2 = cache.pipelineState(name: "simHashScan", device: device),
              let c3 = cache.pipelineState(name: "simHashScatter", device: device) else { return nil }
        self.tableSize = tableSize
        counts = ccBuf; offsets = coBuf; sorted = srBuf
        ccBuf.label = "\(label).hashCounts"; coBuf.label = "\(label).hashOffsets"; srBuf.label = "\(label).hashSorted"
        clear = c0; count = c1; scan = c2; scatter = c3
    }

    /// Clear → count → scan → scatter over `particleCount` records of `byteStride` bytes.
    public func encodeBuild(_ cb: MTLCommandBuffer, particles: MTLBuffer, particleCount n: Int, cellSize: Float,
                            byteStride: Int = 32, posOffset: Int = 0, label: String) {
        guard n > 0 else { return }
        var uniforms = SimHashBuildUniformsCPU(count: UInt32(n), tableSize: UInt32(tableSize), cellSize: cellSize,
                                               byteStride: UInt32(byteStride), posOffset: UInt32(posOffset),
                                               _pad0: 0, _pad1: 0, _pad2: 0)
        var table = UInt32(tableSize)
        func dispatch(_ e: MTLComputeCommandEncoder, _ p: MTLComputePipelineState, _ n: Int) {
            e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: max(1, min(n, p.maxTotalThreadsPerThreadgroup)), height: 1, depth: 1))
        }
        if let e = cb.makeComputeCommandEncoder() {
            e.label = "\(label).hashClear"; e.setComputePipelineState(clear)
            e.setBuffer(counts, offset: 0, index: 0); e.setBytes(&table, length: 4, index: 1)
            dispatch(e, clear, tableSize); e.endEncoding()
        }
        if let e = cb.makeComputeCommandEncoder() {
            e.label = "\(label).hashCount"; e.setComputePipelineState(count)
            e.setBuffer(particles, offset: 0, index: 0); e.setBuffer(counts, offset: 0, index: 1)
            e.setBytes(&uniforms, length: MemoryLayout<SimHashBuildUniformsCPU>.stride, index: 2)
            dispatch(e, count, n); e.endEncoding()
        }
        if let e = cb.makeComputeCommandEncoder() {
            e.label = "\(label).hashScan"; e.setComputePipelineState(scan)
            e.setBuffer(counts, offset: 0, index: 0); e.setBuffer(offsets, offset: 0, index: 1)
            e.setBytes(&table, length: 4, index: 2)
            e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: min(1024, scan.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            e.endEncoding()
        }
        if let e = cb.makeComputeCommandEncoder() {
            e.label = "\(label).hashScatter"; e.setComputePipelineState(scatter)
            e.setBuffer(particles, offset: 0, index: 0); e.setBuffer(counts, offset: 0, index: 1)
            e.setBuffer(offsets, offset: 0, index: 2); e.setBuffer(sorted, offset: 0, index: 3)
            e.setBytes(&uniforms, length: MemoryLayout<SimHashBuildUniformsCPU>.stride, index: 4)
            dispatch(e, scatter, n); e.endEncoding()
        }
    }
}

// Keep field order IDENTICAL to `struct SimHashBuildUniforms` in SimSpatialHash.h.
struct SimHashBuildUniformsCPU {
    var count: UInt32
    var tableSize: UInt32
    var cellSize: Float
    var byteStride: UInt32
    var posOffset: UInt32
    var _pad0: UInt32, _pad1: UInt32, _pad2: UInt32
}
