import Metal
import simd
import Signack

/// DH-1466 — a MAX-depth mip pyramid of the host's depth (`Shaders/SignackHiZ.metal`), the
/// occluder record for Signack's occlusion test (`SignackGPUOcclusion`).
///
/// Level 0 is half the depth's size rounded up to a power of two, each further level half again,
/// down to 1 × 1; every texel holds the farthest depth over its footprint. Standard depth (0 near, 1 far) only.
/// Built by the host after its depth pass; reallocated when the depth's size changes.
public final class SignackHiZ: @unchecked Sendable {
    public let device: MTLDevice
    public private(set) var texture: MTLTexture?
    public private(set) var sourceSize = SIMD2<Int>(0, 0)
    private var levels: [MTLTexture] = []

    public init(device: MTLDevice) { self.device = device }

    public var mipCount: Int { levels.count }
    public var bytes: Int { texture.map { $0.allocatedSize } ?? 0 }

    /// Encode the pyramid build from `depth` (a depth texture, shader-readable). False when the
    /// pyramid could not be allocated (the caller then must not test against it).
    @discardableResult
    public func encodeBuild(depth: MTLTexture, pipelines: SignackGPUPipelines, commandBuffer cb: MTLCommandBuffer) -> Bool {
        guard depth.width > 1, depth.height > 1, ensure(width: depth.width, height: depth.height),
              let enc = cb.makeComputeCommandEncoder() else { return false }
        enc.label = "Signack.hiz"
        for (l, dst) in levels.enumerated() {
            enc.setComputePipelineState(l == 0 ? pipelines.hizSeed : pipelines.hizReduce)
            enc.setTexture(l == 0 ? depth : levels[l - 1], index: 0)
            enc.setTexture(dst, index: 1)
            let p = l == 0 ? pipelines.hizSeed : pipelines.hizReduce
            let tw = p.threadExecutionWidth, th = max(1, p.maxTotalThreadsPerThreadgroup / tw)
            enc.dispatchThreads(MTLSize(width: dst.width, height: dst.height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tw, height: min(th, 8), depth: 1))
        }
        enc.endEncoding()
        return true
    }

    private func ensure(width: Int, height: Int) -> Bool {
        if texture != nil, sourceSize == SIMD2(width, height) { return true }
        // Power-of-two level 0 (`SignackOcclusion.pyramid`): Metal's mip chain halves by floor, so
        // only a power of two keeps every level an exact 2× reduction of the one above.
        let w0 = SignackOcclusion.pow2((width + 1) / 2), h0 = SignackOcclusion.pow2((height + 1) / 2)
        var n = 1, w = w0, h = h0
        while w > 1 || h > 1 { w = max(1, w / 2); h = max(1, h / 2); n += 1 }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: w0, height: h0, mipmapped: true)
        d.mipmapLevelCount = n
        d.usage = [.shaderRead, .shaderWrite, .pixelFormatView]
        d.storageMode = .private
        guard let t = device.makeTexture(descriptor: d) else { texture = nil; levels = []; return false }
        t.label = "Signack.hiz"
        levels = (0..<n).compactMap { t.makeTextureView(pixelFormat: .r32Float, textureType: .type2D,
                                                        levels: $0..<($0 + 1), slices: 0..<1) }
        guard levels.count == n else { texture = nil; levels = []; return false }
        texture = t
        sourceSize = SIMD2(width, height)
        return true
    }
}

/// One frame's occlusion source for a cut or a batch: the pyramid, and the transform from the
/// instance's space (object for a cut, world for a batch) to the clip space the pyramid's depth was
/// rendered in.
public struct SignackGPUOcclusion: @unchecked Sendable {
    public var hiz: SignackHiZ
    public var toClip: simd_float4x4
    /// Pixels (of the depth) each footprint grows by — TAA jitter.
    public var dilatePixels: Float
    public init(hiz: SignackHiZ, toClip: simd_float4x4, dilatePixels: Float = 2) {
        self.hiz = hiz; self.toClip = toClip; self.dilatePixels = dilatePixels
    }

    /// The mirror of `SignackOcclusionUniforms` (96 B) in the shaders.
    struct Uniforms {
        var toClip: simd_float4x4
        var depthSize: SIMD2<Float>
        var mipCount: UInt32
        var enabled: UInt32
        var dilate: Float
        var pad0: Float = 0, pad1: Float = 0, pad2: Float = 0
    }

    static func bind(_ o: SignackGPUOcclusion?, _ enc: MTLComputeCommandEncoder, buffer: Int, texture: Int,
                     fallback: MTLTexture) {
        var u = Uniforms(toClip: o?.toClip ?? matrix_identity_float4x4,
                         depthSize: SIMD2(Float(o?.hiz.sourceSize.x ?? 1), Float(o?.hiz.sourceSize.y ?? 1)),
                         mipCount: UInt32(o?.hiz.mipCount ?? 0),
                         enabled: o?.hiz.texture != nil ? 1 : 0, dilate: o?.dilatePixels ?? 0)
        enc.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: buffer)
        enc.setTexture(o?.hiz.texture ?? fallback, index: texture)
    }
}
