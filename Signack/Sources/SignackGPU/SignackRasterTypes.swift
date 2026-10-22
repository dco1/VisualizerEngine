import Metal
import simd

/// One window-space triangle for the software rasterizer — the byte-for-byte mirror of
/// `SignackRasterTriangle` in `SignackRaster.metal` (48 B; the shader static_asserts the size).
///
/// The HOST's vertex stage writes these (it owns the vertex deformation, so the soft path lands on
/// the same surface as its hardware path): window x, y in pixels (y down, origin top-left, pixel
/// centres at +½) and depth z/w in [0, 1] per vertex, plus the id the rasterizer stores where the
/// triangle wins. `idX` bit 31 is an INPUT flag on the way in (cull back faces) and the
/// front-facing bit on the way out; the other 63 bits are the host's.
public struct SignackRasterTriangle: Sendable, Equatable {
    public var v0x, v0y, v0z: Float
    public var v1x, v1y, v1z: Float
    public var v2x, v2y, v2z: Float
    public var idX: UInt32
    public var idY: UInt32
    public var flags: UInt32 = 0

    /// Set on `idX` by the host: drop the triangle when it faces away (single-sided geometry).
    public static let cullBackFaces: UInt32 = 0x8000_0000

    public init(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>, idX: UInt32, idY: UInt32) {
        (v0x, v0y, v0z) = (a.x, a.y, a.z)
        (v1x, v1y, v1z) = (b.x, b.y, b.z)
        (v2x, v2y, v2z) = (c.x, c.y, c.z)
        self.idX = idX
        self.idY = idY
    }
}

/// The per-dispatch constants — mirror of `SignackRasterUniforms` (32 B).
struct SignackRasterUniforms {
    var width: UInt32, height: UInt32
    var x0: UInt32, y0: UInt32, x1: UInt32, y1: UInt32
    var capacity: UInt32
    var maxSpan: UInt32
}

/// A pixel rectangle `[x, x + width) × [y, y + height)` the rasterizer may write.
public struct SignackRasterRect: Sendable, Equatable {
    public var x: Int, y: Int, width: Int, height: Int
    public init(x: Int, y: Int, width: Int, height: Int) {
        self.x = x; self.y = y; self.width = width; self.height = height
    }
}

/// S4 — how a GPU cut routes its on-screen triangles: those projecting under `maxArea` px² (and with
/// a box under `maxSpan` px per axis) go to the software rasterizer's range of the output. Set on
/// `SignackGPUCut.routing` per frame; nil routes nothing (exactly S2).
public struct SignackGPURouting {
    /// Object → clip (Metal convention), unjittered — the instance's model matrix times the camera.
    public var objectToClip: simd_float4x4
    /// The target the software rasterizer draws (px) — the host's G-buffer.
    public var viewport: SIMD2<Float>
    public var maxArea: Float
    public var maxSpan: Float
    /// The host's vertex buffer (the hierarchy indexes it); a float3 position at offset 0.
    public var vertices: MTLBuffer
    public var vertexStride: Int

    public init(objectToClip: simd_float4x4, viewport: SIMD2<Float>, maxArea: Float, maxSpan: Float = 4,
                vertices: MTLBuffer, vertexStride: Int) {
        self.objectToClip = objectToClip
        self.viewport = viewport
        self.maxArea = maxArea
        self.maxSpan = maxSpan
        self.vertices = vertices
        self.vertexStride = vertexStride
    }
}

/// Mirror of `SignackGPURouteUniforms` (SignackCut.metal, 80 B).
struct SignackGPURouteUniforms {
    var objectToClip: simd_float4x4
    var viewport: SIMD2<Float>
    var maxSpan: Float
    var vertexStride: UInt32
}
