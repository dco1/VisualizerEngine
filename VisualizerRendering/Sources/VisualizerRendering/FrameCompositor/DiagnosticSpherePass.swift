import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── DIAGNOSTIC 3D SPHERE PASS ───────────────────────────────────────────────
//
// Encodes the `diagnosticSphere` Metal compute kernel into the compositor's
// HDR target. Exists purely to prove that a real ray-traced 3D sphere
// (Lambert + Blinn-Phong specular, OFFSET highlight, lit/unlit hemispheres)
// renders correctly through the Metal compositor pipeline — the same path
// the burst particles, clouds, and shells use.
//
// If this sphere reads as 3D on screen, the answer for the particle
// shader is "do what this kernel does, but scaled per-particle":
//   • Compute a per-fragment surface normal of an analytic sphere
//   • Shade with Lambert + specular against a light direction
//   • The OFFSET specular highlight is what creates the 3D illusion
//
// If this sphere DOESN'T render or doesn't read as 3D, we've discovered
// something deeper about how the compositor pipeline behaves and that
// changes the whole approach.
//
// USAGE
//   let pass = DiagnosticSpherePass(engine: .shared)
//   pass.center = SIMD3(0, 22, -10)
//   pass.radius = 4.0
//   pass.encode(into: cb, target: comp,
//               cameraPosition: camPos,
//               viewProjection: vp)

@MainActor
public final class DiagnosticSpherePass {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "DiagnosticSpherePass")

    public let device: MTLDevice
    public let engine: SimEngine

    private let pipeline: MTLComputePipelineState
    private let uniformBuffer: MTLBuffer

    // ── Live tunables ───────────────────────────────────────────────────────
    public var center: SIMD3<Float> = SIMD3(0, 22, -10)
    public var radius: Float = 4.0
    public var albedo: SIMD3<Float> = SIMD3(0.85, 0.88, 0.92)
    public var lightDirection: SIMD3<Float> = SIMD3(0.6, 0.7, 0.4)

    public init?(engine: SimEngine) {
        guard let pipeline = engine.pipeline("diagnosticSphere") else {
            Self.log.error("diagnosticSphere pipeline lookup failed — metal kernel missing from library")
            return nil
        }
        let device = engine.device
        guard let uBuf = device.makeBuffer(
            length: MemoryLayout<DiagnosticSphereUniforms>.stride,
            options: .storageModeShared
        ) else {
            Self.log.error("DiagnosticSphere uniform buffer alloc failed")
            return nil
        }
        uBuf.label = "DiagnosticSphere.uniforms"
        self.device = device
        self.engine = engine
        self.pipeline = pipeline
        self.uniformBuffer = uBuf
    }

    /// Encode a single dispatch that ray-traces the sphere into the
    /// compositor's HDR target. Should be called AFTER the cloud /
    /// shells / streaks passes and BEFORE the tonemap.
    public func encode(
        into cb: MTLCommandBuffer,
        target: FrameCompositor,
        cameraPosition: SIMD3<Float>,
        viewProjection: float4x4
    ) {
        let invVP = simd_inverse(viewProjection)

        // Build uniforms — invVP rows are columns of the column-major
        // matrix transposed into row-major for the shader's dot-product
        // unprojection.
        let lightDirNormalized = simd_normalize(lightDirection)
        var u = DiagnosticSphereUniforms(
            invViewProjRow0: SIMD4(invVP.columns.0.x, invVP.columns.1.x, invVP.columns.2.x, invVP.columns.3.x),
            invViewProjRow1: SIMD4(invVP.columns.0.y, invVP.columns.1.y, invVP.columns.2.y, invVP.columns.3.y),
            invViewProjRow2: SIMD4(invVP.columns.0.z, invVP.columns.1.z, invVP.columns.2.z, invVP.columns.3.z),
            invViewProjRow3: SIMD4(invVP.columns.0.w, invVP.columns.1.w, invVP.columns.2.w, invVP.columns.3.w),
            cameraPos: SIMD4(cameraPosition.x, cameraPosition.y, cameraPosition.z, 0),
            sphereCenterRadius: SIMD4(center.x, center.y, center.z, radius),
            lightDir: SIMD4(lightDirNormalized.x, lightDirNormalized.y, lightDirNormalized.z, 0),
            albedo: SIMD4(albedo.x, albedo.y, albedo.z, 0)
        )
        memcpy(uniformBuffer.contents(), &u, MemoryLayout<DiagnosticSphereUniforms>.size)

        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "DiagnosticSpherePass"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(target.hdrColorTexture, index: 0)
        enc.setBuffer(uniformBuffer, offset: 0, index: 0)

        let w = target.size.x
        let h = target.size.y
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let groups = MTLSize(
            width:  (w + tg.width  - 1) / tg.width,
            height: (h + tg.height - 1) / tg.height,
            depth: 1
        )
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}

// ── Uniform struct mirroring the Metal-side `DiagnosticSphereUniforms` ──────
//
// Field order MUST match the Metal struct byte-for-byte. Using SIMD4
// throughout sidesteps the SIMD3 alignment trap (SIMD3 has stride 16 in
// Swift but packed_float3 in Metal has stride 12, which silently corrupts
// uniform layouts unless you control both sides).
private struct DiagnosticSphereUniforms {
    var invViewProjRow0: SIMD4<Float>
    var invViewProjRow1: SIMD4<Float>
    var invViewProjRow2: SIMD4<Float>
    var invViewProjRow3: SIMD4<Float>
    var cameraPos: SIMD4<Float>
    var sphereCenterRadius: SIMD4<Float>
    var lightDir: SIMD4<Float>
    var albedo: SIMD4<Float>
}
