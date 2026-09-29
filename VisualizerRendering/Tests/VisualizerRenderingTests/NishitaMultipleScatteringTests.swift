import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// VZ-0159 — physically based MULTIPLE SCATTERING for the Nishita sky
/// (`VolumetricCloudRenderer.Params.atmosphereMultipleScattering`, opt-in), pinned as numbers:
///
///   1. The GPU LUT sky against a BRUTE-FORCE reference of the same atmosphere
///      (`NishitaReferenceIntegrator`: double precision, orders 1…6 resolved by backward path
///      tracing with the single scatter at every vertex integrated deterministically, the true
///      Rayleigh / Cornette–Shanks phases, planet shadow at every quadrature point, Lambertian
///      ground 0.1; 400 000 paths per configuration) — stored below as `reference`.
///   2. The zenith and the 30°-up / 90°-from-the-sun twilight luminance against MEASURED clear
///      skies: Koomen, Lock, Packer, Scolnik, Tousey & Hulburt (1952), "Measurements of the
///      Brightness of the Twilight Sky", JOSA 42:353, Table II (Maryland, 30 m, photopic), and
///      Patat, Ugolnikov & Postylyakov (2006), "UBVRI twilight sky brightness at ESO-Paranal",
///      A&A 455:385, Table 1 (V).
///   3. The flag OFF is bit-identical to the pre-VZ-0159 shader (engine commit 2fd9cc9): dome /
///      IBL kernel, cloud-light prepass and Illuminatorama's in-view kernel.
///   4. The lunar hand-off: with the flag on the sky never dips below the moonlit sky.
///   5. The in-view pass draws the same sky as the dome (the LUTs reach it through the uniforms
///      buffer), and the GPU cost of the LUT build and the march.
///
/// Geometry of every probe: the viewer 1 m up at the pole (the shader's), the sun at azimuth 0
/// (+X) at elevation `el`; "window" = 30° up, 90° in azimuth from the sun (VZ-0159's south
/// window at dusk); "antiN" = N° up at the anti-solar azimuth. Radiance is per unit solar
/// irradiance (`atmosphereIntensity` 1, grade off).
final class NishitaMultipleScatteringTests: XCTestCase {

    // MARK: - Reference (NishitaReferenceIntegrator, 4e5 paths, orders 1…6, albedo 0.1, 100 km top)

    enum View: CaseIterable {
        case zenith, window, anti0_5, anti5, anti10, anti15, anti20, anti30
        var dir: SIMD3<Float> {
            func anti(_ el: Float) -> SIMD3<Float> { SIMD3(-cos(el * .pi / 180), sin(el * .pi / 180), 0) }
            switch self {
            case .zenith:  return SIMD3(0, 1, 0)
            case .window:  return SIMD3(0, sin(Float.pi / 6), cos(Float.pi / 6))
            case .anti0_5: return anti(0.5)
            case .anti5:   return anti(5)
            case .anti10:  return anti(10)
            case .anti15:  return anti(15)
            case .anti20:  return anti(20)
            case .anti30:  return anti(30)
            }
        }
    }
    struct R {
        let el: Float; let view: View; let rgb: SIMD3<Float>; let relSE: Float
        init(_ el: Float, _ view: View, _ r: Float, _ g: Float, _ b: Float, _ se: Float) {
            self.el = el; self.view = view; rgb = SIMD3(r, g, b); relSE = se
        }
    }
    /// Radiance per unit solar irradiance (all orders), and the standard error of the Monte
    /// Carlo part relative to the total.
    static let reference: [R] = [
        R(10, .zenith, 3.03546e-03, 5.49639e-03, 1.16563e-02, 0.0007),
        R(10, .window, 5.88333e-03, 1.04431e-02, 2.06102e-02, 0.0008),
        R(10, .anti0_5, 3.94246e-02, 4.15401e-02, 3.52618e-02, 0.0014),
        R(10, .anti5, 3.57298e-02, 4.83225e-02, 5.10669e-02, 0.0009),
        R(10, .anti10, 2.35517e-02, 3.64206e-02, 5.02760e-02, 0.0008),
        R(10, .anti15, 1.68414e-02, 2.76871e-02, 4.38071e-02, 0.0008),
        R(10, .anti20, 1.28381e-02, 2.18309e-02, 3.73701e-02, 0.0007),
        R(10, .anti30, 8.32889e-03, 1.47137e-02, 2.75596e-02, 0.0007),
        R(5, .zenith, 2.49614e-03, 3.84196e-03, 7.28840e-03, 0.0006),
        R(5, .window, 4.93012e-03, 7.36932e-03, 1.27865e-02, 0.0007),
        R(5, .anti0_5, 2.90435e-02, 2.28835e-02, 1.43843e-02, 0.0014),
        R(5, .anti5, 2.98006e-02, 3.18275e-02, 2.54369e-02, 0.0008),
        R(5, .anti10, 2.01717e-02, 2.53756e-02, 2.85431e-02, 0.0007),
        R(5, .anti15, 1.46319e-02, 1.97804e-02, 2.62748e-02, 0.0006),
        R(5, .anti20, 1.12501e-02, 1.58278e-02, 2.30987e-02, 0.0006),
        R(5, .anti30, 7.40652e-03, 1.08800e-02, 1.76173e-02, 0.0006),
        R(3, .zenith, 2.13049e-03, 2.86956e-03, 5.10662e-03, 0.0006),
        R(3, .window, 4.22215e-03, 5.49307e-03, 8.84786e-03, 0.0007),
        R(3, .anti0_5, 2.11959e-02, 1.30169e-02, 7.38496e-03, 0.0016),
        R(3, .anti5, 2.47205e-02, 2.14891e-02, 1.42378e-02, 0.0008),
        R(3, .anti10, 1.72187e-02, 1.81898e-02, 1.79083e-02, 0.0007),
        R(3, .anti15, 1.26230e-02, 1.45172e-02, 1.73108e-02, 0.0006),
        R(3, .anti20, 9.77600e-03, 1.17789e-02, 1.56284e-02, 0.0006),
        R(3, .anti30, 6.48689e-03, 8.21835e-03, 1.22375e-02, 0.0006),
        R(2, .zenith, 1.88392e-03, 2.31275e-03, 4.00084e-03, 0.0006),
        R(2, .window, 3.73206e-03, 4.41184e-03, 6.87233e-03, 0.0007),
        R(2, .anti0_5, 1.57950e-02, 8.17950e-03, 4.92192e-03, 0.0019),
        R(2, .anti5, 2.10625e-02, 1.56741e-02, 9.41113e-03, 0.0008),
        R(2, .anti10, 1.50323e-02, 1.39889e-02, 1.28468e-02, 0.0006),
        R(2, .anti15, 1.11503e-02, 1.14103e-02, 1.28554e-02, 0.0006),
        R(2, .anti20, 8.67824e-03, 9.36030e-03, 1.18323e-02, 0.0006),
        R(2, .anti30, 5.79524e-03, 6.61199e-03, 9.45882e-03, 0.0006),
        R(0, .zenith, 1.20731e-03, 1.17547e-03, 2.03759e-03, 0.0005),
        R(0, .window, 2.37516e-03, 2.20691e-03, 3.41887e-03, 0.0006),
        R(0, .anti0_5, 3.55130e-03, 1.87315e-03, 2.04711e-03, 0.0029),
        R(0, .anti5, 1.08000e-02, 5.07201e-03, 3.12085e-03, 0.0008),
        R(0, .anti10, 8.83368e-03, 5.74879e-03, 5.01442e-03, 0.0006),
        R(0, .anti15, 6.85540e-03, 5.11391e-03, 5.54708e-03, 0.0005),
        R(0, .anti20, 5.46190e-03, 4.38526e-03, 5.38375e-03, 0.0005),
        R(0, .anti30, 3.73949e-03, 3.24379e-03, 4.55312e-03, 0.0005),
        R(-1, .zenith, 8.12757e-04, 7.05423e-04, 1.28011e-03, 0.0005),
        R(-1, .window, 1.58778e-03, 1.30984e-03, 2.12629e-03, 0.0006),
        R(-1, .anti0_5, 1.44491e-03, 8.96939e-04, 1.19932e-03, 0.0031),
        R(-1, .anti5, 5.20024e-03, 1.90362e-03, 1.58023e-03, 0.0010),
        R(-1, .anti10, 5.21012e-03, 2.75745e-03, 2.59809e-03, 0.0006),
        R(-1, .anti15, 4.28402e-03, 2.68010e-03, 3.06955e-03, 0.0005),
        R(-1, .anti20, 3.51124e-03, 2.40097e-03, 3.09652e-03, 0.0005),
        R(-1, .anti30, 2.47172e-03, 1.85421e-03, 2.72366e-03, 0.0005),
        R(-2, .zenith, 4.72366e-04, 3.67805e-04, 7.18094e-04, 0.0005),
        R(-2, .window, 9.14965e-04, 6.73992e-04, 1.18115e-03, 0.0005),
        R(-2, .anti0_5, 6.17361e-04, 3.92710e-04, 6.32242e-04, 0.0028),
        R(-2, .anti5, 1.42398e-03, 5.26081e-04, 7.57892e-04, 0.0015),
        R(-2, .anti10, 2.29312e-03, 9.97101e-04, 1.12977e-03, 0.0007),
        R(-2, .anti15, 2.12288e-03, 1.11706e-03, 1.43232e-03, 0.0005),
        R(-2, .anti20, 1.83493e-03, 1.07349e-03, 1.52076e-03, 0.0005),
        R(-2, .anti30, 1.35796e-03, 8.87586e-04, 1.41550e-03, 0.0004),
        R(-3, .zenith, 2.29839e-04, 1.63695e-04, 3.51745e-04, 0.0005),
        R(-3, .window, 4.41442e-04, 2.96464e-04, 5.75889e-04, 0.0006),
        R(-3, .anti0_5, 2.31464e-04, 1.55710e-04, 2.96136e-04, 0.0026),
        R(-3, .anti5, 2.25908e-04, 1.65745e-04, 3.51719e-04, 0.0024),
        R(-3, .anti10, 6.47112e-04, 2.62599e-04, 4.20876e-04, 0.0011),
        R(-3, .anti15, 7.61476e-04, 3.52040e-04, 5.49404e-04, 0.0007),
        R(-3, .anti20, 7.27801e-04, 3.75128e-04, 6.17756e-04, 0.0006),
        R(-3, .anti30, 5.90174e-04, 3.43974e-04, 6.17941e-04, 0.0005),
        R(-4, .zenith, 9.17845e-05, 6.20250e-05, 1.47422e-04, 0.0006),
        R(-4, .window, 1.74940e-04, 1.11341e-04, 2.41367e-04, 0.0007),
        R(-4, .anti0_5, 8.16220e-05, 5.81491e-05, 1.23103e-04, 0.0028),
        R(-4, .anti5, 7.86487e-05, 6.46492e-05, 1.49348e-04, 0.0029),
        R(-4, .anti10, 1.15212e-04, 6.60019e-05, 1.54903e-04, 0.0023),
        R(-4, .anti15, 1.85571e-04, 8.87756e-05, 1.80522e-04, 0.0015),
        R(-4, .anti20, 2.08164e-04, 1.03864e-04, 2.06747e-04, 0.0011),
        R(-4, .anti30, 1.95439e-04, 1.07668e-04, 2.22436e-04, 0.0008),
        R(-6, .zenith, 8.75021e-06, 6.11188e-06, 1.69435e-05, 0.0018),
        R(-6, .window, 1.69404e-05, 1.12606e-05, 2.88594e-05, 0.0019),
        R(-6, .anti0_5, 1.14800e-05, 8.29337e-06, 1.77572e-05, 0.0051),
        R(-6, .anti5, 1.37566e-05, 1.04828e-05, 2.28967e-05, 0.0049),
        R(-6, .anti10, 1.13215e-05, 9.40548e-06, 2.44305e-05, 0.0054),
        R(-6, .anti15, 9.64091e-06, 8.06557e-06, 2.33136e-05, 0.0052),
        R(-6, .anti20, 1.00372e-05, 7.49384e-06, 2.19447e-05, 0.0048),
        R(-6, .anti30, 1.12992e-05, 7.48215e-06, 2.09414e-05, 0.0036),
        R(-8, .zenith, 6.69681e-07, 5.16273e-07, 1.80841e-06, 0.0076),
        R(-8, .window, 1.39844e-06, 1.03616e-06, 3.27731e-06, 0.0067),
        R(-8, .anti0_5, 2.00656e-06, 1.30905e-06, 2.50485e-06, 0.0135),
        R(-8, .anti5, 2.77857e-06, 1.84484e-06, 3.49529e-06, 0.0109),
        R(-8, .anti10, 2.34625e-06, 1.71025e-06, 4.01763e-06, 0.0089),
        R(-8, .anti15, 1.93917e-06, 1.47947e-06, 3.98092e-06, 0.0090),
        R(-8, .anti20, 1.59346e-06, 1.26083e-06, 3.70009e-06, 0.0091),
        R(-8, .anti30, 1.18640e-06, 9.54493e-07, 3.07858e-06, 0.0101),
        R(-9, .zenith, 2.41654e-07, 1.94748e-07, 7.43627e-07, 0.0330),
        R(-9, .window, 5.20956e-07, 4.03294e-07, 1.36115e-06, 0.0285),
        R(-9, .anti0_5, 8.39648e-07, 5.20929e-07, 9.64790e-07, 0.0101),
        R(-9, .anti5, 1.25288e-06, 7.81030e-07, 1.37892e-06, 0.0105),
        R(-9, .anti10, 1.11577e-06, 7.78781e-07, 1.67970e-06, 0.0323),
        R(-9, .anti15, 9.22459e-07, 6.80614e-07, 1.73319e-06, 0.0340),
        R(-9, .anti20, 7.61371e-07, 5.86105e-07, 1.66819e-06, 0.0354),
        R(-9, .anti30, 5.82388e-07, 4.50500e-07, 1.43162e-06, 0.0458),
        R(-10, .zenith, 1.05891e-07, 8.28255e-08, 3.14285e-07, 0.0179),
        R(-10, .window, 2.27174e-07, 1.67745e-07, 5.62849e-07, 0.0122),
        R(-10, .anti0_5, 3.54423e-07, 2.11094e-07, 4.21100e-07, 0.0171),
        R(-10, .anti5, 5.77141e-07, 3.35148e-07, 5.90907e-07, 0.0135),
        R(-10, .anti10, 5.12444e-07, 3.26781e-07, 7.19366e-07, 0.0114),
        R(-10, .anti15, 4.28217e-07, 2.89102e-07, 7.36917e-07, 0.0114),
        R(-10, .anti20, 3.53698e-07, 2.48729e-07, 7.05987e-07, 0.0117),
        R(-10, .anti30, 2.64320e-07, 1.92328e-07, 6.05395e-07, 0.0122),
        R(-12, .zenith, 2.21563e-08, 1.57496e-08, 6.00845e-08, 0.0167),
        R(-12, .window, 4.83078e-08, 3.22012e-08, 1.09153e-07, 0.0155),
        R(-12, .anti0_5, 6.16527e-08, 3.35341e-08, 6.77196e-08, 0.0388),
        R(-12, .anti5, 1.04733e-07, 5.64734e-08, 9.80927e-08, 0.0166),
        R(-12, .anti10, 1.02473e-07, 5.94020e-08, 1.22090e-07, 0.0170),
        R(-12, .anti15, 8.82904e-08, 5.34711e-08, 1.31700e-07, 0.0161),
        R(-12, .anti20, 7.78208e-08, 4.80495e-08, 1.30473e-07, 0.0195),
        R(-12, .anti30, 5.62244e-08, 3.67694e-08, 1.13044e-07, 0.0154),
        R(-15, .zenith, 1.80424e-09, 1.11257e-09, 4.37164e-09, 0.0280),
        R(-15, .window, 3.79848e-09, 2.34382e-09, 8.56590e-09, 0.0549),
        R(-15, .anti0_5, 3.48636e-09, 1.72563e-09, 4.54566e-09, 0.0421),
        R(-15, .anti5, 6.46005e-09, 2.87615e-09, 5.26470e-09, 0.0228),
        R(-15, .anti10, 6.85329e-09, 3.38590e-09, 7.64445e-09, 0.0277),
        R(-15, .anti15, 6.52284e-09, 3.43587e-09, 8.81411e-09, 0.0270),
        R(-15, .anti20, 5.60559e-09, 3.09613e-09, 8.92518e-09, 0.0293),
        R(-15, .anti30, 4.42584e-09, 2.51871e-09, 7.79637e-09, 0.0290),
    ]

    static let referenceElevations: [Float] = [10, 5, 3, 2, 0, -1, -2, -3, -4, -6, -8, -9, -10, -12, -15]

    // MARK: - Harness

    static let luma = SIMD3<Float>(0.2126, 0.7152, 0.0722)
    static func toSun(_ elDeg: Float) -> SIMD3<Float> {
        SIMD3(cos(elDeg * .pi / 180), sin(elDeg * .pi / 180), 0)
    }
    static func stops(_ a: Float, _ b: Float) -> Float { log2(a / b) }

    static let shadersDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Sources/VisualizerRendering/Shaders")

    /// VolumetricSky.metal + a probe kernel that evaluates the dome's own per-texel atmosphere
    /// (`nishitaAtmosphereColor`) for exact directions — no equirect quantisation.
    static let probeSource = """
    kernel void msProbeSky(device float4* out [[buffer(0)]],
                           constant SkyUniforms &u [[buffer(1)]],
                           device const float4* lut [[buffer(2)]],
                           device const float4* dirs [[buffer(3)]],
                           constant float4 &flags [[buffer(4)]],
                           uint tid [[thread_position_in_grid]]) {
        out[tid] = float4(nishitaAtmosphereColor(normalize(dirs[tid].xyz), u, lut, flags.x > 0.5f), 0.0f);
    }
    """

    @MainActor final class Harness {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let probe: MTLComputePipelineState
        let cloud: VolumetricCloudRenderer

        init() throws {
            let engine = SimEngine.shared
            device = engine.device
            queue = engine.commandQueue
            let src = try MetalSourceLoader.source(contentsOf: NishitaMultipleScatteringTests.shadersDir
                .appendingPathComponent("VolumetricSky.metal"))
            let lib = try device.makeLibrary(source: src + "\n" + NishitaMultipleScatteringTests.probeSource, options: nil)
            guard let fn = lib.makeFunction(name: "msProbeSky") else { throw XCTSkip("probe missing") }
            probe = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness, once
            cloud = VolumetricCloudRenderer(engine: engine, resolution: SIMD2(64, 32), iblResolution: SIMD2(32, 16))
            cloud.minRenderInterval = 0
        }

        /// Test-only: wait for everything queued so far.
        func drain() {
            guard let cb = queue.makeCommandBuffer() else { return }
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test-only drain before reading results
        }

        /// The sky the renderer's kernels see for `p`, along `dirs` (renders once — which builds the
        /// LUTs when the flag is on — then evaluates the atmosphere from the renderer's own buffer).
        func sky(_ p: VolumetricCloudRenderer.Params, _ dirs: [SIMD3<Float>], lutBound: Bool = true) throws -> [SIMD3<Float>] {
            cloud.render(params: p)
            drain()
            let n = dirs.count
            var d4 = dirs.map { SIMD4<Float>($0, 0) }
            var flags = SIMD4<Float>(lutBound ? 1 : 0, 0, 0, 0)
            guard let db = device.makeBuffer(bytes: &d4, length: n * 16, options: .storageModeShared),
                  let ob = device.makeBuffer(length: n * 16, options: .storageModeShared),
                  let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no resources") }
            let u = cloud.skyUniformsBuffer
            enc.setComputePipelineState(probe)
            enc.setBuffer(ob, offset: 0, index: 0)
            enc.setBuffer(u, offset: 0, index: 1)
            enc.setBuffer(u, offset: lutBound ? VolumetricCloudRenderer.atmosphereLUTOffset : 0, index: 2)
            enc.setBuffer(db, offset: 0, index: 3)
            enc.setBytes(&flags, length: 16, index: 4)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(n, 64), height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness
            if let e = cb.error { throw e }
            let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: n)
            return (0..<n).map { SIMD3(o[$0].x, o[$0].y, o[$0].z) }
        }
    }

    /// Mirror of VolumetricSky.metal's `CloudInViewUniforms` (112 bytes).
    struct InViewUniforms { var invVP: simd_float4x4; var cam: SIMD4<Float>; var extra: SIMD4<Float>; var night: SIMD4<Float> }

    /// A depth buffer cleared to 1 (= all sky for the in-view kernel's clip).
    static func clearedDepth(_ device: MTLDevice, _ queue: MTLCommandQueue, _ W: Int, _ H: Int) throws -> MTLTexture {
        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: W, height: H, mipmapped: false)
        dd.usage = [.renderTarget, .shaderRead]; dd.storageMode = .private
        guard let depth = device.makeTexture(descriptor: dd), let cb = queue.makeCommandBuffer() else { throw XCTSkip("no depth") }
        let rp = MTLRenderPassDescriptor()
        rp.depthAttachment.texture = depth; rp.depthAttachment.loadAction = .clear
        rp.depthAttachment.clearDepth = 1; rp.depthAttachment.storeAction = .store
        cb.makeRenderCommandEncoder(descriptor: rp)?.endEncoding()
        cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        return depth
    }

    /// Physical sky, no clouds / cirrus / celestials, unit irradiance, no grade.
    static func bareSky(el: Float, ms: Bool) -> VolumetricCloudRenderer.Params {
        var p = VolumetricCloudRenderer.Params()
        p.atmosphere = .nishita
        p.atmosphereMultipleScattering = ms
        p.atmosphereIntensity = 1
        p.skySaturation = 1
        p.skyBlueLift = 1
        p.coverage = 0
        p.density = 0
        p.cirrusCoverage = 0
        p.celestialsInDome = false
        p.sunDir = -toSun(el)
        return p
    }

    // MARK: - 0. The Swift mirror of the LUT layout

    @MainActor
    func testLUTLayoutMirrorMatchesTheShader() throws {
        let engine = SimEngine.shared
        guard let pso = engine.pipeline("volSkyAtmosphereLUTLayout"),
              let out = engine.device.makeBuffer(length: 48, options: .storageModeShared),
              let cb = engine.commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            return XCTFail("volSkyAtmosphereLUTLayout missing")
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: test harness
        let v = out.contents().bindMemory(to: SIMD4<Int32>.self, capacity: 3)
        let L = VolumetricCloudRenderer.AtmosphereLUT.self
        XCTAssertEqual(Int(v[0].x), L.float4Count, "kLUTLength")
        XCTAssertEqual(Int(v[0].y), 1, "kLUTTrans")
        XCTAssertEqual(Int(v[0].z), 1 + L.transW * L.transH, "kLUTMS")
        XCTAssertEqual(Int(v[1].x), L.transW); XCTAssertEqual(Int(v[1].y), L.transH)
        XCTAssertEqual(Int(v[1].z), L.msW); XCTAssertEqual(Int(v[1].w), L.msH)
        XCTAssertEqual(Int(v[2].x), L.spectralBands, "kSpecBands")
        XCTAssertEqual(Int(v[2].y), L.float4Count - 1, "kLUTBandStride")
        XCTAssertEqual(VolumetricCloudRenderer.skyUniformsBufferLength,
                       VolumetricCloudRenderer.atmosphereLUTOffset + L.regionFloat4Count * 16, "room for every band")
        XCTAssertGreaterThanOrEqual(VolumetricCloudRenderer.atmosphereLUTOffset, MemoryLayout<SkyUniforms>.stride)
        XCTAssertEqual(VolumetricCloudRenderer.atmosphereLUTOffset % 256, 0)
        // The GPU-owned cloud-lighting tail is still the last 3 float4s of the uniforms.
        XCTAssertEqual(SkyUniforms.hostPrefixLength, MemoryLayout<SkyUniforms>.stride - 3 * MemoryLayout<SIMD4<Float>>.stride)
        // Off by default, and only the nishita sky has it.
        var p = VolumetricCloudRenderer.Params()
        XCTAssertFalse(p.atmosphereMultipleScattering)
        XCTAssertEqual(SkyUniforms(params: p, time: 0).msParams.x, 0)
        p.atmosphereMultipleScattering = true
        XCTAssertEqual(SkyUniforms(params: p, time: 0).msParams.x, 1)
        p.atmosphere = .proceduralGradient
        XCTAssertEqual(SkyUniforms(params: p, time: 0).msParams.x, 0, "needs the nishita sky")
    }

    // MARK: - 1. Against the brute-force reference

    @MainActor
    func testLUTSkyMatchesTheBruteForceReference() throws {
        let h = try Harness()
        var worstZW: Float = 0, worstAnti: Float = 0
        var table = "el     view      ref_Y       gpu_Y       stops   ref_rgb             gpu_rgb\n"
        for el in Self.referenceElevations {
            let rows = Self.reference.filter { $0.el == el }
            let got = try h.sky(Self.bareSky(el: el, ms: true), rows.map { $0.view.dir })
            for (r, g) in zip(rows, got) {
                let yr = simd_dot(r.rgb, Self.luma), yg = simd_dot(g, Self.luma)
                let s = Self.stops(yg, yr)
                func c(_ v: SIMD3<Float>) -> String {
                    let m = max(v.x, v.y, v.z)
                    return String(format: "(%.2f,%.2f,%.2f)", v.x / m, v.y / m, v.z / m)
                }
                table += String(format: "%5.1f  %@  %.4e  %.4e  %+.3f  %@  %@\n", el,
                                "\(r.view)".padding(toLength: 8, withPad: " ", startingAt: 0), yr, yg, s, c(r.rgb), c(g))
                XCTAssertTrue(s.isFinite, "\(el)° \(r.view): \(g)")
                switch r.view {
                case .zenith, .window:
                    worstZW = max(worstZW, abs(s))
                    XCTAssertLessThan(abs(s), 0.25, "\(el)° \(r.view): \(s) stops from the reference")
                    // The non-solar sky stays BLUE all the way through (the reference's is).
                    XCTAssertGreaterThan(g.z, g.x, "\(el)° \(r.view) blue > red: \(g)")
                    XCTAssertGreaterThan(g.z, g.y, "\(el)° \(r.view) blue > green: \(g)")
                default:
                    worstAnti = max(worstAnti, abs(s))
                    XCTAssertLessThan(abs(s), 0.6, "\(el)° \(r.view): \(s) stops from the reference")
                }
                // Hue: red/blue within 35 % of the reference's — 60 % on the anti-solar horizon in
                // mid twilight, where Mie's last scatter is isotropic (the documented model limit:
                // the Earth-shadow band prints lavender-blue where the reference is violet-blue).
                let rbRef = r.rgb.x / r.rgb.z, rbGot = g.x / g.z
                let hueTol: Float = r.view == .anti0_5 ? 1.6 : 1.35
                XCTAssertLessThan(abs(log(rbGot / rbRef)), log(hueTol), "\(el)° \(r.view) R/B \(rbGot) vs \(rbRef)")
            }
        }
        print("── NishitaMS vs brute-force reference (per unit E) ──\n" + table
              + String(format: "worst |stops|: zenith/window %.3f, anti-solar %.3f", worstZW, worstAnti))
    }

    /// The anti-solar sky in civil twilight: the bluish EARTH-SHADOW band on the horizon under the
    /// pink BELT OF VENUS — in the reference, and on the GPU.
    @MainActor
    func testBeltOfVenusStandsAboveTheBluishEarthShadow() throws {
        let h = try Harness()
        let shadow: [View] = [.anti0_5, .anti5], belt: [View] = [.anti10, .anti15, .anti20]
        for el: Float in [-2, -3, -4] {
            let rows = Self.reference.filter { $0.el == el }
            let got = try h.sky(Self.bareSky(el: el, ms: true), rows.map { $0.view.dir })
            for (label, rgbOf) in [("reference", { (v: View) in rows.first { $0.view == v }!.rgb }),
                                   ("gpu", { (v: View) in got[rows.firstIndex { $0.view == v }!] })] {
                let shadowRB = shadow.map { rgbOf($0).x / rgbOf($0).z }.min()!
                let beltRB = belt.map { rgbOf($0).x / rgbOf($0).z }.max()!
                XCTAssertGreaterThanOrEqual(beltRB, 1.0, "\(label) \(el)°: the belt is pink (R ≥ B): \(beltRB)")
                XCTAssertLessThan(shadowRB, 0.8 * beltRB, "\(label) \(el)°: the shadow band is bluer than the belt")
                XCTAssertLessThanOrEqual(shadowRB, 1.0, "\(label) \(el)°: the shadow band is blue-violet (R ≤ B): \(shadowRB)")
            }
        }
    }

    /// Single scatter goes EXACTLY black once the 60 km shell is in the Earth's shadow (−7.8°) and
    /// prints the window magenta-red on the way (VZ-0159's measured colours, reproduced here from
    /// the flag-off path). Multiple scattering keeps it lit and blue.
    @MainActor
    func testTwilightStaysBlueAndLitPastTheShellShadow() throws {
        let h = try Harness()
        let window = View.window.dir
        // VZ-0159's measurements (graded like Digital Clock: saturation 1.1, blue lift 1.3).
        let issue: [(Float, SIMD3<Float>)] = [(0.4, SIMD3(0.74, 0.64, 1.00)), (-2, SIMD3(0.87, 0.55, 1.00)),
                                              (-4.15, SIMD3(0.89, 0.44, 1.00)), (-6, SIMD3(1.00, 0.33, 0.59))]
        for (el, want) in issue {
            var p = Self.bareSky(el: el, ms: false)
            p.atmosphereIntensity = 20; p.skySaturation = 1.1; p.skyBlueLift = 1.3
            let s = try h.sky(p, [window])[0]
            let n = s / max(s.x, s.y, s.z)
            XCTAssertLessThan(simd_length(n - want), 0.05, "flag off at \(el)° reproduces VZ-0159: \(n) vs \(want)")
            p.atmosphereMultipleScattering = true
            let m = try h.sky(p, [window])[0]
            XCTAssertGreaterThan(m.z, m.x, "flag on at \(el)°: the window is blue, not magenta: \(m)")
        }
        for el: Float in [-8, -10, -12] {
            let off = try h.sky(Self.bareSky(el: el, ms: false), [View.zenith.dir, window])
            let on = try h.sky(Self.bareSky(el: el, ms: true), [View.zenith.dir, window])
            for (o, m) in zip(off, on) {
                XCTAssertEqual(simd_reduce_add(o), 0, "single scatter at \(el)° is black: \(o)")
                XCTAssertGreaterThan(simd_dot(m, Self.luma), 0, "multiple scattering at \(el)° is not")
                XCTAssertGreaterThan(m.z, max(m.x, m.y), "and it is blue: \(m)")
            }
        }
    }

    /// The stored table is reproducible from the integrator in this target
    /// (`NishitaReferenceIntegrator.swift`). Opt-in — the test target builds at -Onone, where a
    /// full sweep takes hours: `VIZ_NISHITA_REFERENCE_PATHS=20000 ./Scripts/test.sh --filter
    /// NishitaMultipleScatteringTests/testReferenceIntegrator`. Checks three configurations
    /// against the stored values within 4 combined standard errors.
    func testReferenceIntegratorReproducesTheStoredTable() throws {
        guard let paths = ProcessInfo.processInfo.environment["VIZ_NISHITA_REFERENCE_PATHS"].flatMap({ Int($0) }) else {
            throw XCTSkip("opt-in: set VIZ_NISHITA_REFERENCE_PATHS")
        }
        typealias NR = NishitaReference
        var atm = NR.Atmosphere()
        atm.albedo = 0.1
        atm.Rt = atm.Rg + 100e3
        let tab = NR.OpticalDepthTable(atm: atm)
        let viewer = NR.V3(0, atm.Rg + 1, 0)
        for (el, view) in [(Float(-12), View.zenith), (-6, .window), (-3, .anti0_5)] {
            let d = view.dir, s = Self.toSun(el)
            let r = NR.mcRadiance(atm: atm, tab: tab, x0: viewer, w0: NR.V3(Double(d.x), Double(d.y), Double(d.z)),
                                  s: NR.V3(Double(s.x), Double(s.y), Double(s.z)), paths: paths, maxOrder: 6, seed: 991)
            let total = r.order[1...].reduce(NR.V3.zero, +)
            let stored = Self.reference.first { $0.el == el && $0.view == view }!
            let y = NR.luma(total), ys = Double(simd_dot(stored.rgb, Self.luma))
            let se = sqrt(pow(NR.luma(r.stderrMS), 2) + pow(Double(stored.relSE) * ys, 2))
            XCTAssertLessThan(abs(y - ys), 4 * se, "\(el)° \(view): \(y) vs stored \(ys) (σ \(se))")
        }
    }

    // MARK: - 2. Against measured twilight skies

    /// Koomen et al. (1952) Table II — Maryland, 30 m, clear, photopic; candles per square foot.
    /// H = +5, +3, 0, −3, −6, −9, −12, −15 (the −15° values are the site's night sky).
    static let koomenH: [Float] = [5, 3, 0, -3, -6, -9, -12]
    static let koomenZenith: [Float] = [43, 35, 15, 2, 0.06, 0.0015, 0.00012]
    static let koomenZ90P30: [Float] = [80, 66, 32, 3.5, 0.09, 0.0034, 0.00025]
    /// Extraterrestrial solar illuminance (IES): the photometric value of unit irradiance here.
    static let solarLux: Float = 127_500
    static let cdPerFt2: Float = 10.7639

    @MainActor
    func testZenithTwilightCurveAgainstMeasuredData() throws {
        let h = try Harness()
        var report = "H     meas_zen   model_zen  stops    meas_Z90P30 model     stops   (cd/m²)\n"
        for (i, H) in Self.koomenH.enumerated() {
            let s = try h.sky(Self.bareSky(el: H, ms: true), [View.zenith.dir, View.window.dir])
            let mz = simd_dot(s[0], Self.luma) * Self.solarLux
            let mw = simd_dot(s[1], Self.luma) * Self.solarLux
            let kz = Self.koomenZenith[i] * Self.cdPerFt2, kw = Self.koomenZ90P30[i] * Self.cdPerFt2
            let ez = Self.stops(mz, kz), ew = Self.stops(mw, kw)
            report += String(format: "%+3.0f  %.3e  %.3e  %+.2f    %.3e    %.3e %+.2f\n", H, kz, mz, ez, kw, mw, ew)
            // To −3° within 0.35 stops. Deeper, the sky is scattered from ever higher air, and the
            // engine's single-exponential Rayleigh profile (H = 8 km) holds 2–3× the real density at
            // 40–80 km (US Standard Atmosphere 1976): +0.6 stops at −6°, ~+1 at −9…−12°. Measured:
            // the same solver on the USSA-76 profile lands within −0.44…−0.15 stops of Koomen there
            // (docs/illuminatorama/nishita-multiple-scattering.md) — the atmosphere, not the solver.
            let tol: Float = H >= -3 ? 0.35 : (H >= -6 ? 0.95 : 1.2)
            XCTAssertLessThan(abs(ez), tol, "zenith at \(H)°: \(ez) stops from Koomen 1952")
            XCTAssertLessThan(abs(ew), tol, "window at \(H)°: \(ew) stops from Koomen 1952")
        }
        // Patat et al. 2006 (V, Paranal 2600 m): SB = 11.84 + 1.518(ζ−95) − 0.057(ζ−95)², 95° ≤ ζ ≤ 105°.
        report += "depression  Patat_V(cd/m²)  model   stops\n"
        for dep: Float in [6, 8, 10, 12] {
            let x = dep - 5
            let sb = 11.84 + 1.518 * x - 0.057 * x * x
            let meas = 10.8e4 * pow(10, -0.4 * sb)
            let s = try h.sky(Self.bareSky(el: -dep, ms: true), [View.zenith.dir])[0]
            let model = simd_dot(s, Self.luma) * Self.solarLux
            report += String(format: "%4.0f°       %.3e       %.3e %+.2f\n", dep, meas, model, Self.stops(model, meas))
            // A 2600 m site (≈30 % dimmer twilight than a low one, Patat 2006) — same caveat.
            XCTAssertLessThan(abs(Self.stops(model, meas)), 1.6, "Patat 2006 at −\(dep)°")
        }
        print("── NishitaMS zenith twilight vs measured ──\n" + report)
    }

    // MARK: - 3. Flag off: bit-identical to the pre-VZ-0159 shader

    /// Engine commit whose VolumetricSky.metal is the pre-VZ-0159 single-scatter shader.
    static let preVZ0159 = "2fd9cc981927fba739b7ace0f3b364c8342feaf8"

    static func gitShow(_ path: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        proc.arguments = ["-C", root.path, "show", "\(preVZ0159):\(path)"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        try proc.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        guard proc.terminationStatus == 0, let s = String(data: data, encoding: .utf8), !s.isEmpty else {
            throw XCTSkip("engine commit \(preVZ0159) not reachable from this checkout")
        }
        return s
    }

    /// The pre-VZ-0159 library, from git (its header too), compiled like the current one.
    static func preVZ0159Library(_ device: MTLDevice) throws -> MTLLibrary {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("preVZ0159-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let base = "VisualizerRendering/Sources/VisualizerRendering/Shaders/"
        try gitShow(base + "IlluminatoramaNightSky.h").write(to: dir.appendingPathComponent("IlluminatoramaNightSky.h"),
                                                              atomically: true, encoding: .utf8)
        try gitShow(base + "VolumetricSky.metal").write(to: dir.appendingPathComponent("VolumetricSky.metal"),
                                                         atomically: true, encoding: .utf8)
        return try MetalSourceLoader.makeLibrary(device: device, contentsOf: dir.appendingPathComponent("VolumetricSky.metal"))
    }

    /// The old struct is this one without `msParams` and the later `nightSkyE` (the skyglow +
    /// limiting-magnitude cluster), which sit together just before the GPU-owned tail.
    static func preVZ0159Bytes(_ u: SkyUniforms) -> [UInt8] {
        var u = u
        let all = withUnsafeBytes(of: &u) { Array($0) }
        let off = MemoryLayout<SkyUniforms>.offset(of: \SkyUniforms.msParams)!
        let tail = MemoryLayout<SkyUniforms>.offset(of: \SkyUniforms.cloudLitSun)!
        precondition(MemoryLayout<SkyUniforms>.offset(of: \SkyUniforms.nightSkyE)! == off + 16 && tail == off + 32,
                     "msParams + nightSkyE must be the two clusters right before the GPU-written tail")
        return Array(all[0..<off]) + Array(all[tail...])
    }

    /// Configurations that exercise every consumer of the nishita march with the flag OFF.
    @MainActor
    static func flagOffConfigs() -> [(String, VolumetricCloudRenderer.Params)] {
        var out: [(String, VolumetricCloudRenderer.Params)] = []
        var day = VolumetricCloudRenderer.Params()
        day.sunDir = -toSun(50); day.cloudLightingFromAtmosphere = true; day.cloudBaseY = 1300; day.cloudTopY = 2600
        day.horizontalScale = 0.0006; day.cirrusCoverage = 0.4; day.skySaturation = 1.15; day.skyBlueLift = 1.42
        out.append(("day, atmosphere-lit deck + cirrus", day))
        var sunset = day; sunset.sunDir = -toSun(0); sunset.groundFromAtmosphere = true
        out.append(("sunset, ground from atmosphere", sunset))
        func night(_ el: Float) -> VolumetricCloudRenderer.Params {
            var p = VolumetricCloudRenderer.Params()
            p.sunDir = -toSun(el); p.nightSkyModel = .physical; p.moonIntensity = 4.6; p.starBrightness = 1
            p.moonDir = simd_normalize(SIMD3<Float>(-0.95, sin(13 * Float.pi / 180), 0.2))
            p.nightRadiance = 2.4e-9; p.celestialsInDome = false; p.cirrusCoverage = 0.3; p.cirrusThreads = true
            p.cirrusThreadsWeight = 0.5; p.skySaturation = 1.1; p.skyBlueLift = 1.3; p.moonLightGain = 0.8
            return p
        }
        out.append(("civil twilight, physical night + moon", night(-4.15)))
        out.append(("VZ-0159 19:40: sun −7.1°, moon 13° up", night(-7.1)))
        var deep = night(-30); deep.nightRadiance = 2.4e-6; deep.airglow = 1.5; deep.zodiacalLight = 1
        out.append(("night, stylised radiance", deep))
        var proc = VolumetricCloudRenderer.Params()
        proc.atmosphere = .proceduralGradient; proc.sunDir = -toSun(20)
        out.append(("procedural gradient", proc))
        var lava = day; lava.lavaLamp = true; lava.lavaFade = 0.5
        out.append(("lava-lamp cross-fade", lava))
        return out
    }

    @MainActor
    func testFlagOffIsBitIdenticalToThePreVZ0159Shader() throws {
        let engine = SimEngine.shared
        let device = engine.device, queue = engine.commandQueue
        let oldLib = try Self.preVZ0159Library(device)
        let newLib = try MetalSourceLoader.makeLibrary(device: device, contentsOf: Self.shadersDir.appendingPathComponent("VolumetricSky.metal"))
        func pso(_ lib: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
            guard let f = lib.makeFunction(name: name) else { throw XCTSkip("\(name) missing") }
            return try device.makeComputePipelineState(function: f)  // gpu-ok: test harness
        }
        let noiseOwner = VolumetricCloudRenderer(engine: engine, resolution: SIMD2(8, 4), iblResolution: SIMD2(8, 4))
        let noise = noiseOwner.noiseTexture
        guard let lights = device.makeBuffer(length: 256, options: .storageModeShared) else { throw XCTSkip("no buffer") }
        let W = 192, H = 96
        var compared = 0
        func run(_ lib: MTLLibrary, _ bytes: [UInt8], old: Bool) throws -> [UInt32] {
            // float32 target: every bit the kernel computes, not just what survives a half-float.
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: W, height: H, mipmapped: false)
            td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .shared
            let len = old ? bytes.count : VolumetricCloudRenderer.skyUniformsBufferLength
            guard let tex = device.makeTexture(descriptor: td),
                  let ub = device.makeBuffer(length: len, options: .storageModeShared),
                  let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no resources") }
            memset(ub.contents(), 0, len)
            bytes.withUnsafeBytes { ub.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes.count) }
            let p = try pso(lib, "volSkyRender")
            enc.setComputePipelineState(p)
            enc.setTexture(tex, index: 0); enc.setTexture(noise, index: 1)
            enc.setBuffer(ub, offset: 0, index: 0); enc.setBuffer(lights, offset: 0, index: 1)
            enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
            var px = [UInt32](repeating: 0, count: W * H * 4)
            px.withUnsafeMutableBytes { tex.getBytes($0.baseAddress!, bytesPerRow: W * 16, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0) }
            return px
        }
        func prepass(_ lib: MTLLibrary, _ bytes: [UInt8], old: Bool) throws -> [UInt8] {
            let len = old ? bytes.count : VolumetricCloudRenderer.skyUniformsBufferLength
            guard let ub = device.makeBuffer(length: len, options: .storageModeShared),
                  let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no resources") }
            memset(ub.contents(), 0, len)
            bytes.withUnsafeBytes { ub.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes.count) }
            enc.setComputePipelineState(try pso(lib, "volSkyCloudLight"))
            enc.setBuffer(ub, offset: 0, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
            let tailStart = bytes.count - 48
            return Array(UnsafeRawBufferPointer(start: ub.contents() + tailStart, count: 48))
        }
        // Illuminatorama's in-view kernel, the same way (camera at the origin looking along −Z).
        let proj = simd_float4x4(columns: (SIMD4(1, 0, 0, 0), SIMD4(0, 2, 0, 0), SIMD4(0, 0, 0, 1), SIMD4(0, 0, 0.1, 0)))
        let view = simd_float4x4(columns: (SIMD4(-1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, -1, 0), SIMD4(0, 0, 0, 1)))
        var cv = InViewUniforms(invVP: (proj * view).inverse, cam: SIMD4(0, 1.7, 0, 0), extra: SIMD4(3.5, -1, -1, -1), night: SIMD4(3.5, 0, 0, 0))
        let depth = try Self.clearedDepth(device, queue, W, H)
        func inView(_ lib: MTLLibrary, _ bytes: [UInt8], old: Bool) throws -> [UInt32] {
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: W, height: H, mipmapped: false)
            td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .shared
            let len = old ? bytes.count : VolumetricCloudRenderer.skyUniformsBufferLength
            guard let tex = device.makeTexture(descriptor: td),
                  let ub = device.makeBuffer(length: len, options: .storageModeShared),
                  let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no resources") }
            memset(ub.contents(), 0, len)
            bytes.withUnsafeBytes { ub.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes.count) }
            enc.setComputePipelineState(try pso(lib, "illumi_cloud_inview"))
            enc.setTexture(tex, index: 0); enc.setTexture(noise, index: 1); enc.setTexture(depth, index: 2)
            enc.setBuffer(ub, offset: 0, index: 0)
            enc.setBytes(&cv, length: MemoryLayout<InViewUniforms>.stride, index: 1)
            enc.setBuffer(lights, offset: 0, index: 2)
            enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
            var px = [UInt32](repeating: 0, count: W * H * 4)
            px.withUnsafeMutableBytes { tex.getBytes($0.baseAddress!, bytesPerRow: W * 16, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0) }
            return px
        }
        for (name, p) in Self.flagOffConfigs() {
            XCTAssertFalse(p.atmosphereMultipleScattering)
            let u = SkyUniforms(params: p, time: 12.5)
            var nu = u
            let newBytes = withUnsafeBytes(of: &nu) { Array($0) }
            let oldBytes = Self.preVZ0159Bytes(u)
            let a = try run(oldLib, oldBytes, old: true), b = try run(newLib, newBytes, old: false)
            let diffIdx = zip(a, b).enumerated().filter { $0.element.0 != $0.element.1 }.map { $0.offset }
            let diff = diffIdx.count
            if let i = diffIdx.first {
                print("DIFF \(name): texel \((i / 4) % W),\((i / 4) / W) lane \(i % 4): \(Float(bitPattern: a[i])) vs \(Float(bitPattern: b[i])) (\(diff) floats)")
            }
            XCTAssertEqual(diff, 0, "dome/IBL kernel, \(name): \(diff) of \(a.count) floats differ")
            let rgbLit = stride(from: 0, to: a.count, by: 4).filter { a[$0] != 0 || a[$0 + 1] != 0 || a[$0 + 2] != 0 }.count
            XCTAssertGreaterThan(rgbLit, a.count / 8, "\(name): the dome rendered (\(rgbLit) lit texels)")
            compared += a.count
            if p.cloudLightingFromAtmosphere {
                let ta = try prepass(oldLib, oldBytes, old: true), tb = try prepass(newLib, newBytes, old: false)
                XCTAssertEqual(ta, tb, "volSkyCloudLight, \(name): GPU-written tail differs")
            }
            let ia = try inView(oldLib, oldBytes, old: true), ib = try inView(newLib, newBytes, old: false)
            let idiff = zip(ia, ib).filter { $0 != $1 }.count
            XCTAssertEqual(idiff, 0, "illumi_cloud_inview, \(name): \(idiff) of \(ia.count) floats differ")
            compared += ia.count
        }
        print("── NishitaMS flag off: \(compared) float32s compared bit-for-bit against engine \(Self.preVZ0159.prefix(7)) ──")
    }

    // MARK: - 4. The lunar hand-off

    /// VZ-0159's second construct: the moon's march was cross-faded in on a FIXED window of sun
    /// elevation (−5.7° … −9.8°), so between −6° and −8° the dying single-scatter sun sky and the
    /// barely weighted moonlit sky summed to LESS than the moonlit sky alone — the Digital Clock
    /// window at 19:40 (sun −7.1°, moon 96 % and 13° up) printed 2.1 stops dimmer than at 23:47.
    /// With multiple scattering the two lights are summed in one march: never below the moon's.
    @MainActor
    func testLunarHandOffNeverDipsBelowTheMoonlitSky() throws {
        let h = try Harness()
        func params(_ el: Float, ms: Bool) -> VolumetricCloudRenderer.Params {
            var p = Self.bareSky(el: el, ms: ms)
            p.atmosphereIntensity = 20
            p.nightSkyModel = .physical
            p.moonIntensity = 4.6
            p.moonDir = simd_normalize(SIMD3<Float>(-cos(13 * Float.pi / 180), sin(13 * Float.pi / 180), 0.25))
            p.moonPhaseOverride = 0.96          // hold the phase: only the sun moves in this sweep
            p.nightRadiance = 2.54e-6 / 1048   // Digital Clock's photometric night unit
            p.airglow = 0; p.zodiacalLight = 0
            return p
        }
        let dir = [View.window.dir]
        let moonOnlyOn = simd_dot(try h.sky(params(-24, ms: true), dir)[0], Self.luma)
        let moonOnlyOff = simd_dot(try h.sky(params(-24, ms: false), dir)[0], Self.luma)
        XCTAssertGreaterThan(moonOnlyOn, 0)
        var prev = Float.infinity
        var minOffRatio = Float.infinity
        var report = "sun    off(stops vs moonlit)  on(stops vs moonlit)\n"
        var el: Float = -3
        while el >= -20 {
            let on = simd_dot(try h.sky(params(el, ms: true), dir)[0], Self.luma)
            let off = simd_dot(try h.sky(params(el, ms: false), dir)[0], Self.luma)
            XCTAssertGreaterThanOrEqual(on, moonOnlyOn * 0.999, "flag on at \(el)°: dips below the moonlit sky")
            XCTAssertLessThanOrEqual(on, prev * 1.001, "flag on at \(el)°: brightens as the sun sinks")
            prev = on
            minOffRatio = min(minOffRatio, off / moonOnlyOff)
            report += String(format: "%+5.1f  %+.2f                  %+.2f\n", el, Self.stops(off, moonOnlyOff), Self.stops(on, moonOnlyOn))
            el -= 0.5
        }
        print("── NishitaMS lunar hand-off, Digital Clock window ──\n" + report)
        // The legacy path's dip is still there (the flag is opt-in) — the defect this replaces.
        XCTAssertLessThan(Self.stops(minOffRatio, 1), -0.5, "flag off: the fixed-window dip")
    }

    // MARK: - 5. Every consumer sees the same sky

    /// `cloudLightingFromAtmosphere` lights the deck from the SAME multiply-scattered sky: past the
    /// 60 km shell's shadow the single-scatter fill is black, the multiple-scattering one is not.
    @MainActor
    func testCloudLightPrepassFollowsTheMultiplyScatteredSky() throws {
        let h = try Harness()
        func tail(_ ms: Bool, _ el: Float) throws -> (sun: SIMD3<Float>, amb: SIMD3<Float>) {
            var p = Self.bareSky(el: el, ms: ms)
            p.atmosphereIntensity = 20; p.cloudLightingFromAtmosphere = true; p.cloudBaseY = 1300
            _ = try h.sky(p, [View.zenith.dir])
            let u = h.cloud.skyUniformsBuffer.contents().bindMemory(to: SkyUniforms.self, capacity: 1).pointee
            return (SIMD3(u.cloudLitSun.x, u.cloudLitSun.y, u.cloudLitSun.z),
                    SIMD3(u.cloudLitAmbient.x, u.cloudLitAmbient.y, u.cloudLitAmbient.z))
        }
        let offDeep = try tail(false, -12), onDeep = try tail(true, -12)
        XCTAssertEqual(simd_reduce_add(offDeep.amb), 0, "single scatter: the deck's sky fill is black at −12°")
        XCTAssertGreaterThan(simd_dot(onDeep.amb, Self.luma), 0, "multiple scattering: it is not")
        XCTAssertGreaterThan(onDeep.amb.z, onDeep.amb.x, "and it is blue")
        let offNoon = try tail(false, 50), onNoon = try tail(true, 50)
        XCTAssertGreaterThan(simd_dot(onNoon.amb, Self.luma), simd_dot(offNoon.amb, Self.luma), "MS adds sky light by day")
        let rs = simd_dot(onNoon.sun, Self.luma) / simd_dot(offNoon.sun, Self.luma)
        XCTAssertEqual(rs, 1, accuracy: 0.02, "the direct sun through the transmittance LUT matches the march's")
    }

    /// Illuminatorama's in-view pass reaches the LUTs through the uniforms buffer it is already
    /// given (`illumi_cloud_inview_ms`, LUT region at buffer(3)) and draws the same sky as the
    /// dome / IBL kernel; with the region not bound it falls back to the single-scatter sky, and
    /// the unchanged `illumi_cloud_inview` draws the single-scatter sky.
    @MainActor
    func testInViewPassDrawsTheSameSkyAsTheDome() throws {
        let h = try Harness()
        guard let inViewMS = SimEngine.shared.pipeline("illumi_cloud_inview_ms"),
              let inViewSS = SimEngine.shared.pipeline("illumi_cloud_inview") else { return XCTFail("in-view kernels missing") }
        let W = 48, H = 24
        let p = Self.bareSky(el: -6, ms: true)
        let proj = simd_float4x4(columns: (SIMD4(1, 0, 0, 0), SIMD4(0, 2, 0, 0), SIMD4(0, 0, 0, 1), SIMD4(0, 0, 0.1, 0)))
        let view = simd_float4x4(columns: (SIMD4(-1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, -1, 0), SIMD4(0, 0, 0, 1)))
        let invVP = (proj * view).inverse
        let dirs: [SIMD3<Float>] = (0..<(W * H)).map { i in
            let uv = SIMD2<Float>((Float(i % W) + 0.5) / Float(W), (Float(i / W) + 0.5) / Float(H))
            let w = invVP * SIMD4<Float>(uv.x * 2 - 1, 1 - uv.y * 2, 1, 1)
            return simd_normalize(SIMD3(w.x, w.y, w.z) / w.w)
        }
        let depth = try Self.clearedDepth(h.device, h.queue, W, H)
        func runInView(_ pso: MTLComputePipelineState, lutBound: Bool) throws -> [SIMD3<Float>] {
            h.cloud.render(params: p); h.drain()
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: W, height: H, mipmapped: false)
            td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .shared
            guard let out = h.device.makeTexture(descriptor: td), let cb = h.queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no resources") }
            var cv = InViewUniforms(invVP: invVP, cam: .zero, extra: SIMD4(0, -1, -1, -1), night: SIMD4(0, lutBound ? 1 : 0, 0, 0))
            let u = h.cloud.skyUniformsBuffer
            enc.setComputePipelineState(pso)
            enc.setTexture(out, index: 0); enc.setTexture(h.cloud.noiseTexture, index: 1); enc.setTexture(depth, index: 2)
            enc.setBuffer(u, offset: 0, index: 0)
            enc.setBytes(&cv, length: MemoryLayout<InViewUniforms>.stride, index: 1)
            enc.setBuffer(h.cloud.fallbackBurstLightBuffer, offset: 0, index: 2)
            if pso === inViewMS { enc.setBuffer(u, offset: lutBound ? VolumetricCloudRenderer.atmosphereLUTOffset : 0, index: 3) }
            enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
            if let e = cb.error { throw e }
            var px = [Float](repeating: 0, count: W * H * 4)
            px.withUnsafeMutableBytes { out.getBytes($0.baseAddress!, bytesPerRow: W * 16, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0) }
            return (0..<(W * H)).map { SIMD3(px[$0 * 4], px[$0 * 4 + 1], px[$0 * 4 + 2]) }
        }
        func worstRel(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>]) -> Float {
            var w: Float = 0
            for (x, y) in zip(a, b) where simd_dot(y, Self.luma) > 0 {
                w = max(w, abs(simd_dot(x, Self.luma) / simd_dot(y, Self.luma) - 1))
            }
            return w
        }
        let msSky = try h.sky(p, dirs)
        let ssSky = try h.sky(p, dirs, lutBound: false)
        XCTAssertTrue(VolumetricCloudRenderer.multipleScatteringRequested(in: h.cloud.skyUniformsBuffer),
                      "the in-view pass can read the flag from the buffer it is given")
        XCTAssertLessThan(worstRel(try runInView(inViewMS, lutBound: true), msSky), 1e-4, "in-view (MS kernel) vs dome sky")
        // (The single-scatter march compiled into two different kernels differs by ~1e-3 at worst
        // near the horizon — exp of a large optical depth under fast-math — hence 2e-3 here.)
        XCTAssertLessThan(worstRel(try runInView(inViewMS, lutBound: false), ssSky), 2e-3, "LUT not bound: single scatter")
        XCTAssertLessThan(worstRel(try runInView(inViewSS, lutBound: false), ssSky), 2e-3, "the unchanged kernel: single scatter")
        // …which at −6° is well below the multiply-scattered sky (above the horizon).
        let up = dirs.indices.filter { dirs[$0].y > 0.02 }
        let msY = up.map { simd_dot(msSky[$0], Self.luma) }.reduce(0, +), ssY = up.map { simd_dot(ssSky[$0], Self.luma) }.reduce(0, +)
        XCTAssertGreaterThan(msY, ssY * 1.2, "multiple scattering at −6°: \(msY) vs single \(ssY)")
    }

    // MARK: - 6. Day look + GPU cost

    /// How far the day sky moves when a host opts in (not a pass/fail on taste — a fence on the
    /// measured change, so a later edit that moves it shows up here).
    @MainActor
    func testDaytimeChangeWhenOptingIn() throws {
        let h = try Harness()
        let dirs: [(String, SIMD3<Float>)] = [("zenith", View.zenith.dir), ("window", View.window.dir),
                                             ("anti10", View.anti10.dir), ("horizon90", SIMD3(0, 0, 1))]
        var report = "sun  view      off_Y       on_Y        stops   sat_off sat_on\n"
        for el: Float in [70, 50, 30, 15] {
            let off = try h.sky(Self.bareSky(el: el, ms: false), dirs.map { $0.1 })
            let on = try h.sky(Self.bareSky(el: el, ms: true), dirs.map { $0.1 })
            for (i, d) in dirs.enumerated() {
                let a = simd_dot(off[i], Self.luma), b = simd_dot(on[i], Self.luma)
                func sat(_ c: SIMD3<Float>) -> Float { let m = max(c.x, c.y, c.z); return (m - min(c.x, c.y, c.z)) / m }
                let s = Self.stops(b, a)
                report += String(format: "%3.0f  %@  %.4e  %.4e  %+.3f  %.3f   %.3f\n", el, d.0.padding(toLength: 9, withPad: " ", startingAt: 0),
                                 a, b, s, sat(off[i]), sat(on[i]))
                if d.0 == "zenith" || d.0 == "window" {
                    XCTAssertTrue((0.25...0.85).contains(s), "\(el)° \(d.0): \(s) stops (expected ~+0.5: MS + quadrature)")
                }
            }
        }
        print("── NishitaMS day sky, flag on vs off ──\n" + report)
    }

    /// The spectral bake (`atmosphereSpectral`): the 3-wavelength march's mauve civil twilight
    /// turns blue (ozone's Chappuis band is sampled where the sRGB red primary sees it), while the
    /// day sky moves only a little. Targets from the offline spectral reference
    /// (scratchpad sky/r3/spectral.py, single scatter, 380–720 nm, CIE 1931): sun −4.2°, the
    /// window direction — 3 wavelengths (1.00, 0.55, 0.93), spectral (0.47, 0.60, 1.00).
    @MainActor
    func testSpectralBakeTurnsTheTwilightWindowBlue() throws {
        let h = try Harness()
        let window = View.window.dir
        var report = "sun   rgb-3λ                         spectral\n"
        for el: Float in [-1.2, -4.2, -7.1] {
            var p = Self.bareSky(el: el, ms: true)
            let rgb = try h.sky(p, [window])[0]
            p.atmosphereSpectral = true
            let sp = try h.sky(p, [window])[0]
            report += String(format: "%5.1f (%.3e %.3e %.3e)  (%.3e %.3e %.3e)\n", el, rgb.x, rgb.y, rgb.z, sp.x, sp.y, sp.z)
            XCTAssertGreaterThan(sp.z, sp.x, "spectral at \(el)°: B > R — \(sp)")
            XCTAssertGreaterThanOrEqual(sp.y, 0.7 * sp.x, "spectral at \(el)°: G ≥ 0.7 R — \(sp)")
            XCTAssertLessThan(rgb.y / rgb.x, sp.y / sp.x, "the 3-wavelength march is the greener-starved one at \(el)°")
        }
        for el: Float in [50, 15] {
            var p = Self.bareSky(el: el, ms: true)
            let rgb = try h.sky(p, [View.zenith.dir, window])
            p.atmosphereSpectral = true
            let sp = try h.sky(p, [View.zenith.dir, window])
            for (a, b) in zip(rgb, sp) {
                XCTAssertLessThan(abs(Self.stops(simd_dot(b, Self.luma), simd_dot(a, Self.luma))), 0.15, "day \(el)°: luma within 0.15 stop")
                let na = a / a.z, nb = b / b.z
                XCTAssertLessThan(simd_length(na - nb), 0.12, "day \(el)°: hue within a few % — \(na) vs \(nb)")
            }
        }
        print("── Nishita spectral bake, window direction ──\n" + report)
    }

    @MainActor
    func testGPUCostOfTheLUTBuildAndTheMarch() throws {
        let engine = SimEngine.shared
        let cloud = VolumetricCloudRenderer(engine: engine, resolution: SIMD2(64, 32), iblResolution: SIMD2(32, 16))
        // The build alone, on a command buffer of its own (the renderer encodes it into the first
        // command buffer after the flag turns on). First run includes nothing but GPU time.
        var buildMs: [Double] = []
        for _ in 0..<3 {
            guard let cb = engine.commandQueue.makeCommandBuffer() else { throw XCTSkip("no cb") }
            XCTAssertTrue(cloud.encodeAtmosphereLUTBuild(into: cb, albedo: 0.1))
            cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness timing
            buildMs.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
        }
        // The march: the dome kernel at 2048 × 1024 with the cloud march minimised, flag off vs on.
        guard let psoOff = engine.pipeline("volSkyRender"), let psoOn = engine.pipeline("volSkyRenderMS") else {
            return XCTFail("volSkyRender / volSkyRenderMS missing")
        }
        let W = 2048, H = 1024
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: W, height: H, mipmapped: false)
        td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .private
        guard let tex = engine.device.makeTexture(descriptor: td),
              let ub = engine.device.makeBuffer(length: VolumetricCloudRenderer.skyUniformsBufferLength, options: .storageModeShared)
        else { throw XCTSkip("no resources") }
        // A built LUT in this buffer too.
        memcpy(ub.contents() + VolumetricCloudRenderer.atmosphereLUTOffset,
               cloud.skyUniformsBuffer.contents() + VolumetricCloudRenderer.atmosphereLUTOffset,
               VolumetricCloudRenderer.skyUniformsBufferLength - VolumetricCloudRenderer.atmosphereLUTOffset)
        func time(_ p: VolumetricCloudRenderer.Params) throws -> Double {
            var u = SkyUniforms(params: p, time: 0)
            memcpy(ub.contents(), &u, MemoryLayout<SkyUniforms>.stride)
            let ms = p.atmosphereMultipleScattering
            var t: [Double] = []
            for _ in 0..<5 {
                guard let cb = engine.commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no cb") }
                enc.setComputePipelineState(ms ? psoOn : psoOff)
                enc.setTexture(tex, index: 0); enc.setTexture(cloud.noiseTexture, index: 1)
                enc.setBuffer(ub, offset: 0, index: 0); enc.setBuffer(cloud.fallbackBurstLightBuffer, offset: 0, index: 1)
                if ms { enc.setBuffer(ub, offset: VolumetricCloudRenderer.atmosphereLUTOffset, index: 2) }
                enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness timing
                t.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
            }
            return t.sorted()[t.count / 2]
        }
        func cheapClouds(_ p: inout VolumetricCloudRenderer.Params) { p.coverage = 0; p.cloudSteps = 1; p.lightSteps = 1; p.cirrusCoverage = 0 }
        var dayOff = Self.bareSky(el: 40, ms: false); cheapClouds(&dayOff)
        var dayOn = dayOff; dayOn.atmosphereMultipleScattering = true
        var nightOff = dayOff
        nightOff.sunDir = -Self.toSun(-8); nightOff.nightSkyModel = .physical; nightOff.moonIntensity = 1
        nightOff.moonDir = simd_normalize(SIMD3<Float>(-0.9, 0.3, 0.1)); nightOff.nightRadiance = 2.4e-6
        var nightOn = nightOff; nightOn.atmosphereMultipleScattering = true
        let tDayOff = try time(dayOff), tDayOn = try time(dayOn), tNightOff = try time(nightOff), tNightOn = try time(nightOn)
        print("pipelines: volSkyRender maxThreads \(psoOff.maxTotalThreadsPerThreadgroup), volSkyRenderMS \(psoOn.maxTotalThreadsPerThreadgroup)")
        print(String(format: "── NishitaMS GPU cost ──\nLUT build (once): %@ ms\n2048×1024 atmosphere march, day (sun 40°): off %.2f ms, on %.2f ms (Δ %+.2f)\n"
                     + "twilight −8° with the moon: off %.2f ms, on %.2f ms (Δ %+.2f)",
                     buildMs.map { String(format: "%.2f", $0) }.joined(separator: " / "),
                     tDayOff, tDayOn, tDayOn - tDayOff, tNightOff, tNightOn, tNightOn - tNightOff))
        // Fences, not specs (measured on an M1 Max: build ≈2 ms; march ≈1.8× — other sessions'
        // GPU work on a shared machine inflates single samples, hence the headroom).
        XCTAssertLessThan(buildMs.min()!, 50, "the one-time build is a frame's worth of GPU, not more")
        XCTAssertLessThan(tDayOn, tDayOff * 3.0, "the multiple-scattering march stays within ~2× the old one")
    }
}
