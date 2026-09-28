import XCTest
import Metal
import simd
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers
@testable import VisualizerRendering

/// Aerosols for the Nishita sky — `VolumetricCloudRenderer.Params.atmosphereAerosol`
/// (`NishitaAerosol`: τ550, Ångström α, scale height, single-scattering albedo, Cornette–Shanks g),
/// pinned as numbers:
///
///   1. The parameterisation: `.builtin` IS the shader's constants (τ550 0.0277, α 0, H 1.2 km,
///      ω₀ 1/1.1, g 0.76), read back from the GPU; the presets sit in OPAC's ranges; Preetham's
///      turbidity maps through Ångström's β; the brute-force reference uses the same definition.
///   2. With the built-in aerosol every kernel is BIT-IDENTICAL to engine aded161: the dome / IBL
///      kernel, the transmittance + multiple-scattering LUT build, the cloud-light prepass and
///      Illuminatorama's in-view kernel, sun +50° … −6°, multiple scattering off and on — float32
///      targets, byte for byte (runtime and offline-compiled libraries).
///   3. No pop: the aerosol kernels at the built-in values draw the built-in sky.
///   4. The GPU sky against the brute-force reference (`NishitaReferenceIntegrator`, per-channel
///      aerosol, orders 1…6, 400 000 paths) at τ550 = built in, 0.2, 0.4 (α 1.3, H 2 km) and the
///      hazy-suburban preset (0.25, H 4 km), sun +79° … −6°.
///   5. Every consumer sees the same aerosol: the in-view pass, the cloud-light prepass (the sun
///      reaching the deck through the SAME air as `Params.sunTransmittance()`), the LUT rebuild.
///   6. Noon: the up-hemisphere E/π (sun disc out) that Daydream Home calibrates its sky against.
///   7. The golden band and the twilight arch — the sunward vertical at +6, +2, −3° — against the
///      reference, Koomen et al. (1952)'s measured clean (Sacramento Peak) and hazy (Maryland)
///      twilight skies, Preetham et al. (1999)'s turbidity model and Ugolnikov & Maslov (2016).
///   8. The comparison sheet (`VIZ_AEROSOL_SHEET=/path.png`; `VIZ_AEROSOL_SWEEP=/path.png` adds an
///      animated sunset sweep).
///   9. GPU cost: the aerosol LUT build and march against the built-in ones.
final class NishitaAerosolTests: XCTestCase {

    typealias MS = NishitaMultipleScatteringTests
    typealias NR = NishitaReference
    static let luma = SIMD3<Float>(0.2126, 0.7152, 0.0722)
    static func stops(_ a: Float, _ b: Float) -> Float { log2(a / b) }

    /// Directions: the viewer 1 m up at the pole, the sun at azimuth 0 (+X). `sunN` = N° up toward
    /// the sun; `window` = 30° up, 90° in azimuth; `anti0_5` = 0.5° up, away from the sun.
    enum View: String, CaseIterable {
        case zenith, sun5, sun15, window, anti0_5, sun1, sun3, sun8, sun12, sun30
        static func dir(el: Float, az: Float) -> SIMD3<Float> {
            SIMD3(cos(el * .pi / 180) * cos(az * .pi / 180), sin(el * .pi / 180), cos(el * .pi / 180) * sin(az * .pi / 180))
        }
        var dir: SIMD3<Float> {
            switch self {
            case .zenith: return SIMD3(0, 1, 0)
            case .window: return Self.dir(el: 30, az: 90)
            case .anti0_5: return Self.dir(el: 0.5, az: 180)
            case .sun1: return Self.dir(el: 1, az: 0)
            case .sun3: return Self.dir(el: 3, az: 0)
            case .sun5: return Self.dir(el: 5, az: 0)
            case .sun8: return Self.dir(el: 8, az: 0)
            case .sun12: return Self.dir(el: 12, az: 0)
            case .sun15: return Self.dir(el: 15, az: 0)
            case .sun30: return Self.dir(el: 30, az: 0)
            }
        }
    }

    /// Physical sky, no clouds / cirrus / celestials, unit irradiance, no grade — `aerosol` in it.
    static func sky(el: Float, ms: Bool, aerosol: NishitaAerosol) -> VolumetricCloudRenderer.Params {
        var p = MS.bareSky(el: el, ms: ms)
        p.atmosphereAerosol = aerosol
        return p
    }

    /// CIE 1931 xy of a linear-sRGB (D65) colour.
    static func xy(_ c: SIMD3<Float>) -> SIMD2<Float> {
        let X = 0.4124 * c.x + 0.3576 * c.y + 0.1805 * c.z
        let Y = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
        let Z = 0.0193 * c.x + 0.1192 * c.y + 0.9505 * c.z
        let s = max(X + Y + Z, 1e-30)
        return SIMD2(X / s, Y / s)
    }

    // MARK: - 1. The parameterisation

    func testParameterisationPresetsAndPacking() throws {
        let b = NishitaAerosol.builtin
        // The built-in aerosol in these terms, derived from the shader's constants.
        XCTAssertEqual(b.opticalDepth, 21e-6 * 1.1 * 1200, accuracy: 1e-7)
        XCTAssertEqual(b.opticalDepth, 0.0277, accuracy: 1e-4)
        XCTAssertEqual(b.angstromExponent, 0)
        XCTAssertEqual(b.scaleHeight, 1200)
        XCTAssertEqual(b.singleScatteringAlbedo, 1 / 1.1, accuracy: 1e-7)
        XCTAssertEqual(b.phaseG, 0.76)
        for c in 0..<3 {
            XCTAssertEqual(b.seaLevelScattering[c], 21e-6, accuracy: 21e-6 * 1e-6, "grey: every channel")
            XCTAssertEqual(b.seaLevelExtinction[c], 23.1e-6, accuracy: 23.1e-6 * 1e-6)
        }
        // Ångström: per-channel optical depth λ^−α at 680 / 550 / 440 nm.
        let h = NishitaAerosol.hazySuburban
        let t = h.channelOpticalDepth
        XCTAssertEqual(t.y, 0.25, accuracy: 1e-7)
        XCTAssertEqual(t.x / t.y, pow(680.0 / 550, -1.3), accuracy: 1e-6)
        XCTAssertEqual(t.z / t.y, pow(440.0 / 550, -1.3), accuracy: 1e-6)
        XCTAssertEqual(h.seaLevelScattering.y / h.seaLevelExtinction.y, 0.92, accuracy: 1e-6)
        // Preetham's turbidity → Ångström β → τ550 (App. A.1: β = 0.04608 T − 0.04586, α 1.3).
        for (T, tau) in [(2.0, 0.1007), (3.0, 0.2010), (3.5, 0.2511), (4.0, 0.3012)] as [(Float, Float)] {
            XCTAssertEqual(NishitaAerosol.preetham(turbidity: T).opticalDepth, tau, accuracy: 5e-4, "T = \(T)")
        }
        XCTAssertEqual(NishitaAerosol.preetham(turbidity: 1).opticalDepth, 0, accuracy: 1e-3)
        // Cornette–Shanks g from a measured asymmetry parameter (mean cosine) and back.
        XCTAssertEqual(NishitaAerosol.cornetteShanksG(asymmetryParameter: 0.8098), 0.76, accuracy: 1e-3)
        for a: Float in [0.6, 0.7, 0.772] {
            let g = NishitaAerosol.cornetteShanksG(asymmetryParameter: a)
            XCTAssertEqual(3 * g * (4 + g * g) / (5 * (2 + g * g)), a, accuracy: 1e-5)
        }
        // The presets are OPAC's types (Hess, Koepke & Schult 1998, Table 3, 80 % RH).
        XCTAssertEqual(NishitaAerosol.cleanContinental.opticalDepth, 0.064)
        XCTAssertEqual(NishitaAerosol.rural.opticalDepth, 0.151)
        XCTAssertEqual(NishitaAerosol.urban.opticalDepth, 0.643)
        XCTAssertEqual(NishitaAerosol.cleanMaritime.opticalDepth, 0.096)
        // Packing: the built-in aerosol packs zeros (the unchanged kernels never read them); any
        // other aerosol packs (βs, g) and (βe, H); the procedural gradient never gets one.
        var p = VolumetricCloudRenderer.Params()
        XCTAssertEqual(p.atmosphereAerosol, .builtin)
        XCTAssertFalse(p.physicalAerosol)
        var u = SkyUniforms(params: p, time: 0)
        XCTAssertEqual(u.aerosolA, .zero); XCTAssertEqual(u.aerosolB, .zero)
        p.atmosphereAerosol = .hazySuburban
        u = SkyUniforms(params: p, time: 0)
        XCTAssertEqual(u.aerosolA, SIMD4(0.25, 1.3, 0.92, NishitaAerosol.hazySuburban.phaseG), "τ550, α, ω₀, g")
        XCTAssertEqual(u.aerosolB, SIMD4(0, 0, 0, 4000), "H_M (the on-flag)")
        p.atmosphere = .proceduralGradient
        XCTAssertEqual(SkyUniforms(params: p, time: 0).aerosolB, .zero, "an aerosol is a property of the nishita sky")
        // A degenerate host value is sanitised, never NaN.
        var bad = NishitaAerosol.hazySuburban
        bad.opticalDepth = .nan; bad.scaleHeight = -5; bad.phaseG = 3
        let s = bad.sanitized
        XCTAssertTrue(s.opticalDepth.isFinite && s.scaleHeight >= 10 && s.phaseG <= 0.95)
        // Layout: the aerosol clusters sit right before the GPU-written tail; the host prefix ends there.
        XCTAssertEqual(MemoryLayout<SkyUniforms>.offset(of: \SkyUniforms.aerosolB)! + 16,
                       MemoryLayout<SkyUniforms>.offset(of: \SkyUniforms.cloudLitSun)!)
        XCTAssertEqual(SkyUniforms.hostPrefixLength, MemoryLayout<SkyUniforms>.stride - 3 * 16)
        XCTAssertEqual(MemoryLayout<SkyUniforms>.stride, 39 * 16, "the Metal struct: 39 float4s")
        XCTAssertEqual(VolumetricCloudRenderer.atmosphereLUTOffset, 768, "the LUT region did not move")
        // The brute-force reference uses the same definition of the aerosol.
        var atm = NR.Atmosphere()
        atm.setAerosol(tau550: Double(h.opticalDepth), alpha: Double(h.angstromExponent), scaleHeight: Double(h.scaleHeight),
                       omega0: Double(h.singleScatteringAlbedo), g: Double(h.phaseG))
        let c = atm.coeffs(atm.Rg)
        for i in 0..<3 {
            XCTAssertEqual(c.sM[i], h.seaLevelScattering[i], accuracy: h.seaLevelScattering[i] * 1e-6)
            XCTAssertEqual(c.t[i] - atm.betaR[i] - atm.betaO[i] * atm.ozone(0), h.seaLevelExtinction[i],
                           accuracy: h.seaLevelExtinction[i] * 1e-6)
        }
    }

    /// One source of truth: the Swift mirror (`NishitaAtmosphere`, which `.builtin` and the CPU sun
    /// transmittance are built from) against the constants the shader was compiled with.
    @MainActor
    func testShaderConstantsMatchTheSwiftMirror() throws {
        let engine = SimEngine.shared
        guard let pso = engine.pipeline("volSkyAtmosphereConstants"),
              let out = engine.device.makeBuffer(length: 5 * 16, options: .storageModeShared),
              let cb = engine.commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            return XCTFail("volSkyAtmosphereConstants missing")
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        let v = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: 5)
        typealias A = NishitaAtmosphere
        XCTAssertEqual(SIMD3(v[0].x, v[0].y, v[0].z), SIMD3<Float>(A.rayleighScattering), "kBetaR")
        XCTAssertEqual(v[0].w, Float(A.rayleighScaleHeight), "kRayleighH")
        XCTAssertEqual(v[1].x, A.builtinMieScattering, "kBetaM"); XCTAssertEqual(v[1].y, v[1].x); XCTAssertEqual(v[1].z, v[1].x)
        XCTAssertEqual(v[1].w, A.builtinMieScaleHeight, "kMieH")
        XCTAssertEqual(SIMD3(v[2].x, v[2].y, v[2].z), SIMD3<Float>(A.ozoneAbsorption), "kBetaO")
        XCTAssertEqual(v[2].w, A.builtinMieG, "kMieG")
        XCTAssertEqual(v[3].x, Float(A.earthRadius), "kEarthRadius")
        XCTAssertEqual(v[3].y - v[3].x, Float(A.singleScatterTop), "kAtmosRadius")
        XCTAssertEqual(v[3].z - v[3].x, Float(A.multipleScatteringTop), "kMSAtmosRadius")
        XCTAssertEqual(v[3].w, Float(A.ozoneCenter), "kOzoneCenter")
        XCTAssertEqual(v[4].x, Float(A.ozoneHalfWidth), "kOzoneWidth")
        XCTAssertEqual(v[4].y, A.builtinMieExtinctionRatio, "kMieExtRatio")

        // The shader's aerosol medium, built from the packed parameters, against the Swift side
        // (`NishitaAerosol.seaLevel*`, which the CPU sun transmittance and the reference use).
        let src = try MetalSourceLoader.source(contentsOf: MS.shadersDir.appendingPathComponent("VolumetricSky.metal")) + """
        kernel void aerosolMediumProbe(device float4* out [[buffer(0)]], constant float4* ab [[buffer(1)]],
                                       uint tid [[thread_position_in_grid]]) {
            if (tid != 0u) return;
            NishitaMieAerosol m(ab[0], ab[1]);
            out[0] = float4(m.betaS, m.g()); out[1] = float4(m.betaE, 1.0f / m.invH);
            out[2] = float4(m.density(1500.0f), 0.0f, 0.0f, 0.0f);
            out[3] = float4(kChannelWavelengths, 0.0f);
        }
        """
        let lib = try engine.device.makeLibrary(source: src, options: nil)
        guard let fn = lib.makeFunction(name: "aerosolMediumProbe") else { return XCTFail("probe") }
        let probe = try engine.device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        for a in [NishitaAerosol.hazySuburban, .urban, .cleanMaritime, .rural] {
            let packed = a.uniforms
            var ab = [packed.a, packed.b]
            guard let ob = engine.device.makeBuffer(length: 4 * 16, options: .storageModeShared),
                  let cb2 = engine.commandQueue.makeCommandBuffer(), let enc2 = cb2.makeComputeCommandEncoder() else { throw XCTSkip("no cb") }
            enc2.setComputePipelineState(probe)
            enc2.setBuffer(ob, offset: 0, index: 0)
            ab.withUnsafeMutableBytes { enc2.setBytes($0.baseAddress!, length: $0.count, index: 1) }
            enc2.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()  // gpu-ok: test harness
            let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: 4)
            for c in 0..<3 {
                XCTAssertEqual(Double(o[0][c]), a.seaLevelScattering[c], accuracy: a.seaLevelScattering[c] * 2e-6, "βs[\(c)]")
                XCTAssertEqual(Double(o[1][c]), a.seaLevelExtinction[c], accuracy: a.seaLevelExtinction[c] * 2e-6, "βe[\(c)]")
            }
            XCTAssertEqual(o[0].w, a.phaseG); XCTAssertEqual(o[1].w, a.scaleHeight, accuracy: a.scaleHeight * 1e-6)
            XCTAssertEqual(o[2].x, exp(-1500 / a.scaleHeight), accuracy: 1e-6)
            XCTAssertEqual(SIMD3(o[3].x, o[3].y, o[3].z), NishitaAerosol.channelWavelengths, "kChannelWavelengths")
        }
    }

    // MARK: - 2. Built-in aerosol: bit-identical to engine aded161

    /// The engine commit this work branches from (Daydream's measurements were made on it).
    static let baseline = "aded161286c66e58d077f0a4de59a517fd7ed390"

    static func gitShow(_ path: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        proc.arguments = ["-C", root.path, "show", "\(baseline):\(path)"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        try proc.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        guard proc.terminationStatus == 0, let s = String(data: data, encoding: .utf8), !s.isEmpty else {
            throw XCTSkip("engine commit \(baseline) not reachable from this checkout")
        }
        return s
    }

    /// aded161's VolumetricSky.metal + its header in a directory of their own.
    static func baselineSources() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("aded161-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let base = "VisualizerRendering/Sources/VisualizerRendering/Shaders/"
        for f in ["IlluminatoramaNightSky.h", "VolumetricSky.metal"] {
            try gitShow(base + f).write(to: dir.appendingPathComponent(f), atomically: true, encoding: .utf8)
        }
        return dir
    }

    /// The old struct is this one without the two aerosol clusters (right before the GPU tail).
    static func baselineBytes(_ u: SkyUniforms) -> [UInt8] {
        var u = u
        let all = withUnsafeBytes(of: &u) { Array($0) }
        let a = MemoryLayout<SkyUniforms>.offset(of: \SkyUniforms.aerosolA)!
        let tail = MemoryLayout<SkyUniforms>.offset(of: \SkyUniforms.cloudLitSun)!
        precondition(tail == a + 32)
        return Array(all[0..<a]) + Array(all[tail...])
    }

    /// `xcrun metal` + `metallib` — the offline compiler the app's default.metallib goes through.
    /// (Its intermediates go to a temporary directory — never beside the source, where SwiftPM's
    /// `.process("Shaders")` would pick them up.)
    static func offlineLibrary(_ device: MTLDevice, source: URL) throws -> MTLLibrary {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("offline-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        func run(_ args: [String]) throws {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
            p.arguments = args
            p.standardOutput = Pipe(); p.standardError = Pipe()
            try p.run(); p.waitUntilExit()
            guard p.terminationStatus == 0 else { throw XCTSkip("xcrun \(args.joined(separator: " ")) failed") }
        }
        let air = dir.appendingPathComponent("\(UUID().uuidString).air")
        let lib = dir.appendingPathComponent("\(UUID().uuidString).metallib")
        try run(["-sdk", "macosx", "metal", "-c", source.path, "-o", air.path])
        try run(["-sdk", "macosx", "metallib", air.path, "-o", lib.path])
        return try device.makeLibrary(URL: lib)
    }

    /// Every consumer of the sky, per configuration: the sky with clouds (atmosphere-lit deck,
    /// ground from the atmosphere), cirrus, the physical night with a moon, a grade.
    static func bitConfig(el: Float, ms: Bool) -> VolumetricCloudRenderer.Params {
        var p = VolumetricCloudRenderer.Params()
        p.sunDir = -MS.toSun(el)
        p.atmosphereMultipleScattering = ms
        p.cloudLightingFromAtmosphere = true; p.groundFromAtmosphere = true
        p.cloudBaseY = 1300; p.cloudTopY = 2600; p.horizontalScale = 0.0006; p.coverage = 0.55
        p.cirrusCoverage = 0.35; p.skySaturation = 1.12; p.skyBlueLift = 1.2
        if el < 0 {
            p.nightSkyModel = .physical; p.moonIntensity = 4.6; p.starBrightness = 1
            p.moonDir = simd_normalize(SIMD3<Float>(-0.9, sin(13 * Float.pi / 180), 0.3))
            p.nightRadiance = 2.4e-9; p.celestialsInDome = false; p.moonLightGain = 0.8
        }
        return p
    }

    struct GPUOut { var dome: [UInt32]; var lut: [UInt8]; var tail: [UInt8]; var inView: [UInt32] }

    /// Run every consumer of one library on `bytes` (its own SkyUniforms layout): the LUT build
    /// (multiple scattering), the cloud-light prepass, the dome kernel, the in-view kernel.
    @MainActor
    static func runConsumers(_ lib: MTLLibrary, bytes: [UInt8], ms: Bool, noise: MTLTexture,
                             depth: MTLTexture, W: Int, H: Int) throws -> GPUOut {
        let engine = SimEngine.shared
        let device = engine.device, queue = engine.commandQueue
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let f = lib.makeFunction(name: name) else { throw XCTSkip("\(name) missing") }
            return try device.makeComputePipelineState(function: f)  // gpu-ok: test harness
        }
        let lutOff = VolumetricCloudRenderer.atmosphereLUTOffset
        let len = VolumetricCloudRenderer.skyUniformsBufferLength
        guard let ub = device.makeBuffer(length: len, options: .storageModeShared),
              let lights = device.makeBuffer(length: 256, options: .storageModeShared) else { throw XCTSkip("no buffers") }
        memset(ub.contents(), 0, len)
        memset(lights.contents(), 0, 256)
        bytes.withUnsafeBytes { ub.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes.count) }
        var lut: [UInt8] = []
        if ms {
            guard let cb = queue.makeCommandBuffer() else { throw XCTSkip("no cb") }
            let L = VolumetricCloudRenderer.AtmosphereLUT.self
            var bp = SIMD4<Float>(0, 0.1, 0, 0)
            func stage(_ p: MTLComputePipelineState, _ body: (MTLComputeCommandEncoder) -> Void) {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(p)
                enc.setBuffer(ub, offset: lutOff, index: 0)
                body(enc)
                enc.endEncoding()
            }
            stage(try pso("volSkyTransmittanceLUT")) {
                $0.dispatchThreads(MTLSize(width: L.transW, height: L.transH, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
            }
            let order = try pso("volSkyMultiScatterOrder")
            for n in 2...L.lastOrder {
                bp.x = Float(n)
                stage(order) {
                    $0.setBytes(&bp, length: 16, index: 1)
                    $0.dispatchThreadgroups(MTLSize(width: L.msW, height: L.msH, depth: 1), threadsPerThreadgroup: MTLSize(width: L.dirThreads, height: 1, depth: 1))
                }
            }
            stage(try pso("volSkyMultiScatterFinalize")) {
                $0.dispatchThreads(MTLSize(width: L.msW, height: L.msH, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
            }
            stage(try pso("volSkyMultiScatterReady")) {
                $0.setBytes(&bp, length: 16, index: 1)
                $0.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }
            cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
            if let e = cb.error { throw e }
            lut = Array(UnsafeRawBufferPointer(start: ub.contents() + lutOff, count: len - lutOff))
        }
        // The cloud-light prepass writes the GPU tail; the dome and in-view kernels then read it.
        guard let cb = queue.makeCommandBuffer() else { throw XCTSkip("no cb") }
        do {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(try pso(ms ? "volSkyCloudLightMS" : "volSkyCloudLight"))
            enc.setBuffer(ub, offset: 0, index: 0)
            if ms { enc.setBuffer(ub, offset: lutOff, index: 1) }
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
            enc.endEncoding()
        }
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: W, height: H, mipmapped: false)
        td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .shared
        guard let dome = device.makeTexture(descriptor: td), let view = device.makeTexture(descriptor: td) else { throw XCTSkip("no textures") }
        do {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(try pso(ms ? "volSkyRenderMS" : "volSkyRender"))
            enc.setTexture(dome, index: 0); enc.setTexture(noise, index: 1)
            enc.setBuffer(ub, offset: 0, index: 0); enc.setBuffer(lights, offset: 0, index: 1)
            if ms { enc.setBuffer(ub, offset: lutOff, index: 2) }
            enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
        }
        do {
            let proj = simd_float4x4(columns: (SIMD4(1, 0, 0, 0), SIMD4(0, 2, 0, 0), SIMD4(0, 0, 0, 1), SIMD4(0, 0, 0.1, 0)))
            let v = simd_float4x4(columns: (SIMD4(-1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, -1, 0), SIMD4(0, 0, 0, 1)))
            var cv = MS.InViewUniforms(invVP: (proj * v).inverse, cam: SIMD4(0, 1.7, 0, 0), extra: SIMD4(3.5, -1, -1, -1),
                                       night: SIMD4(3.5, ms ? 1 : 0, 0, 0))
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(try pso(ms ? "illumi_cloud_inview_ms" : "illumi_cloud_inview"))
            enc.setTexture(view, index: 0); enc.setTexture(noise, index: 1); enc.setTexture(depth, index: 2)
            enc.setBuffer(ub, offset: 0, index: 0)
            enc.setBytes(&cv, length: MemoryLayout<MS.InViewUniforms>.stride, index: 1)
            enc.setBuffer(lights, offset: 0, index: 2)
            if ms { enc.setBuffer(ub, offset: lutOff, index: 3) }
            enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
        }
        cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
        if let e = cb.error { throw e }
        func px(_ t: MTLTexture) -> [UInt32] {
            var a = [UInt32](repeating: 0, count: W * H * 4)
            a.withUnsafeMutableBytes { t.getBytes($0.baseAddress!, bytesPerRow: W * 16, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0) }
            return a
        }
        let tailOff = bytes.count - 48
        let tail = Array(UnsafeRawBufferPointer(start: ub.contents() + tailOff, count: 48))
        return GPUOut(dome: px(dome), lut: lut, tail: tail, inView: px(view))
    }

    @MainActor
    func testBuiltinAerosolIsBitIdenticalToAded161() throws {
        let engine = SimEngine.shared
        let device = engine.device
        let dir = try Self.baselineSources()
        defer { try? FileManager.default.removeItem(at: dir) }
        let current = MS.shadersDir.appendingPathComponent("VolumetricSky.metal")
        var libraries: [(String, MTLLibrary, MTLLibrary)] = [
            ("runtime", try MetalSourceLoader.makeLibrary(device: device, contentsOf: dir.appendingPathComponent("VolumetricSky.metal")),
             try MetalSourceLoader.makeLibrary(device: device, contentsOf: current)),
        ]
        if let old = try? Self.offlineLibrary(device, source: dir.appendingPathComponent("VolumetricSky.metal")),
           let new = try? Self.offlineLibrary(device, source: current) {
            libraries.append(("offline", old, new))
        } else {
            print("(offline metal compiler unavailable — runtime libraries only)")
        }
        let noiseOwner = VolumetricCloudRenderer(engine: engine, resolution: SIMD2(8, 4), iblResolution: SIMD2(8, 4))
        let W = 160, H = 80
        let depth = try MS.clearedDepth(device, engine.commandQueue, W, H)
        var compared = 0, lutBytes = 0
        var report = ""
        for (libName, oldLib, newLib) in libraries {
            for el: Float in [50, 6, 2, -3, -6] {
                for ms in [false, true] {
                    let p = Self.bitConfig(el: el, ms: ms)
                    XCTAssertEqual(p.atmosphereAerosol, .builtin)
                    let u = SkyUniforms(params: p, time: 12.5)
                    var nu = u
                    let newBytes = withUnsafeBytes(of: &nu) { Array($0) }
                    let a = try Self.runConsumers(oldLib, bytes: Self.baselineBytes(u), ms: ms, noise: noiseOwner.noiseTexture, depth: depth, W: W, H: H)
                    let b = try Self.runConsumers(newLib, bytes: newBytes, ms: ms, noise: noiseOwner.noiseTexture, depth: depth, W: W, H: H)
                    let name = String(format: "%@ sun %+.0f° MS %@", libName, el, ms ? "on " : "off")
                    let dd = zip(a.dome, b.dome).filter { $0 != $1 }.count
                    let vd = zip(a.inView, b.inView).filter { $0 != $1 }.count
                    XCTAssertEqual(dd, 0, "dome/IBL kernel, \(name): \(dd) of \(a.dome.count) floats differ")
                    XCTAssertEqual(vd, 0, "in-view kernel, \(name): \(vd) of \(a.inView.count) floats differ")
                    XCTAssertEqual(a.tail, b.tail, "cloud-light prepass, \(name): the GPU-written tail differs")
                    XCTAssertEqual(a.lut.count, b.lut.count)
                    let ld = zip(a.lut, b.lut).filter { $0 != $1 }.count
                    XCTAssertEqual(ld, 0, "atmosphere LUTs, \(name): \(ld) of \(a.lut.count) bytes differ")
                    // The dome actually rendered (and, with MS, the LUT was built — header ready).
                    XCTAssertGreaterThan(stride(from: 0, to: b.dome.count, by: 4).filter { b.dome[$0] != 0 }.count, b.dome.count / 16)
                    if ms {
                        let ready = b.lut.withUnsafeBytes { $0.load(as: Float.self) }
                        XCTAssertEqual(ready, 1, "\(name): LUT built")
                    }
                    compared += a.dome.count + a.inView.count + a.tail.count / 4
                    lutBytes += a.lut.count
                    report += "\(name): dome \(dd), in-view \(vd), prepass tail \(a.tail == b.tail ? 0 : 1), LUT \(ld) differing\n"
                }
            }
        }
        print("── Aerosol: built-in sky vs engine aded161 ──\n" + report
              + "\(compared) float32 outputs + \(lutBytes) LUT bytes compared bit-for-bit against \(Self.baseline.prefix(7))")
    }

    // MARK: - 3. No pop at the built-in values

    /// Any change to the aerosol switches to the `…Aerosol` kernels. Single scatter: the same march
    /// on runtime numbers — at the built-in values the sky is the built-in sky. Multiple scattering:
    /// the aerosol path adds its refinements (order-2 phase renormalisation, the Mie last bounce
    /// and higher-order re-scatter through the field's own Mie shape, a finer altitude axis and
    /// transmittance march), so at the
    /// built-in values it is the built-in sky with those corrections — measured here against the
    /// brute-force reference: the step off `.builtin` moves the horizon TOWARD the reference.
    @MainActor
    func testAerosolKernelsAtTheBuiltinValuesDrawTheBuiltinSky() throws {
        let h = try MS.Harness()
        var near = NishitaAerosol.builtin
        near.opticalDepth = near.opticalDepth.nextUp
        XCTAssertNotEqual(near, .builtin, "a 1-ulp change already selects the aerosol kernels")
        var dirs: [SIMD3<Float>] = []
        for az in stride(from: Float(0), to: 360, by: 30) {
            for el: Float in [0.5, 2, 5, 10, 20, 35, 55, 75] { dirs.append(View.dir(el: el, az: az)) }
        }
        dirs.append(SIMD3(0, 1, 0))
        var report = "sun   MS   worst |stops|  mean |stops|   (sky above the horizon, 97 directions)\n"
        for ms in [false, true] {
            var worstAll: Float = 0
            for el: Float in [50, 20, 6, 2, 0, -3, -6] {
                let a = try h.sky(Self.sky(el: el, ms: ms, aerosol: .builtin), dirs)
                let b = try h.sky(Self.sky(el: el, ms: ms, aerosol: near), dirs)
                var worst: Float = 0, sum: Float = 0, n: Float = 0
                for (x, y) in zip(a, b) where simd_dot(x, Self.luma) > 0 {
                    let d = abs(Self.stops(simd_dot(y, Self.luma), simd_dot(x, Self.luma)))
                    worst = max(worst, d); sum += d; n += 1
                }
                worstAll = max(worstAll, worst)
                report += String(format: "%+3.0f  %@   %.4f         %.4f\n", el, ms ? "on " : "off", worst, sum / max(n, 1))
            }
            // MS: the step off `.builtin` is the built-in kernels' own error being corrected (VZ-0159:
            // they sit 0.41 stop bright on the anti-solar horizon at −6°, 0.33 dark on the sunward
            // horizon at −3° — the rows below pin that the step goes TOWARD the reference).
            XCTAssertLessThan(worstAll, ms ? 0.5 : 1e-3, "MS \(ms): the aerosol kernels at the built-in values")
        }
        // Against the reference, the built-in rows: the aerosol path at the built-in values is at
        // least as close as the built-in kernels (the switch is a correction, not a drift).
        var errBuiltin: Float = 0, errAerosol: Float = 0
        report += "reference rows (built-in aerosol), stops:  built-in kernels | aerosol kernels at the built-in values\n"
        for el in Set(Self.reference.filter { $0.cfg == .builtin }.map { $0.el }).sorted(by: >) {
            let rows = Self.reference.filter { $0.cfg == .builtin && $0.el == el }
            let a = try h.sky(Self.sky(el: el, ms: true, aerosol: .builtin), rows.map { $0.view.dir })
            let b = try h.sky(Self.sky(el: el, ms: true, aerosol: near), rows.map { $0.view.dir })
            for (i, r) in rows.enumerated() {
                let yr = simd_dot(r.rgb, Self.luma)
                let ea = Self.stops(simd_dot(a[i], Self.luma), yr), eb = Self.stops(simd_dot(b[i], Self.luma), yr)
                errBuiltin += abs(ea); errAerosol += abs(eb)
                report += String(format: "  %+3.0f %-8@ %+.3f | %+.3f\n", el, r.view.rawValue, ea, eb)
            }
        }
        report += String(format: "Σ|stops|: built-in kernels %.2f, aerosol kernels %.2f\n", errBuiltin, errAerosol)
        XCTAssertLessThanOrEqual(errAerosol, errBuiltin * 0.5, "the aerosol path at the built-in values is closer to the reference")
        print("── Aerosol kernels at the built-in values vs the built-in kernels ──\n" + report)
    }

    // MARK: - 4. Against the brute-force reference

    enum RefCfg: String { case builtin, tau02, tau04, hazy
        var aerosol: NishitaAerosol {
            switch self {
            case .builtin: return .builtin
            case .tau02: var a = NishitaAerosol.hazySuburban; a.opticalDepth = 0.2; a.scaleHeight = 2000; return a
            case .tau04: var a = NishitaAerosol.hazySuburban; a.opticalDepth = 0.4; a.scaleHeight = 2000; return a
            case .hazy: return .hazySuburban
            }
        }
    }
    struct R {
        let cfg: RefCfg; let el: Float; let view: View; let rgb: SIMD3<Float>; let ms: SIMD3<Float>; let relSE: Float
        init(_ cfg: RefCfg, _ el: Float, _ view: View, _ r: Float, _ g: Float, _ b: Float,
             _ mr: Float, _ mg: Float, _ mb: Float, _ se: Float) {
            self.cfg = cfg; self.el = el; self.view = view; rgb = SIMD3(r, g, b); ms = SIMD3(mr, mg, mb); relSE = se
        }
    }

    /// `NishitaReferenceIntegrator` (double precision, orders 1…6 by backward path tracing, the true
    /// Rayleigh / Cornette–Shanks phases at every vertex, the single scatter integrated
    /// deterministically, planet shadow at every point, Lambertian ground 0.1, 100 km top),
    /// 400 000 paths per configuration, release build. Per unit solar irradiance: total (rgb),
    /// its orders ≥ 2 (rgb), and the Monte Carlo standard error relative to the total.
    /// tau02 / tau04 = `hazySuburban`'s particles (α 1.3, ω₀ 0.92, mean cosine 0.70) at τ550 0.2 /
    /// 0.4 packed into a 2 km layer (the hardest case for the LUT: a thick haze near the ground);
    /// hazy = `hazySuburban` itself (τ550 0.25, H 4 km).
    static let reference: [R] = [
        R(.builtin, 79, .zenith, 4.18179e-02, 4.65878e-02, 6.16491e-02, 1.31474e-03, 3.13404e-03, 9.89568e-03, 0.0001),
        R(.builtin, 79, .sun5, 3.35932e-02, 5.55146e-02, 8.43613e-02, 1.21532e-02, 2.23925e-02, 4.27203e-02, 0.0007),
        R(.builtin, 79, .sun15, 1.72998e-02, 3.31712e-02, 6.63596e-02, 4.96461e-03, 1.12625e-02, 2.95176e-02, 0.0006),
        R(.builtin, 79, .window, 9.86800e-03, 1.98912e-02, 4.45386e-02, 2.42205e-03, 6.08861e-03, 1.84700e-02, 0.0005),
        R(.builtin, 79, .anti0_5, 4.28828e-02, 5.69637e-02, 7.37497e-02, 2.22592e-02, 2.99973e-02, 4.26761e-02, 0.0010),
        R(.builtin, 20, .zenith, 3.75557e-03, 7.44658e-03, 1.70960e-02, 6.86532e-04, 1.89036e-03, 6.59926e-03, 0.0005),
        R(.builtin, 20, .sun5, 2.48179e-01, 2.18812e-01, 1.74545e-01, 2.10117e-02, 2.59463e-02, 3.69626e-02, 0.0004),
        R(.builtin, 20, .sun15, 2.06039e-01, 1.83954e-01, 1.65917e-01, 7.85285e-03, 1.18963e-02, 2.47345e-02, 0.0002),
        R(.builtin, 20, .window, 6.79058e-03, 1.34307e-02, 2.93040e-02, 1.60720e-03, 4.16316e-03, 1.30379e-02, 0.0007),
        R(.builtin, 20, .anti0_5, 4.59278e-02, 5.76773e-02, 6.24586e-02, 1.66492e-02, 2.30529e-02, 3.04529e-02, 0.0011),
        R(.builtin, 6, .zenith, 2.62986e-03, 4.24500e-03, 8.29652e-03, 4.82000e-04, 1.07969e-03, 3.29126e-03, 0.0007),
        R(.builtin, 6, .sun5, 4.16444e-01, 2.31982e-01, 9.24041e-02, 2.46297e-02, 1.88305e-02, 1.75374e-02, 0.0004),
        R(.builtin, 6, .sun15, 1.15195e-01, 7.62036e-02, 5.14574e-02, 7.33643e-03, 7.57149e-03, 1.20999e-02, 0.0005),
        R(.builtin, 6, .window, 5.18335e-03, 8.13712e-03, 1.45962e-02, 1.16324e-03, 2.42586e-03, 6.50246e-03, 0.0008),
        R(.builtin, 6, .anti0_5, 3.19159e-02, 2.73813e-02, 1.85182e-02, 1.14994e-02, 1.22212e-02, 1.23637e-02, 0.0014),
        R(.builtin, 2, .zenith, 1.88533e-03, 2.31422e-03, 4.00237e-03, 3.30157e-04, 5.63564e-04, 1.55260e-03, 0.0006),
        R(.builtin, 2, .sun5, 2.21072e-01, 7.72662e-02, 1.89803e-02, 1.56731e-02, 8.61275e-03, 7.21946e-03, 0.0004),
        R(.builtin, 2, .sun15, 4.96227e-02, 2.44792e-02, 1.66359e-02, 4.65829e-03, 3.62854e-03, 5.42267e-03, 0.0005),
        R(.builtin, 2, .window, 3.73386e-03, 4.41098e-03, 6.86370e-03, 8.01371e-04, 1.26653e-03, 3.03060e-03, 0.0007),
        R(.builtin, 2, .anti0_5, 1.58101e-02, 8.19459e-03, 4.92081e-03, 6.68840e-03, 5.04432e-03, 4.65318e-03, 0.0019),
        R(.builtin, 0, .zenith, 1.20698e-03, 1.17546e-03, 2.03585e-03, 1.94521e-04, 2.71908e-04, 7.70107e-04, 0.0005),
        R(.builtin, 0, .sun5, 7.01605e-02, 1.86375e-02, 6.69264e-03, 7.03429e-03, 3.46932e-03, 3.50603e-03, 0.0005),
        R(.builtin, 0, .sun15, 1.72428e-02, 8.26513e-03, 7.90487e-03, 2.20808e-03, 1.58818e-03, 2.70823e-03, 0.0006),
        R(.builtin, 0, .window, 2.37649e-03, 2.20715e-03, 3.41238e-03, 4.77239e-04, 6.09045e-04, 1.48440e-03, 0.0006),
        R(.builtin, 0, .anti0_5, 3.54242e-03, 1.87090e-03, 2.04374e-03, 2.87867e-03, 1.81106e-03, 2.04339e-03, 0.0030),
        R(.builtin, -3, .zenith, 2.29798e-04, 1.63695e-04, 3.51640e-04, 3.23881e-05, 3.66634e-05, 1.31333e-04, 0.0005),
        R(.builtin, -3, .sun5, 4.55711e-03, 2.26743e-03, 1.50484e-03, 8.04697e-04, 5.32255e-04, 7.12989e-04, 0.0006),
        R(.builtin, -3, .sun15, 1.90797e-03, 1.24687e-03, 1.73832e-03, 2.89995e-04, 2.38232e-04, 5.25410e-04, 0.0005),
        R(.builtin, -3, .window, 4.41326e-04, 2.96115e-04, 5.74132e-04, 7.92855e-05, 8.01955e-05, 2.49344e-04, 0.0006),
        R(.builtin, -3, .anti0_5, 2.31756e-04, 1.55356e-04, 2.94992e-04, 2.31756e-04, 1.55356e-04, 2.94992e-04, 0.0026),
        R(.builtin, -6, .zenith, 8.74732e-06, 6.11788e-06, 1.69796e-05, 2.36592e-06, 2.35985e-06, 9.45275e-06, 0.0018),
        R(.builtin, -6, .sun5, 6.58182e-04, 2.88239e-04, 2.13117e-04, 9.77588e-05, 5.08362e-05, 6.71068e-05, 0.0007),
        R(.builtin, -6, .sun15, 1.68363e-04, 1.03618e-04, 1.71867e-04, 3.10362e-05, 2.04655e-05, 4.68029e-05, 0.0009),
        R(.builtin, -6, .window, 1.69282e-05, 1.12298e-05, 2.86483e-05, 5.53759e-06, 5.05296e-06, 1.77927e-05, 0.0019),
        R(.builtin, -6, .anti0_5, 1.13739e-05, 8.24248e-06, 1.77974e-05, 1.13739e-05, 8.24248e-06, 1.77974e-05, 0.0051),
        R(.tau02, 79, .zenith, 1.17135e-01, 1.43499e-01, 1.76659e-01, 5.57148e-03, 1.04923e-02, 2.27724e-02, 0.0002),
        R(.tau02, 79, .sun5, 5.18768e-02, 6.28936e-02, 7.57325e-02, 2.72526e-02, 3.75556e-02, 5.25154e-02, 0.0016),
        R(.tau02, 79, .sun15, 3.72434e-02, 5.35033e-02, 7.89296e-02, 1.48778e-02, 2.54571e-02, 4.63614e-02, 0.0012),
        R(.tau02, 79, .window, 2.31964e-02, 3.58380e-02, 6.05307e-02, 7.36010e-03, 1.43266e-02, 3.15009e-02, 0.0009),
        R(.tau02, 79, .anti0_5, 4.20295e-02, 4.75277e-02, 5.69711e-02, 2.81291e-02, 3.36739e-02, 4.33889e-02, 0.0018),
        R(.tau02, 20, .zenith, 7.81511e-03, 1.20881e-02, 2.17501e-02, 2.46837e-03, 4.90376e-03, 1.16047e-02, 0.0010),
        R(.tau02, 20, .sun5, 4.51225e-01, 3.60966e-01, 2.31844e-01, 8.49563e-02, 8.67714e-02, 7.63609e-02, 0.0012),
        R(.tau02, 20, .sun15, 3.70684e-01, 3.48782e-01, 2.81772e-01, 4.49361e-02, 5.58578e-02, 6.54074e-02, 0.0008),
        R(.tau02, 20, .window, 1.32235e-02, 2.03748e-02, 3.41640e-02, 5.81688e-03, 1.06675e-02, 2.17463e-02, 0.0016),
        R(.tau02, 20, .anti0_5, 2.88408e-02, 3.04475e-02, 3.23972e-02, 1.77793e-02, 2.09835e-02, 2.53438e-02, 0.0026),
        R(.tau02, 6, .zenith, 3.92155e-03, 5.10430e-03, 8.60051e-03, 1.45835e-03, 2.29776e-03, 4.85143e-03, 0.0013),
        R(.tau02, 6, .sun5, 3.10142e-01, 1.40395e-01, 4.09602e-02, 5.51032e-02, 3.51959e-02, 1.91866e-02, 0.0013),
        R(.tau02, 6, .sun15, 1.73921e-01, 1.07075e-01, 5.39603e-02, 2.88903e-02, 2.39735e-02, 1.97054e-02, 0.0012),
        R(.tau02, 6, .window, 7.67227e-03, 9.39127e-03, 1.36746e-02, 3.56082e-03, 5.07651e-03, 8.88943e-03, 0.0019),
        R(.tau02, 6, .anti0_5, 1.19583e-02, 9.24682e-03, 8.42646e-03, 8.63490e-03, 7.91261e-03, 8.13433e-03, 0.0042),
        R(.tau02, 2, .zenith, 2.07122e-03, 2.29335e-03, 3.98558e-03, 7.48394e-04, 9.82972e-04, 2.16703e-03, 0.0010),
        R(.tau02, 2, .sun5, 9.32321e-02, 2.53803e-02, 7.05393e-03, 2.13914e-02, 9.65566e-03, 5.67489e-03, 0.0013),
        R(.tau02, 2, .sun15, 5.64482e-02, 2.48604e-02, 1.27989e-02, 1.22531e-02, 7.55963e-03, 6.59310e-03, 0.0010),
        R(.tau02, 2, .window, 4.01962e-03, 4.09996e-03, 6.08219e-03, 1.80399e-03, 2.12264e-03, 3.85410e-03, 0.0013),
        R(.tau02, 2, .anti0_5, 3.90257e-03, 2.86126e-03, 3.19958e-03, 3.67661e-03, 2.83941e-03, 3.19918e-03, 0.0056),
        R(.tau02, 0, .zenith, 1.08519e-03, 1.09544e-03, 2.04591e-03, 3.56768e-04, 4.37959e-04, 1.07863e-03, 0.0009),
        R(.tau02, 0, .sun5, 2.44712e-02, 6.26407e-03, 2.87885e-03, 7.16092e-03, 3.15506e-03, 2.57904e-03, 0.0017),
        R(.tau02, 0, .sun15, 1.66537e-02, 7.17617e-03, 5.32068e-03, 4.35019e-03, 2.61553e-03, 3.03542e-03, 0.0012),
        R(.tau02, 0, .window, 2.05787e-03, 1.90627e-03, 3.04952e-03, 8.53578e-04, 9.33207e-04, 1.89205e-03, 0.0012),
        R(.tau02, 0, .anti0_5, 1.60651e-03, 1.16201e-03, 1.50607e-03, 1.60643e-03, 1.16201e-03, 1.50607e-03, 0.0055),
        R(.tau02, -3, .zenith, 1.88038e-04, 1.53933e-04, 3.56799e-04, 5.23563e-05, 5.73966e-05, 1.84817e-04, 0.0008),
        R(.tau02, -3, .sun5, 1.80079e-03, 6.64636e-04, 5.55116e-04, 7.19598e-04, 3.99114e-04, 4.98369e-04, 0.0022),
        R(.tau02, -3, .sun15, 1.33334e-03, 8.20449e-04, 1.06831e-03, 4.46721e-04, 3.32376e-04, 5.87970e-04, 0.0013),
        R(.tau02, -3, .window, 3.43256e-04, 2.57182e-04, 5.19044e-04, 1.24096e-04, 1.19467e-04, 3.19433e-04, 0.0011),
        R(.tau02, -3, .anti0_5, 1.70509e-04, 1.14066e-04, 2.18701e-04, 1.70509e-04, 1.14066e-04, 2.18701e-04, 0.0047),
        R(.tau02, -6, .zenith, 7.42643e-06, 5.97371e-06, 1.73207e-05, 2.92927e-06, 3.02505e-06, 1.14341e-05, 0.0024),
        R(.tau02, -6, .sun5, 1.91203e-04, 6.70917e-05, 5.13542e-05, 6.71349e-05, 3.30931e-05, 4.09907e-05, 0.0031),
        R(.tau02, -6, .sun15, 1.12896e-04, 6.81309e-05, 9.97974e-05, 4.03903e-05, 2.71011e-05, 5.02045e-05, 0.0018),
        R(.tau02, -6, .window, 1.39984e-05, 1.03848e-05, 2.64767e-05, 6.91317e-06, 6.30558e-06, 1.97915e-05, 0.0026),
        R(.tau02, -6, .anti0_5, 5.82126e-06, 4.38254e-06, 1.06879e-05, 5.82126e-06, 4.38254e-06, 1.06879e-05, 0.0086),
        R(.tau04, 79, .zenith, 2.03230e-01, 2.35605e-01, 2.65716e-01, 1.55561e-02, 2.55472e-02, 4.45015e-02, 0.0004),
        R(.tau04, 79, .sun5, 6.11678e-02, 6.72977e-02, 7.41069e-02, 3.96914e-02, 4.83005e-02, 5.86941e-02, 0.0025),
        R(.tau04, 79, .sun15, 5.45447e-02, 6.86869e-02, 8.59238e-02, 2.82422e-02, 4.11739e-02, 6.02328e-02, 0.0019),
        R(.tau04, 79, .window, 3.67101e-02, 5.04780e-02, 7.27162e-02, 1.54438e-02, 2.56975e-02, 4.53279e-02, 0.0013),
        R(.tau04, 79, .anti0_5, 4.33260e-02, 4.69487e-02, 5.25446e-02, 3.26870e-02, 3.73073e-02, 4.41574e-02, 0.0024),
        R(.tau04, 20, .zenith, 1.18221e-02, 1.61239e-02, 2.49126e-02, 5.31694e-03, 8.71808e-03, 1.62603e-02, 0.0017),
        R(.tau04, 20, .sun5, 4.30898e-01, 2.94288e-01, 1.61526e-01, 1.28699e-01, 1.09528e-01, 7.80043e-02, 0.0022),
        R(.tau04, 20, .sun15, 4.79167e-01, 3.89300e-01, 2.57060e-01, 9.43303e-02, 9.79201e-02, 8.80804e-02, 0.0014),
        R(.tau04, 20, .window, 1.93698e-02, 2.55389e-02, 3.52724e-02, 1.14831e-02, 1.71258e-02, 2.68636e-02, 0.0028),
        R(.tau04, 20, .anti0_5, 2.32605e-02, 2.25578e-02, 2.21596e-02, 1.75201e-02, 1.85446e-02, 1.98285e-02, 0.0040),
        R(.tau04, 6, .zenith, 4.68972e-03, 5.50184e-03, 8.59584e-03, 2.46179e-03, 3.26186e-03, 5.84912e-03, 0.0021),
        R(.tau04, 6, .sun5, 1.61328e-01, 5.39276e-02, 1.56960e-02, 5.31356e-02, 2.66593e-02, 1.28480e-02, 0.0037),
        R(.tau04, 6, .sun15, 1.58504e-01, 7.99837e-02, 3.37108e-02, 4.31661e-02, 2.88174e-02, 1.84109e-02, 0.0024),
        R(.tau04, 6, .window, 8.77262e-03, 9.29056e-03, 1.19209e-02, 5.52590e-03, 6.44850e-03, 9.24144e-03, 0.0033),
        R(.tau04, 6, .anti0_5, 7.00329e-03, 5.57954e-03, 5.58492e-03, 6.35997e-03, 5.42334e-03, 5.56748e-03, 0.0065),
        R(.tau04, 2, .zenith, 2.14962e-03, 2.29833e-03, 3.92650e-03, 1.09147e-03, 1.29509e-03, 2.57781e-03, 0.0015),
        R(.tau04, 2, .sun5, 3.14008e-02, 8.23387e-03, 3.96871e-03, 1.53989e-02, 6.22296e-03, 3.88298e-03, 0.0035),
        R(.tau04, 2, .sun15, 4.14231e-02, 1.65595e-02, 8.17497e-03, 1.42835e-02, 7.61475e-03, 5.65889e-03, 0.0018),
        R(.tau04, 2, .window, 3.91378e-03, 3.70940e-03, 5.20186e-03, 2.39138e-03, 2.47304e-03, 3.93957e-03, 0.0021),
        R(.tau04, 2, .anti0_5, 2.50542e-03, 1.98762e-03, 2.35130e-03, 2.50090e-03, 1.98749e-03, 2.35130e-03, 0.0079),
        R(.tau04, 0, .zenith, 1.06668e-03, 1.07539e-03, 2.01843e-03, 4.95607e-04, 5.70019e-04, 1.29123e-03, 0.0014),
        R(.tau04, 0, .sun5, 8.61315e-03, 2.46508e-03, 1.80694e-03, 4.95265e-03, 2.06881e-03, 1.78901e-03, 0.0041),
        R(.tau04, 0, .sun15, 1.22382e-02, 4.87459e-03, 3.43593e-03, 4.80256e-03, 2.56101e-03, 2.57160e-03, 0.0021),
        R(.tau04, 0, .window, 1.88571e-03, 1.68432e-03, 2.61374e-03, 1.07547e-03, 1.07316e-03, 1.94798e-03, 0.0020),
        R(.tau04, 0, .anti0_5, 1.13509e-03, 8.52276e-04, 1.14256e-03, 1.13509e-03, 8.52276e-04, 1.14256e-03, 0.0080),
        R(.tau04, -3, .zenith, 1.76614e-04, 1.50028e-04, 3.52915e-04, 7.09291e-05, 7.47098e-05, 2.22099e-04, 0.0013),
        R(.tau04, -3, .sun5, 6.76983e-04, 2.80341e-04, 3.39554e-04, 4.65176e-04, 2.50721e-04, 3.36574e-04, 0.0044),
        R(.tau04, -3, .sun15, 9.22117e-04, 5.27184e-04, 6.60614e-04, 4.51933e-04, 3.07660e-04, 4.89426e-04, 0.0023),
        R(.tau04, -3, .window, 2.98597e-04, 2.25441e-04, 4.47238e-04, 1.52044e-04, 1.37553e-04, 3.30958e-04, 0.0019),
        R(.tau04, -3, .anti0_5, 1.35210e-04, 9.12219e-05, 1.72089e-04, 1.35210e-04, 9.12219e-05, 1.72089e-04, 0.0069),
        R(.tau04, -6, .zenith, 7.12013e-06, 5.96109e-06, 1.71986e-05, 3.58592e-06, 3.62846e-06, 1.27152e-05, 0.0029),
        R(.tau04, -6, .sun5, 5.98755e-05, 2.18393e-05, 2.50216e-05, 3.79530e-05, 1.81877e-05, 2.44820e-05, 0.0056),
        R(.tau04, -6, .sun15, 7.50341e-05, 4.19873e-05, 5.62719e-05, 3.71095e-05, 2.33809e-05, 3.85848e-05, 0.0029),
        R(.tau04, -6, .window, 1.25681e-05, 9.41173e-06, 2.29658e-05, 7.78429e-06, 6.76897e-06, 1.90661e-05, 0.0036),
        R(.tau04, -6, .anti0_5, 4.40609e-06, 3.25873e-06, 7.77117e-06, 4.40609e-06, 3.25873e-06, 7.77117e-06, 0.0102),
        R(.hazy, 79, .zenith, 1.40907e-01, 1.69847e-01, 2.03580e-01, 7.70884e-03, 1.38405e-02, 2.78696e-02, 0.0003),
        R(.hazy, 79, .sun5, 5.58129e-02, 6.63498e-02, 7.85966e-02, 3.09257e-02, 4.12320e-02, 5.56210e-02, 0.0016),
        R(.hazy, 79, .sun15, 4.21641e-02, 5.84218e-02, 8.25769e-02, 1.81501e-02, 2.96930e-02, 5.08826e-02, 0.0013),
        R(.hazy, 79, .window, 2.67857e-02, 3.99507e-02, 6.45440e-02, 9.23541e-03, 1.71534e-02, 3.54141e-02, 0.0009),
        R(.hazy, 79, .anti0_5, 4.49010e-02, 5.11754e-02, 6.10873e-02, 3.01044e-02, 3.61061e-02, 4.61775e-02, 0.0017),
        R(.hazy, 20, .zenith, 8.86363e-03, 1.31343e-02, 2.24408e-02, 3.16771e-03, 5.90368e-03, 1.28790e-02, 0.0012),
        R(.hazy, 20, .sun5, 4.46206e-01, 3.33326e-01, 1.96934e-01, 9.65977e-02, 9.16280e-02, 7.37442e-02, 0.0013),
        R(.hazy, 20, .sun15, 4.10909e-01, 3.69708e-01, 2.79266e-01, 5.72287e-02, 6.71780e-02, 7.22165e-02, 0.0008),
        R(.hazy, 20, .window, 1.48696e-02, 2.18510e-02, 3.45982e-02, 7.25207e-03, 1.24801e-02, 2.34675e-02, 0.0021),
        R(.hazy, 20, .anti0_5, 3.10052e-02, 3.23502e-02, 3.31299e-02, 1.92181e-02, 2.22397e-02, 2.58223e-02, 0.0027),
        R(.hazy, 6, .zenith, 4.02527e-03, 4.96275e-03, 8.10515e-03, 1.72978e-03, 2.53568e-03, 5.01331e-03, 0.0020),
        R(.hazy, 6, .sun5, 2.77239e-01, 1.15132e-01, 3.22238e-02, 5.86181e-02, 3.49267e-02, 1.82491e-02, 0.0017),
        R(.hazy, 6, .sun15, 1.84647e-01, 1.11620e-01, 5.69843e-02, 3.50159e-02, 2.79201e-02, 2.18410e-02, 0.0013),
        R(.hazy, 6, .window, 7.83147e-03, 9.00510e-03, 1.25974e-02, 4.11221e-03, 5.42879e-03, 8.89790e-03, 0.0022),
        R(.hazy, 6, .anti0_5, 1.11559e-02, 8.30528e-03, 7.48189e-03, 8.30445e-03, 7.23217e-03, 7.27382e-03, 0.0041),
        R(.hazy, 2, .zenith, 1.94482e-03, 2.05384e-03, 3.63270e-03, 8.46371e-04, 1.02836e-03, 2.18190e-03, 0.0011),
        R(.hazy, 2, .sun5, 8.74932e-02, 2.38685e-02, 7.04603e-03, 2.41173e-02, 1.05881e-02, 5.86983e-03, 0.0017),
        R(.hazy, 2, .sun15, 6.45556e-02, 2.98297e-02, 1.57594e-02, 1.56777e-02, 9.54945e-03, 7.67546e-03, 0.0011),
        R(.hazy, 2, .window, 3.76604e-03, 3.62874e-03, 5.42710e-03, 1.98717e-03, 2.15731e-03, 3.77049e-03, 0.0015),
        R(.hazy, 2, .anti0_5, 3.33099e-03, 2.40595e-03, 2.81786e-03, 3.17723e-03, 2.39244e-03, 2.81766e-03, 0.0050),
        R(.hazy, 0, .zenith, 9.50864e-04, 9.26089e-04, 1.84788e-03, 4.00246e-04, 4.42402e-04, 1.07417e-03, 0.0011),
        R(.hazy, 0, .sun5, 3.03762e-02, 7.53672e-03, 2.97421e-03, 9.96802e-03, 3.90461e-03, 2.63693e-03, 0.0022),
        R(.hazy, 0, .sun15, 2.41172e-02, 1.01144e-02, 6.46601e-03, 6.73607e-03, 3.64971e-03, 3.46733e-03, 0.0014),
        R(.hazy, 0, .window, 1.80632e-03, 1.59225e-03, 2.69512e-03, 9.27943e-04, 9.13864e-04, 1.83178e-03, 0.0015),
        R(.hazy, 0, .anti0_5, 1.28592e-03, 9.23147e-04, 1.31501e-03, 1.28572e-03, 9.23146e-04, 1.31501e-03, 0.0048),
        R(.hazy, -3, .zenith, 1.37823e-04, 1.21154e-04, 3.19154e-04, 5.29132e-05, 5.32846e-05, 1.78757e-04, 0.0013),
        R(.hazy, -3, .sun5, 3.58157e-03, 8.08087e-04, 5.04693e-04, 1.27621e-03, 4.49092e-04, 4.55241e-04, 0.0032),
        R(.hazy, -3, .sun15, 2.29204e-03, 9.50477e-04, 1.03148e-03, 8.33921e-04, 4.13095e-04, 5.98884e-04, 0.0025),
        R(.hazy, -3, .window, 2.53011e-04, 1.99384e-04, 4.53109e-04, 1.21119e-04, 1.07793e-04, 3.00779e-04, 0.0016),
        R(.hazy, -3, .anti0_5, 1.25243e-04, 8.78125e-05, 1.90513e-04, 1.25243e-04, 8.78125e-05, 1.90513e-04, 0.0045),
        R(.hazy, -6, .zenith, 5.49178e-06, 4.87411e-06, 1.53479e-05, 2.77597e-06, 2.70198e-06, 1.05267e-05, 0.0037),
        R(.hazy, -6, .sun5, 2.09887e-04, 5.44074e-05, 4.01345e-05, 8.50887e-05, 3.02233e-05, 3.41874e-05, 0.0054),
        R(.hazy, -6, .sun15, 1.08166e-04, 5.86680e-05, 8.43093e-05, 5.56859e-05, 2.82857e-05, 4.82765e-05, 0.0097),
        R(.hazy, -6, .window, 1.04514e-05, 8.37139e-06, 2.28124e-05, 6.33707e-06, 5.51555e-06, 1.76906e-05, 0.0042),
        R(.hazy, -6, .anti0_5, 4.66526e-06, 3.55187e-06, 9.30337e-06, 4.66526e-06, 3.55187e-06, 9.30337e-06, 0.0093),
        R(.builtin, 6, .sun1, 7.89498e-01, 3.65069e-01, 9.71869e-02, 5.59301e-02, 3.20229e-02, 1.78079e-02, 0.0005),
        R(.builtin, 6, .sun3, 5.57810e-01, 2.91060e-01, 9.73811e-02, 3.53346e-02, 2.42492e-02, 1.80964e-02, 0.0004),
        R(.builtin, 6, .sun8, 2.83596e-01, 1.67395e-01, 8.04380e-02, 1.59332e-02, 1.36177e-02, 1.59683e-02, 0.0004),
        R(.builtin, 6, .sun12, 1.70551e-01, 1.07090e-01, 6.26957e-02, 9.96946e-03, 9.53121e-03, 1.36170e-02, 0.0004),
        R(.builtin, 6, .sun30, 2.12922e-02, 2.00955e-02, 2.46030e-02, 2.26677e-03, 3.26070e-03, 7.31555e-03, 0.0006),
        R(.builtin, 2, .sun1, 4.65985e-01, 1.21892e-01, 1.32990e-02, 3.50821e-02, 1.35224e-02, 6.34459e-03, 0.0004),
        R(.builtin, 2, .sun3, 3.17930e-01, 9.97319e-02, 1.67576e-02, 2.24013e-02, 1.07811e-02, 7.02286e-03, 0.0004),
        R(.builtin, 2, .sun8, 1.35747e-01, 5.30406e-02, 1.96649e-02, 1.01980e-02, 6.36371e-03, 6.84815e-03, 0.0004),
        R(.builtin, 2, .sun12, 7.48509e-02, 3.33549e-02, 1.82037e-02, 6.36179e-03, 4.53563e-03, 6.04990e-03, 0.0005),
        R(.builtin, 2, .sun30, 1.10241e-02, 8.71186e-03, 1.06579e-02, 1.45118e-03, 1.63444e-03, 3.40101e-03, 0.0005),
        R(.builtin, -3, .sun1, 4.73757e-03, 1.26293e-03, 6.04007e-04, 1.43948e-03, 6.67104e-04, 5.79995e-04, 0.0015),
        R(.builtin, -3, .sun3, 5.35232e-03, 2.16203e-03, 1.00800e-03, 1.08261e-03, 6.35122e-04, 6.76031e-04, 0.0008),
        R(.builtin, -3, .sun8, 3.38436e-03, 1.95337e-03, 1.87537e-03, 5.52552e-04, 4.02806e-04, 6.80707e-04, 0.0006),
        R(.builtin, -3, .sun12, 2.38306e-03, 1.50334e-03, 1.87173e-03, 3.72360e-04, 2.92950e-04, 5.91619e-04, 0.0005),
        R(.builtin, -3, .sun30, 8.62824e-04, 6.02222e-04, 1.07124e-03, 1.18795e-04, 1.12578e-04, 3.15482e-04, 0.0004),
        R(.hazy, 6, .sun1, 1.91883e-01, 6.54551e-02, 1.86010e-02, 5.30649e-02, 2.72568e-02, 1.37192e-02, 0.0028),
        R(.hazy, 6, .sun3, 2.51414e-01, 9.14155e-02, 2.40847e-02, 5.95404e-02, 3.23393e-02, 1.59864e-02, 0.0021),
        R(.hazy, 6, .sun8, 2.67905e-01, 1.31476e-01, 4.56070e-02, 5.18829e-02, 3.49601e-02, 2.09635e-02, 0.0014),
        R(.hazy, 6, .sun12, 2.22609e-01, 1.25506e-01, 5.57727e-02, 4.17716e-02, 3.13563e-02, 2.21621e-02, 0.0013),
        R(.hazy, 6, .sun30, 6.10277e-02, 4.54734e-02, 3.56254e-02, 1.49627e-02, 1.43724e-02, 1.56374e-02, 0.0016),
        R(.hazy, 2, .sun1, 3.58979e-02, 8.54425e-03, 4.37811e-03, 1.88369e-02, 7.33899e-03, 4.36532e-03, 0.0038),
        R(.hazy, 2, .sun3, 6.71648e-02, 1.50357e-02, 5.28240e-03, 2.31392e-02, 9.21551e-03, 5.09618e-03, 0.0024),
        R(.hazy, 2, .sun8, 9.19435e-02, 3.20391e-02, 1.09620e-02, 2.24064e-02, 1.12789e-02, 6.95023e-03, 0.0014),
        R(.hazy, 2, .sun12, 7.78460e-02, 3.27933e-02, 1.47431e-02, 1.84590e-02, 1.05138e-02, 7.62708e-03, 0.0012),
        R(.hazy, 2, .sun30, 2.19888e-02, 1.33945e-02, 1.17059e-02, 6.80385e-03, 5.15501e-03, 5.90950e-03, 0.0012),
        R(.hazy, -3, .sun1, 1.01722e-03, 3.00178e-04, 3.41062e-04, 8.73248e-04, 2.97220e-04, 3.41056e-04, 0.0097),
        R(.hazy, -3, .sun3, 2.54576e-03, 4.99394e-04, 4.00500e-04, 1.18775e-03, 3.81923e-04, 3.95951e-04, 0.0051),
        R(.hazy, -3, .sun8, 3.62581e-03, 1.06300e-03, 7.39305e-04, 1.20531e-03, 4.85830e-04, 5.40636e-04, 0.0028),
        R(.hazy, -3, .sun12, 2.86265e-03, 1.05017e-03, 9.63422e-04, 9.99067e-04, 4.54823e-04, 5.93755e-04, 0.0037),
        R(.hazy, -3, .sun30, 8.43498e-04, 5.00791e-04, 8.62888e-04, 3.56163e-04, 2.28065e-04, 4.68002e-04, 0.0019),
    ]

    @MainActor
    func testLUTSkyMatchesTheBruteForceReferenceWithAerosols() throws {
        let h = try MS.Harness()
        var worst: [String: Float] = [:]
        var table = "τ550    sun  view     ref_Y       gpu_Y       stops   ref R/B  gpu R/B  ref orders≥2\n"
        var keys: [(RefCfg, Float)] = []
        for r in Self.reference where !keys.contains(where: { $0.0 == r.cfg && $0.1 == r.el }) { keys.append((r.cfg, r.el)) }
        for (cfg, el) in keys {
            let rows = Self.reference.filter { $0.cfg == cfg && $0.el == el }
            let got = try h.sky(Self.sky(el: el, ms: true, aerosol: cfg.aerosol), rows.map { $0.view.dir })
            for (r, g) in zip(rows, got) {
                let yr = simd_dot(r.rgb, Self.luma), yg = simd_dot(g, Self.luma)
                let s = Self.stops(yg, yr)
                XCTAssertTrue(s.isFinite, "\(cfg) \(el)° \(r.view): \(g)")
                let key = "\(cfg.rawValue)/\(r.view.rawValue)"
                worst[key] = abs(s) > abs(worst[key] ?? 0) ? s : (worst[key] ?? 0)
                let name = cfg == .builtin ? "0.028" : String(format: "%.3g", cfg.aerosol.opticalDepth)
                table += String(format: "%-6@ %+4.0f  %-8@ %.4e  %.4e  %+.3f   %6.2f   %6.2f   %3.0f %%\n", name, el, r.view.rawValue, yr, yg, s,
                                r.rgb.x / r.rgb.z, g.x / g.z, 100 * simd_dot(r.ms, Self.luma) / yr)
                // Tolerances (stops): the built-in kernels' own documented limits (VZ-0159: the
                // anti-solar horizon to 0.41, the sunward horizon 1° up at −3° to 0.33); the
                // aerosol path within 0.2 everywhere (0.15 measured — its Mie shape tables carry
                // the forward-scattered sunward field through the last bounce and the re-scatter).
                let tol: Float
                switch (cfg, r.view) {
                case (.builtin, .anti0_5): tol = 0.45
                case (.builtin, .sun1): tol = 0.35
                default: tol = 0.2
                }
                XCTAssertLessThan(abs(s), tol, "\(cfg) \(el)° \(r.view): \(s) stops from the reference")
                // Chromaticity: R/B within ±0.25 stop of the reference's on the aerosol path; the
                // built-in kernels' limits (±0.6, ±0.9 on the sunward horizon at low sun where R/B
                // is 4–40 and falls steeply with elevation, ±0.7 on the anti-solar horizon — the
                // documented lavender-for-violet limit).
                let rb = Self.stops(g.x / g.z, r.rgb.x / r.rgb.z)
                let rbTol: Float = cfg != .builtin ? 0.25
                    : ([.sun1, .sun3].contains(r.view) ? 0.9 : (r.view == .anti0_5 ? 0.7 : 0.6))
                XCTAssertLessThan(abs(rb), rbTol, "\(cfg) \(el)° \(r.view): R/B \(g.x / g.z) vs \(r.rgb.x / r.rgb.z)")
            }
        }
        let summary = worst.sorted { $0.key < $1.key }.map { String(format: "%@ %+.2f", $0.key, $0.value) }.joined(separator: ", ")
        print("── Aerosol MS sky vs brute-force reference (per unit E) ──\n" + table + "worst: " + summary)
    }

    // MARK: - 5. Every consumer sees the same aerosol

    @MainActor
    func testEveryConsumerSeesTheAerosol() throws {
        let h = try MS.Harness()
        let engine = SimEngine.shared
        // The flags Illuminatorama's in-view pass reads from the buffer it is given.
        _ = try h.sky(Self.sky(el: 20, ms: true, aerosol: .builtin), [SIMD3(0, 1, 0)])
        XCTAssertFalse(VolumetricCloudRenderer.aerosolRequested(in: h.cloud.skyUniformsBuffer))
        let lutRange = VolumetricCloudRenderer.atmosphereLUTOffset..<VolumetricCloudRenderer.skyUniformsBufferLength
        // The built-in region (`kLUTLength`), then the aerosol medium's Mie shape tables.
        let builtinRange = lutRange.lowerBound..<(lutRange.lowerBound
            + VolumetricCloudRenderer.AtmosphereLUT.builtinFloat4Count * MemoryLayout<SIMD4<Float>>.stride)
        func lutBytes(_ r: Range<Int>) -> [UInt8] {
            Array(UnsafeRawBufferPointer(start: h.cloud.skyUniformsBuffer.contents() + r.lowerBound, count: r.count))
        }
        let lut0 = lutBytes(builtinRange)
        XCTAssertTrue(lutBytes(builtinRange.upperBound..<lutRange.upperBound).allSatisfy { $0 == 0 },
                      "the built-in build never touches the Mie shape tables")
        _ = try h.sky(Self.sky(el: 20, ms: true, aerosol: .hazySuburban), [SIMD3(0, 1, 0)])
        XCTAssertTrue(VolumetricCloudRenderer.aerosolRequested(in: h.cloud.skyUniformsBuffer))
        XCTAssertTrue(VolumetricCloudRenderer.multipleScatteringRequested(in: h.cloud.skyUniformsBuffer))
        XCTAssertNotEqual(lutBytes(builtinRange), lut0, "a new aerosol rebuilds the LUTs")
        let hazyLUT = lutBytes(lutRange)
        let tables = lutBytes(builtinRange.upperBound..<lutRange.upperBound)
        XCTAssertGreaterThan(tables.filter { $0 != 0 }.count, tables.count / 2, "the aerosol build writes its Mie shape tables")
        _ = try h.sky(Self.sky(el: 20, ms: true, aerosol: .builtin), [SIMD3(0, 1, 0)])
        XCTAssertEqual(lutBytes(builtinRange), lut0, "and back: the built-in LUTs, bit for bit (the build is deterministic)")
        _ = try h.sky(Self.sky(el: 20, ms: true, aerosol: .hazySuburban), [SIMD3(0, 1, 0)])
        XCTAssertEqual(lutBytes(lutRange), hazyLUT, "and the aerosol build, tables included, is deterministic too")

        // The in-view pass draws the same sky as the dome for the same rays (the `_aerosol` kernels).
        let W = 48, H = 24
        let proj = simd_float4x4(columns: (SIMD4(1, 0, 0, 0), SIMD4(0, 2, 0, 0), SIMD4(0, 0, 0, 1), SIMD4(0, 0, 0.1, 0)))
        let view = simd_float4x4(columns: (SIMD4(0, 0, -1, 0), SIMD4(0, 1, 0, 0), SIMD4(1, 0, 0, 0), SIMD4(0, 0, 0, 1)))
        let invVP = (proj * view).inverse
        let rays: [SIMD3<Float>] = (0..<(W * H)).map { i in
            let uv = SIMD2<Float>((Float(i % W) + 0.5) / Float(W), (Float(i / W) + 0.5) / Float(H))
            let w = invVP * SIMD4<Float>(uv.x * 2 - 1, 1 - uv.y * 2, 1, 1)
            return simd_normalize(SIMD3(w.x, w.y, w.z) / w.w)
        }
        let depth = try MS.clearedDepth(h.device, h.queue, W, H)
        for ms in [false, true] {
            let p = Self.sky(el: 2, ms: ms, aerosol: .hazySuburban)
            let dome = try h.sky(p, rays)
            guard let pso = engine.pipeline(ms ? "illumi_cloud_inview_ms_aerosol" : "illumi_cloud_inview_aerosol") else {
                return XCTFail("aerosol in-view kernels missing")
            }
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: W, height: H, mipmapped: false)
            td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .shared
            guard let out = h.device.makeTexture(descriptor: td), let cb = h.queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no resources") }
            var cv = MS.InViewUniforms(invVP: invVP, cam: .zero, extra: SIMD4(0, -1, -1, -1), night: SIMD4(0, ms ? 1 : 0, 0, 0))
            let u = h.cloud.skyUniformsBuffer
            enc.setComputePipelineState(pso)
            enc.setTexture(out, index: 0); enc.setTexture(h.cloud.noiseTexture, index: 1); enc.setTexture(depth, index: 2)
            enc.setBuffer(u, offset: 0, index: 0)
            enc.setBytes(&cv, length: MemoryLayout<MS.InViewUniforms>.stride, index: 1)
            enc.setBuffer(h.cloud.fallbackBurstLightBuffer, offset: 0, index: 2)
            if ms { enc.setBuffer(u, offset: VolumetricCloudRenderer.atmosphereLUTOffset, index: 3) }
            enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
            if let e = cb.error { throw e }
            var px = [Float](repeating: 0, count: W * H * 4)
            px.withUnsafeMutableBytes { out.getBytes($0.baseAddress!, bytesPerRow: W * 16, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0) }
            var worst: Float = 0
            for i in 0..<(W * H) where simd_dot(dome[i], Self.luma) > 1e-9 {
                let c = SIMD3(px[i * 4], px[i * 4 + 1], px[i * 4 + 2])
                worst = max(worst, abs(simd_dot(c, Self.luma) / simd_dot(dome[i], Self.luma) - 1))
            }
            // (As for the built-in pair: the same march compiled into two kernels differs in the last
            // bits under fast-math — ~1e-3 at worst on the single-scatter horizon.)
            XCTAssertLessThan(worst, ms ? 1e-4 : 2e-3, "in-view (\(ms ? "MS" : "SS") aerosol kernel) vs the dome's sky")
        }
    }

    /// The sun disc and a host's directional light get `Params.sunTransmittance()`; the deck's sun
    /// (`cloudLightingFromAtmosphere`) is the GPU's march through the same air. They agree.
    @MainActor
    func testSunTransmittanceIsTheSkysOwn() throws {
        let h = try MS.Harness()
        var report = "aerosol          MS   sun    CPU T (r,g,b)            GPU T (r,g,b)            worst Δ\n"
        for (name, a) in [("built-in", NishitaAerosol.builtin), ("rural", .rural), ("hazy suburban", .hazySuburban)] {
            for ms in [false, true] {
                for el: Float in [60, 30, 15, 6, 3, 1] {
                    var p = Self.sky(el: el, ms: ms, aerosol: a)
                    p.cloudLightingFromAtmosphere = true
                    p.cloudBaseY = 0            // the prepass evaluates the sun at max(base, 1 m): the ground
                    _ = try h.sky(p, [SIMD3(0, 1, 0)])
                    let u = h.cloud.skyUniformsBuffer.contents().bindMemory(to: SkyUniforms.self, capacity: 1).pointee
                    let gpu = SIMD3(u.cloudLitSun.x, u.cloudLitSun.y, u.cloudLitSun.z) / p.atmosphereIntensity
                    let cpu = p.sunTransmittance(altitude: 1)
                    let rel = abs(gpu / cpu - 1)
                    let worst = max(rel.x, rel.y, rel.z)
                    report += String(format: "%-15@ %@  %+3.0f   (%.4f %.4f %.4f)   (%.4f %.4f %.4f)   %.3f\n", name, ms ? "on " : "off", el,
                                     cpu.x, cpu.y, cpu.z, gpu.x, gpu.y, gpu.z, worst)
                    // The aerosol kernels march the sun path finely (LUT: 160 steps; single-scatter
                    // prepass: 64 crowded to the observer). The BUILT-IN kernels keep their legacy
                    // quadrature, bit for bit: 40 LUT steps (≤ 3 %) and 16 uniform steps (≤ 8 %).
                    let tol: Float = a == .builtin ? (ms ? 0.03 : 0.08) : 0.02
                    XCTAssertLessThan(worst, tol, "\(name) MS \(ms) sun \(el)°: GPU \(gpu) vs CPU \(cpu)")
                }
            }
        }
        // More aerosol: less and redder sunlight.
        let tb = NishitaAtmosphere.transmittance(toLight: MS.toSun(6), aerosol: .builtin)
        let th = NishitaAtmosphere.transmittance(toLight: MS.toSun(6), aerosol: .hazySuburban)
        XCTAssertLessThan(th.y, tb.y)
        XCTAssertGreaterThan(th.x / th.z, tb.x / tb.z)
        XCTAssertEqual(NishitaAtmosphere.transmittance(toLight: MS.toSun(-1), aerosol: .hazySuburban), .zero, "below the horizon: shadowed")
        print("── Sun transmittance: CPU (Params.sunTransmittance) vs the GPU prepass ──\n" + report)
    }

    // MARK: - 6. Noon: up-hemisphere E/π (Daydream's calibration)

    /// Daydream Home holds its noon sky's up-hemisphere irradiance (E/π, the sun disc left out —
    /// `sunDiscHalfAngle` 11°) at the single-scatter sky's with `atmosphereIntensity` × 0.6737 under
    /// multiple scattering. Its reference instant is lat 34°, day 172, noon: the sun 79.4° up. The
    /// dome kernel's own output, the probe kernel's own integral; the factor is solved at
    /// intensity 20 (the dome's soft Reinhard makes it not quite linear).
    /// Daydream's noon: lat 34°, day 172 → the sun 79.44° up.
    static let noonElevation: Float = 79.44

    /// The dome's up-facing E/π at noon for this sky — (sun cone of 11° left out, whole hemisphere) —
    /// through the renderer's own dome bake and the radiance-probe kernel.
    @MainActor
    final class NoonProbe {
        let engine = SimEngine.shared
        let cloud: VolumetricCloudRenderer
        let probe: MTLComputePipelineState
        let out: MTLBuffer
        init() throws {
            cloud = VolumetricCloudRenderer(engine: engine, resolution: SIMD2(1024, 512), iblResolution: SIMD2(32, 16))
            cloud.minRenderInterval = 0
            guard let p = engine.pipeline("skyRadianceProbe"),
                  let o = engine.device.makeBuffer(length: 2 * 16, options: .storageModeShared) else { throw XCTSkip("no probe") }
            probe = p; out = o
        }
        func ePi(_ a: NishitaAerosol, ms: Bool, intensity: Float) throws -> (SIMD3<Float>, SIMD3<Float>) {
            let el = NishitaAerosolTests.noonElevation
            let probes: [VolumetricCloudRenderer.RadianceProbe] = [
                .init(normal: SIMD3(0, 1, 0), excludeDirection: MS.toSun(el), excludeHalfAngle: 11 * .pi / 180),
                .init(normal: SIMD3(0, 1, 0)),
            ]
            var p = NishitaAerosolTests.sky(el: el, ms: ms, aerosol: a)
            p.atmosphereIntensity = intensity
            cloud.renderNow(params: p, blocking: true)
            guard let cb = engine.commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no cb") }
            let gpu = VolumetricCloudRenderer.radianceProbeGPUData(probes)
            enc.setComputePipelineState(probe)
            enc.setTexture(cloud.outputTexture, index: 0)
            gpu.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 0) }
            enc.setBuffer(out, offset: 0, index: 1)
            enc.dispatchThreadgroups(MTLSize(width: probes.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
            let v = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: 2)
            return (SIMD3(v[0].x, v[0].y, v[0].z), SIMD3(v[1].x, v[1].y, v[1].z))
        }
    }

    @MainActor
    func testNoonUpHemisphereIrradiance() throws {
        let noon = try NoonProbe()
        let el = Self.noonElevation
        func ePi(_ a: NishitaAerosol, ms: Bool, intensity: Float) throws -> (SIMD3<Float>, SIMD3<Float>) {
            try noon.ePi(a, ms: ms, intensity: intensity)
        }
        let ss = try ePi(.builtin, ms: false, intensity: 20)
        let target = simd_dot(ss.0, Self.luma)
        var report = String(format: "sun %.2f° up, up-facing E/π (luma; rgb), atmosphereIntensity 20:\n", el)
        report += String(format: "  SS built-in (Daydream's reference): %.4f  (%.4f %.4f %.4f)  [sun cone kept: %.4f]\n",
                         target, ss.0.x, ss.0.y, ss.0.z, simd_dot(ss.1, Self.luma))
        var factors: [String: Float] = [:]
        for (name, a) in [("MS built-in", NishitaAerosol.builtin), ("MS rural", .rural), ("MS hazy suburban", .hazySuburban), ("MS urban", .urban)] {
            let e20 = try ePi(a, ms: true, intensity: 20)
            // Solve atmosphereIntensity = 20·k so the sun-cone-excluded luma matches SS's (secant).
            var k0: Float = 0.5, k1: Float = 0.8
            var f0 = simd_dot(try ePi(a, ms: true, intensity: 20 * k0).0, Self.luma) - target
            var f1 = simd_dot(try ePi(a, ms: true, intensity: 20 * k1).0, Self.luma) - target
            for _ in 0..<8 where abs(f1) > target * 1e-4 {
                let k2 = k1 - f1 * (k1 - k0) / (f1 - f0)
                k0 = k1; f0 = f1; k1 = max(0.05, k2)
                f1 = simd_dot(try ePi(a, ms: true, intensity: 20 * k1).0, Self.luma) - target
            }
            factors[name] = k1
            report += String(format: "  %-17@ at 20: %.4f  (%.4f %.4f %.4f)  %+.2f stops vs SS  [sun cone kept: %.4f]  → atmosphereIntensity = 20 × %.4f holds SS's E/π\n",
                             name, simd_dot(e20.0, Self.luma), e20.0.x, e20.0.y, e20.0.z, Self.stops(simd_dot(e20.0, Self.luma), target),
                             simd_dot(e20.1, Self.luma), k1)
        }
        print("── Noon up-hemisphere E/π (sun disc out) ──\n" + report)
        // Daydream measured 0.6737 for the built-in multiply-scattered sky on its own dome (its grade
        // is luma-preserving); this bare dome must land close to it.
        XCTAssertEqual(factors["MS built-in"]!, 0.6737, accuracy: 0.03)
        // More aerosol brightens the noon hemisphere (more diffuse light): a smaller factor.
        XCTAssertLessThan(factors["MS hazy suburban"]!, factors["MS built-in"]!)
    }

    // MARK: - 7. The golden band and the twilight arch

    /// Koomen, Lock, Packer, Scolnik, Tousey & Hulburt (1952), "Measurements of the Brightness of
    /// the Twilight Sky", JOSA 42:353, Tables I and II (candles/ft², photopic): [H][Z][P] with
    /// H = sun +5, +3, 0, −3, −6°; Z = 0 (sunward), 180° (anti-sunward); P = 0, 10, 30, 50, 70, 90°
    /// (zenith). Sacramento Peak (2800 m, clean desert air) and Maryland (30 m, humid, hazier).
    static let koomenH: [Float] = [5, 3, 0, -3, -6]
    static let koomenP: [Float] = [0, 10, 30, 50, 70, 90]
    static let koomenMaryland: [[[Float]]] = [   // [H][Z0, Z180][P]
        [[1000, 1000, 350, 150, 80, 43], [180, 170, 110, 65, 50, 43]],
        [[.nan, 600, 220, 100, 62, 35], [120, 120, 78, 54, 40, 35]],
        [[.nan, 150, 74, 35, 21, 15], [42, 40, 35, 23, 19, 15]],
        [[17, 15, 9.5, 4.0, 3.0, 2], [2, 2, 4, 3.2, 2, 2]],
        [[0.8, 0.7, 0.3, 0.12, 0.08, 0.06], [0.09, 0.08, 0.07, 0.07, 0.05, 0.06]],
    ]
    static let koomenSacramentoPeak: [[[Float]]] = [
        [[1000, 1000, 150, 63, 35, 31], [190, 170, 80, 50, 35, 31]],
        [[.nan, 560, 97, 44, 24, 21], [120, 115, 60, 36, 24, 21]],
        [[.nan, 150, 37, 16.5, 9.5, 8.0], [38, 37, 22, 14, 9.5, 8.0]],
        [[20, 15, 4.7, 1.9, 1.3, 1.0], [2.4, 2.0, 2.1, 1.6, 1.1, 1.0]],
        [[0.9, 0.7, 0.18, 0.06, 0.033, 0.022], [0.078, 0.068, 0.043, 0.035, 0.025, 0.022]],
    ]

    /// Preetham, Shirley & Smits (1999) "A Practical Analytic Model for Daylight", App. A.2: the
    /// Perez distribution with turbidity-linear coefficients, zenith luminance (kcd/m²) and zenith
    /// chromaticity. A fit to a double-scatter simulation for T = 2…6; Zotti, Wilkie & Purgathofer
    /// (2007) show it unreliable for a low sun (too-flat circumsolar peak, antisolar sky 2–5× too
    /// bright for θs ≥ 60°) — a sanity anchor for chromaticity by day, not ground truth at +6°.
    enum Preetham {
        static func perez(_ th: Double, _ ga: Double, _ c: [Double]) -> Double {
            (1 + c[0] * exp(c[1] / cos(th))) * (1 + c[2] * exp(c[3] * ga) + c[4] * cos(ga) * cos(ga))
        }
        /// (Y cd/m², x, y) for the sun `sunEl`° up (azimuth 0), view `el`° up at azimuth `az`°.
        static func sky(T: Double, sunEl: Double, el: Double, az: Double) -> SIMD3<Double> {
            let ts = (90 - sunEl) * .pi / 180, th = (90 - el) * .pi / 180
            let s = SIMD3<Double>(cos(sunEl * .pi / 180), sin(sunEl * .pi / 180), 0)
            let v = SIMD3<Double>(cos(el * .pi / 180) * cos(az * .pi / 180), sin(el * .pi / 180), cos(el * .pi / 180) * sin(az * .pi / 180))
            let ga = acos(min(1, max(-1, simd_dot(s, v))))
            let cY = [0.1787 * T - 1.4630, -0.3554 * T + 0.4275, -0.0227 * T + 5.3251, 0.1206 * T - 2.5771, -0.0670 * T + 0.3703]
            let cx = [-0.0193 * T - 0.2592, -0.0665 * T + 0.0008, -0.0004 * T + 0.2125, -0.0641 * T - 0.8989, -0.0033 * T + 0.0452]
            let cy = [-0.0167 * T - 0.2608, -0.0950 * T + 0.0092, -0.0079 * T + 0.2102, -0.0441 * T - 1.6537, -0.0109 * T + 0.0529]
            let chi = (4.0 / 9 - T / 120) * (.pi - 2 * ts)
            let Yz = ((4.0453 * T - 4.9710) * tan(chi) - 0.2155 * T + 2.4192) * 1000
            let t3 = ts * ts * ts, t2 = ts * ts
            let xz = T * T * (0.00166 * t3 - 0.00375 * t2 + 0.00209 * ts) + T * (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * ts + 0.00394)
                + (0.11693 * t3 - 0.21196 * t2 + 0.06052 * ts + 0.25886)
            let yz = T * T * (0.00275 * t3 - 0.00610 * t2 + 0.00317 * ts) + T * (-0.04214 * t3 + 0.08970 * t2 - 0.04153 * ts + 0.00516)
                + (0.15346 * t3 - 0.26756 * t2 + 0.06670 * ts + 0.26688)
            return SIMD3(Yz * perez(th, ga, cY) / perez(0, ts, cY), xz * perez(th, ga, cx) / perez(0, ts, cx),
                         yz * perez(th, ga, cy) / perez(0, ts, cy))
        }
    }

    /// The sunward vertical at +6, +2 and −3° (and 30° off the sun's azimuth), single scatter vs
    /// multiple scattering with the built-in, rural and hazy-suburban air: where the golden band
    /// is, how tall, how bright against the zenith — and the twilight arch. Then the same skies
    /// against Koomen et al.'s measured luminance distributions, Preetham's turbidity model and
    /// Ugolnikov & Maslov (2016)'s multicolour twilight photometry.
    @MainActor
    func testGoldenBandAndTwilightArch() throws {
        let h = try MS.Harness()
        let els: [Float] = [0.5, 1, 2, 3, 5, 8, 12, 16, 20, 30, 45, 60, 90]
        let cfgs: [(String, NishitaAerosol, Bool)] = [("SS built-in", .builtin, false), ("MS built-in", .builtin, true),
                                                       ("MS rural", .rural, true), ("MS hazy", .hazySuburban, true)]
        var report = ""
        // Golden band: the highest elevation where the sky is still gold-orange (CIE x ≥ 0.40).
        var bandTop: [String: Float] = [:]
        var profile: [String: [SIMD3<Float>]] = [:]
        for sunEl: Float in [6, 2, -3] {
            report += String(format: "sun %+.0f° — sunward vertical: elevation: L/L_zenith  x,y  (R/B)\n", sunEl)
            for az: Float in [0, 30] {
                let dirs = els.map { View.dir(el: $0, az: az) }
                for (name, a, ms) in cfgs {
                    let s = try h.sky(Self.sky(el: sunEl, ms: ms, aerosol: a), dirs)
                    let zl = simd_dot(s.last!, Self.luma)
                    var line = String(format: "  az %2.0f %-12@", az, name)
                    var top: Float = 0
                    for (e, c) in zip(els, s) {
                        let q = Self.xy(c)
                        if q.x >= 0.40 && e != 90 { top = max(top, e) }
                        if [1, 3, 5, 8, 12, 20, 30, 90].contains(e) {
                            line += String(format: " %g°:%.1f (%.3f,%.3f)(%.1f)", e, simd_dot(c, Self.luma) / zl, q.x, q.y, c.x / c.z)
                        }
                    }
                    bandTop["\(name) \(sunEl) \(az)"] = top
                    if az == 0 { profile["\(name) \(sunEl)"] = s }
                    report += line + String(format: "  | gold (x≥0.40) up to %g°\n", top)
                }
            }
        }
        // The golden band is taller with haze, at the sun's azimuth and 30° off it.
        for sunEl: Float in [6, 2] {
            for az: Float in [0, 30] {
                XCTAssertGreaterThan(bandTop["MS hazy \(sunEl) \(az)"]!, bandTop["MS built-in \(sunEl) \(az)"]!,
                                     "sun \(sunEl)°, az \(az)°: the hazy golden band is taller")
            }
        }
        // The twilight arch at −3°: orange above the sunward horizon, blue overhead. With the hazy
        // preset (its haze reaching the air still sunlit above the Earth's shadow) the arch is
        // warmer than the built-in air across the band a house scene sees above its roofline
        // (5–20°) and at least as red as the single-scatter sky there — Daydream's hand-off gap,
        // closed from the physical side.
        let hz = profile["MS hazy -3.0"]!, bi = profile["MS built-in -3.0"]!, ss = profile["SS built-in -3.0"]!
        for (i, e) in els.enumerated() where [5, 8, 12, 16, 20].contains(e) {
            XCTAssertGreaterThan(Self.xy(hz[i]).x, Self.xy(bi[i]).x, "−3°, \(e)° up: hazy warmer than the built-in air")
            XCTAssertGreaterThan(Self.stops(hz[i].x / hz[i].z, ss[i].x / ss[i].z), -0.3, "−3°, \(e)° up: hazy MS as red as SS")
        }
        let warmest = zip(els, hz).max { Self.xy($0.1).x < Self.xy($1.1).x }!
        XCTAssertGreaterThanOrEqual(warmest.0, 3); XCTAssertLessThanOrEqual(warmest.0, 12)
        XCTAssertGreaterThan(Self.xy(warmest.1).x, 0.44, "an orange arch, not pale blue")
        XCTAssertGreaterThan(hz.last!.z, hz.last!.x, "blue overhead")
        report += String(format: "twilight arch, hazy, −3°: warmest at %g° (x,y %.3f,%.3f); zenith (%.3f,%.3f)\n",
                         warmest.0, Self.xy(warmest.1).x, Self.xy(warmest.1).y, Self.xy(hz.last!).x, Self.xy(hz.last!).y)

        // ── Koomen et al. (1952): luminance relative to the zenith, sunward and anti-sunward ──
        report += "Koomen 1952 — L/L_zenith, model vs Maryland (M) and Sacramento Peak (S); P0 taken at 1°:\n"
        var rmsLine = ""
        for (name, a, ms) in cfgs {
            var eM: [Float] = [], eS: [Float] = []
            var zenLine = ""
            var black = 0   // points the model leaves black (the single-scatter sky in the Earth's shadow)
            for (hi, H) in Self.koomenH.enumerated() {
                var dirs: [SIMD3<Float>] = []
                for az: Float in [0, 180] { for P in Self.koomenP { dirs.append(View.dir(el: max(P, 1), az: az)) } }
                let s = try h.sky(Self.sky(el: H, ms: ms, aerosol: a), dirs)
                let zen = simd_dot(s[5], Self.luma)
                zenLine += String(format: " %+.0f°:%+.2f", H, Self.stops(zen * MS.solarLux, Self.koomenMaryland[hi][0][5] * MS.cdPerFt2))
                for zi in 0..<2 {
                    for pi in 0..<5 {
                        let m = simd_dot(s[zi * 6 + pi], Self.luma) / zen
                        guard m > 0, m.isFinite else { black += 1; continue }
                        let km = Self.koomenMaryland[hi][zi][pi] / Self.koomenMaryland[hi][zi][5]
                        let ks = Self.koomenSacramentoPeak[hi][zi][pi] / Self.koomenSacramentoPeak[hi][zi][5]
                        if km.isFinite { eM.append(Self.stops(m, km)) }
                        if ks.isFinite { eS.append(Self.stops(m, ks)) }
                    }
                }
            }
            func rms(_ e: [Float]) -> Float { sqrt(e.map { $0 * $0 }.reduce(0, +) / Float(e.count)) }
            rmsLine += String(format: "  %-12@ rms vs Maryland %.2f stops, vs Sacramento Peak %.2f (%d points%@); zenith vs Maryland:%@\n",
                              name, rms(eM), rms(eS), eM.count, black > 0 ? ", \(black) black in the model — left out" : "", zenLine)
            if name == "MS hazy" || name == "MS rural" {
                XCTAssertLessThan(rms(eM), 0.85, "\(name): Koomen's hazier site within 0.85 stop rms")
            }
        }
        report += rmsLine
        // Sunward P10 and P30 at +5° (the golden band) — the numbers behind the rms.
        do {
            let dirs = [View.dir(el: 10, az: 0), View.dir(el: 30, az: 0), SIMD3<Float>(0, 1, 0)]
            for (name, a, ms) in cfgs {
                let s = try h.sky(Self.sky(el: 5, ms: ms, aerosol: a), dirs)
                let z = simd_dot(s[2], Self.luma)
                report += String(format: "  +5° sunward P10 %.1f× / P30 %.1f× the zenith — %@ (Maryland 23.3 / 8.1, Sacramento Peak 32.3 / 4.8)\n",
                                 simd_dot(s[0], Self.luma) / z, simd_dot(s[1], Self.luma) / z, name)
            }
        }

        // ── Preetham et al. (1999) at +20° and +6°: T 3.5 ≈ τ550 0.25 (hazy), T 2.5 ≈ 0.15 (rural) ──
        report += "Preetham 1999 — (x,y) and L/L_zenith; model vs Preetham at the same turbidity:\n"
        let pv: [(String, Float, Float)] = [("zenith", 90, 0), ("sun5", 5, 0), ("sun15", 15, 0), ("window", 30, 90), ("anti0.5", 0.5, 180)]
        for (name, a, T) in [("MS hazy", NishitaAerosol.hazySuburban, 3.5), ("MS rural", NishitaAerosol.rural, 2.5), ("MS built-in", NishitaAerosol.builtin, 2.0)] as [(String, NishitaAerosol, Double)] {
            for sunEl: Float in [20, 6] {
                let s = try h.sky(Self.sky(el: sunEl, ms: true, aerosol: a), pv.map { View.dir(el: $0.1, az: $0.2) })
                let zl = simd_dot(s[0], Self.luma)
                let pz = Preetham.sky(T: T, sunEl: Double(sunEl), el: 90, az: 0)
                var line = String(format: "  %-11@ vs T %.1f, sun %+.0f°:", name, T, sunEl)
                for (i, v) in pv.enumerated() {
                    let q = Self.xy(s[i]); let pr = Preetham.sky(T: T, sunEl: Double(sunEl), el: Double(v.1), az: Double(v.2))
                    line += String(format: " %@ (%.3f,%.3f)/(%.3f,%.3f) %.1f/%.1f×", v.0, q.x, q.y, pr.y, pr.z,
                                   simd_dot(s[i], Self.luma) / zl, pr.x / pz.x)
                }
                report += line + String(format: "  zenith %.0f vs %.0f cd/m²\n", zl * MS.solarLux, pz.x)
            }
        }

        // ── Ugolnikov & Maslov (2016), Fig. 1: I(ζ = +45°)/I(ζ = −45°) in the solar vertical ──
        // (RGB camera, 624 / 540 / 461 nm, central Russia, 27 March 2016). Read off the figure:
        // sun 0° (z 90°): R 1.68, G 1.52, B 1.40; sun −3° (z 93°): R ≈ 1.95, G ≈ 1.75, B ≈ 1.60.
        report += "Ugolnikov & Maslov 2016 — I(45° up, sunward)/I(45° up, anti-sunward), model (680/550/440 nm) vs measured (624/540/461):\n"
        for (name, a, ms) in cfgs {
            var line = String(format: "  %-12@", name)
            for (sunEl, meas) in [(Float(0), SIMD3<Float>(1.68, 1.52, 1.40)), (-3, SIMD3<Float>(1.95, 1.75, 1.60))] {
                let s = try h.sky(Self.sky(el: sunEl, ms: ms, aerosol: a), [View.dir(el: 45, az: 0), View.dir(el: 45, az: 180)])
                let r = s[0] / s[1]
                line += String(format: "  sun %+.0f°: (%.2f %.2f %.2f) vs (%.2f %.2f %.2f)", sunEl, r.x, r.y, r.z, meas.x, meas.y, meas.z)
            }
            report += line + "\n"
        }
        print("── Golden band + twilight arch ──\n" + report)
    }

    // MARK: - 8. The comparison sheet (and a sunset sweep)

    /// Renders a perspective view of the sky with Illuminatorama's in-view kernel (the production
    /// path: `illumi_cloud_inview[_ms][_aerosol]` reading the renderer's own uniforms + LUTs),
    /// 90° horizontal field of view, pitched 20° up, from 2 m. Linear HDR, per unit irradiance.
    @MainActor
    final class SheetCamera {
        let engine = SimEngine.shared
        let cloud: VolumetricCloudRenderer
        let depth: MTLTexture
        let out: MTLTexture
        let W: Int, H: Int
        init(W: Int, H: Int) throws {
            self.W = W; self.H = H
            cloud = VolumetricCloudRenderer(engine: engine, resolution: SIMD2(64, 32), iblResolution: SIMD2(32, 16))
            cloud.minRenderInterval = 0
            depth = try MS.clearedDepth(engine.device, engine.commandQueue, W, H)
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: W, height: H, mipmapped: false)
            td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .shared
            guard let t = engine.device.makeTexture(descriptor: td) else { throw XCTSkip("no texture") }
            out = t
        }
        /// The sky for `p`, looking toward azimuth `yawDeg` (0 = +X = the sun's azimuth).
        func render(_ p: VolumetricCloudRenderer.Params, yawDeg: Float, pitchDeg: Float = 20) throws -> [SIMD3<Float>] {
            cloud.render(params: p)
            let ms = p.physicalMultipleScattering, aerosol = p.physicalAerosol
            let name = "illumi_cloud_inview" + (ms ? "_ms" : "") + (aerosol ? "_aerosol" : "")
            guard let pso = engine.pipeline(name), let cb = engine.commandQueue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("\(name) missing") }
            let yaw = yawDeg * .pi / 180, pitch = pitchDeg * .pi / 180
            let f = SIMD3<Float>(cos(pitch) * cos(yaw), sin(pitch), cos(pitch) * sin(yaw))
            let right = simd_normalize(simd_cross(f, SIMD3(0, 1, 0)))
            let up = simd_cross(right, f)
            let ro = SIMD3<Float>(0, 2, 0)
            let aspect = Float(H) / Float(W)
            let m = simd_float4x4(columns: (SIMD4(right, 0), SIMD4(up * aspect, 0), SIMD4(f, 0), SIMD4(ro, 1)))
            var cv = MS.InViewUniforms(invVP: m, cam: SIMD4(ro, 0), extra: SIMD4(0, -1, -1, -1), night: SIMD4(0, ms ? 1 : 0, 0, 0))
            let u = cloud.skyUniformsBuffer
            enc.setComputePipelineState(pso)
            enc.setTexture(out, index: 0); enc.setTexture(cloud.noiseTexture, index: 1); enc.setTexture(depth, index: 2)
            enc.setBuffer(u, offset: 0, index: 0)
            enc.setBytes(&cv, length: MemoryLayout<MS.InViewUniforms>.stride, index: 1)
            enc.setBuffer(cloud.fallbackBurstLightBuffer, offset: 0, index: 2)
            if ms { enc.setBuffer(u, offset: VolumetricCloudRenderer.atmosphereLUTOffset, index: 3) }
            enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness
            if let e = cb.error { throw e }
            var px = [Float](repeating: 0, count: W * H * 4)
            px.withUnsafeMutableBytes { out.getBytes($0.baseAddress!, bytesPerRow: W * 16, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0) }
            var img = (0..<(W * H)).map { SIMD3(px[$0 * 4], px[$0 * 4 + 1], px[$0 * 4 + 2]) }
            // The sun: a small disc in its own transmitted colour (the dome's stylised disc + halo is
            // off here so it does not paint over the aureole being compared).
            let toSun = -simd_normalize(p.sunDir)
            if toSun.y > -0.005 {
                let T = p.sunTransmittance()
                for i in 0..<(W * H) {
                    let x = (Float(i % W) + 0.5) / Float(W) * 2 - 1, y = 1 - (Float(i / W) + 0.5) / Float(H) * 2
                    let d = simd_normalize(right * x + up * (y * aspect) + f)
                    let c = simd_dot(d, toSun)
                    if c > cos(0.55 * Float.pi / 180) { img[i] += T * 40 * smoothstep(cos(0.55 * Float.pi / 180), cos(0.3 * Float.pi / 180), c) }
                }
            }
            return img
        }
        func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float { let t = min(max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t) }
    }

    /// The sheet's sky: bare physical sky, unit irradiance, the ground lit by the same atmosphere
    /// (sky share + the direct sun through `sunTransmittance()` on an albedo-0.1 ground).
    static func sheetParams(el: Float, ms: Bool, aerosol: NishitaAerosol) -> VolumetricCloudRenderer.Params {
        var p = Self.sky(el: el, ms: ms, aerosol: aerosol)
        p.sunIntensity = 0
        p.cloudLightingFromAtmosphere = true
        p.groundFromAtmosphere = true
        let albedo = SIMD3<Float>(0.10, 0.10, 0.08)
        p.cloudGroundAlbedo = albedo
        p.groundColor = albedo / .pi * p.sunTransmittance() * max(0, -simd_normalize(p.sunDir).y) * p.atmosphereIntensity
        return p
    }

    /// Luminance-driven filmic curve (ACES fit on luma, chroma kept, then clipped) → sRGB bytes.
    static func tonemap(_ img: [SIMD3<Float>], exposure: Float) -> [UInt8] {
        var out = [UInt8](repeating: 255, count: img.count * 4)
        for (i, c0) in img.enumerated() {
            let c = simd_max(c0, .zero) * exposure
            let l = simd_dot(c, luma)
            let a = l * (2.51 * l + 0.03) / (l * (2.43 * l + 0.59) + 0.14)
            var d = l > 1e-9 ? c * (a / l) : .zero
            let mx = max(d.x, d.y, d.z)
            if mx > 1 { d = d + (SIMD3(repeating: mx) - d) * min((mx - 1) / mx, 1) ; d /= mx }   // desaturate the clip
            for k in 0..<3 {
                let v = min(max(d[k], 0), 1)
                let s = v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1 / 2.4) - 0.055
                out[i * 4 + k] = UInt8(min(max(s * 255 + 0.5, 0), 255))
            }
        }
        return out
    }

    static func drawText(_ ctx: CGContext, _ s: String, x: CGFloat, y: CGFloat, size: CGFloat,
                         bold: Bool = false, color: CGColor = CGColor(gray: 0.92, alpha: 1)) {
        let font = CTFontCreateWithName((bold ? "Helvetica-Bold" : "Helvetica") as CFString, size, nil)
        let attrs: [NSAttributedString.Key: Any] = [NSAttributedString.Key(kCTFontAttributeName as String): font,
                                                    NSAttributedString.Key(kCTForegroundColorAttributeName as String): color]
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attrs))
        ctx.textPosition = CGPoint(x: x, y: y)
        CTLineDraw(line, ctx)
    }

    static func image(_ bytes: [UInt8], W: Int, H: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(width: W, height: H, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: W * 4,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    @MainActor
    func testComparisonSheet() throws {
        let env = ProcessInfo.processInfo.environment
        let sheetPath = env["VIZ_AEROSOL_SHEET"] ?? FileManager.default.temporaryDirectory.appendingPathComponent("nishita_aerosol_sheet.png").path
        let W = 400, H = 300
        let cam = try SheetCamera(W: W, H: H)
        let columns: [(String, NishitaAerosol, Bool)] = [
            ("single scatter · built-in air", .builtin, false),
            ("multiple scattering · built-in air", .builtin, true),
            ("MS · rural (τ550 0.15, H 3 km)", .rural, true),
            ("MS · hazy suburban (τ550 0.25, H 4 km)", .hazySuburban, true),
        ]
        let nc = columns.count
        let rows: [Float] = [6, 2, -3, -6]
        // Each sky as a host calibrated like Daydream shows it: scaled so its NOON up-hemisphere E/π
        // (sun disc out) equals the single-scatter sky's — `atmosphereIntensity` × k.
        let noon = try NoonProbe()
        let ref = simd_dot(try noon.ePi(.builtin, ms: false, intensity: 1).0, Self.luma)
        var calib: [Float] = []
        for c in columns { calib.append(ref / simd_dot(try noon.ePi(c.1, ms: c.2, intensity: 1).0, Self.luma)) }
        let margin = 24, labelW = 150, gap = 10, header = 138, footer = 96
        let totalW = margin * 2 + labelW + 2 * nc * W + (2 * nc - 1) * gap + 2 * gap
        let totalH = header + rows.count * (H + gap) + footer
        guard let ctx = CGContext(data: nil, width: totalW, height: totalH, bitsPerComponent: 8, bytesPerRow: totalW * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw XCTSkip("no context")
        }
        ctx.setFillColor(CGColor(red: 0.075, green: 0.078, blue: 0.09, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: totalW, height: totalH))
        let top = CGFloat(totalH)
        Self.drawText(ctx, "Nishita sky with aerosols  —  single scatter vs multiple scattering, built-in air vs rural vs hazy suburban",
                      x: CGFloat(margin), y: top - 40, size: 26, bold: true)
        Self.drawText(ctx, "hazy suburban: τ550 0.25 (Preetham T ≈ 3.5), Ångström α 1.3, H 4 km, ω₀ 0.92, mean cosine 0.70 · rural: OPAC continental average, τ550 0.151, H 3 km · built-in: τ550 0.028, grey, H 1.2 km · 90° FOV, 20° up · each sky × k (its noon up-hemisphere E/π = the single-scatter sky's, as Daydream calibrates) · one exposure per row · ACES-fit tonemap on luminance",
                      x: CGFloat(margin), y: top - 68, size: 14, color: CGColor(gray: 0.7, alpha: 1))
        let x0 = margin + labelW
        let groupW = nc * W + (nc - 1) * gap
        Self.drawText(ctx, "WEST — looking into the sun", x: CGFloat(x0), y: top - 100, size: 19, bold: true,
                      color: CGColor(red: 1, green: 0.78, blue: 0.45, alpha: 1))
        Self.drawText(ctx, "EAST — looking away from the sun", x: CGFloat(x0 + groupW + 2 * gap), y: top - 100, size: 19, bold: true,
                      color: CGColor(red: 0.6, green: 0.75, blue: 1, alpha: 1))
        for g in 0..<2 {
            for (ci, c) in columns.enumerated() {
                let cx = x0 + g * (groupW + 2 * gap) + ci * (W + gap)
                Self.drawText(ctx, c.0 + String(format: "  · ×%.2f", calib[ci]), x: CGFloat(cx), y: top - 126, size: 14.5)
            }
        }
        var notes: [String] = []
        for (ri, el) in rows.enumerated() {
            // Row exposure: the multiple-scattering built-in west view's median luminance → 0.2.
            let key = try cam.render(Self.sheetParams(el: el, ms: true, aerosol: .builtin), yawDeg: 0)
            let lums = key.map { simd_dot($0, Self.luma) * calib[1] }.sorted()
            let exposure = 0.2 / max(lums[lums.count / 2], 1e-12)
            let cy = totalH - header - (ri + 1) * (H + gap) + gap
            Self.drawText(ctx, String(format: "sun %+.0f°", el), x: CGFloat(margin), y: CGFloat(cy + H / 2 + 12), size: 26, bold: true)
            Self.drawText(ctx, String(format: "EV %+.1f", log2(exposure / 20)), x: CGFloat(margin), y: CGFloat(cy + H / 2 - 16), size: 14,
                          color: CGColor(gray: 0.65, alpha: 1))
            for g in 0..<2 {
                for (ci, c) in columns.enumerated() {
                    let img = try cam.render(Self.sheetParams(el: el, ms: c.2, aerosol: c.1), yawDeg: g == 0 ? 0 : 180)
                    guard let cg = Self.image(Self.tonemap(img, exposure: exposure * calib[ci]), W: W, H: H) else { continue }
                    let cx = x0 + g * (groupW + 2 * gap) + ci * (W + gap)
                    ctx.draw(cg, in: CGRect(x: cx, y: cy, width: W, height: H))
                }
            }
            // A number per row: the sunward sky 5° up and 15° up, CIE x,y — built-in MS vs hazy.
            let h = try MS.Harness()
            let dirs = [View.dir(el: 5, az: 0), View.dir(el: 15, az: 0), SIMD3<Float>(0, 1, 0)]
            let b = try h.sky(Self.sky(el: el, ms: true, aerosol: .builtin), dirs)
            let z = try h.sky(Self.sky(el: el, ms: true, aerosol: .hazySuburban), dirs)
            func f(_ c: SIMD3<Float>) -> String { let q = Self.xy(c); return String(format: "(%.3f, %.3f)", q.x, q.y) }
            notes.append(String(format: "sun %+.0f°: sunward 5° up %@ → hazy %@; 15° up %@ → %@", el, f(b[0]), f(z[0]), f(b[1]), f(z[1])))
        }
        Self.drawText(ctx, "CIE x,y of the sky toward the sun, multiple scattering, built-in air → hazy suburban:", x: CGFloat(margin), y: CGFloat(footer - 30), size: 14,
                      color: CGColor(gray: 0.75, alpha: 1))
        Self.drawText(ctx, notes.joined(separator: "   ·   "), x: CGFloat(margin), y: CGFloat(footer - 54), size: 13,
                      color: CGColor(gray: 0.62, alpha: 1))
        Self.drawText(ctx, "VisualizerEngine turbidity branch · Params.atmosphereAerosol = .hazySuburban · rendered with Illuminatorama's in-view sky kernels (illumi_cloud_inview[_ms][_aerosol])",
                      x: CGFloat(margin), y: CGFloat(footer - 80), size: 12, color: CGColor(gray: 0.5, alpha: 1))
        guard let sheet = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: sheetPath) as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw XCTSkip("cannot write \(sheetPath)")
        }
        CGImageDestinationAddImage(dest, sheet, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        print("── comparison sheet: \(sheetPath) (\(totalW)×\(totalH)) ──")

        // A sunset sweep (+10° … −9°), looking west into the sun, multiple scattering: built-in air |
        // rural | hazy suburban — an animated PNG (no palette banding), the exposure adapting like an eye.
        guard let sweepPath = env["VIZ_AEROSOL_SWEEP"] else { return }
        let panels: [(String, NishitaAerosol)] = [("built-in air (τ550 0.028)", .builtin), ("rural (τ550 0.15, H 3 km)", .rural),
                                                   ("hazy suburban (τ550 0.25, H 4 km)", .hazySuburban)]
        let frames = 39
        let gw = panels.count * W + (panels.count - 1) * gap + 2 * margin, gh = H + 64
        guard let apng = CGImageDestinationCreateWithURL(URL(fileURLWithPath: sweepPath) as CFURL, UTType.png.identifier as CFString, frames, nil) else { return }
        CGImageDestinationSetProperties(apng, [kCGImagePropertyPNGDictionary as String: [kCGImagePropertyAPNGLoopCount as String: 0]] as CFDictionary)
        var exposure: Float = 0
        var sweepCalib: [Float] = []
        for pnl in panels { sweepCalib.append(ref / simd_dot(try noon.ePi(pnl.1, ms: true, intensity: 1).0, Self.luma)) }
        for f in 0..<frames {
            let el = 10 - Float(f) * 0.5
            guard let fctx = CGContext(data: nil, width: gw, height: gh, bitsPerComponent: 8, bytesPerRow: gw * 4,
                                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { break }
            fctx.setFillColor(CGColor(red: 0.075, green: 0.078, blue: 0.09, alpha: 1))
            fctx.fill(CGRect(x: 0, y: 0, width: gw, height: gh))
            var imgs: [[SIMD3<Float>]] = []
            for pnl in panels { imgs.append(try cam.render(Self.sheetParams(el: el, ms: true, aerosol: pnl.1), yawDeg: 0)) }
            let lums = imgs[0].map { simd_dot($0, Self.luma) * sweepCalib[0] }.sorted()
            let target = 0.2 / max(lums[lums.count / 2], 1e-12)
            exposure = exposure == 0 ? target : exposure * pow(target / exposure, 0.6)
            for (i, img) in imgs.enumerated() {
                if let cg = Self.image(Self.tonemap(img, exposure: exposure * sweepCalib[i]), W: W, H: H) {
                    fctx.draw(cg, in: CGRect(x: margin + i * (W + gap), y: 10, width: W, height: H))
                }
                Self.drawText(fctx, panels[i].0, x: CGFloat(margin + i * (W + gap)), y: CGFloat(gh - 44), size: 15,
                              color: i == 0 ? CGColor(gray: 0.85, alpha: 1) : CGColor(red: 1, green: 0.82, blue: 0.55, alpha: 1))
            }
            Self.drawText(fctx, String(format: "sun %+.1f°  ·  looking west, multiple scattering", el), x: CGFloat(margin), y: CGFloat(gh - 22), size: 15, bold: true)
            if let frame = fctx.makeImage() {
                CGImageDestinationAddImage(apng, frame, [kCGImagePropertyPNGDictionary as String: [kCGImagePropertyAPNGDelayTime as String: f == 0 || f == frames - 1 ? 0.8 : 0.12]] as CFDictionary)
            }
        }
        XCTAssertTrue(CGImageDestinationFinalize(apng))
        print("── sunset sweep: \(sweepPath) ──")
    }

    // MARK: - 9. GPU cost

    /// The aerosol medium's LUT build (the Mie shape tables add a phase convolution per order and a
    /// 4-D lookup per re-scatter sample) and its march, against the built-in ones — a build runs
    /// once per aerosol change, never per frame.
    @MainActor
    func testGPUCostOfTheAerosolPath() throws {
        let engine = SimEngine.shared
        let cloud = VolumetricCloudRenderer(engine: engine, resolution: SIMD2(64, 32), iblResolution: SIMD2(32, 16))
        func build(_ aerosol: NishitaAerosol) throws -> Double {
            var t: [Double] = []
            for _ in 0..<4 {
                guard let cb = engine.commandQueue.makeCommandBuffer() else { throw XCTSkip("no cb") }
                let u = aerosol.uniforms
                XCTAssertTrue(cloud.encodeAtmosphereLUTBuild(into: cb, albedo: 0.1, aerosol: aerosol == .builtin ? nil : (u.a, u.b)))
                cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness timing
                if let e = cb.error { throw e }
                t.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
            }
            return t.sorted()[t.count / 2]
        }
        let buildBuiltin = try build(.builtin), buildHazy = try build(.hazySuburban)
        // The march: the dome kernel at 2048 × 1024 with the cloud march minimised (the LUT in the
        // renderer's own buffer is the hazy one just built).
        let W = 2048, H = 1024
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: W, height: H, mipmapped: false)
        td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .private
        guard let tex = engine.device.makeTexture(descriptor: td),
              let ub = engine.device.makeBuffer(length: VolumetricCloudRenderer.skyUniformsBufferLength, options: .storageModeShared)
        else { throw XCTSkip("no resources") }
        func time(_ p: VolumetricCloudRenderer.Params, _ kernel: String) throws -> Double {
            guard let pso = engine.pipeline(kernel) else { throw XCTSkip("\(kernel) missing") }
            let ms = p.atmosphereMultipleScattering
            var u = SkyUniforms(params: p, time: 0)
            memcpy(ub.contents(), &u, MemoryLayout<SkyUniforms>.stride)
            var t: [Double] = []
            for _ in 0..<5 {
                guard let cb = engine.commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw XCTSkip("no cb") }
                enc.setComputePipelineState(pso)
                enc.setTexture(tex, index: 0); enc.setTexture(cloud.noiseTexture, index: 1)
                enc.setBuffer(ub, offset: 0, index: 0); enc.setBuffer(cloud.fallbackBurstLightBuffer, offset: 0, index: 1)
                if ms { enc.setBuffer(ub, offset: VolumetricCloudRenderer.atmosphereLUTOffset, index: 2) }
                enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness timing
                if let e = cb.error { throw e }
                t.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
            }
            return t.sorted()[t.count / 2]
        }
        func cheap(_ el: Float, ms: Bool, _ a: NishitaAerosol) -> VolumetricCloudRenderer.Params {
            var p = Self.sky(el: el, ms: ms, aerosol: a)
            p.coverage = 0; p.cloudSteps = 1; p.lightSteps = 1; p.cirrusCoverage = 0
            return p
        }
        var report = "── Aerosol GPU cost ──\n"
        report += String(format: "LUT build (once per change): built-in %.2f ms, hazy suburban %.2f ms\n", buildBuiltin, buildHazy)
        var worstMS: Double = 0
        for el: Float in [40, -3] {
            // Built-in LUT for the built-in march, the hazy LUT for the aerosol one.
            _ = try build(.builtin)
            memcpy(ub.contents() + VolumetricCloudRenderer.atmosphereLUTOffset,
                   cloud.skyUniformsBuffer.contents() + VolumetricCloudRenderer.atmosphereLUTOffset,
                   VolumetricCloudRenderer.skyUniformsBufferLength - VolumetricCloudRenderer.atmosphereLUTOffset)
            let ssB = try time(cheap(el, ms: false, .builtin), "volSkyRender")
            let msB = try time(cheap(el, ms: true, .builtin), "volSkyRenderMS")
            _ = try build(.hazySuburban)
            memcpy(ub.contents() + VolumetricCloudRenderer.atmosphereLUTOffset,
                   cloud.skyUniformsBuffer.contents() + VolumetricCloudRenderer.atmosphereLUTOffset,
                   VolumetricCloudRenderer.skyUniformsBufferLength - VolumetricCloudRenderer.atmosphereLUTOffset)
            let ssA = try time(cheap(el, ms: false, .hazySuburban), "volSkyRenderAerosol")
            let msA = try time(cheap(el, ms: true, .hazySuburban), "volSkyRenderMSAerosol")
            worstMS = max(worstMS, msA / msB)
            report += String(format: "2048×1024 march, sun %+.0f°: single scatter built-in %.2f / aerosol %.2f ms; multiple scattering built-in %.2f / aerosol %.2f ms (×%.2f)\n",
                             el, ssB, ssA, msB, msA, msA / msB)
        }
        print(report)
        // Fences, not specs (other sessions' GPU work on a shared machine inflates single samples).
        XCTAssertLessThan(buildHazy, 100, "the one-time aerosol build stays within a few frames of GPU")
        XCTAssertLessThan(worstMS, 2.0, "the aerosol march (a 4-D table lookup per sample in the haze) stays within 2× the built-in one")
    }
}
