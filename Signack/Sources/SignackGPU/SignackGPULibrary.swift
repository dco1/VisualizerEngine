import Foundation
import Metal

/// The compute pipelines of the S2 cut pass, the S4 software rasterizer and the S6 instance batch,
/// built once per device.
///
/// **How the shader gets compiled.** Same mechanism as the engine (`SimPipelineCache`): the
/// `.metal` file is a `.process` resource of this target, so in an Xcode build (the app, its test
/// bundle) it is compiled into this module's `default.metallib` and loaded with
/// `makeDefaultLibrary(bundle: .module)`. A bare `swift build` / `swift test` does NOT compile
/// Metal — it copies the source into the bundle — so when no metallib is there the source is
/// compiled at run time instead (safe math). That is what lets `SignackGPUTests` run the real kernel
/// under `swift test`, with no app and no Xcode.
public final class SignackGPUPipelines: @unchecked Sendable {
    public let device: MTLDevice
    let classify: MTLComputePipelineState
    let write: MTLComputePipelineState
    let finalize: MTLComputePipelineState
    let viewsPrefix: MTLComputePipelineState   // DH-1465
    let rasterPrepare: MTLComputePipelineState
    let rasterDepth: MTLComputePipelineState
    let rasterIDs: MTLComputePipelineState
    let rasterClear: MTLComputePipelineState
    let instancesClassify: MTLComputePipelineState
    let instancesFinalize: MTLComputePipelineState
    /// "metallib" or "source" — which way the library was obtained (logged by hosts/tests).
    public let librarySource: String

    nonisolated(unsafe) private static var cache: [ObjectIdentifier: SignackGPUPipelines] = [:]
    private static let lock = NSLock()

    /// The pipelines for `device`, built on first use. nil if the library or a kernel is missing.
    public static func shared(device: MTLDevice) -> SignackGPUPipelines? {
        lock.lock(); defer { lock.unlock() }
        let key = ObjectIdentifier(device)
        if let p = cache[key] { return p }
        guard let p = SignackGPUPipelines(device: device) else { return nil }
        cache[key] = p
        return p
    }

    private init?(device: MTLDevice) {
        guard let (lib, source) = Self.library(device) else { return nil }
        func pso(_ name: String) -> MTLComputePipelineState? {
            guard let fn = lib.makeFunction(name: name) else { return nil }
            return try? device.makeComputePipelineState(function: fn)
        }
        guard let c = pso("signack_cut_classify"), let w = pso("signack_cut_write"),
              let f = pso("signack_cut_finalize"), let vp = pso("signack_cut_views_prefix"), let rp = pso("signack_raster_prepare"),
              let rd = pso("signack_raster_depth"), let ri = pso("signack_raster_ids"),
              let rc = pso("signack_raster_clear"),
              let ic = pso("signack_instances_classify"), let inf = pso("signack_instances_finalize") else { return nil }
        self.device = device
        classify = c; write = w; finalize = f; viewsPrefix = vp
        rasterPrepare = rp; rasterDepth = rd; rasterIDs = ri; rasterClear = rc
        instancesClassify = ic; instancesFinalize = inf
        librarySource = source
    }

    private static func library(_ device: MTLDevice) -> (MTLLibrary, String)? {
        if let lib = try? device.makeDefaultLibrary(bundle: .module),
           lib.functionNames.contains("signack_cut_classify"), lib.functionNames.contains("signack_raster_depth"),
           lib.functionNames.contains("signack_instances_classify"),
           lib.functionNames.contains("signack_cut_views_prefix") {
            return (lib, "metallib")
        }
        // Each .metal is self-contained (no includes), so the sources concatenate into one library.
        var src = ""
        for name in ["SignackCut", "SignackRaster", "SignackInstances"] {
            let url = Bundle.module.url(forResource: name, withExtension: "metal")
                ?? Bundle.module.url(forResource: name, withExtension: "metal", subdirectory: "Shaders")
            guard let url, let s = try? String(contentsOf: url, encoding: .utf8) else { return nil }
            src += s + "\n"
        }
        let opts = MTLCompileOptions()
        if #available(macOS 15.0, iOS 18.0, tvOS 18.0, *) { opts.mathMode = .safe } else { opts.fastMathEnabled = false }
        guard let lib = try? device.makeLibrary(source: src, options: opts) else { return nil }
        return (lib, "source")
    }
}
