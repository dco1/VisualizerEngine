import Darwin
import Metal
import OSLog

public enum GPUCapabilities {
    private static let log = Logger(subsystem: AppLog.subsystem, category: "gpuCaps")

    public static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    public static let supportsRaytracing: Bool = {
        guard let d = device else { return false }
        return d.supportsRaytracing
    }()

    public static let supportsMetal3: Bool = {
        guard let d = device else { return false }
        return d.supportsFamily(.metal3)
    }()

    public static let isAppleSilicon: Bool = {
        guard let d = device else { return false }
        return d.supportsFamily(.apple1)
    }()

    public static let isRunningUnderRosetta: Bool = {
        var translated: Int32 = 0
        var size = MemoryLayout<Int32>.size
        let rc = sysctlbyname("sysctl.proc_translated", &translated, &size, nil, 0)
        return rc == 0 && translated == 1
    }()

    public static func summary() -> String {
        guard let d = device else { return "gpu: no Metal device" }
        return """
            gpu: \(d.name) | \
            raytracing=\(supportsRaytracing) | \
            metal3=\(supportsMetal3) | \
            apple-silicon=\(isAppleSilicon) | \
            rosetta=\(isRunningUnderRosetta)
            """
    }

    public static func logSummary() {
        log.notice("\(summary())")
        if isRunningUnderRosetta {
            log.warning("""
                running under Rosetta translation — Metal applies \
                behaviour-equivalence workarounds at a perf cost. \
                Build for arm64 for full performance.
                """)
        }
    }
}
