import Foundation
import OSLog
import VisualizerCore

// ── ILLUMINATORAMA INIT-PHASE TIMING ─────────────────────────────────────────
//
// `IlluminatoramaRenderer.init` is a single ~900-line blocking call that a host
// pays once per process, on the main thread, before the first frame — in Daydream
// Home it was 2.5–2.6 s of a 2.8 s cold launch. A single wall-clock number around
// the whole thing can't say WHICH of the twenty-odd things it does (pipeline
// builds, LUT bakes, cubemaps, shadow maps, atlases, render targets, buffer rings)
// owns the time, and guessing wastes rounds: the first guess here — shader pipeline
// compilation — was disproved only after a full `MTLBinaryArchive` was built and
// A/B'd (61/61 pipelines served from disk moved launch by 0.4 %).
//
// So init carries its own stopwatch. Marks are recorded unconditionally (twenty
// `DispatchTime.now()` reads, well under a microsecond total) and exposed on the
// renderer as `initTiming`, so a host can log the breakdown into its own launch
// timeline without a special build. Setting `VIZ_ILLUMI_INIT_TIMING` additionally
// emits `os_signpost` intervals (for Instruments) and mirrors the table to the
// unified log — and, if the variable's value is a path, appends it to that file,
// the same sidecar channel `VIZ_ILLUMI_LTC_LOG` uses to get diagnostics out of a
// sandboxed bundle whose stderr is disconnected.
public struct IlluminatoramaInitTiming: Sendable {

    /// One measured span of `init`, in the order the marks were taken.
    public struct Phase: Sendable {
        public let label: String
        public let milliseconds: Double
    }

    public private(set) var phases: [Phase] = []

    /// Total measured init time (ms) — the sum of every phase.
    public var totalMilliseconds: Double { phases.reduce(0) { $0 + $1.milliseconds } }

    /// The single most expensive phase, or nil if nothing was measured.
    public var dominantPhase: Phase? { phases.max { $0.milliseconds < $1.milliseconds } }

    private var lastMark: UInt64
    private let signposter: OSSignposter?
    private let sidecarPath: String?

    private static let log = Logger(subsystem: AppLog.subsystem, category: "illuminatoramaInit")

    public init() {
        let env = ProcessInfo.processInfo.environment["VIZ_ILLUMI_INIT_TIMING"]
        let enabled = (env != nil && env != "0")
        signposter = enabled
            ? OSSignposter(subsystem: AppLog.subsystem, category: "illuminatoramaInit") : nil
        // A value that looks like a path doubles as the sidecar destination; "1" just
        // turns on signposts + unified logging.
        sidecarPath = (env?.hasPrefix("/") == true) ? env : nil
        lastMark = DispatchTime.now().uptimeNanoseconds
    }

    /// Close the span that started at the previous mark (or at `init`) and label it.
    public mutating func mark(_ label: String) {
        let now = DispatchTime.now().uptimeNanoseconds
        let ms = Double(now &- lastMark) / 1_000_000
        lastMark = now
        phases.append(Phase(label: label, milliseconds: ms))
        if let signposter {
            // Emit as a zero-length event rather than a begin/end pair: the marks are
            // strictly sequential, so the timeline of events already gives Instruments
            // the spans, and an event can't leak an unclosed interval on a throw.
            signposter.emitEvent("initPhase", "\(label, privacy: .public) \(ms, privacy: .public) ms")
            Self.log.notice("init \(label, privacy: .public) — \(ms, privacy: .public) ms")
        }
    }

    /// Aligned one-phase-per-line table, slowest phase last-word tagged. For a host
    /// log or the sidecar file.
    public var summary: String {
        let width = phases.map(\.label.count).max() ?? 0
        let slowest = dominantPhase?.milliseconds ?? 0
        return phases.map { p in
            String(format: "  %@%@ %8.1f ms%@",
                   p.label, String(repeating: " ", count: width - p.label.count),
                   p.milliseconds,
                   p.milliseconds == slowest ? "   ← dominant" : "")
        }.joined(separator: "\n")
    }

    /// Called once at the end of `init`. Writes the sidecar file if one was named.
    public func finish() {
        guard let sidecarPath else { return }
        let text = String(format: "IlluminatoramaRenderer.init — %.1f ms total\n", totalMilliseconds)
            + summary + "\n"
        if let h = FileHandle(forWritingAtPath: sidecarPath) {
            defer { try? h.close() }
            _ = try? h.seekToEnd()
            try? h.write(contentsOf: Data(text.utf8))
        } else {
            try? text.write(toFile: sidecarPath, atomically: true, encoding: .utf8)
        }
    }
}
