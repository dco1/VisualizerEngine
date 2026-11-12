import Foundation
import Darwin
import os

/// A black-box recorder: once a second it appends one line of vitals to its own file, from a
/// thread that is NOT the main thread, straight through `write(2)` + `fsync` — so after a hang,
/// a force-quit, a jetsam kill or a hard power-cycle, the last seconds before it are on disk.
///
/// What a post-mortem reads from the file:
/// - `dt=` — how long the recorder's own tick really took. A 1 s sleep that took 9 s means the
///   WHOLE MACHINE was starved (memory thrash, GPU/WindowServer starvation), not just this app.
/// - `mem` / `pressure` / `avail` / `swap` — footprint climbing, kernel pressure, swapping.
/// - `main=STALLED` + `lag` — the main thread stopped answering; a symbolicated stack of it is
///   logged at the start of the stall and every few seconds while it lasts.
/// - `ctx` and probes — what the app said it was doing (open document, bake, export) and any
///   app-supplied numbers (GPU in-flight time, scene size…).
/// - `PREVIOUS SESSION DID NOT END CLEANLY` at the next launch — the file's last line wasn't a
///   `SESSION END`, i.e. a crash, force-quit or freeze.
///
/// App-agnostic: the app passes a log URL and optional probes. Start it ON THE MAIN THREAD.
public final class FlightRecorder: @unchecked Sendable {
    public struct Configuration: Sendable {
        public var url: URL
        public var interval: TimeInterval = 1.0
        /// Main-thread unresponsiveness (s) before a stack sample is logged.
        public var stallThreshold: TimeInterval = 0.5
        /// While a stall continues, re-sample the stack this often (s).
        public var resampleEvery: TimeInterval = 3.0
        /// Rotate to `<name>.1` past this many bytes.
        public var maxBytes: Int = 8 * 1_048_576
        public init(url: URL) { self.url = url }
    }

    private struct State {
        var context = ""
        var probes: [(name: String, read: @Sendable () -> String)] = []
        var pingOutstandingSince: UInt64 = 0        // ns; 0 = main answered the last ping
        var running = false
        var ended = false                           // SESSION END written: nothing may follow it
    }

    private let config: Configuration
    private let state = OSAllocatedUnfairLock(initialState: State())
    private let sampler: StackSampler
    private var fd: Int32 = -1
    private var bytesWritten = 0
    private var thread: Thread?

    /// Process-wide instance, set by `start`.
    public nonisolated(unsafe) private(set) static var shared: FlightRecorder?

    private init(_ config: Configuration) {
        self.config = config
        self.sampler = StackSampler()            // captures the calling (main) thread
    }

    // MARK: Public API

    /// Begin recording. Must be called on the main thread (it is the thread that gets sampled).
    @discardableResult
    public static func start(_ config: Configuration) -> FlightRecorder {
        precondition(Thread.isMainThread, "FlightRecorder.start must run on the main thread")
        if let s = shared { return s }
        let r = FlightRecorder(config)
        shared = r
        r.open()
        r.state.withLock { $0.running = true }
        let t = Thread { r.loop() }
        t.name = "FlightRecorder"
        t.qualityOfService = .userInteractive
        r.thread = t
        t.start()
        return r
    }

    /// What the app is doing right now ("opening Kitchen.daydreamhome", "export High"). Cheap.
    public func setContext(_ text: String) { state.withLock { $0.context = text } }

    /// A named value read on the recorder thread each tick. `read` must be thread-safe and fast
    /// (read an atomic / lock-protected counter — never touch main-actor state).
    public func addProbe(_ name: String, _ read: @escaping @Sendable () -> String) {
        state.withLock { $0.probes.append((name, read)) }
    }

    /// Free-form line, stamped, flushed immediately (from any thread).
    public func note(_ text: String) { write("note \(text)") }

    /// Mark an orderly exit. Without this line the next launch reports an unclean end.
    public func stop() {
        state.withLock { $0.running = false }
        write("SESSION END", final: true)
    }

    // MARK: File

    private func open() {
        let dir = config.url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let previousClean = Self.endedCleanly(config.url)
        rotateIfNeeded(force: false)
        fd = Darwin.open(config.url.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        let pid = ProcessInfo.processInfo.processIdentifier
        let cores = ProcessInfo.processInfo.activeProcessorCount
        let ram = ProcessInfo.processInfo.physicalMemory / 1_073_741_824
        if previousClean == false {
            write("!! PREVIOUS SESSION DID NOT END CLEANLY — crash, force-quit, jetsam or hard freeze")
        }
        write("SESSION START pid=\(pid) cores=\(cores) ram=\(ram)GB interval=\(config.interval)s")
    }

    /// nil = no previous file / empty.
    static func endedCleanly(_ url: URL) -> Bool? {
        guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? h.close() }
        let size = (try? h.seekToEnd()) ?? 0
        guard size > 0 else { return nil }
        try? h.seek(toOffset: size > 512 ? size - 512 : 0)
        let tail = String(decoding: (try? h.readToEnd()) ?? Data(), as: UTF8.self)
        guard let last = tail.split(separator: "\n").last else { return nil }
        return last.contains("SESSION END")
    }

    private func rotateIfNeeded(force: Bool) {
        let size = (try? FileManager.default.attributesOfItem(atPath: config.url.path)[.size] as? Int) ?? 0
        guard force || size > config.maxBytes else { return }
        let old = config.url.appendingPathExtension("1")
        try? FileManager.default.removeItem(at: old)
        try? FileManager.default.moveItem(at: config.url, to: old)
    }

    private func write(_ line: String, final: Bool = false) {
        let stamped = Self.stamp() + " " + line + "\n"
        state.withLock { st in                     // serialises the heartbeat thread vs note()/stop()
            guard fd >= 0, !st.ended else { return }
            if final { st.ended = true }
            var s = stamped
            s.withUTF8 { _ = Darwin.write(fd, $0.baseAddress, $0.count) }
            fsync(fd)
            bytesWritten += stamped.utf8.count
            if bytesWritten > config.maxBytes {
                Darwin.close(fd)
                rotateIfNeeded(force: true)
                fd = Darwin.open(config.url.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
                bytesWritten = 0
            }
        }
    }

    private static func stamp() -> String {
        var tv = timeval(); gettimeofday(&tv, nil)
        var t = time_t(tv.tv_sec); var tmv = tm(); localtime_r(&t, &tmv)
        return String(format: "%04d-%02d-%02d %02d:%02d:%02d.%03d", tmv.tm_year + 1900, tmv.tm_mon + 1,
                      tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec, Int(tv.tv_usec) / 1000)
    }

    // MARK: Loop

    private func loop() {
        var lastTick = DispatchTime.now().uptimeNanoseconds
        var lastCPU = FlightVitals.cpuSeconds()
        var stallStart: UInt64 = 0
        var lastSample: UInt64 = 0
        while state.withLock({ $0.running }) {
            Thread.sleep(forTimeInterval: config.interval)
            let now = DispatchTime.now().uptimeNanoseconds
            let dt = Double(now &- lastTick) / 1e9
            lastTick = now

            // Ping main; "lag" is how long the oldest unanswered ping has waited.
            let since: UInt64 = state.withLock { st in
                if st.pingOutstandingSince == 0 { st.pingOutstandingSince = now; return 0 }
                return st.pingOutstandingSince
            }
            let postedNow = since == 0
            if postedNow {
                DispatchQueue.main.async { [weak self] in
                    self?.state.withLock { $0.pingOutstandingSince = 0 }
                }
            }
            let lag = since == 0 ? 0 : Double(now &- since) / 1e9
            let stalled = lag >= config.stallThreshold

            let cpu = FlightVitals.cpuSeconds()
            let cpuPct = Int(((cpu - lastCPU) / max(dt, 0.001)) * 100)
            lastCPU = cpu

            let (ctx, probes) = state.withLock { ($0.context, $0.probes) }
            let probeText = probes.map { "\($0.name)=\($0.read())" }.joined(separator: " ")
            let mb = FlightVitals.footprintBytes() / 1_048_576
            var line = "hb dt=\(String(format: "%.2f", dt))s"
            if dt > config.interval * 3 { line += " ⚠︎MACHINE-STALL(recorder itself was starved)" }
            line += " mem=\(mb)MB peak=\(FlightVitals.peakFootprintBytes() / 1_048_576)MB"
            line += " pressure=\(FlightVitals.pressure()) avail=\(FlightVitals.availablePercent())%"
            line += " swap=\(FlightVitals.swapUsedMB())MB cpu=\(cpuPct)% thermal=\(FlightVitals.thermal())"
            line += stalled ? " main=STALLED lag=\(String(format: "%.1f", lag))s" : " main=ok"
            if !ctx.isEmpty { line += " ctx=\"\(ctx)\"" }
            if !probeText.isEmpty { line += " " + probeText }
            write(line)

            if stalled {
                if stallStart == 0 { stallStart = now; lastSample = 0 }
                if lastSample == 0 || Double(now &- lastSample) / 1e9 >= config.resampleEvery {
                    lastSample = now
                    let frames = StackSampler.symbolicate(sampler.capture())
                    write("main-stack (stalled \(String(format: "%.1f", lag))s):\n    "
                          + (frames.isEmpty ? "(unavailable)" : frames.joined(separator: "\n    ")))
                }
            } else if stallStart != 0 {
                write("main recovered after \(String(format: "%.1f", Double(now &- stallStart) / 1e9))s stall")
                stallStart = 0
            }
        }
    }
}

/// Test seam for the file-level clean-exit check.
enum FlightRecorderProbe {
    static func endedCleanly(_ url: URL) -> Bool? { FlightRecorder.endedCleanly(url) }
}
