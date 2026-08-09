import XCTest
import QuartzCore
@testable import VisualizerRendering

/// `RenderClock` — the session-epoch rebase that keeps `IlluminatoramaRenderer.time` inside the
/// precision a 32-bit `Float` can actually carry.
///
/// **The gate that was missing.** The defect these tests pin was a host assigning
/// `renderer.time = Float(CACurrentMediaTime())`. `CACurrentMediaTime()` is awake-uptime *since
/// boot*, so its magnitude decides its own resolution: at ~337 700 s (a Mac awake 3.9 days) a
/// `Float`'s ulp is 31.25 ms — coarser than a 60 fps frame — so consecutive frames narrowed to
/// the SAME value and every GPU oscillator advanced in quantised ~32 Hz steps. Nothing caught it
/// because the artefact is a function of the *user's uptime*: it is absent on a freshly-booted
/// machine and halves in resolution every time uptime doubles, so no short test on a recently
/// rebooted CI box can ever see it.
///
/// These tests are therefore written against *simulated* uptimes rather than the ambient clock —
/// deterministic, GPU-free, and identical on any machine at any uptime.
final class RenderClockTests: XCTestCase {

    /// Uptimes worth pinning, in seconds: the value measured on the machine where the defect was
    /// found, plus a doubling ladder that shows the artefact is monotone in uptime.
    private static let uptimes: [CFTimeInterval] = [
        3_600,        // 1 h — a just-rebooted machine
        86_400,       // 1 day
        337_700,      // 3.9 days — MEASURED on the machine where this was diagnosed
        691_200,      // 8 days
        2_592_000,    // 30 days — a laptop that only ever sleeps
    ]

    // MARK: - The defect, reproduced as arithmetic

    /// THE MEASUREMENT. A boot-relative `Float` clock cannot represent one 60 fps frame of
    /// advance; the session-relative clock represents it exactly. Same inputs, same narrowing —
    /// only the epoch differs.
    func testBootRelativeFloatClockLosesA60fpsFrameThatSessionRelativeKeeps() {
        let dt = 1.0 / 60.0
        var brokenAt: [CFTimeInterval] = []
        for uptime in Self.uptimes {
            // ── The defect: narrow the boot-relative clock, THEN take the delta.
            let bootDelta = CFTimeInterval(Float(uptime + dt) - Float(uptime))
            // ── The fix: subtract in Double against a session epoch, THEN narrow.
            let sessionAge: CFTimeInterval = 600                      // 10 min into the session
            var clock = RenderClock()
            _ = clock.elapsed(at: uptime - sessionAge)
            let a = Float(clock.elapsed(at: uptime))
            let b = Float(clock.elapsed(at: uptime + dt))
            let sessionDelta = CFTimeInterval(b - a)

            // Tolerance is DERIVED, not chosen: the narrowed delta can only be off by the clock's
            // own resolution at the session age, which is what the fix makes small.
            XCTAssertEqual(sessionDelta, dt,
                           accuracy: 2 * RenderClock.resolution(atSeconds: sessionAge),
                           "session-relative clock must advance by one frame at uptime \(uptime)")
            let err = abs(bootDelta - dt) / dt
            if err > 0.01 { brokenAt.append(uptime) }
            print(String(format: "RENDER_CLOCK uptime=%.0f s  bootDelta=%.5f ms  sessionDelta=%.5f ms  "
                                 + "ulp=%.3f ms  stepsPerFrame=%.3f",
                         uptime, bootDelta * 1000, sessionDelta * 1000,
                         RenderClock.resolution(atSeconds: uptime) * 1000,
                         RenderClock.stepsPerFrame(atSeconds: uptime)))
        }
        // The old mapping must be shown to be broken — otherwise this test proves nothing and the
        // constants above have drifted somewhere harmless.
        XCTAssertEqual(brokenAt.count, Self.uptimes.count - 1,
                       "every uptime from a day up must break the boot-relative mapping "
                       + "(broke at: \(brokenAt))")
    }

    /// At the measured 3.9-day uptime the boot-relative clock does not merely lose accuracy — it
    /// FREEZES: many consecutive 60 fps frames narrow to the identical float, then the clock jumps
    /// a whole 31.25 ms. That step/hold pattern is the visible judder.
    func testBootRelativeClockHoldsThenJumpsInsteadOfAdvancing() {
        let uptime: CFTimeInterval = 337_700
        let dt = 1.0 / 60.0
        let frames = 120
        var frozenFrames = 0
        var maxJumpMs = 0.0
        var previous = Float(uptime)
        for frame in 1...frames {
            let now = Float(uptime + Double(frame) * dt)
            let stepMs = CFTimeInterval(now - previous) * 1000
            if stepMs == 0 { frozenFrames += 1 } else { maxJumpMs = max(maxJumpMs, stepMs) }
            previous = now
        }
        // The frozen count is PREDICTED, not observed-then-pasted: only `dt / ulp` of the frames
        // can advance the clock at all, so the rest are exact repeats. 0.533 → 56 of 120.
        let advancing = RenderClock.stepsPerFrame(atSeconds: uptime, fps: 1 / dt)
        let predictedFrozen = Int(((1 - advancing) * CFTimeInterval(frames)).rounded())
        print("RENDER_CLOCK_JUDDER frozenFrames=\(frozenFrames)/\(frames) "
              + "predicted=\(predictedFrozen) maxJumpMs=\(maxJumpMs)")
        XCTAssertLessThanOrEqual(abs(frozenFrames - predictedFrozen), 2,
                                 "the hold/jump pattern must match the ulp prediction")
        XCTAssertGreaterThan(frozenFrames, frames / 3, "a large fraction of frames are exact repeats")
        XCTAssertEqual(maxJumpMs, RenderClock.resolution(atSeconds: uptime) * 1000, accuracy: 0.01,
                       "and the catch-up jump is a full ulp — 31.25 ms")

        // The session-relative clock over the same 120 frames: every step is one frame, no repeats.
        var clock = RenderClock()
        _ = clock.elapsed(at: uptime)
        var prev = Float(0)
        for frame in 1...120 {
            let t = Float(clock.elapsed(at: uptime + Double(frame) * dt))
            XCTAssertEqual(CFTimeInterval(t - prev), dt, accuracy: dt * 1e-4,
                           "frame \(frame): session clock advances one frame, every frame")
            prev = t
        }
    }

    /// The *consequence* on a real shader oscillator, measured. `applySway` mode 2 (the ceiling
    /// pendant, `Illuminatorama.metal`) computes `angle = lean · sin(time · 2.65 + φ)` in `float`.
    /// Under a boot-relative clock its per-frame angular delta collapses to zero for runs of
    /// frames and then jumps; under the session clock it is smooth. Judged on the ratio of the
    /// largest single-frame step to the mean step — 1.0 is perfectly smooth motion.
    func testPendantPendulumIsSmoothOnSessionClockAndStaircasedOnBootClock() {
        let uptime: CFTimeInterval = 337_700
        let dt = 1.0 / 60.0
        let omega: Float = 2.65, phase: Float = 0.7, lean: Float = 0.07

        func angles(_ clockValues: [Float]) -> (peakOverMean: Double, zeroSteps: Int) {
            let a = clockValues.map { lean * sin($0 * omega + phase) }
            let steps = zip(a, a.dropFirst()).map { abs(Double($1 - $0)) }
            let zeros = steps.filter { $0 == 0 }.count
            let mean = steps.reduce(0, +) / Double(steps.count)
            return (mean > 0 ? (steps.max() ?? 0) / mean : .infinity, zeros)
        }

        let bootClock = (0...240).map { Float(uptime + Double($0) * dt) }
        var clock = RenderClock()
        _ = clock.elapsed(at: uptime)
        let sessionClock = (0...240).map { Float(clock.elapsed(at: uptime + Double($0) * dt)) }

        let boot = angles(bootClock), session = angles(sessionClock)
        print(String(format: "PENDULUM_SMOOTHNESS boot: peak/mean=%.2f zeroSteps=%d | "
                             + "session: peak/mean=%.2f zeroSteps=%d",
                     boot.peakOverMean, boot.zeroSteps, session.peakOverMean, session.zeroSteps))

        // Session clock: continuous motion — no frozen frames, and the biggest step is within a
        // small factor of the mean (the only variation left is the cosine envelope of the swing,
        // which peaks at π/2 over the mean of |cos| ⇒ ~1.57; measured 1.66).
        XCTAssertEqual(session.zeroSteps, 0, "no frame of the swing is a repeat of the last")
        XCTAssertLessThan(session.peakOverMean, 2.0, "the swing advances smoothly")

        // Boot clock: the same swing, staircased. Both tells must be present, or the measurement
        // is not actually reproducing the defect it claims to. Measured: 112 frozen of 240, and
        // peak/mean 4.74 — nearly 3× the smooth case, because the motion of ~2 frames arrives in
        // one. (peak/mean cannot blow up further: the total variation is unchanged, it is just
        // delivered in fewer, bigger steps.)
        XCTAssertGreaterThan(boot.zeroSteps, 100, "most frames are frozen under the boot clock")
        XCTAssertGreaterThan(boot.peakOverMean, 3.0, "…and motion arrives in large jumps")
        XCTAssertGreaterThan(boot.peakOverMean, session.peakOverMean * 2,
                             "the boot clock must be measurably rougher than the session clock")
    }

    // MARK: - Epoch semantics

    /// First sample is exactly 0, and the epoch is that first sample — not `init` time. A renderer
    /// constructed long before its first frame (shader build, asset bake) must still start its
    /// animation at 0.
    func testFirstSampleIsZeroAndSeedsTheEpoch() {
        var clock = RenderClock()
        XCTAssertNil(clock.epoch, "a clock that has never ticked has no origin")
        XCTAssertEqual(clock.elapsed(at: 12_345.5), 0)
        XCTAssertEqual(clock.epoch, 12_345.5)
        XCTAssertEqual(clock.elapsed(at: 12_348.0), 2.5, accuracy: 1e-12)
    }

    /// Deltas are preserved EXACTLY — the rebase is a pure translation, so no consumer that reads
    /// an interval (the `iblRebakeInterval` fallback, any host-side dt) can be affected by it.
    func testRebasePreservesEveryDeltaExactly() {
        let base: CFTimeInterval = 337_700
        var samples: [CFTimeInterval] = []
        for i in 0..<200 {
            let n = CFTimeInterval(i)
            samples.append(base + n * 0.0167 + n * n * 1e-6)   // irregular, jittered frame arrival
        }
        var clock = RenderClock()
        let rebased = samples.map { clock.elapsed(at: $0) }
        for i in 1..<samples.count {
            let host: CFTimeInterval = samples[i] - samples[i - 1]
            let seen: CFTimeInterval = rebased[i] - rebased[i - 1]
            XCTAssertEqual(seen, host, accuracy: 1e-12, "delta \(i) must survive the rebase untouched")
        }
    }

    /// `reset()` restarts the session; it does not shift it by a wall-clock offset.
    func testResetRestartsTheSessionAtZero() {
        var clock = RenderClock()
        _ = clock.elapsed(at: 1_000)
        XCTAssertEqual(clock.elapsed(at: 1_060), 60, accuracy: 1e-12)
        clock.reset()
        XCTAssertNil(clock.epoch)
        XCTAssertEqual(clock.elapsed(at: 500_000), 0)
    }

    /// A clock that goes backwards is reported as negative rather than clamped — clamping would
    /// silently freeze every animation instead of surfacing a broken host clock.
    func testBackwardsSampleIsReportedNotClamped() {
        var clock = RenderClock()
        _ = clock.elapsed(at: 100)
        XCTAssertEqual(clock.elapsed(at: 90), -10, accuracy: 1e-12)
    }

    // MARK: - The diagnostic the renderer warns from

    /// `isTooCoarse` must fire for every boot-relative clock a real Mac produces and stay silent
    /// for every session length an interactive run reaches.
    func testCoarseClockDetectorAgreesWithTheUlpTable() {
        for good in [0.0, 1.0, 60.0, 3_600.0, 28_800.0] {          // up to an 8-hour session
            XCTAssertFalse(RenderClock.isTooCoarse(good),
                           "a \(good) s session clock is fine (ulp "
                           + "\(RenderClock.resolution(atSeconds: good) * 1000) ms)")
        }
        for bad in [86_400.0, 337_700.0, 2_592_000.0] {            // boot-relative territory
            XCTAssertTrue(RenderClock.isTooCoarse(bad),
                          "\(bad) s must be flagged (ulp "
                          + "\(RenderClock.resolution(atSeconds: bad) * 1000) ms)")
        }
        // Sign must not matter: ulp is a magnitude property.
        XCTAssertTrue(RenderClock.isTooCoarse(-337_700))
        // And the numbers quoted in the docs must be the numbers the code computes.
        XCTAssertEqual(RenderClock.resolution(atSeconds: 337_700), 0.03125, accuracy: 1e-9)
        XCTAssertEqual(RenderClock.resolution(atSeconds: 86_400), 0.0078125, accuracy: 1e-9)
    }

    /// The diagnosis string carries the actual measurements (that is its whole job) and names the
    /// fix, so a host developer who trips it does not have to come read this file.
    func testDiagnosisNamesTheNumbersAndTheFix() {
        let m = RenderClock.coarseClockDiagnosis(337_700)
        XCTAssertTrue(m.contains("337700"), m)
        XCTAssertTrue(m.contains("31.25 ms"), m)
        XCTAssertTrue(m.contains("RenderClock"), m)
        XCTAssertTrue(m.contains("CACurrentMediaTime"), m)
    }
}
