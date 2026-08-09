import Foundation
import QuartzCore

/// The animation clock a host hands `IlluminatoramaRenderer.time` — and the ONE place the
/// **boot-relative → session-relative** rebase is written.
///
/// ## Why this type exists
///
/// `IlluminatoramaRenderer.time` is a 32-bit `Float`, and every GPU animation that reads it
/// (`applyTreeWind`, `applySway`'s self-oscillating pendant pendulum, the shadow passes'
/// `shadowTime`, `applyCurveWind`, the film-grain reseed) uses it as a **phase**. A `Float`'s
/// resolution is proportional to its magnitude (`t.ulp`), so the clock's *absolute value*
/// decides whether a frame-to-frame step survives the narrowing at all:
///
/// | clock value | what produces it | `Float.ulp` | resolves a 60 fps frame (16.7 ms)? |
/// |---|---|---|---|
/// | 60 s | one minute into a session | 3.8 µs | yes, 4 400× over |
/// | 3 600 s | an hour-long session | 244 µs | yes, 68× over |
/// | 86 400 s | a day-long session | 7.8 ms | marginal — 2 clock steps per frame |
/// | 337 700 s | `CACurrentMediaTime()` on a Mac awake 3.9 days | **31.25 ms** | **NO** |
///
/// `CACurrentMediaTime()` is awake-uptime *since boot*. Narrowing it straight into a `Float`
/// uniform makes every shader oscillator advance in quantised ~32 Hz steps instead of smoothly
/// — consecutive 60 fps frames land on the SAME float, then jump a whole 31.25 ms — and **the
/// quantisation halves in resolution every time uptime doubles**, so the artefact gets worse
/// the longer the user's machine stays awake and disappears entirely on a freshly-booted one.
/// It is invisible to any short test, which is exactly how it shipped: the film-grain hash died
/// completely (see the `Illuminatorama.metal` grain comment) before anyone noticed the clock.
///
/// Every `Visualizer` scene controller already honours the contract by accident — they all
/// accumulate `elapsed` from `dt` starting at 0 and send `Float(elapsed)`. This type makes that
/// contract explicit, single-sourced and testable instead of a convention each host re-derives.
///
/// ## The fix, and the candidate that was rejected
///
/// **Session epoch (this type).** Capture the first uptime sample; send `now − epoch`. The
/// subtraction happens in `Double`, so no precision is lost; every frame-to-frame delta is
/// bit-identical to what the host measured; and the value narrowed into the `Float` uniform
/// starts at 0 and stays fine-grained for any plausible session length (table above).
///
/// **Wrapping** — `now.truncatingRemainder(dividingBy: N)` — was rejected. It needs no state,
/// but the value is a *phase*, so every wrap is a discontinuity in `sin(t·ω + φ)` for every
/// oscillator whose period does not divide `N`. Wind gust (0.43 Hz), tree macro sway (0.9 Hz),
/// leaf flutter (6.5 / 5.7 / 6.0 Hz) and the pendant pendulum (2.65 rad/s) are all live at
/// once, and any future effect adds another, so no `N` is knowable. Trading a permanent,
/// uptime-dependent judder for a guaranteed periodic pop is not a fix.
///
/// ## Usage
///
/// ```swift
/// var clock = RenderClock()                       // one per renderer / scene
/// let t = clock.elapsed(at: CACurrentMediaTime())  // Double, session-relative
/// renderer.time = Float(t)                         // safe to narrow
/// ```
///
/// A host that already has a session-relative clock (an accumulated `elapsed`, or a *pinned*
/// simulation time in a deterministic capture) should assign `renderer.time` directly and skip
/// this type — rebasing an already-small clock would only re-zero it.
public struct RenderClock {

    /// The uptime sample that this clock calls zero — the first one it was handed. `nil` until
    /// the first `elapsed(at:)` call, so a clock that has never ticked has no arbitrary origin.
    ///
    /// Seeded from the *first sample* rather than from `init` deliberately: a renderer built
    /// well before its first frame (shader compilation, asset bakes, an offscreen harness that
    /// is constructed and only later driven) would otherwise start its animation clock partway
    /// through, and a test-constructed clock would carry a wall-clock origin it never used.
    public private(set) var epoch: CFTimeInterval?

    public init() {}

    /// Rebase a host monotonic-uptime sample (`CACurrentMediaTime()`, `mach_absolute_time()`
    /// in seconds, …) onto this clock's session epoch. Returns seconds since the first sample —
    /// `0` exactly on the first call.
    ///
    /// `Double` in, `Double` out: the narrowing to the renderer's `Float` is the caller's, and
    /// must happen AFTER this subtraction. `Float(now) - Float(epoch)` is not the same thing —
    /// that is the bug this type exists to prevent.
    ///
    /// Monotonicity is the caller's to supply. A sample that goes *backwards* is returned as
    /// given (a negative elapsed) rather than clamped, because silently flattening it would
    /// freeze every animation instead of showing that the host's clock is wrong.
    public mutating func elapsed(at uptime: CFTimeInterval) -> CFTimeInterval {
        guard let epoch else {
            self.epoch = uptime
            return 0
        }
        return uptime - epoch
    }

    /// Forget the epoch, so the next `elapsed(at:)` restarts the session at 0. For a host that
    /// genuinely restarts a scene's animation (scene reload, a "replay" control) — NOT something
    /// to call per frame or on resize, which would stall every oscillator at 0.
    public mutating func reset() { epoch = nil }

    // MARK: - Precision of a Float clock (the diagnostic side)

    /// The finest time step a `Float` clock reading `seconds` can represent, in seconds. This
    /// IS `Float(seconds).ulp` — named because it is the quantity the whole defect turns on:
    /// two frames closer together than this narrow to the same float and animate not at all.
    public static func resolution(atSeconds seconds: CFTimeInterval) -> CFTimeInterval {
        CFTimeInterval(Float(seconds).ulp)
    }

    /// How many `Float` clock steps fit inside one frame at `fps` — the honest "how smooth is
    /// this animation" number. ≥ 4 is comfortable; 1 means every frame lands on a different
    /// float but the phase is visibly staircased; < 1 means consecutive frames are IDENTICAL
    /// and the animation stutters at `1 / resolution` Hz instead of at the frame rate.
    public static func stepsPerFrame(atSeconds seconds: CFTimeInterval, fps: CFTimeInterval = 60) -> CFTimeInterval {
        let res = resolution(atSeconds: seconds)
        guard res > 0 else { return .infinity }
        return (1.0 / fps) / res
    }

    /// True when a `Float` clock reading `seconds` can no longer carry a smooth `fps` animation
    /// — fewer than `minStepsPerFrame` representable clock steps per frame.
    ///
    /// The default (4 steps per 60 fps frame ⇒ 4.17 ms) trips above ~35 000 s. That is ~9.7 h of
    /// *session* time, which no interactive session reaches before a relaunch, and it is tripped
    /// instantly by any boot-relative clock on a machine that has been awake more than a few
    /// hours — which is the case worth shouting about.
    public static func isTooCoarse(_ seconds: CFTimeInterval,
                                  fps: CFTimeInterval = 60,
                                  minStepsPerFrame: CFTimeInterval = 4) -> Bool {
        stepsPerFrame(atSeconds: seconds, fps: fps) < minStepsPerFrame
    }

    /// One-line diagnosis for a clock that failed `isTooCoarse` — the numbers, not a vibe.
    public static func coarseClockDiagnosis(_ seconds: CFTimeInterval, fps: CFTimeInterval = 60) -> String {
        String(format: """
               Illuminatorama: renderer.time = %.1f s is too coarse for a Float animation clock \
               (ulp %.2f ms ⇒ %.2f clock steps per %.0f fps frame). Every GPU oscillator \
               (tree wind, pendant sway, curve wind, grain) will judder, and it worsens as the \
               value grows. Feed a SESSION-relative clock: RenderClock().elapsed(at: \
               CACurrentMediaTime()) — not CACurrentMediaTime() itself, which is uptime since boot.
               """,
               seconds,
               resolution(atSeconds: seconds) * 1000,
               stepsPerFrame(atSeconds: seconds, fps: fps),
               fps)
    }
}
