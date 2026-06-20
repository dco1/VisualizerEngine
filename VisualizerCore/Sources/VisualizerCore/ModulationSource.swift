import Foundation
import Observation

/// A live signal that can drive scene settings — the source side of the
/// "modulation" concept (issue #63). Audio (ScreenCaptureKit level/band/beat)
/// and, later, MIDI are concrete `ModulationSource`s; both publish a small set
/// of named, normalised 0…1 **channels** that a `ModulationRouter` maps into a
/// `KeyframeBinding`'s range. They are peers of the keyframe timeline: three
/// ways of writing the same `binding.set`.
///
/// Concurrency: `value(forChannel:)` is read every frame on the main actor from
/// `ModulationRouter.apply`. A capture-thread source (audio) must publish its
/// latest analysis with a lock-free hand-off so this stays non-blocking; it
/// returns the most recent latched value and never waits.
@MainActor
public protocol ModulationSource: AnyObject {
    /// Stable identifier used in an assignment (e.g. `"audio"`, `"lfo"`).
    var id: String { get }
    /// Human-facing name for the source picker (e.g. `"Audio In"`).
    var displayName: String { get }
    /// The channels this source exposes, in display order.
    var channels: [ModulationChannel] { get }
    /// Whether the source is currently producing meaningful signal (capturing /
    /// connected). Drives the "● live" indicator and lets the UI grey out a
    /// source that isn't running. A source with no runtime gate returns `true`.
    var isLive: Bool { get }
    /// Latest normalised 0…1 value for a channel. Returns 0 for an unknown id.
    /// Must be cheap and non-blocking — called once per assignment per frame.
    func value(forChannel channelID: String) -> Double
    /// Advance any internal time-based state by `dt` seconds. Pumped exactly
    /// once per frame by the active scene's timeline (only one scene ticks at a
    /// time). Push-driven sources (audio) treat this as a latch and may ignore
    /// `dt`; generated sources (LFO) advance their phase here.
    func frame(dt: Double)
}

/// One named output of a `ModulationSource`.
public struct ModulationChannel: Identifiable, Hashable, Sendable {
    public let id: String
    public let displayName: String
    /// A momentary spike (e.g. a beat onset) rather than a continuous level.
    /// The UI draws it as a flash and a router may treat it as a trigger.
    public let isImpulse: Bool

    public init(id: String, displayName: String, isImpulse: Bool = false) {
        self.id = id
        self.displayName = displayName
        self.isImpulse = isImpulse
    }
}

/// Process-wide registry of available modulation sources. The app registers its
/// concrete sources (e.g. `AudioModulationSource`) at launch; a `ModulationRouter`
/// resolves an assignment's `sourceID` through here, and the timeline pumps every
/// registered source once per frame via `frame(dt:)`.
///
/// Lives in `VisualizerCore` (alongside `SceneTimeline`) so the router can reach
/// it without depending on the app target, where the audio source actually lives.
@MainActor
@Observable
public final class ModulationRegistry {
    public static let shared = ModulationRegistry()

    public private(set) var sources: [any ModulationSource] = []

    private init() {
        // A built-in low-frequency oscillator, always available. It validates
        // the whole modulation path (assignment → envelope → additive write →
        // live lane meter) before any audio capture exists, and stays useful as
        // a "test signal" / pure-synthetic modulator.
        register(LFOModulationSource())
    }

    public func register(_ source: any ModulationSource) {
        guard !sources.contains(where: { $0.id == source.id }) else { return }
        sources.append(source)
    }

    public func source(id: String) -> (any ModulationSource)? {
        sources.first { $0.id == id }
    }

    /// Pump every registered source once. Called by the active scene's
    /// `SceneTimeline.apply()` — exactly one timeline ticks at a time, so each
    /// source advances once per displayed frame.
    public func frame(dt: Double) {
        for s in sources { s.frame(dt: dt) }
    }
}

/// A self-contained low-frequency oscillator source. No capture, no
/// dependencies — purely generated — so it can validate the modulation pipeline
/// and serve as a deliberate synthetic modulator. Exposes a sine and a
/// (unipolar) triangle, plus a derived "pulse" impulse at each cycle start.
@MainActor
@Observable
public final class LFOModulationSource: ModulationSource {
    public let id = "lfo"
    public let displayName = "Test LFO"
    public let channels: [ModulationChannel] = [
        ModulationChannel(id: "sine", displayName: "Sine"),
        ModulationChannel(id: "triangle", displayName: "Triangle"),
        ModulationChannel(id: "pulse", displayName: "Pulse", isImpulse: true),
    ]
    public var isLive: Bool { true }

    /// Cycles per second.
    public var rateHz: Double = 0.5

    private var phase: Double = 0          // 0…1
    private var pulse: Double = 0          // decays after each wrap

    public init() {}

    public func frame(dt: Double) {
        let prev = phase
        phase += rateHz * max(0, dt)
        if phase >= 1 {
            phase -= floor(phase)
            pulse = 1
        } else if phase < prev {            // safety for large dt wraps
            pulse = 1
        }
        pulse = max(0, pulse - max(0, dt) * 6)   // ~0.17s flash
    }

    public func value(forChannel channelID: String) -> Double {
        switch channelID {
        case "sine":     return 0.5 + 0.5 * sin(phase * 2 * .pi)
        case "triangle": return phase < 0.5 ? phase * 2 : (1 - phase) * 2
        case "pulse":    return pulse
        default:         return 0
        }
    }
}
