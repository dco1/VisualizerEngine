import Foundation
import Observation

/// One live modulation of a single `KeyframeBinding` by a source channel.
///
/// The mapped contribution at full signal is `(offset + depth) · span`, where
/// `span` is the binding range's width; at zero signal it's `offset · span`.
/// `depth` may be negative to invert the swing without flipping the resting
/// point. The value is smoothed by an attack/release envelope so visuals don't
/// strobe (the `CameraEasing` tau idea, per the issue).
public struct ModulationAssignment: Identifiable, Sendable, Equatable {
    public var id: UUID
    public let bindingName: String
    public var sourceID: String
    public var channelID: String
    /// Swing as a fraction of the binding range, signed. −1…1.
    public var depth: Double
    /// Resting offset as a fraction of the binding range, applied on top of the
    /// base value. −1…1.
    public var offset: Double
    /// Envelope rise time (s) — how fast the smoothed signal chases a rise.
    public var attack: Double
    /// Envelope fall time (s).
    public var release: Double
    /// Mirror the channel signal (1 − s) before applying depth.
    public var invert: Bool

    public init(
        id: UUID = UUID(),
        bindingName: String,
        sourceID: String,
        channelID: String,
        depth: Double = 1.0,
        offset: Double = 0.0,
        attack: Double = 0.03,
        release: Double = 0.18,
        invert: Bool = false
    ) {
        self.id = id
        self.bindingName = bindingName
        self.sourceID = sourceID
        self.channelID = channelID
        self.depth = depth
        self.offset = offset
        self.attack = attack
        self.release = release
        self.invert = invert
    }
}

/// The live-modulation peer of keyframe sampling. Holds the per-scene set of
/// `ModulationAssignment`s and, each frame, blends active sources additively on
/// top of each binding's *base* value (the keyframe-sampled value if the
/// setting also has a track, else the value captured when it was first
/// assigned). Owned by `SceneTimeline`, which runs it right after `apply()`'s
/// keyframe pass — so every timeline-capable scene gets modulation with no
/// per-controller change.
///
/// Composition (issue #63, "adds on top"): `value = base + Σ (offset + depth·s)·span`,
/// clamped to the binding range. A keyframed setting keeps its automation as the
/// moving base; a non-keyframed one holds its captured base and just gets the
/// live swing.
@MainActor
@Observable
public final class ModulationRouter {
    public private(set) var assignments: [ModulationAssignment] = []

    /// Smoothed envelope value per assignment id (0…1), persisted across frames.
    private var envelopes: [UUID: Double] = [:]
    /// Base value captured for a non-keyframed binding when first assigned, so
    /// modulation adds onto a stable anchor instead of feeding back on itself.
    private var capturedBase: [String: Double] = [:]
    /// Last raw signal per assignment, exposed to the lane meter without a
    /// re-read of the source (which the UI can't safely poll every redraw).
    private var lastSignal: [UUID: Double] = [:]

    public init() {}

    // ── Assignment management ────────────────────────────────────

    public func assignments(forBindingNamed name: String) -> [ModulationAssignment] {
        assignments.filter { $0.bindingName == name }
    }

    public func isModulated(_ name: String) -> Bool {
        assignments.contains { $0.bindingName == name }
    }

    /// Add (or no-op if an identical binding/source/channel triple exists). The
    /// base for a non-keyframed binding is captured here via `currentValue`.
    @discardableResult
    public func assign(
        bindingName: String,
        sourceID: String,
        channelID: String,
        currentValue: Double
    ) -> ModulationAssignment {
        if let existing = assignments.first(where: {
            $0.bindingName == bindingName && $0.sourceID == sourceID && $0.channelID == channelID
        }) {
            return existing
        }
        let a = ModulationAssignment(bindingName: bindingName, sourceID: sourceID, channelID: channelID)
        assignments.append(a)
        if capturedBase[bindingName] == nil { capturedBase[bindingName] = currentValue }
        envelopes[a.id] = 0
        return a
    }

    public func update(_ assignment: ModulationAssignment) {
        guard let i = assignments.firstIndex(where: { $0.id == assignment.id }) else { return }
        assignments[i] = assignment
    }

    /// Remove an assignment. `restore` is called with the binding's base value
    /// when this was the binding's *last* assignment, so the setting returns to
    /// where it rested rather than freezing at its last modulated value.
    public func remove(_ id: UUID, restore: (_ bindingName: String, _ base: Double) -> Void) {
        guard let i = assignments.firstIndex(where: { $0.id == id }) else { return }
        let name = assignments[i].bindingName
        assignments.remove(at: i)
        envelopes[id] = nil
        lastSignal[id] = nil
        if !isModulated(name) {
            if let base = capturedBase[name] { restore(name, base) }
            capturedBase[name] = nil
        }
    }

    public func clear(restore: (_ bindingName: String, _ base: Double) -> Void) {
        for (name, base) in capturedBase { restore(name, base) }
        assignments.removeAll()
        envelopes.removeAll()
        lastSignal.removeAll()
        capturedBase.removeAll()
    }

    /// Latest smoothed envelope (0…1) for a lane meter. Returns 0 if unknown.
    public func envelope(for id: UUID) -> Double { envelopes[id] ?? 0 }

    /// Latest raw (pre-envelope) channel signal (0…1), for a lane meter's
    /// "incoming" tick. Returns 0 if unknown.
    public func rawSignal(for id: UUID) -> Double { lastSignal[id] ?? 0 }

    // ── Per-frame application ────────────────────────────────────

    /// Blend every assignment onto its binding. `keyframeValues` holds the
    /// values the keyframe pass just wrote this frame (used as the moving base
    /// for keyframed settings); `dt` drives the envelope.
    public func apply(
        bindings: [String: KeyframeBinding],
        keyframeValues: [String: Double],
        dt: Double
    ) {
        guard !assignments.isEmpty else { return }

        // Group by binding so multiple assignments on one setting sum cleanly
        // and we call `set` once per binding.
        var contributionByBinding: [String: Double] = [:]
        for a in assignments {
            guard let source = ModulationRegistry.shared.source(id: a.sourceID) else {
                lastSignal[a.id] = 0
                continue
            }
            let raw = max(0, min(1, source.value(forChannel: a.channelID)))
            lastSignal[a.id] = raw

            // Attack/release: chase `raw` with the time constant for the
            // direction of travel. tau→0 means "snap" (no smoothing).
            let prev = envelopes[a.id] ?? 0
            let tau = raw >= prev ? a.attack : a.release
            let env: Double
            if tau <= 1e-4 {
                env = raw
            } else {
                let alpha = 1 - exp(-max(0, dt) / tau)
                env = prev + (raw - prev) * alpha
            }
            envelopes[a.id] = env

            let s = a.invert ? (1 - env) : env
            contributionByBinding[a.bindingName, default: 0] += a.offset + a.depth * s
        }

        for (name, contribFraction) in contributionByBinding {
            guard let b = bindings[name] else { continue }
            let base = keyframeValues[name] ?? capturedBase[name] ?? b.get()
            let span = b.range.upperBound - b.range.lowerBound
            let value = base + contribFraction * span
            let clamped = min(b.range.upperBound, max(b.range.lowerBound, value))
            b.set(clamped)
        }
    }
}
