import Foundation
import Observation

@MainActor
@Observable
public final class SceneTimeline {
    public var duration: Double
    public var currentTime: Double = 0
    public var isPlaying: Bool = false
    public var loops: Bool = true
    public private(set) var tracks: [SceneTrack]

    /// When true the timeline is in automation-WRITE mode: each tick reads every
    /// binding's live value and writes keyframes for any that changed (instead of
    /// sampling tracks INTO the settings). The playhead advances in real time and
    /// `duration` grows to follow it — recording is open-ended. Mutually exclusive
    /// with `isPlaying`.
    public private(set) var isRecording: Bool = false

    private var bindings: [String: KeyframeBinding]
    private var bindingOrder: [String]
    private var lastAppliedTime: Double?

    /// Playhead time at the previous record tick, used to pin a "hold" keyframe
    /// at the old value just before a change so a setting reads as held-then-moved
    /// rather than ramping from the record-start anchor across an idle stretch.
    private var recPrevTime: Double = 0
    /// Last value written per binding while recording, to detect changes.
    private var recLastValues: [String: Double] = [:]

    public var selectedTrackID: UUID?

    public init(
        duration: Double = 10,
        bindings: [KeyframeBinding] = [],
        initialTracks: [SceneTrack] = []
    ) {
        self.duration = duration
        self.tracks = initialTracks
        var byName: [String: KeyframeBinding] = [:]
        var order: [String] = []
        for b in bindings {
            byName[b.name] = b
            order.append(b.name)
        }
        self.bindings = byName
        self.bindingOrder = order
        self.selectedTrackID = initialTracks.first?.id
    }

    // ── Track lookup ─────────────────────────────────────────────

    public func track(named name: String) -> SceneTrack? {
        tracks.first { $0.name == name }
    }

    public func selectedTrack() -> SceneTrack? {
        guard let id = selectedTrackID else { return nil }
        return tracks.first { $0.id == id }
    }

    public func select(_ track: SceneTrack?) {
        selectedTrackID = track?.id
    }

    // ── Bindings catalog ─────────────────────────────────────────

    public var allBindings: [KeyframeBinding] {
        bindingOrder.compactMap { bindings[$0] }
    }

    public var availableBindings: [KeyframeBinding] {
        let active = Set(tracks.map(\.name))
        return allBindings.filter { !active.contains($0.name) }
    }

    public func binding(named name: String) -> KeyframeBinding? {
        bindings[name]
    }

    // ── Track add / remove / reorder ─────────────────────────────

    @discardableResult
    public func addTrack(forBindingNamed name: String) -> SceneTrack? {
        guard let b = bindings[name] else { return nil }
        if let existing = tracks.first(where: { $0.name == b.name }) {
            return existing
        }
        let track = SceneTrack(
            name: b.name,
            displayName: b.displayName,
            range: b.range,
            defaultEasing: b.defaultEasing
        )
        tracks.append(track)
        if selectedTrackID == nil { selectedTrackID = track.id }
        return track
    }

    public func removeTrack(_ id: UUID) {
        tracks.removeAll { $0.id == id }
        if selectedTrackID == id {
            selectedTrackID = tracks.first?.id
        }
    }

    public func moveTrack(id: UUID, to destination: Int) {
        guard let from = tracks.firstIndex(where: { $0.id == id }) else { return }
        let track = tracks.remove(at: from)
        let clamped = max(0, min(destination, tracks.count))
        tracks.insert(track, at: clamped)
    }

    public func moveTrackUp(_ id: UUID) {
        guard let i = tracks.firstIndex(where: { $0.id == id }), i > 0 else { return }
        tracks.swapAt(i, i - 1)
    }

    public func moveTrackDown(_ id: UUID) {
        guard let i = tracks.firstIndex(where: { $0.id == id }), i < tracks.count - 1 else { return }
        tracks.swapAt(i, i + 1)
    }

    // ── Sampling ─────────────────────────────────────────────────

    public func apply() {
        // While recording, settings flow INTO the timeline (capture), not out of
        // it — applying would overwrite the user's live edits. Stand down.
        if isRecording { return }
        let t = currentTime
        if !isPlaying, let last = lastAppliedTime, last == t {
            return
        }
        lastAppliedTime = t
        for track in tracks {
            guard let v = track.sample(at: t) else { continue }
            bindings[track.name]?.set(v)
        }
    }

    // ── Automation recording ─────────────────────────────────────

    /// Arm automation-write recording. Ensures every binding has a track and
    /// seeds each with the current value at the playhead, so every setting is
    /// "prepared to be recorded." Subsequent ticks capture changes live.
    public func startRecording() {
        guard !isRecording else { return }
        isPlaying = false
        loops = false          // open-ended; the duration just keeps going
        recLastValues.removeAll()
        let t = currentTime
        for name in bindingOrder {
            guard let b = bindings[name] else { continue }
            let track = addTrack(forBindingNamed: name) ?? tracks.first { $0.name == name }
            let v = b.get()
            track?.addKeyframe(time: t, value: v)   // start anchor for this setting
            recLastValues[name] = v
        }
        recPrevTime = t
        isRecording = true
    }

    /// Stop recording. The duration is pinned to wherever the playhead reached.
    public func stopRecording() {
        guard isRecording else { return }
        isRecording = false
        duration = max(duration, currentTime)
        recLastValues.removeAll()
        lastAppliedTime = nil   // force the next apply() to re-sample
    }

    public func toggleRecording() {
        if isRecording { stopRecording() } else { startRecording() }
    }

    /// One record tick: advance time open-endedly, grow the duration, and write a
    /// keyframe for every binding whose value changed since the last tick.
    private func recordTick(dt: Double) {
        let prevT = recPrevTime
        let t = currentTime + max(0, dt)
        currentTime = t
        if t > duration { duration = t }

        for name in bindingOrder {
            guard let b = bindings[name],
                  let track = tracks.first(where: { $0.name == name }) else { continue }
            let v = b.get()
            let last = recLastValues[name]
            let changed: Bool
            switch b.kind {
            case .boolean:   changed = last == nil || v != last!
            case .numerical: changed = last == nil || abs(v - last!) > 1e-6
            }
            guard changed else { continue }
            // Pin the old value at the previous tick so the setting holds flat
            // until it actually moved (skip if a keyframe already sits at/after
            // prevT, i.e. during a continuous drag).
            if let lastV = last,
               let latest = track.keyframes.last, latest.time < prevT - 1e-9 {
                track.addKeyframe(time: prevT, value: lastV)
            }
            track.addKeyframe(time: t, value: v)
            recLastValues[name] = v
        }
        recPrevTime = t
    }

    public func addKeyframeAtPlayhead(forTrackNamed name: String) {
        guard let track = track(named: name), let b = bindings[name] else { return }
        track.addKeyframe(time: currentTime, value: b.get())
    }

    // ── Playhead ─────────────────────────────────────────────────

    public func advance(by dt: Double) {
        if isRecording {
            recordTick(dt: dt)
            return
        }
        guard isPlaying else { return }
        var next = currentTime + dt
        if duration <= 0 {
            currentTime = 0
            return
        }
        if next >= duration {
            if loops {
                next = next.truncatingRemainder(dividingBy: duration)
            } else {
                next = duration
                isPlaying = false
            }
        }
        currentTime = next
    }

    public func play() {
        if isRecording { stopRecording() }
        isPlaying = true
    }
    public func pause() { isPlaying = false }
    public func toggle() {
        if isRecording { stopRecording() }
        isPlaying.toggle()
    }
    public func rewind() {
        if isRecording { stopRecording() }
        currentTime = 0
    }

    public func seek(to t: Double) {
        currentTime = max(0, min(duration, t))
    }
}
