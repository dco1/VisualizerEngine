import Foundation
import Observation

@MainActor
@Observable
public final class SceneTrack: Identifiable {
    public let id: UUID
    public let name: String
    public let displayName: String
    public let range: ClosedRange<Double>
    public let defaultEasing: Easing
    public private(set) var keyframes: [Keyframe]

    public init(
        id: UUID = UUID(),
        name: String,
        displayName: String,
        range: ClosedRange<Double>,
        defaultEasing: Easing = .easeInOut,
        keyframes: [Keyframe] = []
    ) {
        self.id = id
        self.name = name
        self.displayName = displayName
        self.range = range
        self.defaultEasing = defaultEasing
        self.keyframes = keyframes.sorted { $0.time < $1.time }
    }

    public func sample(at t: Double) -> Double? {
        guard let first = keyframes.first, let last = keyframes.last else { return nil }
        if t <= first.time { return first.value }
        if t >= last.time { return last.value }
        for i in 0..<(keyframes.count - 1) {
            let a = keyframes[i]
            let b = keyframes[i + 1]
            if t >= a.time && t <= b.time {
                let span = b.time - a.time
                if span <= 0 { return b.value }
                let f = (t - a.time) / span
                let eased = a.easing.interpolate(progress: f)
                return a.value + (b.value - a.value) * eased
            }
        }
        return last.value
    }

    @discardableResult
    public func addKeyframe(time: Double, value: Double) -> Keyframe {
        let kf = Keyframe(time: time, value: value, easing: defaultEasing)
        keyframes.append(kf)
        keyframes.sort { $0.time < $1.time }
        return kf
    }

    public func removeKeyframe(_ id: UUID) {
        keyframes.removeAll { $0.id == id }
    }

    public func removeKeyframe(_ id: UUID, undoManager: UndoManager?) {
        guard let index = keyframes.firstIndex(where: { $0.id == id }) else { return }
        let removed = keyframes.remove(at: index)
        undoManager?.registerUndo(withTarget: self) { target in
            target.reinsertKeyframe(removed, undoManager: undoManager)
        }
        undoManager?.setActionName("Delete Keyframe")
    }

    private func reinsertKeyframe(_ keyframe: Keyframe, undoManager: UndoManager?) {
        keyframes.append(keyframe)
        keyframes.sort { $0.time < $1.time }
        undoManager?.registerUndo(withTarget: self) { target in
            target.removeKeyframe(keyframe.id, undoManager: undoManager)
        }
    }

    public func setEasing(_ easing: Easing, for id: UUID, undoManager: UndoManager?) {
        guard let index = keyframes.firstIndex(where: { $0.id == id }) else { return }
        let previous = keyframes[index].easing
        guard previous != easing else { return }
        keyframes[index].easing = easing
        undoManager?.registerUndo(withTarget: self) { target in
            target.setEasing(previous, for: id, undoManager: undoManager)
        }
        undoManager?.setActionName("Change Easing")
    }

    @discardableResult
    public func removeKeyframe(near time: Double, tolerance: Double = 0.5) -> Keyframe? {
        guard !keyframes.isEmpty else { return nil }
        var bestIndex = 0
        var bestDistance = abs(keyframes[0].time - time)
        for i in 1..<keyframes.count {
            let d = abs(keyframes[i].time - time)
            if d < bestDistance {
                bestDistance = d
                bestIndex = i
            }
        }
        guard bestDistance <= tolerance else { return nil }
        return keyframes.remove(at: bestIndex)
    }

    public func clear() {
        keyframes.removeAll()
    }
}
