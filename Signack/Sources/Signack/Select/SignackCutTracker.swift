import simd

/// Hysteresis for a host that re-selects a cut on the CPU: only look again once the camera has
/// moved enough, relative to its distance, that the cut could have changed meaningfully.
///
/// Any cut the selector returned is valid (crack-free) for EVERY view — a stale one is merely a
/// little finer or coarser than ideal. So holding the old cut until the camera has moved
/// `relativeMove` of its distance (or zoomed by `scaleChange`) costs nothing in correctness and
/// keeps an orbit, a pan or a TAA jitter from rewriting index buffers every frame.
public struct SignackCutTracker: Sendable, Equatable {
    public var relativeMove: Float
    public var scaleChange: Float
    public private(set) var lastPosition: SIMD3<Float>?
    public private(set) var lastScale: Float = 0

    public init(relativeMove: Float = 0.12, scaleChange: Float = 1.1) {
        self.relativeMove = relativeMove
        self.scaleChange = scaleChange
    }

    /// Should the host re-select for `view`, given geometry centred at `center` with `radius`?
    /// Records the view when it answers yes.
    public mutating func shouldReselect(_ view: SignackView, center: SIMD3<Float>, radius: Float) -> Bool {
        guard let last = lastPosition else { record(view); return true }
        let ratio = view.pixelScale / max(lastScale, 1e-12)
        if ratio > scaleChange || ratio < 1 / scaleChange { record(view); return true }
        if view.isOrthographic { return false }
        let reach = max(simd_distance(last, center), radius, 1e-3)
        if simd_distance(view.position, last) > relativeMove * reach { record(view); return true }
        return false
    }

    /// Forget the recorded view (the next call re-selects).
    public mutating func reset() { lastPosition = nil; lastScale = 0 }

    private mutating func record(_ view: SignackView) {
        lastPosition = view.position
        lastScale = view.pixelScale
    }
}
