import Foundation

// MARK: – Comparable.clamped

public extension Comparable {
    /// Clamp into a closed range. The codebase had ~25 hand-written `min(max(x, lo), hi)`
    /// spellings of this; the nesting is easy to invert (`max(min(...))` reads the same at a
    /// glance but clamps to the wrong bound when lo > hi) and it never says *why* a value is
    /// bounded. One named helper in the shared layer, so it's usable from both `swift test`
    /// and the app.
    func clamped(to r: ClosedRange<Self>) -> Self {
        min(max(self, r.lowerBound), r.upperBound)
    }
}
