import Metal

/// Tracks whether the renderer's output texture has been replaced (e.g. after
/// a canvas resize) and returns the new texture only when it changes — so the
/// caller can rebind `scene.background.contents` exactly once per swap rather
/// than every tick. Replaces the repeated `boundOutputTexture: ObjectIdentifier?`
/// property that every native Illuminatorama scene carried.
///
/// Usage:
/// ```swift
/// // stored property
/// private var outputTracker = OutputTextureTracker()
///
/// // in tick():
/// if let tex = outputTracker.texture(from: renderer) {
///     scene.background.contents = tex  // visual-ok: snapshot-seam rebind only
/// }
/// ```
public struct OutputTextureTracker {
    private var id: ObjectIdentifier?

    public init() {}

    /// Returns the renderer's current output texture if it changed since the
    /// last call, or `nil` if it's the same instance.
    @MainActor
    public mutating func texture(from renderer: IlluminatoramaRenderer) -> MTLTexture? {
        let tex = renderer.outputTexture
        let newId = ObjectIdentifier(tex)
        guard newId != id else { return nil }
        id = newId
        return tex
    }

    /// Advances the tracker to the renderer's current texture WITHOUT returning
    /// it. Use after an explicit `scene.background.contents = renderer.outputTexture`
    /// set (e.g. after a `renderer.resize(...)`) so the next per-tick call to
    /// `texture(from:)` doesn't fire a redundant rebind.
    @MainActor
    public mutating func sync(to renderer: IlluminatoramaRenderer) {
        id = ObjectIdentifier(renderer.outputTexture)
    }
}
