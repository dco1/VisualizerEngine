import Foundation

/// Built-in colour-grade "looks" selectable from the Illuminatorama Settings
/// panel (issue #65). Each maps to a procedurally-generated 3D LUT the renderer
/// bakes once when the look changes; `colorLUTAmount` then blends it over the
/// ungraded image. `.none` is a pure no-op (the renderer keeps the identity LUT).
///
/// These are deliberately simple, broadly-flattering film grades — a scene that
/// wants a bespoke look ships its own `.cube` via `IlluminatoramaRenderer.loadCubeLUT`.
public enum IlluminatoramaColorGradeLook: String, CaseIterable, Identifiable, Sendable {
    case none       = "None"
    case tealOrange = "Teal / Orange"
    case warm       = "Warm"
    case cool       = "Cool"
    case noir       = "Noir B&W"
    case vibrant    = "Vibrant"
    public var id: String { rawValue }
}
