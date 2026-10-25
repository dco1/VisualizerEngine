import Foundation

/// **How many rays the still's one-bounce GI pass casts per pixel per accumulated frame** —
/// the RT lighting pass DH-0689 measured at 89 % of a still's frame time (24 MP gallery
/// export: `rtLightingTLAS` 22.6 s of a 25.4 s frame).
///
/// Deliberately NOT a `QualityTier` rung. That ladder governs the LIVE canvas
/// (`RenderQuality.working` / `.settled`), and this pass never runs live under any tier —
/// `rtOpaqueLightingEnabled` is `capturingPhoto`-only, so a "Maximum" rung folded into the
/// shared ladder would render pixel-identical to "High" on screen and only diverge at the
/// moment of an actual export. That is exactly the shape of the `ultra` rung removed
/// 2026-08-31 for tying `photo` on every live measurement (`RenderQualityTests
/// .testTheLadderIsMonotonic`: "a rung that costs the same and adds nothing is a lie to the
/// user"). This is its own export-only preference instead, read only by `applyReflectionsPolicy`.
///
/// **Three rungs since 2026-10-01 (Danny):** the old High is now **Standard**, the old Maximum is
/// now **High**, and a new **Maximum** carries the photographic route of DH-0994 — the
/// path-traced GI lane (DH-0989), volumetric window light (DH-0990) and the ISO sensor model +
/// bloom PSF (DH-0992). The RAW VALUES did not move with the names: `"high"` is still the
/// cheap rung and `"maximum"` the multi-bounce cache, so a stored choice keeps rendering exactly
/// what the user picked, under its new label. The new rung's raw value is `"pathTraced"`.
///
/// `.standard` is the shipped default (DH-0689). A 2026-09-07 sweep
/// (`build/goal-hallway/gi-rays.sh`) measured `.high`'s 1-ray frame against the original 4-ray
/// frame under the SAME 32-frame photo settle: flat-tile noise 0.824 vs 0.798, a difference the
/// Ollama judge could not see (it scored the cheaper frame HIGHER — 65/70 vs 35/40 — though
/// that judge is noisy run to run and is not adjudication on its own). `.high` keeps the
/// original per-ray budget for anyone willing to trade export time for the sharpest indirect
/// light.
///
/// **`.high` also bounces light more than once (DH-0622).** It turns on the engine's
/// surface-radiance cache for the still, so a GI hit carries the light that surface itself
/// received second-hand — ceilings and shadowed sides a single bounce cannot reach fill in.
/// Measured on 4000 Sunset's Primary Bedroom (walkthrough, photo lane): +15.2 % light with the
/// export's auto-exposure on, +16 % GPU time per frame plus ~1 s of warm-up.
///
/// **Standard is traced too, since 2026-10-15 (Danny, DH-1347).** Standard's one bounce plus fills
/// was measured against a 4000-path reference on Oak Credenza Hall: −1.82 stops in the alcove,
/// −1.03 on the ceiling, grey, and blotchy (6.3 % wall mottle — the "mottle is unacceptable" defect,
/// DH-1344). On the path lane with the radiance cache (DH-1346) the same export reads within 0.04
/// stops of the reference everywhere, with 1.1–1.4 % mottle, at 16 frames: 80 s against 59 s.
/// Against Danny's photos it was a wash (Oak closer 0.50 → 0.46, Sunset Gallery 0.70 → 0.75 — too
/// orange, which is that scene's red wood and terracotta, not the method). Danny: "Standard on the
/// path lane + radiance cache, 16 frames looks good to me." Standard keeps none of the top rung's
/// extras (haze, glare PSF, sensor).
///
/// **Two rungs since 2026-10-15 (Danny, DH-1352):** the surface-cache "High" (one bounce over a cached
/// irradiance, +0.5 stops over the path reference, a look unlike the traced rungs either side of it)
/// is gone, and the photographic rung formerly labelled "Maximum" is now **High**. The name "Maximum"
/// is deliberately left free for a better rung later. Raw values: `"high"` is still Standard,
/// `"pathTraced"` is High; a stored `"maximum"` (the removed rung) is unrecognised and falls back to
/// `.default`, as any retired value does.
public enum PhotoGIQuality: String, CaseIterable, Codable, Sendable {
    /// Was "High" (then the cheap rung) — raw value kept so a stored choice is unchanged.
    case standard = "high"
    /// The photographic rung (DH-0994): path-traced GI, volumetric light, sensor model. Was "Maximum".
    case high = "pathTraced"

    public static let `default`: PhotoGIQuality = .standard

    /// `UserDefaults` key — machine-scoped like `QualityAxis`'s keys: how long you're willing
    /// to wait for an export is a property of the graphics card in front of the user, not of
    /// the house being designed, so it does not live in the `.daydreamhome` document.
    public static let defaultsKey = "photoGIQuality"

    public var displayName: String {
        switch self {
        case .standard: return "Standard"
        case .high:     return "High"
        }
    }

    public var blurb: String {
        switch self {
        case .standard:
            return "Light traced around the room, every bounce. The faster export."
        case .high:
            return "Every bounce traced like a photograph, with sunbeams in the air. Slower."
        }
    }

    /// Whether the export adds the photographic extras to the path-traced lane (DH-0989): haze,
    /// physical glare, sensor model, every-roughness reflections. Both rungs trace the diffuse light.
    public var pathTraced: Bool { self == .high }

    /// Frames a still accumulates. Standard settles in 16 (DH-1347): with the radiance cache the walls
    /// measured the same against the 4000-path reference at 16 as at 32 (left alcove 1.52 vs 1.50 %),
    /// so the other 16 bought nothing visible. High keeps `PhotoQuality.settleFrames`.
    public var settleFrames: Int {
        switch self {
        case .standard: return 16
        case .high:     return PhotoQuality.settleFrames
        }
    }

    /// GI rays per pixel per accumulated frame — `IlluminatoramaRenderer.rtGIRays`. On the
    /// path-traced rung each ray is a whole multi-bounce path, and ONE is the budget: measured
    /// against the 256-frame reference on 4000 Sunset's Primary Bedroom (DH-0642), one path is
    /// −0.05 stops / mean tile error 0.054 at 2.5 s a hero frame, two are −0.03 / 0.034 at 4.2 s,
    /// with the same flat-wall noise after the GI filter. A twentieth of a stop is not worth 1.7×.
    public var giRaysPerFrame: Int { 1 }

    /// The user's stored choice, or `.default` when nothing (or something unrecognised) is
    /// stored — the same tolerant-fallback shape `QualityTier` reads use.
    public static func stored(in defaults: UserDefaults = .standard) -> PhotoGIQuality {
        defaults.string(forKey: defaultsKey).flatMap(PhotoGIQuality.init(rawValue:)) ?? .default
    }
}
