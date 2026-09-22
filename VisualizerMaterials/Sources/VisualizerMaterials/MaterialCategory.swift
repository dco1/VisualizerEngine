/// **Broad material families** (MATERIALS_AND_TEXTURES §7 library).
///
/// A generator declares which family it drew, and the auditors branch on it — `TextureAudit`
/// expects a grain tangent from `.wood`/`.laminate` and exempts `.metal`/`.glass`/`.mirror`
/// from the dielectric albedo band; `MaterialScaleAudit` reads it to know what feature size a
/// surface ought to carry. That is the whole of its meaning HERE.
///
/// **Where a family is allowed to be applied is deliberately not in this type.** A host's
/// material↔surface matrix (a home designer's "paint belongs on a wall, not a countertop") is
/// product IA, not material physics, and it stays with the host — in Daydream Home that is
/// `MaterialApplicability` over its own `SurfaceRole`. A different host wants a different
/// matrix over the same families, which only works if the families do not carry one.
public enum MaterialCategory: String, Sendable, Equatable, Codable, CaseIterable {
    case paint, plaster, wallpaper
    case wood, laminate
    case tile, stone, terrazzo, concrete, brick
    /// **Glazed / bisque-fired clay** — a thrown or slip-cast ceramic BODY, not a tiled surface.
    /// Its own family rather than `.tile` (which is unit masonry: a grid of pieces in grout) or
    /// `.stone` (a quarried mineral): the thing that makes a pot look like a pot is a poured coat
    /// over a clay body plus the rings of the wheel, and neither sibling has any of that. It is
    /// also the reason the family is COHERENT for anti-tiling — those rings are a regular
    /// directional period, and the hex de-repeat overlays offset copies of one into moiré, the
    /// same trade `.fabric`'s weave makes.
    case ceramic
    case carpet, fabric, leather
    case metal, glass, mirror
    /// Yard / site surfaces — grass, asphalt, sidewalk, pavers, soil.
    case ground
    /// A bitmap the USER brought in, rather than anything this package generated. Its channels
    /// are whatever the import produced, so the auditors treat it as unconstrained.
    case custom
}
