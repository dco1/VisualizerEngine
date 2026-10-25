/// **The measured optics of each family's reflecting surface** (Daydream DH-1300 — OpenPBR
/// `specular_ior`).
///
/// Every opaque dielectric used to reflect with F0 = 0.04 — the normal-incidence reflectance of an
/// interface of refractive index 1.5. That is right for paint binders, lacquers, cellulose and most
/// polymers, and it is the default here. The families below are the ones whose reflecting surface is
/// a different material, and whose number is not a matter of taste but a handbook value:
///
///   • `.stone` / `.terrazzo` — **1.56.** The polished face of a marble is calcite (ordinary 1.658,
///     extraordinary 1.486; a polycrystalline face averages ~1.57), of a granite quartz (1.544/1.553)
///     and feldspars (1.52–1.53), of an engineered quartz counter quartz in a polyester binder (1.55).
///     A terrazzo is marble chips polished flush in a cement or epoxy matrix (1.55–1.57). F0 0.048.
///   • `.tile` / `.ceramic` — **1.54.** A fired glaze is a silicate glass; soda-lime-alumina glazes run
///     1.50–1.58, a typical stoneware or wall-tile glaze ~1.54. F0 0.045.
///   • `.laminate` — **1.65.** A high-pressure laminate's wear face is melamine-formaldehyde resin
///     (1.65). F0 0.060 — half again the default, the most visible of the set.
///
/// Values: CRC Handbook of Chemistry and Physics (minerals, polymers); Taylor & Bull, *Ceramic Glaze
/// Technology* (glazes). Families not listed — paint, plaster, wood under its finish, fabrics,
/// leather, ground, concrete, brick — stay at 1.5, which is within ±0.05 of their own surfaces and
/// would move F0 by less than the half-float the G-buffer stores it in. Metals and glass do not use
/// this (a metal's F0 is its albedo; glass carries its own IOR).
///
/// Material physics, not product IA — the same family reflects the same way in every host — so it
/// lives with the families rather than with a host's catalogue. A per-material override, if one is
/// ever needed, belongs on the bake (`MaterialChannels`), not here.
public extension MaterialCategory {
    /// OpenPBR `specular_ior` of this family's reflecting surface; 1.5 (F0 0.04) where the family's
    /// own surface is within the default's tolerance.
    var specularIOR: Double {
        switch self {
        case .stone, .terrazzo: return 1.56
        case .tile, .ceramic:   return 1.54
        case .laminate:         return 1.65
        default:                return 1.5
        }
    }

    /// **DH-1298 — OpenPBR `coat_ior` of the family's coat**, where its `clearcoat` lobe is one: a fired
    /// glaze (`.tile`, `.ceramic`) 1.54; a lacquer / polyurethane / acrylic finish (`.wood`, `.laminate`,
    /// `.leather`, `.paint`) 1.50; a polished mineral face (`.stone`, `.terrazzo`, `.concrete`) is not a
    /// film — its "coat" is its own polish — so it takes the mineral's index, 1.56.
    var coatIOR: Double {
        switch self {
        case .tile, .ceramic:              return 1.54
        case .stone, .terrazzo, .concrete: return 1.56
        default:                           return 1.5
        }
    }

    /// **DH-1298 — OpenPBR `coat_darkening`** [0, 1]: 1 where the coat is a real FILM over a separate
    /// base (a glaze over its clay body, a finish over wood / laminate / leather / paint) — light the
    /// base scatters back is partly trapped under the film and returns to be absorbed again, which is
    /// why wet stone and oiled wood read darker and richer. 0 where the "coat" is the material's own
    /// polished face (stone, terrazzo, concrete): there is no film to trap anything.
    var coatDarkening: Double {
        switch self {
        case .tile, .ceramic, .wood, .laminate, .leather, .paint: return 1
        default:                                                  return 0
        }
    }
}
