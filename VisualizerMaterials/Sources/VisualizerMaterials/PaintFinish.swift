import simd

/// **The paint sheen ladder** — the six steps a paint store actually sells, and the ONE place
/// each step's optical consequences are written down. A finish is not a colour modifier: it is
/// how much the film levels as it dries, which sets three things at once — how tight the
/// specular lobe is (`roughness`), whether there is a polish layer over the pigment
/// (`clearcoat`), and how much of the substrate's tooth survives (`reliefScale`).
///
/// The `reliefScale` column is the one that looks like a free parameter and isn't. A flat paint
/// is high-PVC — more pigment and extender, less binder — so it dries with the roller's stipple
/// intact; a gloss enamel is binder-rich and flows level, burying it. That is why a flat ceiling
/// shows every lap mark and a gloss door shows none, and it is why sheen and tooth cannot be
/// dialled independently here without producing a surface that exists in no paint aisle.
///
/// Sheen bands are 60° gloss units, the number on the can.
public enum PaintFinish: String, CaseIterable, Equatable, Hashable, Sendable, Codable {
    case flat, matte, eggshell, satin, semiGloss, gloss

    /// The default for any newly painted surface. Eggshell, not satin: it is the standard
    /// interior wall specification, and it is where a white wall stops reading as a glossy
    /// card (Danny, 2026-08-15 — *"a perfect glossy white"* at the old 0.62 satin base).
    public static let `default`: PaintFinish = .eggshell

    public var displayName: String {
        switch self {
        case .flat:      return "Flat"
        case .matte:     return "Matte"
        case .eggshell:  return "Eggshell"
        case .satin:     return "Satin"
        case .semiGloss: return "Semi-Gloss"
        case .gloss:     return "Gloss"
        }
    }

    /// Base roughness the generator centres its spatial variation on. The generator adds the
    /// micro/tooth breakup around this value, so this is a mean, never a flat scalar — paint
    /// that does not vary its roughness trips `TextureAudit`'s flat-roughness tell.
    public var roughness: Double {
        switch self {
        case .flat:      return 0.94
        case .matte:     return 0.87
        case .eggshell:  return 0.78
        case .satin:     return 0.62
        case .semiGloss: return 0.42
        case .gloss:     return 0.26
        }
    }

    /// The polish lobe over the pigment. Flat and matte have no film to speak of; the sheens
    /// build one. This is what puts a reflected window on a gloss door and nothing on a flat wall.
    public var clearcoat: Double {
        switch self {
        case .flat:      return 0.00
        case .matte:     return 0.00
        case .eggshell:  return 0.04
        case .satin:     return 0.12
        case .semiGloss: return 0.26
        case .gloss:     return 0.45
        }
    }

    /// How much of the roller's tooth survives the film levelling — scales the surface-relief
    /// normal amplitude and the detail-band micro-occlusion. See the type note: this tracks
    /// sheen because both are consequences of the same binder content.
    public var reliefScale: Double {
        switch self {
        case .flat:      return 1.00
        case .matte:     return 0.90
        case .eggshell:  return 0.78
        case .satin:     return 0.62
        case .semiGloss: return 0.45
        case .gloss:     return 0.30
        }
    }

    /// The 60° gloss-unit band this finish occupies — shown as picker help so the choice reads
    /// as a paint spec rather than a slider.
    public var sheenDescription: String {
        switch self {
        case .flat:      return "Under 5 GU — ceilings, hides flaws, cannot be scrubbed"
        case .matte:     return "5–10 GU — low-traffic walls, very soft"
        case .eggshell:  return "10–25 GU — the standard interior wall"
        case .satin:     return "25–35 GU — kitchens, baths, wipeable"
        case .semiGloss: return "35–70 GU — trim, doors, cabinetry"
        case .gloss:     return "Over 70 GU — accent doors, high shine"
        }
    }
}

/// A paint assignment's customization: the colour, and the sheen it is mixed in.
///
/// Colour is **linear** albedo (the picker converts from sRGB), matching
/// `WallFinish.linearAlbedoVec3`.
public struct PaintParams: Equatable, Hashable, Sendable, Codable {
    public var color: Vec3
    public var finish: PaintFinish

    public init(color: Vec3, finish: PaintFinish = .default) {
        self.color = color
        self.finish = finish
    }

    private enum CodingKeys: String, CodingKey { case color, finish }

    /// **A bare colour IS a paint at the default sheen — that is the format, not a fallback.**
    ///
    /// Defining it this way rather than as a migration shim has two consequences worth having.
    /// A document written before the ladder existed is a *valid current document*, so no file
    /// has to be rewritten to keep opening — and rewriting them was the alternative, on a
    /// house document that is the layout truth for a real remodel. And a paint left at the
    /// default finish still saves byte-identically to before, so adding the ladder did not
    /// churn every `.daydreamhome` in the repo.
    ///
    /// Only a NON-default finish spends the keyed form, which is also the only case where the
    /// extra bytes carry information.
    public init(from decoder: Decoder) throws {
        if let single = try? decoder.singleValueContainer(), let c = try? single.decode(Vec3.self) {
            self.init(color: c)
            return
        }
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(color: try c.decode(Vec3.self, forKey: .color),
                  finish: try c.decodeIfPresent(PaintFinish.self, forKey: .finish) ?? .default)
    }

    public func encode(to encoder: Encoder) throws {
        guard finish != .default else {
            var single = encoder.singleValueContainer()
            try single.encode(color)
            return
        }
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(color, forKey: .color)
        try c.encode(finish, forKey: .finish)
    }
}
