import simd

/// **The glaze ladder** — the finishes a studio potter actually pulls out of a kiln, and the ONE
/// place each one's optical consequences are written down. The sibling of `PaintFinish`, one kiln
/// over, and it is the same physical argument: a glaze is a *glass melt*, and how far it flows
/// before it freezes decides four things at once, not one.
///
/// A fluid, flux-rich glaze (gloss) levels as it melts. It buries
/// the throwing rings, pools into a thick even skin, and freezes to a smooth glassy face: tight
/// specular lobe, strong clearcoat, no tooth. A stiff, alumina- and magnesia-rich glaze (matte)
/// barely moves. It stays where the potter poured it, keeps every ring the wheel left, and
/// freezes micro-crystalline — a broad scatter, almost no clearcoat, and every bit of the
/// substrate's relief still showing. That single variable is why `roughness`, `clearcoat`,
/// `throwRelief` and `bodyBreak` cannot be dialled independently here without describing a pot
/// that came out of no kiln.
///
/// Three of the seven are not sheens at all but **glaze recipes** — speckled, crackle, reactive —
/// and they belong in the same picker for the reason a pottery catalogue puts them there: the
/// character and the sheen come out of one bucket of glaze, so a person choosing a finish is
/// choosing both at once. Each still declares its own sheen, so nothing is guessed downstream.
public enum GlazeFinish: String, CaseIterable, Equatable, Hashable, Sendable, Codable {
    /// A clear fluid glaze fired to a wet, glassy skin. The vase-on-the-mantel finish.
    case gloss
    /// A soft semi-matte glaze — the everyday studio-pottery surface. The DEFAULT.
    case satin
    /// A stiff, dry, micro-crystalline matte. Reads as stone-like, almost chalky.
    case matte
    /// A satin glaze over an iron-bearing stoneware body: dark iron specks bloom up THROUGH
    /// the glaze during the firing. The most recognisably hand-made of the seven.
    case speckled
    /// A crazed glaze — the glaze shrinks more than the clay on cooling and freezes as a fine
    /// network of hairline cracks, tea-stained darker with age. Glossy, because a crackle glaze
    /// has to be a fluid one to craze at all.
    case crackle
    /// A thick variegated glaze (rutile / ash / wood-fired) that runs while it melts: it pools
    /// dark and deep in the hollows and **breaks** to the bare clay body over the high ground.
    /// The strongest colour-variation of the seven, and the least predictable — which is the point.
    case reactive
    /// No glaze at all — the bisque-fired clay body itself. Porous, matte, and it keeps every
    /// throwing ring and every bit of grog the potter wedged into it. The colour tints the CLAY,
    /// so it desaturates toward earthenware rather than sitting on top as a coat.
    case unglazed

    /// The default a freshly picked glazed-ceramic surface wears. Satin, not gloss: a full-gloss
    /// pot in a room lit by one window is a bright specular blob, and satin is what the great
    /// majority of studio pottery actually is.
    public static let `default`: GlazeFinish = .satin

    public var displayName: String {
        switch self {
        case .gloss:    return "Gloss"
        case .satin:    return "Satin"
        case .matte:    return "Matte"
        case .speckled: return "Speckled"
        case .crackle:  return "Crackle"
        case .reactive: return "Reactive"
        case .unglazed: return "Unglazed"
        }
    }

    /// One line of picker help — what the potter did, not what the shader does.
    public var summary: String {
        switch self {
        case .gloss:    return "A fluid clear glaze fired glassy — wet, deep, reflective"
        case .satin:    return "A soft semi-matte glaze — the everyday studio surface"
        case .matte:    return "A stiff dry glaze, micro-crystalline — stone-like, no shine"
        case .speckled: return "Satin glaze over an iron body — dark specks bloom through"
        case .crackle:  return "A crazed glossy glaze — a fine tea-stained network of hairlines"
        case .reactive: return "A thick running glaze — pools deep, breaks to bare clay on the edges"
        case .unglazed: return "Bare bisque-fired clay — porous, matte, every throwing ring showing"
        }
    }

    // MARK: – optical consequences (all four move together; see the type note)

    /// Base roughness the generator centres its spatial variation on. A mean, never a flat
    /// scalar — a glaze that does not vary its roughness trips `TextureAudit.roughnessIsFlat`.
    public var roughness: Double {
        switch self {
        case .gloss:    return 0.10
        case .satin:    return 0.32
        case .matte:    return 0.66
        case .speckled: return 0.34
        case .crackle:  return 0.13
        case .reactive: return 0.24
        case .unglazed: return 0.86
        }
    }

    /// The glassy lobe over the pigment — the wet look. A fired glaze is genuinely glossier than
    /// any paint, which is why `gloss` sits above `PaintFinish.gloss`'s 0.45.
    public var clearcoat: Double {
        switch self {
        case .gloss:    return 0.66
        case .satin:    return 0.24
        case .matte:    return 0.03
        case .speckled: return 0.22
        case .crackle:  return 0.58
        case .reactive: return 0.42
        case .unglazed: return 0.00
        }
    }

    /// Width of that glassy lobe (DH-0140). A well-melted gloss freezes optically flat and gives a
    /// tight, sharp reflection; a stiff matte's lobe is broad and diffuse where it exists at all.
    public var clearcoatRoughness: Double {
        switch self {
        case .gloss:    return 0.05
        case .satin:    return 0.14
        case .matte:    return 0.30
        case .speckled: return 0.14
        case .crackle:  return 0.06
        case .reactive: return 0.10
        case .unglazed: return 0.08     // inert — clearcoat is 0
        }
    }

    /// How much of the wheel's **throwing rings** survives the melt, 0…1. The tell that separates
    /// a thrown pot from an injection-moulded planter, and the direct analogue of
    /// `PaintFinish.reliefScale`: a fluid glaze floods the spiral, a stiff one preserves it, and
    /// a bare bisque body shows every turn of it.
    public var throwRelief: Double {
        switch self {
        case .gloss:    return 0.25
        case .satin:    return 0.45
        case .matte:    return 0.70
        case .speckled: return 0.50
        case .crackle:  return 0.30
        case .reactive: return 0.35
        case .unglazed: return 1.00
        }
    }

    /// Amplitude of the **glaze-thickness** field, 0…1 — how unevenly the coat pooled. Drives the
    /// depth-of-colour variation (a thick glaze reads deeper and glossier) and, with `bodyBreak`,
    /// how far the thin ground goes toward bare clay.
    public var pooling: Double {
        switch self {
        case .gloss:    return 0.30
        case .satin:    return 0.42
        case .matte:    return 0.50
        case .speckled: return 0.44
        case .crackle:  return 0.34
        case .reactive: return 1.00
        case .unglazed: return 0.00     // no coat to pool
        }
    }

    /// How far the thinnest ground breaks toward the bare CLAY BODY, 0…1. This is the reason a
    /// hand-glazed pot is not one flat colour: the coat is thinner over every high point, and
    /// what shows through there is the clay, not a lighter version of the glaze.
    public var bodyBreak: Double {
        switch self {
        case .gloss:    return 0.08
        case .satin:    return 0.14
        case .matte:    return 0.20
        case .speckled: return 0.16
        case .crackle:  return 0.10
        case .reactive: return 0.55
        case .unglazed: return 1.00     // it IS the body
        }
    }

    /// Iron **speckle** density from the clay body, 0…1 — dark flecks that bloom up through the
    /// coat during the firing. Not a decal on top: they are the body showing itself, so they read
    /// through a glaze rather than over it.
    public var speckle: Double {
        switch self {
        case .speckled: return 1.00
        case .reactive: return 0.35     // a reactive glaze is usually on a speckled body too
        case .unglazed: return 0.30     // grog in the raw clay
        default:        return 0.00
        }
    }

    /// **Crazing** density, 0…1 — the hairline network a glaze freezes into when it shrinks more
    /// than the clay under it.
    public var crackle: Double { self == .crackle ? 1.0 : 0.0 }

    /// Whether this finish has a glaze coat at all. `unglazed` is the one that does not, and
    /// several terms below are gated on it rather than being given a zero they'd have to be
    /// trusted to keep.
    public var isGlazed: Bool { self != .unglazed }
}

/// A glazed-ceramic assignment's customization — **the two axes Danny asked for**: the colour of
/// the glaze, and the finish it was fired to.
///
/// Colour is **linear** albedo (the picker converts from sRGB), matching `PaintParams.color` and
/// `WallFinish.linearAlbedoVec3`. On `.unglazed` it stains the CLAY rather than coating it, so the
/// generator takes the colour's HUE onto the clay's own value (`MaterialGenerator.clayTinted(by:)`)
/// instead of printing it flat — a bisque pot in "Cobalt" is a pale blue-grey clay, not a blue
/// coat, and pretending otherwise is the one way this material could lie about what it is.
public struct GlazeParams: Equatable, Hashable, Sendable, Codable {
    public var color: Vec3
    public var finish: GlazeFinish

    public init(color: Vec3 = GlazeParams.defaultColor, finish: GlazeFinish = .default) {
        self.color = color
        self.finish = finish
    }

    /// The colour a freshly picked glazed ceramic wears — a warm chalk white, the commonest
    /// studio glaze there is and the one that reads as pottery rather than as sanitaryware
    /// (which is a hair COOL of neutral; see `MaterialGenerator.ceramicAlbedo`).
    public static let defaultColor = Vec3(0.80, 0.77, 0.71)

    private enum CodingKeys: String, CodingKey { case color, finish }

    /// Tolerant decode, matching `PaintParams`: a stored colour with no finish is a glaze at the
    /// default sheen, so the two axes can be extended independently later without a migration.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        color = try c.decode(Vec3.self, forKey: .color)
        finish = try c.decodeIfPresent(GlazeFinish.self, forKey: .finish) ?? .default
    }
}
