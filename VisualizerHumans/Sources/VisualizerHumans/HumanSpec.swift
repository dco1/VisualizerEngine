import simd

// Silhouette archetypes from the MakeHuman bodyshapes set. Each maps to the
// gendered "elvs" target(s); the blend respects the gender continuum.
public enum BodyArchetype: String, CaseIterable, Sendable {
    case none, apple, pear, hourglass, invertedTriangle, rectangle, leanColumn, trapezoid

    var feminineTarget: String? {
        switch self {
        case .none: nil
        case .apple: "bodyshapes/bodyshapes-elvs-fem-apple"
        case .pear: "bodyshapes/bodyshapes-elvs-fem-triangle"
        case .hourglass: "bodyshapes/bodyshapes-elvs-fem-full-hourglass"
        case .invertedTriangle: "bodyshapes/bodyshapes-elvs-fem-invert-triangle"
        case .rectangle: "bodyshapes/bodyshapes-elvs-fem-rectangle"
        case .leanColumn: "bodyshapes/bodyshapes-elvs-fem-lean-column"
        case .trapezoid: "bodyshapes/bodyshapes-elvs-fem-rectangle"   // no fem trapezoid shipped
        }
    }

    var masculineTarget: String? {
        switch self {
        case .none: nil
        case .apple: "bodyshapes/bodyshapes-elvs-man-apple"
        case .pear: "bodyshapes/bodyshapes-elvs-man-triangle"
        case .hourglass: "bodyshapes/bodyshapes-elvs-man-rectangle"   // no masc hourglass shipped
        case .invertedTriangle: "bodyshapes/bodyshapes-elvs-man-invert-triangle"
        case .rectangle: "bodyshapes/bodyshapes-elvs-man-rectangle"
        case .leanColumn: "bodyshapes/bodyshapes-elvs-man-lean-column"
        case .trapezoid: "bodyshapes/bodyshapes-elvs-man-trapezoid"
        }
    }
}

// Everything that defines one generated person. All continuous fields are 0…1
// except the physical ones, which use real-world units. Deterministic per seed.
public struct HumanSpec: Sendable, Hashable {
    public var ageYears: Float          // 1…90
    public var gender: Float            // 0 female … 1 male (continuum, matches MakeHuman)
    public var muscle: Float            // 0…1, 0.5 average
    public var weight: Float            // 0…1, 0.5 average
    public var height: Float            // 0…1, 0.5 average (relative within age/sex range)
    public var proportions: Float       // 0 uncommon … 0.5 neutral … 1 idealized
    public var african: Float           // ethnicity blend; normalized internally
    public var asian: Float
    public var caucasian: Float
    public var skinTone: Float          // 0 lightest … 1 darkest (melanin ramp)
    public var archetype: BodyArchetype
    public var archetypeAmount: Float   // 0…1 blend of the archetype silhouette
    public var belly: Float             // 0 toned flat … 0.5 neutral … 1 heavy paunch
    public var glutes: Float            // 0…0.5…1 buttocks volume
    public var hips: Float              // 0…0.5…1 hip width
    public var cup: Float               // female-weighted chest size, 0…0.5…1
    public var firmness: Float          // female-weighted chest firmness, 0…0.5…1
    public var seed: UInt64

    public init(
        ageYears: Float = 30, gender: Float = 0.5,
        muscle: Float = 0.5, weight: Float = 0.5, height: Float = 0.5,
        proportions: Float = 0.5,
        african: Float = 0, asian: Float = 0, caucasian: Float = 1,
        skinTone: Float = 0.35,
        archetype: BodyArchetype = .none, archetypeAmount: Float = 0,
        belly: Float = 0.5, glutes: Float = 0.5, hips: Float = 0.5,
        cup: Float = 0.5, firmness: Float = 0.5,
        seed: UInt64 = 1
    ) {
        self.archetype = archetype
        self.archetypeAmount = min(max(archetypeAmount, 0), 1)
        self.belly = min(max(belly, 0), 1)
        self.glutes = min(max(glutes, 0), 1)
        self.hips = min(max(hips, 0), 1)
        self.cup = min(max(cup, 0), 1)
        self.firmness = min(max(firmness, 0), 1)
        self.ageYears = min(max(ageYears, 1), 90)
        self.gender = min(max(gender, 0), 1)
        self.muscle = min(max(muscle, 0), 1)
        self.weight = min(max(weight, 0), 1)
        self.height = min(max(height, 0), 1)
        self.proportions = min(max(proportions, 0), 1)
        self.african = max(african, 0)
        self.asian = max(asian, 0)
        self.caucasian = max(caucasian, 0)
        self.skinTone = min(max(skinTone, 0), 1)
        self.seed = seed
    }

    // A plausible random person. Ages skew adult; ethnicity is a random simplex point.
    public static func random(seed: UInt64) -> HumanSpec {
        var rng = seed == 0 ? 0x9E3779B97F4A7C15 : seed
        func next() -> Float {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
            return Float(rng >> 40) / Float(1 << 24)
        }
        let age: Float = 5 + pow(next(), 0.7) * 75
        let a = next(), b = next(), c = next()
        let sum = max(a + b + c, 0.001)
        let archetypes: [BodyArchetype] = [.apple, .pear, .hourglass, .invertedTriangle, .rectangle, .leanColumn, .trapezoid]
        let rollArchetype = next() < 0.55
        return HumanSpec(
            ageYears: age,
            gender: next() < 0.5 ? next() * 0.25 : 0.75 + next() * 0.25,
            muscle: 0.2 + next() * 0.6,
            weight: 0.15 + next() * 0.7,
            height: 0.2 + next() * 0.6,
            proportions: 0.3 + next() * 0.5,
            african: a / sum, asian: b / sum, caucasian: c / sum,
            skinTone: next(),
            archetype: rollArchetype ? archetypes[Int(next() * 0.999 * Float(archetypes.count))] : .none,
            archetypeAmount: 0.35 + next() * 0.5,
            belly: 0.25 + next() * 0.65,
            glutes: 0.25 + next() * 0.5,
            hips: 0.3 + next() * 0.4,
            cup: 0.2 + next() * 0.6,
            firmness: 0.3 + next() * 0.5,
            seed: seed)
    }
}
