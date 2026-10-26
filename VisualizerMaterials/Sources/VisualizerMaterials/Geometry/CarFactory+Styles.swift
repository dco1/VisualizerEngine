// plausibility: real — every number below is a measured production-car dimension (length, track,
// wheelbase, roof height, wheel radius); the shapes are authored lofts with true wheel arches, not
// primitives dressed up. See CarFactory.swift.

/// Body style of a `CarSpec`.
public enum CarStyle: String, CaseIterable, Sendable {
    case sedan, hatchback, suv, pickup, van, taxi, bus, coupe
}

/// Period styling: corner radii, lamp shape, bumper treatment, tyre section.
public enum CarEra: String, CaseIterable, Sendable {
    case classic1950s, nineties, modern
}

/// Per-style nominal dimensions + side profile. Distances `d` are metres measured BACK from the
/// front of the body; heights are metres above ground. Internal to the factory.
struct CarBlueprint {
    var length: Double, width: Double, height: Double
    var wheelbase: Double, frontOverhang: Double
    var wheelRadius: Double, wheelWidth: Double
    /// Underside of the body between the wheels, and at the very nose/tail.
    var sill: Double, endBottom: Double
    /// Outer surface height of the body (hood → beltline → deck), `(d, y)`.
    var top: [(Double, Double)]
    /// Outer surface of the greenhouse (windscreen rake, roof, rear glass), `(d, y)`; empty for a bus.
    var cabin: [(Double, Double)]
    var belt: Double
    var cabinWidthFrac: Double
    var cornerRadius: Double
    /// Plan-view narrowing of the nose / tail (fraction of full half-width at the very end).
    var noseTaper: Double, tailTaper: Double
    var boxy = false
    var roofSign = false
    var axleCount = 2

    var frontZ: Double { wheelbase / 2 + frontOverhang }
    var rearZ: Double { frontZ - length }

    /// Reference vehicle behind each style. L/W/H, wheelbase and tyre size are manufacturer-published
    /// (width excludes mirrors); the front/rear overhang split, ground clearance, hood and beltline
    /// heights are read from spec sheets / side elevations (±3 cm) — OEMs do not publish them.
    ///   sedan, taxi  Toyota Camry (2025)       4.915 × 1.839 × 1.445, wb 2.825, 235/45R18
    ///   hatchback    VW Golf 8                 4.284 × 1.789 × 1.456, wb 2.619, 205/55R16
    ///   suv          Toyota RAV4 (2024)        4.600 × 1.855 × 1.685, wb 2.690, 225/65R17
    ///   pickup       Ford F-150 SuperCrew 5.5′ 5.885 × 1.997 × 1.960, wb 3.683, 275/65R18
    ///   van          Ford Transit Custom SWB   4.973 × 1.986 × 1.970, wb 2.933, 215/65R16
    ///   coupe        Ford Mustang GT           4.811 × 1.915 × 1.397, wb 2.720, 255/40R19
    ///   bus          New Flyer Xcelsior 40′    12.19 × 2.600 × 3.350, wb 7.000, 305/70R22.5
    ///   1950s era    1957 Chevrolet Bel Air (sedan/taxi/coupe) 4.990 × 1.880 × 1.500, wb 2.921, 7.50-14
    static func make(_ style: CarStyle, era: CarEra = .modern) -> CarBlueprint {
        var b = authored(style)
        func fit(_ L: Double, _ W: Double, _ H: Double, wb: Double, fo: Double, R: Double, tw: Double,
                 clearance: Double, nose: Double) {
            let kL = L / b.length, kH = H / b.height
            b.top = b.top.map { ($0.0 * kL, $0.1 * kH) }
            b.cabin = b.cabin.map { ($0.0 * kL, $0.1 * kH) }
            b.belt *= kH
            b.length = L; b.width = W; b.height = H
            b.wheelbase = wb; b.frontOverhang = fo; b.wheelRadius = R; b.wheelWidth = tw
            b.sill = clearance; b.endBottom = nose
        }
        if era == .classic1950s, [.sedan, .taxi, .coupe].contains(style) {
            fit(4.99, 1.88, 1.50, wb: 2.921, fo: 0.99, R: 0.335, tw: 0.19, clearance: 0.23, nose: 0.36)
            return b
        }
        switch style {
        case .sedan, .taxi: fit(4.915, 1.839, 1.445, wb: 2.825, fo: 0.95, R: 0.345, tw: 0.235, clearance: 0.16, nose: 0.28)
        case .hatchback:    fit(4.284, 1.789, 1.456, wb: 2.619, fo: 0.87, R: 0.316, tw: 0.205, clearance: 0.15, nose: 0.27)
        case .suv:          fit(4.600, 1.855, 1.685, wb: 2.690, fo: 0.90, R: 0.362, tw: 0.225, clearance: 0.21, nose: 0.36)
        case .pickup:       fit(5.885, 1.997, 1.960, wb: 3.683, fo: 1.04, R: 0.407, tw: 0.275, clearance: 0.24, nose: 0.46)
        case .van:          fit(4.973, 1.986, 1.970, wb: 2.933, fo: 0.88, R: 0.343, tw: 0.215, clearance: 0.17, nose: 0.34)
        case .coupe:        fit(4.811, 1.915, 1.397, wb: 2.720, fo: 0.98, R: 0.343, tw: 0.255, clearance: 0.12, nose: 0.25)
        case .bus:          fit(12.19, 2.60, 3.35, wb: 7.00, fo: 2.30, R: 0.499, tw: 0.305, clearance: 0.42, nose: 0.50)
        }
        return b
    }

    private static func authored(_ style: CarStyle) -> CarBlueprint {
        switch style {
        case .sedan, .taxi:
            var b = CarBlueprint(
                length: 4.6, width: 1.85, height: 1.45, wheelbase: 2.7, frontOverhang: 0.95,
                wheelRadius: 0.33, wheelWidth: 0.215, sill: 0.30, endBottom: 0.36,
                top: [(0, 0.72), (0.12, 0.82), (1.30, 0.95), (3.95, 0.97), (4.50, 0.90), (4.6, 0.78)],
                cabin: [(1.30, 0.95), (2.15, 1.45), (3.30, 1.45), (4.00, 0.97)],
                belt: 0.95, cabinWidthFrac: 0.88, cornerRadius: 0.14, noseTaper: 0.84, tailTaper: 0.88)
            b.roofSign = style == .taxi
            return b
        case .hatchback:
            return CarBlueprint(
                length: 4.0, width: 1.75, height: 1.50, wheelbase: 2.55, frontOverhang: 0.80,
                wheelRadius: 0.31, wheelWidth: 0.195, sill: 0.28, endBottom: 0.34,
                top: [(0, 0.66), (0.10, 0.76), (1.05, 0.88), (3.2, 0.92), (3.6, 0.86), (4.0, 0.76)],
                cabin: [(1.05, 0.90), (1.80, 1.50), (3.30, 1.50), (3.85, 0.95)],
                belt: 0.90, cabinWidthFrac: 0.88, cornerRadius: 0.13, noseTaper: 0.84, tailTaper: 0.90)
        case .suv:
            return CarBlueprint(
                length: 4.75, width: 1.95, height: 1.75, wheelbase: 2.85, frontOverhang: 0.95,
                wheelRadius: 0.38, wheelWidth: 0.235, sill: 0.38, endBottom: 0.44,
                top: [(0, 0.85), (0.12, 0.98), (1.35, 1.10), (4.45, 1.12), (4.75, 1.0)],
                cabin: [(1.35, 1.10), (2.15, 1.75), (4.35, 1.75), (4.65, 1.15)],
                belt: 1.10, cabinWidthFrac: 0.90, cornerRadius: 0.16, noseTaper: 0.86, tailTaper: 0.92)
        case .pickup:
            return CarBlueprint(
                length: 5.3, width: 2.0, height: 1.9, wheelbase: 3.3, frontOverhang: 1.0,
                wheelRadius: 0.38, wheelWidth: 0.25, sill: 0.40, endBottom: 0.46,
                top: [(0, 0.88), (0.12, 1.0), (1.45, 1.12), (3.55, 1.14), (3.62, 1.0), (5.3, 1.0)],
                cabin: [(1.45, 1.12), (2.15, 1.90), (3.0, 1.90), (3.45, 1.15)],
                belt: 1.12, cabinWidthFrac: 0.90, cornerRadius: 0.13, noseTaper: 0.88, tailTaper: 0.95)
        case .van:
            return CarBlueprint(
                length: 5.2, width: 2.0, height: 2.0, wheelbase: 3.2, frontOverhang: 0.9,
                wheelRadius: 0.36, wheelWidth: 0.225, sill: 0.36, endBottom: 0.42,
                top: [(0, 0.75), (0.12, 0.9), (0.9, 1.05), (2.3, 1.05), (2.34, 2.0), (5.0, 2.0), (5.2, 1.9)],
                cabin: [(0.9, 1.05), (1.6, 2.0), (2.2, 2.0), (2.3, 2.0)],
                belt: 1.05, cabinWidthFrac: 0.92, cornerRadius: 0.15, noseTaper: 0.88, tailTaper: 0.96)
        case .coupe:
            return CarBlueprint(
                length: 4.9, width: 1.9, height: 1.45, wheelbase: 2.9, frontOverhang: 1.0,
                wheelRadius: 0.34, wheelWidth: 0.22, sill: 0.30, endBottom: 0.36,
                top: [(0, 0.70), (0.12, 0.82), (1.5, 0.96), (3.9, 0.98), (4.5, 0.90), (4.9, 0.76)],
                cabin: [(1.55, 0.96), (2.45, 1.45), (3.15, 1.45), (4.0, 0.98)],
                belt: 0.96, cabinWidthFrac: 0.84, cornerRadius: 0.16, noseTaper: 0.84, tailTaper: 0.86)
        case .bus:
            var b = CarBlueprint(
                length: 11.8, width: 2.55, height: 3.2, wheelbase: 5.9, frontOverhang: 2.6,
                wheelRadius: 0.50, wheelWidth: 0.28, sill: 0.62, endBottom: 0.55,
                top: [(0, 3.0), (0.6, 3.2), (11.2, 3.2), (11.8, 3.1)],
                cabin: [], belt: 1.3, cabinWidthFrac: 1.0, cornerRadius: 0.16, noseTaper: 0.96, tailTaper: 0.97)
            b.boxy = true
            return b
        }
    }
}
