import Foundation
import simd

// plausibility: real — bevelled/lofted solids at real lamp/bumper positions on the shell's own outline

extension CarContext {
    /// Bumpers, headlamps, tail lamps, fog lamps, grille / destination sign, plates, third brake light.
    /// Appends to `a.trim`, `a.chrome`, `a.headlights`, `a.tailLights`, `a.signals`, `a.plates`.
    ///
    /// **Mesh → function mapping.** Every function is its own closed shell, appended in a fixed order, so a host can
    /// drive each emissive independently. `CarFactory.lampShellMap(_:)` returns the exact list (mesh, function, side,
    /// triangle range inside that mesh); the ranges are exact for the meshes only this pass writes
    /// (`headlights`, `tailLights`, `signals`, `plates`).
    ///
    /// | mesh        | shells (function, per side unless noted)                                                        |
    /// |-------------|--------------------------------------------------------------------------------------------------|
    /// | headlights  | `headlampLens` (cover; nineties: two rectangular lenses; modern: + projector dome lenses; 1950s:    |
    /// |             | round sealed-beam dome) · `fogLamp` lens                                                         |
    /// | tailLights  | `brakeLamp` red lens per side · `thirdBrake` (one, centred)                                      |
    /// | signals     | `drl` strip · `frontIndicator` (amber; 1950s: dome lens in the bumper) · `sideRepeater` ·       |
    /// |             | `rearIndicator` (amber, wraps the corner) · `reverseLamp` (white) · `destinationSign` (bus LED)  |
    /// | trim        | `headlampHousing`, `tailHousing`, `fogHousing`, `grille` bezel, `bumper`, `bumperLip`,           |
    /// |             | `plateFrame`, `destinationSign` housing                                                          |
    /// | chrome      | `headlampReflector` bowls + projector rings, 1950s bezels / grille / bumper blades / `overrider` |
    /// |             | bullets / `exhaustTip`, `badge` discs, grille bars and honeycomb studs                           |
    /// | plates      | `plate` (front and rear, US 0.305 × 0.152 m, recessed behind its frame)                          |
    ///
    /// Era: 1950s = round chrome-bezelled sealed beams, toothed chrome grille, chrome blade bumpers; nineties =
    /// rectangular two-unit halogen headlamps, black plastic bumpers; modern = swept LED units wrapping into the
    /// fender with projectors and DRL strip.
    func buildLamps(_ a: inout CarAccumulator) {
        var rec = LampRecorder()
        buildLamps(&a, &rec)
    }

    func buildLamps(_ a: inout CarAccumulator, _ rec: inout LampRecorder) {
        buildFront(&a, &rec)
        buildRear(&a, &rec)
    }
}

public extension CarFactory {
    /// The lamp pass's shells for `spec` (see `CarContext.buildLamps`): which closed shell in which mesh is which
    /// lamp function, so a host can light brake / reverse / indicator / DRL independently.
    static func lampShellMap(_ spec: CarSpec) -> [CarLampShell] {
        let c = CarContext.lampContext(spec)
        var a = CarAccumulator(), rec = LampRecorder()
        c.buildLamps(&a, &rec)
        return rec.shells
    }
}

extension CarContext {
    /// The same construction context `CarFactory.build` makes for `spec` (used by the shell map and the tests).
    static func lampContext(_ spec: CarSpec) -> CarContext {
        var b = CarBlueprint.make(spec.style, era: spec.era)
        switch spec.era {
        case .classic1950s: b.cornerRadius *= 1.6; b.cabinWidthFrac -= 0.04
        case .nineties:     break
        case .modern:       b.cornerRadius *= 0.85
        }
        let L = spec.length ?? b.length, W = spec.width ?? b.width, H = spec.height ?? b.height
        let R = spec.wheelRadius ?? b.wheelRadius
        return CarContext(bp: b, era: spec.era, L: L, W: W, H: H, R: R, rL: L / b.length, rH: H / b.height)
    }
}
