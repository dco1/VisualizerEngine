import XCTest
@testable import VisualizerMaterials

/// Standalone smoke test for the carpet generator — the audits it must clear before the
/// registry/rug wiring in DaydreamCore can rely on it. Kept in the engine package so it runs
/// under `Scripts/bump-engine.sh`'s standalone gate.
final class CarpetStandaloneTests: XCTestCase {

    /// The rug renders at the carpet mesh UV scale — mirrored here (the engine can't import the
    /// app's `carpetMeshUVScale`). Keep in lockstep with `HouseRenderBridge.carpetMeshUVScale`.
    let rugRun = 0.30

    func testCutPileIsPlausible() {
        for pile in [CarpetPile.cutPile, .loop, .flatweave] {
            let a = TextureAudit.audit(MaterialGenerator.carpet(size: 128, pile: pile))
            XCTAssertTrue(a.isPlausible,
                "carpet(\(pile)) is not plausible: outOfRange=\(a.albedoOutOfRange) "
                + "roughFlat=\(a.roughnessIsFlat) flatColor=\(a.isFlatColor) seamy=\(a.isSeamy) "
                + "missingGrain=\(a.missingGrain) normalsUnit=\(a.normalsAreUnit)")
        }
    }

    func testCarriesACoherentNapAndSheen() {
        let ch = MaterialGenerator.carpet(size: 128)
        XCTAssertNotNil(ch.grainTangent, "a carpet's nap is its whole tell — it must carry a grain tangent")
        XCTAssertGreaterThan(ch.sheen, 0, "a carpet needs the cloth grazing lobe")
        let a = TextureAudit.audit(ch)
        XCTAssertGreaterThanOrEqual(a.grainCoherence, TextureAudit.grainFloor,
            "the nap must be coherent (a laid pile), not random: \(a.grainCoherence)")
    }

    /// The velvet failure was a SCALE failure: the pile must resolve BELOW the 3 mm touch band
    /// at the run the rug renders it at, and it must not be under-resolved there.
    func testPileIsSubTouchScaleAtRugRun() {
        let sc = MaterialScaleAudit.audit(MaterialGenerator.carpet(size: 512), runMeters: rugRun)
        XCTAssertFalse(sc.isUnderResolved,
            "carpet at the rug run \(rugRun) m is under-resolved (finest \(sc.finestRepresentableMM) mm > 3 mm)")
        XCTAssertGreaterThanOrEqual(sc.touchScaleFraction, MaterialScaleAudit.touchScaleFloor,
            "carpet carries no touch-scale pile detail at the rug run: \(sc.touchScaleFraction)")
        XCTAssertTrue(sc.hasTouchScaleDetail, "carpet must read as a real pile up close at rug size")
    }

    func testIsDeterministicPerSeed() {
        let a = MaterialGenerator.carpet(size: 64, seed: 3)
        let b = MaterialGenerator.carpet(size: 64, seed: 3)
        XCTAssertEqual(a.albedo, b.albedo, "same seed must bake the same pile")
    }
}
