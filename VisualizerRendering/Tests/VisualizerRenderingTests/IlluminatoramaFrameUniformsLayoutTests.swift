import XCTest
import simd
@testable import VisualizerRendering

/// `IlluminatoramaFrameUniforms` is a hand-maintained byte-for-byte mirror of the
/// Metal `FrameUniforms` in `Shaders/Illuminatorama.metal`. Nothing in the build
/// checks that mirror: SwiftPM only *copies* the `.metal` into the resource bundle
/// (the library is compiled by Xcode in the host app), so a Swift-side field added
/// without its Metal counterpart — or a cluster inserted in the middle rather than
/// appended — compiles clean and then silently shifts every uniform past the
/// insertion point. The symptom is never "the new knob is broken"; it's the whole
/// frame going wrong.
///
/// These are the numbers the Metal side actually reports. To re-derive them, append
/// `static_assert(sizeof(FrameUniforms) == N, "");` to a scratch copy of
/// Illuminatorama.metal and compile it with `xcrun metal -c`.
///
/// Adding a uniform cluster? Add it to BOTH structs, then update the expectations
/// here — this test failing is the reminder that the Metal struct needs the same edit.
final class IlluminatoramaFrameUniformsLayoutTests: XCTestCase {

    /// Metal's `sizeof(FrameUniforms)` (verified via static_assert). The Swift stride
    /// must match exactly or `uploadFrameUniforms`'s copy writes a differently-shaped
    /// blob than the shader reads.
    /// 1264 → 1312 with the three diagram clusters (`diagramParams` / `diagramOutline` /
    /// `diagramEdge`); 1312 → 1328 with `taaJitterDelta` (S4.2); 1328 → 1376 with the three
    /// interior irradiance bands (`interiorIrrUp/Side/Down` — the lawn-green-ceiling fix);
    /// 1376 → 1392 with the RT soft-sun-shadow cluster (`rtSunShadowSeed/Angle/RayCount` +
    /// pad — S4.1); 1392 → 1536 with the per-room band-gain table (S3.5 Stage E —
    /// `interiorRoomGain[8]` + `interiorRoomGainMeta`, NINE clusters: 32 gains packed four
    /// to a vector, one per light-layer bit, plus the enable word); 1536 → 1552 with the
    /// photographic-finish cluster (`highlightChromaRolloff` + the split-tone shadow /
    /// highlight temperatures + a pad).
    /// Verified against Metal by compiling a scratch kernel carrying
    /// `static_assert(sizeof(FrameUniforms) == 1552)`, which holds while the same assert at
    /// 1536 fails — i.e. the assert is live, not a tautology. (`offsetof` is not available
    /// in Metal; the tail offsets below are the Swift-side half of the check, and the fields
    /// are APPENDED, so stride pins them.)
    private static let metalStride = 1552

    func testFrameUniformsStrideMatchesMetal() {
        XCTAssertEqual(MemoryLayout<IlluminatoramaFrameUniforms>.stride,
                       Self.metalStride,
                       "Swift IlluminatoramaFrameUniforms stride diverged from the Metal "
                       + "FrameUniforms. Both structs must be edited together.")
    }

    /// The trailing clusters, at the offsets the Metal struct places them. A field
    /// inserted ABOVE these (rather than appended) can keep the stride correct while
    /// scrambling the tail — the stride check alone wouldn't catch that.
    func testTrailingClustersSitAtMetalOffsets() {
        assertOffset(\.nightSkyParams,  1152, "nightSkyParams")
        assertOffset(\.nightMoonDir,    1168, "nightMoonDir")
        assertOffset(\.nightSunDir,     1184, "nightSunDir")
        assertOffset(\.lensFlareParams, 1200, "lensFlareParams")
        assertOffset(\.halationParams,  1216, "halationParams")
        assertOffset(\.halationTint,    1232, "halationTint")
        assertOffset(\.bloomParams,     1248, "bloomParams")
        assertOffset(\.diagramParams,   1264, "diagramParams")
        assertOffset(\.diagramOutline,  1280, "diagramOutline")
        assertOffset(\.diagramEdge,     1296, "diagramEdge")
        assertOffset(\.taaJitterDelta,  1312, "taaJitterDelta")
        assertOffset(\.interiorIrrUp,   1328, "interiorIrrUp")
        assertOffset(\.interiorIrrSide, 1344, "interiorIrrSide")
        assertOffset(\.interiorIrrDown, 1360, "interiorIrrDown")
        // S4.1 — RT soft-sun-shadow cluster: three scalars + a pad in ONE 16-byte
        // cluster (uint/float/uint/float, so the scalar offsets pin the packing).
        assertOffset(\.rtSunShadowSeed,     1376, "rtSunShadowSeed")
        assertOffset(\.rtSunShadowAngle,    1380, "rtSunShadowAngle")
        assertOffset(\.rtSunShadowRayCount, 1384, "rtSunShadowRayCount")
        // S3.5 Stage E — the per-room band-gain table. Every vector is checked, not just
        // the first: the shader indexes it as `gains[b >> 2][b & 3]`, so a gap anywhere in
        // the run sends one room's level to another room's fragments, which reads as a
        // brightness bug in a room nobody edited.
        assertOffset(\.interiorRoomGain0,   1392, "interiorRoomGain0")
        assertOffset(\.interiorRoomGain1,   1408, "interiorRoomGain1")
        assertOffset(\.interiorRoomGain2,   1424, "interiorRoomGain2")
        assertOffset(\.interiorRoomGain3,   1440, "interiorRoomGain3")
        assertOffset(\.interiorRoomGain4,   1456, "interiorRoomGain4")
        assertOffset(\.interiorRoomGain5,   1472, "interiorRoomGain5")
        assertOffset(\.interiorRoomGain6,   1488, "interiorRoomGain6")
        assertOffset(\.interiorRoomGain7,   1504, "interiorRoomGain7")
        assertOffset(\.interiorRoomGainMeta, 1520, "interiorRoomGainMeta")
        // Photographic finish — three scalars + a pad in ONE 16-byte cluster, so the
        // scalar offsets pin the packing the way the RT-sun-shadow cluster's do.
        assertOffset(\.highlightChromaRolloff, 1536, "highlightChromaRolloff")
        assertOffset(\.shadowTemperatureK,     1540, "shadowTemperatureK")
        assertOffset(\.highlightTemperatureK,  1544, "highlightTemperatureK")
    }

    /// The packing the shader's `gains[b >> 2][b & 3]` assumes, held on the Swift side that
    /// writes it. A table whose lanes are laid out differently from the way they are read is
    /// the same defect as a stride drift, and no stride check can see it. Tested through
    /// `InteriorRoomGains.pack` because that is the ONE place the packing is written — the
    /// frame, glass and RT-instanced uniform blocks all delegate to it.
    func testGainPackingPutsEachLayerBitInTheLaneTheShaderReads() {
        let p = InteriorRoomGains.pack((0..<32).map { Float($0) + 1 }, enabled: true)
        XCTAssertEqual(p.v.count, 8, "32 gains, four to a vector")
        for b in 0..<InteriorRoomGains.capacity {
            XCTAssertEqual(p.v[b >> 2][b & 3], Float(b) + 1, "layer bit \(b)")
        }
        XCTAssertEqual(p.meta.x, 1, "the enable word must be stamped")

        // Disabled is the inert default: all-ones vectors and a zero enable, which the
        // shader short-circuits to exactly 1.0.
        let off = InteriorRoomGains.pack([0.1, 0.2], enabled: false)
        XCTAssertEqual(off.meta.x, 0)
        XCTAssertEqual(off.v[0], SIMD4<Float>.one)

        // A short table leaves the rest inert rather than zeroing them — a zero gain would
        // black out every room whose bit fell past the end.
        let short = InteriorRoomGains.pack([0.5], enabled: true)
        XCTAssertEqual(short.v[0][0], 0.5)
        XCTAssertEqual(short.v[0][1], 1.0)
        XCTAssertEqual(short.v[7], SIMD4<Float>.one)
    }

    private func assertOffset(_ key: PartialKeyPath<IlluminatoramaFrameUniforms>,
                              _ expected: Int,
                              _ name: String,
                              file: StaticString = #filePath, line: UInt = #line) {
        guard let actual = MemoryLayout<IlluminatoramaFrameUniforms>.offset(of: key) else {
            return XCTFail("\(name) has no stored-property offset", file: file, line: line)
        }
        XCTAssertEqual(actual, expected,
                       "\(name) moved — the Metal FrameUniforms places it at byte \(expected)",
                       file: file, line: line)
    }
}
