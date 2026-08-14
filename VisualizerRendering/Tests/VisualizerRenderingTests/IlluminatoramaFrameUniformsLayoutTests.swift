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
    /// `diagramEdge`); 1312 → 1328 with `taaJitterDelta` (S4.2). Verified against Metal by
    /// compiling a scratch kernel carrying `static_assert(sizeof(FrameUniforms) == 1328)`,
    /// which holds while the same assert at 1312 fails — i.e. the assert is live, not a
    /// tautology. (`offsetof` is not available in Metal; the tail offsets below are the
    /// Swift-side half of the check, and the field is APPENDED, so stride pins it.)
    private static let metalStride = 1328

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
