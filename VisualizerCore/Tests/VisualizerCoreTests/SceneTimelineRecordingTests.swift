import XCTest
@testable import VisualizerCore

/// Headless gate for the timeline's automation-WRITE recorder: arming captures
/// every binding, live changes are written as keyframes, the duration grows
/// open-ended, and the recording plays back through `apply()`.
@MainActor
final class SceneTimelineRecordingTests: XCTestCase {

    /// Tiny settings stand-in the bindings read/write.
    private final class Backing {
        var value: Double = 1.0
        var flag: Bool = false
    }

    private func makeTimeline(_ s: Backing) -> SceneTimeline {
        // Start short so the test exercises open-ended duration growth.
        SceneTimeline(
            duration: 0.05,
            bindings: [
                .double("value", "Value", 0...10, get: { s.value }, set: { s.value = $0 }),
                .bool("flag", "Flag", get: { s.flag }, set: { s.flag = $0 }),
            ]
        )
    }

    /// Arming creates a track for EVERY binding, each seeded with the current
    /// value at the playhead — "every setting is prepared to be recorded."
    func testArmingPreparesEverySetting() {
        let s = Backing()
        let tl = makeTimeline(s)
        XCTAssertTrue(tl.tracks.isEmpty)

        tl.startRecording()
        XCTAssertTrue(tl.isRecording)
        XCTAssertFalse(tl.isPlaying)
        XCTAssertEqual(tl.tracks.count, 2, "one track per binding")
        XCTAssertEqual(tl.track(named: "value")?.keyframes.count, 1, "start anchor")
        XCTAssertEqual(tl.track(named: "flag")?.keyframes.count, 1)
    }

    /// A change to a setting mid-record is captured, the duration grows to follow
    /// the playhead, and the result plays back through `apply()`.
    func testRecordsChangesAndPlaysBack() {
        let s = Backing()
        let tl = makeTimeline(s)
        tl.startRecording()                     // t=0: value=1, flag=false anchors

        // idle tick — no change, no new keyframe beyond the anchor.
        tl.advance(by: 0.1)                      // t=0.1
        XCTAssertEqual(tl.track(named: "value")?.keyframes.count, 1)

        s.value = 5                              // user moves the slider
        tl.advance(by: 0.1)                      // t=0.2 → hold@0.1 + move@0.2

        s.flag = true                            // user flips the toggle
        tl.advance(by: 0.1)                      // t=0.3

        XCTAssertGreaterThan(tl.duration, 0.05, "duration grew past its initial cap")
        XCTAssertEqual(tl.duration, 0.3, accuracy: 1e-9, "duration follows the playhead")
        tl.stopRecording()
        XCTAssertFalse(tl.isRecording)
        XCTAssertEqual(tl.duration, 0.3, accuracy: 1e-9)

        // Playback: scrub + apply writes the recorded values back into settings.
        s.value = 0; s.flag = false             // clobber so apply must restore

        tl.seek(to: 0.0); tl.apply()
        XCTAssertEqual(s.value, 1, accuracy: 1e-6, "value held at start before the move")
        XCTAssertFalse(s.flag, "flag held false before the flip")

        tl.seek(to: 0.3); tl.apply()
        XCTAssertEqual(s.value, 5, accuracy: 1e-6, "recorded slider value")
        XCTAssertTrue(s.flag, "recorded toggle flip")
    }

    /// While recording, `apply()` must NOT write the timeline back into settings —
    /// the data flows settings → timeline, so live edits aren't clobbered.
    func testApplyIsInertWhileRecording() {
        let s = Backing()
        let tl = makeTimeline(s)
        tl.startRecording()
        tl.advance(by: 0.1)

        s.value = 7                              // a live edit not yet keyframed
        tl.apply()                               // must be a no-op while recording
        XCTAssertEqual(s.value, 7, accuracy: 1e-9, "apply didn't overwrite the live edit")
    }

    /// Pressing play (or rewind) while armed stops recording rather than fighting it.
    func testPlayStopsRecording() {
        let s = Backing()
        let tl = makeTimeline(s)
        tl.startRecording()
        tl.toggle()                              // play
        XCTAssertFalse(tl.isRecording)
        XCTAssertTrue(tl.isPlaying)
    }
}
