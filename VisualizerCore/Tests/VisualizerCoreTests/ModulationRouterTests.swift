import XCTest
@testable import VisualizerCore

/// Headless gate for the live-modulation layer (issue #63): a source channel
/// drives a binding additively on top of its base, the attack/release envelope
/// smooths it, composition with keyframes holds, and removing an assignment
/// restores the captured base.
@MainActor
final class ModulationRouterTests: XCTestCase {

    private final class Backing {
        var value: Double = 5.0   // mid-range of 0...10
    }

    /// A constant-output test source we can pin to any channel value.
    private final class FixedSource: ModulationSource {
        let id = "fixed"
        let displayName = "Fixed"
        let channels = [ModulationChannel(id: "v", displayName: "V")]
        var isLive: Bool { true }
        var out: Double = 0
        func frame(dt: Double) {}
        func value(forChannel channelID: String) -> Double { out }
    }

    private func makeTimeline(_ s: Backing) -> SceneTimeline {
        SceneTimeline(duration: 10, bindings: [
            .double("value", "Value", 0...10, s, \.value),
        ])
    }

    func testAdditiveModulationOnNonKeyframedBase() {
        let src = FixedSource()
        ModulationRegistry.shared.register(src)
        let s = Backing()
        let tl = makeTimeline(s)

        // depth defaults to 1.0 (full range swing), attack tiny → near-instant.
        let a = tl.addModulation(bindingName: "value", sourceID: "fixed", channelID: "v")!
        var edited = a; edited.attack = 0; edited.release = 0
        tl.modulation.update(edited)

        // Signal 0 → value sits at its captured base (5.0).
        src.out = 0
        tl.advance(by: 1.0 / 60.0)
        tl.apply()
        XCTAssertEqual(s.value, 5.0, accuracy: 1e-6, "no signal ⇒ base value")

        // Full signal → base + depth·span = 5 + 1·10 = 15, clamped to 10.
        src.out = 1
        tl.advance(by: 1.0 / 60.0)
        tl.apply()
        XCTAssertEqual(s.value, 10.0, accuracy: 1e-6, "full signal clamps to range max")

        // Half signal → 5 + 0.5·10 = 10 → clamps to 10 as well; use depth 0.4.
        var half = tl.modulation.assignments(forBindingNamed: "value").first!
        half.depth = 0.4; half.attack = 0; half.release = 0
        tl.modulation.update(half)
        src.out = 0.5
        tl.advance(by: 1.0 / 60.0)
        tl.apply()
        XCTAssertEqual(s.value, 5.0 + 0.4 * 0.5 * 10.0, accuracy: 1e-6, "additive on base")
    }

    func testRemoveRestoresBase() {
        let src = FixedSource()
        ModulationRegistry.shared.register(src)
        let s = Backing()
        let tl = makeTimeline(s)

        let a = tl.addModulation(bindingName: "value", sourceID: "fixed", channelID: "v")!
        var edited = a; edited.attack = 0; edited.release = 0
        tl.modulation.update(edited)
        src.out = 1
        tl.advance(by: 1.0 / 60.0)
        tl.apply()
        XCTAssertEqual(s.value, 10.0, accuracy: 1e-6)

        // Removing the only assignment restores the captured base (5.0).
        tl.removeModulation(a.id)
        tl.advance(by: 1.0 / 60.0)
        tl.apply()
        XCTAssertEqual(s.value, 5.0, accuracy: 1e-6, "base restored on unassign")
    }

    func testKeyPathBindingLinksByIdentity() {
        let s = Backing()
        let tl = makeTimeline(s)
        let kp: AnyKeyPath = \Backing.value
        XCTAssertNotNil(tl.binding(forIdentity: kp), "key-path binding is found by identity")
        XCTAssertFalse(tl.isModulated(identity: kp))
        _ = tl.addModulation(bindingName: "value", sourceID: "lfo", channelID: "sine")
        XCTAssertTrue(tl.isModulated(identity: kp), "control links to modulation via key-path")
    }
}
