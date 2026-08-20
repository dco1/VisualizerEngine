// swift-tools-version: 6.0
import PackageDescription

// VisualizerMaterials — the CPU-side procedural material engine.
//
// Pure Swift numerics: seeded noise, a PBR channel set, and the generators that
// author it. NO AppKit / SwiftUI / Metal / Foundation-UI — this target has zero
// dependencies on purpose, because its two consumers pull it in from opposite
// ends. `DaydreamCore` is a headless package that must keep building and testing
// under a bare `swift test`, and the Visualizer scenes reach it through
// `VisualizerRendering`. `VisualizerCore` was the obvious home and is the wrong
// one: twelve of its files import AppKit/SwiftUI/Metal, so depending on it would
// put a UI framework underneath a package whose defining property is not having
// one.
//
// The platform floor is macOS 14, one below the rest of the engine, because
// `DaydreamCore` declares 14 and a dependency cannot raise its dependent's floor
// without forcing an unrelated bump.
let package = Package(
    name: "VisualizerMaterials",
    platforms: [.macOS(.v14), .iOS(.v18), .tvOS(.v18)],
    products: [
        .library(name: "VisualizerMaterials", targets: ["VisualizerMaterials"]),
    ],
    targets: [
        .target(
            name: "VisualizerMaterials",
            swiftSettings: [
                .swiftLanguageMode(.v6),
                // OPTIMIZE EVEN IN DEBUG — this setting travelled here WITH the code, and
                // dropping it would silently undo a shipped launch-time fix.
                //
                // These generators are per-pixel numeric kernels: ~45 registry materials ×
                // 512² × 4 channels of noise/fBm math. At `-Onone` (the Debug default) Swift
                // emits no generic specialization or inlining and keeps full retain/release
                // traffic through the Vec3 math, which costs ~62× — measured on the whole
                // registry, same machine, while this code lived in DaydreamCore:
                //
                //     Debug (-Onone)  serial 58,033 ms · parallelMap 18,852 ms
                //     Release (-O)    serial  1,875 ms · parallelMap    301 ms
                //
                // That is the "intense hanging on first launch" report: the host bakes the
                // library before the first frame, so a Debug build — which is what running
                // from Xcode gives you — spent ~45 s on a splash screen a Release build
                // clears in well under a second. Debug is the config these apps are
                // DEVELOPED in, so it is the config launch has to be usable in.
                //
                // Safe: this target contains zero `assert()` calls, and `-O` (unlike
                // `-Ounchecked`) keeps every precondition, bounds and overflow trap.
                //
                // `unsafeFlags` is permitted because BOTH consumers (Daydream Home,
                // Visualizer) depend on this package by PATH. It is deliberately absent from
                // the repo-root umbrella manifest, which exists so a third party can depend
                // on this repository by git URL — SwiftPM rejects unsafeFlags there. Same
                // two-manifest split `VisualizerRendering` already uses.
                .unsafeFlags(["-O"], .when(configuration: .debug)),
            ]
        ),
        .testTarget(
            name: "VisualizerMaterialsTests",
            dependencies: ["VisualizerMaterials"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
