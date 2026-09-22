// swift-tools-version: 6.0
import PackageDescription

// VisualizerVegetation — the shared CPU-side botanical construction engine.
//
// ONE leaf/petal/frond/needle/pad constructor that every species in every consuming
// app is described *against*, instead of each plant forking its own strip-stitch. This
// is the module docs/VEGETATION_SHARED_ARCHITECTURE.md specified in 2026-06 and that
// never got built — the gap that let two independent leaf-card systems grow (DH-0651).
//
// WHY THIS IS ITS OWN PACKAGE AND NOT `VisualizerRendering/Vegetation/`, which is where
// the 2026-06 plan put it: `DaydreamCore` is a headless package that must keep building
// and testing under a bare `swift test`, and it is a REQUIRED consumer — the potted
// plants, garden plants and planters all generate their geometry there. Depending on
// `VisualizerRendering` would (a) hard-error, because it declares `.macOS(.v15)` against
// DaydreamCore's `.v14` floor, and (b) put Metal + SceneKit + AppKit + SwiftUI
// underneath a package whose defining property is not having them. So the CPU geometry
// lives here, at the VisualizerMaterials floor, reachable from BOTH the headless core
// and the Metal renderer. The per-frame Metal lifecycle (`VegetationRenderSet`,
// `GrassRenderSet`) lives in `VisualizerRendering/Vegetation/`, where it belongs — the
// split is by "needs a GPU", not by subject.
//
// Same platform-floor rule, and for the same reason, as VisualizerMaterials: macOS 14,
// one below the rest of the engine, because a dependency cannot raise its dependent's
// floor without forcing an unrelated bump.
let package = Package(
    name: "VisualizerVegetation",
    platforms: [.macOS(.v14), .iOS(.v18), .tvOS(.v18)],
    products: [
        .library(name: "VisualizerVegetation", targets: ["VisualizerVegetation"]),
    ],
    dependencies: [
        .package(path: "../VisualizerMaterials"),
    ],
    targets: [
        .target(
            name: "VisualizerVegetation",
            dependencies: [.product(name: "VisualizerMaterials", package: "VisualizerMaterials")],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                // OPTIMIZE EVEN IN DEBUG — same rule, same reason, as VisualizerMaterials and
                // DaydreamCore, whose manifests carry the measurement. This target is the same
                // shape of code: tight `Vec3` loops stitching thousands of leaf cards per plant,
                // run before the first frame is drawn. `-O` (unlike `-Ounchecked`) keeps every
                // precondition, bounds and overflow trap, and this target has zero `assert()`s.
                //
                // `unsafeFlags` is permitted because every consumer depends by PATH. It is
                // deliberately absent from the repo-root umbrella manifest (SwiftPM rejects it
                // for a by-URL consumer) — the same two-manifest split the sibling packages use.
                .unsafeFlags(["-O"], .when(configuration: .debug)),
            ]
        ),
        .testTarget(
            name: "VisualizerVegetationTests",
            dependencies: ["VisualizerVegetation"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
