// swift-tools-version: 6.0
import PackageDescription

// Umbrella manifest for the VisualizerEngine repo. It exposes both packages as
// products from the repo root so a third party can depend on this repository by
// git URL (SPM requires a manifest at the repo root for url-based deps).
//
// Each package ALSO keeps its own manifest under VisualizerCore/ and
// VisualizerRendering/, so the same checkout can be consumed as a local path
// package or git submodule (which is how the parent Visualizer app uses it).
// The two manifest forms are independent — nobody reads both at once.
let package = Package(
    name: "VisualizerEngine",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "VisualizerCore", targets: ["VisualizerCore"]),
        .library(name: "VisualizerRendering", targets: ["VisualizerRendering"]),
    ],
    targets: [
        .target(
            name: "VisualizerCore",
            path: "VisualizerCore/Sources/VisualizerCore",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "VisualizerCoreTests",
            dependencies: ["VisualizerCore"],
            path: "VisualizerCore/Tests/VisualizerCoreTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "VisualizerRendering",
            dependencies: ["VisualizerCore"],
            path: "VisualizerRendering/Sources/VisualizerRendering",
            resources: [.process("Shaders")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "VisualizerRenderingTests",
            dependencies: ["VisualizerRendering"],
            path: "VisualizerRendering/Tests/VisualizerRenderingTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
