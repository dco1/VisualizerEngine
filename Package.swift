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
    platforms: [.macOS(.v15), .iOS(.v18), .tvOS(.v18)],
    products: [
        .library(name: "VisualizerCore", targets: ["VisualizerCore"]),
        .library(name: "VisualizerMaterials", targets: ["VisualizerMaterials"]),
        .library(name: "VisualizerRendering", targets: ["VisualizerRendering"]),
        .library(name: "VisualizerHumans", targets: ["VisualizerHumans"]),
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
        // The CPU-side procedural material engine. Deliberately depends on NOTHING — see
        // VisualizerMaterials/Package.swift. Note this umbrella omits that manifest's
        // `-O`-in-Debug `unsafeFlags`, which SwiftPM forbids for a by-URL consumer; the
        // path-based sub-manifest both apps actually use carries it.
        .target(
            name: "VisualizerMaterials",
            path: "VisualizerMaterials/Sources/VisualizerMaterials",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "VisualizerMaterialsTests",
            dependencies: ["VisualizerMaterials"],
            path: "VisualizerMaterials/Tests/VisualizerMaterialsTests",
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
        .target(
            name: "VisualizerHumans",
            dependencies: ["VisualizerRendering"],
            path: "VisualizerHumans/Sources/VisualizerHumans",
            resources: [.process("Resources")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "VisualizerHumansTests",
            dependencies: ["VisualizerHumans"],
            path: "VisualizerHumans/Tests/VisualizerHumansTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "humans-preview",
            dependencies: ["VisualizerHumans", "VisualizerRendering"],
            path: "tools/humans-preview",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
