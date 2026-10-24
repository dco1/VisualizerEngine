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
        .library(name: "VisualizerStill", targets: ["VisualizerStill"]),
        .library(name: "VisualizerRendering", targets: ["VisualizerRendering"]),
        .library(name: "VisualizerVegetation", targets: ["VisualizerVegetation"]),
        .library(name: "VisualizerFlora", targets: ["VisualizerFlora"]),
        .library(name: "VisualizerHumans", targets: ["VisualizerHumans"]),
        .library(name: "Signack", targets: ["Signack"]),
        .library(name: "SignackGPU", targets: ["SignackGPU"]),
        .library(name: "SignackScatter", targets: ["SignackScatter"]),
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
        // The still/converge quality policy — pure, zero dependencies; see VisualizerStill/Package.swift.
        .target(
            name: "VisualizerStill",
            path: "VisualizerStill/Sources/VisualizerStill",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "VisualizerMaterialsTests",
            dependencies: ["VisualizerMaterials"],
            path: "VisualizerMaterials/Tests/VisualizerMaterialsTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // The shared CPU-side botanical construction engine — one leaf/petal/frond/needle/pad
        // constructor every species is described against. Depends only on VisualizerMaterials so
        // the headless `DaydreamCore` can consume it; see VisualizerVegetation/Package.swift for
        // why it is not a directory inside VisualizerRendering. As with VisualizerMaterials, this
        // umbrella omits that manifest's `-O`-in-Debug `unsafeFlags`.
        .target(
            name: "VisualizerVegetation",
            dependencies: ["VisualizerMaterials"],
            path: "VisualizerVegetation/Sources/VisualizerVegetation",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "VisualizerVegetationTests",
            dependencies: ["VisualizerVegetation"],
            path: "VisualizerVegetation/Tests/VisualizerVegetationTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "VisualizerRendering",
            dependencies: ["VisualizerCore", "VisualizerStill"],
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
            name: "VisualizerFlora",
            dependencies: ["VisualizerMaterials", "VisualizerRendering", "VisualizerVegetation"],
            path: "VisualizerFlora/Sources/VisualizerFlora",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "VisualizerFloraTests",
            dependencies: ["VisualizerFlora"],
            path: "VisualizerFlora/Tests/VisualizerFloraTests",
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
        // Signack — the Nanite-class geometry system (Daydream DH-1461). Pure Swift + simd, no
        // dependencies; see Signack/Package.swift for the phase map and why it is its own package.
        .target(
            name: "Signack",
            path: "Signack/Sources/Signack",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "SignackTests",
            dependencies: ["Signack", "SignackGPU"],
            path: "Signack/Tests/SignackTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "SignackGPU",
            dependencies: ["Signack"],
            path: "Signack/Sources/SignackGPU",
            resources: [.process("Shaders")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // S6 — rule-driven scatter (PCG-equivalent): rules → seeded plant points. Pure Swift + simd.
        .target(
            name: "SignackScatter",
            path: "Signack/Sources/SignackScatter",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "SignackScatterTests",
            dependencies: ["SignackScatter"],
            path: "Signack/Tests/SignackScatterTests",
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
