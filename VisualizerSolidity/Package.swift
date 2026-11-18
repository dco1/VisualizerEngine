// swift-tools-version: 6.0
import PackageDescription

// VisualizerSolidity — "is every element a real, solid thing?" for Mesh3 geometry.
// The auditors (MeshSoundness, ZFightDetector, PenetrationProbing) live in VisualizerCore, which knows no
// meshes; Mesh3 lives in VisualizerMaterials, which depends on nothing. This tiny package is the bridge
// that welds a Mesh3, turns its faces into SurfacePatches and runs the auditors, so every app's solidity
// tests share one implementation instead of each copying it.
let package = Package(
    name: "VisualizerSolidity",
    platforms: [.macOS(.v15), .iOS(.v18), .tvOS(.v18)],
    products: [
        .library(name: "VisualizerSolidity", targets: ["VisualizerSolidity"]),
    ],
    dependencies: [
        .package(path: "../VisualizerCore"),
        .package(path: "../VisualizerMaterials"),
    ],
    targets: [
        .target(
            name: "VisualizerSolidity",
            dependencies: [
                .product(name: "VisualizerCore", package: "VisualizerCore"),
                .product(name: "VisualizerMaterials", package: "VisualizerMaterials"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "VisualizerSolidityTests",
            dependencies: ["VisualizerSolidity"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
