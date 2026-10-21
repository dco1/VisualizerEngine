// swift-tools-version: 6.0
import PackageDescription

// VisualizerFlora — streamed wildflower FIELDS for any app on the engine.
//
// Lifted from Visualizer's Superbloom Hills scene so Daydream Home's yard and that scene grow the
// same field from the same code: the engine's `WildflowerPlantingMesh` library baked once into GPU
// meshes (full + lite LOD), a deterministic per-cell scatter (the same world cell always grows the
// same plants), near plants as instances, mid-distance plants as merged cards derived from the SAME
// variants, all streamed around a camera. It needs BOTH the renderer (meshes, instances) and the
// vegetation library (the plants), which neither of those packages may depend on — hence its own
// target.
let package = Package(
    name: "VisualizerFlora",
    platforms: [.macOS(.v15), .iOS(.v18), .tvOS(.v18)],
    products: [
        .library(name: "VisualizerFlora", targets: ["VisualizerFlora"]),
    ],
    dependencies: [
        .package(path: "../VisualizerMaterials"),
        .package(path: "../VisualizerRendering"),
        .package(path: "../VisualizerVegetation"),
    ],
    targets: [
        .target(
            name: "VisualizerFlora",
            dependencies: [
                .product(name: "VisualizerMaterials", package: "VisualizerMaterials"),
                .product(name: "VisualizerRendering", package: "VisualizerRendering"),
                .product(name: "VisualizerVegetation", package: "VisualizerVegetation"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "VisualizerFloraTests",
            dependencies: ["VisualizerFlora"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
