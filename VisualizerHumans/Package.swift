// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VisualizerHumans",
    platforms: [.macOS(.v15), .iOS(.v18), .tvOS(.v18)],
    products: [
        .library(name: "VisualizerHumans", targets: ["VisualizerHumans"]),
    ],
    dependencies: [
        .package(path: "../VisualizerRendering"),
    ],
    targets: [
        .target(
            name: "VisualizerHumans",
            dependencies: ["VisualizerRendering"],
            // Resources holds HumanDataset.bin — the baked CC0 MakeHuman body
            // dataset (mesh + macro morphs + skeleton + skin weights) produced
            // by tools/humans/import_makehuman.py.
            resources: [.process("Resources")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "VisualizerHumansTests",
            dependencies: ["VisualizerHumans"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
