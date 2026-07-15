// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VisualizerCore",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "VisualizerCore", targets: ["VisualizerCore"])
    ],
    targets: [
        .target(
            name: "VisualizerCore",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "VisualizerCoreTests",
            dependencies: ["VisualizerCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
