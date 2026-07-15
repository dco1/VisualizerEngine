// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VisualizerRendering",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "VisualizerRendering", targets: ["VisualizerRendering"]),
    ],
    dependencies: [
        .package(path: "../VisualizerCore"),
    ],
    targets: [
        .target(
            name: "VisualizerRendering",
            dependencies: ["VisualizerCore"],
            // The Shaders/ folder holds the GPU kernels for every solver this
            // package ships (PBD, PBDField, MLS-MPM, Marching Cubes, Foam).
            // `.process` lets SwiftPM hand the .metal files to Xcode's Metal
            // compiler, which produces a `default.metallib` inside the
            // generated `VisualizerRendering_VisualizerRendering` resource
            // bundle. `SimPipelineCache.library(for:)` loads that bundle via
            // `Bundle.module`, so kernels travel with the package instead of
            // needing to be re-added to every host app's main target.
            resources: [.process("Shaders")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "VisualizerRenderingTests",
            dependencies: ["VisualizerRendering"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
