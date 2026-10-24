// swift-tools-version: 6.0
import PackageDescription

// VisualizerStill — what a STILL (and a settled canvas) is worth: the measured quality table
// (`PhotoQuality`: supersample, megapixel cap, output size, jitter, 1/k accumulation), the export
// tiers (`PhotoGIQuality`), the output-size contract, the converge-and-hold state machine
// (`ConvergenceLatch`) and the adaptive-still statistics (`AdaptiveStill`).
//
// Pure Swift numerics with ZERO dependencies, for the same reason as VisualizerMaterials:
// `DaydreamCore` is headless and must keep building under a bare `swift test`, while
// `VisualizerRendering` (Metal) feeds these numbers to `IlluminatoramaRenderer.renderStill`. Neither can
// import the other, so the policy lives below both. macOS 14 floor to match DaydreamCore.
let package = Package(
    name: "VisualizerStill",
    platforms: [.macOS(.v14), .iOS(.v18), .tvOS(.v18)],
    products: [
        .library(name: "VisualizerStill", targets: ["VisualizerStill"]),
    ],
    targets: [
        .target(name: "VisualizerStill", swiftSettings: [.swiftLanguageMode(.v6)]),
    ]
)
