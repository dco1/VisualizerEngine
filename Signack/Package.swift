// swift-tools-version: 6.0
import PackageDescription

// Signack — the Illuminatorama engine's Nanite-class geometry system (Daydream DH-1461).
//
// Goal: cost that tracks PIXELS, not triangles. No hand-made LOD chains, no flat per-mesh
// triangle cap that thins a canopy evenly. Built in phases, each with its own directory here:
//
//   Budget/  S0 — spend a triangle budget by visibility importance (this phase)
//   LOD/     S0 — pick among pre-baked budgets by projected size (cheap, per frame, no rebake)
//   Mesh/    S0 — small mesh utilities the later phases build on (vertex welding)
//   Cluster/ Simplify/ DAG/ Select/ Debug/  S1 — cluster LOD DAG, simplifiers, CPU reference selection
//   SignackGPU (target)  S2 — the GPU cut pass: per-cluster select + frustum cull + compaction
//                         into indirect draws (Metal; `Sources/SignackGPU`, own Shaders/)
//   (S3 VisBuffer/ · S4 Raster/ · S5 Foliage/ · S6 Scatter/ — not yet)
//
// WHY ITS OWN PACKAGE: Danny, 2026-10-21 — "None of this 12,000 line stuff. Make it proper."
// Signack is not a corner of `IlluminatoramaRenderer.swift`; it is a module with a small public
// API, small single-purpose files, and a thin seam into its hosts.
//
// The CPU half (S0/S1) depends on NOTHING but `simd`, so it is testable GPU-free under a bare
// `swift test` and reachable from the headless `DaydreamCore` as well as the app. Same macOS 14
// floor as VisualizerMaterials / VisualizerVegetation for the same reason: a dependency cannot
// raise its dependent's floor. The GPU phase lives in the sibling `SignackGPU` target (Metal only —
// it does NOT depend on VisualizerRendering; the host glues its draw to the engine), so the CPU
// target never drags Metal in.
let package = Package(
    name: "Signack",
    platforms: [.macOS(.v14), .iOS(.v18), .tvOS(.v18)],
    products: [
        .library(name: "Signack", targets: ["Signack"]),
        .library(name: "SignackGPU", targets: ["SignackGPU"]),
    ],
    targets: [
        .target(
            name: "Signack",
            swiftSettings: [
                .swiftLanguageMode(.v6),
                // Optimise in Debug too — the ranking walks tens of thousands of leaf cards
                // before a tree's first frame. Same rule as VisualizerVegetation; `-O` keeps
                // every precondition and bounds trap. Path consumers only (the umbrella
                // manifest omits it, as SwiftPM rejects unsafeFlags for a by-URL consumer).
                .unsafeFlags(["-O"], .when(configuration: .debug)),
            ]
        ),
        // S2 — GPU cut selection. Its `.metal` is a `.process` resource: Xcode compiles it into this
        // module's default.metallib; a bare `swift test` copies the source and `SignackGPUPipelines`
        // compiles it at run time (see there).
        .target(
            name: "SignackGPU",
            dependencies: ["Signack"],
            resources: [.process("Shaders")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "SignackTests",
            dependencies: ["Signack", "SignackGPU"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
