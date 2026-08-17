// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VisualizerRendering",
    platforms: [.macOS(.v15), .iOS(.v18), .tvOS(.v18)],
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
            swiftSettings: [
                .swiftLanguageMode(.v6),
                // OPTIMIZE EVEN IN DEBUG — same reasoning DaydreamCore already applies
                // to itself (see DaydreamCore/Package.swift), extended to the renderer.
                //
                // This target holds IlluminatoramaRenderer (~9.5k lines) and the whole
                // per-frame encode path: pass construction, instance upload, shadow and
                // G-buffer encoding. Host apps build their Debug configuration by default
                // (`Scripts/run.sh`, Xcode's Run action), so ALL of that ran at `-Onone`
                // in the config the apps are actually developed and judged in. Measured
                // on Daydream Home 2026-08-03: app-target code was ~5x slower in Debug
                // (17.8 ms vs 3.6 ms for a warm structural rebuild), and this package was
                // paying the same tax with no one having noticed.
                //
                // These are real-time renderers judged by feel. A default build several
                // times slower than the shipping one silently distorts every judgement
                // about how the render performs.
                //
                // CAVEAT, deliberately not papered over: `-O` compiles out `assert()`
                // (`precondition` and all bounds/overflow traps survive). This target has
                // four asserts — three solver-mode invariants in CoinDEMSolver, and the
                // IlluminatoramaInstance stride check in IlluminatoramaTypes. They no
                // longer fire in Debug. If any of them is load-bearing, promote it to
                // `precondition` rather than reverting this setting.
                //
                // `unsafeFlags` is permitted because BOTH consumers (Daydream Home,
                // Visualizer) depend on this package by PATH — see the `path:` entries in
                // each project.yml. A versioned/remote consumer would make SwiftPM reject
                // it; swap to a Release build or an xcconfig override then.
                .unsafeFlags(["-O"], .when(configuration: .debug)),
            ]
        ),
        .testTarget(
            name: "VisualizerRenderingTests",
            dependencies: ["VisualizerRendering"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
