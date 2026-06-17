# VisualizerEngine

Real-time **GPU simulation + rendering** for SceneKit on Apple Silicon, in two
Swift packages:

| Package                | What it is                                                            |
| ---------------------- | -------------------------------------------------------------------- |
| **`VisualizerRendering`** | The engine: GPU solvers (XPBD, MLS-MPM fluid, SPH foam, CoinDEM rigid-body, marching cubes), GPU→SceneKit mesh bridges, and the **Illuminatorama** deferred-PBR + image-based-lighting + hardware ray-tracing path. Ships its own Metal kernels and loads them via `Bundle.module`. |
| **`VisualizerCore`**      | The shared substrate `VisualizerRendering` (and host apps) build on: scene-controller / descriptor plumbing, GPU + display capability probes, a keyframe/timeline model, procedural-texture helpers, and a set of deterministic geometry auditors (winding, mesh soundness, spline continuity, z-fight, penetration). |

Originally extracted from a macOS SceneKit app and shaped into something a
third party could drop into their own SceneKit project. The parent app keeps its
scene library private and consumes this repo as a git submodule.

> **Status:** pre-1.0. The public API is usable but not yet frozen.

Full engine documentation — every solver, renderer, and the architecture rules
behind them — lives in
[`VisualizerRendering/README.md`](VisualizerRendering/README.md).

---

## Requirements

- macOS 15+
- Swift 6.0+ · Xcode 16+
- An Apple Silicon Mac (kernels target the unified-memory model; Intel +
  discrete-GPU paths aren't tested)

---

## Install

As a Swift Package dependency:

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/dco1/VisualizerEngine.git", from: "0.1.0"),
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "VisualizerRendering", package: "VisualizerEngine"),
        // VisualizerCore is pulled in transitively; depend on it directly only
        // if you use its auditors / timeline / capability probes on their own.
    ]),
]
```

The repo root ships an umbrella `Package.swift` exposing both products. Each
package also keeps its own manifest under `VisualizerCore/` and
`VisualizerRendering/`, so it can equally be consumed as a **local path package**
or **git submodule** (which is how the parent app uses it):

```yaml
# XcodeGen project.yml
packages:
  VisualizerCore:
    path: Packages/VisualizerCore        # submodule mounted at Packages/
  VisualizerRendering:
    path: Packages/VisualizerRendering
```

---

## Quick start

A 12-rope chain hanging under gravity, rendered as swept tubes:

```swift
import SceneKit
import VisualizerRendering

let engine = SimEngine.shared

let solver = PBDSolver(
    engine: engine,
    chainCount: 12,
    nodesPerChain: 32,
    nodeSpacing: 0.04
)!

let tubes = PBDTubeRenderer(solver: solver, radius: 0.012)!
scene.rootNode.addChildNode(tubes.node)

// Step once per frame from your SCNSceneRendererDelegate:
sceneView.delegate = StepDelegate(solver: solver)
```

…where `StepDelegate.renderer(_:updateAtTime:)` calls `solver.step(dt:)`. See
[`VisualizerRendering/README.md`](VisualizerRendering/README.md) for the full
catalogue and the architecture rules.

---

## Unlicense

Public domain ([The Unlicense](https://unlicense.org)).

This was built by an AI using publicly available research code, scraped code, and the public's water and electricity. As such, it should be owned by everyone. It is a shame this even exists. I feel shame for having created it. What wicked part of me was unable to be anything but weak enough to be seduced by AI? And in seduction, all I am left with is shame.
