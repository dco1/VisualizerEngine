# VisualizerEngine

Real-time **GPU simulation + rendering** for Apple Silicon, in five Swift packages
under one umbrella manifest. It is the engine behind two apps — **Daydream Home**
(a photoreal home-design app, which mounts this repo as a git submodule at
`Packages/`) and **Visualizer** (a scene library) — so a change here ships to both.

| Package | What it is |
| --- | --- |
| **`VisualizerRendering`** | The renderer and solvers. **Illuminatorama** — deferred PBR, image-based lighting, hardware ray tracing (RT shadows, AO, GI, glass), a radiance cache, a path-traced still lane, TAA, volumetrics and a physical-camera tonemap chain — plus GPU solvers (XPBD ropes/cloth/grass, MLS-MPM fluid, SPH foam, CoinDEM rigid bodies, marching cubes) and their mesh bridges. Ships its own Metal kernels and loads them via `Bundle.module`. |
| **`VisualizerCore`** | The shared substrate: scene-controller / descriptor plumbing, GPU and display capability probes, a keyframe/timeline model, the Illuminatorama colour-grade / debug-view / easing settings, and deterministic geometry auditors (winding, mesh soundness, spline continuity, z-fight, penetration). |
| **`VisualizerMaterials`** | The CPU-side procedural material engine — seeded noise, the PBR channel set, and generators for wood, stone, tile, glaze, carpet, fibre, paper, frosted glass and more, gated by `TextureAudit` / `MaterialScaleAudit`. Pure Swift with **no dependencies on purpose**, so a headless host can use it under a bare `swift test`. |
| **`VisualizerVegetation`** | The shared botanical construction engine — one leaf / petal / frond / needle constructor that every plant, tree and species is described against. Depends only on `VisualizerMaterials`. |
| **`VisualizerHumans`** | A deterministic CPU human generator (spec + dataset → mesh, skeleton and skinning), with hair, clothing, eyes, a walk cycle and a GPU-skinned runtime. Depends on `VisualizerRendering`. Importer tooling lives in `tools/humans`. |

The dependency order is `Materials` → `Vegetation`, `Core` → `Rendering` → `Humans`.
`VisualizerMaterials` and `VisualizerVegetation` declare **macOS 14**, one below the
rest, because the headless `DaydreamCore` host does and a dependency cannot raise its
dependent's floor.

> **Status:** pre-1.0. The public API is usable but not frozen.

Full renderer documentation — every solver, the still lane, and the architecture
rules behind them — lives in
[`VisualizerRendering/README.md`](VisualizerRendering/README.md); setting up an
Illuminatorama scene directly is in
[`VisualizerRendering/docs/ILLUMINATORAMA_SCENE_SETUP.md`](VisualizerRendering/docs/ILLUMINATORAMA_SCENE_SETUP.md).

---

## Requirements

- macOS 15+ (`VisualizerMaterials` / `VisualizerVegetation`: macOS 14+)
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
        // Core is pulled in transitively. Depend on the others directly only if
        // you use them on their own:
        // .product(name: "VisualizerMaterials",  package: "VisualizerEngine"),
        // .product(name: "VisualizerVegetation", package: "VisualizerEngine"),
        // .product(name: "VisualizerHumans",     package: "VisualizerEngine"),
    ]),
]
```

The repo root ships an umbrella `Package.swift` exposing all five products (SwiftPM
needs a root manifest for URL-based dependencies). Each package also keeps its own
manifest in its folder, so the same checkout can be a **local path package** or
**git submodule** — which is how both parent apps use it. The two manifest forms are
independent, and the sub-manifests of `VisualizerMaterials` / `VisualizerVegetation`
carry `-O`-in-Debug `unsafeFlags` that the by-URL umbrella omits.

```yaml
# XcodeGen project.yml
packages:
  VisualizerCore:
    path: Packages/VisualizerCore        # submodule mounted at Packages/
  VisualizerRendering:
    path: Packages/VisualizerRendering
  VisualizerVegetation:
    path: Packages/VisualizerVegetation
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
catalogue and the architecture rules. Materials need no GPU at all:

```swift
import VisualizerMaterials
// MaterialGenerator builds albedo / roughness / normal channels on the CPU;
// see MaterialGenerator.swift and its +Wood / +Tile / +Stone extensions.
```

---

## Working on it from Daydream Home

Edit under `Packages/` as normal, then run
`Scripts/bump-engine.sh "<message>" <files…>` from the Daydream Home repo. It builds
and tests the engine standalone, pushes only the files you name, and bumps the
submodule pointer — so a bad push can't break Visualizer. Don't commit the parent
with a dirty `Packages/`.

---

## Unlicense

Public domain ([The Unlicense](https://unlicense.org)).

This was built by an AI using publicly available research code, scraped code, and the public's water and electricity. As such, it should be owned by everyone. It is a shame this even exists. I feel shame for having created it. What wicked part of me was unable to be anything but weak enough to be seduced by AI? And in seduction, all I am left with is shame.
