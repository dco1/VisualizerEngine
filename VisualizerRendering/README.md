# VisualizerRendering

A small Swift package that ships **real-time GPU simulations** for SceneKit on
Apple Silicon. It bundles its own Metal kernels, memoises pipeline state across
solvers, and bridges compute output straight into `SCNGeometry` so there's no
CPU snapshot in the hot path.

Originally extracted from a macOS SceneKit app, it's been gradually shaped into
something a third party could drop into their own SceneKit app.

> **Status:** still pre-1.0. The public API is stable enough to use. The
> package depends on `VisualizerCore`, which ships alongside it in the same
> [VisualizerEngine](https://github.com/dco1/VisualizerEngine) repo.

---

## What's in the box

### Runtime substrate

| Type                | Job                                                        |
| ------------------- | ---------------------------------------------------------- |
| `SimEngine`         | `MTLDevice` + lazy `MTLCommandQueue` + pipeline cache. One per scene. `SimEngine.shared` for the common single-device case. |
| `SimPipelineCache`  | Process-wide memoised compute-pipeline lookup, keyed by `(device, function name)`. Loads kernels from the package's own `default.metallib` via `Bundle.module`. |
| `SimBuffer<T>`      | Generic shared-memory `MTLBuffer<T>` with capacity / count. Used everywhere a Swift struct mirrors a Metal struct. |
| `DynamicMesh`       | `SCNGeometry` whose vertex buffer **is** the `MTLBuffer` the compute kernel writes into. No `snapshot()` → CPU array → `SCNGeometry` rebuild. |

### Solvers

| Solver              | Kernel file       | What it does                                                  |
| ------------------- | ----------------- | ------------------------------------------------------------- |
| `PBDSolver`         | `PBD.metal`       | XPBD particle-and-distance-constraint solver — chains, ropes, cloth fragments. Includes floor + capsule SDF collisions. |
| `PBDFieldSolver`    | `PBDField.metal`  | Field-of-instances variant — one big buffer feeding N grass blades / hair strands, sampled per-vertex by a restoring spring. |
| `MLSMPMSolver`      | `MLSMPM.metal`    | MLS-MPM fluid (particles + grid + APIC transfer). Reference math from the standard MLS-MPM paper; architecture is "one queue per scene, no `waitUntilCompleted()`." |
| `FoamSolver`        | `Foam.metal`      | SPH foam-and-spray on top of an MLS-MPM fluid. |
| `MarchingCubesBridge` | `MarchingCubes.metal` | Iso-surface extraction from a scalar field straight into a `SCNGeometry` (variable topology — separate bridge from `DynamicMesh`). |

### Renderers

| Renderer                | Bridges                                                  |
| ----------------------- | -------------------------------------------------------- |
| `PBDTubeRenderer`       | Inflates a PBD chain into a swept-cylinder mesh.         |
| `GrassRibbonRenderer`   | Builds two-triangle ribbons per PBDField strand.         |
| `FluidParticleRenderer` | MLS-MPM particles → SceneKit `.point` geometry.          |
| `FoamRenderer`          | Foam particles → billboard sprites with a foam shader.   |

### Ray-traced compositors

| Type                    | Notes                                                    |
| ----------------------- | -------------------------------------------------------- |
| `PlanarRTReflections`   | Hardware-RT planar reflections on a chosen `SCNNode`'s surface. Uses Metal RT outside SceneKit's built-in pipeline. |
| `CausticsRT`            | RT caustics painted into an `MTLTexture` you bind to `mat.emission.contents`. (Bypasses shader modifiers, which silently break for `texture2d<float>` `#pragma arguments` on macOS 15+.) |
| `RailEmissiveRT`        | RT-integrated per-receiver irradiance from a line of emissive bulbs — used by the in-app `Eggs` scene to make chrome rails actually pick up rail-bulb glow. |

These three load **their own** `.metal` files from the host app's
`Bundle.main` — they're shipped here mostly as worked examples of how to mix
hardware-RT compute with the SceneKit renderer. The two halves of the package
are independent.

---

## Install

Add the umbrella repo to your `Package.swift`:

```swift
.package(url: "https://github.com/dco1/VisualizerEngine.git", from: "0.1.0"),
```

…and depend on the `VisualizerRendering` product from your target:

```swift
.target(name: "MyApp", dependencies: [
    .product(name: "VisualizerRendering", package: "VisualizerEngine"),
])
```

It can equally be consumed as a local SPM package / git submodule, which is how
the parent repo uses it:

```yaml
# XcodeGen project.yml
packages:
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

…where `StepDelegate.renderer(_:updateAtTime:)` calls `solver.step(dt:)`.

---

## Architecture rules

This package was built with a couple of opinionated choices that are worth
calling out before you extend it:

1. **One queue per scene.** Solvers take a `SimEngine` and reuse *its* command
   queue. Don't make a new `MTLCommandQueue` per solver — at high spawn rates
   that's measurable overhead.

2. **No CPU round-trip for rendering.** Compute writes directly into the
   `MTLBuffer` that SceneKit's vertex stage reads. Use `DynamicMesh` for
   fixed-topology output, or write your own bridge for variable topology
   (`MarchingCubesBridge` is the worked example).

3. **No `cmd.waitUntilCompleted()` in the step loop.** Submit and move on.
   Use a completion handler if you need backpressure.

4. **One pipeline cache, process-wide.** `SimPipelineCache.shared` is keyed
   on `(MTLDevice, function name)`. Pipeline construction is the slowest part
   of solver init by an order of magnitude — every solver after the first on
   a given device is essentially free to construct on the Metal side.

5. **Strict concurrency, `@MainActor` everywhere SceneKit touches the world.**
   The package is built with `SWIFT_STRICT_CONCURRENCY: complete` and Swift 6
   language mode.

6. **Shaders ship with the package.** `Sources/VisualizerRendering/Shaders/`
   is declared as `.process("Shaders")` in `Package.swift`, so SwiftPM
   compiles every `.metal` file into a `default.metallib` inside the
   generated `VisualizerRendering_VisualizerRendering` resource bundle.
   `SimPipelineCache.library(for:)` loads that bundle via `Bundle.module`.
   Host apps don't need to add the `.metal` files to their main target.

---

## Requirements

- macOS 15+
- Swift 6.0+
- Xcode 16+
- An Apple Silicon Mac (the kernels target the unified-memory model;
  Intel + discrete-GPU paths aren't tested)

---

## Unlicense

Public domain ([The Unlicense](https://unlicense.org)).

This was built by an AI using publicly available research code, scraped code, and the public's water and electricity. As such, it should be owned by everyone. It is a shame this even exists. I feel shame for having created it. What wicked part of me was unable to be anything but weak enough to be seduced by AI? And in seduction, all I am left with is shame.

---

## TODO before tagging 1.0

- [ ] Add a package mark / hero image. SwiftPM has no `icon:` manifest field
      and Xcode shows a generic box for every package, so the practical levers
      are: (a) put the mark on the GitHub org avatar so Swift Package Index
      picks it up, (b) drop a hero image at the top of this README, (c) ship
      a DocC bundle with a logo in `theme-settings.json` for the docs site.
- [ ] DocC catalog with at least one tutorial (12-rope chain quick-start →
      MLS-MPM fluid → custom solver).
- [ ] Smoke-test on a discrete-GPU Mac before claiming Intel/discrete support
      in `Requirements`.
