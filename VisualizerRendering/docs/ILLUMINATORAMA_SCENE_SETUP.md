# Setting up an Illuminatorama scene

How to drive the **Illuminatorama** Metal renderer directly — build geometry, give it
materials, light it, and get a frame — without going through `SCNScene` /
`IlluminatoramaSceneExtractor`. This is the path you want when your geometry comes from
your own model (a procedural house, a CAD kernel, a game world) rather than a SceneKit
graph.

> **Working references.** The canonical, battle-tested examples live in the Visualizer
> app: `Visualizer/Scenes/IlluminatoramaRoom/IlluminatoramaRoomController.swift` and
> `…/IlluminatoramaHouse/IlluminatoramaHouseController.swift`. When in doubt, mirror
> those. This doc distills their pattern and the non-obvious traps.

---

## The mental model

Each frame you hand the renderer three things and ask it to draw:

1. **Instances** — `renderer.instances: [InstanceRef]`. One `InstanceRef` = *one mesh
   kind* + *one `IlluminatoramaInstance`* (a transform + PBR material). This is the unit
   of drawing.
2. **Lights + environment** — a directional sun, ambient, point/spot/area lights, and an
   **`equirectSky`** environment texture that feeds image-based lighting (IBL).
3. **Camera** — an `IlluminatoramaCamera` (position/target/up/fov/aspect/near/far).

Then `renderer.render(blocking:)` composites everything into `renderer.outputTexture`.

---

## Minimal end-to-end example

```swift
import VisualizerRendering
import Metal
import simd

@MainActor
final class MyScene {
    let renderer: IlluminatoramaRenderer
    private let sky: VolumetricCloudRenderer
    private var handles: [IlluminatoramaMeshHandle] = []   // ⚠️ MUST retain — see Gotcha 1

    init() throws {
        let camera = IlluminatoramaCamera(
            position: SIMD3(8, 7, 11), target: SIMD3(3, 1, -2),
            up: SIMD3(0, 1, 0), fovYRadians: .pi / 3,
            aspect: 1280.0 / 800.0, zNear: 0.05, zFar: 300)
        renderer = try IlluminatoramaRenderer(engine: .shared, width: 1280, height: 800,
                                              camera: camera)

        // ⚠️ Without an environment, the lighting pass renders MAGENTA — see Gotcha 3.
        sky = VolumetricCloudRenderer(engine: .shared, resolution: SIMD2(2048, 1024))
        renderer.equirectSky = sky.outputTexture
        renderer.autoExposureTargetEV = -1.05
    }

    func loadGeometry(_ device: MTLDevice) {
        handles.removeAll()
        var refs: [IlluminatoramaRenderer.InstanceRef] = []

        // (a) a custom mesh you built yourself
        let mesh = IlluminatoramaMesh(device: device, vertices: myVertices, indices: myIndices)
        mesh.doubleSided = true                            // ⚠️ winding — see Gotcha 2
        let handle = renderer.registerMesh(mesh)
        handles.append(handle)                             // ⚠️ retain!
        refs.append(.init(meshKind: handle.kind, data: IlluminatoramaInstance(
            modelMatrix: matrix_identity_float4x4,
            albedo: SIMD3(0.82, 0.80, 0.76), metallic: 0, roughness: 0.9)))

        // (b) a built-in primitive, scaled/placed via the model matrix (no mesh to make)
        refs.append(.init(meshKind: .box, data: IlluminatoramaInstance(
            modelMatrix: scaleTranslate(scale: SIMD3(2, 1, 0.2), translate: SIMD3(0, 0.5, -2)),
            albedo: SIMD3(0.5, 0.1, 0.1), metallic: 0, roughness: 0.6)))

        renderer.instances = refs
    }

    /// Call every frame (e.g. from a Timer / CADisplayLink / MTKView delegate).
    func renderFrame() -> Bool {
        // 1) light it
        let sun = normalize(SIMD3<Float>(-0.4, 0.85, 0.5))
        renderer.directionalLightDirection = sun                     // TOWARD the sun
        renderer.directionalLightColor = SIMD3(1.0, 0.96, 0.88) * 3.4 // pre-multiplied HDR
        renderer.ambientColor = SIMD3(0.19, 0.165, 0.13)
        renderer.iblEnabled = true
        renderer.iblIntensity = 1.0
        renderer.exposure = 1.0

        // 2) render the sky each frame (feeds IBL + shows through windows)
        var p = VolumetricCloudRenderer.Params()
        p.cameraPos = renderer.camera.position
        p.sunDir = -sun
        p.skyZenith = SIMD3(0.05, 0.20, 0.74)
        p.skyHorizon = SIMD3(0.28, 0.48, 0.82)
        sky.render(params: p)
        renderer.markIBLDirty()        // re-bake IBL when the sky changes (gate this for perf)

        // 3) composite
        return renderer.render(blocking: false)   // outputTexture valid only when this is true
    }
}

private func scaleTranslate(scale: SIMD3<Float>, translate: SIMD3<Float>) -> simd_float4x4 {
    var m = matrix_identity_float4x4
    m.columns.0.x = scale.x; m.columns.1.y = scale.y; m.columns.2.z = scale.z
    m.columns.3 = SIMD4(translate.x, translate.y, translate.z, 1)
    return m
}
```

Display the result by binding `renderer.outputTexture` to an `MTKView` (the live path the
Visualizer scenes use) or `scene.background.contents` (the SceneKit-composite path).

---

## Geometry: built-in vs. custom meshes

| Need | Use |
| --- | --- |
| Boxes / spheres / a ground plane | Built-in `MeshKind.box` / `.sphere` / `.ground`, sized + placed by the instance's `modelMatrix`. No mesh allocation. |
| Arbitrary geometry | `IlluminatoramaMesh(device:, vertices: [IlluminatoramaVertex], indices: [UInt16] or [UInt32])`, then `renderer.registerMesh(mesh)` → handle, or `renderer.setMesh(stableKind, mesh)` for a kind you control. |

`IlluminatoramaVertex` is `position`, `normal`, `uv`, `tangent` (xyz + handedness in `w`;
zero = "no normal-map data"), and per-vertex `color` (defaults to white, multiplied into
albedo). Build a TBN-consistent tangent if you ship normal maps.

---

## Materials

`IlluminatoramaInstance` carries scalar PBR (`albedo`, `metallic`, `roughness`,
`emission`) **plus** texture-slice indices into two atlases the renderer owns:

- `renderer.albedoAtlas` — **sRGB** (`.bgra8Unorm_srgb`). Albedo + emission (colour data).
- `renderer.nonColorAtlas` — **linear** (`.bgra8Unorm`). Metallic, roughness, normal maps.

```swift
let slice = renderer.albedoAtlas.register(contents: nsImageOrCGImage) ?? -1
instance.albedoTextureSlice = slice          // -1 = use the scalar `albedo` instead
```

`register(contents:)` takes an `NSImage`/`CGImage`, letterboxes it into a square slice
(recording the aspect in `uvScale`), and returns an `Int32` slice index. **Hold a strong
reference to the source image** for the lifetime of the slice — the atlas caches by object
identity. Encode albedo/emission as sRGB; keep data maps (roughness/metallic/normal)
linear.

---

## Lighting & environment

- **Sun:** `directionalLightDirection` points *toward* the sun; `directionalLightColor` is
  pre-multiplied linear HDR (values > 1).
- **Ambient:** `ambientColor` lifts shadows so they don't crush to black.
- **IBL:** set `renderer.equirectSky` to an equirectangular HDR texture (a
  `VolumetricCloudRenderer.outputTexture` is the easy default). Re-render the sky each
  frame and call `renderer.markIBLDirty()` when it changes so the irradiance/specular
  probes re-bake. `iblEnabled` / `iblIntensity` scale the contribution.
- **Extra lights:** push `pointLights` / `spotLights` / `areaLights` arrays (all colours
  pre-multiplied HDR).
- **Exposure:** `autoExposureTargetEV` (in init) biases auto-exposure; `exposure` is a
  per-frame multiplier.

---

## The three gotchas that will cost you an afternoon

### 1. Retain your mesh handles, or nothing draws

`registerMesh` returns an `IlluminatoramaMeshHandle` whose **`deinit` evicts the mesh from
the renderer's draw table** (handle lifetime drives cleanup). If you write
`let handle = renderer.registerMesh(mesh)` inside a loop and don't store it, the mesh is
gone before you ever render — and you get an empty frame. **Keep every handle in a property
for as long as the mesh should draw.**

### 2. The G-buffer culls back faces by winding

The forward/G-buffer pass uses `setCullMode(.back)` with Metal's default **clockwise**
front-facing winding. If your mesh winds counter-clockwise-outward (the common math
convention), every visible triangle is culled and you see only the sky. Either flip your
index winding to match, or set **`mesh.doubleSided = true`** (also the right call for an
architectural cutaway where you want both faces).

### 3. No `equirectSky` ⇒ a uniform MAGENTA frame

This is the big one. If you never set `renderer.equirectSky`, the lighting pass has no
environment to sample and renders a **flat magenta frame** — easy to mistake for "nothing
drew." Set `renderer.equirectSky` (e.g. from a `VolumetricCloudRenderer`) **and render
that sky every frame**. Once the environment is bound, your geometry appears.

> **Debugging tip:** when you verify a render programmatically (reading the output texture
> back), measure **spatial** variation — count distinct pixels across the image. A flat
> magenta frame still spans the full 0–255 byte range, so a naïve min/max check passes on
> a frame that drew nothing.

---

## Per-frame checklist

1. Update the camera (`renderer.camera = …`).
2. Set sun / ambient / IBL knobs.
3. `sky.render(params:)` → `renderer.markIBLDirty()` (when the sky changed).
4. `let ok = renderer.render(blocking:)`.
5. Present `renderer.outputTexture` **only when `ok == true`** (it's triple-buffered and
   promoted to readable after GPU completion).

`render()` handles temporal accumulation (TAA / SVGF GI / AO) internally — no warmup loop
needed, though the first frame after a sky change wants the IBL re-bake. Avoid calling
`renderer.resize(...)` per frame; it reallocates every GPU target and resets accumulation.
