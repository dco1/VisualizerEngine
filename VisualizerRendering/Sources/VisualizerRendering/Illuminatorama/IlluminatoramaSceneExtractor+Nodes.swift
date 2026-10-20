import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

extension IlluminatoramaSceneExtractor {

    // ── Emissive-as-light synthesis (Phase 4.27) ───────────────────────────────

    /// Append a synthesised point light for each strongly-emissive instance.
    /// Position is the instance's world-space origin (model matrix translation
    /// — a fair proxy for a glowing object's centre); colour is its emission ×
    /// `emissiveLightGain`. When more than `maxEmissiveLights` qualify, only the
    /// brightest are kept so the per-pixel point-light loop stays bounded.
    func synthesiseEmissiveLights(
        from instances: [IlluminatoramaRenderer.InstanceRef],
        into pointLights: inout [IlluminatoramaPointLight]
    ) {
        var lights: [IlluminatoramaPointLight] = []
        for inst in instances {
            // `lightEmission` is the effective radiance for lighting — solid
            // emission colour OR an emission texture's average, × intensity.
            // (Phase 4.27a; texture-driven glow lights the scene through this.)
            let e = inst.lightEmission
            // Guard against non-finite emission (a material that resolved to
            // NaN/Inf) — a bad light position/colour can abort the render.
            guard e.x.isFinite, e.y.isFinite, e.z.isFinite else { continue }
            let peak = max(e.x, max(e.y, e.z))
            guard peak >= emissiveLightThreshold else { continue }
            let m = inst.data.modelMatrix
            let pos = SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
            guard pos.x.isFinite, pos.y.isFinite, pos.z.isFinite else { continue }
            // Clamp the pre-multiplied colour so a runaway emissive value
            // can't produce an absurd light that blows out the composite.
            let scaled = e * emissiveLightGain
            let cMax = max(scaled.x, max(scaled.y, scaled.z))
            let color = cMax > emissiveLightMaxColor
                ? scaled * (emissiveLightMaxColor / cMax)
                : scaled
            lights.append(IlluminatoramaPointLight(
                position: pos,
                radius: emissiveLightRadius,
                color: color))
        }
        if lights.count > maxEmissiveLights {
            // Luminance-sort descending; keep the brightest N.
            lights.sort { a, b in
                let la = 0.2126 * a.color.x + 0.7152 * a.color.y + 0.0722 * a.color.z
                let lb = 0.2126 * b.color.x + 0.7152 * b.color.y + 0.0722 * b.color.z
                return la > lb
            }
            lights.removeLast(lights.count - maxEmissiveLights)
        }
        pointLights.append(contentsOf: lights)
    }

    // ── Tree walk ─────────────────────────────────────────────────────────────

    func walk(node: SCNNode,
                      parentHidden: Bool,
                      renderer: IlluminatoramaRenderer,
                      instances: inout [IlluminatoramaRenderer.InstanceRef],
                      pointLights: inout [IlluminatoramaPointLight],
                      spotLights: inout [IlluminatoramaSpotLight],
                      areaLights: inout [IlluminatoramaAreaLight],
                      directionalDir: inout SIMD3<Float>?,
                      directionalColor: inout SIMD3<Float>?,
                      extraDirectionals: inout [IlluminatoramaDirectionalLight],
                      ambientFromAmbientLights: inout SIMD3<Float>) {
        let hidden = parentHidden || (respectHidden && node.isHidden)
        if !hidden {
            extractGeometry(node: node, renderer: renderer, into: &instances)
            extractLight(node: node,
                         pointLights: &pointLights,
                         spotLights: &spotLights,
                         areaLights: &areaLights,
                         directionalDir: &directionalDir,
                         directionalColor: &directionalColor,
                         extraDirectionals: &extraDirectionals,
                         ambientFromAmbientLights: &ambientFromAmbientLights)
        }
        // Continue walking even hidden subtrees? No — SceneKit treats a
        // hidden ancestor as hiding the whole subtree, and walking through
        // a hidden helper rig (which often contains thousands of cosmetic
        // children) would balloon the instance count for no visible gain.
        if hidden { return }
        for child in node.childNodes {
            walk(node: child,
                 parentHidden: hidden,
                 renderer: renderer,
                 instances: &instances,
                 pointLights: &pointLights,
                 spotLights: &spotLights,
                 areaLights: &areaLights,
                 directionalDir: &directionalDir,
                 directionalColor: &directionalColor,
                 extraDirectionals: &extraDirectionals,
                 ambientFromAmbientLights: &ambientFromAmbientLights)
        }
    }

    func extractGeometry(node: SCNNode,
                                  renderer: IlluminatoramaRenderer,
                                  into instances: inout [IlluminatoramaRenderer.InstanceRef]) {
        guard let geometry = node.geometry else { return }
        let world = effectiveWorldTransform(of: node)
        let modelMatrix = simdFromSCN(world)

        // AAA glass (#60): a node flagged `illuminatoramaGlass` is routed into the
        // renderer's TLAS glass path (true ray-traced refraction) instead of the
        // deferred opaque pipeline — which can't render transparency honestly (the
        // "transparent PBR pane renders black" gotcha). Register each element's
        // mesh and emit a glass instance; the node never reaches the opaque loop.
        if let glassMat = node.illuminatoramaGlass {
            for elementIndex in 0..<max(1, geometry.elements.count) {
                guard let entry = registerMesh(geometry, elementIndex: elementIndex,
                                               into: renderer) else { continue }
                glassEntriesThisFrame.append((entry.kind,
                    IlluminatoramaGlassInstance(modelMatrix: modelMatrix, material: glassMat)))
            }
            return
        }

        // Phase 4.3 — walk every element / submesh. SCNGeometry.material(at:)
        // wraps SCN's materials array indexing AND its 1-material fallback
        // rule: when `materials.count == 1` it's reused across every
        // element, which is the common case for hand-built meshes; when
        // `materials.count >= elements.count` each element gets its
        // matching material; otherwise we get nil for the out-of-range
        // tail and fall through to the default-material branch.
        // Host-owned GPU particle buffers (Foam spray, Firework bursts) are
        // NOT discovered here — they're published into `SimEngine.particleFields`
        // by their renderer-shim and fed to the renderer in `extractFrame`
        // (filtered to the active scene). The deferred opaque pipeline can't
        // draw `.point` + additive geometry anyway, so such geometry that
        // reaches the walk just falls through the unsupported-mesh path.
        let elementCount = max(1, geometry.elements.count)
        let materials = geometry.materials
        for elementIndex in 0..<elementCount {
            let scnMaterial: SCNMaterial?
            if elementIndex < materials.count {
                scnMaterial = materials[elementIndex]
            } else if materials.count == 1 {
                scnMaterial = materials[0]
            } else {
                scnMaterial = nil
            }
            // Phase 4.18 — skip materials that the deferred opaque
            // pipeline can't reproduce honestly. Three buckets, all of
            // which read as solid-block artefacts when forced through
            // the G-buffer pass:
            //
            //   • `blendMode == .add` — additive emissive volumes
            //     (CityStreet's manhole light beam, every glow shell
            //     scene authors layer over an emissive light source).
            //     These rely on SCN's transparent forward pass adding
            //     to the framebuffer; the deferred pipeline writes
            //     opaque to the G-buffer and the cone shows up as a
            //     solid red wall blocking the rest of the scene.
            //   • `transparency < 1` AND `writesToDepthBuffer == false`
            //     — translucent volumes (water shells, fog cards,
            //     stylised glow domes) that depend on alpha compositing
            //     SCN does for free. Deferred opaque can't fake it.
            //
            // Skipping is the honest fix until Illuminatorama gains a
            // proper transparent/forward pass. Scene authors can opt
            // a node back in via `illuminatoramaOverride` if they
            // want the deferred pass to render it as solid PBR
            // regardless (e.g. by overriding emission to the intended
            // tint).
            if let m = scnMaterial {
                let isAdditive = (m.blendMode == .add)
                let isAlphaVolume = (m.transparency < 0.999
                                     && !m.writesToDepthBuffer)
                if isAdditive || isAlphaVolume {
                    if node.illuminatoramaOverride == nil { continue }
                }
            }
            guard let entry = registerMesh(geometry, elementIndex: elementIndex,
                                            into: renderer) else { continue }
            var material = extractMaterial(scnMaterial, renderer: renderer)
            // Phase 4.14 — per-node material overrides. Lets a scene
            // share one SCNGeometry archetype across many SCNNodes (huge
            // perf win via Phase 4.12 instancing) while keeping per-
            // instance colour / metallic / roughness / emission variety
            // via lightweight `SCNNode.userData` keys. The host stashes
            // overrides on each node; extractor applies them here on
            // top of the geometry's base material.
            applyNodeOverrides(node: node, to: &material)
            var inst = IlluminatoramaInstance(
                modelMatrix: modelMatrix,
                albedo: material.albedo,
                metallic: material.metallic,
                roughness: material.roughness,
                emission: material.emission,
                albedoTextureSlice: material.albedoTextureSlice,
                metallicTextureSlice: material.metallicTextureSlice,
                roughnessTextureSlice: material.roughnessTextureSlice,
                normalTextureSlice: material.normalTextureSlice,
                emissionTextureSlice: material.emissionTextureSlice,
                emissionIntensity: material.emissionIntensity
            )
            // Raster-only node (#60 item 7): keeps its G-buffer draw, gets
            // TLAS mask 0 at AS rebuild (its RT twin is supplied elsewhere,
            // e.g. a registered curve set).
            if node.illuminatoramaOverride?.rtExclude == true { inst.rtExclude = 1 }
            instances.append(.init(meshKind: entry.kind, data: inst,
                                   lightEmission: material.lightEmission))

            // RT soup: gather a lightweight ref + fold a cheap scene hash. The
            // world-space vertex bake is deferred to `finalizeRTGeometry` and
            // only runs when this hash changes (static scenes build the AS once).
            // Normally skipped when the renderer can build a TLAS (RT hardware) —
            // the renderer instances its own meshes, no CPU soup needed. BUT the
            // surface cache (Phase 4.38) rides the soup primitive AS, so when it's
            // enabled we DO gather the soup even on TLAS hardware.
            if renderer.buildRTFromExtractedScene && renderer.rtSupported
                && (!renderer.rtTLASSupported || renderer.surfaceCacheEnabled) {
                let key = RTSoupKey(geom: ObjectIdentifier(geometry), elem: elementIndex)
                rtFrameRefs.append((geom: geometry, elem: elementIndex,
                                    model: modelMatrix, albedo: material.albedo, key: key))
                // Fold translation AND the rotation/scale columns so an object
                // spinning or scaling in place (origin unchanged) still dirties
                // the hash and triggers a rebuild — not just one that moves.
                let c0 = modelMatrix.columns.0, c1 = modelMatrix.columns.1
                let t = modelMatrix.columns.3
                rtSceneHash = rtSceneHash &* 31 &+ key.hashValue
                rtSceneHash = rtSceneHash &+ Int((t.x * 97 + t.y * 89 + t.z * 83
                    + c0.x * 53 + c0.y * 47 + c0.z * 43
                    + c1.x * 41 + c1.y * 37 + c1.z * 31) * 64)
            }
        }
    }

    private func extractLight(node: SCNNode,
                              pointLights: inout [IlluminatoramaPointLight],
                              spotLights: inout [IlluminatoramaSpotLight],
                              areaLights: inout [IlluminatoramaAreaLight],
                              directionalDir: inout SIMD3<Float>?,
                              directionalColor: inout SIMD3<Float>?,
                              extraDirectionals: inout [IlluminatoramaDirectionalLight],
                              ambientFromAmbientLights: inout SIMD3<Float>) {
        guard let light = node.light else { return }
        let intensity = Float(light.intensity) * lightIntensityScale
        let color = (scnContentsToRGB(light.color) ?? SIMD3(repeating: 1)) * intensity
        switch light.type {
        case .directional:
            // SCNLight directional points down -Z in its own space (same as
            // SCNCamera). World direction TO the light = +Z of the node.
            let world = effectiveWorldTransform(of: node)
            let dirToLight = simd_normalize(SIMD3<Float>(-Float(world.m31),
                                                         -Float(world.m32),
                                                         -Float(world.m33)))
            // First directional becomes the shadow-casting primary —
            // the deferred pipeline has one cascade rig and we hand it
            // to the first one we see (SCN convention is "key first").
            // #60 task 5: subsequent directionals (fill / back) are carried
            // as REAL directional lights in `extraDirectionals` and shaded
            // with the same NdotL + GGX-specular BRDF the primary uses (full
            // colour, not the old 0.5× ambient fold). They don't get an
            // individual cascade rig — SCN fill/back lights ship
            // `castsShadow = false`, so this is the complete best-version for
            // real 3-point rigs. (A scene with >1 *shadow-casting* directional
            // would need an N-cascade rig — declared out of scope; no current
            // scene has one.)
            if directionalDir == nil {
                directionalDir = dirToLight
                directionalColor = color
            } else {
                extraDirectionals.append(
                    IlluminatoramaDirectionalLight(dir: dirToLight, color: color))
            }
        case .omni:
            let world = effectiveWorldTransform(of: node)
            let pos = SIMD3<Float>(Float(world.m41), Float(world.m42), Float(world.m43))
            // SceneKit's `attenuationEndDistance` is the cutoff; map it to
            // the renderer's `radius`. 0 (the SCNLight default) is "no
            // attenuation, infinite radius" — clamp to something finite so
            // the renderer's window function still works.
            let radius = Float(light.attenuationEndDistance > 0
                                 ? light.attenuationEndDistance
                                 : 25.0)
            pointLights.append(.init(position: pos, radius: radius, color: color))
        case .spot:
            // SCNLight `.spot` cones extend along the node's local -Z axis
            // (same convention as `.directional` and `SCNCamera`). Apex
            // sits at the node's world position; the cone opens in the
            // -Z direction. `SCNMatrix4` is column-major, so the third
            // column (m31/m32/m33) is the local +Z axis in world space;
            // light travels along the NEGATIVE of that — pass the
            // negated vector so the lighting kernel's `dot(direction, -L)`
            // gives the cosine relative to the actual cone axis.
            let world = effectiveWorldTransform(of: node)
            let pos = SIMD3<Float>(Float(world.m41), Float(world.m42), Float(world.m43))
            let dirLightTravels = simd_normalize(SIMD3<Float>(-Float(world.m31),
                                                               -Float(world.m32),
                                                               -Float(world.m33)))
            // `spotInnerAngle` / `spotOuterAngle` are SceneKit FULL cone
            // angles in degrees. The lighting kernel compares the dot
            // product of the cone axis against the light-to-fragment
            // direction, so we store the cosines of the HALF-angles.
            let deg2rad = Float.pi / 180.0
            let innerCos = cos(Float(light.spotInnerAngle) * 0.5 * deg2rad)
            let outerCos = cos(Float(light.spotOuterAngle) * 0.5 * deg2rad)
            // Same fallback for `attenuationEndDistance == 0` ("infinite")
            // as omni — clamp to something finite for the windowing math.
            let radius = Float(light.attenuationEndDistance > 0
                                 ? light.attenuationEndDistance
                                 : 25.0)
            spotLights.append(.init(position: pos,
                                     direction: dirLightTravels,
                                     innerCone: innerCos,
                                     outerCone: outerCos,
                                     color: color,
                                     radius: radius))
        case .area:
            // #60 task 5 — closed-form rectangular area light, retiring the 4.24
            // five-spot approximation (one centre + four corner spots, which read
            // as five discrete speculars on a glossy surface instead of one
            // elongated area reflection). The node's local +X / +Y axes span the
            // emitting plane; `areaExtents.xy` are the half-width / half-height in
            // metres (centre-relative). We pass the half-EDGE vectors `ex` / `ey`
            // (unit axis × half-extent) so the kernel's corners are simply
            // `centre ± ex ± ey`; the diffuse term is then the exact polygon
            // clamped-cosine integral (LTC with M = identity) and the specular a
            // most-representative-point sample.
            let world = effectiveWorldTransform(of: node)
            let localX = simd_normalize(SIMD3<Float>(Float(world.m11),
                                                       Float(world.m12),
                                                       Float(world.m13)))
            let localY = simd_normalize(SIMD3<Float>(Float(world.m21),
                                                       Float(world.m22),
                                                       Float(world.m23)))
            let extents = light.areaExtents
            let halfX: Float = max(0.02, extents.x)
            let halfY: Float = max(0.02, extents.y)
            let centre = SIMD3<Float>(Float(world.m41),
                                        Float(world.m42),
                                        Float(world.m43))
            let radius = Float(light.attenuationEndDistance > 0
                                 ? light.attenuationEndDistance
                                 : 25.0)
            // SCN `.area` lights emit from one face (the local −Z side); the
            // kernel's one-sided gate uses the +normal = cross(ex, ey) face, so
            // order the edges to make that normal point the way the light travels.
            areaLights.append(.init(center: centre,
                                     ex: localX * halfX,
                                     ey: localY * halfY,
                                     color: color,
                                     radius: radius,
                                     twoSided: false))
        case .ambient:
            // Genuine SCN ambient lights → the scene-wide ambient supplement.
            // They illuminate all surfaces equally (no NdotL). The 0.5× scale
            // keeps the unmodulated `mix(0.4·amb, amb, upness)·albedo` term in
            // the lighting kernel from overshooting. (As of #60 task 5 this term
            // is fed ONLY by `.ambient` lights — secondary directionals now
            // shade with a real BRDF via `extraDirectionals`.)
            ambientFromAmbientLights += color * 0.5
        default:
            // IES, probe — not extracted yet. Skip silently.
            break
        }
    }
}
