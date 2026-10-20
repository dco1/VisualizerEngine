import Metal
import simd

/// The bit → (vector, lane) packing of the per-room band gains (S3.5 Stage E).
///
/// ONE place it is written, because THREE uniform blocks carry the table — the
/// deferred frame, the glass pass and the RT-instanced pass — and the whole point of
/// carrying it in all three is that a room seen through a pane, a room seen in a
/// reflection and a room seen directly agree. Three hand-rolled copies of `b >> 2`
/// is exactly how they would stop agreeing. Mirrors the shader-side indexing in
/// `interiorRoomBandGain` (IlluminatoramaSecondary.h).
public enum InteriorRoomGains {
    /// The layer-bit pool size — 32, one per bit of an `IlluminatoramaInstance.layer`.
    public static let capacity = 32

    /// Pack a flat table (index = layer bit) into eight vectors + the enable word.
    /// Disabled, or an empty table, yields all-ones vectors and a zero meta, which the
    /// shader resolve short-circuits to exactly 1.0.
    public static func pack(_ gains: [Float], enabled: Bool)
        -> (v: [SIMD4<Float>], meta: SIMD4<Float>) {
        var v = [SIMD4<Float>](repeating: .one, count: 8)
        guard enabled, !gains.isEmpty else { return (v, .zero) }
        for b in 0..<min(gains.count, capacity) {
            v[b >> 2][b & 3] = max(0, gains[b])
        }
        return (v, SIMD4(1, 0, 0, 0))
    }
}

/// World-space secondary directional light (#60 task 5 — retires the 4.20
/// "ambient fold" that collapsed every directional past the first into a flat
/// hemispheric ambient supplement). `dir` points TOWARD the light (pre-
/// normalized); `color` is the premultiplied linear-HDR intensity. The lighting
/// kernel shades it with the same diffuse + GGX-specular BRDF the primary sun
/// uses — no shadow map (SCN fill/back lights are `castsShadow = false`).
/// Field-for-field mirror of the Metal `DirectionalLight` (SIMD3 + pad, ×2 = 64 B
/// — `float3`/`SIMD3<Float>` are 16-byte-sized, so each (vec + pad) pair is 32 B;
/// the same layout as `IlluminatoramaPointLight`).
public struct IlluminatoramaDirectionalLight {
    public var dir: SIMD3<Float>             // toward the light (world, normalized)
    public var _pad0: Float = 0
    public var color: SIMD3<Float>           // pre-multiplied linear-HDR intensity
    public var _pad1: Float = 0

    public init(dir: SIMD3<Float>, color: SIMD3<Float>) {
        self.dir = dir
        self.color = color
    }
}

/// World-space rectangular area light (#60 task 5 — replaces the 4.24 5-spot
/// `.area` approximation). The rectangle is `center ± ex ± ey`, where `ex`/`ey`
/// are the half-width / half-height EDGE vectors in world space (so the four
/// corners need no extra basis), and the emitting normal is `normalize(cross(ex,
/// ey))`. The lighting kernel evaluates a closed-form polygon clamped-cosine
/// integral for diffuse (exact Lambert form-factor) plus a most-representative-
/// point specular. Field-for-field mirror of the Metal `AreaLight` (4× float4).
public struct IlluminatoramaAreaLight {
    public var center: SIMD3<Float>
    /// 1 = emits from both faces; 0 = front only (the `+normal` hemisphere).
    public var twoSided: Float
    public var ex: SIMD3<Float>              // half-width edge vector (world)
    /// Light-layer mask (was `_pad0` — same 4 bytes, stride unchanged; the rule
    /// PointLight/SpotLight already follow). Default 0xFFFFFFFF ⇒ affects every
    /// fragment — byte-identical for every existing caller. An area light has no
    /// shadow map and no visibility term, so this is its only containment.
    public var layerMask: UInt32 = 0xFFFF_FFFF
    public var ey: SIMD3<Float>              // half-height edge vector (world)
    /// Daydream DH-0989 — for an area light that stands over a real OPENING (a window portal), the
    /// fraction of light its pane STOPS (1 − diffuse transmittance); 0 ⇒ an open hole. Read only by
    /// the path-traced GI lane, which samples the opening as an aperture onto the real sky. Was
    /// `_pad1` — same 4 bytes, stride unchanged; the deferred kernel never reads it.
    public var apertureOpacity: Float = 0
    public var color: SIMD3<Float>           // pre-multiplied intensity
    public var radius: Float                 // distance-falloff range (metres)
    /// DH-0601 — portal VISIBILITY shadow. World → light-space NDC for the depth map
    /// rendered from the portal centre into the shared spot-shadow atlas. Computed by the
    /// renderer each frame (`updateSpotShadows`) from `(center, ex, ey, radius)`; identity
    /// until a slice is assigned. Mirrors `IlluminatoramaSpotLight.shadowMatrix`.
    public var shadowMatrix: simd_float4x4 = matrix_identity_float4x4
    /// Slice into the shared `spotShadowAtlas`. `< 0` ⇒ no map this frame; the light
    /// contributes as fully visible (the exact pre-DH-0601 path — so an area light that
    /// never opts in is byte-identical). Set by the renderer.
    public var shadowSliceIndex: Int32 = -1
    /// 1 ⇒ this area light may claim a shadow-atlas slice (a window portal). 0 ⇒ it never
    /// does — a diffuse cove/softbox that should not occlude (Visualizer, light strips).
    /// Default 0, so every existing caller is unchanged. Mirrors `IlluminatoramaPointLight
    /// .castsShadow`.
    public var castsShadow: Int32 = 0
    /// Daydream DH-0989 — 1 ⇒ this rect stands over a real OPENING onto the sky (a window portal),
    /// not an emitter. The path-traced GI lane samples an aperture as a window onto the real sky
    /// (its fitted `color` only weights the choice between apertures) and takes over its diffuse
    /// share from the deferred pass; an emitter (a light strip, a skylight lens) stays an emitter.
    /// Was `_pad2` — same 4 bytes. Default 0 ⇒ every existing caller is an emitter, unchanged.
    public var isAperture: Float = 0
    public var _pad3: Float = 0

    public init(center: SIMD3<Float>, ex: SIMD3<Float>, ey: SIMD3<Float>,
                color: SIMD3<Float>, radius: Float, twoSided: Bool = false,
                layerMask: UInt32 = 0xFFFF_FFFF, castsShadow: Bool = false) {
        self.center = center
        self.ex = ex
        self.ey = ey
        self.color = color
        self.radius = radius
        self.twoSided = twoSided ? 1 : 0
        self.layerMask = layerMask
        self.castsShadow = castsShadow ? 1 : 0
    }
}

/// A daylight APERTURE for the interior-irradiance gradation (S3.5 Stage D) — a glazed
/// opening's rectangle, reduced to what the per-pixel sky-visibility estimate needs:
/// where it is, which way it faces into the room, and its area. The lighting kernel sums
/// an approximate solid angle over these per interior fragment and grades the interior
/// irradiance BANDS by it, which is what lets a wall be brighter 0.5 m from its window
/// than 8 m from it — the flatness no normal-only term could fix. 32 bytes, mirror of the
/// Metal `InteriorAperture` (buffer 7 of the lighting kernel).
public struct IlluminatoramaInteriorAperture {
    public var center: SIMD3<Float>          // aperture centre (world)
    public var width: Float                  // metres
    public var inward: SIMD3<Float>          // unit normal INTO the room
    public var height: Float                 // metres

    public init(center: SIMD3<Float>, inward: SIMD3<Float>, width: Float, height: Float) {
        self.center = center
        self.inward = inward
        self.width = width
        self.height = height
    }
}

public struct IlluminatoramaPointLight {
    public var position: SIMD3<Float>
    public var radius: Float
    public var color: SIMD3<Float>           // pre-multiplied intensity
    /// Light-layer mask (mirrors the Metal `PointLight.layerMask`, was `_pad`). A
    /// light contributes to a fragment only when `(layerMask & instance.layer) != 0`.
    /// Default `0xFFFFFFFF` ⇒ affects every fragment (byte-identical to the prior
    /// behaviour, so scenes that never touch it — Visualizer — are unchanged).
    public var layerMask: UInt32 = 0xFFFF_FFFF
    // ── Point-light cubemap shadows (opt-in) ──────────────────────────────────
    /// 1 ⇒ this point light casts a real depth-cubemap shadow (blocks at geometry,
    /// passes through openings, casts intra-room object shadows). Default 0 ⇒ NO
    /// shadow map rendered and the lighting kernel takes the exact pre-change path,
    /// so every scene that never sets this (Visualizer) is byte-identical. Was the
    /// former `_padPointShadow0` trailing slot; reinterpreted, so the struct stride
    /// is unchanged. Mirrors the Metal `PointLight.castsShadow`.
    public var castsShadow: UInt32 = 0
    /// Cube-array index (set by the renderer each frame in `updatePointShadows`).
    /// `< 0` ⇒ no cube page for this light this frame — the kernel treats it as fully
    /// visible. The renderer assigns the first `pointShadowCubeCapacity` shadow-casting
    /// lights a page; the rest fall back to unshadowed. Mirrors the Metal
    /// `PointLight.shadowCubeIndex`. Default `-1`.
    public var shadowCubeIndex: Int32 = -1
    /// **The physical size of the lamp at this light** (metres; 0 ⇒ a point, the shipped law).
    /// Read ONLY by the path-traced still lane (`rtPathLampBulbs`), at path vertices: there a lamp's
    /// falloff is `1/max(d² + softRadius², bulbRadius²)` — a luminous SPHERE of this radius is
    /// exactly `1/d²` outside itself, and no surface can sit inside the glass, so this is the honest
    /// finite-bulb law rather than a clamp (Daydream DH-1293/DH-1295). The deferred pass and every
    /// real-time kernel ignore it, so a host that sets it changes nothing off the path lane. Was the
    /// former `_padPointShadow0` (same 4 bytes, stride unchanged). Mirrors the Metal
    /// `PointLight.bulbRadius` / `RTPointLight.bulbRadius`.
    public var bulbRadius: Float = 0
    /// Source-size term for the near-field falloff (was `_padPointShadow1`; reinterpreted, so
    /// the struct stride is unchanged). The lighting kernel attenuates by `1/(d² + softRadius²)`
    /// instead of `1/d²`: a point source of zero size explodes as d→0, so a wall a metre from a
    /// bulb reads as a hard blob. `softRadius` gives the source a finite apparent size — the near
    /// field flattens into a soft halo while the far field stays honest inverse-square. Default 0
    /// ⇒ exactly `1/d²`, byte-identical to the prior behaviour (Visualizer never sets it).
    public var softRadius: Float = 0
    /// DH-0872 — 1 (default) ⇒ visible to the GI/reflection SECONDARY-ray local-light fill
    /// (`secondaryLocalLightFill`) as well as the deferred pass; 0 ⇒ deferred-only. That
    /// secondary path has no occlusion test (falloff + layer-mask only), so a light whose
    /// containment depends on a wall actually blocking it — `nightWindowGlow`: untrapped, no
    /// shadow map, origin just outboard of its own wall, relying on the deferred pass's own
    /// normal-facing/cone-direction rejection — leaks its full, un-occluded, facade-calibrated
    /// brightness onto nearby interior GI bounces. Mirrors the Metal `PointLight.giVisible`.
    /// Daydream DH-1320 — `secondaryOnly` (2): the other way round — the light reaches secondary
    /// hits ONLY and the deferred pass skips it: a stand-in for the light a traced bounce's HIT
    /// receives (a lamp's first bounce, while the traced bounce delivers that bounce to the primary).
    public var giVisible: UInt32 = 1
    public static let secondaryOnly: UInt32 = 2
    /// **The radius of the luminous OPENING the light leaves through** (metres; 0 ⇒ none). Read ONLY
    /// by the path-traced still lane at path vertices, beside `bulbRadius` (`rtPathLampBulbs`): the
    /// falloff is `1/max(d² + max(softRadius, openingRadius)², bulbRadius²)` — `1/(d² + r²)` is the
    /// exact on-axis irradiance of a Lambertian DISC of radius r (a shade's rim, a can's aperture, a
    /// pane), `1/d²` far away and finite at the rim. A host authors it per fixture from the same
    /// geometry it builds the fixture with (Daydream DH-1363); unset, the law is DH-1295's exactly.
    /// Appended after `giVisible` (size 72 → 76): inside the float3-aligned stride's tail padding,
    /// so the stride stays 80 on both sides. Mirrors the Metal `PointLight` / `RTPointLight
    /// .openingRadius` — keep in lockstep.
    public var openingRadius: Float = 0
    /// Daydream DH-1363 inc. 2 — where the opening sits along the light's travel direction, from `position` (m;
    /// a point light has no axis, so this is carried for the layout only and ignored). Size 76 → 80 = the stride.
    public var openingOffset: Float = 0

    public init(position: SIMD3<Float>, radius: Float, color: SIMD3<Float>,
                layerMask: UInt32 = 0xFFFF_FFFF,
                castsShadow: Bool = false, softRadius: Float = 0, giVisible: Bool = true,
                bulbRadius: Float = 0, openingRadius: Float = 0, openingOffset: Float = 0) {
        self.position = position
        self.radius = radius
        self.color = color
        self.layerMask = layerMask
        self.castsShadow = castsShadow ? 1 : 0
        self.softRadius = softRadius
        self.giVisible = giVisible ? 1 : 0
        self.bulbRadius = bulbRadius
        self.openingRadius = openingRadius
        self.openingOffset = openingOffset
    }
}

/// World-space spot light. Cone angles are stored as the cosines of the
/// half-angles so the lighting kernel can compare a `dot()` result
/// directly. `direction` points AWAY from the apex (the way light
/// travels), matching the convention `SCNLight` uses for `.spot`.
/// DH-0540 — the length of one row of the spot-profile table (`setSpotLightProfiles`): samples uniformly over
/// 0..90° off the cone axis. Mirrors `ILLUMI_SPOT_PROFILE_SAMPLES`.
public let illuminatoramaSpotProfileSampleCount = 128

public struct IlluminatoramaSpotLight {
    public var position: SIMD3<Float>
    /// `cos(spotInnerAngle / 2)` — full intensity inside this cone.
    public var innerCone: Float
    /// Daydream DH-0540 — the row of the host's spot-profile table (`setSpotLightProfiles`) this cone's
    /// intensity is shaped by, or −1 for the analytic cone (byte-identical to before the field existed).
    /// Sits in the 12-byte alignment gap between `innerCone` (offset 16) and the float3 `direction`
    /// (offset 32), so the stride stays 192 in Swift and in every Metal mirror (`SpotLight`,
    /// `RTSpotLight`, `VolSpotLight`) — keep all four in lockstep.
    public var profileIndex: Int32 = -1
    public var direction: SIMD3<Float>
    /// `cos(spotOuterAngle / 2)` — zero intensity outside this cone.
    /// `innerCone >= outerCone` since smaller angles → larger cosines.
    public var outerCone: Float
    public var color: SIMD3<Float>            // pre-multiplied intensity
    public var radius: Float
    /// Phase 4.10 — light-space view-projection for the spot's shadow
    /// map. Computed by the renderer each frame from
    /// `(position, direction, outerCone, radius)`. The lighting kernel
    /// projects each fragment into this space, compares against the
    /// stored depth, and modulates the BRDF contribution by the
    /// visibility term.
    public var shadowMatrix: simd_float4x4 = matrix_identity_float4x4
    /// Slice index into `IlluminatoramaRenderer.spotShadowAtlas`. `< 0`
    /// means "no shadow map for this spot" — the spot still contributes
    /// direct lighting, but with full visibility (no occlusion). Set by
    /// the renderer based on the spot's position in the array and
    /// `spotShadowAtlasCapacity`.
    public var shadowSliceIndex: Int32 = -1
    /// Light-layer mask (mirrors the Metal `SpotLight.layerMask`, was `_padSpot0`).
    /// Same masking rule as `IlluminatoramaPointLight`. Default `0xFFFFFFFF` ⇒ affects
    /// every fragment (byte-identical to prior behaviour).
    public var layerMask: UInt32 = 0xFFFF_FFFF
    /// 1 ⇒ this spot may claim a slice of the shadow atlas. 0 ⇒ it NEVER does, however few
    /// spots the scene has — it lights without occluding anything. Default 1, so a scene that
    /// never sets it is byte-identical to the pre-change behaviour. Was `_padSpot1`;
    /// reinterpreted, so the struct stride is unchanged. Mirrors `IlluminatoramaPointLight
    /// .castsShadow`, which has worked this way since the cubemap path landed.
    ///
    /// **Why a flag and not "sort it last".** A slice used to be purely the light's INDEX:
    /// `updateSpotShadows` handed 0..<capacity to the first `capacity` entries. A host that
    /// wanted an unshadowed cone could only push it toward the back of the array, which does
    /// nothing at all while the scene is under capacity — the cone lands under the cap anyway
    /// and is shadowed anyway. Daydream Home shipped exactly that: its lamp up-cones are
    /// declared untrapped, the night hero carries 7 shadow-wanting cones into 8 slices, and the
    /// spare slice went to an up-cone, which then rasterised the lamp's own finial into it and
    /// printed a hard black hole in the middle of that lamp's ceiling pool (measured on the
    /// 2026-08-15 tree: blob/surround 0.740 with the slice, 1.014 without). One lamp had it and
    /// the other did not, decided by nothing but which one inherited the spare.
    public var castsShadow: Int32 = 1
    /// **Bit flag OR-ed into `castsShadow` (DH-0631): the host asserts that no GPU-fed geometry can
    /// enter this cone's frustum.** Shadow maps are light-space, so a parked scene could reuse
    /// them — but the reuse was gated on the WHOLE scene being CPU-visible (no compute-fed or
    /// GPU-written geometry), and a yard's grass field is always live, so every document with a
    /// lawn re-rasterised all its spot slices every frame into a byte-identical atlas. A cone
    /// that sits INSIDE a room (Daydream masks such lights to their room's layer, so they cannot
    /// light an exterior pixel at all) can never have a blade of grass between it and anything it
    /// lights; the host says so with this bit, and `encodeSpotShadowPasses` then reuses that
    /// slice while the CPU-visible scene holds. The bit keeps `castsShadow` non-zero, so every
    /// `!= 0` test still reads "casts"; a host must not set it on a cone with `castsShadow == 0`.
    /// Exterior cones (a deck lamp, a sconce over the lawn) leave it clear and re-render as before.
    public static let castsShadowIgnoresGPUGeometry: Int32 = 2
    /// Source-size term for the near-field falloff (was `_padSpot2`; reinterpreted, so the
    /// struct stride is unchanged — 176). Same rule as `IlluminatoramaPointLight.softRadius`:
    /// the lighting kernel attenuates by `1/(d² + softRadius²)`, so a cone from a finite-size
    /// source flattens into a soft halo near the emitter instead of a hard blob on a nearby
    /// wall. Default 0 ⇒ exactly `1/d²`, byte-identical (Visualizer never sets it).
    public var softRadius: Float = 0
    /// DH-0872 — see `IlluminatoramaPointLight.giVisible`; same field, same reason, same
    /// default. Mirrors the Metal `SpotLight.giVisible` (size grows 176 → 180; stride 192).
    public var giVisible: UInt32 = 1
    /// The physical size of the lamp at this cone's apex — see `IlluminatoramaPointLight
    /// .bulbRadius`; same field, same law, read only by the path-traced still lane. Appended after
    /// `giVisible` (size 180 → 184): it sits inside the float4x4-aligned stride's tail padding, so
    /// the stride stays 192 on both sides. Mirrors the Metal `SpotLight` / `RTSpotLight` /
    /// `VolSpotLight.bulbRadius` — keep all four in lockstep.
    public var bulbRadius: Float = 0
    /// The radius of the luminous OPENING at this cone's apex — see `IlluminatoramaPointLight
    /// .openingRadius`; same field, same law, read only by the path-traced still lane. Appended
    /// after `bulbRadius` (size 184 → 188): still inside the float4x4-aligned stride's tail
    /// padding, so the stride stays 192 on both sides. Mirrors the Metal `SpotLight` /
    /// `RTSpotLight` / `VolSpotLight.openingRadius` — keep all four in lockstep.
    public var openingRadius: Float = 0
    /// Daydream DH-1363 inc. 2 — where the opening disc sits along `direction` from `position` (m; 0 ⇒ at the apex).
    /// The path lane draws a lamp's NEE target on that disc and evaluates the cone's law from it. Appended after
    /// `openingRadius` (size 188 → 192 = the stride). Mirrors `SpotLight` / `RTSpotLight` / `VolSpotLight.openingOffset`.
    public var openingOffset: Float = 0

    public init(position: SIMD3<Float>, direction: SIMD3<Float>,
                innerCone: Float, outerCone: Float,
                color: SIMD3<Float>, radius: Float,
                layerMask: UInt32 = 0xFFFF_FFFF,
                castsShadow: Bool = true, softRadius: Float = 0, giVisible: Bool = true,
                bulbRadius: Float = 0, openingRadius: Float = 0, profileIndex: Int32 = -1, openingOffset: Float = 0) {
        self.bulbRadius = bulbRadius
        self.openingRadius = openingRadius
        self.openingOffset = openingOffset
        self.profileIndex = profileIndex
        self.position = position
        self.direction = direction
        self.innerCone = innerCone
        self.outerCone = outerCone
        self.color = color
        self.radius = radius
        self.layerMask = layerMask
        self.castsShadow = castsShadow ? 1 : 0
        self.softRadius = softRadius
        self.giVisible = giVisible ? 1 : 0
    }
}
