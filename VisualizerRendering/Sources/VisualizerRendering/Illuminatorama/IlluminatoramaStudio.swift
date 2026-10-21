//  The engine's default PRODUCT STUDIO.
//
//  A piece of polished metal or glass has no colour of its own: it shows what stands around it. Against an empty sky it
//  renders as flat pale plastic whatever the lighting (measured 2026-10-20: every fork, spoon and toaster came out cream
//  or white until the room around it was built). This is that room — a black studio with a floor, a stepped-gradient
//  softbox, a bright side softbox, a tall front strip, a dim fill and two black flags — captured as DATA so every host
//  (the Daydream Home piece studio, the Visualizer scenes, a catalog view) gets the same look and none re-derives it.
//
//  It is a rig of `Card`s: boxes with an albedo and an emission, lying around a piece that rests on y = 0. `instances()`
//  turns them into ordinary renderer instances, so they light the scene (path-traced GI) AND appear in reflections
//  (ray-traced reflections). By default the CARDS are hidden from the camera (`cardsHiddenFromCamera`): they are lights,
//  not set dressing, and a product shot is of the piece — not of the softbox. They give their own mesh kind to the
//  renderer, which skips it in every raster pass (frame, shadow maps) but keeps it in the ray-tracing structure, so the
//  lighting and the reflections are untouched. The FLOOR is always visible: it is what the piece stands on.
//
//  What each card is for — each one fixed a specific bad render:
//   · floor            a dark matte slab with its TOP at exactly y = 0. The bare terrain sits lower than the origin, so a
//                      piece "on the ground" hovered 2 cm above its own shadow; this gives it a surface to rest on.
//   · gradient bank    five adjacent overhead strips, bright → dark. A flat face mirrors ONE angle of the room, so a
//                      single bright card washes it white; a ramp shows a surface's form as a gradient across it.
//   · right softbox    a large bright panel at one side: long highlights along a piece's length.
//   · front strip      a tall narrow strip in front: a vertical face (a toaster front, a fridge door) mirrors the room in
//                      FRONT of it and reads as flat grey paint without something bright there.
//   · far fill         dim, so shadows are not pure black and the silhouette separates from the floor.
//   · black flags      opposite the lights: dark reflections are what make metal read as metal (a mirror is mostly the
//                      dark around its highlights).
//
//  The rig is sized for a piece of ~0.05–1 m. Cards sit on a 2.4 m ring so they stay out of frame at normal camera
//  distances; `scaled(by:)` grows the whole rig for a bigger subject (the angular size of each card, and so the look, is
//  preserved — emission is radiance, so it does not change with scale).
//
//  HOST RESPONSIBILITIES — the rig is only the room. Also needed for the look (the engine does not own the grade):
//   · ray-traced REFLECTIONS ON (`raytracedReflections`). They ship OFF in the Daydream app, and a metal then mirrors
//     only the sky probe: flat cream, whatever the cards do (DH-1444).
//   · NO SUN (`sunEnabled == false`): a black studio, so the cards are the only light.
//   · white balance `whiteBalanceK` (neutral 6500 K — a 9000 K noon-sky balance turns a black studio warm) and exposure
//     `exposureScale` × the host's own: a big polished surface blows out at 1×.
//  `apply(to:)` sets the two things the renderer itself owns; the rest are values for the host's grade.

import simd

public struct IlluminatoramaStudio: Sendable, Equatable {

    /// One light or flag: a box centred on `center`, `size` metres across, uniformly emitting `emission` (0 = a flag).
    public struct Card: Sendable, Equatable {
        public var name: String
        public var center: SIMD3<Float>
        public var size: SIMD3<Float>
        public var emission: Float
        public var albedo: Float

        public init(_ name: String, center: SIMD3<Float>, size: SIMD3<Float>, emission: Float = 0, albedo: Float = 0.02) {
            self.name = name; self.center = center; self.size = size; self.emission = emission; self.albedo = albedo
        }
    }

    /// The surface the piece rests on; its top face is at `center.y + size.y / 2`.
    public var floor: Card
    public var cards: [Card]

    /// Recommended grade (host applies): neutral white balance in kelvin, and an exposure multiplier.
    public var whiteBalanceK: Float
    public var exposureScale: Float
    /// Recommended renderer state: ray-traced reflections on, no sun.
    public var raytracedReflections: Bool
    public var sunEnabled: Bool
    /// The light cards light and reflect but are not drawn to the camera (needs `apply(to:)`, which registers the mesh).
    public var cardsHiddenFromCamera: Bool

    public init(floor: Card, cards: [Card], whiteBalanceK: Float = 6500, exposureScale: Float = 0.55,
                raytracedReflections: Bool = true, sunEnabled: Bool = false, cardsHiddenFromCamera: Bool = true) {
        self.floor = floor; self.cards = cards
        self.whiteBalanceK = whiteBalanceK; self.exposureScale = exposureScale
        self.raytracedReflections = raytracedReflections; self.sunEnabled = sunEnabled
        self.cardsHiddenFromCamera = cardsHiddenFromCamera
    }

    /// The light cards' own mesh kind. A separate kind (not the shared `.box`) because the renderer hides a whole GROUP from
    /// the camera, and other boxes in the scene must still draw.
    public static let cardKind = IlluminatoramaRenderer.MeshKind.custom("illuminatorama.studio.card")

    /// The default: a black product studio for a piece resting on y = 0.
    public static let productShot = IlluminatoramaStudio(
        floor: Card("floor", center: SIMD3(0, -0.01, 0), size: SIMD3(8, 0.02, 8), emission: 0, albedo: 0.05),
        cards: (0..<5).map { i in
            Card("gradient-\(i)", center: SIMD3(-0.7 + 0.30 * Float(i), 2.0, -0.2), size: SIMD3(0.30, 0.02, 2.6),
                 emission: [7, 3, 1.0, 0.3, 0.1][i])                            // stepped-gradient softbox, bright → dark
        } + [
            Card("right-softbox", center: SIMD3(2.4, 1.6, 0.0), size: SIMD3(0.05, 1.6, 2.4), emission: 6.0),
            Card("left-flag", center: SIMD3(-2.4, 1.6, 0.0), size: SIMD3(0.05, 1.6, 2.4), emission: 0),
            Card("far-fill", center: SIMD3(0.0, 1.6, 2.4), size: SIMD3(2.4, 1.6, 0.05), emission: 0.35),
            Card("near-flag", center: SIMD3(0.0, 1.6, -2.4), size: SIMD3(2.4, 1.6, 0.05), emission: 0),
            Card("front-strip", center: SIMD3(1.0, 1.0, 2.2), size: SIMD3(0.35, 1.9, 0.04), emission: 6.0),
        ])

    /// This rig with every card's position and size multiplied by `k` (emission, a radiance, is unchanged).
    public func scaled(by k: Float) -> IlluminatoramaStudio {
        func s(_ c: Card) -> Card { var c = c; c.center *= k; c.size *= k; return c }
        var out = self
        out.floor = s(floor)
        out.cards = cards.map(s)
        return out
    }

    /// This rig moved by `delta` (a piece standing somewhere other than the origin: the floor and every card go with it).
    public func translated(by delta: SIMD3<Float>) -> IlluminatoramaStudio {
        var out = self
        out.floor.center += delta
        out.cards = cards.map { var c = $0; c.center += delta; return c }
        return out
    }

    /// The floor and every card as renderer instances — hand them to the scene next to the piece. Call `apply(to:)` first
    /// when `cardsHiddenFromCamera` (the default): it registers the cards' mesh, without which they would not exist at all.
    public func instances() -> [IlluminatoramaRenderer.InstanceRef] {
        ([floor] + cards).enumerated().map { i, c in
            var m = matrix_identity_float4x4
            m.columns.0 = SIMD4<Float>(c.size.x, 0, 0, 0)
            m.columns.1 = SIMD4<Float>(0, c.size.y, 0, 0)
            m.columns.2 = SIMD4<Float>(0, 0, c.size.z, 0)
            m.columns.3 = SIMD4<Float>(c.center.x, c.center.y, c.center.z, 1)
            let isCard = i > 0 && cardsHiddenFromCamera
            return IlluminatoramaRenderer.InstanceRef(
                meshKind: isCard ? Self.cardKind : .box,
                data: IlluminatoramaInstance(modelMatrix: m, albedo: SIMD3(repeating: c.albedo), metallic: 0, roughness: 0.9,
                                             emission: SIMD3(repeating: c.emission)))
        }
    }

    /// Set what the RENDERER owns: ray-traced reflections and a fixed exposure (an auto-exposure would re-meter the
    /// black studio to grey). White balance and exposure scale are the host's grade — read them off this value.
    @MainActor
    public func apply(to renderer: IlluminatoramaRenderer) {
        renderer.rtReflectionsEnabled = raytracedReflections
        renderer.autoExposureEnabled = false
        if cardsHiddenFromCamera {
            renderer.setMesh(Self.cardKind, IlluminatoramaMesh.unitBox(device: renderer.device))   // a named kind: nothing to retain
            renderer.cameraHiddenKinds.insert(Self.cardKind)
        } else {
            renderer.cameraHiddenKinds.remove(Self.cardKind)
        }
    }

    /// Undo `apply`: the cards are visible to the camera again (a host leaving the studio).
    @MainActor
    public static func remove(from renderer: IlluminatoramaRenderer) {
        renderer.cameraHiddenKinds.remove(cardKind)
    }
}
