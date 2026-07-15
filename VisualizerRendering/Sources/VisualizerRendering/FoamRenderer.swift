#if canImport(AppKit) && !targetEnvironment(macCatalyst)
import AppKit
#else
import UIKit
#endif
import Foundation
import Metal
import SceneKit
import simd
import VisualizerCore

// ── FOAM RENDERER ────────────────────────────────────────────────────────────
//
// Bridges a FoamSolver's particle buffer into an SCNGeometry rendered as
// hardware points. Same zero-copy contract as FluidParticleRenderer: SceneKit
// reads `positionLife.xyz` from the same MTLBuffer the foam-advect kernel
// writes into. No CPU snapshot.
//
// PER-PARTICLE FADE & TINT
// ────────────────────────
// The FoamParticle struct's `positionLife.w` is the remaining-life seconds
// (set on spawn, decremented per advect step). We alias the same MTLBuffer a
// second time as a `.color` semantic SCNGeometrySource at offset 0 — that
// gives the fragment shader access to a per-particle float4 where `.a` is the
// life value. A surface-shader modifier turns that into:
//
//   • alpha = a smooth fade-in (first 15% of life), full bright for the
//     bright middle, then a long fade-out into nothing. Without this the
//     foam read as a hard "pop in / pop out" — droplets blinking on and
//     off rather than spraying.
//   • per-particle tint = bright white-blue when fresh, fading toward sky-
//     cyan as the droplet ages so the long-lived spray reads as wisp,
//     not the same crisp white that just left the splash crown.
//
// Dead foam particles are PARKED by the advect kernel at `boundsMin.y - 1000`,
// so they still get drawn but are off-screen. The alpha modulation handles
// near-dead-but-still-on-screen particles gracefully via the fade-out band.
//
// Material is constant-lit + additive-blend so spray reads as self-luminous
// against the dark water surface. Point size is screen-space-scaled so close-
// up droplets cover real pixels.

@MainActor
public final class FoamRenderer {

    public let node: SCNNode
    public let solver: FoamSolver

    private let geometryElement: SCNGeometryElement
    private let indexBuffer: MTLBuffer

    // ── Phase 4.23 — Illuminatorama bridge (held for `publishToIlluminatorama`) ──
    //
    // The foam-particle ring buffer interleaves `positionLife` + `velocityAge`
    // (two SIMD4<Float>, 32 B = 8 floats stride). The Phase 4.23 external
    // point pipeline reads `(x, y, z)` from the head of every stride-slot, so
    // we can publish `solver.particleBuffer` straight as the position source
    // with `positionStrideFloats = 8`.
    //
    // Colour for foam is a uniform mustard / spray tint (no per-particle
    // colour), but the external-emitter contract still needs one float
    // per vertex to index into. We allocate a `capacity`-long colour
    // buffer populated once with the chosen foam tint; subsequent frames
    // never touch it. 64 KB at 18k capacity — trivial.
    private let illumiColorBuffer: MTLBuffer
    private var currentPointSize: Float
    private var currentColour: SIMD3<Float>
    /// SCNScene identity captured by `registerParticleField(scene:)`, reused
    /// by `publishToIlluminatorama()` so live re-registration (point-size
    /// changes) can re-scope the source without the caller re-passing it.
    private var registeredOwnerScene: ObjectIdentifier?

    public init?(solver: FoamSolver,
                 pointSize: CGFloat = 9.0,
                 colour: SIMD3<Float> = SIMD3(0.95, 0.97, 1.0)) {
        let device = solver.device
        let cap = solver.particleBuffer.capacity

        // Phase 4.23 colour buffer. 4 floats per vertex (SIMD4 stride) so
        // the descriptor below can use `colorStrideFloats: 4`. We fill it
        // with the foam tint at full brightness; `colorScale` on the
        // descriptor lets a host attenuate at runtime without touching
        // the buffer.
        let colStride = MemoryLayout<SIMD4<Float>>.stride
        guard let illumiCBuf = device.makeBuffer(length: colStride * cap,
                                                  options: .storageModeShared)
        else { return nil }
        illumiCBuf.label = "Foam.illumiColors"
        let cPtr = illumiCBuf.contents().bindMemory(to: SIMD4<Float>.self,
                                                     capacity: cap)
        let tint = SIMD4<Float>(colour.x, colour.y, colour.z, 1)
        for i in 0..<cap { cPtr[i] = tint }
        self.illumiColorBuffer = illumiCBuf
        self.currentPointSize = Float(pointSize)
        self.currentColour = colour

        let idxStride = MemoryLayout<UInt32>.stride
        guard let idxBuf = device.makeBuffer(length: idxStride * cap,
                                             options: .storageModeShared)
        else { return nil }
        idxBuf.label = "Foam.particleIndices"
        let idxPtr = idxBuf.contents().bindMemory(to: UInt32.self, capacity: cap)
        for i in 0..<cap { idxPtr[i] = UInt32(i) }

        // FoamParticle starts with positionLife (SIMD4<Float>); .xyz at byte
        // offset 0 is the world position.
        //
        // NOTE: an earlier version aliased the same buffer as a `.color`
        // SCNGeometrySource (intent: read `.a` as life-remaining from a
        // shader modifier). Two problems killed that path:
        //   1) Shader modifiers on `.point` primitives silently disable
        //      rasterisation (see memory: shader_modifiers_break_point_primitives).
        //   2) WITHOUT the modifier consuming the alias, SceneKit's default
        //      fragment path interprets the float3 prefix of `.color` as
        //      per-vertex RGB — producing position-tinted droplets
        //      (foam reads as multicoloured because each droplet's world
        //      XYZ becomes its RGB). Don't re-add that alias unless a
        //      working consumer is wired up at the same time.
        let particleStride = MemoryLayout<FoamParticle>.stride
        let posSource = SCNGeometrySource(
            buffer: solver.particleBuffer.buffer,
            vertexFormat: .float3,
            semantic: .vertex,
            vertexCount: cap,
            dataOffset: 0,
            dataStride: particleStride
        )

        let element = SCNGeometryElement(
            buffer: idxBuf,
            primitiveType: .point,
            primitiveCount: cap,
            bytesPerIndex: idxStride
        )
        element.minimumPointScreenSpaceRadius = pointSize * 0.5
        element.maximumPointScreenSpaceRadius = pointSize * 1.8
        element.pointSize = pointSize

        let geom = SCNGeometry(sources: [posSource], elements: [element])

        // Constant-lit so visibility doesn't depend on scene lighting — foam
        // is supposed to read as self-luminous spray. Additive blending makes
        // overlapping droplets brighten the surface beneath them, the way
        // real foam concentrates on splash crowns.
        let mat = SCNMaterial()
        mat.lightingModel = .constant
        let col = PlatformColor(deviceRed: CGFloat(colour.x),
                          green:     CGFloat(colour.y),
                          blue:      CGFloat(colour.z),
                          alpha:     1)
        mat.diffuse.contents  = col
        mat.emission.contents = col
        mat.blendMode = .add
        mat.writesToDepthBuffer = false
        mat.readsFromDepthBuffer = true

        // ── Per-particle fade modifier ───────────────────────────────────
        //
        // `_surface.diffuse` arrives interpolated from the vertex colour
        // attribute, which we aliased over `positionLife` — so `.a` is the
        // remaining-life seconds. lifeMax is hard-coded to match FoamSolver's
        // typical .lifeMax range (1.4s); it sets the inverse used to normalise
        // life into 0..1. A scene-level uniform could feed the exact value
        // but the curve is forgiving (a 2x error just shifts when fade-in
        // ends), and an extra uniform binding per renderer adds surface area
        // without buying much.
        //
        // Curve:
        //   life fraction (life remaining / lifeMax) maps to age fraction
        //   t = 1 - lifeFrac. Then alpha = smoothstep fade-in over t<0.15,
        //   full bright in the middle, smoothstep fade-out over t>0.55.
        //   Young droplets are crisp white; older ones tint toward cyan
        //   so the lingering wisp reads as foam dispersing into the air.
        //
        // `_surface.transparent.a` is multiplied into the final alpha by
        // SceneKit's transparency pipeline. We feed our computed alpha
        // through both `_surface.diffuse.a` and `_surface.transparent.a`
        // so the additive blend respects it.
        // NOTE on per-particle fade — SceneKit shader modifiers don't appear
        // to compose with `.point` primitives the way they do for triangles
        // (both `.surface` and `.fragment` entry points silently make the
        // points stop drawing). Per-particle alpha-by-life is therefore
        // handled in the FOAM SOLVER kernel: when life drops below a fade
        // threshold the advect kernel pushes the particle's xy into a
        // wider drift so it dissipates as a sparser cloud rather than
        // hanging at full brightness then popping out. The dead-park is
        // unchanged. Additive blending + dense spawn rate gives the
        // perceived "foam concentration" gradient without needing
        // explicit per-particle alpha.

        geom.materials = [mat]

        let n = SCNNode(geometry: geom)
        n.name = "FoamParticles"
        // Render after the fluid surface so it composites on top.
        n.renderingOrder = 100

        self.solver          = solver
        self.geometryElement = element
        self.indexBuffer     = idxBuf
        self.node            = n
    }

    // ── Live-tunable appearance ────────────────────────────────────────────

    /// Update the rendered point radius. Applies immediately to the SCN
    /// forward path. The Illuminatorama path retains the size from
    /// `publishToIlluminatorama()`; call that again after this if you need
    /// both in sync (cheap — just rewrites the associated-object descriptor).
    public func setPointSize(_ size: CGFloat) {
        let s = Float(size)
        guard s != currentPointSize else { return }
        currentPointSize = s
        geometryElement.minimumPointScreenSpaceRadius = size * 0.5
        geometryElement.maximumPointScreenSpaceRadius = size * 1.8
        geometryElement.pointSize = size
        // Re-publish so the Illuminatorama descriptor reflects the new size.
        // The extractor caches by positionBuffer identity on first encounter,
        // so this only takes effect if publishToIlluminatorama() is called
        // before the extractor has registered this geometry. The SCN path
        // above always takes effect immediately.
        publishToIlluminatorama()
    }

    /// Update the foam tint. Applies to both the SCN material and the
    /// Illuminatorama colour buffer on the next frame (shared GPU memory).
    public func setColour(_ colour: SIMD3<Float>) {
        guard colour != currentColour else { return }
        currentColour = colour
        let col = PlatformColor(deviceRed: CGFloat(colour.x),
                          green:     CGFloat(colour.y),
                          blue:      CGFloat(colour.z),
                          alpha:     1)
        node.geometry?.materials.first?.diffuse.contents  = col
        node.geometry?.materials.first?.emission.contents = col
        let cap  = solver.particleBuffer.capacity
        let cPtr = illumiColorBuffer.contents()
            .bindMemory(to: SIMD4<Float>.self, capacity: cap)
        let tint = SIMD4<Float>(colour.x, colour.y, colour.z, 1)
        for i in 0..<cap { cPtr[i] = tint }
    }

    // ── Illuminatorama particle-field registration ────────────────────────
    //
    // Publishes the foam particle buffer into the shared
    // `SimEngine.particleFields` registry so the Illuminatorama overlay draws
    // it as additive HDR point sprites. The SCN `.point`/`.add` node still
    // draws on SceneKit's own forward pass (Phase 4.18's additive-skip filter
    // keeps the deferred opaque pass off it; the two renderers composite into
    // separate targets, so no double-draw).
    //
    // Call `registerParticleField(scene:)` once from the controller; it stores
    // the owning SCNScene's identity (the overlay's active-scene filter key,
    // which stays correct even though AppModel caches controllers across scene
    // switches). `publishToIlluminatorama()` re-registers in place — the
    // registry is keyed by position-buffer identity — so a live point-size
    // change takes effect on the very next overlay frame.
    //
    // Layout:
    //   • positionBuffer = solver.particleBuffer.buffer. `FoamParticle` is
    //     `(SIMD4<Float> positionLife, SIMD4<Float> velocityAge)` = 32 B =
    //     8-float stride; positionLife.xyz at offset 0 is the world position.
    //   • colorBuffer = `illumiColorBuffer` — pre-filled foam tint, 4-float
    //     stride, offset 0. Foam has no per-particle colour.
    //
    // Dead (parked) particles sit at `solver.boundsMin.y - 1000`, far below
    // any frustum, so each parked slot costs one VS invocation + a clipped
    // primitive — negligible.
    public func registerParticleField(scene: SCNScene) {
        registeredOwnerScene = ObjectIdentifier(scene)
        publishToIlluminatorama()
    }

    /// Re-register the foam field with the current point size. No-op until
    /// `registerParticleField(scene:)` has captured the owning scene.
    public func publishToIlluminatorama() {
        guard let owner = registeredOwnerScene else { return }
        SimEngine.particleFields.register(ParticleFieldSource(
            positionBuffer:       solver.particleBuffer.buffer,
            colorBuffer:          illumiColorBuffer,
            vertexCount:          solver.particleBuffer.capacity,
            ownerScene:           owner,
            positionStrideFloats: 8,                       // FoamParticle stride / 4
            colorStrideFloats:    4,                       // SIMD4<Float>
            positionOffsetFloats: 0,
            colorOffsetFloats:    0,
            pointSize:            currentPointSize,
            colorScale:           1.0
        ))
    }
}
