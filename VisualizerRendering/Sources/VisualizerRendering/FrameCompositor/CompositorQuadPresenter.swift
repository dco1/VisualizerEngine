import AppKit
import Foundation
import Metal
import SceneKit
import simd

// ── COMPOSITOR QUAD PRESENTER ───────────────────────────────────────────────
//
// Bridges a `FrameCompositor`'s `displayTexture` into a SceneKit scene by
// hosting a camera-attached quad whose material samples the texture. The
// host's existing SCNView keeps owning the window/drawable surface; the
// quad shows whatever the compositor wrote each frame.
//
// REUSABLE for any scene that wants to display a custom-Metal-rendered
// image inside a SceneKit camera view.
//
// USAGE
//
//   let presenter = CompositorQuadPresenter(outputTexture: comp.displayTexture)
//   presenter.attach(to: cameraNode)        // parents the quad to the camera
//   presenter.update(fovDegrees: cam.fov, aspect: 16/9)
//
// Parenting the quad directly under the camera (rather than world-rooted
// and re-applying the camera's world transform each tick) is what makes
// SCN render the quad facing the camera: SCNPlane's front face is its
// +Z direction in local coords, and the camera looks down its own -Z;
// as a child of the camera node the plane sits at a NEGATIVE local Z
// (in front of the camera) with its normal pointing back AT the camera,
// which is the face SCN's renderer treats as the visible front of the
// plane. (The world-rooted-then-copy-transform variant put the plane's
// normal facing AWAY from the camera, leaving SCN with only the back
// face — and even with `isDoubleSided = true` the result wasn't visible
// in our setup.)
//
// Render order:
//   • `renderingOrder = +1000` so the quad renders LAST.
//   • `writesToDepthBuffer = false`, `readsFromDepthBuffer = false` —
//     the quad is a screen-space overlay; it never participates in depth
//     testing. SCN keeps its own scene geometry visible behind it where
//     the compositor's output is transparent.
//
// Transparency:
//   The compositor's `fcTonemap` kernel writes premultiplied alpha — RGB
//   is scaled by per-pixel coverage (luminance), and the alpha channel
//   carries that same coverage. The presenter's material uses the
//   default alpha-blend mode; pixels the compositor didn't touch end
//   up as (0,0,0,0) and the SCN scene below shows through unaltered.

@MainActor
public final class CompositorQuadPresenter {

    public let node: SCNNode
    public let plane: SCNPlane
    public let material: SCNMaterial

    /// Construct the presenter. `outputTexture` is typically
    /// `FrameCompositor.displayTexture` — the LDR texture the compositor
    /// writes each frame. The presenter samples it with linear filtering.
    public init(outputTexture: MTLTexture) {
        // The plane gets sized in `update(fovDegrees:aspect:distance:)`
        // each tick — base 2×2 is just a starting size.
        let plane = SCNPlane(width: 2.0, height: 2.0)
        self.plane = plane

        let mat = SCNMaterial()
        mat.lightingModel = .constant
        mat.diffuse.contents = outputTexture
        mat.diffuse.magnificationFilter = .linear
        mat.diffuse.minificationFilter = .linear
        mat.diffuse.wrapS = .clamp
        mat.diffuse.wrapT = .clamp
        mat.writesToDepthBuffer = false
        mat.readsFromDepthBuffer = false
        mat.isDoubleSided = true
        // Default `.alpha` blend mode. The tonemap writes premultiplied
        // alpha (RGB *= coverage, A = coverage), which composes correctly
        // over the SCN scene with standard `src + dst*(1 - src.a)` math.
        mat.blendMode = .alpha
        plane.materials = [mat]
        self.material = mat

        let planeNode = SCNNode(geometry: plane)
        planeNode.name = "CompositorQuadPresenter.plane"
        planeNode.renderingOrder = 1000   // overlay — drawn last
        planeNode.castsShadow = false
        // The plane's front face is +Z in its local coords. As a child
        // of the camera (which looks down -Z), we want the plane's face
        // to point TOWARD the camera — i.e. its normal should be the
        // camera's forward direction (+Z of the plane = -Z of camera).
        // Rotating 180° around Y flips the plane to face the camera
        // without affecting UVs (vs. flipping around X which would mirror).
        planeNode.eulerAngles = SCNVector3(0, CGFloat.pi, 0)

        self.node = planeNode
    }

    /// Attach the presenter quad as a child of the given camera node so
    /// it follows the camera's transform automatically (including any
    /// SCNLookAtConstraint or animated rotation). Idempotent — calling
    /// twice with the same camera is a no-op.
    public func attach(to cameraNode: SCNNode) {
        if node.parent === cameraNode { return }
        node.removeFromParentNode()
        cameraNode.addChildNode(node)
    }

    /// Rebind the material's diffuse to a new output texture. Use this
    /// after the host's `FrameCompositor.resize(to:)` — `resize` swaps
    /// the display texture for a fresh one matching the new size, and
    /// the presenter's material was caching a reference to the previous
    /// (now dangling) MTLTexture.
    public func setOutputTexture(_ texture: MTLTexture) {
        material.diffuse.contents = texture
    }

    /// Size the quad each frame so it exactly covers the camera's
    /// frustum at `distance` metres in front of the camera. `fovDegrees`
    /// is the camera's `fieldOfView`; `aspect` is the compositor
    /// texture's width/height ratio (and should match the screen aspect
    /// for an undistorted result). `distance` is the local -Z offset
    /// from the camera; the default 0.25 m sits comfortably past the
    /// typical near plane (0.1 m).
    public func update(
        fovDegrees: CGFloat,
        aspect: Float,
        distance: Float = 0.25
    ) {
        let fovRad = Float(fovDegrees) * .pi / 180
        // Horizontal FOV → half-width at `distance`. Tiny oversize (5%)
        // keeps the quad just past the visible frustum edge so subpixel
        // jitter doesn't expose a sliver of background at the edges.
        let oversize: Float = 1.05
        let halfW = tanf(fovRad * 0.5) * distance * oversize
        // Aspect-matched height. The HOST is expected to keep the
        // compositor's texture aspect in sync with the SCN viewport's
        // aspect (cover-fill resize — FireworksUltra does this each
        // tick). With that invariant, `aspect` is the viewport's
        // aspect, and matching the quad to the same aspect produces a
        // pixel-accurate 1:1 mapping of texture → display with no
        // stretching, no letterbox, no crop. The previous formula
        // `max(halfW/aspect, halfW)` was a holdover from before the
        // cover-fill change — when the texture was fixed 16:9 and the
        // quad had to overscan to handle aspect mismatch — and now
        // actively breaks things: on a wider-than-tall viewport it
        // forces the quad SQUARE, the texture content gets vertically
        // stretched ~1.8× to fit, only the centre 27% of the texture's
        // height ends up on-screen, and that band upsamples ~5× to
        // fill the viewport (the "soft / low-rez" appearance).
        let halfH = halfW / max(aspect, 1e-3)
        // Position in camera-local coords: -distance along the camera's
        // forward direction (camera looks down -Z).
        node.simdPosition = SIMD3(0, 0, -distance)
        // `SCNPlane` is 2×2 in local coords (centred on origin), so a
        // scale of `halfW` produces total width `2 × halfW`.
        node.scale = SCNVector3(CGFloat(halfW), CGFloat(halfH), 1)
    }

    /// Legacy update path retained so FireworksPlus's experimental
    /// `useCustomCompositor` toggle keeps compiling. The world-rooted
    /// + push-camera-transform scheme this signature implies is fragile
    /// (see the `attach(to:)` docstring above for the failure mode the
    /// camera-parented path was introduced to fix). New scenes should
    /// call `attach(to: cameraNode)` once + `update(fovDegrees:…)`
    /// each tick; this overload just pushes the supplied world transform
    /// onto the presenter's node and then forwards to the modern sizer.
    public func update(
        cameraWorld: float4x4,
        fovDegrees: CGFloat,
        aspect: Float,
        distance: Float = 0.25
    ) {
        node.simdWorldTransform = cameraWorld
        update(fovDegrees: fovDegrees, aspect: aspect, distance: distance)
    }
}
