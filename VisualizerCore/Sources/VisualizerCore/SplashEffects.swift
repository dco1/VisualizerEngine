import AppKit
import SceneKit

/// Liquid-surface splash effects shared between scenes.
///
/// Each scene builds a `SplashEffects.Style` describing the liquid (its
/// PBR material, its constant-lit droplet sprite, the world-Y of its
/// surface, and a few tuning knobs) and calls into the static spawn
/// methods to add splash debris under a caller-chosen parent node. The
/// parent is the caller's responsibility because different scenes want
/// different containment: Hotdog Drop+ uses a top-level `splashes` node;
/// Well Drop uses a node nested inside the well so debris that escapes
/// the action's clip distance still gets occluded by the well lip.
///
/// The splash anatomy mirrors the Worthington photographs:
///
///   1. Crown — a wide thin ring that grows outward and fades. Reads as
///      the cavity-rim sheet that forms when something punches into
///      liquid.
///   2. Jet — a narrow column that rises then collapses. Drives the eye
///      upward and reads as the secondary jet that climbs after the
///      crown collapses.
///   3. Droplet fan — billboarded radial-gradient sprites flung outward
///      and upward on ballistic arcs. Two-segment SCNAction.move(to:)
///      pairs approximate the parabola without paying for `customAction`
///      or `@Sendable` closures.
public enum SplashEffects {

    /// Per-liquid configuration. Captured once at scene build and reused
    /// for every spawn — both materials are shared `SCNMaterial`
    /// references so droplet count doesn't multiply uniform uploads.
    public struct Style {
        public let liquidMaterial: SCNMaterial
        public let dropletSpriteMaterial: SCNMaterial
        /// Optional material for the crown / jet / pinch-off droplet —
        /// the bits whose orientation sweeps across the env-map every
        /// frame. A glossy clearcoat liquid material (good for the
        /// puddle and ambient ripples, which sit flat on the pool) will
        /// pick up cool zenith reflections on the upward-facing parts
        /// of a flared crown — a yellow mustard splash then reads as
        /// olive. Pass a less-reflective variant here to keep the
        /// splash anatomy anchored to its base colour. Falls back to
        /// `liquidMaterial` when nil.
        public var splashMaterial: SCNMaterial? = nil
        /// World-Y of the liquid's surface. Splash debris is offset just
        /// above this; droplets land slightly below it so they disappear
        /// by entering the liquid rather than vanishing mid-air.
        public let surfaceY: CGFloat
        /// World-scale fudge applied to crown / jet / droplet sizes. 1.0
        /// matches Hotdog Drop+'s tuning where the surface is roughly at
        /// the camera's eyeline. Drop below 1 for scenes where the
        /// liquid is far below the camera (Well Drop drops to ~0.55 so
        /// the splash reads correctly when viewed from 15 m up).
        public let scale: CGFloat
        /// Gravity used by the ballistic droplet trajectories. Splashes
        /// are animation, not physics — but matching the scene's
        /// physics gravity keeps the arcs reading right next to the
        /// real objects' fall.
        public let gravity: Double

        public init(
            liquidMaterial: SCNMaterial,
            dropletSpriteMaterial: SCNMaterial,
            surfaceY: CGFloat,
            scale: CGFloat,
            gravity: Double,
            splashMaterial: SCNMaterial? = nil
        ) {
            self.liquidMaterial = liquidMaterial
            self.dropletSpriteMaterial = dropletSpriteMaterial
            self.splashMaterial = splashMaterial
            self.surfaceY = surfaceY
            self.scale = scale
            self.gravity = gravity
        }
    }

    /// Full Worthington-style splash. Caller passes the impact point in
    /// world coordinates of `parent`; the Y is snapped to the style's
    /// `surfaceY` plus a tiny lift so geometry doesn't z-fight. The
    /// `intensity` knob (typical range 0.3 – 1.6, clamped at 2.0)
    /// scales crown spread, jet height, droplet count, and droplet
    /// speeds — light glancing blows produce a small flick of debris;
    /// a heavy impact throws a tall jet and a wide fan of droplets.
    /// Defaults to 1.0 so callers that don't yet pass an impulse
    /// behave exactly as before.
    public static func spawnFullSplash(
        in parent: SCNNode,
        at impactPoint: SCNVector3,
        style: Style,
        intensity: Double = 1.0
    ) {
        let s = style.scale
        let i = CGFloat(min(2.0, max(0.25, intensity)))
        let id = Double(i)
        let origin = SCNVector3(impactPoint.x, style.surfaceY + 0.01, impactPoint.z)
        let anim = style.splashMaterial ?? style.liquidMaterial

        let crownBottom = 0.07 * s * i
        let crownTop = 0.18 * s * i
        let crownHeight = 0.06 * s * i
        let crown = SCNCone(
            topRadius: crownTop,
            bottomRadius: crownBottom,
            height: crownHeight
        )
        crown.radialSegmentCount = 36
        crown.heightSegmentCount = 1
        crown.materials = [makeDoubleSided(anim)]
        let crownNode = SCNNode(geometry: crown)
        crownNode.position = SCNVector3(origin.x, origin.y + crownHeight * 0.5, origin.z)
        crownNode.opacity = 0.78
        parent.addChildNode(crownNode)
        let crownSpread = SCNAction.scale(to: CGFloat(5.5 + 3.0 * id), duration: 0.5)
        crownSpread.timingMode = .easeOut
        let crownRise = SCNAction.moveBy(
            x: 0, y: 0.04 * s * i, z: 0, duration: 0.5
        )
        crownRise.timingMode = .easeOut
        crownNode.runAction(.sequence([
            .group([
                crownSpread,
                crownRise,
                .fadeOut(duration: 0.55)
            ]),
            .removeFromParentNode()
        ]))

        let jetHeight = (0.35 * s) * (0.7 + 0.6 * i)
        let jetRadius = 0.022 * s
        let jet = SCNCylinder(radius: jetRadius, height: jetHeight)
        jet.radialSegmentCount = 14
        jet.heightSegmentCount = 1
        jet.materials = [anim]
        let jetGeo = SCNNode(geometry: jet)
        jetGeo.position = SCNVector3(0, jetHeight * 0.5, 0)
        let jetNode = SCNNode()
        jetNode.position = origin
        jetNode.scale = SCNVector3(0.4, 0.05, 0.4)
        jetNode.opacity = 0.95
        jetNode.addChildNode(jetGeo)
        parent.addChildNode(jetNode)
        let jetRise = SCNAction.scale(to: 1.0, duration: 0.16)
        jetRise.timingMode = .easeOut
        let jetCollapse = SCNAction.scale(to: 0.1, duration: 0.32)
        jetCollapse.timingMode = .easeIn
        jetNode.runAction(.sequence([
            jetRise,
            .group([jetCollapse, .fadeOut(duration: 0.32)]),
            .removeFromParentNode()
        ]))

        let headRadius = 0.04 * s * i
        let head = SCNSphere(radius: headRadius)
        head.segmentCount = 14
        head.materials = [anim]
        let headNode = SCNNode(geometry: head)
        let topY = origin.y + jetHeight
        headNode.position = SCNVector3(origin.x, origin.y + 0.02 * s, origin.z)
        headNode.opacity = 0.95
        parent.addChildNode(headNode)
        let headRise = SCNAction.move(
            to: SCNVector3(origin.x, topY, origin.z),
            duration: 0.16
        )
        headRise.timingMode = .easeOut
        let headApex = SCNAction.move(
            to: SCNVector3(origin.x, topY + 0.06 * s * i, origin.z),
            duration: 0.10
        )
        headApex.timingMode = .easeOut
        let headFall = SCNAction.move(
            to: SCNVector3(origin.x, style.surfaceY - 0.05, origin.z),
            duration: 0.30
        )
        headFall.timingMode = .easeIn
        headNode.runAction(.sequence([
            headRise,
            headApex,
            .group([headFall, .fadeOut(duration: 0.30)]),
            .removeFromParentNode()
        ]))

        let baseCount = 6 + Int((id * 7).rounded())
        let count = min(18, max(4, baseCount + Int.random(in: -1...2)))
        for _ in 0..<count {
            spawnDroplet(
                in: parent,
                from: origin,
                radius: CGFloat.random(in: 0.045 * s ... 0.11 * s) * (0.7 + 0.4 * i),
                upSpeed: Double.random(in: 2.4...4.2) * Double(s) * (0.7 + 0.5 * id),
                outSpeed: Double.random(in: 1.4...3.0) * Double(s) * (0.7 + 0.6 * id),
                style: style
            )
        }
    }

    private static func makeDoubleSided(_ m: SCNMaterial) -> SCNMaterial {
        let copy = m.copy() as! SCNMaterial
        copy.isDoubleSided = true
        return copy
    }

    /// A slow expanding ripple ring near `center`.
    public static func spawnAmbientRipple(
        in parent: SCNNode,
        around center: SCNVector3,
        radius: CGFloat,
        style: Style
    ) {
        let s = style.scale
        let x = center.x + CGFloat.random(in: -radius ... radius)
        let z = center.z + CGFloat.random(in: -radius ... radius)
        let ring = SCNTorus(ringRadius: 0.12 * s, pipeRadius: 0.018 * s)
        ring.ringSegmentCount = 28
        ring.pipeSegmentCount = 8
        ring.materials = [style.liquidMaterial]
        let node = SCNNode(geometry: ring)
        node.position = SCNVector3(x, style.surfaceY + 0.006, z)
        node.opacity = 0.55
        parent.addChildNode(node)
        node.runAction(.sequence([
            .group([
                .scale(to: 7.0, duration: 1.6),
                .fadeOut(duration: 1.6)
            ]),
            .removeFromParentNode()
        ]))
    }

    private static func spawnDroplet(
        in parent: SCNNode,
        from origin: SCNVector3,
        radius: CGFloat,
        upSpeed: Double,
        outSpeed: Double,
        style: Style
    ) {
        let plane = SCNPlane(width: radius * 2, height: radius * 2)
        plane.firstMaterial = style.dropletSpriteMaterial
        let node = SCNNode(geometry: plane)
        node.constraints = [SCNBillboardConstraint()]
        node.position = SCNVector3(origin.x, origin.y + 0.04 * style.scale, origin.z)
        parent.addChildNode(node)

        let angle = Double.random(in: 0..<(2 * .pi))
        let vx = cos(angle) * outSpeed
        let vz = sin(angle) * outSpeed
        let g = style.gravity

        let riseTime = upSpeed / g
        let peakY = Double(origin.y) + 0.04 * Double(style.scale)
            + upSpeed * riseTime - 0.5 * g * riseTime * riseTime
        let fallDistance = peakY - (Double(style.surfaceY) - 0.05)
        let fallTime = max(0.05, sqrt(max(0, 2 * fallDistance / g)))

        let peakPos = SCNVector3(
            origin.x + CGFloat(vx * riseTime),
            CGFloat(peakY),
            origin.z + CGFloat(vz * riseTime)
        )
        let landPos = SCNVector3(
            peakPos.x + CGFloat(vx * fallTime),
            style.surfaceY - 0.05,
            peakPos.z + CGFloat(vz * fallTime)
        )

        let rise = SCNAction.move(to: peakPos, duration: riseTime)
        rise.timingMode = .linear
        let fall = SCNAction.move(to: landPos, duration: fallTime)
        fall.timingMode = .linear
        node.runAction(.sequence([
            rise,
            fall,
            .removeFromParentNode()
        ]))
    }

    /// Constant-lit billboard sprite material used by droplet sprites.
    public static func makeSpriteMaterial(color: NSColor) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = makeSpriteImage(color: color)
        m.isDoubleSided = true
        m.writesToDepthBuffer = false
        m.transparencyMode = .default
        return m
    }

    /// Shared PBR liquid material.
    public static func makeLiquidMaterial(diffuse: NSColor, emission: NSColor) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        m.diffuse.contents = diffuse
        m.metalness.contents = 0.0
        m.roughness.contents = 0.16
        m.clearCoat.contents = 0.9
        m.clearCoatRoughness.contents = 0.04
        m.emission.contents = emission
        return m
    }

    private static func makeSpriteImage(color: NSColor) -> NSImage {
        let size: CGFloat = 32
        let img = NSImage(size: NSSize(width: size, height: size))
        img.lockFocus()
        defer { img.unlockFocus() }
        guard let ctx = NSGraphicsContext.current?.cgContext else { return img }
        guard let rgb = color.usingColorSpace(.deviceRGB) else { return img }
        let centre = CGPoint(x: size / 2, y: size / 2)
        let dark = NSColor(
            calibratedRed: rgb.redComponent * 0.7,
            green: rgb.greenComponent * 0.7,
            blue: rgb.blueComponent * 0.7,
            alpha: 1
        )
        let edge = NSColor(
            calibratedRed: rgb.redComponent * 0.55,
            green: rgb.greenComponent * 0.55,
            blue: rgb.blueComponent * 0.55,
            alpha: 0
        )
        let space = CGColorSpaceCreateDeviceRGB()
        if let gradient = CGGradient(
            colorsSpace: space,
            colors: [rgb.cgColor, dark.cgColor, edge.cgColor] as CFArray,
            locations: [0, 0.7, 1]
        ) {
            ctx.drawRadialGradient(
                gradient,
                startCenter: centre, startRadius: 0,
                endCenter: centre, endRadius: size / 2,
                options: []
            )
        }
        return img
    }
}
