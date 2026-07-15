import Foundation
import SceneKit
#if canImport(AppKit) && !targetEnvironment(macCatalyst)
import AppKit
#else
import UIKit
#endif
import simd
import VisualizerCore

// ── NeonTube ──────────────────────────────────────────────────────────────────
//
// Reusable glowing-glass neon tubing. Give it a path (polyline) and it builds a
// swept-circle tube with the canonical neon recipe (see the `emissive_surface_recipe`
// memory): a constant-lit near-black glass body + HDR emission in the gas colour,
// paired with real SCNLights distributed along the path so the neon actually
// spills coloured light onto nearby surfaces (emissive alone doesn't light the
// world). Path points can spell letters, border a marquee, or underline a trough.
//
// Research (Coin Pusher §4): glass tube ⌀10–15 mm, a bright near-white-desaturated
// hot core + a saturated halo (bloom carries it), and a strong coloured cast onto
// adjacent glossy/chrome surfaces. The core/halo split is approximated with a hot
// emission colour + the camera's bloom; the cast comes from the paired lights.
//
// Returns an SCNNode (tube + child light nodes). For native Illuminatorama, read
// `geometry`/`lightSpecs` and feed them to the renderer (emissive instance + point
// lights). Optional `flicker(_:)` drives a subtle buzz on the emission intensity.

@MainActor
public final class NeonTube {

    public struct LightSpec {
        public var position: SIMD3<Float>
        public var color: SIMD3<Float>     // pre-multiplied intensity (for Illuminatorama)
        public var radius: Float
    }

    public let node: SCNNode
    public let geometry: SCNGeometry
    public private(set) var lightSpecs: [LightSpec]

    private let material: SCNMaterial
    private let baseEmissionIntensity: CGFloat

    /// Build a neon tube.
    /// - `path`: ≥2 world-space points (the tube centreline).
    /// - `radius`: tube radius (≈ glass radius; 0.05–0.09 reads as 10–15 mm at
    ///   token scale).
    /// - `color`: gas glow colour (use a saturated arcade hue).
    /// - `emissionIntensity`: HDR emission multiplier (>1 to bloom).
    /// - `lightColor`/`lightIntensity`: paired SCNLight colour spill.
    /// - `lightSpacing`: emit one light every N path points.
    public init(path: [SIMD3<Float>],
                radius: Float = 0.07,
                color: PlatformColor = PlatformColor(srgbRed: 1.0, green: 0.12, blue: 0.6, alpha: 1),
                emissionIntensity: CGFloat = 3.0,
                lightColor: PlatformColor? = nil,
                lightIntensity: CGFloat = 600,
                lightSpacing: Int = 4,
                radialSegments: Int = 12,
                closed: Bool = false) {
        let built = NeonTube.buildGeometry(path: path, radius: radius,
                                           radialSegments: radialSegments, closed: closed)
        GeometryWinding.audit(positions: built.positions, normals: built.normals,
                              indices: built.indices.map { UInt32(bitPattern: $0) },
                              label: "NeonTube")
        self.geometry = built.geometry
        self.baseEmissionIntensity = emissionIntensity

        // ── Neon material: constant-lit black glass + HDR emission ────────────
        let m = SCNMaterial()
        m.lightingModel = .constant            // the tube IS a light, not lit by the scene
        m.diffuse.contents = PlatformColor.black
        m.emission.contents = color
        m.emission.intensity = emissionIntensity
        m.isDoubleSided = false
        m.writesToDepthBuffer = true
        // A faint fresnel-ish rim via multiply on the reflective slot keeps the
        // glass edge reading; cheap and modifier-free (modifiers are finicky here).
        m.multiply.contents = PlatformColor(white: 1.0, alpha: 1)
        geometry.firstMaterial = m
        self.material = m

        let root = SCNNode(geometry: geometry)
        root.name = "neonTube"
        root.castsShadow = false               // a light source shouldn't cast a shadow

        // ── Paired lights along the path ──────────────────────────────────────
        let lc = lightColor ?? color
        let lcc = lc.srgbComponents ?? SIMD4(1, 1, 1, 1)
        var specs: [LightSpec] = []
        var i = 0
        while i < path.count {
            let light = SCNLight()
            light.type = .omni
            light.color = lc
            light.intensity = lightIntensity
            light.attenuationStartDistance = CGFloat(radius * 2)
            light.attenuationEndDistance = CGFloat(radius * 40)
            light.castsShadow = false          // sparse spill; one shadow-caster max if needed
            let ln = SCNNode()
            ln.light = light
            ln.simdPosition = path[i]
            root.addChildNode(ln)
            let scale = Float(lightIntensity) / 600.0
            specs.append(LightSpec(
                position: path[i],
                color: SIMD3(Float(lcc.x), Float(lcc.y), Float(lcc.z)) * (5.0 * scale),
                radius: radius * 40))
            i += max(1, lightSpacing)
        }
        self.lightSpecs = specs
        self.node = root
    }

    /// Subtle buzz/flicker on the emission (call per tick with elapsed seconds).
    /// Default amplitude ±4% at a few Hz — real neon shimmer, not a broken strobe.
    public func flicker(_ time: Float, amplitude: CGFloat = 0.04) {
        let f = 1.0 + amplitude * CGFloat(sin(time * 7.0) * 0.6 + sin(time * 23.0) * 0.4)
        material.emission.intensity = baseEmissionIntensity * f
    }

    // ── Geometry ──────────────────────────────────────────────────────────────

    struct Built {
        let geometry: SCNGeometry
        let positions: [SIMD3<Float>]
        let normals: [SIMD3<Float>]
        let indices: [Int32]
    }

    static func buildGeometry(path: [SIMD3<Float>], radius: Float,
                              radialSegments seg: Int, closed: Bool) -> Built {
        let n = max(6, seg)
        var pts = path
        if pts.count < 2 { pts = [SIMD3(0,0,0), SIMD3(0,1,0)] }

        var pos: [SIMD3<Float>] = []
        var nrm: [SIMD3<Float>] = []
        var idx: [Int32] = []

        // Parallel-transport a frame along the path to avoid twist.
        func tangentAt(_ i: Int) -> SIMD3<Float> {
            let a = pts[max(0, i - 1)], b = pts[min(pts.count - 1, i + 1)]
            let t = b - a
            return simd_length(t) > 1e-6 ? simd_normalize(t) : SIMD3(0, 1, 0)
        }
        var t0 = tangentAt(0)
        // Initial reference perpendicular.
        var up = abs(t0.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
        var right = simd_normalize(simd_cross(t0, up))
        var fwd = t0

        var ringStart: [Int] = []
        for i in 0..<pts.count {
            let t = tangentAt(i)
            // Rotate the frame from the previous tangent to this one (PT).
            let axis = simd_cross(fwd, t)
            let axisLen = simd_length(axis)
            if axisLen > 1e-5 {
                let angle = asin(min(1, axisLen))
                let q = simd_quatf(angle: angle, axis: axis / axisLen)
                right = simd_normalize(q.act(right))
            }
            fwd = t
            up = simd_normalize(simd_cross(right, fwd))

            ringStart.append(pos.count)
            for s in 0..<n {
                let a = Float(s) / Float(n) * 2 * .pi
                let dir = cos(a) * right + sin(a) * up
                pos.append(pts[i] + radius * dir)
                nrm.append(dir)
            }
        }

        // Connect consecutive rings (CW front-face for SceneKit custom geometry).
        let ringCount = pts.count
        let segPairs = closed ? ringCount : ringCount - 1
        for k in 0..<segPairs {
            let a = ringStart[k]
            let b = ringStart[(k + 1) % ringCount]
            for s in 0..<n {
                let sn = (s + 1) % n
                let a0 = Int32(a + s), a1 = Int32(a + sn)
                let b0 = Int32(b + s), b1 = Int32(b + sn)
                idx += [a0, b0, a1]
                idx += [a1, b0, b1]
            }
        }
        // Flat end caps (open tubes only).
        if !closed {
            func cap(ringBase: Int, center: SIMD3<Float>, normal: SIMD3<Float>, flip: Bool) {
                let c = Int32(pos.count); pos.append(center); nrm.append(normal)
                for s in 0..<n {
                    let sn = (s + 1) % n
                    let r0 = Int32(ringBase + s), r1 = Int32(ringBase + sn)
                    if flip { idx += [c, r1, r0] } else { idx += [c, r0, r1] }
                }
            }
            cap(ringBase: ringStart[0], center: pts[0], normal: -tangentAt(0), flip: false)
            cap(ringBase: ringStart[ringCount - 1], center: pts[ringCount - 1],
                normal: tangentAt(ringCount - 1), flip: true)
        }

        let posData = Data(bytes: pos, count: pos.count * MemoryLayout<SIMD3<Float>>.stride)
        let nrmData = Data(bytes: nrm, count: nrm.count * MemoryLayout<SIMD3<Float>>.stride)
        let posSource = SCNGeometrySource(data: posData, semantic: .vertex,
            vectorCount: pos.count, usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: MemoryLayout<SIMD3<Float>>.stride)
        let nrmSource = SCNGeometrySource(data: nrmData, semantic: .normal,
            vectorCount: nrm.count, usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: MemoryLayout<SIMD3<Float>>.stride)
        let element = SCNGeometryElement(indices: idx, primitiveType: .triangles)
        // winding-ok: audited via GeometryWinding.audit in init against these arrays.
        let geo = SCNGeometry(sources: [posSource, nrmSource], elements: [element])
        geo.name = "NeonTube"
        return Built(geometry: geo, positions: pos, normals: nrm, indices: idx)
    }
}
