import Foundation
import Metal
import OSLog
import VisualizerCore
import simd

/// Surface-patch extraction for **native Illuminatorama scenes**, so the
/// deterministic geometry auditors can see them at all.
///
/// ## Why this exists
///
/// `SceneAudit` builds its input by walking `SCNNode`s and reading their world
/// bounding boxes. A native Illuminatorama scene has no such tree — its geometry
/// is an array of `IlluminatoramaRenderer.InstanceRef`, uploaded straight to the
/// GPU — so the walk finds nothing and every check reports a clean, empty scene.
/// That is a silent false PASS, which is worse than no check: `scene-audit.sh`
/// prints a verdict for a scene it never looked at.
///
/// ## Why it is FACES, not objects
///
/// The SceneKit path emits one patch per drawn object and only for objects that
/// are already flat (`minExtent < 20 mm`), i.e. decals and thin panels laid on a
/// surface. That model cannot express the most common z-fight in a built scene:
/// two SOLID boxes that share a face plane. A facade with a shopfront box buried
/// in it, a cornice sunk into a parapet, a sign plate flush with a wall — none of
/// those are flat objects, so none of them ever produced a patch, and the pair
/// went unreported on the SceneKit path too. (Measured: City Street Ultra's
/// shopfronts sat exactly coplanar with their brick facades and the detector saw
/// zero patches for them.)
///
/// So this emits **six patches per instance** — one per face of the instance's
/// oriented bounding box — and lets `ZFightDetector` do the pairing it already
/// does. A box-vs-box coplanar face lands as an ordinary patch pair.
///
/// ## Same-facing vs back-to-back
///
/// Two coplanar faces pointing in OPPOSITE directions are ordinary construction:
/// a kerb abutting a slab, a cornice resting on a mass. The back one is culled
/// and nothing fights. The defect is two coplanar faces pointing the SAME way —
/// both rasterised, both writing the same depth, winner chosen per pixel. Callers
/// should pass `requireSameFacing: true` to `ZFightDetector.audit` for this
/// patch set, or the report drowns in legal contact.
public enum IlluminatoramaSurfaceAudit {

    private static let log = Logger(subsystem: AppLog.subsystem,
                                    category: "illuminatoramaSurfaceAudit")

    /// One instance's contribution: its six OBB face patches.
    ///
    /// - Parameters:
    ///   - model: the instance's model matrix.
    ///   - normalMatrix: its inverse-transpose (already carried on every
    ///     `IlluminatoramaInstance`), used so a non-uniformly scaled box still
    ///     reports perpendicular face normals.
    ///   - localMin/localMax: the mesh's object-space AABB.
    ///   - name: label used in the report.
    public static func facePatches(model: simd_float4x4,
                                   normalMatrix: simd_float4x4,
                                   localMin: SIMD3<Float>,
                                   localMax: SIMD3<Float>,
                                   name: String,
                                   owner: Int? = nil) -> [SurfacePatch] {
        let centre = (localMin + localMax) * 0.5
        let half = (localMax - localMin) * 0.5
        guard half.x.isFinite, half.y.isFinite, half.z.isFinite else { return [] }

        let m3 = simd_float3x3(SIMD3(model.columns.0.x, model.columns.0.y, model.columns.0.z),
                               SIMD3(model.columns.1.x, model.columns.1.y, model.columns.1.z),
                               SIMD3(model.columns.2.x, model.columns.2.y, model.columns.2.z))
        let n3 = simd_float3x3(SIMD3(normalMatrix.columns.0.x, normalMatrix.columns.0.y, normalMatrix.columns.0.z),
                               SIMD3(normalMatrix.columns.1.x, normalMatrix.columns.1.y, normalMatrix.columns.1.z),
                               SIMD3(normalMatrix.columns.2.x, normalMatrix.columns.2.y, normalMatrix.columns.2.z))

        var out: [SurfacePatch] = []
        out.reserveCapacity(6)
        let axes: [(SIMD3<Float>, Int)] = [(SIMD3(1, 0, 0), 0), (SIMD3(0, 1, 0), 1), (SIMD3(0, 0, 1), 2)]
        for (axis, k) in axes {
            let hk = half[k]
            // The two in-plane axes, for the face's lateral footprint.
            let a = axes[(k + 1) % 3].0 * half[(k + 1) % 3]
            let b = axes[(k + 2) % 3].0 * half[(k + 2) % 3]
            // The face's true in-plane half-extents, transformed to world. These
            // are what makes the overlap test a rectangle test — a circumscribed
            // radius on a 136 m roadway "overlaps" the whole scene.
            let halfU = m3 * a
            let halfV = m3 * b
            let radius = simd_length(halfU + halfV)   // fallback only
            // A degenerate face has no area to fight over. Flat meshes (a ground
            // quad is zero-thick) produce four of these per instance, and they
            // would pair with every other degenerate face in the scene.
            guard simd_length(halfU) > 1e-5, simd_length(halfV) > 1e-5 else { continue }
            // A SHEET (zero extent along this axis — a ground quad, a decal
            // plane) has ONE surface, not two coincident opposite ones. Emitting
            // both gives every sheet a phantom underside that sits coplanar with
            // the underside of everything resting on it, and those pair up as
            // same-facing fights that no camera can ever see: the road's
            // downward face against the downward face of every lane dash and
            // manhole lying on it (measured: 295 findings, all of them this).
            // The scene is never viewed from below the ground.
            let signs: [Float] = hk > 1e-5 ? [1, -1] : [1]
            for sign in signs {
                let localCentre = centre + axis * (sign * hk)
                let world4 = model * SIMD4<Float>(localCentre, 1)
                let worldN = n3 * (axis * sign)
                guard simd_length(worldN) > 1e-9 else { continue }
                out.append(SurfacePatch(name: "\(name)\(faceSuffix(k, sign))",
                                        center: SIMD3(world4.x, world4.y, world4.z),
                                        normal: worldN,
                                        radius: radius,
                                        halfU: halfU, halfV: halfV, owner: owner))
            }
        }
        return out
    }

    /// A cheap identity for a transform: translation plus the three basis
    /// vectors' lengths, quantised. Two instances that agree on all of it are
    /// the same placed object as far as this audit is concerned.
    private static func transformKey(_ m: simd_float4x4) -> SIMD8<Float> {
        func q(_ v: Float) -> Float { (v * 1000).rounded() / 1000 }
        let c0 = SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z)
        let c1 = SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z)
        let c2 = SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z)
        return SIMD8(q(m.columns.3.x), q(m.columns.3.y), q(m.columns.3.z),
                     q(simd_length(c0)), q(simd_length(c1)), q(simd_length(c2)),
                     q(c0.x), q(c2.z))
    }

    private static func faceSuffix(_ axis: Int, _ sign: Float) -> String {
        let names = ["x", "y", "z"]
        return ".\(sign > 0 ? "+" : "-")\(names[axis])"
    }

    /// Face patches for a whole instance list.
    ///
    /// `bounds` resolves a mesh kind to its object-space AABB; instances whose
    /// mesh has no resolvable bounds are skipped (and counted in the log rather
    /// than silently dropped, because a skipped instance is an unaudited one).
    public static func facePatches<Kind: Hashable>(
        instances: [(kind: Kind, model: simd_float4x4, normalMatrix: simd_float4x4)],
        bounds: (Kind) -> (min: SIMD3<Float>, max: SIMD3<Float>)?,
        name: (Int) -> String = { "instance\($0)" }
    ) -> [SurfacePatch] {
        var out: [SurfacePatch] = []
        var skipped = 0
        var owners: [SIMD8<Float>: Int] = [:]
        out.reserveCapacity(instances.count * 6)
        for (i, inst) in instances.enumerated() {
            guard let b = bounds(inst.kind) else { skipped += 1; continue }
            // Instances that share a transform are ONE object (a car's body,
            // glass, wheels and lamps ride the same matrix), so they group under
            // one owner and never report against each other.
            let key = transformKey(inst.model)
            let owner = owners[key] ?? { owners[key] = owners.count; return owners.count - 1 }()
            out.append(contentsOf: facePatches(model: inst.model,
                                               normalMatrix: inst.normalMatrix,
                                               localMin: b.min, localMax: b.max,
                                               name: name(i), owner: owner))
        }
        if skipped > 0 {
            log.warning("surface audit: \(skipped, privacy: .public) instance(s) skipped — no mesh bounds")
        }
        return out
    }
}
