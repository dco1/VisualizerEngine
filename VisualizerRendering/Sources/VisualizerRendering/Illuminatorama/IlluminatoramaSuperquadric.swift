import Foundation
import Metal
import OSLog
import VisualizerCore
import simd

// ── Perfect analytic superquadric — the reusable hero primitive ──────────────
//
// A superquadric is offered to native Illuminatorama scenes as a SINGLE shape
// that becomes sphere / ellipsoid / rounded-box / capsule / cylinder via three
// extents + two roundness exponents — rendered the CORRECT way (ray–surface
// intersection in a fragment shader, not a tessellated mesh), so its silhouette
// is mathematically exact at any zoom.
//
// One logical object is drawn as TWO instances sharing one `modelMatrix`:
//   1. IMPOSTOR — a [-1,1] bounding box drawn with the impostor pipeline; the
//      fragment ray-traces the analytic surface and writes the G-buffer +
//      analytic depth + analytic motion vectors. This is the perfect camera view.
//   2. RT PROXY — a moderate-tessellation triangle mesh, raster-skipped, that
//      lives only in the TLAS so the object still casts RT shadows / appears in
//      RT GI & reflections (the RT path is hardware-triangle-only).
//
// Extents fold into the `modelMatrix` scale, so object space is always the UNIT
// superquadric (a=b=c=1). The impostor's ellipsoid fast path and its
// `normalize(localHit/extents²)` normal then fall straight out of the existing
// `modelMatrix` / `normalMatrix` — no special-casing.
//
// Convention (matched byte-for-byte by `sqField`/`SQParam` in Illuminatorama.metal):
//   F(x,y,z) = (|x|^(2/e2) + |y|^(2/e2))^(e2/e1) + |z|^(2/e1)   (z is the polar axis)
//   surface = F==1 ; e1==e2==1 → unit sphere. e1 = polar squareness, e2 = equatorial.

private let sqLog = Logger(subsystem: "com.visualizer-engine", category: "superquadric")

extension IlluminatoramaMesh {

    /// Axis-aligned box spanning [-1,1]³ (per-face flat normals). The impostor's
    /// bounding proxy: scaled by the instance's semi-axis extents it bounds the
    /// analytic surface exactly, and the fragment intersection does the rest.
    public static func biunitBox(device: MTLDevice) -> IlluminatoramaMesh {
        let h: Float = 1.0
        let faces: [(SIMD3<Float>, [SIMD3<Float>])] = [
            (SIMD3( 1, 0, 0), [SIMD3( h,-h, h), SIMD3( h,-h,-h), SIMD3( h, h,-h), SIMD3( h, h, h)]),
            (SIMD3(-1, 0, 0), [SIMD3(-h,-h,-h), SIMD3(-h,-h, h), SIMD3(-h, h, h), SIMD3(-h, h,-h)]),
            (SIMD3( 0, 1, 0), [SIMD3(-h, h, h), SIMD3( h, h, h), SIMD3( h, h,-h), SIMD3(-h, h,-h)]),
            (SIMD3( 0,-1, 0), [SIMD3(-h,-h,-h), SIMD3( h,-h,-h), SIMD3( h,-h, h), SIMD3(-h,-h, h)]),
            (SIMD3( 0, 0, 1), [SIMD3(-h,-h, h), SIMD3( h,-h, h), SIMD3( h, h, h), SIMD3(-h, h, h)]),
            (SIMD3( 0, 0,-1), [SIMD3( h,-h,-h), SIMD3(-h,-h,-h), SIMD3(-h, h,-h), SIMD3( h, h,-h)]),
        ]
        var verts: [IlluminatoramaVertex] = []
        var indices: [UInt16] = []
        for (normal, quad) in faces {
            let base = UInt16(verts.count)
            for (i, p) in quad.enumerated() {
                let uv = SIMD2<Float>(Float(i & 1), Float((i >> 1) & 1))
                verts.append(IlluminatoramaVertex(position: p, normal: normal, uv: uv))
            }
            indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
        }
        return IlluminatoramaMesh(device: device, vertices: verts, indices: indices)
    }

    /// Superquadric inside–outside field F (object space, unit shape). Mirrors
    /// `sqField` in Illuminatorama.metal so the proxy and the impostor agree.
    static func superquadricField(_ p: SIMD3<Float>, e1: Float, e2: Float) -> Float {
        let ax = powf(abs(p.x), 2.0 / e2)
        let ay = powf(abs(p.y), 2.0 / e2)
        let az = powf(abs(p.z), 2.0 / e1)
        return powf(ax + ay, e2 / e1) + az
    }

    /// Moderate-tessellation UNIT superquadric (a=b=c=1) for the RT proxy. Same
    /// lat-long topology + outward-CCW winding as `unitSphere`; positions use the
    /// signed-power superellipsoid parameterization; normals are the central-
    /// difference gradient of F (robust at the axis/pole singularities where the
    /// analytic derivative of |x|^p, p<1, blows up). Runs `MeshSoundness` +
    /// `GeometryWinding` audits once (logged) so a bad winding/degenerate count
    /// shows up before the mesh ships.
    public static func superquadricProxy(
        device: MTLDevice,
        e1: Float,
        e2: Float,
        meridians: Int = 48,
        parallels: Int = 24
    ) -> IlluminatoramaMesh {
        let (verts, indices) = superquadricProxyGeometry(
            e1: e1, e2: e2, meridians: meridians, parallels: parallels)

        // Audit once (logged, non-blocking) — the proxy is closed/watertight.
        let positions = verts.map(\.position)
        let normals = verts.map(\.normal)
        let idx32 = indices.map(UInt32.init)
        let label = "superquadricProxy(e1=\(e1),e2=\(e2))"
        let report = MeshSoundness.audit(positions: positions, indices: idx32,
                                         normals: normals, label: label, expectClosed: true)
        let windFrac = GeometryWinding.audit(positions: positions, normals: normals,
                                             indices: idx32, label: label)
        sqLog.info("\(label): \(verts.count) verts, degenerateTris=\(report.degenerateTriangles), windingInconsistent=\(String(format: "%.1f%%", windFrac * 100))")

        return IlluminatoramaMesh(device: device, vertices: verts, indices: indices)
    }

    /// Pure CPU geometry for the unit superquadric proxy (no device, no audit) —
    /// the testable core of `superquadricProxy`. Positions use the signed-power
    /// superellipsoid parameterization; normals are the central-difference
    /// gradient of `superquadricField`, so the proxy and the in-fragment impostor
    /// share one surface definition.
    public static func superquadricProxyGeometry(
        e1: Float,
        e2: Float,
        meridians: Int = 48,
        parallels: Int = 24
    ) -> ([IlluminatoramaVertex], [UInt16]) {
        // sign-preserving power: sign(cos)·|cos|^m etc.
        func cpow(_ t: Float, _ m: Float) -> Float {
            let c = cosf(t); return (c < 0 ? -1 : 1) * powf(abs(c), m)
        }
        func spow(_ t: Float, _ m: Float) -> Float {
            let s = sinf(t); return (s < 0 ? -1 : 1) * powf(abs(s), m)
        }
        func gradF(_ p: SIMD3<Float>) -> SIMD3<Float> {
            let h: Float = 2e-3
            let gx = superquadricField(p + SIMD3(h,0,0), e1: e1, e2: e2)
                   - superquadricField(p - SIMD3(h,0,0), e1: e1, e2: e2)
            let gy = superquadricField(p + SIMD3(0,h,0), e1: e1, e2: e2)
                   - superquadricField(p - SIMD3(0,h,0), e1: e1, e2: e2)
            let gz = superquadricField(p + SIMD3(0,0,h), e1: e1, e2: e2)
                   - superquadricField(p - SIMD3(0,0,h), e1: e1, e2: e2)
            let g = SIMD3<Float>(gx, gy, gz)
            let len = simd_length(g)
            return len > 1e-8 ? g / len : SIMD3(0, 0, 1)
        }

        var verts: [IlluminatoramaVertex] = []
        verts.reserveCapacity((parallels + 1) * (meridians + 1))
        for j in 0...parallels {
            let v = Float(j) / Float(parallels)
            let eta = .pi / 2 - v * .pi          // +π/2 (north, z=+1) → -π/2 (south)
            for i in 0...meridians {
                let u = Float(i) / Float(meridians)
                let omega = u * 2 * .pi - .pi     // -π → +π
                // z is the polar axis (matches the F convention).
                let pos = SIMD3<Float>(
                    cpow(eta, e1) * cpow(omega, e2),
                    cpow(eta, e1) * spow(omega, e2),
                    spow(eta, e1)
                )
                verts.append(IlluminatoramaVertex(position: pos, normal: gradF(pos), uv: SIMD2(u, v)))
            }
        }
        var indices: [UInt16] = []
        let stride = meridians + 1
        for j in 0..<parallels {
            for i in 0..<meridians {
                let a = UInt16(j * stride + i)
                let b = UInt16(j * stride + i + 1)
                let c = UInt16((j + 1) * stride + i)
                let d = UInt16((j + 1) * stride + i + 1)
                // Reversed vs unitSphere's [a,b,c,b,d,c]: this parameterization's
                // (latitude north→south, longitude −π→+π) grid is opposite-handed,
                // so CCW-from-outside is the swapped order (verified by
                // GeometryWinding.audit vs the outward ∇F normals).
                indices.append(contentsOf: [a, c, b, b, c, d])
            }
        }
        return (verts, indices)
    }
}

extension IlluminatoramaRenderer {

    /// The two instances that make up one perfect analytic superquadric. A scene
    /// appends `refs` to `renderer.instances` (re-driven each tick, like any other
    /// instance). The impostor + proxy share `modelMatrix`, so they stay aligned.
    public struct SuperquadricHandle {
        public var impostor: InstanceRef
        public var proxy: InstanceRef
        /// Append both to the renderer's instance list. The impostor must keep a
        /// STABLE slot frame-to-frame (TAA grouping invariant) — drive the same
        /// handle each tick rather than reordering.
        public var refs: [InstanceRef] { [impostor, proxy] }
    }

    /// Build a perfect analytic superquadric. `extents` are the world semi-axes;
    /// `(e1, e2)` the roundness exponents (1,1 = ellipsoid/sphere; small = box;
    /// >1 = pinched). `transform` is the object's world placement (rotation +
    /// translation); extents are folded into the model scale internally.
    ///
    /// Lazily registers the shared impostor bounding-box mesh and a per-roundness
    /// proxy mesh, and records their kinds so the G-buffer pass routes them
    /// correctly. Returns a handle whose `refs` the scene appends to `instances`.
    public func makeSuperquadric(
        extents: SIMD3<Float>,
        e1: Float,
        e2: Float,
        transform: simd_float4x4 = matrix_identity_float4x4,
        albedo: SIMD3<Float> = SIMD3(0.8, 0.8, 0.8),
        metallic: Float = 0.0,
        roughness: Float = 0.4,
        emission: SIMD3<Float> = .zero
    ) -> SuperquadricHandle {
        let ce1 = min(max(e1, 0.05), 8.0)
        let ce2 = min(max(e2, 0.05), 8.0)

        // Lazily register the shared impostor box (extent-independent).
        let boxKind = MeshKind.custom("__sqImpostorBox")
        if !impostorMeshKinds.contains(boxKind) {
            setMesh(boxKind, .biunitBox(device: device))
            impostorMeshKinds.insert(boxKind)
        }
        // Lazily register the proxy for this roundness pair (one mesh serves all
        // sizes — extents ride in the modelMatrix, so the BLAS is reused).
        let key = String(format: "__sqProxy:%.3f_%.3f", ce1, ce2)
        let proxyKind = MeshKind.custom(key)
        if !rtProxyMeshKinds.contains(proxyKind) {
            setMesh(proxyKind, .superquadricProxy(device: device, e1: ce1, e2: ce2))
            rtProxyMeshKinds.insert(proxyKind)
        }

        // Shared model matrix: object-space unit shape → scale by extents → place.
        var scale = matrix_identity_float4x4
        scale.columns.0.x = extents.x
        scale.columns.1.y = extents.y
        scale.columns.2.z = extents.z
        let model = transform * scale

        var impostorData = IlluminatoramaInstance(
            modelMatrix: model, albedo: albedo, metallic: metallic,
            roughness: roughness, emission: emission)
        impostorData.rtExclude = 1   // the box is raster-only; RT sees the proxy

        let proxyData = IlluminatoramaInstance(
            modelMatrix: model, albedo: albedo, metallic: metallic,
            roughness: roughness, emission: emission)

        let impostor = InstanceRef(meshKind: boxKind, data: impostorData,
                                   superquadricShape: SIMD2(ce1, ce2))
        let proxy = InstanceRef(meshKind: proxyKind, data: proxyData)
        return SuperquadricHandle(impostor: impostor, proxy: proxy)
    }
}
