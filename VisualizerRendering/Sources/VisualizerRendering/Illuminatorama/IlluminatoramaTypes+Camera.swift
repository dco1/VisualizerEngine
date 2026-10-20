import Metal
import simd

/// Vertex format consumed by the G-buffer vertex shader. Tightly packed so the
/// `MTLVertexDescriptor` is unambiguous and matches the `Vertex` struct in the
/// shader byte-for-byte. Phase 4.5 added the tangent + handedness slot to
/// support tangent-space normal-map sampling; stride goes 48 → 64 (33%
/// vertex-buffer growth, applied uniformly because the vertex shader can't
/// branch on per-mesh schema).
public struct IlluminatoramaVertex {
    public var position: SIMD3<Float>
    public var _padPos: Float = 0      // align next SIMD3 to 16
    public var normal: SIMD3<Float>
    public var _padNrm: Float = 0
    public var uv: SIMD2<Float>
    public var _padUv: SIMD2<Float> = .zero  // pad to 16 so stride was 48
    /// Tangent in object space (xyz) + handedness in `w`. The fragment
    /// shader builds the bitangent as `cross(normal, tangent) * w` so a
    /// mirrored mesh's TBN frame stays consistent. Zero tangent = "no
    /// normal-map data"; the fragment shader falls through to interpolated
    /// vertex normal.
    public var tangent: SIMD4<Float>
    /// Phase 4.17 — per-vertex RGBA color, multiplied into the
    /// fragment's albedo at shading time. Scenes that paint pattern
    /// detail (HotAirBalloon's chevron stripes, GiantGummyBears' candy
    /// gradients, anything with a `SCNGeometrySource(semantic: .color)`)
    /// store their pattern here so the deferred pass doesn't lose it.
    /// Default `(1, 1, 1, 1)` is a no-op — the fragment shader's
    /// `albedo *= color.rgb` then leaves the material's diffuse
    /// untouched for meshes that don't ship a colour source. Stride
    /// grows 96 → 112 (one float4 column). Synthesised tangents in the
    /// 4.13a repack kernel default the colour to white too.
    public var color: SIMD4<Float> = SIMD4(1, 1, 1, 1)

    public init(position: SIMD3<Float>,
                normal: SIMD3<Float>,
                uv: SIMD2<Float>,
                tangent: SIMD4<Float> = .zero,
                color: SIMD4<Float> = SIMD4(1, 1, 1, 1)) {
        self.position = position
        self.normal = normal
        self.uv = uv
        self.tangent = tangent
        self.color = color
    }
}

/// Camera state the host supplies each frame. Illuminatorama derives all
/// matrices from this — the host doesn't need to compute view/projection.
public struct IlluminatoramaCamera {
    /// How the camera flattens the world onto the frame.
    ///
    /// `.perspective` is the shipped default and the only mode the renderer had until
    /// the axonometric/diagram look needed a parallel projection. Under `.orthographic`
    /// there is no vanishing point: parallel world lines stay parallel on screen, which
    /// is what makes an isometric drawing read as a drawing rather than as a photograph
    /// taken from far away with a long lens.
    ///
    /// **`fovYRadians` is ignored under `.orthographic`** — the frame is sized by
    /// `halfHeight` in METRES instead of by an angle, so the framing no longer depends
    /// on how far the eye is from the subject. Anything deriving a cone half-angle from
    /// the camera must branch on this (`halfExtents(atViewDistance:)` is the one place
    /// that does; the shadow-cascade fitter reads it).
    public enum Projection: Equatable, Sendable {
        case perspective
        /// Parallel projection framing `halfHeight` metres above and below the view
        /// centre. Frame half-width follows from `aspect`, exactly as it does for the
        /// perspective case.
        case orthographic(halfHeight: Float)
    }

    public var position: SIMD3<Float>
    public var target: SIMD3<Float>
    public var up: SIMD3<Float>
    public var fovYRadians: Float
    public var aspect: Float
    public var zNear: Float
    public var zFar: Float
    /// Perspective (default) or parallel. See `Projection`.
    public var projection: Projection

    public init(
        position: SIMD3<Float>,
        target: SIMD3<Float> = .zero,
        up: SIMD3<Float> = SIMD3(0, 1, 0),
        fovYRadians: Float = .pi / 3,
        aspect: Float = 16.0 / 9.0,
        zNear: Float = 0.1,
        zFar: Float = 200,
        projection: Projection = .perspective
    ) {
        self.position = position
        self.target = target
        self.up = up
        self.fovYRadians = fovYRadians
        self.aspect = aspect
        self.zNear = zNear
        self.zFar = zFar
        self.projection = projection
    }

    /// True when this camera projects in parallel. Read it rather than pattern-matching
    /// at call sites, so a future third projection doesn't have to be found by grep.
    public var isOrthographic: Bool {
        if case .orthographic = projection { return true }
        return false
    }

    /// Half-height and half-width of the view frustum at view-space distance `d` — the
    /// ONE place the two projections' framing differs, so anything that needs frustum
    /// extents (cascade fitting, culling) gets both modes right by construction rather
    /// than by each call site remembering to branch.
    ///
    /// Perspective: extents grow linearly with distance (`d · tan(fov/2)`).
    /// Orthographic: extents are CONSTANT — that is the whole definition of parallel.
    public func halfExtents(atViewDistance d: Float) -> (height: Float, width: Float) {
        switch projection {
        case .perspective:
            let h = d * tan(fovYRadians * 0.5)
            return (h, h * aspect)
        case .orthographic(let halfHeight):
            return (halfHeight, halfHeight * aspect)
        }
    }

    public var viewMatrix: simd_float4x4 {
        let f = simd_normalize(target - position)
        let s = simd_normalize(simd_cross(f, up))
        let u = simd_cross(s, f)
        return simd_float4x4(
            SIMD4( s.x,  u.x, -f.x, 0),
            SIMD4( s.y,  u.y, -f.y, 0),
            SIMD4( s.z,  u.z, -f.z, 0),
            SIMD4(-simd_dot(s, position),
                  -simd_dot(u, position),
                   simd_dot(f, position), 1)
        )
    }

    public var projectionMatrix: simd_float4x4 {
        let zRange = zFar - zNear
        switch projection {
        case .perspective:
            // Right-handed perspective with depth range [0, 1] (Metal-conventional).
            let yScale = 1 / tan(fovYRadians * 0.5)
            let xScale = yScale / aspect
            return simd_float4x4(
                SIMD4(xScale, 0, 0, 0),
                SIMD4(0, yScale, 0, 0),
                SIMD4(0, 0, -zFar / zRange, -1),
                SIMD4(0, 0, -(zFar * zNear) / zRange, 0)
            )
        case .orthographic(let halfHeight):
            // Right-handed parallel projection, same [0, 1] depth convention. Note the
            // w row is (0,0,0,1), not (0,0,-1,0): clip.w is a constant 1, so the
            // perspective divide is a no-op and screen size stops depending on depth.
            // Centred on the view axis, so left/right and bottom/top are symmetric and
            // the general off-centre ortho form collapses to two scales + a depth bias.
            let hH = max(halfHeight, 1e-4)
            let hW = max(hH * aspect, 1e-4)
            return simd_float4x4(
                SIMD4(1 / hW, 0, 0, 0),
                SIMD4(0, 1 / hH, 0, 0),
                SIMD4(0, 0, -1 / zRange, 0),
                SIMD4(0, 0, -zNear / zRange, 1)
            )
        }
    }
}


// plausibility: real — an analytic ray-box room behind a pane, lit by a point lamp (inverse-square
// × cosine, exact for an empty convex box) + a uniform one-bounce term. See
// `IlluminatoramaRenderer.interiorMappedMeshKinds` and IlluminatoramaGBuffer.metal
// `interiorMapRadiance`.
/// Interior mapping (van Dongen 2008) for one mesh kind: the room box behind each pane.
public struct IlluminatoramaInteriorMapping: Equatable, Sendable {
    /// Room width, height, mean depth (m).
    public var roomSize: SIMD3<Float>
    /// Floor → the pane's bottom edge (m).
    public var sillHeight: Float
    /// The pane's width × height (m) — what its UVs span.
    public var paneSize: SIMD2<Float>
    /// Subtracted from uv.x before use (a host that biases u, e.g. to keep it positive).
    public var uvBias: Float
    /// ± fraction of the depth each window's room varies by (hashed per window).
    public var depthJitter: Float
    /// Wall albedo (linear); floor and ceiling as scales of it.
    public var wallAlbedo: SIMD3<Float>
    public var floorScale: Float
    public var ceilingScale: Float
    /// The lamp: height above the floor, inset from the back wall (m).
    public var lampHeight: Float
    public var lampInset: Float
    /// Uniform one-bounce term, as a fraction of the back wall's direct light.
    public var ambient: Float

    public init(roomSize: SIMD3<Float> = SIMD3(3.6, 2.5, 4.0), sillHeight: Float = 0.8,
                paneSize: SIMD2<Float>, uvBias: Float = 0, depthJitter: Float = 0.25,
                wallAlbedo: SIMD3<Float> = SIMD3(0.55, 0.50, 0.44), floorScale: Float = 0.45,
                ceilingScale: Float = 1.3, lampHeight: Float = 1.1, lampInset: Float = 0.7,
                ambient: Float = 0.12) {
        self.roomSize = roomSize; self.sillHeight = sillHeight; self.paneSize = paneSize
        self.uvBias = uvBias; self.depthJitter = depthJitter; self.wallAlbedo = wallAlbedo
        self.floorScale = floorScale; self.ceilingScale = ceilingScale
        self.lampHeight = lampHeight; self.lampInset = lampInset; self.ambient = ambient
    }

    /// Mirror of the shader's `InteriorMapParams` (four float4s).
    struct GPU { var room: SIMD4<Float>; var pane: SIMD4<Float>; var wall: SIMD4<Float>; var lamp: SIMD4<Float> }
    var gpu: GPU {
        GPU(room: SIMD4(roomSize.x, roomSize.y, roomSize.z, sillHeight),
            pane: SIMD4(paneSize.x, paneSize.y, uvBias, depthJitter),
            wall: SIMD4(wallAlbedo, floorScale),
            lamp: SIMD4(lampHeight, lampInset, ceilingScale, ambient))
    }
}
