#include <metal_stdlib>
using namespace metal;

// ── EggMotion.metal ──────────────────────────────────────────────────────────
//
// Advances per-egg arc-length, samples the pre-baked network LUT, builds the
// egg's world-space orientation matrix, and writes a 3×4 transform per egg.
// Replaces the CPU `placeEgg` + `network.sample` hot path for the Eggs scene.
//
// The LUT layout — see `EggsRails.Network.bakeLUT()`:
//
//   `lut[4*i + 0].xyz` = pos at arc-length `i * step`
//   `lut[4*i + 1].xyz` = tangent (unit)
//   `lut[4*i + 2].xyz` = right (unit)
//   `lut[4*i + 3].xyz` = up (unit)
//
// Linear interpolation between adjacent slots gives sub-metre accuracy at
// the 0.1 m step the controller bakes with.
//
// Two motion modes mirror the CPU enum:
//   0 = roll  — barrel pose, rolls around `right` with `rollAccum`
//   1 = glide — torpedo pose, points along `tangent`
//
// Swift mirror: `EggMotionState`, `EggMotionUniforms`, `EggTransform` in
// EggMotionKernel.swift. Same alignment rule (float4 only in shared structs).

struct EggMotionState {
    float s;             // arc length on the loop
    float speed;         // metres/sec
    float liftCurrent;   // current cradle-up offset (m)
    float rollOffset;    // initial roll seed
    float rollAccum;     // accumulated roll about `right`
    float rollingRadius; // metres — drives roll per metre advanced
    uint  motionMode;    // 0=roll, 1=glide
    uint  povBoost;      // 1 = apply pov-boost uniform, else 0
    float scale;         // uniform per-egg scale (clobber-safe simdTransform)
    float _pad0;
    float _pad1;
    float _pad2;
};

struct EggMotionUniforms {
    float dt;
    float globalSpeed;
    float povBoost;       // 1.0 = no boost; >1.0 = chaser boost
    float loopLength;
    float lutStep;        // metres between LUT entries
    uint  lutSlotCount;
    uint  eggCount;
    float _pad0;
};

/// 3×4 world transform written for each egg. Row-major (matches simd float4x4
/// column convention via row-vector packing — see Swift mirror for details).
struct EggTransform {
    float4 col0;  // (rotation0.xyz, 0)
    float4 col1;  // (rotation1.xyz, 0)
    float4 col2;  // (rotation2.xyz, 0)
    float4 col3;  // (position.xyz, 1)
};

// Linear interpolation between two LUT slots.
static void sampleLUT(
    float s,
    constant EggMotionUniforms& U,
    device const float4* lut,
    thread float3& pos,
    thread float3& tangent,
    thread float3& right,
    thread float3& up
) {
    // Wrap arc-length into [0, loopLength)
    float wrapped = fmod(s, U.loopLength);
    if (wrapped < 0.0) wrapped += U.loopLength;

    float slotF = wrapped / U.lutStep;
    uint i0 = uint(floor(slotF));
    uint i1 = i0 + 1u;
    if (i0 >= U.lutSlotCount) i0 = U.lutSlotCount - 1u;
    if (i1 >= U.lutSlotCount) i1 = 0u;  // wrap to first slot
    float frac = slotF - float(i0);

    float3 p0 = lut[i0 * 4u + 0u].xyz;
    float3 t0 = lut[i0 * 4u + 1u].xyz;
    float3 r0 = lut[i0 * 4u + 2u].xyz;
    float3 u0 = lut[i0 * 4u + 3u].xyz;
    float3 p1 = lut[i1 * 4u + 0u].xyz;
    float3 t1 = lut[i1 * 4u + 1u].xyz;
    float3 r1 = lut[i1 * 4u + 2u].xyz;
    float3 u1 = lut[i1 * 4u + 3u].xyz;

    pos     = mix(p0, p1, frac);
    tangent = normalize(mix(t0, t1, frac));
    right   = normalize(mix(r0, r1, frac));
    up      = normalize(mix(u0, u1, frac));
}

// Quaternion to 3×3 rotation. Standard formula.
static float3x3 quatToMat3(float4 q) {
    float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    return float3x3(
        float3(1.0 - 2.0 * (yy + zz),     2.0 * (xy + wz),     2.0 * (xz - wy)),
        float3(    2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz),     2.0 * (yz + wx)),
        float3(    2.0 * (xz + wy),     2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy))
    );
}

// Axis-angle to quaternion.
static float4 axisAngleQuat(float3 axis, float angle) {
    float h = angle * 0.5;
    float s = sin(h);
    return float4(axis * s, cos(h));
}

// Quat × Quat (Hamilton product). q.xyz = vector part, q.w = scalar.
static float4 quatMul(float4 a, float4 b) {
    return float4(
        a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz)
    );
}

// Build a quaternion from a 3×3 orthonormal basis (columns = basis vectors).
// Shepherd's method — stable across all orientations.
static float4 quatFromMat3(float3x3 m) {
    float trace = m[0][0] + m[1][1] + m[2][2];
    float4 q;
    if (trace > 0.0) {
        float s = sqrt(trace + 1.0) * 2.0;
        q.w = 0.25 * s;
        q.x = (m[1][2] - m[2][1]) / s;
        q.y = (m[2][0] - m[0][2]) / s;
        q.z = (m[0][1] - m[1][0]) / s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        float s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        q.w = (m[1][2] - m[2][1]) / s;
        q.x = 0.25 * s;
        q.y = (m[1][0] + m[0][1]) / s;
        q.z = (m[2][0] + m[0][2]) / s;
    } else if (m[1][1] > m[2][2]) {
        float s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        q.w = (m[2][0] - m[0][2]) / s;
        q.x = (m[1][0] + m[0][1]) / s;
        q.y = 0.25 * s;
        q.z = (m[2][1] + m[1][2]) / s;
    } else {
        float s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        q.w = (m[0][1] - m[1][0]) / s;
        q.x = (m[2][0] + m[0][2]) / s;
        q.y = (m[2][1] + m[1][2]) / s;
        q.z = 0.25 * s;
    }
    return normalize(q);
}

// ── Instanced-render expand kernel ───────────────────────────────────────────
//
// Reads the unit-egg position + normal sources (shared across every instance)
// + the per-instance `EggTransform` output of `eggs_advance_motion`, and
// writes the transformed vertices for every instance into a big mesh's
// position + normal buffers. Result: one SCNGeometry with all eggs' vertices
// in it, one draw call.
//
// One thread per (instance, vertex_in_instance). The grid is sized to
// `eggCount * vertsPerInstance`; threads past that range early-out.
//
// Vertex colour is set ONCE at populate time (per-egg base color baked into
// per-vertex colours), so no per-tick work needed for colour.

struct EggExpandUniforms {
    uint  eggCount;
    uint  vertsPerInstance;
    float _pad0;
    float _pad1;
};

kernel void eggs_expand_instances(
    constant EggExpandUniforms& U      [[buffer(0)]],
    device const EggTransform* xforms  [[buffer(1)]],
    device const float4* unitPos       [[buffer(2)]],   // xyz = pos, w unused
    device const float4* unitNrm       [[buffer(3)]],   // xyz = normal, w unused
    device float4* outPos              [[buffer(4)]],   // xyz = pos (w unused, stride 16)
    device float4* outNrm              [[buffer(5)]],   // xyz = normal (w unused, stride 16)
    uint gid                           [[thread_position_in_grid]]
) {
    if (U.vertsPerInstance == 0u) return;
    uint instance = gid / U.vertsPerInstance;
    uint vertIdx  = gid - instance * U.vertsPerInstance;
    if (instance >= U.eggCount) return;

    EggTransform M = xforms[instance];
    // M.col0..col2 already have the per-egg uniform scale baked in;
    // M.col3 is the world position. The scale we recover for transforming
    // normals is M_basis's column length (= |scale|).
    float3x3 basis = float3x3(M.col0.xyz, M.col1.xyz, M.col2.xyz);
    float3 worldPos = basis * unitPos[vertIdx].xyz + M.col3.xyz;
    // For normals we want the rotation only (basis / scale), then re-norm.
    float3 nLocal = unitNrm[vertIdx].xyz;
    float3 nWorld = normalize(basis * nLocal);

    uint outIdx = instance * U.vertsPerInstance + vertIdx;
    outPos[outIdx] = float4(worldPos, 0.0);
    outNrm[outIdx] = float4(nWorld, 0.0);
}

kernel void eggs_advance_motion(
    constant EggMotionUniforms& U          [[buffer(0)]],
    device   EggMotionState* state         [[buffer(1)]],
    device   EggTransform*   transforms    [[buffer(2)]],
    device const float4*     lut           [[buffer(3)]],
    uint eggIdx                            [[thread_position_in_grid]]
) {
    if (eggIdx >= U.eggCount) return;

    EggMotionState st = state[eggIdx];

    // Advance arc-length. POV-egg boost is gated by the per-egg flag set
    // CPU-side (the controller knows which slot is the chaser).
    float boost = (st.povBoost == 1u) ? U.povBoost : 1.0;
    float ds = st.speed * boost * U.globalSpeed * U.dt;
    st.s += ds;

    // Accumulate roll for barrel mode. Sign matches the CPU path: forward
    // motion rolls the top toward +tangent.
    float dRoll = -ds / max(0.001, st.rollingRadius);
    st.rollAccum += dRoll;

    // Sample LUT for the new arc-length.
    float3 pos, tangent, right, up;
    sampleLUT(st.s, U, lut, pos, tangent, right, up);

    float3 worldPos = pos + up * st.liftCurrent;

    // Orientation depends on motionMode.
    float3x3 basis;
    if (st.motionMode == 0u) {
        // Barrel pose: local Y → world right; egg lies sideways across rails.
        // Then roll about `right` by (rollOffset + rollAccum).
        basis = float3x3(-up, right, tangent);
        float angle = st.rollOffset + st.rollAccum;
        float4 baseQ = quatFromMat3(basis);
        float4 rollQ = axisAngleQuat(right, angle);
        float4 q = quatMul(rollQ, baseQ);
        basis = quatToMat3(q);
    } else {
        // Torpedo pose: local Y → tangent, pointy end forward.
        basis = float3x3(-up, tangent, right);
    }

    // Bake uniform scale into the basis columns so SceneKit gets the
    // node's per-egg size back when the controller writes simdTransform.
    // Without this, simdTransform=M (a pure rotation+translation) clobbers
    // node.scale and every egg shrinks to its unit size.
    float sc = st.scale;
    transforms[eggIdx] = EggTransform{
        float4(basis[0] * sc, 0.0),
        float4(basis[1] * sc, 0.0),
        float4(basis[2] * sc, 0.0),
        float4(worldPos, 1.0)
    };

    state[eggIdx] = st;
}
