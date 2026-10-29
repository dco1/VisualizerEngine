// SimSettle — the GPU half of the shared rest contract (SimSettling.swift).
//
// Two tiny kernels, run at the end of a solver's step:
//   simSettleReduce   max speed over a particle (or body) buffer, into one word.
//   simSettleGate     folds that speed into the solver's rest state: seconds spent under the policy speed, and
//                     whether the solver is frozen. The CPU reads ONE word of this state on demand (`isSettled`)
//                     and, like CoinDEM's whole-pile sleep, skips encoding a frozen solver.
//
// ONE reduction serves every solver. The caller says where a record's position and previous position live
// (byte stride + offsets), so PBDParticle (cloth, cords, grass), SMParticle (soft bodies) and CoinBody (rigid) all
// feed the same kernel. Speed is either (pos - prev) / dt, or — velocity mode — a velocity field read directly.

#include <metal_stdlib>
using namespace metal;

struct SettleReduceUniforms {
    uint  count;          // records to look at (after sampling)
    uint  byteStride;     // bytes between records
    uint  posOffset;      // byte offset of the float4 position (w = inverse mass; <= 0 means pinned / inactive)
    uint  prevOffset;     // byte offset of the float4 previous position — or, in velocity mode, of the float4 velocity
    uint  sampleStride;   // look at every n-th record (very large fields)
    uint  velocityMode;   // 1: prevOffset is a velocity, not a previous position
    uint  useAsleep;      // 1: skip records whose `asleep` word is non-zero
    float invDt;          // 1 / (time one step covers)
};

// atomic_uint[4]: [0] max speed as float bits (non-negative floats order like their bit patterns), [1] "a sample landed",
// [2] sum of clamped speeds (mm/s fixed point), [3] records counted
kernel void simSettleReduce(device const uchar*        records [[buffer(0)]],
                            device atomic_uint*        peak    [[buffer(1)]],
                            constant SettleReduceUniforms& u   [[buffer(2)]],
                            device const uint*         asleep  [[buffer(3)]],
                            uint gid  [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]])
{
    float s = 0.0f;
    const uint i = gid * max(u.sampleStride, 1u);
    if (i < u.count) {
        device const uchar* r = records + (ulong)i * u.byteStride;
        const float4 a = *(device const float4*)(r + u.posOffset);
        const float4 b = *(device const float4*)(r + u.prevOffset);
        const bool skip = (a.w <= 0.0f) || (u.useAsleep != 0u && asleep[i] != 0u);
        if (!skip) {
            const float3 d = (u.velocityMode != 0u) ? b.xyz : (a.xyz - b.xyz) * u.invDt;
            s = length(d);
            if (!isfinite(s)) s = INFINITY;          // a diverged particle is NOT at rest
        }
    }
    // MEAN statistic inputs: speed (clamped at 20 m/s, millimetres/second fixed point) and the number of records that counted.
    const bool counted = (i < u.count) && isfinite(s) && s >= 0.0f;
    const float sc = (i < u.count) ? ((isfinite(s) ? min(s, 20.0f) : 20.0f)) : 0.0f;
    const uint  fixedSpeed = uint(sc * 1000.0f);
    const uint  simdSum   = simd_sum(fixedSpeed);
    const uint  simdCount = simd_sum(counted && s >= 0.0f && (i < u.count) ? 1u : 0u);
    const float m = simd_max(s);
    if (lane == 0u) {
        atomic_fetch_max_explicit(&peak[0], as_type<uint>(m), memory_order_relaxed);
        atomic_fetch_add_explicit(&peak[2], simdSum, memory_order_relaxed);
        atomic_fetch_add_explicit(&peak[3], simdCount, memory_order_relaxed);
        atomic_store_explicit(&peak[1], 1u, memory_order_relaxed);
    }
}

struct SettleGateUniforms {
    float speed;          // policy: peak speed under this ...
    float seconds;        // ... for this long (simulated time) counts as rest
    uint  autoFreeze;     // 1: freeze the solver when it rests
    float elapsed;        // simulated seconds this observation covers
    uint  useMean;        // 1: judge the MEAN speed instead of the fastest (a few buzzing contacts must not hold a pile awake)
};

// uint[4] state: [0] seconds under speed (float bits), [1] frozen, [2] last peak speed (float bits), [3] observations
kernel void simSettleGate(device uint*        state [[buffer(0)]],
                          device atomic_uint* peak  [[buffer(1)]],
                          constant SettleGateUniforms& g [[buffer(2)]],
                          uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    const uint sampled = atomic_exchange_explicit(&peak[1], 0u, memory_order_relaxed);
    const uint bits    = atomic_exchange_explicit(&peak[0], 0u, memory_order_relaxed);
    if (sampled == 0u || state[1] != 0u) return;     // nothing measured, or already frozen (state is held)
    const uint  sumFixed = atomic_exchange_explicit(&peak[2], 0u, memory_order_relaxed);
    const uint  cnt      = atomic_exchange_explicit(&peak[3], 0u, memory_order_relaxed);
    const float mean     = cnt > 0u ? float(sumFixed) * 0.001f / float(cnt) : 0.0f;
    // A diverged particle (infinite peak) is never at rest, whatever the statistic.
    const float p = (g.useMean != 0u && isfinite(as_type<float>(bits))) ? mean : as_type<float>(bits);
    float below = as_type<float>(state[0]);
    below = (isfinite(p) && p < g.speed) ? below + g.elapsed : 0.0f;
    state[0] = as_type<uint>(below);
    state[2] = as_type<uint>(p);
    state[3] += 1u;
    if (g.autoFreeze != 0u && below >= g.seconds) state[1] = 1u;
}
