#include <metal_stdlib>
#include "SimSpatialHash.h"
using namespace metal;

// Counting-sort build of the shared spatial hash: clear → count → scan → scatter. After scatter `counts` holds the
// per-bucket count again (it doubles as the scatter cursor) and `sorted[offsets[b] ..< offsets[b] + counts[b]]` lists
// the particle ids in bucket b. Reads positions through (byteStride, posOffset) so any particle record feeds it.

static inline float3 simHashPosition(device const uchar* records, constant SimHashBuildUniforms& u, uint id) {
    return (*(device const float4*)(records + (ulong)id * u.byteStride + u.posOffset)).xyz;
}

kernel void simHashClear(device atomic_uint* counts [[buffer(0)]],
                         constant uint& tableSize   [[buffer(1)]],
                         uint id [[thread_position_in_grid]])
{
    if (id >= tableSize) return;
    atomic_store_explicit(&counts[id], 0u, memory_order_relaxed);
}

kernel void simHashCount(device const uchar* records [[buffer(0)]],
                         device atomic_uint* counts  [[buffer(1)]],
                         constant SimHashBuildUniforms& u [[buffer(2)]],
                         uint id [[thread_position_in_grid]])
{
    if (id >= u.count) return;
    uint h = simHashCell(simCellCoord(simHashPosition(records, u, id), u.cellSize), u.tableSize);
    atomic_fetch_add_explicit(&counts[h], 1u, memory_order_relaxed);
}

// Exclusive prefix sum of the bucket counts in ONE threadgroup (DH-1530: a single GPU thread walking all 16 384
// buckets was ~45 % of a towel's settle). Integer arithmetic, so identical to the serial scan. Re-zeros `counts` for
// the scatter cursor. Dispatch ONE threadgroup of at most 1024 threads (`sums` is sized for that).
kernel void simHashScan(device atomic_uint* counts [[buffer(0)]],
                        device uint* offsets        [[buffer(1)]],
                        constant uint& tableSize    [[buffer(2)]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tgs [[threads_per_threadgroup]])
{
    threadgroup uint sums[1024];
    uint chunk = (tableSize + tgs - 1u) / tgs;
    uint lo = min(tid * chunk, tableSize), hi = min(lo + chunk, tableSize);
    uint mine = 0u;
    for (uint i = lo; i < hi; ++i) mine += atomic_load_explicit(&counts[i], memory_order_relaxed);
    sums[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1u; off < tgs; off <<= 1) {          // Hillis-Steele inclusive scan of the chunk totals
        uint v = (tid >= off) ? sums[tid - off] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sums[tid] += v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint acc = sums[tid] - mine;                          // exclusive: everything before this chunk
    for (uint i = lo; i < hi; ++i) {
        uint c = atomic_load_explicit(&counts[i], memory_order_relaxed);
        offsets[i] = acc;
        acc += c;
        atomic_store_explicit(&counts[i], 0u, memory_order_relaxed);
    }
}

kernel void simHashScatter(device const uchar* records [[buffer(0)]],
                           device atomic_uint* cursor  [[buffer(1)]],
                           device const uint* offsets  [[buffer(2)]],
                           device uint* sorted         [[buffer(3)]],
                           constant SimHashBuildUniforms& u [[buffer(4)]],
                           uint id [[thread_position_in_grid]])
{
    if (id >= u.count) return;
    uint h = simHashCell(simCellCoord(simHashPosition(records, u, id), u.cellSize), u.tableSize);
    uint slot = atomic_fetch_add_explicit(&cursor[h], 1u, memory_order_relaxed);
    sorted[offsets[h] + slot] = id;
}
