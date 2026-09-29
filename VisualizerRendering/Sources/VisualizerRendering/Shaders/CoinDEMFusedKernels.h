// ── CoinDEMFusedKernels.h — the multi-dispatch substep's small kernels, fused (engine plan 3b) ──
//
// Every dispatch of the constraint substep drains the GPU before the next can start (~10 µs each on
// an M1 Max, measured), so the bookkeeping steps between the real work cost more in bubbles than in
// arithmetic. These kernels fold them together — the same step functions (the `cd…Body` / `cd…TG`
// functions the one-step kernels wrap), in the same order, so the substep is the same simulation
// (the B2 parity worlds step bit-identically; CoinDEMSmallWorldTests):
//
//   coinSubstepBegin              velocity integration + every clear the substep needs before contact
//                                 generation (cells, append cursors, per-body contact counts, colour
//                                 buckets): 5 dispatches → 1.
//   coinBroadphaseTG              the spatial hash — count, the block scan, scatter — in ONE
//                                 threadgroup (the old path already ran each of its five kernels as a
//                                 single threadgroup for ≤ 1024 bodies / blocks): 5 → 1.
//   coinColorPrepareTG            clamp the append cursor + the contact-indexed dispatch args, per-body
//                                 contact lists, the colouring seed: 3 → 1.
//   coinColorFinishTG             colour writeback, the colour buckets (count, scan, scatter), the
//                                 uncoloured bucket's sub-colours and the per-colour dispatch args: 5 → 1.
//   coinIntegratePositionFinalize position integration + finalize, both per body: 2 → 1.
//
// The integrations keep `u` a constant kernel argument (as in their one-step kernels): arithmetic on
// uniform values is rounded on its own there, and a different shape would round it differently
// (see CoinDEMSmallWorld.h). The single-threadgroup kernels loop over any count, so a large world
// (thousands of contacts) runs them too — except coinBroadphaseTG past ONE chunk of its group
// (> 1 024 scan blocks, ~1 M cells): its scan is then serial passes on one core, slower than the
// one-step kernels over the whole GPU (+90 ms/frame at 8 M cells, measured), so the host takes it
// only while the grid fits one pass (CoinDEMSolver.fusedBroadphaseFits).

kernel void coinSubstepBegin(
    device CoinBody*       coins            [[ buffer(0) ]],
    device float4*         bias             [[ buffer(1) ]],
    constant CoinUniforms& u                [[ buffer(2) ]],
    device const uint*     asleep           [[ buffer(3) ]],
    device atomic_uint*    cellCounts       [[ buffer(4) ]],
    constant uint&         numCells         [[ buffer(5) ]],
    device atomic_uint*    contactCount     [[ buffer(6) ]],
    device atomic_uint*    polyPairCount    [[ buffer(7) ]],   // (the contact cursor again when no list exists)
    device atomic_uint*    bodyContactCount [[ buffer(8) ]],
    constant uint&         bodySlots        [[ buffer(9) ]],
    device atomic_uint*    colorCount       [[ buffer(10) ]],
    uint id [[ thread_position_in_grid ]])
{
    cdIntegrateVelocityBody(id, coins, bias, u, asleep);
    if (id < numCells) atomic_store_explicit(&cellCounts[id], 0u, memory_order_relaxed);
    if (id < bodySlots) atomic_store_explicit(&bodyContactCount[id], 0u, memory_order_relaxed);
    if (id <= CD_UNCOLORED_BUCKET) atomic_store_explicit(&colorCount[id], 0u, memory_order_relaxed);
    if (id == 0u) {
        atomic_store_explicit(contactCount, 0u, memory_order_relaxed);
        atomic_store_explicit(polyPairCount, 0u, memory_order_relaxed);
    }
}

// The cell-offset scan's contract, unchanged (coinCellBlockSums / BlockScan / OffsetsApply):
// cellOffsets = the exclusive prefix sum of the counts with the grand total in
// cellOffsets[totalCells], and the counts zeroed as the scatter's cursors. Blocks of
// CD_SCAN_BLOCK cells, one per thread, `tgs` blocks per chunk; each chunk's block sums are scanned
// with SIMD-group prefix sums and a carry — integer arithmetic, so the offsets are exact.
constant uint CD_FUSED_MAX_SIMDGROUPS = 32u;     // 1024 threads / 32

kernel void coinBroadphaseTG(
    device const CoinBody* coins         [[ buffer(0) ]],
    device atomic_uint*    cellCounts    [[ buffer(1) ]],   // zeroed by coinSubstepBegin
    device uint*           cellOffsets   [[ buffer(2) ]],
    device uint*           sortedIndices [[ buffer(3) ]],
    constant CoinUniforms& u             [[ buffer(4) ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tgs  [[ threads_per_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]],
    uint sg   [[ simdgroup_index_in_threadgroup ]],
    uint nsg  [[ simdgroups_per_threadgroup ]])
{
    threadgroup uint groupTotal[CD_FUSED_MAX_SIMDGROUPS];
    threadgroup uint groupBase[CD_FUSED_MAX_SIMDGROUPS];
    threadgroup uint chunkTotal;
    threadgroup uint carry;
    device uint* counts = reinterpret_cast<device uint*>(cellCounts);

    // Count (coinCellCount).
    for (uint id = tid; id < u.coinCount; id += tgs) {
        if (coins[id].posInvMass.w == 0.0) continue;
        atomic_fetch_add_explicit(&cellCounts[cdCellIndex(coins[id].posInvMass.xyz, u)], 1u, memory_order_relaxed);
    }
    if (tid == 0u) carry = 0u;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // Scan, a chunk of `tgs` blocks at a time.
    uint total = u.gridResX * u.gridResY * u.gridResZ;
    uint blocks = (total + CD_SCAN_BLOCK - 1u) / CD_SCAN_BLOCK;
    for (uint base = 0u; base < blocks; base += tgs) {
        uint b = base + tid;
        uint start = b * CD_SCAN_BLOCK, end = min(start + CD_SCAN_BLOCK, total);
        uint s = 0u;
        if (b < blocks) for (uint i = start; i < end; ++i) s += counts[i];
        uint incl = simd_prefix_inclusive_sum(s);
        if (lane == 31u || tid == tgs - 1u) groupTotal[sg] = incl;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            uint gt = (lane < nsg) ? groupTotal[lane] : 0u;
            groupBase[lane] = simd_prefix_exclusive_sum(gt);
            if (lane == nsg - 1u) chunkTotal = groupBase[lane] + gt;       // the chunk's total (read below)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint run = carry + groupBase[sg] + (incl - s);
        if (b < blocks) {
            for (uint i = start; i < end; ++i) {
                uint c = counts[i];
                cellOffsets[i] = run;
                run += c;
                counts[i] = 0u;                          // reused as the scatter's write cursor
            }
            if (end == total) cellOffsets[total] = run;   // the tail block owns the grand total
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) carry += chunkTotal;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Scatter (coinScatter).
    for (uint id = tid; id < u.coinCount; id += tgs) {
        if (coins[id].posInvMass.w == 0.0) continue;
        uint cell = cdCellIndex(coins[id].posInvMass.xyz, u);
        uint slot = atomic_fetch_add_explicit(&cellCounts[cell], 1u, memory_order_relaxed);
        sortedIndices[cellOffsets[cell] + slot] = id;
    }
}

// Clamp the append cursor to the buffer and write the contact-indexed dispatch args
// (coinWriteContactArgs), then the per-body contact lists (coinBuildBodyContacts, counts zeroed by
// coinSubstepBegin) and the colouring seed (coinColorInit) — one threadgroup.
kernel void coinColorPrepareTG(
    device atomic_uint&       contactCount     [[ buffer(0) ]],
    device uint*              contactArgs      [[ buffer(1) ]],
    constant uint&            argsTGSize       [[ buffer(2) ]],
    constant uint&            maxContacts      [[ buffer(3) ]],
    device const CoinContact* contacts         [[ buffer(4) ]],
    device uint*              bodyContacts     [[ buffer(5) ]],
    device atomic_uint*       bodyContactCount [[ buffer(6) ]],
    device atomic_uint*       stats            [[ buffer(7) ]],
    device uint*              priority         [[ buffer(8) ]],
    device uint*              color            [[ buffer(9) ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint tgs [[ threads_per_threadgroup ]])
{
    uint n = min(atomic_load_explicit(&contactCount, memory_order_relaxed), maxContacts);
    threadgroup_barrier(mem_flags::mem_device);      // every thread has read the cursor before it is clamped
    if (tid == 0u) {
        atomic_store_explicit(&contactCount, n, memory_order_relaxed);
        contactArgs[0] = (n + argsTGSize - 1u) / argsTGSize;
        contactArgs[1] = 1u;
        contactArgs[2] = 1u;
    }
    for (uint cid = tid; cid < n; cid += tgs) {
        cdBuildBodyContactsBody(cid, n, contacts, bodyContacts, bodyContactCount, stats);
        cdColorInitBody(cid, n, contacts, priority, color);
    }
}

// Colour writeback (coinColorWriteback), the colour buckets (coinColorBucketCount / Scan / Scatter —
// counts zeroed by coinSubstepBegin), the uncoloured bucket's sub-colours and the per-colour
// dispatch args (coinWriteColorArgs) — one threadgroup.
kernel void coinColorFinishTG(
    device CoinContact*       contacts         [[ buffer(0) ]],
    device const atomic_uint& contactCount     [[ buffer(1) ]],
    device const uint*        color            [[ buffer(2) ]],   // the final colours
    device atomic_uint*       stats            [[ buffer(3) ]],
    constant uint&            dispatchedColors [[ buffer(4) ]],
    device atomic_uint*       colorCount       [[ buffer(5) ]],
    device uint*              colorOffset      [[ buffer(6) ]],
    device uint*              colorContacts    [[ buffer(7) ]],
    device const uint*        priority         [[ buffer(8) ]],
    device uint*              uncolSub         [[ buffer(9) ]],
    device uint*              args             [[ buffer(10) ]],  // 4 uints per colour (16-B stride)
    constant uint&            argsTGSize       [[ buffer(11) ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint tgs [[ threads_per_threadgroup ]],
    uint w   [[ threads_per_simdgroup ]])
{
    threadgroup uint  sCid[CD_UNCOLORED_SUB_MAX];
    threadgroup uint  sPri[CD_UNCOLORED_SUB_MAX];
    threadgroup uint4 sId[CD_UNCOLORED_SUB_MAX];
    threadgroup uint2 sAB[CD_UNCOLORED_SUB_MAX];
    threadgroup uint  sSub[CD_UNCOLORED_SUB_MAX];
    threadgroup atomic_uint sUsed[4];
    threadgroup uint  sFail;
    uint n = atomic_load_explicit(&contactCount, memory_order_relaxed);
    for (uint cid = tid; cid < n; cid += tgs) {
        cdColorWritebackBody(cid, n, contacts, color, stats, dispatchedColors);
        cdColorBucketCountBody(cid, n, color, colorCount);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid < w) cdColorBucketScanSG(tid, w, reinterpret_cast<device uint*>(colorCount), colorOffset);   // the first SIMD group
    threadgroup_barrier(mem_flags::mem_device);
    for (uint cid = tid; cid < n; cid += tgs)
        cdColorBucketScatterBody(cid, n, color, colorCount, colorOffset, colorContacts, contacts);
    threadgroup_barrier(mem_flags::mem_device);
    cdUncolouredBucketTG(tid, tgs, colorOffset, colorContacts, priority, contacts, uncolSub, stats,
                         sCid, sPri, sId, sAB, sSub, sUsed, sFail);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint c = tid; c < CD_MAX_COLORS; c += tgs) {
        uint count = colorOffset[c + 1] - colorOffset[c];
        args[c * 4 + 0] = (count + argsTGSize - 1u) / argsTGSize;
        args[c * 4 + 1] = 1u;
        args[c * 4 + 2] = 1u;
        args[c * 4 + 3] = 0u;
    }
}

kernel void coinIntegratePositionFinalize(
    device CoinBody*       coins  [[ buffer(0) ]],
    device const float4*   bias   [[ buffer(1) ]],
    constant CoinUniforms& u      [[ buffer(2) ]],
    device const uint*     asleep [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    cdIntegratePositionBody(id, coins, bias, u, asleep);
    cdFinalizeBody(id, coins, u, asleep, bias);
}
