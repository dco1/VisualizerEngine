#include <metal_stdlib>
using namespace metal;

// ── MLSMPM.metal ──────────────────────────────────────────────────────────────
//
// Moving-Least-Squares Material Point Method fluid kernels. Four kernels:
//
//   1.  mlsGridClear    — zero grid_mass + grid_mom atomics (one thread per voxel)
//   2.  mlsP2G          — scatter particle mass + APIC momentum + stress to grid
//   3.  mlsGridUpdate   — normalise to velocity, add gravity, AABB clamp, write GridNode
//   4.  mlsG2P          — gather grid velocity, update C / F, advect particle
//
// Struct layouts MUST match the Swift mirrors in MLSMPMSolver.swift. All shared
// structs use float4 / float3x3 / uint4 — no bare float3 (Metal's float3 is
// 16-byte aligned and disagrees with Swift's SIMD3<Float> stride).
//
// Math reference: Hu et al. 2018 "A Moving Least Squares Material Point Method
// with Displacement Discontinuity and Two-Way Rigid Body Coupling" — APIC
// formulation in their §3, fluid stress (weakly compressible Neo-Hookean) in §5.
// The reference Visualizer/Reference/fluid-code/metal/FluidSim.metal in this
// repo carries the same maths in a different harness; this file shares the
// math, not the queue / sync model.

// ── Shared structs ──────────────────────────────────────────────────────────

struct MLSParticle {
    float4   positionMass;   // xyz = position, w = mass
    float4   velocityJp;     // xyz = velocity, w = Jp
    float3x3 C;              // APIC affine velocity matrix
    float3x3 F;              // elastic deformation gradient
    float4   misc;           // x = density estimate, y = age, z/w = pad
};

struct MLSGridNode {
    float4 momentumMass;     // xyz = momentum, w = mass
    float4 velocityPad;      // xyz = velocity, w = pad
};

struct MLSUniforms {
    uint4  gridRes;          // xyz = gridX/Y/Z, w = particleCount
    float4 dxParams;         // x = dx, y = invDx, z = dt, w = time
    float4 matParams;        // x = bulkModulus, y = gamma, z = restDensity, w = viscosity
    float4 boundsMin;        // xyz = boundsMin, w = friction
    float4 boundsMax;        // xyz = boundsMax, w = settleDamping
    float4 gravity;          // xyz = gravity, w = pad
    // Elastoplastic "droopy sauce" params (ignored when materialMode == 0).
    float4 plasticA;         // x = mu0, y = lambda0, z = xi (hardening), w = thetaC
    float4 plasticB;         // x = thetaS, y = materialMode (0=fluid,1=elastoplastic), zw = pad
};

// ── Utilities ───────────────────────────────────────────────────────────────

// Quadratic B-spline weights (MLS-MPM §3.1). fx is in [0.5, 1.5).
inline float3 bspline_weights(float fx) {
    float3 w;
    w.x = 0.5f * (1.5f - fx) * (1.5f - fx);
    w.y = 0.75f - (fx - 1.0f) * (fx - 1.0f);
    w.z = 0.5f * (fx - 0.5f) * (fx - 0.5f);
    return w;
}

inline uint grid_idx(uint3 cell, uint3 res) {
    return cell.x + res.x * (cell.y + res.y * cell.z);
}

inline float3x3 outer3(float3 a, float3 b) {
    return float3x3(a * b.x, a * b.y, a * b.z);
}

// Neo-Hookean fluid Kirchhoff stress τ = (Piola) * Fᵀ. Weakly compressible
// water with Newtonian viscosity.
//
// In `.fluid` mode the g2p kernel resets F to isotropic (cbrtJ · I) each
// substep — shear deformation is intentionally forgotten so the fluid
// flows freely. That means `symF = (F+Fᵀ)/2 − I ≈ 0` always, and the
// old `2·μ·symF` "viscous" term was almost entirely cosmetic. Honey
// labeled with viscosity = 50 ran like water for exactly this reason.
//
// FIX: use the per-particle velocity gradient `C` (computed in g2p as
// the APIC affine matrix) for the viscous stress. `2·μ·sym(C)` is the
// proper Newtonian viscous stress: it damps STRAIN-RATE, scaling
// linearly with how fast the fluid is shearing. Honey now feels honey-
// thick on contact + spread, while pure free fall (where C ≈ 0 across
// the rigidly-translating blob) sees no extra drag.
inline float3x3 fluid_stress(float3x3 F,
                             float3x3 C,
                             float bulkModulus,
                             float gammaP,
                             float mu) {
    float J = determinant(F);
    // Equation of state: p = k((ρ/ρ₀)^γ − 1) → with ρ ∝ 1/J:
    float pressure = bulkModulus * (pow(1.0f / max(J, 1e-4f), gammaP) - 1.0f);
    float3x3 I = float3x3(1);
    // Newtonian viscous stress from the velocity gradient.
    float3x3 symC = 0.5f * (C + transpose(C));
    float3x3 tau_visc = 2.0f * mu * symC;
    float3x3 tau_pressure = -pressure * J * I;
    return tau_pressure + tau_visc;
}

// ── Elastoplastic "mash" model (Stomakhin 2013 snow) ─────────────────────────
//
// The avocado is a FIRM elastic solid (fixed-corotated) that PLASTICALLY YIELDS
// when over-compressed/stretched. SVD F = U Σ Vᵀ; clamp Σ into
// [1−thetaC, 1+thetaS]; the clamped-off part becomes plastic flow (the mashing),
// tracked in Jp. Hardening exp(xi(1−Jp)) softens worked material so it flows
// into paste. R = U Vᵀ drives the corotated stress.

// Analytic 3×3 SVD via cyclic one-sided Jacobi on A = FᵀF (symmetric PSD).
// Returns U,S,V with F ≈ U·diag(S)·Vᵀ, S ≥ 0, det(U)=det(V)=+1.
// (Metal float3x3 is column-major: m[col][row]; A[p][q] = col p, row q.)
inline void svd3(float3x3 F, thread float3x3 &U, thread float3 &S, thread float3x3 &V) {
    float3x3 A = transpose(F) * F;          // symmetric
    V = float3x3(1);
    const int P[3] = {0, 0, 1};
    const int Q[3] = {1, 2, 2};
    for (int sweep = 0; sweep < 6; ++sweep) {
        for (int k = 0; k < 3; ++k) {
            int p = P[k], q = Q[k];
            float apq = A[p][q];
            if (fabs(apq) < 1e-10f) continue;
            float app = A[p][p], aqq = A[q][q];
            float phi = 0.5f * atan2(2.0f * apq, aqq - app);
            float c = cos(phi), s = sin(phi);
            float3x3 J = float3x3(1);        // Jacobi rotation in the (p,q) plane
            J[p][p] = c;  J[q][q] = c;
            J[q][p] = s;  J[p][q] = -s;      // col q,row p = s ; col p,row q = -s
            A = transpose(J) * A * J;        // A ← JᵀAJ
            V = V * J;
        }
    }
    float3 sig = float3(sqrt(max(A[0][0], 0.0f)),
                        sqrt(max(A[1][1], 0.0f)),
                        sqrt(max(A[2][2], 0.0f)));
    float3 inv = float3(sig.x > 1e-8f ? 1.0f / sig.x : 0.0f,
                        sig.y > 1e-8f ? 1.0f / sig.y : 0.0f,
                        sig.z > 1e-8f ? 1.0f / sig.z : 0.0f);
    float3x3 FV = F * V;                      // U = F V Σ⁻¹
    U = float3x3(FV[0] * inv.x, FV[1] * inv.y, FV[2] * inv.z);
    S = sig;
    if (determinant(V) < 0.0f) { V[2] = -V[2]; S.z = -S.z; }
    if (determinant(U) < 0.0f) { U[2] = -U[2]; S.z = -S.z; }
}

// Fixed-corotated elastic Kirchhoff stress with snow hardening, given the
// elastic deformation gradient Fe and plastic determinant Jp.
//   mu,lambda  : Lamé params at rest
//   xi         : hardening coefficient (higher → stiffens/softens faster)
//   viscosity  : Newtonian viscous coefficient. Without this term, the
//                elastoplastic stress can only RESIST instantaneous
//                deformation (via μ, λ) and yield to PLASTIC deformation
//                (via the return map outside) — but there's no term that
//                resists the RATE of deformation. The cheese ends up
//                acting like a rigid solid that snaps to its new yielded
//                shape rather than FLOWING off the chip edge over time.
//                Adding the same 2·viscosity·sym(Fe-I) term that the
//                fluid path uses gives the elastoplastic material a
//                speed-dependent resistance: at high strain rate it
//                pushes back; at low strain rate it lets the cheese
//                droop and drip naturally.
// τ = 2μ(Fe−R)Feᵀ + λ(J−1)J I + 2·visc·sym(Fe−I), with μ,λ scaled by
//                                                  exp(xi(1−Jp)).
inline float3x3 snow_stress(float3x3 Fe, float Jp,
                            float mu0, float lambda0, float xi,
                            float viscosity) {
    float3x3 U, V; float3 S;
    svd3(Fe, U, S, V);
    float3x3 R = U * transpose(V);           // rotation (polar)
    float J = S.x * S.y * S.z;               // det(Fe)
    float h = exp(xi * (1.0f - Jp));         // hardening (Jp<1 ⇒ harden)
    float mu = mu0 * h, lambda = lambda0 * h;
    float3x3 I = float3x3(1);
    float3x3 symF = 0.5f * (Fe + transpose(Fe)) - I;
    float3x3 tau = (2.0f * mu) * (Fe - R) * transpose(Fe)
                 + (lambda * J * (J - 1.0f)) * I
                 + (2.0f * viscosity) * symF;
    return tau;
}

// Snow return-mapping: clamp the elastic singular values into
// [1−thetaC, 1+thetaS]; fold the clamped-off stretch into Jp. Returns the
// projected elastic Fe and updates Jp. Ftotal is F before projection.
inline float3x3 snow_return_map(float3x3 Ftotal, thread float &Jp,
                                float thetaC, float thetaS) {
    float3x3 U, V; float3 S;
    svd3(Ftotal, U, S, V);
    float Jtotal = S.x * S.y * S.z;
    float3 Sc = clamp(S, 1.0f - thetaC, 1.0f + thetaS);
    float Jelastic = Sc.x * Sc.y * Sc.z;
    // Plastic det carries the part we clamped off; keep total volume consistent.
    Jp = clamp(Jp * Jtotal / max(Jelastic, 1e-6f), 0.2f, 1.6f);
    // Fe = U · diag(Sc) · Vᵀ
    float3x3 Sigma = float3x3(float3(Sc.x, 0, 0), float3(0, Sc.y, 0), float3(0, 0, Sc.z));
    return U * Sigma * transpose(V);
}

// ═══════════════════════════════════════════════════════════════════════════
//  PIPELINE NOTE — Gather-mode P2G via cell-sorted particles
// ═══════════════════════════════════════════════════════════════════════════
//
// The naïve scatter P2G (each particle scatters mass + momentum into its 27
// stencil cells via atomic adds) dies on contention once particle count >
// ~50k: 108 atomic ops per particle × ~50k particles × multiple substeps
// adds up to tens of millions of contended atomics per frame, and the
// 60 fps GPU budget evaporates.
//
// The replacement is the standard MPM perf trick: SORT particles into bins
// by their centre cell, then run P2G as a GATHER — one thread per *cell*,
// summing contributions from particles in the 27-cell neighbourhood. No
// atomics in the hot path; the only atomics live in the (much cheaper)
// counting + scattering passes that build the bins.
//
// Per-substep pipeline:
//
//   1.  mlsCellClear        — zero `cellCounts` (one thread per cell)
//   2.  mlsCellCount        — atomic-inc `cellCounts[centreCell]` per particle
//   3.  mlsCellOffsetsScan  — exclusive prefix-sum cellCounts → cellOffsets;
//                             *also* re-zeros cellCounts to serve as the
//                             write cursor for the scatter pass
//   4.  mlsScatterParticles — atomic-add a slot in cellCounts, write the
//                             particle's index into sortedIndices at
//                             cellOffsets[centre] + slot
//   5.  mlsP2GGather        — one thread per grid cell. Iterates the 27
//                             neighbour bins, reads particles, contributes
//                             via the same B-spline weights as the scatter
//                             version, writes (not adds) into gridMass/Mom
//   6.  mlsGridUpdate       — unchanged from the original pipeline
//   7.  mlsG2P              — unchanged
//
// The atomic count drops from ~108 per particle to ~2 (one for counting,
// one for scatter slot allocation). Net effect at 157k particles in the
// FluidTest scene: P2G GPU time goes from ~50 ms back down to single-digit
// ms; total frame time fits comfortably in the 60 fps budget again.
//
// Cells with no particles contribute nothing and exit early. The kernel's
// per-cell cost is dominated by reading particles from neighbour bins —
// memory-bound, not atomic-bound — and parallelises cleanly across the
// 32k-cell grid.
//
// Comparison-only NOTE: the original scatter mlsP2G has been deleted; the
// reference implementation lives in Visualizer/Reference/fluid-code/metal/
// FluidSim.metal if you want to see the contrast.

// ── KERNEL 1 — Cell-count buffer clear ─────────────────────────────────────
//
// Single-purpose: zero the cellCounts buffer so the next mlsCellCount pass
// starts from a clean slate. Faster than blit-fill from the host side
// (`MTLBlitCommandEncoder.fill` on a 128 KB buffer at 60 Hz is fine in
// isolation, but stacking it with the other clears would force a CPU stall).

kernel void mlsCellClear(
    device atomic_uint* cellCounts [[ buffer(0) ]],
    uint                idx        [[ thread_position_in_grid ]])
{
    atomic_store_explicit(&cellCounts[idx], 0u, memory_order_relaxed);
}

// ── KERNEL 2 — Particle → cell binning (count phase) ──────────────────────
//
// For each particle, atomic-increment the count of its centre cell. The
// centre cell is `base + 1` where `base` is the lower-left corner of the
// 3×3×3 quadratic-B-spline stencil — i.e. the cell the particle is closest
// to. Any particle whose stencil includes cell C must have its centre in
// [C-1, C+1]; that's the contract the gather pass relies on.
//
// Clamps the centre cell to [0, gridRes - 1] so a particle that drifts
// slightly outside the grid still gets binned (it just won't contribute
// meaningfully — the gather pass filters by `rel ∈ [0, 2]`).

kernel void mlsCellCount(
    device const MLSParticle* particles  [[ buffer(0) ]],
    device atomic_uint*       cellCounts [[ buffer(1) ]],
    constant MLSUniforms&     U          [[ buffer(2) ]],
    uint                      pid        [[ thread_position_in_grid ]])
{
    uint pCount = U.gridRes.w;
    if (pid >= pCount) return;

    MLSParticle p = particles[pid];
    float3 fpos = (p.positionMass.xyz - U.boundsMin.xyz) * U.dxParams.y;
    int3   base = int3(floor(fpos - 0.5f));
    int3   centre = clamp(base + int3(1), int3(0), int3(U.gridRes.xyz) - 1);
    uint   idx = uint(centre.x) + U.gridRes.x * (uint(centre.y) + U.gridRes.y * uint(centre.z));
    atomic_fetch_add_explicit(&cellCounts[idx], 1u, memory_order_relaxed);
}

// ── KERNEL 3 — Exclusive prefix-sum of cellCounts → cellOffsets ────────────
//
// Single-thread sequential scan. Wasteful in parallelism terms but trivially
// correct, and for ~32k cells (32 µs at ~1 ns/iter on Apple Silicon) it's
// well under the noise floor of the rest of the pipeline.
//
// As a side effect, RE-ZEROES cellCounts so the scatter pass can reuse it
// as a per-cell write cursor. Saves one extra kernel + one buffer.
//
// For larger grids (≫ 256k cells) a proper multi-pass parallel scan would
// be worth it, but the cost wouldn't appear before then.

kernel void mlsCellOffsetsScan(
    device uint*          cellCounts  [[ buffer(0) ]],
    device uint*          cellOffsets [[ buffer(1) ]],
    constant MLSUniforms& U           [[ buffer(2) ]],
    uint                  gid         [[ thread_position_in_grid ]])
{
    if (gid != 0) return;
    uint total = U.gridRes.x * U.gridRes.y * U.gridRes.z;
    uint sum = 0;
    for (uint i = 0; i < total; i++) {
        cellOffsets[i] = sum;
        sum += cellCounts[i];
        cellCounts[i] = 0;     // reused as write cursor by mlsScatterParticles
    }
    cellOffsets[total] = sum;
}

// ── KERNEL 4 — Scatter particle indices into per-cell bins ─────────────────
//
// For each particle: claim a slot via atomic-inc of its bin's cursor, write
// the particle's flat index into `sortedIndices[cellOffsets[centre] + slot]`.
// After this kernel completes, `sortedIndices[cellOffsets[c]..cellOffsets[c+1]]`
// holds the indices of every particle whose centre cell is `c`.

kernel void mlsScatterParticles(
    device const MLSParticle* particles      [[ buffer(0) ]],
    device atomic_uint*       cellCursors    [[ buffer(1) ]],   // == cellCounts, zeroed by scan
    device const uint*        cellOffsets    [[ buffer(2) ]],
    device uint*              sortedIndices  [[ buffer(3) ]],
    constant MLSUniforms&     U              [[ buffer(4) ]],
    uint                      pid            [[ thread_position_in_grid ]])
{
    uint pCount = U.gridRes.w;
    if (pid >= pCount) return;

    MLSParticle p = particles[pid];
    float3 fpos = (p.positionMass.xyz - U.boundsMin.xyz) * U.dxParams.y;
    int3   base = int3(floor(fpos - 0.5f));
    int3   centre = clamp(base + int3(1), int3(0), int3(U.gridRes.xyz) - 1);
    uint   binIdx = uint(centre.x) + U.gridRes.x * (uint(centre.y) + U.gridRes.y * uint(centre.z));
    uint   slot = atomic_fetch_add_explicit(&cellCursors[binIdx], 1u, memory_order_relaxed);
    sortedIndices[cellOffsets[binIdx] + slot] = pid;
}

// ── KERNEL 5 — Gather-mode P2G ─────────────────────────────────────────────
//
// One thread per grid cell. For every particle whose centre cell is in the
// 27-cell neighbourhood, compute the B-spline weight from particle to this
// cell and accumulate mass + APIC-corrected momentum. Writes (NOT adds) into
// gridMass + gridMom, so the previous-substep values are overwritten without
// needing a separate clear pass.

kernel void mlsP2GGather(
    device const MLSParticle* particles      [[ buffer(0) ]],
    device const uint*        sortedIndices  [[ buffer(1) ]],
    device const uint*        cellOffsets    [[ buffer(2) ]],
    device float*             gridMass       [[ buffer(3) ]],
    device float*             gridMom        [[ buffer(4) ]],
    constant MLSUniforms&     U              [[ buffer(5) ]],
    uint3                     cellId         [[ thread_position_in_grid ]])
{
    uint3 res = U.gridRes.xyz;
    if (any(cellId >= res)) return;

    float dx     = U.dxParams.x;
    float invDx  = U.dxParams.y;
    float dt     = U.dxParams.z;
    float bulkK  = U.matParams.x;
    float gammaP = U.matParams.y;
    float rho0   = U.matParams.z;
    float mu     = U.matParams.w;

    // Cell's world-space position (matching the convention in P2G/G2P:
    // grid origin lives at boundsMin, cells are indexed from 0).
    float3 cellPosWorld = float3(cellId) * dx + U.boundsMin.xyz;

    float  massSum = 0;
    float3 momSum  = float3(0);

    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dxn = -1; dxn <= 1; dxn++) {
        int3 nb = int3(cellId) + int3(dxn, dy, dz);
        if (any(nb < int3(0)) || any(nb >= int3(res))) continue;

        uint binIdx = uint(nb.x) + res.x * (uint(nb.y) + res.y * uint(nb.z));
        uint start  = cellOffsets[binIdx];
        uint end    = cellOffsets[binIdx + 1];

        for (uint i = start; i < end; i++) {
            uint pid = sortedIndices[i];
            MLSParticle p = particles[pid];

            // Re-derive the particle's stencil base from its world position.
            // Reading the particle once and reusing all its fields beats
            // pre-computing weights in a separate pass (which would need
            // extra storage per particle).
            float3 ppos = p.positionMass.xyz;
            float  pmass = p.positionMass.w;
            float3 pvel  = p.velocityJp.xyz;

            float3 fpos = (ppos - U.boundsMin.xyz) * invDx;
            int3   base = int3(floor(fpos - 0.5f));
            int3   rel  = int3(cellId) - base;
            // Filter particles whose stencil doesn't actually cover this cell.
            // Cheap early-out — the centre-cell binning means most candidates
            // pass; this catches the off-by-one cases at bin boundaries.
            if (any(rel < int3(0)) || any(rel > int3(2))) continue;

            float3 fx = fpos - float3(base);
            float3 wxV = bspline_weights(fx.x);
            float3 wyV = bspline_weights(fx.y);
            float3 wzV = bspline_weights(fx.z);
            float  w = wxV[rel.x] * wyV[rel.y] * wzV[rel.z];

            // Same APIC stress + affine assembly as scatter P2G. Done inside
            // the particle loop because each particle contributes to multiple
            // cells; we have no per-particle cache to amortise it across the
            // 27 cells without an extra pass.
            //   • FLUID (materialMode 0): weakly-compressible Neo-Hookean.
            //   • ELASTOPLASTIC (materialMode 1, "droopy sauce"): fixed-corotated
            //     snow stress (SVD-based). The SVD runs per (particle, cell) here;
            //     if profiling demands it, hoist to a per-particle precompute pass.
            float3x3 stress;
            if (U.plasticB.y > 0.5f) {
                stress = snow_stress(p.F, p.velocityJp.w,        // Fe, Jp
                                     U.plasticA.x, U.plasticA.y, // mu0, lambda0
                                     U.plasticA.z,               // xi
                                     mu);                        // viscosity
            } else {
                stress = fluid_stress(p.F, p.C, bulkK, gammaP, mu);
            }
            float D_inv = 4.0f * invDx * invDx;
            float3x3 affine = pmass * p.C - (dt * pmass / max(rho0, 1e-4f)) * stress * D_inv;

            float3 offset = cellPosWorld - ppos;
            float3 mv = pmass * (pvel + affine * offset);

            massSum += w * pmass;
            momSum  += w * mv;
        }
    }

    uint idx = grid_idx(cellId, res);
    gridMass[idx]      = massSum;
    gridMom[idx*3 + 0] = momSum.x;
    gridMom[idx*3 + 1] = momSum.y;
    gridMom[idx*3 + 2] = momSum.z;
}

// ── KERNEL 3 — Grid Update ─────────────────────────────────────────────────

kernel void mlsGridUpdate(
    device const float*     gridMass [[ buffer(0) ]],
    device const float*     gridMom  [[ buffer(1) ]],
    device MLSGridNode*     gridOut  [[ buffer(2) ]],
    constant MLSUniforms&   U        [[ buffer(3) ]],
    uint3                   gid      [[ thread_position_in_grid ]])
{
    uint3 res = U.gridRes.xyz;
    if (any(gid >= res)) return;

    uint idx  = grid_idx(gid, res);
    float m   = gridMass[idx];

    MLSGridNode node;
    node.momentumMass = float4(0, 0, 0, m);
    node.velocityPad  = float4(0, 0, 0, 0);

    if (m < 1e-10f) {
        gridOut[idx] = node;
        return;
    }

    float3 mom = float3(gridMom[idx*3+0], gridMom[idx*3+1], gridMom[idx*3+2]);
    float3 vel = mom / m;

    // Gravity.
    vel += U.dxParams.z * U.gravity.xyz;

    // AABB-clamp velocity at the simulation bounds. Use cell indices directly
    // (more robust than comparing world positions to bounds — no FP-equality
    // edge cases at the cell-0 boundary). Two cells of wall on each side
    // because the quadratic B-spline stencil reaches one cell past the
    // particle's centre cell, so a particle sitting in cell 1 still needs
    // cell 0 to behave as a wall.
    const uint wall = 2u;
    if (gid.x <  wall            && vel.x < 0) vel.x = 0;
    if (gid.x >= res.x - wall    && vel.x > 0) vel.x = 0;
    if (gid.y <  wall            && vel.y < 0) vel.y = 0;
    if (gid.y >= res.y - wall    && vel.y > 0) vel.y = 0;
    if (gid.z <  wall            && vel.z < 0) vel.z = 0;
    if (gid.z >= res.z - wall    && vel.z > 0) vel.z = 0;

    // Coulomb friction on the floor.
    float dx = U.dxParams.x;
    if (gid.y < wall + 1u && vel.y <= 0) {
        float frictionCoef = U.boundsMin.w;
        float2 tangent = vel.xz;
        float tangLen = length(tangent);
        if (tangLen > 1e-6f) {
            float frictionMag = min(frictionCoef * fabs(vel.y), tangLen);
            vel.xz -= frictionMag * normalize(tangent);
        }
    }

    // Soft settle damping. Once the bulk fluid is moving slowly the
    // weakly-compressible EOS keeps producing micro-jitter (visible as
    // caustic flicker / surface shimmer that never goes still). Apply
    // a velocity-magnitude-dependent damping: above ~0.5 m/s no damping
    // (real splashes preserve momentum), below 0.5 m/s damp toward
    // zero exponentially so settled fluid actually stops. Coefficient
    // U.boundsMax.w is per-frame (clamped to [0,1] host-side); the
    // smoothstep falls off the damping for fast cells.
    float damp = U.boundsMax.w;
    if (damp > 0.0f) {
        float vmag = length(vel);
        float keep = smoothstep(0.05f, 0.5f, vmag);   // 1 = fast, 0 = slow
        float dampFactor = mix(damp, 1.0f, keep);
        vel *= dampFactor;
    }

    node.momentumMass = float4(mom, m);
    node.velocityPad  = float4(vel, 0);
    gridOut[idx] = node;
}

// ── KERNEL 4 — G2P ─────────────────────────────────────────────────────────

kernel void mlsG2P(
    device MLSParticle*       particles [[ buffer(0) ]],
    device const MLSGridNode* grid      [[ buffer(1) ]],
    constant MLSUniforms&     U         [[ buffer(2) ]],
    uint                      pid       [[ thread_position_in_grid ]])
{
    uint pCount = U.gridRes.w;
    if (pid >= pCount) return;

    MLSParticle p = particles[pid];
    float3 pos = p.positionMass.xyz;
    float dx    = U.dxParams.x;
    float invDx = U.dxParams.y;
    float dt    = U.dxParams.z;

    // Same grid-local mapping as P2G — see the comment there.
    float3 gridOrigin = U.boundsMin.xyz;
    float3 fpos = (pos - gridOrigin) * invDx;
    int3   base = int3(floor(fpos - 0.5f));
    float3 fx   = fpos - float3(base);

    float3 wx = bspline_weights(fx.x);
    float3 wy = bspline_weights(fx.y);
    float3 wz = bspline_weights(fx.z);

    uint3 res = U.gridRes.xyz;

    float3   vel_new = float3(0);
    float3x3 C_new   = float3x3(0);
    float    density = 0;

    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int k = 0; k < 3; k++) {
        int3 cell = base + int3(i, j, k);
        if (any(cell < int3(0)) || any(cell >= int3(res))) continue;
        float w = wx[i] * wy[j] * wz[k];
        uint idx = grid_idx(uint3(cell), res);
        float3 gv = grid[idx].velocityPad.xyz;
        float3 off = (float3(cell) - fpos) * dx;
        vel_new += w * gv;
        C_new   += 4.0f * invDx * invDx * w * outer3(gv, off);
        density += w * grid[idx].momentumMass.w;
    }

    // Update deformation gradient: F_new = (I + dt·C) F_old  (BOTH materials).
    float3x3 I = float3x3(1);
    float3x3 F_new = (I + dt * C_new) * p.F;

    if (U.plasticB.y > 0.5f) {
        // ELASTOPLASTIC ("droopy sauce"): KEEP the full deformation gradient —
        // that anisotropy is what lets stretched regions read as strands. Run
        // the snow return-mapping: clamp the elastic singular values into
        // [1−θc, 1+θs]; the clamped-off stretch folds into Jp (the mashing/
        // yielding). NO isotropic fluid reset.
        float Jp = p.velocityJp.w;
        p.F = snow_return_map(F_new, Jp, U.plasticA.w, U.plasticB.x); // thetaC, thetaS
        p.velocityJp.w = Jp;
    } else {
        // FLUID: tight J clamp + liquid projection — reset F to an isotropic
        // matrix encoding only J. This is what makes a fluid actually flow
        // (otherwise the solver would remember its original orientation).
        float J = clamp(determinant(F_new), 0.5f, 2.0f);
        float cbrtJ = pow(J, 1.0f / 3.0f);
        p.F  = cbrtJ * I;
        p.velocityJp.w = J;
    }

    // Safety cap on per-particle speed. Numerically a healthy fluid at our
    // scale tops out around 10–15 m/s under gravity from this height; if
    // we ever see speeds >> that it's a sign of a tuning bug, and letting
    // it propagate just produces visible explosions + huge GPU cost as the
    // EOS does pow() on absurd values. 50 m/s is comfortably above any
    // physically-plausible value and short-circuits the runaway.
    float speed = length(vel_new);
    const float maxSpeed = 50.0f;
    if (speed > maxSpeed) vel_new *= maxSpeed / speed;

    p.C            = C_new;
    p.velocityJp.xyz = vel_new;
    p.positionMass.xyz = pos + dt * vel_new;
    p.misc.x       = density;
    p.misc.y       = p.misc.y + dt;

    // Hard-clamp position to the bounds (safety net — the grid velocity
    // boundary should already prevent escape, but FP slop at the bounds can
    // produce a one-frame leak otherwise).
    p.positionMass.xyz = clamp(p.positionMass.xyz,
                               U.boundsMin.xyz + 1e-3f,
                               U.boundsMax.xyz - 1e-3f);

    particles[pid] = p;
}

// ── KERNEL 5 — Capsule collider push-out ───────────────────────────────────
//
// One-way coupling: hot-dog spine capsules push water particles out of their
// volume. Reads the same PBDCollider buffer the PBD solvers populate each
// substep — no extra collider-build work; the hot dogs are already publishing
// their capsules for inter-tube SDF collision.
//
// For each fluid particle, walk every collider. If the particle is inside the
// capsule (distance to the line segment < radius), push it to the surface
// along the outward normal and zero the inward component of velocity. We do
// NOT bounce — water doesn't visibly bounce off a sausage; it sloshes around
// it. The energy that would have produced a bounce becomes the tangential
// flow that the foam spawn picks up as a splash.
//
// PBDCollider mirror — MUST match the layout in PBD.metal / PBDSolver.swift.
// We can't include the PBD header from here, so the struct is duplicated
// verbatim. If you change one, change all three.

struct PBDColliderMLS {
    float4 a;       // xyz = endpoint A, w = type tag (bit-cast uint: 1 = capsule)
    float4 b;       // xyz = endpoint B, w = radius
    uint4  meta;    // x = ownerID (unused here; only the PBD self-skip needs it)
};

struct MLSColliderUniforms {
    uint  particleCount;
    uint  colliderCount;
    float skinDepth;        // small padding (m) so particles don't sit *exactly* on the surface
    float restitution;      // bounce coefficient on the inward velocity component (0 = no bounce, 1 = full reflect)
    float boost;            // outward velocity kick per metre of penetration (default 2.5; 0 = inert static collider)
    float tangentFriction;  // 0..1 — fraction of tangential velocity REMOVED at contact each substep.
                            // 0 = frictionless slide (water default). 1 = particles stick to the
                            // collider surface and stop sliding entirely (honey clings to chip).
                            // Implements the "adhesion" force that bulk-fluid MPM doesn't model.
    float _pad1;
    float _pad2;
};

kernel void mlsCollideCapsules(
    device MLSParticle*                particles  [[ buffer(0) ]],
    device const PBDColliderMLS*       colliders  [[ buffer(1) ]],
    constant MLSColliderUniforms&      U          [[ buffer(2) ]],
    uint                               pid        [[ thread_position_in_grid ]])
{
    if (pid >= U.particleCount) return;
    if (U.colliderCount == 0u) return;

    MLSParticle p = particles[pid];
    float3 pos = p.positionMass.xyz;
    float3 vel = p.velocityJp.xyz;

    bool   touched = false;

    for (uint c = 0; c < U.colliderCount; c++) {
        PBDColliderMLS col = colliders[c];
        // Type tag at a.w; we only consume capsules here.
        uint kind = as_type<uint>(col.a.w);
        if (kind != 1u) continue;

        float3 a = col.a.xyz;
        float3 b = col.b.xyz;
        float  r = col.b.w + U.skinDepth;

        // Closest point on segment AB to pos.
        float3 ab = b - a;
        float  ab2 = max(dot(ab, ab), 1e-8f);
        float  t = clamp(dot(pos - a, ab) / ab2, 0.0f, 1.0f);
        float3 q = a + ab * t;
        float3 d = pos - q;
        float  dist2 = dot(d, d);
        float  r2 = r * r;
        if (dist2 >= r2) continue;

        // Inside the capsule. Push out along the outward normal — but
        // with the downward component clamped to zero.
        //
        // Why the clamp: the naïve "push to nearest surface" rule
        // biases every below-axis particle DOWN (its normal points
        // away from the axis, i.e., further down). For a half-
        // submerged horizontal capsule this drives the entire bulk
        // fluid beneath it into the box floor, compressing it. Mass
        // is conserved per-particle, but the weakly-compressible EOS
        // doesn't push the surface back up fast enough — the pool
        // sinks ~30 cm over a few seconds. Confirmed in the diag log:
        // pMax.y drifted from -0.02 down to -0.39 over a 50 s run.
        //
        // Clamping n.y >= 0 mimics real displacement physics: a solid
        // object pushes fluid OUT and UP, not into the floor. Volume
        // gets pushed laterally + upward around the object instead of
        // crushed downward under it.
        float dist = sqrt(max(dist2, 1e-12f));
        float3 n;
        if (dist > 1e-5f) {
            n = d / dist;
        } else {
            // Degenerate: particle is on the capsule's axis. Pick any
            // perpendicular to AB so we don't end up with a zero normal.
            float3 axis = normalize(ab);
            float3 fallback = (abs(axis.y) < 0.9f) ? float3(0, 1, 0) : float3(1, 0, 0);
            n = normalize(fallback - axis * dot(fallback, axis));
        }
        // No-downward-push clamp.
        if (n.y < 0.0f) {
            n.y = 0.0f;
            float lat = length(n.xz);
            if (lat > 1e-5f) {
                n = n / length(n);   // renormalize after zeroing y
            } else {
                // Particle was directly below the axis (purely vertical
                // n). Default to straight up so it escapes the capsule's
                // lower hemisphere by going over the top of the fluid.
                n = float3(0, 1, 0);
            }
        }
        // Penetration depth before the push (how far INSIDE the surface
        // the particle was). Drives the velocity boost below — deeper
        // penetration = stronger outward kick, so a hot dog sweeping
        // through the fluid leaves a stronger wake than one resting on it.
        float penetration = max(0.0f, r - dist);

        pos = q + n * r;

        // Two-part velocity update:
        //
        //  1. Reflect the inward velocity component with restitution > 0
        //     so an impacting hot dog actually launches fluid outward.
        //     Restitution 0 just kills the inward component; restitution
        //     1 fully reflects it. Tangential flow is preserved.
        //
        //  2. Add an outward velocity boost proportional to penetration
        //     depth. WITHOUT this, a settled hot dog (zero relative
        //     velocity) produces only spatial displacement — fluid just
        //     deforms around the capsule and reaches a new static
        //     equilibrium with no visible motion. The boost imparts a
        //     small "escape" velocity to displaced particles, which
        //     keeps the surface continuously sloshing around resting
        //     hot dogs and gives the foam solver fast-moving particles
        //     to spawn spray from.
        float vN = dot(vel, n);
        if (vN < 0.0f) {
            vel -= (1.0f + U.restitution) * vN * n;
        }
        // Empirical scale, host-driven via U.boost. Default 2.5 for fluid
        // (wake / continuous slosh around static immersed objects); 0 for
        // elastoplastic on a static ledge so the sauce can actually rest
        // on it instead of being kicked outward indefinitely.
        vel += n * (penetration * U.boost);

        // Tangent friction — the "adhesion" force that bulk-fluid MPM
        // doesn't model. Bulk viscosity damps with velocity GRADIENT,
        // so a small remaining blob has little gradient and accelerates
        // freely off the chip — wrong for honey, which clings to the
        // surface harder per-volume the smaller the blob gets (real
        // surface tension / adhesion). Removing a fraction of the
        // tangential velocity at every contact substep simulates that
        // surface stickiness: last particles to leave the chip get
        // dragged just like the bulk did.
        if (U.tangentFriction > 0.0f) {
            float3 vTangent = vel - dot(vel, n) * n;
            vel -= vTangent * U.tangentFriction;
        }
        touched = true;
    }

    if (touched) {
        p.positionMass.xyz = pos;
        p.velocityJp.xyz   = vel;
        particles[pid] = p;
    }
}

// ── KINEMATIC GRIP (bonded pull) ─────────────────────────────────────────────
//
// Two crust halves of a fried mozzarella stick grip the cheese at each torn
// face and drag it apart. A band of cheese particles near each face is PINNED
// to its half: it moves rigidly with the half (a moving boundary), while the
// unpinned middle is left to the constitutive model — so it necks, thins, and
// snaps under the growing separation. This is the bonded-grip the repel-only
// capsule collider can't express.
//
// `misc.z` tags the pin owner (0 = free, 1 = cap A / −X half, 2 = cap B / +X
// half), set at seed time. `gripRest[pid].xyz` is the particle's CAPTURED rest
// position (also at seed time). Each substep a pinned particle is snapped to
// rest + the host-supplied rigid displacement of its cap and given the cap's
// velocity, so the NEXT P2G drags its free neighbours along. Translation-only
// today (caps separate along the stick axis); a future angled/rotating pull
// would pass a full transform instead of a displacement.
struct MLSGripUniforms {
    uint   particleCount;
    uint   enabled;       // 0 = no-op (kernel returns immediately)
    float  _pad0;
    float  _pad1;
    float4 dispA;         // xyz = cap A rigid displacement from rest, w = pad
    float4 dispB;         // xyz = cap B rigid displacement from rest, w = pad
    float4 velA;          // xyz = cap A velocity, w = pad
    float4 velB;          // xyz = cap B velocity, w = pad
};

kernel void mlsGripPin(
    device MLSParticle*            particles  [[ buffer(0) ]],
    device const float4*           gripRest   [[ buffer(1) ]],
    constant MLSGripUniforms&      U          [[ buffer(2) ]],
    uint                           pid        [[ thread_position_in_grid ]])
{
    if (U.enabled == 0u) return;
    if (pid >= U.particleCount) return;

    float flag = particles[pid].misc.z;
    if (flag < 0.5f) return;                  // free particle — untouched

    float3 disp = (flag < 1.5f) ? U.dispA.xyz : U.dispB.xyz;
    float3 vel  = (flag < 1.5f) ? U.velA.xyz  : U.velB.xyz;

    float3 rest = gripRest[pid].xyz;
    particles[pid].positionMass.xyz = rest + disp;
    particles[pid].velocityJp.xyz   = vel;
    // Rigid: kill the affine velocity matrix so the pinned particle injects
    // no spurious strain into the grid, and reset F so it never accumulates
    // plastic deformation (it's a boundary, not deforming cheese).
    particles[pid].C = float3x3(0.0f);
    particles[pid].F = float3x3(1.0f);
}

// ── Conveyor recycle (opt-in endless column) ─────────────────────────────────
//
// A Y-WRAP treadmill for an upward-flowing column (the Hot Dog Press tube): any
// particle that has risen past `lipY` (cleared the exit) is wrapped straight down
// by `wrapHeight` (= lipY − floorY), preserving its X/Z, velocity, and deformation
// state. Wrapping the WHOLE particle (not a scatter re-seed) keeps seeded clusters
// — the distinct frankfurters — intact as they loop, the way Infinite Soft Serve's
// scroll preserves its flavour bands. Particle COUNT is constant (no growth /
// compaction). The seam is at the bottom (off-frame) and, because the column is
// periodic over `wrapHeight`, a wrapped particle lands among matching neighbours,
// so the flow stays continuous. No-op unless the host enables it.
struct MLSRecycleUniforms {
    uint   particleCount;
    uint   frame;          // reserved (animation variation)
    float  lipY;           // wrap threshold (world Y)
    float  wrapHeight;     // distance wrapped down (= lipY − floorY)
    float  driveSpeed;     // target upward conveyor speed (m/s); 0 = drive off
    float  driveRelax;     // 0..1 per-frame relaxation toward driveSpeed
    float  _pad0;
    float  _pad1;
};

kernel void mlsConveyorRecycle(
    device MLSParticle*            particles  [[ buffer(0) ]],
    constant MLSRecycleUniforms&   U          [[ buffer(1) ]],
    uint                           pid        [[ thread_position_in_grid ]])
{
    if (pid >= U.particleCount) return;

    // CONVEYOR DRIVE: relax vertical velocity toward a uniform target speed so the
    // whole packed column rides up TOGETHER (no body-force acceleration → no
    // stretching/gaps). Where franks jam against each other the incompressible
    // pressure overrides this and they squish — the deformation cue. Applied to
    // every particle; horizontal velocity is left to the solver (settling/jostle).
    if (U.driveSpeed != 0.0f) {
        float vy = particles[pid].velocityJp.y;
        particles[pid].velocityJp.y = vy + (U.driveSpeed - vy) * U.driveRelax;
    }

    // Y-WRAP: a particle that cleared the lip wraps down one column height,
    // preserving shape/strain so the frank cluster survives the loop seamlessly.
    if (particles[pid].positionMass.y > U.lipY) {
        particles[pid].positionMass.y -= U.wrapHeight;
    }
}
