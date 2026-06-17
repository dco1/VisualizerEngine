#include <metal_stdlib>
using namespace metal;

// ── MarchingCubes.metal ───────────────────────────────────────────────────────
//
// One compute kernel: marchingCubesExtract. Walks the (gridRes - 1) cells of a
// scalar density field, runs the classic Lorensen-Cline lookup against each
// cell's 8 corner densities, and writes up to 5 triangles' worth of vertex
// positions + normals into the cell's per-cell output slot.
//
// The crucial design choice is FIXED PER-CELL SLOT ALLOCATION:
//
//   • Output buffer is sized to (cellsX*cellsY*cellsZ * 15) vertices, where
//     `cellsAxis = gridRes.axis - 1` — one slot of 15 vertices (5 triangles ×
//     3 vertices) per cell, the worst case for MC.
//   • Each cell writes exactly 15 vertices: real triangles for the case
//     enumerated by the corner densities, then degenerate (all-zero) triangles
//     to fill the remaining slots.
//
// This avoids the alternatives (atomic vertex counter + CPU readback, or
// per-frame SCNGeometryElement rebuilds), at the cost of GPU vertex-shader
// work on degenerate triangles. The rasterizer culls those at zero-area; the
// only real cost is ~3× more vertices loaded by the vertex stage than
// strictly necessary. Cheap on Apple Silicon.
//
// ALIGNMENT: outPositions / outNormals are `packed_float3` (12-byte stride)
// so they alias directly into an SCNGeometrySource with `vertexFormat: .float3`
// and `dataStride: 12`. Same pattern as DynamicMesh.

struct MCUniforms {
    uint4  gridRes;      // xyz = density-field point resolution, w = unused
    float4 dxOrigin;     // x = cell size (metres), yzw = grid origin (world)
    float4 isoLevel;     // x = iso threshold, y = fuse radius (cells),
                         // z = cylindrical clip radius (world m, 0 = off), w = unused
};

// ── DENSITY SMOOTHING KERNEL ────────────────────────────────────────────────
//
// Single-pass 27-tap box blur over the scalar density field. Multiple
// iterations are dispatched ping-pong style by the bridge to amplify the
// effect. Smoothing the field before marching does two things:
//
//   • Smooths the per-cell density values → MC's edge-interpolation
//     positions get pulled toward "averaged" surfaces, eliminating the
//     visible faceting at cell boundaries.
//   • Smooths the gradient → MC's per-vertex normals (computed via central
//     differences in the extract kernel) read as continuous curvature
//     instead of cell-aligned plateaus.
//
// 27-tap (cube of 27 cells, weighted) gives noticeably smoother results
// than 7-tap (face neighbours only) at marginal extra cost. Centre cell
// gets a heavier weight so 1-2 iterations don't over-blur away genuine
// surface features.

kernel void mcSmoothDensity(
    device const float*  inField     [[ buffer(0) ]],
    device const float*  sourceField [[ buffer(1) ]],
    device float*        outField    [[ buffer(2) ]],
    constant MCUniforms& U           [[ buffer(3) ]],
    uint3                gid         [[ thread_position_in_grid ]])
{
    uint3 res = U.gridRes.xyz;
    if (any(gid >= res)) return;

    int3 g = int3(gid);
    int3 R = int3(res);

    float sum = 0;
    float weightSum = 0;
    // Mass-containment cap. The naïve box blur lets density diffuse into
    // cells that had no source mass, growing a halo of phantom fluid one
    // cell thicker per iteration — at 5 iters the surface visibly escapes
    // the actual fluid volume. By clamping each output cell to the maximum
    // SOURCE value in its 27-cell neighbourhood, the halo can never extend
    // more than one cell past the genuine fluid boundary, no matter how
    // many smoothing iterations we run.
    float maxSource = 0;
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        int3 nb = g + int3(dx, dy, dz);
        if (any(nb < int3(0)) || any(nb >= R)) continue;
        // Centre weight 8 vs neighbour weight 1: keeps the iso surface from
        // drifting too far across iterations while still buying significant
        // smoothing. Tuned by eye against the FluidTest at rest.
        float w = (dx == 0 && dy == 0 && dz == 0) ? 8.0f : 1.0f;
        uint  nidx = uint(nb.x) + res.x * (uint(nb.y) + res.y * uint(nb.z));
        sum += inField[nidx] * w;
        weightSum += w;
        maxSource = max(maxSource, sourceField[nidx]);
    }
    uint idx = gid.x + res.x * (gid.y + res.y * gid.z);
    float smoothed = sum / weightSum;
    outField[idx] = min(smoothed, maxSource);
}

// ── DENSITY FUSE KERNEL (wide Gaussian, UNCAPPED) ────────────────────────────
//
// `mcSmoothDensity` is a single-cell, MASS-CAPPED box blur — it deliberately
// cannot spread mass more than one cell past the genuine fluid boundary, which
// is correct for water (keeps the surface hugging the volume) but is exactly
// what makes a low-particle-count PASTE read as a "string of pearls": each
// particle clump keeps its own local density maximum, so neighbouring clumps
// bulge into separate metaballs instead of fusing into one field.
//
// This kernel is the fix for that. It runs a wider separable-style Gaussian
// (full radius R cells, sigma = R/2) with NO mass cap, so neighbouring clumps
// genuinely merge into a single smooth density field BEFORE isosurfacing. The
// caller widens R to ≈1.6–2.0× particle spacing for the guac and then re-tunes
// the isolevel up to compensate for the spread (the paste would otherwise
// balloon). Water scenes leave R at 0 and never invoke this kernel, so their
// look is unchanged.
//
// `U.isoLevel.y` carries the fuse radius in cells (read as an int). The
// neighbourhood is clamped to a hard 3-cell radius so the inner loop cost stays
// bounded (7³ = 343 taps worst case); at the guac's modest grid that's a
// single-digit-millisecond GPU pass.
kernel void mcFuseDensity(
    device const float*  inField  [[ buffer(0) ]],
    device float*        outField [[ buffer(2) ]],
    constant MCUniforms& U        [[ buffer(3) ]],
    uint3                gid      [[ thread_position_in_grid ]])
{
    uint3 res = U.gridRes.xyz;
    if (any(gid >= res)) return;

    int R = clamp(int(U.isoLevel.y + 0.5f), 1, 3);
    float sigma = max(0.5f, float(R) * 0.5f);
    float inv2s2 = 1.0f / (2.0f * sigma * sigma);

    int3 g = int3(gid);
    int3 RES = int3(res);
    float sum = 0;
    float wsum = 0;
    for (int dz = -R; dz <= R; dz++)
    for (int dy = -R; dy <= R; dy++)
    for (int dx = -R; dx <= R; dx++) {
        int3 nb = g + int3(dx, dy, dz);
        if (any(nb < int3(0)) || any(nb >= RES)) continue;
        float d2 = float(dx * dx + dy * dy + dz * dz);
        float w = exp(-d2 * inv2s2);
        uint nidx = uint(nb.x) + res.x * (uint(nb.y) + res.y * uint(nb.z));
        sum += inField[nidx] * w;
        wsum += w;
    }
    uint idx = gid.x + res.x * (gid.y + res.y * gid.z);
    // UNCAPPED — the whole point is to let mass migrate between clumps so the
    // surface fuses. A normalised Gaussian conserves total mass on average.
    outField[idx] = (wsum > 0) ? (sum / wsum) : 0;
}

// 8 cube corners in cell-local coordinates (0 or 1 along each axis).
// Order matches the Lorensen-Cline bit assignment: corner i's bit is set in
// the case index iff densityField at that corner ≥ iso.
constant int3 cornerOffsets[8] = {
    int3(0, 0, 0),  // 0
    int3(1, 0, 0),  // 1
    int3(1, 1, 0),  // 2
    int3(0, 1, 0),  // 3
    int3(0, 0, 1),  // 4
    int3(1, 0, 1),  // 5
    int3(1, 1, 1),  // 6
    int3(0, 1, 1)   // 7
};

// 12 cube edges, each a pair of corner indices. Matches the comment block in
// MarchingCubesTables.swift exactly.
constant int2 edgeCorners[12] = {
    int2(0, 1), int2(1, 2), int2(2, 3), int2(3, 0),
    int2(4, 5), int2(5, 6), int2(6, 7), int2(7, 4),
    int2(0, 4), int2(1, 5), int2(2, 6), int2(3, 7)
};

inline uint flatIdx(uint3 c, uint3 res) {
    return c.x + res.x * (c.y + res.y * c.z);
}

// Read a density-field sample with simple bounds clamping. Inside the MC
// kernel we never query OOB, but the gradient sampler needs ±1 neighbours.
inline float densityAt(int3 cell, uint3 res,
                       device const float* densityField) {
    cell = clamp(cell, int3(0), int3(res) - 1);
    return densityField[flatIdx(uint3(cell), res)];
}

// Central-difference gradient of the density field at one grid point. The
// surface normal at an isosurface vertex is the NEGATIVE gradient (mass
// density increases inward, so the outward-pointing normal is -∇ρ).
inline float3 gradAt(int3 cell, uint3 res,
                     device const float* densityField) {
    float dxp = densityAt(int3(cell.x + 1, cell.y, cell.z), res, densityField);
    float dxm = densityAt(int3(cell.x - 1, cell.y, cell.z), res, densityField);
    float dyp = densityAt(int3(cell.x, cell.y + 1, cell.z), res, densityField);
    float dym = densityAt(int3(cell.x, cell.y - 1, cell.z), res, densityField);
    float dzp = densityAt(int3(cell.x, cell.y, cell.z + 1), res, densityField);
    float dzm = densityAt(int3(cell.x, cell.y, cell.z - 1), res, densityField);
    return float3(dxp - dxm, dyp - dym, dzp - dzm);
}

kernel void marchingCubesExtract(
    device const float*    densityField [[ buffer(0) ]],
    device packed_float3*  outPositions [[ buffer(1) ]],
    device packed_float3*  outNormals   [[ buffer(2) ]],
    constant int*          triTable     [[ buffer(3) ]],
    constant MCUniforms&   U            [[ buffer(4) ]],
    uint3                  gid          [[ thread_position_in_grid ]])
{
    uint3 res = U.gridRes.xyz;
    uint3 cellRes = res - 1u;
    if (any(gid >= cellRes)) return;

    float  iso     = U.isoLevel.x;
    float  dx      = U.dxOrigin.x;
    float3 origin  = U.dxOrigin.yzw;

    // Sample the 8 corner densities and build the case index.
    //
    // BOUNDARY-CLOSING: when a corner sits at the GRID EDGE (index 0
    // or res-1 on any axis) treat it as density 0. The grid extends to
    // the simulation box boundary, but there's no fluid OUTSIDE the
    // box — so the boundary face is a real density transition that
    // should produce MC triangles. Without this, water that touches
    // the box wall (floor, side walls) generates NO surface there
    // because the kernel only sees uniformly-high density at the
    // boundary corners. The result was a "shell" mesh with only the
    // top surface, no bottom or sides — and the visible body had
    // nothing rendering between the top film and the floor.
    //
    // Treating boundary corners as zero produces a CLOSED mesh that
    // wraps the entire water column: top + sides + bottom — so an
    // opaque material on the bridge renders the water as a solid mass
    // from any angle.
    int3 R = int3(res);
    // Optional cylindrical density clip (isoLevel.z = world radius, 0 = off),
    // axis vertical through the grid's XZ centre. A vessel wall thinner than
    // the P2G + fuse + smooth smear (~4 cells total) cannot stop the density
    // FIELD from extending past it, so the isosurface pokes through solid
    // geometry (HotdogDropUltra's mustard vat rendered a phantom collar
    // around its outside base). Zeroing corner density outside the vessel's
    // interior radius closes the surface exactly at the wall — the same
    // boundary-closing treatment the grid edge gets above.
    float clipR2 = U.isoLevel.z * U.isoLevel.z;
    float clipCx = origin.x + 0.5f * dx * float(R.x - 1);
    float clipCz = origin.z + 0.5f * dx * float(R.z - 1);
    float cornerD[8];
    uint  caseIdx = 0;
    for (int i = 0; i < 8; i++) {
        int3 c = int3(gid) + cornerOffsets[i];
        bool atEdge = (c.x == 0) || (c.x == R.x - 1) ||
                      (c.y == 0) || (c.y == R.y - 1) ||
                      (c.z == 0) || (c.z == R.z - 1);
        bool clipped = false;
        if (U.isoLevel.z > 0.0f) {
            float wx = origin.x + float(c.x) * dx - clipCx;
            float wz = origin.z + float(c.z) * dx - clipCz;
            clipped = (wx * wx + wz * wz > clipR2);
        }
        cornerD[i] = (atEdge || clipped) ? 0.0f : densityField[flatIdx(uint3(c), res)];
        if (cornerD[i] >= iso) caseIdx |= (1u << uint(i));
    }

    // Where this cell's 15 vertex slots start in the output buffer.
    uint baseSlot = flatIdx(gid, cellRes) * 15u;

    // Walk the triangle list for this case — up to 5 triangles, terminated
    // by -1 in the LUT.
    uint slotsUsed = 0;
    for (int t = 0; t < 5; t++) {
        int e0 = triTable[caseIdx * 16u + t * 3 + 0];
        if (e0 < 0) break;
        int e1 = triTable[caseIdx * 16u + t * 3 + 1];
        int e2 = triTable[caseIdx * 16u + t * 3 + 2];
        // Swap e1 ↔ e2 to flip CCW → CW. The Lorensen-Cline triangle
        // table is wound CCW from the outside (standard OpenGL
        // convention), but SceneKit treats CW as front-facing for
        // custom SCNGeometry. Without this swap, our triangles are
        // back-facing in SceneKit's frame — survivable only via
        // `isDoubleSided = true`, but with wrong-side lighting and
        // the wrong sign on dot(N, V) for ordering-sensitive shader
        // modifiers. Swapping at the kernel makes the geometry
        // correct-by-construction.
        int edges[3] = { e0, e2, e1 };

        for (int j = 0; j < 3; j++) {
            int  edge    = edges[j];
            int2 cIdx    = edgeCorners[edge];
            int3 cA      = int3(gid) + cornerOffsets[cIdx.x];
            int3 cB      = int3(gid) + cornerOffsets[cIdx.y];
            float dA     = cornerD[cIdx.x];
            float dB     = cornerD[cIdx.y];

            // Linear interpolation along the edge to find the iso crossing.
            // Guard the denominator — if the two corners somehow had equal
            // density we'd divide by zero; clamp t to a midpoint default.
            float denom = dB - dA;
            float tParam = (fabs(denom) > 1e-6f) ? ((iso - dA) / denom) : 0.5f;
            tParam = clamp(tParam, 0.0f, 1.0f);

            float3 posA = origin + float3(cA) * dx;
            float3 posB = origin + float3(cB) * dx;
            float3 pos  = mix(posA, posB, tParam);

            float3 nA   = gradAt(cA, res, densityField);
            float3 nB   = gradAt(cB, res, densityField);
            float3 nRaw = mix(nA, nB, tParam);
            // Outward = negative gradient; epsilon in case the gradient is
            // exactly zero (rare; usually means we're on a flat plateau).
            float3 normal = normalize(-nRaw + float3(0, 1e-6, 0));

            outPositions[baseSlot + slotsUsed + j] = packed_float3(pos);
            outNormals  [baseSlot + slotsUsed + j] = packed_float3(normal);
        }
        slotsUsed += 3;
    }

    // Fill remaining slots with degenerate triangles. All three vertices of
    // an unused triangle live at the same point so the rasterizer culls
    // them as zero-area. Normal direction is irrelevant for culled
    // triangles but we set y=1 anyway to avoid a denormal that some GPUs
    // get fussy about.
    for (uint k = slotsUsed; k < 15u; k++) {
        outPositions[baseSlot + k] = packed_float3(0, 0, 0);
        outNormals  [baseSlot + k] = packed_float3(0, 1, 0);
    }
}
