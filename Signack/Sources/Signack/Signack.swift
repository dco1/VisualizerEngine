/// Signack — a Nanite-class geometry system for the Illuminatorama engine (Daydream DH-1461).
///
/// The public surface, by phase:
///
/// - **S0 · Budget by importance** — `SignackCard` (one indivisible unit of geometry, e.g. a leaf
///   card), `SignackLeafBudget.plan` (rank cards by visibility importance and cut a ladder of
///   NESTED budgets from that one ranking), `SignackLODSelector` (pick a ladder rung per frame from
///   projected size, with hysteresis). A host bakes every rung ONCE, keyed by its own params, and
///   switches between them by distance — never a per-frame rebake.
/// - **S1 · Cluster LOD DAG** — `SignackHierarchyBuilder.build` (cluster → group → simplify with the
///   group boundary locked → re-cluster, level by level) over ONE shared vertex buffer, behind the
///   `SignackSimplifier` seam: `SignackEdgeCollapse` (connected surfaces, no new vertices) and
///   `SignackCardSimplifier` (foliage cards, importance-ranked removal + quantised coverage scale).
///   `SignackSelection` is the CPU reference of S2's GPU pass: a per-cluster, state-free,
///   crack-free cut test (`isInCut`), a fixed object-space cut for ray-traced proxies, and frustum
///   culling; `SignackCutTracker` is the host's re-select hysteresis; `SignackDebugPalette` colours
///   clusters / levels.
/// - **S2 · GPU cut selection** (the `SignackGPU` target) — `SignackGPUHierarchy` makes a DAG
///   resident ONCE per object type; `SignackGPUCut` is one instance's GPU-written index list + two
///   indirect records (on-screen for the camera, whole cut for shadows); `SignackGPUCutPass` encodes
///   the per-cluster `isInCut` + padded frustum test + compaction for a batch of instances. The CPU
///   `SignackSelection` stays the reference it is gated against (`SignackGPUParityTests`).
/// - **S3 · Visibility buffer** (host glue in the engine: `IlluminatoramaRenderer+VisBuffer`,
///   `IlluminatoramaVisBuffer.metal`) — a mesh flagged `visBuffer` rasterises only a triangle id in
///   the G-buffer pass and is shaded once per pixel by a resolve that writes the same G-buffer
///   targets. Signack's part is `SignackGPUCut.outputClusterIDs`: the cluster of every compacted
///   triangle, so a pixel's triangle id names its cluster with one load.
/// - **S4 · Software rasterizer for sub-pixel triangles** (`SignackGPU`) — `SignackGPURouting` makes
///   a cut send each on-screen triangle projecting under `maxArea` px² to a soft record (written from
///   the top of the output); `SignackRaster` resolves window-space triangles (the host's vertex stage
///   writes them) into a per-pixel nearest depth + id with the hardware's own coverage rule (1/256
///   snap, top-left, `SignackRasterParityTests` checks it pixel for pixel against the real
///   rasterizer). Two 32-bit passes — Apple7 has no 64-bit atomics. The engine's half
///   (`IlluminatoramaRenderer+SoftRaster`) merges it through the G-buffer pass's depth test into the
///   S3 id attachment. Built, parity-proven, lever OFF: at the 1 px cut its triangles cost the
///   hardware nothing measurable (DH-1461 S4).
/// - **Mesh utilities** — `SignackVertexWeld` (turn an unindexed soup into an indexed mesh).
///
/// Everything here is pure, deterministic and seeded: the same input bakes the same output on
/// every run, so host-side spec-keyed mesh caches stay valid.
public enum Signack {
    /// The phase this module has reached. Bumped as phases land, so a host can log what it runs.
    public static let phase = "S4"
}
