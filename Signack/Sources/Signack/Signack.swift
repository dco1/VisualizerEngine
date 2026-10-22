/// Signack — a Nanite-class geometry system for the Illuminatorama engine (Daydream DH-1461).
///
/// The public surface, by phase:
///
/// - **S0 · Budget by importance** — `SignackCard` (one indivisible unit of geometry, e.g. a leaf
///   card), `SignackLeafBudget.plan` (rank cards by visibility importance and cut a ladder of
///   NESTED budgets from that one ranking), `SignackLODSelector` (pick a ladder rung per frame from
///   projected size, with hysteresis). A host bakes every rung ONCE, keyed by its own params, and
///   switches between them by distance — never a per-frame rebake.
/// - **Mesh utilities** — `SignackVertexWeld` (turn an unindexed soup into an indexed mesh).
///
/// Everything here is pure, deterministic and seeded: the same input bakes the same output on
/// every run, so host-side spec-keyed mesh caches stay valid.
public enum Signack {
    /// The phase this module has reached. Bumped as phases land, so a host can log what it runs.
    public static let phase = "S0"
}
