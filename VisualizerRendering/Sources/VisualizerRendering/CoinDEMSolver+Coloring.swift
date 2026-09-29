import Foundation

// ── CoinDEMSolver + Coloring ─────────────────────────────────────────────────
//
// VZ-0160 (stage B2): a contact colouring that CONVERGES in a few rounds however busy a body
// is. The constraint solver runs its contacts colour by colour (Gauss–Seidel between colours,
// in parallel within one), so every contact needs a colour no neighbour sharing a body has.
//
// JONES–PLASSMANN (the default, unchanged). A contact takes the lowest free colour only in a
// round where it outranks every still-uncoloured neighbour, so a round colours at most ONE
// contact per body: a body of degree d needs ≥ d rounds, and in practice the rounds follow
// the longest decreasing-priority chain through the graph — ≈ 2.4× the max degree. At the
// usual 16–24 rounds, a heap whose busiest bodies touch 20–40 others leaves hundreds to
// thousands of contacts uncoloured per substep (a Daydream-Home-scale egg rain, a 7×7 grid of
// resting cubes); the tail solves up to 256 of them per substep and the rest go UNSOLVED for
// that substep (the 800-egg rain at 24 rounds: 2.15 M unsolved over 1 500 frames).
//
// SPECULATIVE (`coloringScheme = .speculative`, opt-in). Every round colours MANY contacts of
// a body at once, in two passes (CoinDEM.metal, coinColorTentative / coinColorResolve):
//   • tentative — each uncoloured contact counts how many uncoloured contacts of each of its
//     two bodies outrank it (Jones–Plassmann's own priority / identity order) and bids for the
//     k-th colour no COLOURED neighbour holds, k = the larger count: a body's contacts spread
//     over distinct colours instead of all queueing for the lowest free one;
//   • resolve — a contact keeps its bid unless an uncoloured neighbour bid the same colour and
//     outranks it; a loser bids again next round. Two neighbours never keep one colour (the
//     higher-ranked one wins), so the colouring stays proper.
// Both passes are pure functions of the previous round's colours (the bids have their own
// buffer; colours ping-pong as in Jones–Plassmann), so the colouring is a function of the
// contact graph alone — reproducible run to run. The price is ≈ 15–35 % more colours than a
// complete Jones–Plassmann colouring (each colour is one dispatch per velocity iteration);
// the win is that ~6 rounds (12 dispatches) colour a heap of degree 24–40 of which
// Jones–Plassmann leaves 45–67 % uncoloured at 16–24 rounds. Measured: CoinDEMColoringTests (28 colours
// for JP's complete 24 on a cube grid, 47 for 40 on an egg heap); the 800-egg rain in
// `docs/coindem-constraint-solver.md`.
//
// OPT-IN. The default stays `.jonesPlassmann`, bit for bit: a scene (or Daydream Home) that
// never sets the scheme colours, solves and settles exactly as before, and never allocates
// the bid buffer. A scene switches with
//     solver.coloringScheme = .speculative      // speculativeColorRounds (6) replaces colorRounds

extension CoinDEMSolver {

    /// How the constraint solver colours each substep's contacts (VZ-0160). See the header.
    public enum ColoringScheme: Sendable {
        /// Jones–Plassmann: one contact per body per round, `colorRounds` rounds. The default.
        case jonesPlassmann
        /// Speculative ranked bids + conflict resolution: many contacts per body per round,
        /// `speculativeColorRounds` rounds of two dispatches each. Opt-in.
        case speculative
    }
}
