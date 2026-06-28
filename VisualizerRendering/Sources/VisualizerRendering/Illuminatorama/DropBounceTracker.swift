import simd

// ── Loss-of-support drop+bounce driver ───────────────────────────────────────
//
// Host-side counterpart to `applySway` in Illuminatorama.metal, a sibling of
// `DragSwayTracker`. When an object that was resting ON something (a lamp on a
// table) loses that support — the table is deleted out from under it — it should
// FALL to where it now rests with a small settling bounce, instead of teleporting
// straight down.
//
// The host already recomputes each object's resting height every frame (its support
// surfaces are queried live). This tracker watches that height per object; when it
// drops suddenly it captures the gap and runs a little gravity fall with a bouncing
// settle, returning a POSITIVE vertical offset (metres) to ADD back on top of the
// new (lower) rest height. Feed that offset into the instance's `swayJostle`
// (world-space Y) — the GPU vertex shader does the actual displacement, so no mesh
// or matrix is rebuilt on the CPU per frame; only this one scalar of state advances.
//
// Why detect by height instead of taking a delete hook: it's fully decoupled. The
// host loop deletes the table → next frame the lamp's support query returns floor
// height → this tracker sees the drop and animates. Dragging a lamp off a table edge
// falls for free through the same path. A rising support (placing a lamp ONTO a
// table) is not a fall and is ignored.
public struct DropBounceTracker {
    /// Per-object animation state: how far ABOVE its final rest the object currently
    /// floats (metres, ≥ 0), and the velocity of that offset (m/s, negative = falling).
    private struct Drop {
        var offset: Float
        var velocity: Float
    }
    private var lastRestY: [Int: Float] = [:]
    private var drops:     [Int: Drop]  = [:]

    public init() {}

    // Constants tuned for the 60 fps render cadence the engine runs at (the same
    // fixed-step convention `DragSwayTracker` uses — no `dt` threaded through).
    //
    // `speed` advances the integrator's clock faster than wall-time to make the whole
    // animation snappier than real-time gravity (Danny: "40% faster"). Scaling the
    // timestep speeds up the fall AND every bounce uniformly, with the SAME bounce
    // count and settle (impact speeds are energy-determined, independent of dt).
    private static let speed:         Float = 1.4    // 1.0 = real-time gravity; 1.4 = 40% faster
    private static let dt:            Float = (1.0 / 60.0) * speed
    private static let gravity:       Float = 9.8    // m/s² — a real, un-floaty fall
    private static let restitution:   Float = 0.38   // fraction of speed kept per bounce
    private static let dropThreshold: Float = 0.02   // ignore sub-2 cm support jitter
    private static let settleSpeed:   Float = 0.18   // m/s — slower than this on contact → rest

    /// Call once per frame per object. `restY` is the object's CURRENT resolved
    /// resting height (world Y of its base) as the host computed it this frame — e.g.
    /// floor height once its support was removed. Returns the vertical offset
    /// (≥ 0, metres) to ADD on top of `restY` so the object appears to fall from where
    /// it was and bounce to rest. Returns 0 when nothing is animating; a settled object
    /// returns exactly 0, so its resolved pose equals an ordinary grounded placement.
    public mutating func update(id: Int, restY: Float) -> Float {
        defer { lastRestY[id] = restY }

        // First sighting: record only. Appearing in the scene is not a fall.
        guard let prev = lastRestY[id] else { return 0 }

        // A sudden DROP in support height is the gap the object must fall through.
        let fell = prev - restY
        if fell > Self.dropThreshold {
            var d = drops[id] ?? Drop(offset: 0, velocity: 0)
            d.offset += fell           // stack onto any in-flight fall (e.g. double delete)
            drops[id] = d
        }

        guard var d = drops[id] else { return 0 }

        // Integrate a gravity fall with a bouncing settle (semi-implicit Euler).
        d.velocity -= Self.gravity * Self.dt
        d.offset   += d.velocity * Self.dt
        if d.offset <= 0 {
            // Reached the rest plane. Settle if slow; otherwise bounce with energy loss.
            if abs(d.velocity) < Self.settleSpeed {
                drops[id] = nil
                return 0
            }
            d.offset = 0
            d.velocity = -d.velocity * Self.restitution
        }
        drops[id] = d
        return max(d.offset, 0)        // never sink below the rest plane
    }
}
