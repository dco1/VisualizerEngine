import simd

// ── Drag / impact secondary-motion driver ────────────────────────────────────
//
// The host-side counterpart to `applySway` in Illuminatorama.metal. The shader
// applies a rigid lean+hop to a swaying instance; THIS decides how much, per
// object, per frame — so any Illuminatorama consumer (Visualizer, Daydream Home,
// …) gets believable secondary motion on dragged/knocked objects for free.
//
// Any placed object that the host drags can participate by calling
// `DragSwayTracker.update(id:pos:angle:)` once per frame. The tracker owns all
// physics state (previous-frame positions, eased sway, recoil springs) and returns
// a `DragSway` snapshot. The *caller* decides what to do with it — feed it into an
// instance's `swayLean`/`swayJostle` for the GPU bend, or interpret it directly:
//
//   • Bookshelf   → books lean (leanX) and jostle (jostle)
//   • Lamp shade  → shade swings as a pendulum (leanX + leanZ)
//   • Pendant     → cord sways, shade tilts (leanX + leanZ)
//   • Chandelier  → arms oscillate (jostle + leanX magnitude)
//
// Collision recoil rides the same channel: when a dragged piece knocks into
// something the host calls `registerImpact(…)` with the momentum lost, and the
// returned leanX/jostle gain a one-shot spring lurch that overshoots and settles.
// Any object already reading `DragSway` recoils for free.

/// Per-object eased animation snapshot returned by `DragSwayTracker`.
///
/// Fields:
///   - `leanX`  — local-X axis rotation (radians). Positive = top swings toward +X.
///   - `leanZ`  — local-Z axis rotation (radians). Reserved (e.g. shade front/back swing).
///   - `jostle` — upward Y displacement (metres).
public struct DragSway {
    public var leanX:  Float
    public var leanZ:  Float
    public var jostle: Float
    public init(leanX: Float = 0, leanZ: Float = 0, jostle: Float = 0) {
        self.leanX = leanX; self.leanZ = leanZ; self.jostle = jostle
    }
}

/// Shared drag-animation physics engine for every placed object in a scene.
///
/// Call `update(id:pos:angle:)` once per frame per object; the tracker diffs
/// positions to get velocity, projects it into the object's local axes, and eases
/// the sway state toward its target. On top of the continuous, velocity-driven lean
/// it carries a one-shot **collision recoil**: when the piece knocks into something
/// the caller hands the tracker the momentum it just lost via `registerImpact(…)`,
/// and a lightly-damped spring lurches the object the way it was heading, rocks back,
/// and settles. The recoil folds into the same `leanX`/`jostle` the eased lean uses,
/// so *every* consumer of `DragSway` gets the knock reaction for free.
///
/// Positions are plan-space (metres): `pos.x` = world X, `pos.y` = world depth (which
/// the renderer maps to −Z). The constants are tuned for the 60 fps frame cadence.
public struct DragSwayTracker {
    private var states:  [Int: DragSway]     = [:]
    private var prevPos: [Int: SIMD2<Float>] = [:]
    private var impacts: [Int: ImpactSpring] = [:]

    public init() {}

    /// A lightly-damped harmonic oscillator, integrated per-frame at the 60 fps
    /// cadence the eased constants below assume. Kicked by `registerImpact`, it swings
    /// past zero a few times and decays to rest — the "knock" recoil, kept separate
    /// from the steady eased lean so a collision reads as a distinct jolt on top of it.
    private struct ImpactSpring {
        var leanPos: Float = 0, leanVel: Float = 0   // local-X recoil (radians)
        var josPos:  Float = 0, josVel:  Float = 0   // vertical pop (metres)
        // k → ~0.33 s period; c → ζ ≈ 0.35, i.e. 2–3 visible swings before rest.
        static let k: Float = 0.10, c: Float = 0.22
        mutating func step() {
            leanVel += -Self.k * leanPos - Self.c * leanVel; leanPos += leanVel
            josVel  += -Self.k * josPos  - Self.c * josVel;  josPos  += josVel
        }
        var atRest: Bool {
            abs(leanPos) < 1e-4 && abs(leanVel) < 1e-4 &&
            abs(josPos)  < 1e-4 && abs(josVel)  < 1e-4
        }
    }

    /// Call once per frame per animated object. Returns the current eased sway plus
    /// any active collision recoil. `pos` is the object's plan-space position (metres);
    /// `angle` is its Y-rotation (radians) — pass 0 for radially symmetric objects
    /// (lamps); the tracker still produces a world-space leanX/leanZ.
    public mutating func update(id: Int, pos: SIMD2<Float>, angle: Float) -> DragSway {
        let cur   = pos
        let delta = (prevPos[id].map { cur - $0 }) ?? .zero
        prevPos[id] = cur

        // Project plan-space delta onto the object's local axes.
        // model col0 = (cos_a, 0, -sin_a) → local X in world.
        // local_vX = Δx·cos_a + Δy·sin_a   (plan Y maps to world -Z, so sign folds in)
        let ca = cos(angle), sa = sin(angle)
        let localVX = delta.x * ca + delta.y * sa
        let speed   = simd_length(delta)
        let moving  = speed > 0.00008

        // Targets: inertia opposes motion (negate velocity) for the lean;
        // jostle is always upward, driven by total speed.
        let tLeanX  = moving ? min(max(-localVX * 4.0, -0.35), 0.35) : Float(0)
        let tJostle = moving ? min(speed * 0.22, Float(0.018))        : Float(0)

        var s = states[id] ?? DragSway()
        if moving {
            s.leanX  += (tLeanX  - s.leanX)  * 0.24
            s.jostle += (tJostle - s.jostle)  * 0.28
        } else {
            s.leanX  *= 0.91   // ~530 ms settle at 60 fps
            s.jostle *= 0.88
        }
        states[id] = s

        // Add the collision recoil on top, then advance / retire its spring.
        var out = s
        if var imp = impacts[id] {
            out.leanX  += imp.leanPos
            out.jostle += imp.josPos
            imp.step()
            if imp.atRest { impacts[id] = nil } else { impacts[id] = imp }
        }
        out.leanX  = min(max(out.leanX, -0.45), 0.45)
        out.jostle = min(max(out.jostle, 0), 0.05)   // ≥0: a knock hops up, never sinks in
        return out
    }

    /// Record a collision knock for object `id`: it was travelling along world-plane
    /// direction `worldDir` at `speed` (metres/frame) when something stopped it. Kicks
    /// the recoil spring so the object lurches the way it was heading, then rocks back
    /// and settles. Caller fires this only on the *rising edge* of contact, so holding
    /// the piece against a wall doesn't buzz. No-op for micro-nudges.
    public mutating func registerImpact(id: Int, worldDir: SIMD2<Float>, speed: Float, angle: Float) {
        let mag = simd_length(worldDir)
        guard mag > 1e-5, speed > 0.004 else { return }
        let dir = worldDir / mag
        // Project travel onto the object's local X — same basis as the eased lean. The
        // kick is +localX (forward): on a sudden stop the object pitches the way it was
        // going, opposite the trailing lean of steady acceleration.
        let ca = cos(angle), sa = sin(angle)
        let localX   = dir.x * ca + dir.y * sa
        let leanKick = min(max(localX * speed * 4.0, -0.16), 0.16)
        let josKick  = min(speed * 1.5, Float(0.020))
        var imp = impacts[id] ?? ImpactSpring()
        imp.leanVel += leanKick
        imp.josVel  += josKick
        impacts[id] = imp
    }
}
