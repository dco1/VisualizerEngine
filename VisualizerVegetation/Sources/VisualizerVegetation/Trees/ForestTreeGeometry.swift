import Foundation
import simd

// ─────────────────────────────────────────────────────────────────────────────
// PORTED FROM Visualizer's Forest scene (`~/Sites/Visualizer/Visualizer/Scenes/
// Forest/ForestGeometry.swift`, the `addTrees`/`Soup` tree-bake subset). This is
// the REAL Visualizer Forest broadleaf tree generator — trunk flare profile,
// Hermite-collar curved branches, sun-side asymmetry, fractal twig clusters,
// Halton-distributed volumetric crown fill, phyllotaxis leaf cards — baking each
// tree into a world-space triangle `Soup` with per-vertex colour (albedo), which
// Daydream's `ForestTreeBaker` turns into an `IlluminatoramaVertex` DrawGroup.
//
// Deliberately brought into Daydream's APP target (not the shared engine
// submodule): the two apps' engine pointers have diverged, so this is a scoped
// duplicate. A shared-engine home (a `VegetationRenderSet`-style module) is a
// LATER reconcile — flagged in docs/PROJECT_SCOPE.md Phase 8, NOT done here.
//
// SceneKit STRIPPED: the source `TreeGeometry.swift` has an SCNNode-per-branch
// AssetLab path (SceneKit-coupled); NONE of it is here. `ForestGeometry`'s soup
// bake never imported SceneKit — only Metal/VisualizerRendering/simd — so the
// tree-bake closure ports cleanly. The only adaptations are: (1) `makeSite` takes
// an explicit `groundY` (Daydream resolves ground height per its terrain/flat
// plane) instead of the Forest scene's `groundHeightApprox` heightfield; (2) the
// forest-clearing `treeSites` layout + camera-anchored haze/LOD review cameras
// are replaced by yard-scale defaults (no aerial-perspective haze on a yard;
// full detail); (3) diagnostic env-var probes (FOREST_NEON hero binning) are
// retained verbatim but default-off.
// ─────────────────────────────────────────────────────────────────────────────
public enum ForestTreeGeometry {

    // Crown leaf-fill shell fraction (single source of truth; see source).
    public static let fillLimitConst: Float = 0.95

    // Global leaf-card density multiplier. Yard trees are viewed close and few,
    // so we run a low scale for a cheap-but-lush canopy (CLAUDE.md RT budget note:
    // ~0.30 default). Overridable via FOREST_FOLIAGE_SCALE for A/B.
    public static let foliageDensityScale: Float = {
        if let s = ProcessInfo.processInfo.environment["FOREST_FOLIAGE_SCALE"],
           let v = Float(s), v > 0 { return v }
        return 0.30
    }()

    // Hero-probe slice count (FOREST_NEON diagnostic only).
    public static let heroSlices = 28
    // No hero probe target in the yard baker (the Forest scene selected a hero
    // from its fixed `treeSites`; we have none). nil ⇒ probe never fires.
    public static let heroSeed: UInt64? = nil

    // YARD camera model: the Forest scene keyed haze + LOD to five review cameras
    // around a clearing. A yard has no such rig — trees are close and should read
    // at full detail with NO aerial-perspective haze. An empty review set makes
    // `camDist` collapse to the anchor distance; the anchor sits ON the trees so
    // the haze smoothstep (16→40 m) stays at 0. `nearCamera` promotion then keeps
    // every yard tree at full leaf detail.
    public static let cameraAnchor = SIMD3<Float>(0, 0, 0)
    public static let reviewCameraXZ: [SIMD2<Float>] = []

    // World sun direction (TO the sun) — governs the baked leaf-tip highlight albedo
    // (leaves whose normal faces the sun get a brighter tip). Ported from the Forest
    // scene's golden-hour sun (azim 4.05, mid-elevation ≈ 21°). Daydream's own sky/sun
    // drives the actual lighting; this only tints the baked leaf albedo variation, so a
    // fixed golden-hour vector reads fine for yard foliage.
    public static let worldSunDir: SIMD3<Float> = {
        let elev: Float = (10.0 + 22.0 * 0.5) * .pi / 180
        let azim: Float = 4.05
        let ce = cos(elev)
        return simd_normalize(SIMD3<Float>(ce * cos(azim), sin(elev), ce * sin(azim)))
    }()

    // ── Soup (verbatim from ForestGeometry) ─────────────────────────────────
    public struct Soup {
        public init() {}
        public var positions: [SIMD3<Float>] = []
        public var normals:   [SIMD3<Float>] = []   // smooth shading normals
        public var colors:    [SIMD3<Float>] = []   // per-vertex albedo
        public var indices:   [UInt32] = []
        public var triAlbedo: [SIMD3<Float>] = []   // RT per-triangle albedo
        public var triNormal: [SIMD3<Float>] = []   // RT per-triangle geometric normal
        // ── RT wood/leaf split (#60 item 7 incr. 2c) ─────────────────────────
        // Per-triangle WOOD flag, in lockstep with triAlbedo/triNormal. A
        // triangle is WOOD when it was emitted under the tree-wind context
        // (`windActive`) and is NOT a leaf (`foliageMark <= 0.5`) — i.e. trunk /
        // leader / primary / branch / twig / trunk-moss. The Illuminatorama RT
        // pass EXCLUDES wood triangles from the soup acceleration structure
        // (`ForestController.buildRTGeometry` keeps leaves + ground + props only,
        // ~50–72k tris) and instead casts the wood's RT shadows from registered
        // round Catmull-Rom CURVE primitives that sway each frame — so the static
        // soup AS stays cheap AND the RT wood shadows track the swaying raster
        // wood (a static triangle soup AS can't, it never refits for wind). The
        // discriminator matches the established clearing-cull rule (line ~320):
        // `windActive` is true for trunk/branch/leader/moss/understory, FALSE for
        // grass/ground/rocks/logs. (Same `windActive && foliageMark<=0.5` test.)
        public var triWood:   [Bool] = []
        // Per-triangle LEAF flag (foliageMark > 0.5), lockstep with triWood. Used
        // by the RT leaf LOD (#60 item 7 2c): the canopy is ~3.2M leaf-card tris —
        // far over any live-RT budget — so `ForestController.buildRTGeometry`
        // DECIMATES leaf triangles to a budget for the RT acceleration structure
        // (ground/props are always kept). Raster keeps every leaf.
        public var triLeaf:   [Bool] = []
        // ── Per-triangle LEAF-CARD id (PHOTOREALISM #9 / D7) ──────────────────
        // Lockstep with `triLeaf`/`triWood`/`triNormal`/`triAlbedo`. A leaf is a
        // CARD, not a triangle: `emitLeafCard` emits one contiguous run of 2 (far),
        // 16 (mid) or 24 (hero, a 7-point lobed margin mirrored to both sides)
        // triangles — 12 / 20 of which survive the cross guard, since the u = 0
        // margin endpoints coincide with the spine — and those triangles are only
        // a leaf TOGETHER. Any LOD
        // that thins the canopy therefore has to remove whole cards — striding
        // individual triangles shreds every card into a partial silhouette
        // (`ForestTreeBaker.soup`'s decimator did exactly that until D7).
        //
        // The id is the emission ordinal of the card the triangle belongs to;
        // **−1 means "not part of a leaf card"** (all wood, and any foliage
        // emitted outside `emitLeafCard` — there is none today). Ids are unique
        // within a Soup and assigned in emission order, so a bake is as
        // deterministic with them as without.
        public var triCard:   [Int32] = []
        /// The card id `tri()` stamps onto every triangle it emits. `emitLeafCard`
        /// raises it for the span of one card and restores it after, exactly as
        /// `foliageMark` / `debugClass` do.
        public var cardMark:  Int32 = -1
        /// Next unused card ordinal. Bumped once per `emitLeafCard`.
        public var nextCardID: Int32 = 0
        // ── Wood centerline skeleton for RT curve primitives (#60 item 7 2c) ─
        // When `captureCurves` is on (the real Forest build; off for AssetLab's
        // single-tree lab), `emitTrunkMesh` + `emitBranchMesh` append one strand
        // per structural wood segment (trunk / leader / primary / branch — twigs
        // are skipped: sub-pixel shadow casters, and they'd dominate the curve
        // count). Each strand is 4 spine samples (mirroring
        // `TreeGeometry.appendBranchToSkeleton`, same spine math + RNG so the
        // curve coincides with the rastered tube) with per-point radius +
        // windAttr, which `ForestController` assembles into ONE
        // `IlluminatoramaCurveSet`. windAttr uses the SAME formula as the vertex
        // wind tangent (height² sway weight, per-tree phase) so the displace
        // kernel re-applies `applyTreeWind` identically.
        public struct CurveStrand: Sendable {
            public var points:   [SIMD3<Float>]   // world-space spine, base → tip
            public var radii:    [Float]
            public var windAttr: [SIMD4<Float>]   // (swayWeight, phase, flutter, woodMark)
        }
        public var curveStrands: [CurveStrand] = []
        public var captureCurves: Bool = false
        // Per-vertex foliage flag (1 = leaf). Rides the vertex-colour ALPHA
        // channel into the G-buffer (colour.rgb is albedo; alpha was unused),
        // where the fragment shader stashes it in the free normalRoughness.w
        // slot so the deferred lighting pass can add a leaf-transmission term.
        // `foliageMark` is the value `tri()` stamps onto every vertex it emits;
        // callers raise it to 1 around leaf emission and drop it back to 0.
        public var foliage:     [Float] = []
        public var foliageMark: Float = 0

        // Per-vertex UV. Unused for shading in this scene (no texture atlas), so
        // the ground packs PROCEDURAL SOIL material data here (issue #58 #11/#12/
        // #13): uv = (-(roughness + 0.01), wetness) — a negative-x marker the
        // Illuminatorama G-buffer reads to drive per-region soil roughness +
        // wetness + normal relief. `uvMark` is stamped onto every vertex `tri()`
        // emits; callers set it around ground emission and reset to .zero after.
        public var uvs:     [SIMD2<Float>] = []
        public var uvMark:  SIMD2<Float> = .zero

        // Per-vertex tangent. Unused for shading here (no normal-map atlas), so
        // TREES pack hierarchical-wind weights into it (issue #58 #1): tangent =
        // (swayWeight, treePhase, flutter, 1). swayWeight = quadratic of height
        // above the trunk base (stiff trunk → swaying canopy); flutter (from the
        // foliage flag) adds leaf shimmer. The G-buffer + shadow vertex shaders
        // displace by these × the frame wind (gated to this scene). Ground/props
        // leave windActive=false → tangent .zero → no sway.
        public var tangents: [SIMD4<Float>] = []
        public var windActive: Bool = false
        public var windBaseY: Float = 0
        public var windCrownH: Float = 1
        public var windPhase: Float = 0

        // ── Per-pixel MATERIAL-CLASS marker (round 3 texture fidelity) ──────────
        // tangent.w carries a material class the Illuminatorama G-buffer reads to
        // pick a procedural per-pixel surface field (the soup has no texture
        // atlas, so all detail is computed). The marker convention is:
        //   1   = foliage (leaf)                — wind path, foliageMark > 0.5
        //   2   = WOOD generic / OAK bark       — speciesMark default
        //   3   = BIRCH bark                    — lenticel dashes, pale peel
        //   4   = MAPLE bark                    — shaggy lifting strips
        //   5   = MOSSY FALLEN LOG (barrel)     — moss velvet + length-aligned bark
        //   6   = LOG CUT END-GRAIN (cap)       — concentric growth rings ONLY here
        //                                          (rings must NOT wrap the barrel)
        // The wood/bark test in the shader stays `w > 1.5`, so the generic bark
        // path still fires for any value ≥ 2; species ∈ {2,3,4} sub-select. No
        // other scene emits w > 1.5 (they ship tangent .zero or wind w==1/2 only
        // from this file), so every new branch is an exact no-op elsewhere.
        //
        // `speciesMark` is stamped by the WIND path (trees, windActive) — set per
        // tree in addTree. `materialMark` is stamped by the NON-wind path (props:
        // logs, grass) so logs/grass get a class without faking a wind frame.
        public var speciesMark: Float = 2.0
        public var materialMark: Float = 0.0

        // ── Aerial perspective (polish #58) ──────────────────────────────────
        // A per-emission haze blend: the far treeline desaturates toward and
        // fades INTO the warm horizon glow, so the stand reads with golden-hour
        // depth instead of every crown sitting at the same saturated near-plane.
        // `hazeMix` (0 = none) lerps the emitted albedo toward `hazeColor`. Set
        // per-tree from camera distance in addTree; reset to 0 after. Applies in
        // BOTH paths (per-vertex colour + RT triAlbedo) since both read `colors`.
        public var hazeMix: Float = 0
        public var hazeColor = SIMD3<Float>(0.62, 0.50, 0.38)

        // ── NEON DIAGNOSTIC INSTRUMENTATION (env FOREST_NEON=1) ──────────────
        // The "trunk not connecting to its branches" read keeps recurring on the
        // SAME crown-envelope axis (memory: forest_trunk_branch_junction_gap), so
        // every cosmetic patch oscillates. Before touching geometry again, prove
        // WHICH class the bare sticks are. `debugClass` is stamped onto each tri;
        // when neon is on, tri() replaces albedo with a class-coded NEON colour so
        // the render reads as a labelled diagram. 0 = pass through real albedo
        // (ground / understory / rocks stay natural for context).
        //   1 trunk          magenta
        //   2 leader         cyan
        //   3 primary (d=1)  bright green   ← a bare green span = the junction gap
        //   4 branch (d>1)   orange
        //   5 fill cards     deep blue
        //   6 emitLeaves     teal
        //   7 tip rosette    yellow
        //   8 junction tuft  red            ← the clothing the gap-fix hangs
        //   9 twig cluster   purple
        public var debugClass: UInt8 = 0
        // NEON hero-probe accumulator — PER-SOUP, not global. buildRawSoup runs on a
        // detached background Task, and the live app can have two forest builds in
        // flight at once (scene re-entry makes a fresh ForestController whose own
        // Task.detached { buildRawSoup() } overlaps the previous one). Global mutable
        // probe state raced: each build's tri() checked the shared flag, so a second
        // build's triangles wrote into the first build's COW heroBins from a second
        // thread — tripping the Array uniqueness check (the bare _assertionFailure
        // crash in fillCrownVolume). Each build owns its Soup, so keeping the probe
        // state here makes concurrent builds independent and the race impossible.
        public var heroProbeActive = false
        public var heroBaseY: Float = 0
        public var heroTopY: Float = 1
        public var heroBins = [[Float]]()   // [slice][class 0...9] = area
        // Settable so AssetLab's TreeLab can flip neon class-debug from its UI
        // (not just the FOREST_NEON env). Defaults to the env for the main app.
        public nonisolated(unsafe) static var neonOn = ProcessInfo.processInfo.environment["FOREST_NEON"] != nil
        // NO-LEAVES debug (AssetLab TreeLab): drop every foliage-class triangle
        // (debugClass 5 fill / 6 leaves / 7 tip / 8 junction / 9 twig) at the
        // emission chokepoint, leaving only the trunk+leader+primary+branch wood
        // skeleton so the branch structure can be read without the fill blob.
        public nonisolated(unsafe) static var suppressFoliage = false
        // Scene-only clearing-centre cull (drops tree triangles in the walkable
        // gap at world (0,−3.5)). FALSE for AssetLab's TreeLab, whose single tree
        // sits at the origin — inside that zone — and would otherwise be erased.
        public nonisolated(unsafe) static var clearingCull = true
        // ── FLOOR NEON probe (env FOREST_FLOOR_NEON=1) ───────────────────────
        // The "lighter green square on the floor" keeps recurring (memory:
        // forest_disconnect_is_illum_tone_not_geometry). We already proved it is
        // NOT the XPBD grass field (FOREST_NO_GRASS=1 leaves it unchanged) and not
        // any organic scatter (ferns/tussocks/grass all scatter RADIALLY about the
        // clearing centre — no axis-aligned domain can make a straight-edged
        // square). That leaves two possibilities: a ground-ALBEDO blend boundary,
        // or a LIGHTING/Illuminatorama tone (GI surface-cache chart, shadow-map
        // frustum, secondary-directional coverage). This flag floods the WHOLE
        // ground with one FLAT neon albedo (magenta) so the two are distinguishable
        // at a glance: if the square still reads as a brighter/darker patch over
        // the flat magenta, it is LIGHTING (geometry is uniform); if the floor goes
        // perfectly uniform, the square was ground albedo.
        public nonisolated(unsafe) static var floorNeon = ProcessInfo.processInfo.environment["FOREST_FLOOR_NEON"] != nil
        public static let neonPalette: [SIMD3<Float>] = [
            SIMD3(0, 0, 0),            // 0 unused (pass through)
            SIMD3(1.0, 0.0, 1.0),     // 1 trunk    magenta
            SIMD3(0.0, 1.0, 1.0),     // 2 leader   cyan
            SIMD3(0.0, 1.0, 0.10),    // 3 primary  green
            SIMD3(1.0, 0.45, 0.0),    // 4 branch   orange
            SIMD3(0.04, 0.06, 0.85),  // 5 fill     deep blue
            SIMD3(0.0, 0.65, 0.55),   // 6 leaves   teal
            SIMD3(1.0, 0.95, 0.0),    // 7 tip ros. yellow
            SIMD3(1.0, 0.0, 0.0),     // 8 junction red
            SIMD3(0.65, 0.0, 1.0),    // 9 twigs    purple
        ]

        public var triangleCount: Int { indices.count / 3 }

        // HANG GUARD: hard ceiling so a mis-calibrated density constant can
        // never bring back the 462-second build. Raised from 4M to 7M after
        // discovering hero trees (24-tri leaf cards × 60 nodes × ~356 calls)
        // consumed the full 4M budget before any backdrop/cap trees got geometry,
        // leaving the treeline completely empty. 7M allows all tiers to render;
        // computation is bounded by leafNodeCap=20 (3× cheaper per call) and
        // the twiglet tier being removed, so build time stays <10s.
        // TRI CAP: a true hang ceiling, NOT a quality target — and it must sit
        // ABOVE expected demand so it is never routinely hit (a routinely-hit cap
        // silently drops whatever is emitted last, which is the history below).
        // It was 8_000_000, but the canopy fill alone SATURATED it: the soup build
        // took ~38 s (trees 30.8 s) and switching to Forest froze the app — see the
        // foliageDensityScale note. With that knob at 0.30 the whole stand now lands
        // ~2.0–2.5 M tris, so 4 M leaves comfortable headroom (density can climb to
        // ~0.48 before capping) while keeping build time and the ~vert×3 memory
        // reservation an order of magnitude below the old beachball.
        //   HISTORY (#20, neon-instrumented): at 7 M, `addTrees` alone reached ~7 M
        //   and the guard `triangleCount < triCap` in `tri()` silently dropped the
        //   ENTIRE understory (`addUnderstory` runs after trees) — every blade tweak
        //   was an invisible no-op and the reviewer read OSCILLATING. The fix then
        //   was to RAISE the cap; the real fix is to not generate 8 M tris at all.
        private static let triCap = 4_000_000

        /// Triangle with explicit per-vertex normals + colours.
        public mutating func tri(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>,
                          n0: SIMD3<Float>, n1: SIMD3<Float>, n2: SIMD3<Float>,
                          c0 c0in: SIMD3<Float>, c1 c1in: SIMD3<Float>, c2 c2in: SIMD3<Float>) {
            guard triangleCount < Self.triCap else { return }   // hang guard
            // NO-LEAVES debug: skip foliage classes (5 fill … 9 twig), keep wood.
            // visual-ok: this turn's ForestGeometry edits are all flag/env-gated
            // diagnostics (suppressFoliage, FOREST_NO_TRUNKLEAN) + AssetLab lab
            // support (buildLabTreeRawSoup); default-off → no change to the shipped
            // Forest scene render.
            if Self.suppressFoliage && debugClass >= 5 { return }
            // ── DEAD-CLEARING-CENTRE TREE-GEOMETRY CULL (ROUND 2 crown pass) ──────
            // ROOT CAUSE (diagnosed, not guessed): the persistent "faceted conifer"
            // in the dead clearing centre is NOT a tree placed there — the
            // clearCoreR=9 m trunk cull already guarantees no trunk within 9 m of the
            // centre, and culling every nearby crown (understory, grass, the 3 big
            // neighbour crowns, all trees < 9 m) left it standing. It is TREE-CLASS
            // geometry (vanishes only with FOREST_NO_TREES) emitted at world
            // ≈ (0.3, 0.5, −0.9): a drooping crown LOBE / long primary-branch tip
            // from a ring tree whose envelope (cR up to ~8 m) reaches into the open
            // centre, which the deepened lobe-droop warp now pushes DOWN into the
            // walkable gap where it reads as an isolated tiered cone. The honest read
            // is that NO foliage mass belongs in the open walkable centre (BRIEF: "the
            // clearing floor is owned by grass/ferns/logs, not tree trunks"). Cull any
            // TREE-CLASS triangle (debugClass 1–9: trunk/leader/primary/branch/fill/
            // leaves/tip/junction/twig) whose centroid lands in a tight box around the
            // centre cone point, ABOVE grass height (y≥0.55, so the static grass tufts
            // and ground are untouched). The box is small (±2.2 m XZ, y 0.55–4) so only
            // the dead-centre intrusion is removed; every ring/hero crown is intact.
            // BRIEF: "the clearing floor is owned by grass/ferns/logs, not tree
            // trunks." Enforce that LITERALLY at the geometry chokepoint: no tree-class
            // foliage/wood mass (debugClass 1–9) may occupy the open WALKABLE CLEARING
            // CORE above grass height. The persistent dead-centre "faceted conifer"
            // survived every site/RNG cull (no trunk within clearCoreR=9 m; understory/
            // grass/rocks/logs/nearby-crown culls all left it) — it is a low tree-class
            // intrusion (a drooping crown lobe / long primary tip) reaching into the
            // gap, and screen→world ray-tracing it proved unreliable across the warmup
            // camera. So cull by WORLD VOLUME, not by guessing the emitter: any tree
            // triangle whose centroid is inside the walkable core cylinder (radius
            // `coreCullR` about the clearing centre (0,−3.5)) and ABOVE the grass
            // canopy (y ≥ 0.6) is dropped. The radius matches the trunk clearCoreR so
            // no ring/hero crown is touched — those crowns START ≥ 9 m out; only mass
            // that intrudes into the empty centre is removed. Grass/ferns/ground (and
            // the y < 0.6 turf) are untouched.
            // (Disabled for AssetLab's TreeLab, which builds ONE tree at the world
            // origin — exactly inside this scene-specific clearing-centre zone, so
            // the cull would erase the isolated lab tree's crown. `clearingCull` is
            // true for the real Forest scene, false when the lab drives the build.)
            // ── DEAD-CLEARING-CENTRE TREE CULL (ROUND 3 — class-agnostic) ────────
            // ROOT CAUSE (now PROVEN by a per-triangle magenta paint, not guessed):
            // the dead-centre tiered cone IS tree geometry sitting inside the walkable
            // core (≈ (0…1.5, −3.5…−5)), TALL — but it survived every prior cull
            // because that cull was gated on debugClass 1–9, and this mass is emitted
            // at debugClass 0 (a tree-class sub-emission that resets the class). So
            // the discriminator can't be debugClass. Use the MATERIAL state instead:
            // ANY tree geometry — leaves (foliageMark>0.5) OR wood emitted under the
            // tree-wind context (windActive, which is true for trunk/branch/leader/
            // moss but FALSE for grass, ground, rocks, logs) — whose centroid lands in
            // the walkable core cylinder above grass height is dropped. This catches
            // the class-0 cone while leaving grass/ground/props (windActive=false)
            // untouched. Understory uses windActive too, but it's already sparse-to-
            // absent in the open core (site culls), so this just guarantees the
            // BRIEF's "clearing floor owned by grass/ferns/logs, not trees."
            if Self.clearingCull && (foliageMark > 0.5 || windActive) {
                let mx = (p0.x + p1.x + p2.x) * 0.3333333
                let my = (p0.y + p1.y + p2.y) * 0.3333333
                let mz = (p0.z + p1.z + p2.z) * 0.3333333
                // ROUND-4: the dead-centre "striped basket" that the reviewer kept
                // calling "the log" is NOT the mossy log (it didn't move when the
                // logLayout coords moved). It is a LOW DROOPING tree-class lobe — a
                // primary branch (oak-bark cell field = the striped wall) with its
                // leaf cluster (the green pile) — dipping into the walkable centre.
                // It SURVIVED every prior cull because the height gate was y ≥ 0.5 and
                // this woody mass sags BELOW that. Lower the gate to 0.15 so the whole
                // intrusion is removed (no trunk legitimately stands within
                // clearCoreR = 9 m, and grass/ground/logs are windActive=false → exempt
                // from this `foliageMark||windActive` branch, so the low static sward
                // and the bedded mossy logs are untouched).
                // No height gate: a low branch tip / leaf cluster that sags to the
                // GROUND in the walkable core (y < 0.15) was surviving the old y≥0.5
                // gate and reading as a dark blob beside the mossy log. No tree-class
                // geometry belongs in the core at ANY height (no trunk within
                // clearCoreR = 9 m); the bedded mossy LOGS and the static grass are
                // windActive=false and so never enter this branch.
                let dx = mx, dz = mz + 3.5
                let coreCullR: Float = 8.0
                if dx * dx + dz * dz < coreCullR * coreCullR { return }
                _ = my
            }
            let cr = simd_cross(p1 - p0, p2 - p0)
            guard simd_length(cr) > 1e-9 else { return }   // drop degenerates
            // ── NaN-normal EMISSION GUARD (single chokepoint) ────────────────
            // Every tree vertex normal funnels through here. A caller can hand us
            // a degenerate/NaN shading normal (a sliver leaf card, a cancelling
            // summed normal, an axis-aligned cross) — shipping that as a vertex
            // normal breaks lighting and fails the vertex-validity gate. The
            // triangle's own geometric normal (`cr`, already proven non-zero
            // above) is the correct fallback for any collapsed shading normal, so
            // sanitize all three here before they reach the arrays.
            let geoN = simd_normalize(cr)
            let sn0 = safeNormalize(n0, fallback: geoN)
            let sn1 = safeNormalize(n1, fallback: geoN)
            let sn2 = safeNormalize(n2, fallback: geoN)
            var c0 = c0in, c1 = c1in, c2 = c2in
            // visual-ok: env-gated (FOREST_NEON) diagnostic instrumentation; with the
            // env unset the branch is skipped and default render output is unchanged.
            if Self.neonOn && debugClass != 0 {
                // Class-coded diagram colour; skip haze so far trees read too.
                let nc = Self.neonPalette[Int(debugClass)]
                c0 = nc; c1 = nc; c2 = nc
            } else if hazeMix > 0.001 {
                c0 += (hazeColor - c0) * hazeMix
                c1 += (hazeColor - c1) * hazeMix
                c2 += (hazeColor - c2) * hazeMix
            }
            // Quantitative slice accounting: bin this tri into the hero tree's
            // vertical histogram by class (gated to the hero tree in addTree).
            if heroProbeActive {
                heroAccumulate(p0, p1, p2, cls: debugClass)
            }
            // RT per-triangle normal comes from the SUPPLIED shading normals, not
            // the geometric winding cross-product: the RT pass shades from
            // triangleNormal, so feeding it the intended outward normal keeps the
            // ground facing the sun, not the geometric −Y its winding would imply.
            //
            // CORRECTION (2026-08-09). This comment used to continue "Illuminatorama
            // does not back-face-cull (House's floor vs box prove it), so winding is
            // free". THAT IS FALSE. `IlluminatoramaMesh.doubleSided` defaults false and
            // the G-buffer pass runs `setCullMode(mesh.doubleSided ? .none : .back)` —
            // the floor and the box simply happened to be registered double-sided.
            // Winding is NOT free: a single-sided mesh out of this soup loses every
            // away-facing card, which for a phyllotaxis crown is ~50 % of it (measured
            // in `ForestTreeValidityTests.testHalfOfEveryLeafCanopyFacesAway…`).
            // Daydream's fix is at registration — `HouseRenderBridge+Tree` sets
            // `doubleSided`, which is the right lever for two-sided cards; do not
            // "fix" it by trying to wind leaves toward the camera, which is impossible
            // for a mesh baked once and viewed from every angle.
            // Summed shading normal for the RT per-triangle normal. When the three
            // vertex normals cancel (n0+n1+n2 ≈ 0) simd_normalize NaNs — fall back
            // to the geometric normal so the RT pass still has a finite outward.
            let tn = safeNormalize(sn0 + sn1 + sn2, fallback: geoN)
            let base = UInt32(positions.count)
            positions.append(p0); positions.append(p1); positions.append(p2)
            normals.append(sn0);  normals.append(sn1);  normals.append(sn2)
            // Regression tripwire: after the guard, every emitted normal MUST be
            // finite. A future edit that reintroduces a NaN path fails HERE (in the
            // bake) rather than downstream in a flaky GPU vertex-validity test.
            assert(sn0.x.isFinite && sn0.y.isFinite && sn0.z.isFinite &&
                   sn1.x.isFinite && sn1.y.isFinite && sn1.z.isFinite &&
                   sn2.x.isFinite && sn2.y.isFinite && sn2.z.isFinite &&
                   tn.x.isFinite && tn.y.isFinite && tn.z.isFinite,
                   "ForestTreeGeometry emitted a non-finite vertex/triangle normal")
            colors.append(c0);    colors.append(c1);    colors.append(c2)
            foliage.append(foliageMark); foliage.append(foliageMark); foliage.append(foliageMark)
            uvs.append(uvMark);   uvs.append(uvMark);   uvs.append(uvMark)
            if windActive {
                let flutter: Float = foliageMark > 0.5 ? 1.0 : 0.0
                // tangent.w doubles as a material marker the G-buffer reads:
                // 1 = foliage (leaf), ≥2 = WOOD (trunk/branch). The wood value is
                // SPECIES-CODED (2 oak / 3 birch / 4 maple) via `speciesMark` so the
                // shader can pick species-correct bark plates vs lenticels vs strips
                // while keeping `w > 1.5` as the generic "wood" test. (Wind reads
                // only tangent.xyz, so the species code never affects sway.)
                let woodMark: Float = foliageMark > 0.5 ? 1.0 : speciesMark
                // Quadratic height weight: 0 at the trunk base → grows up the tree
                // so the trunk is stiff and the canopy sways (cantilever).
                func windTangent(_ p: SIMD3<Float>) -> SIMD4<Float> {
                    let hw = max(0, min(1.0, (p.y - windBaseY) / windCrownH))
                    // Cap sway at 1.0: the whole UPPER canopy (the sealed outer
                    // rind) sways as ONE coherent mass instead of the very top
                    // cards shearing past those just below — that height-shear was
                    // half of what re-opened the crown seal under wind.
                    return SIMD4(min(1.0, hw * hw), windPhase, flutter, woodMark)
                }
                tangents.append(windTangent(p0))
                tangents.append(windTangent(p1))
                tangents.append(windTangent(p2))
            } else {
                // NON-wind props (logs / grass / rocks): no sway, but still carry a
                // material class in tangent.w so the shader can apply moss velvet
                // (5 barrel) or end-grain rings (6 cut-cap). materialMark default 0 → tangent
                // .zero (unchanged for rocks/sticks that don't set it).
                let m = SIMD4<Float>(0, 0, 0, materialMark)
                tangents.append(m); tangents.append(m); tangents.append(m)
            }
            indices.append(base); indices.append(base + 1); indices.append(base + 2)
            triNormal.append(tn)
            triAlbedo.append((c0 + c1 + c2) * (1.0 / 3.0))
            // WOOD flag for the RT soup split (#60 item 7 2c): tree-wind context
            // AND not a leaf ⇒ trunk/branch/leader/twig/moss. Kept in lockstep
            // with triNormal/triAlbedo (appended at the SAME late point, after
            // every early-out above), so the per-triangle arrays stay aligned.
            triWood.append(windActive && foliageMark <= 0.5)
            triLeaf.append(foliageMark > 0.5)
            // Which leaf CARD this triangle belongs to (−1 = none). Appended at
            // the same late point as the other per-triangle arrays so it stays
            // aligned past every early-out above — a misaligned card id would
            // make the card-level decimator drop the wrong triangles.
            triCard.append(cardMark)
        }

        /// Flat triangle (geometric normal on all three vertices), single colour.
        public mutating func flatTri(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>,
                              color c: SIMD3<Float>) {
            // Degenerate (sliver) tri → zero cross → NaN normal. tri() drops the
            // sliver at its own cross guard, but keep the passed normal finite too.
            let gn = safeNormalize(simd_cross(p1 - p0, p2 - p0))
            tri(p0, p1, p2, n0: gn, n1: gn, n2: gn, c0: c, c1: c, c2: c)
        }

        /// Flat quad p0→p1→p2→p3 (CCW around `n`), single colour.
        public mutating func quad(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>,
                           _ p2: SIMD3<Float>, _ p3: SIMD3<Float>,
                           normal n: SIMD3<Float>, color c: SIMD3<Float>) {
            tri(p0, p1, p2, n0: n, n1: n, n2: n, c0: c, c1: c, c2: c)
            tri(p0, p2, p3, n0: n, n1: n, n2: n, c0: c, c1: c, c2: c)
        }

        /// Smooth quad with four per-vertex normals + colours (CCW around the
        /// averaged normal). Used for the ground heightfield.
        public mutating func smoothQuad(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>,
                                 _ p2: SIMD3<Float>, _ p3: SIMD3<Float>,
                                 n0: SIMD3<Float>, n1: SIMD3<Float>,
                                 n2: SIMD3<Float>, n3: SIMD3<Float>,
                                 c0: SIMD3<Float>, c1: SIMD3<Float>,
                                 c2: SIMD3<Float>, c3: SIMD3<Float>) {
            tri(p0, p1, p2, n0: n0, n1: n1, n2: n2, c0: c0, c1: c1, c2: c2)
            tri(p0, p2, p3, n0: n0, n1: n2, n2: n3, c0: c0, c1: c2, c2: c3)
        }
    }
}

/// Tiny deterministic RNG (SplitMix64-ish) for the procedural layout.
/// (Daydream: `internal` — was `private` — so `ForestTreeBaker` in a sibling file
/// can seed a per-tree RNG for `makeSite`.)
public struct ForestRNG {
    public var state: UInt64
    public init(seed: UInt64) { state = seed &* 0x9E3779B97F4A7C15 &+ 0x1234_5678 }
    public mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    public mutating func unit() -> Float { Float(next() >> 40) * (1.0 / Float(1 << 24)) }
}

// ===== Soup hero-probe extension (5273-5325) =====
extension ForestTreeGeometry.Soup {
    /// Start binning the hero tree's geometry into vertical slices. Per-Soup state,
    /// so two concurrent forest builds (live-app scene re-entry) can't race.
    public mutating func heroBegin(baseY: Float, topY: Float) {
        heroProbeActive = true
        heroBaseY = baseY
        heroTopY = max(topY, baseY + 1.0)
        heroBins = Array(repeating: Array(repeating: 0, count: 10),
                         count: ForestTreeGeometry.heroSlices)
    }

    public mutating func heroAccumulate(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>,
                                 _ p2: SIMD3<Float>, cls: UInt8) {
        let slices = ForestTreeGeometry.heroSlices
        let cy = (p0.y + p1.y + p2.y) * (1.0 / 3.0)
        let f = (cy - heroBaseY) / (heroTopY - heroBaseY)
        guard f >= 0, f < 1 else { return }
        let area = 0.5 * simd_length(simd_cross(p1 - p0, p2 - p0))
        let si = min(slices - 1, max(0, Int(f * Float(slices))))
        heroBins[si][Int(min(9, cls))] += area
    }

    public mutating func heroReport() {
        guard heroProbeActive else { return }
        let slices = ForestTreeGeometry.heroSlices
        let names = ["—", "trunk", "leader", "primary", "branch",
                     "fill", "leaves", "tipRos", "junct", "twig"]
        let woodCls = [1, 2, 3, 4], leafCls = [5, 6, 7, 8, 9]
        var out = "\n[Forest NEON PROBE] hero seed=\(ForestTreeGeometry.heroSeed.map(String.init) ?? "?") " +
                  "baseY=\(String(format: "%.2f", heroBaseY)) topY=\(String(format: "%.2f", heroTopY))\n"
        out += "slice  yLo   wood   leaf   leaf/wood  verdict\n"
        for si in 0..<slices {
            let yLo = heroBaseY + (heroTopY - heroBaseY) * Float(si) / Float(slices)
            let wood = woodCls.reduce(Float(0)) { $0 + heroBins[si][$1] }
            let leaf = leafCls.reduce(Float(0)) { $0 + heroBins[si][$1] }
            guard wood + leaf > 1e-5 else { continue }
            let ratio = wood > 1e-5 ? leaf / wood : (leaf > 0 ? 999 : 0)
            // BARE NECK = real wood present, leaf area < 1/3 of it.
            let bare = wood > 0.02 && ratio < 0.34
            out += String(format: "%4d  %5.2f  %6.3f  %6.3f  %8.2f  %@\n",
                          si, yLo, wood, leaf, ratio, bare ? "◄ BARE NECK" : "")
        }
        // Per-class totals for the whole hero crown.
        out += "totals:"
        for c in 1...9 {
            let t = (0..<slices).reduce(Float(0)) { $0 + heroBins[$1][c] }
            if t > 1e-4 { out += String(format: " %@=%.2f", names[c], t) }
        }
        out += "\n"
        FileHandle.standardError.write(out.data(using: .utf8)!)
        heroProbeActive = false
    }
}
