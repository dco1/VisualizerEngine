// ── ILLUMINATORAMA — G-BUFFER PROCEDURAL BARK (shared) ────────────────────────────────────────
//
// Species-correct bark (oak plates, birch lenticels, maple strips) and the mossy fallen log's
// barrel + cut end, for wood marked by its species code in worldTangent.w (> 1.5 — Forest wood
// only; an exact no-op for everything else). Split out of IlluminatoramaGBufferShade.h (Signack
// S4, Daydream DH-1461); body unchanged.
#pragma once

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaMaterial.h"
#include "IlluminatoramaGBufferDeform.h"
using namespace metal;

static inline __attribute__((always_inline)) void illumiSurfaceBark(VSOut in, Instance inst,
                                     thread float3& n, thread float3& albedo, thread float& roughness) {
    // ── Procedural SPECIES-CORRECT bark + mossy-log surface (#58 / round 3) ──
    // ForestGeometry marks WOOD in worldTangent.w with a SPECIES/material code:
    //   2 = oak (deep furrowed PLATES)   3 = birch (pale + lenticel dashes)
    //   4 = maple (shaggy lifting strips) 5 = mossy fallen LOG (moss velvet)
    // Leaves are 1, everything else 0/±1, and NO other scene emits w > 1.5 (every
    // non-Forest mesh ships tangent .zero or, from this same file, wind w==1/2 on
    // the *same* Forest scene only). So `w > 1.5` gates cleanly to Forest wood and
    // is an exact no-op for foliage, ground, every prop without a marker, and every
    // other scene. The detail is COMPUTED (the soup has no texture atlas) and is far
    // crisper than the coarse 18-radial trunk mesh could carry.
    if (inst.normalTextureSlice < 0 && in.worldTangent.w > 1.5) {
        float species = in.worldTangent.w;                 // 2/3/4/5
        float ang = atan2(n.z, n.x);                       // sweeps around the trunk
        float3 tang = normalize(cross(n, float3(0.0, 1.0, 0.0)) + float3(1e-4, 0, 0));
        float yW = in.worldPos.y;
        float e  = 0.14;
        // Bark is a dry, dusty, fissured dielectric — about the roughest thing in a yard
        // (DH-0629). At the instance's 0.82 the plate relief's steep normals caught the sky in
        // the specular-IBL lobe, and a hero trunk read as BLUE-GREY GLOSSY SLATE.
        roughness = max(roughness, 0.95f);

#if FOREST_SPECIES_DEBUG
        // TEMP marker probe: flat colour per species marker so the contract
        // (geometry stamp vs shader gate) can be read off one render.
        // oak2=red, birch3=white, maple4=blue, log5=green, cap6=yellow.
        if (species < 2.5)      albedo = float3(1.0, 0.0, 0.0);
        else if (species < 3.5) albedo = float3(1.0, 1.0, 1.0);
        else if (species < 4.5) albedo = float3(0.0, 0.3, 1.0);
        else if (species < 5.5) albedo = float3(0.0, 1.0, 0.0);
        else                    albedo = float3(1.0, 1.0, 0.0);
#else
        if (species < 2.5) {
            // ── OAK — deep furrows broken into discrete shedding PLATE CELLS ────
            // ROUND-3 RE-ARCHITECT (the previous furrow-only field still read as a
            // smooth airbrushed gradient in hero_oakbark_tight — no plate structure
            // survived to 3–6 m). The fix the brief asks for is a real CELL field:
            // a Worley F1 lattice (barkCellF1) carves the trunk into discrete
            // plates, the inter-cell DISTANCE drives a dark furrow crack between
            // them, and each cell takes a COHERENT ridge tint by its per-cell ID.
            // The lattice is ANISOTROPIC (ang×3.4 around, yW×1.1 up ⇒ ≈18 cm wide ×
            // ≈30 cm tall plate cells — oak-correct vertical-ish plates) so it reads
            // as stacked vertical bark scales, not bubbles. A vertical furrow octave
            // grooves WITHIN each plate; a fine octave adds the close-up grain.
            float2 cellP = float2(ang * 3.4, yW * 1.1);
            float2 cell = barkCellF1(cellP);
            float crackDist = cell.x;                 // small ⇒ near a plate crack
            float cellID    = cell.y;                 // coherent per-plate random
            // Crack mask: 1 deep in a plate, 0 in the dark furrow border between
            // plates. Tight band so plates have flat lit faces + sharp dark grooves.
            float plate = smoothstep(0.06, 0.24, crackDist);
            // Vertical grooving within the plate (oak's lengthwise fissures).
            float2 vc  = float2(ang * 9.0, yW * 0.7);
            float vf   = soilValueNoise(vc);
            float vfa  = soilValueNoise(vc + float2(e, 0));
            float vfy  = soilValueNoise(vc + float2(0, e));
            float vGroove = smoothstep(0.38, 0.60, vf);
            float fine = soilValueNoise(float2(ang * 38.0, yW * 9.0)) - 0.5;
            // ── ROUND-5 STRUCTURAL #1: DOMAIN-WARPED BRICK-OFFSET HORIZONTAL CRACKS ─
            // The round-4 field (`hPhase = yW*7 + lowFreqAngNoise`) used a phase that
            // varied only SLOWLY with `ang` (×1.7), so adjacent furrow columns shared
            // nearly the same crack height — the horizontal cracks LINED UP across the
            // trunk into continuous rows. Combined with the regular ×9 vertical groove
            // that reads as a KNITTED MESH / SNAKESKIN grid (the basket-weave defect).
            //
            // The fix the reviewer prescribes: per-furrow-COLUMN brick-offset so plate
            // seams stagger between adjacent columns (never align into a grid), plus
            // low-freq jitter on BOTH crack spacing AND amplitude so some plates are
            // tall, some squat, and some cracks fade out entirely — cells read as
            // IRREGULAR polygons. Derive a discrete column index from `ang`, hash it
            // for an independent phase OFFSET (the brick stagger) and a per-column row
            // SPACING + a per-column crack DEPTH, so no two columns share a row line.
            float colW = 6.5;                                  // ~6–7 furrow columns/m circumference
            float colF = ang * colW;
            float colIdx = floor(colF);                        // furrow-column index
            // Brick stagger: each column gets its own phase offset in [0,1) so the
            // crack rows of neighbouring columns never align into a continuous course.
            float colPhase = soilHash(float2(colIdx, 3.1));
            // Per-column row SPACING jitter (±35 %): some columns stack squat plates,
            // some tall — so the horizontal seams don't form a regular ladder.
            float colSpace = 5.5 + 4.0 * soilHash(float2(colIdx, 11.7));   // 5.5–9.5 rows/m
            // Per-column crack DEPTH jitter: some columns' cracks fade out (shallow),
            // so a few plates merge vertically — irregular polygon cells, not a weave.
            // ROUND-5: raise the FLOOR (0.62) so EVERY column shows real horizontal
            // cracks — the prior 0.45 floor let too many columns merge vertically, so
            // the trunk read as continuous combed strands (shaggy maple) instead of
            // discrete stacked OAK plates. A few still merge (0.62→1.0), keeping the
            // irregular-polygon read without losing the plate-row structure.
            float colDepth = 0.62 + 0.38 * soilHash(float2(colIdx, 23.9));
            // Smoothly blend the per-column phase/spacing across the column boundary so
            // the brick offset doesn't itself read as a hard vertical seam line.
            float colMix = smoothstep(0.0, 1.0, fract(colF));
            float colPhaseB = soilHash(float2(colIdx + 1.0, 3.1));
            float colSpaceB = 5.5 + 4.0 * soilHash(float2(colIdx + 1.0, 11.7));
            float colDepthB = 0.45 + 0.55 * soilHash(float2(colIdx + 1.0, 23.9));
            float phase = mix(colPhase, colPhaseB, colMix);
            float space = mix(colSpace, colSpaceB, colMix);
            float crackDepthCol = mix(colDepth, colDepthB, colMix);
            // Low-freq AMPLITUDE jitter on top (slow in y) so a plate is taller here,
            // squatter there even within a column run.
            float ampJ = 0.7 + 0.6 * soilValueNoise(float2(colIdx * 0.7, yW * 0.6));
            float hPhase = (yW * space) * ampJ + phase * 1.31;
            float hTri = abs(fract(hPhase) - 0.5) * 2.0;        // 0 at crack → 1 mid-cell
            // crackDepthCol scales the crack WIDTH: shallow columns barely groove (the
            // "some cracks fade out" cue) so plates merge into irregular polygons.
            // ROUND-5: WIDEN the crack band (0.10→0.34) so the dark horizontal seam is
            // fat enough to read at hero range and physically separate the plate rows.
            float hCrack = mix(1.0, smoothstep(0.12, 0.34, hTri), crackDepthCol);
            // Combined "is this lit ridge-top vs dark damp floor": a pixel reads as
            // ridge only when inside a plate AND on a vertical ridge AND not in a
            // horizontal crack — so the ridges break into discrete blocky plate cells.
            // ROUND-6 STRUCTURAL #2 (oak side): the round-5 field STILL read as soft
            // painted streaks at hero distance — the horizontal cracks barely resolved
            // into discrete stacked plate CELLS. Two reasons: (a) the `ridge` value
            // floored at plate·0.45·0.18≈0.08 so the cracks never reached a true dark
            // floor, and (b) the albedo split (crackTint 0.22 → ridgeTint ~1.3) was a
            // ~6× spread but applied to a non-zero `ridge` ⇒ washed grey, not damp-dark
            // vs grey-lifted. Push BOTH: zero the floors so cracks cut to a true dark
            // furrow floor, and split harder (darker damp floor, brighter ridge top),
            // so each plate reads as a discrete lit cell with sharp dark seams.
            // The horizontal crack now FULLY gates the ridge (multiplicative, no floor)
            // so a horizontal seam carves the vertical ridge into stacked plate cells —
            // the oak-vs-maple tell the reviewer keeps flagging.
            float ridge = plate * (0.20 + 0.80 * vGroove) * hCrack;
            // ≈5× luminance split: weathered grey-lifted ridge tops ↔ DARK damp furrow
            // floors. Per-plate value jitter so adjacent plates differ.
            float plateVal = 0.78 + 0.5 * cellID;          // 0.78..1.28 per plate
            float3 ridgeTint = float3(1.44, 1.37, 1.22) * plateVal;  // bright weathered grey
            float3 crackTint = float3(0.20, 0.155, 0.11);           // dark damp furrow floor (DH-0629: was 0.135/0.105/0.075 — over the vertex seam darkening it crushed the bole to near-black)
            albedo *= mix(crackTint, ridgeTint, ridge) * (0.90 + 0.22 * fine);
            // Sparse lichen blotches — pale grey-green, slightly SMOOTHER — only on
            // the lit plate tops, keyed to whole plates (per-cell) so they blotch a
            // plate, not a pixel.
            float lichSel = soilHash(floor(cellP) + 5.1);
            float lich = soilValueNoise(float2(ang * 3.5 + 31.0, yW * 0.9 + 9.0));
            float lichM = step(0.82, lichSel) * smoothstep(0.55, 0.85, lich) * plate;
            albedo = mix(albedo, float3(0.42, 0.47, 0.36), lichM * 0.6);
            roughness = mix(roughness, roughness * 0.82, lichM);
            // Lift plate edges: push the dark cracks IN (recessed grooves) and tilt
            // the vertical grooving within plates.
            float gA = (vfa - vf) / e, gY = (vfy - vf) / e;
            float edge = 1.0 - plate;                       // 1 at the vertical plate crack
            float hEdge = 1.0 - hCrack;                     // 1 at the horizontal crack
            // ROUND-6: deepen the recess on BOTH crack axes so the plate edges lift in
            // relief (discrete cells under the rake light), and tilt the lit ridge top
            // OUTWARD along +n so the grey ridge faces the camera (plate-top read).
            n = normalize(n + tang * (gA * 1.1 + fine * 0.45)
                            + float3(0,1,0) * (gY * 0.30)
                            + n * (ridge * 0.22)                  // ridge tops lift toward camera
                            - n * (edge * 0.55 + hEdge * 0.55));  // recess BOTH crack axes deeper
        } else if (species < 3.5) {
            // ── BIRCH — pale paper bark, DARK HORIZONTAL lenticel dashes ────────
            // ROUND-3 RETUNE: at 3–10 m the old dashes (ang×22, smoothstep 0.62–0.80
            // — a narrow ~18 % duty) were too sparse + too thin to survive, and the
            // warm paper tint pushed the trunk pinkish-tan. Now: FATTER, ~3× DENSER,
            // higher-CONTRAST near-black dashes (lower angular freq ≈10–14 cm dashes,
            // wider duty), a brighter NEUTRAL-COOL paper white, and a dark peel band.
            float2 lc  = float2(ang * 2.4, yW * 0.9);          // band selector (big)
            float band = soilValueNoise(lc);
            // Dash field: MODERATE angular freq (fat dashes) broken into short
            // segments by a vertical mask. Wider smoothstep windows → ~3× the duty
            // cycle of the old field so dashes are common, not rare.
            // ROUND-5 polish #5: hero-facing birch dashes read too FAINT vs the
            // side-view birch that passes — at hero distance the moderate duty + 0.94
            // mix wasn't surviving. Widen the duty windows (denser dashes), and raise
            // the band coverage so dashes appear over more of the trunk, so hero
            // birches match the side one.
            float dashRun = soilValueNoise(float2(ang * 12.0, yW * 0.5));
            float dashGap = soilValueNoise(float2(ang * 4.5,  yW * 5.5));
            float dash = smoothstep(0.40, 0.60, dashRun) * smoothstep(0.34, 0.56, dashGap);
            // Lenticels concentrate in horizontal bands but appear broadly — wider
            // band coverage so dashes carry across more of the hero-facing trunk.
            float lent = dash * smoothstep(0.22, 0.52, band);
            // Occasional darker peel band / triangular scar (low freq, rare).
            float peel = smoothstep(0.76, 0.90, soilValueNoise(float2(ang * 1.6 + 7.0, yW * 0.5)));
            // NEUTRAL-COOL paper — but DON'T push >1.0 hard: the hero/foreground
            // birches were blowing to a featureless WHITE PAINT-DRIP smear (the
            // 1.18–1.22 multiplier compounded with the golden-hour sun term and
            // clipped, washing out the dashes). Hold the paper just under unity so
            // the trunk reads as bright bark with VISIBLE structure, not a clipped
            // white column. A faint per-trunk grey value noise breaks the flat sheet.
            float papGrain = soilValueNoise(float2(ang * 1.3 + 4.0, yW * 0.4)) - 0.5;
            float3 paper = float3(0.96, 0.97, 0.99) * (1.0 + 0.10 * papGrain);
            albedo *= mix(paper, float3(0.50, 0.45, 0.42), peel * 0.65);
            // DARK, high-contrast lenticel — drive nearly to black so the dash reads
            // even against the bright paper (full-weight mix so hero dashes hold).
            albedo = mix(albedo, float3(0.035, 0.032, 0.032), lent * 0.98);
            // Paper-smooth roughness; peel edges lift the normal slightly.
            roughness = clamp(roughness * 0.85, 0.30, 0.70);
            n = normalize(n + float3(0,1,0) * (lent * 0.22) + tang * (peel - 0.5) * 0.14);
        } else if (species < 4.5) {
            // ── MAPLE — narrow BROKEN vertical shaggy strips, lifted/curling edges ─
            // ROUND-5 STRUCTURAL #2: the round-3 tuning read as a smooth tan pole at
            // hero distance — its strips were too wide + too low-contrast to survive,
            // so no foreground trunk read as maple (it looked like generic oak/birch
            // wood). The reviewer wants maple UNMISTAKABLY DISTINCT from oak: NARROW,
            // BROKEN vertical shaggy strips with strong tangential normal at the strip
            // borders (lifted/curling edges) and NO horizontal cracking — it must NOT
            // share the oak crack field. Build a vertical-strip field, BREAK each strip
            // vertically into shaggy segments (a separate vertical break mask), drive a
            // wide albedo split (dark damp furrow ↔ pale grey lifted strip), and lift
            // the curling edges HARD (strong tangential tilt at the strip-vertical
            // borders, away from the strip — the peeling-up shag of maple bark).
            // ROUND-6 STRUCTURAL #2 (maple side): the reviewer says maple's strips
            // don't read NARROWER / more BROKEN than oak. Make the difference
            // unmistakable: narrow the strips further (ang×18 ≈ 6–7 cm vs oak's plate
            // cells), break them harder vertically into short shaggy SEGMENTS, push the
            // albedo split MORE (pure-black furrow ↔ pale lifted strip), and lift the
            // strip BORDERS hard with tangential normal (the peeling-up shag of maple),
            // with ZERO horizontal cracking (no oak crack field touches this branch).
            float2 sc  = float2(ang * 18.0, yW * 0.40);        // narrower strips (~6–7 cm)
            float2 sc2 = float2(ang * 38.0, yW * 1.8);
            float f   = soilValueNoise(sc) * 0.70 + soilValueNoise(sc2) * 0.30;
            float fa  = soilValueNoise(sc + float2(e,0)) * 0.70 + soilValueNoise(sc2 + float2(e,0)) * 0.30;
            float fy  = soilValueNoise(sc + float2(0,e)) * 0.70 + soilValueNoise(sc2 + float2(0,e)) * 0.30;
            float fine = soilValueNoise(float2(ang * 50.0, yW * 14.0)) - 0.5;
            // Vertical BREAK: strips snap into short shaggy SEGMENTS (the maple tell vs
            // oak's stacked plate cells). Stronger break (higher vertical freq, wider
            // gap window) so the strips are visibly BROKEN, not continuous combed lines.
            float vbreak = soilValueNoise(float2(ang * 7.0, yW * 4.2));
            float segMask = smoothstep(0.34, 0.50, vbreak);    // 0 at a strip break gap
            // Sharper, NARROWER strip mask → the shaggy lift concentrates at the borders.
            float strip = smoothstep(0.44, 0.52, f) * segMask;
            float border = (1.0 - abs(smoothstep(0.44, 0.52, f) - 0.5) * 2.0) * segMask;
            // WIDER albedo split, pulled DARKER/greyer than oak so the two species never
            // read alike: near-BLACK damp furrow ↔ pale weathered grey lifted strip.
            albedo *= mix(0.18, 1.22, strip) * (0.80 + 0.32 * fine);
            float gA = (fa - f) / e, gY = (fy - f) / e;
            // CURLING-EDGE lift: very strong tangential tilt at the vertical strip
            // borders (the bark peeling up), plus an outward push at the segment break
            // so the shaggy ends catch the rake light.
            float segEdge = (1.0 - segMask);                   // 1 at a vertical break
            n = normalize(n + tang * (gA * 2.7 + border * 1.5 + fine * 0.60)
                            + float3(0,1,0) * (gY * 0.30)
                            + n * (border * 0.35 - segEdge * 0.22));
        } else if (species < 5.5) {
            // ── MOSSY FALLEN LOG — BARREL (damp moss velvet + length-aligned bark) ─
            // ROUND-4 MATERIAL-GATING FIX: the concentric end-grain RING field was
            // firing on the BARREL circumference as alternating pale/dark candy
            // stripes (the "striped wicker basket" read). Rings now live ONLY on the
            // cut-end cap (marker 6, the `species >= 5.5` branch below) — this branch
            // is the BARREL only: moss velvet on top, bark on the underside, with
            // bark RIDGES running ALONG the log's length (not around it).
            float3 ref = (abs(n.y) > 0.9) ? float3(1,0,0) : float3(0,1,0);
            float3 lt = normalize(cross(n, ref));
            float3 lb = cross(n, lt);
            float3 wp = in.worldPos;
            float mu = dot(wp, lt), mv = dot(wp, lb);
            // ── ROUND-6 STRUCTURAL #1 — BARREL MATERIAL REBUILT ──────────────────
            // DIAGNOSIS (why the previous branch rendered a smooth tan dowel even
            // though the marker-5 gate provably fired): the moss was gated to the
            // GEOMETRIC up component `n.y` via `topness = smoothstep(-0.20,0.50,n.y)`.
            // A fallen log is HORIZONTAL, and the hero/low camera sees its SIDE, where
            // n.y≈0 ⇒ topness≈0.29 ⇒ `surf = mix(bark, velvet, 0.29·0.85+0.15≈0.40)`
            // — i.e. the visible side was 60 % BARK. The moss velvet DID survive to the
            // G-buffer, but only on the thin n.y≈1 crest the camera barely grazes. The
            // tan dowel WAS the bark underside, shown on the side. Plus `bareBark`
            // (coverage<0.46) punched even more bark through.
            // REBUILD: dense moss velvet over the whole UPPER HEMISPHERE (top crest AND
            // both upper flanks the camera actually sees), bark only on the genuine
            // UNDERSIDE (n.y<0), smooth blend at the equator. The split is on `topness`
            // but with a much LOWER, wider ramp so the sides read mossy; bark
            // breakthrough is confined to the underside and a few small dry clumps.
            float topness = smoothstep(-0.55, 0.10, n.y);   // moss covers upper hemi + sides
            // Velvet mottle: COARSE clumps dominate (≈12–30 cm), one fine octave for
            // the close fuzz. Weighted toward the low freq so it survives distance.
            float m0 = soilValueNoise(float2(mu * 4.5,  mv * 4.5));   // ≈20 cm patches
            float m1 = soilValueNoise(float2(mu * 11.0, mv * 11.0));  // ≈9 cm
            float m2 = soilValueNoise(float2(mu * 40.0, mv * 40.0));  // close fuzz
            float mottle = m0 * 0.58 + m1 * 0.28 + m2 * 0.14;
            // HIGH-CONTRAST damp velvet: dark olive-green floor ↔ bright yellow-green
            // tip clumps. Floor lifted off black (a damp forest moss is dark olive).
            float3 mossDamp = float3(0.040, 0.072, 0.030);
            float3 mossTip  = float3(0.245, 0.300, 0.082);
            float3 velvet = mix(mossDamp, mossTip, smoothstep(0.34, 0.70, mottle));
            // Bark breakthrough now SPARSE on the moss: only small DRY clumps on the
            // upper surface (rare), but the underside is fully bark. Tightened so the
            // barrel reads UNMISTAKABLY mossy from the side, not 60 % bark.
            float cover = soilValueNoise(float2(mu * 2.4 + 13.0, mv * 2.4 + 5.0));
            float dryClump = smoothstep(0.30, 0.14, cover);  // rare small dry patch
            // LENGTH-ALIGNED bark ridges on the exposed bark: the bark coordinate
            // `mv` runs roughly along the bole, so high-freq in `mv`, low-freq in `mu`
            // makes ridges that run ALONG the log — not stripes that wrap it.
            float barkRidge = soilValueNoise(float2(mu * 2.2, mv * 16.0));
            float3 barkUnder = float3(0.21, 0.135, 0.090) * (0.6 + 0.7 * m1)
                             * (0.78 + 0.5 * barkRidge);   // ridges run lengthwise
            // Moss dominates the upper hemisphere; bark shows on the underside (low
            // topness) and a few small dry clumps on top. The blend is on `topness`
            // (smooth at the equator) with only a sparse dry-clump break on top.
            float3 mossSurf = mix(velvet, barkUnder, dryClump * 0.5 * topness);
            float3 surf = mix(barkUnder, mossSurf, topness);
            albedo = surf;
            // Bark coverage (for the normal/roughness) is high only where bark wins.
            float barkAmt = (1.0 - topness) + dryClump * 0.5 * topness;
            // Fine fuzz normal (where moss is) so velvet catches the rake light as a
            // soft fur. Coarse mottle bumps the silhouette. Bark grooves on underside.
            float ep = 0.05;
            float gU = (soilValueNoise(float2((mu+ep)*40.0, mv*40.0)) - m2);
            float gV = (soilValueNoise(float2(mu*40.0, (mv+ep)*40.0)) - m2);
            float gC0 = (soilValueNoise(float2((mu+ep)*3.2, mv*3.2)) - m0);   // coarse bump
            float gC1 = (soilValueNoise(float2(mu*3.2, (mv+ep)*3.2)) - m0);
            float bg = (soilValueNoise(float2(mu*2.2 + ep, mv*16.0)) - barkRidge);
            n = normalize(n + (lt * (gU + gC0 * 2.6 + bg * barkAmt * 1.2)
                               + lb * (gV + gC1 * 2.6)) * (0.85 * topness + 0.20 * barkAmt)
                            + float3(0,1,0) * (mottle - 0.5) * 0.20);
            // Velvet is matte (high roughness); damp bark a touch glossier.
            roughness = mix(0.97, 0.90, barkAmt);
        } else {
            // ── MOSSY FALLEN LOG — CUT END-GRAIN CAP (marker 6) ─────────────────
            // ROUND-4: rings fire ONLY here. This face is the sawn/snapped END of the
            // bole (addMossyLog stamps materialMark = 6 on the cut-end disc; no other
            // surface emits w ≈ 6). Build a polar radial coordinate centred on the
            // cap's pith and draw concentric growth rings as a luminance alternation —
            // so rings appear on the cut face and NEVER wrap the barrel circumference.
            // The cap's geometric normal is ≈ axial, so the radial frame is built in
            // the cap plane (perpendicular to n).
            float3 ref = (abs(n.y) > 0.9) ? float3(1,0,0) : float3(0,1,0);
            float3 ct = normalize(cross(n, ref));
            float3 cb = cross(n, ct);
            float3 wp = in.worldPos;
            // Radial distance from the local pith. The cap is small (≈0.3 m), so use a
            // fract-free world projection; the baked vertex gradient handles absolute
            // centring, this adds the crisp ring alternation on top.
            float cu = dot(wp, ct), cv = dot(wp, cb);
            // A coherent per-cap centre from the low-freq field so the rings are
            // roughly concentric without packing the true pith position.
            float2 pith = (soilValueNoise(float2(cu, cv)) - 0.5) * 0.05;
            float rr = length(float2(cu, cv) - pith);
            // ≈5 rings over the cap radius: alternate light sap / dark latewood.
            float ringPhase = rr * 22.0;
            float ring = 0.5 + 0.5 * cos(ringPhase * 2.0 * 3.14159265);
            // Heartwood pith → paler sapwood radial base (the baked vertex colour
            // already trends this way; reinforce + add the ring alternation).
            // Warm wood, not black: the cut face is sap-bright with a darker (but not
            // black) latewood line so the END-GRAIN rings read as warm wood, not a
            // hole punched in the log.
            float3 latewood = float3(0.30, 0.20, 0.115);   // darker growth-ring line
            float3 earlywood = max(albedo, float3(0.34, 0.24, 0.15)) * 1.18;  // bright sap
            albedo = mix(latewood, earlywood, smoothstep(0.30, 0.70, ring));
            // Faint radial checks / rays + slight recess at each dark ring line.
            float ray = soilValueNoise(float2(atan2(cv, cu) * 6.0, rr * 3.0));
            albedo *= 0.88 + 0.24 * ray;
            roughness = mix(0.62, 0.85, ring);             // latewood a touch glossier
            n = normalize(n - n * (1.0 - ring) * 0.06);    // dark rings recess slightly
        }
#endif
    }
}
