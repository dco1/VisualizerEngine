# Modelling kit (`VisualizerMaterials/Geometry`)

Blender-parity tools for what a hat / prop builder uses: topology editing, push/pull, robust booleans, SDF modelling,
remesh/decimate, modifiers, UVs, sculpting, a procedural graph and import/export. Everything is `Mesh3`-in, `Mesh3`-out
unless it says otherwise. Every file's header states what it guarantees and what it doesn't. Pure Swift 6 + Metal, no
third-party code.

## What's there

| Area | Files (`Geometry/…`) | Core types |
|---|---|---|
| Half-edge mesh (n-gons, adjacency, selection) | `HalfEdge/HalfEdgeMesh*.swift` | `HalfEdgeMesh`, `HESelection` |
| Topology edit tools | `HalfEdge/HalfEdge+Extrude/Inset/LoopCut/Bevel/Bridge/Dissolve/Knife/Smooth/Spin.swift` | methods on `HalfEdgeMesh` |
| Robust booleans | `Mesh3+Boolean*.swift` | `Mesh3.boolean(_:_:options:)`, `BooleanOptions`, `BooleanStats` |
| SDF modelling + surface nets | `SDF*.swift` | `SDF`, `SDFMeshOptions` |
| Remesh / decimate / weld / fill holes | `Mesh3+Remesh/Decimate/Weld/FillHoles*.swift` | `isotropicRemeshed`, `decimated`, `welded`, `filledHoles` |
| Modifier stack | `Modifiers/*.swift` | `Modifier`, `ModifierStack`, `SolidifyModifier`, `DisplaceModifier`, `SimpleDeformModifier`, `LatticeDeformModifier`, `ShrinkwrapModifier`, `MirrorModifier`, `ArrayModifier`, `BevelModifier`, `SubdivideModifier`, `BooleanModifier` |
| UV tools | `UV*.swift` | `UVProjection`, `UVTopology`, `UVSeams`, `UVUnwrap`, `UVPacker`, `UVReport` |
| Sculpt brushes | `Sculpt*.swift` | `SculptVolume`, `SculptBrush`, `SculptStroke` |
| Procedural graph | `Graph/*.swift` | `ModelGraph`, `GraphBuilder`, `NodeKind` |
| Import / export | `MeshIO*.swift` | `MeshIO` (OBJ, PLY, STL read/write) |
| Pre-existing (unchanged) | `Mesh3+Build/Sweep/Revolve/Subdivide/Bevel/Pattern/CSG.swift` | `sweep`, `loft`, `tube`, `revolve`, `subdivided`, `beveled`, `mirrored`, `arrayed`, BSP CSG |
| Demos | `Demo/GuitarCarve*.swift`, `Tests/…/ModellingDemoTests.swift` | `GuitarCarve.steps()` |

## API cheat sheet

```swift
// ---- half-edge: any n-gon topology edit, then back to triangles
var he = HalfEdgeMesh.box(size: Vec3(0.06, 0.04, 0.02))          // or HalfEdgeMesh(mesh3)  (weld by position)
let top = he.faces.first { he.faceNormal($0).z > 0.9 }!
let stem = he.extrude(faces: he.selection(.face, [top]), distance: 0.03)!   // faces / edges / vertices; individual: true
let cap  = he.inset(stem, thickness: 0.006, depth: 0)!                      // region or individual
he.pushPull(cap, distance: 0.008)                                           // move faces along normals, no new topology
he.bevel(edges, width: 0.002); he.loopCut(ringFrom: e, cuts: 2); he.bridge(sel); he.knife(planeOrigin:normal:)
he.spin(sel, axisOrigin:axisDirection:angle:steps:); he.smoothVertices(sel); he.dissolve(sel); he.merge(sel)
let m = he.toMesh3()

// ---- robust boolean: closed, welded, outward-wound output (MeshAudit.boundaryEdges == 0)
let r = a.booleanDifference(b)        // booleanUnion / booleanIntersection; a.slice(point:normal:) -> (above, below)
let full = a.boolean(.union, b, options: opts)   // .mesh + .stats; opts.weld = .keepCreases keeps hard-edge normals

// ---- SDF: primitives, transforms, smooth booleans, offset/shell, carve, surface nets (GPU if a device exists)
let s = SDF.roundedBox(size: v, radius: r).union(SDF.capsule(from: a, to: b, radius: r), smooth: 0.02)
let engraved = SDF.mesh(m).carved(tool: tool, depth: 0.0015, falloff: 0.001)
let mesh = engraved.mesh(lo: lo, hi: hi, cell: 0.0025, options: .init(uvs: .boxProjected(metersPerUnit: 1)))

// ---- remesh / decimate / weld / holes
m.decimated(targetTriangles: 8000)  m.isotropicRemeshed(targetEdgeLength: 0.004)  m.welded(distance: 1e-5)  m.filledHoles()

// ---- modifiers (re-evaluable, ordered)
var stack = ModifierStack()
_ = stack.add(SimpleDeformModifier(.bend(angle: 0.05), origin: o, axis: Vec3(1,0,0), lower: 0, upper: 0.5))
_ = stack.add(DisplaceModifier.noise(strength: 0.0008, frequency: 90))
let out = stack.evaluate(mesh)         // SolidifyModifier, LatticeDeform, Shrinkwrap, Mirror, Array, Bevel, Subdivide, Boolean too

// ---- UVs
let box = mesh.projectedUVs(UVProjection(.box, scale: Vec2(4, 4))).packedUVs()          // projection + pack
let lscm = mesh.unwrappedUVs(seams: UVTopology(mesh).seams(angleDegrees: 60))             // seams -> LSCM per island -> pack
lscm.report.texelDensity(textureSize: 2048)

// ---- sculpt (voxel/SDF volume, Metal brushes, undo)
let vol = SculptVolume(mesh: m, cell: 0.005)             // or (sdf:lo:hi:cell:)
let st = vol.beginStroke(SculptBrush(.draw, radius: 0.08, strength: 0.16, subtract: true))
st.stamp(position: p, normal: n, pressure: 1); st.end(); vol.undo(); let sculpted = vol.mesh()

// ---- procedural graph: Codable DAG, deterministic, cached per node
let g = GraphBuilder(); let k = g.importMesh(knob).arrayed(count: 2, offset: Vec3(0.05, 0, 0))
let result = try g.build(output: k).evaluateMesh()       // JSON round-trips: graph.jsonData()

// ---- import / export
let data = MeshIO.write([mesh], format: .obj)            // .ply .stl (binary or ascii via options)
let back = try MeshIO.read(url: url)                      // .meshes, names, groups
```

## Blender → engine map

| Blender | Engine |
|---|---|
| Extrude faces / edges / vertices | `HalfEdgeMesh.extrude(faces:/edges:/vertices:)` |
| Inset faces (region / individual) | `HalfEdgeMesh.inset(_:thickness:depth:individual:)` |
| Push/Pull (Alt+E → along normals; Shrink/Fatten) | `HalfEdgeMesh.pushPull` |
| Loop cut + slide, edge ring select | `loopCut(ringFrom:cuts:slide:)`, `slideEdges`, `HalfEdgeMesh+Loops/Selection` |
| Bevel (edges) | `HalfEdgeMesh.bevel`, `Mesh3.beveled` (crease-driven), `BevelModifier` |
| Bridge edge loops | `HalfEdgeMesh.bridge` |
| Fill / Grid fill hole | `HalfEdgeMesh.fillHoles/deleteAndFill`, `Mesh3.filledHoles` |
| Knife / Bisect | `HalfEdgeMesh.knife`, `Mesh3.slice` |
| Spin | `HalfEdgeMesh.spin` |
| Smooth vertices | `HalfEdgeMesh.smoothVertices` |
| Dissolve, Merge, Merge by distance | `dissolve`, `merge`, `mergeByDistance`; `Mesh3.welded(distance:)` |
| Boolean modifier / tool (Exact) | `Mesh3.booleanUnion/Difference/Intersection`, `BooleanModifier` (takes an engine closure) |
| Remesh (voxel) | `SDF.mesh` / `SculptVolume.mesh` (voxel); `Mesh3.isotropicRemeshed` (quad-ish isotropic) |
| Decimate (Collapse) | `Mesh3.decimated(targetTriangles:/ratio:)` |
| Solidify | `SolidifyModifier` (thickness, offset, rim) |
| Displace | `DisplaceModifier` (closure / `.noise`) |
| Simple Deform (bend, twist, taper, stretch) | `SimpleDeformModifier` |
| Lattice | `LatticeDeformModifier` |
| Shrinkwrap | `ShrinkwrapModifier` (nearest, project, offset) |
| Mirror / Array | `MirrorModifier`, `ArrayModifier`, `Mesh3.mirrored/arrayed/radialArray` |
| Subdivision Surface | `SubdivideModifier`, `Mesh3.subdivided` (Loop) |
| Modifier stack | `ModifierStack` |
| UV: Project from view / Cube / Cylinder / Sphere | `UVProjection(.planar/.box/.cylinder/.sphere)` |
| Mark seam, Seams from angle | `UVTopology.seams(edges:)` / `seams(angleDegrees:)` |
| Unwrap (Angle based / Conformal) | `Mesh3.unwrappedUVs` (LSCM) |
| Pack Islands | `UVPacker`, `packedUVs` |
| Texel density | `UVReport.texelDensity` |
| Sculpt: Draw, Clay, Grab, Smooth, Inflate, Crease, Flatten, Pinch, mask, symmetry | `SculptBrush(.draw/.clay/.grab/…)`, `SculptVolume.setMask`, `SculptSymmetry` |
| Dyntopo / Multires | `SculptVolume.refine()` (global refinement by brush size) |
| Sculpt undo | `SculptVolume.undo/redo` (per-block snapshots) |
| SDF-style booleans / smooth union (metaballs, Voxel Remesher) | `SDF.union/subtracting/intersecting(smooth:)`, `offset`, `shell`, `carved` |
| Geometry Nodes (lite) | `ModelGraph` / `GraphBuilder` (primitive, transform, boolean, SDF, modifier, instance-on-points, join) |
| Import / Export OBJ, PLY, STL | `MeshIO.read/write` |
| Curve → mesh (sweep), Loft, Spin/Screw | `Mesh3.sweep/loft/tube/revolve` |

## GPU vs CPU, per tool (measured; Apple GPU, see each ticket item)

| Tool | GPU (Metal compute) | CPU twin / stays CPU | Measured |
|---|---|---|---|
| Half-edge bulk work (normals, areas, edge table, Mesh3 import weld) | yes (`HalfEdgeMesh+GPU*`) | topology pointer-chasing, edits | 476k-face import 713 → 77 ms |
| Half-edge push/pull offsets | yes above `HEEdit.gpuThreshold` verts | below it | n/a |
| Robust boolean | broadphase (grid/sort), narrow-phase float filter, winding, assembly | exact predicates for uncertain pairs, intersection points (double), retriangulation | 2×51.6k-tri union 259 ms CPU → 31–39 ms; 2×201k 1150 → 70 ms; shaders ~52 ms first use |
| SDF eval + surface nets | grid, topology (prefix-scan quad emit), vertex refine | closures (`SDF.custom`, mesh-SDF) CPU | 131k-tri mesh 1665 ms CPU → 12 ms all-GPU |
| Decimate | quadric build, collapse cost, parallel independent-set rounds | BVH build + final assembly | 82k → 8k tris 5.6–6.4 s → 0.17–0.29 s |
| Remesh | smoothing + projection (BVH closest point) | topology ops | 1.8 s → 0.2 s |
| Weld | hash/sort | — | 0.20 s → 0.03 s |
| Modifiers: Displace (incl. noise), SimpleDeform, Lattice, Shrinkwrap, Solidify | yes | closure Displace; Mirror/Array/Bevel/Subdivide/Boolean wrappers | 200k-vertex 5-modifier stack 814 ms → 18 ms warm |
| UV projections, distortion report, pack scoring, LSCM solve (CG) | yes | island discovery, seams, topology | 200k-tri sphere projection 124 → 2.8 ms; LSCM 400 CG its 2450 → 181 ms (wins above ~30–50k tris) |
| Sculpt brushes | all (Metal) | CPU twin `SculptKernel` | 0.10 ms a stamp |
| MeshIO binary STL/PLY encode+decode, weld+smooth | yes | text formats (OBJ, ascii) | 1M-tri STL weld+smooth 1102 → 29 ms; PLY read 250 → 16 ms |
| Graph evaluation | via the tools above | DAG walk | — |

Backends: `MeshBackend` (`.cpu/.gpu/.auto`; `.auto` goes GPU from ~20k triangles). Without a Metal device (CI) every GPU
test `XCTSkip`s and the CPU path runs.

## Known gaps

- **Booleans**: inputs must be closed, manifold and outward-wound; operands touching only along an edge/vertex give a
  non-manifold result; intersection points and retriangulation are CPU; thin (< ~1e-9 relative) slivers fail. Near
  coplanar overlaps of a few tenths of a millimetre (a plate 0.2 mm proud of a face) can leave open edges: make
  attachments overlap by ≥ 1 mm or flush-coplanar (see `GuitarCarve+Parts`). A `ModelGraph` `boolean` chain over
  float-identical inputs gave different closure than the same chain on `Mesh3` once in the demo (3 open edges): audit
  graph output with `openEdges` too.
- **SDF**: no dual contouring; sharp edges round by about one cell; features thinner than ~2 cells vanish; closure
  based fields (`custom`) are CPU-only.
- **Half-edge**: pointer-chasing topology edits stay CPU; `bevel` on a selection of triangulated, noisy meshes (e.g. a
  surface-nets mesh) can fail and returns nil: merge coplanar faces first.
- **Remesh** drops UVs; **decimate** can stop short of target and GPU error is ~3× the CPU's (not bit-reproducible).
- **Modifiers**: Solidify has no self-intersection guard; Boolean modifier takes an engine closure; closure Displace,
  Mirror/Array/Bevel/Subdivide/Boolean are CPU.
- **UV**: LSCM can fold strongly curved islands; shelf packing is 70–85% fill; island discovery is CPU.
- **Sculpt**: dense grid (memory ∝ volume); refinement is global rather than block-local; the field is not
  re-distanced after strokes.
- **MeshIO**: no MTL read; `Mesh3` has no colour channel so PLY colours ride beside it; FBX is out of scope (USD stays
  with the app's USDZ path).
- Out of scope: FBX, full geometry nodes, rigging/animation, NURBS, a GUI.

## Demo models

- `ModellingDemoTests.testModelADrawerPullWithTheWholeKit`: a drawer pull through half-edge extrude/inset/push-pull,
  a robust-boolean bore, an SDF carve (groove), a taper + grain modifier stack, LSCM UV unwrap and an OBJ export to
  `/private/tmp/dhf-debug/modelling-kit/demo.obj`; every stage is checked closed.
- `GuitarCarve.steps()`: eight steps from a 1.06 × 0.36 × 0.05 m block to a double-cutaway electric guitar with neck,
  headstock, pickups, pickguard, bridge, knobs, frets, strings and tuners; `GuitarCarve.write(to:)` writes every step's OBJ and
  `steps.json` (default folder `/private/tmp/dhf-debug/guitar-carve`). The test asserts non-empty, closed, and that
  the four carving steps strictly lose volume.

## Running the tests

```sh
cd VisualizerMaterials
swift test                                              # everything (~125 s, 289 tests, 1 skip: TileRealismPreview needs TILE_OUT)
swift test --filter ModellingDemoTests                  # one object, the whole kit
swift test --filter GuitarCarveTests                    # the guitar (~6 s)
swift test --filter "MeshBoolean|SDF|HalfEdge|Modifier|UV|MeshIO|Sculpt|ModelGraph"   # per area
```
Outputs that can be audited are written under `/private/tmp/dhf-debug/…` (the app's Stop hook posts them).
