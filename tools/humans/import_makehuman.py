#!/usr/bin/env python3
"""Bake MakeHuman CC0 data into the VisualizerHumans binary dataset.

Input (mh-data/, staged by fetch-makehuman-data.sh):
  base.obj                  19k-vertex quad mesh; groups: body / helper-* / joint-*
  rigs/default.mhskel       163 bones; joint positions = centroids of mesh vertex sets
  rigs/default_weights.mhw  per-bone sparse [vertexIndex, weight]
  macrodetails/**.target    sparse per-vertex deltas for the macro grid
                            (ethnicity x gender x age, universal muscle/weight,
                             height/, proportions/)

Output: HumanDataset.bin (format MHB1, little-endian, documented in
VisualizerHumans/Sources/VisualizerHumans/HumanDataset.swift which must stay in sync).

Units: MakeHuman works in decimeters, Y-up; everything here is scaled to meters.
Only the "body" OBJ group becomes render geometry; helper-* groups are dropped;
joint-* cubes are dropped too (the .mhskel joints dict references raw vertex
indices directly, so the cubes' vertices stay in the orig-vertex table).
"""

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "mh-data"
OUT = ROOT.parent.parent / "VisualizerHumans" / "Sources" / "VisualizerHumans" / "Resources" / "HumanDataset.bin"

SCALE = 0.1  # dm -> m


def parse_obj(path):
    orig_positions = []
    uvs = []
    render_map = {}      # (v, vt) -> render index
    render_orig = []
    render_uv = []
    tris = []
    in_body = False

    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                orig_positions.append((float(x) * SCALE, float(y) * SCALE, float(z) * SCALE))
            elif line.startswith("vt "):
                parts = line.split()
                uvs.append((float(parts[1]), float(parts[2])))
            elif line.startswith("g "):
                in_body = line.strip() == "g body"
            elif line.startswith("f ") and in_body:
                corners = []
                for tok in line.split()[1:]:
                    vi, ti = (int(p) for p in tok.split("/")[:2])
                    key = (vi - 1, ti - 1)
                    if key not in render_map:
                        render_map[key] = len(render_orig)
                        render_orig.append(vi - 1)
                        render_uv.append(uvs[ti - 1])
                    corners.append(render_map[key])
                for a, b in [(1, 2), (2, 3)] if len(corners) == 4 else [(1, 2)]:
                    tris += [corners[0], corners[a], corners[b]]
    return orig_positions, render_orig, render_uv, tris


PRUNE = 2e-5  # meters; below visual relevance even stacked

def parse_target(path):
    entries = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            d = (float(parts[1]) * SCALE, float(parts[2]) * SCALE, float(parts[3]) * SCALE)
            if max(abs(c) for c in d) < PRUNE:
                continue
            entries.append((int(parts[0]), *d))
    return entries


def w_str(buf, s):
    b = s.encode()
    buf += struct.pack("<H", len(b)) + b


def main():
    orig, render_orig, render_uv, tris = parse_obj(DATA / "base.obj")
    ys = [orig[i][1] for i in set(render_orig)]
    print(f"orig verts {len(orig)}, render verts {len(render_orig)}, tris {len(tris)//3}, "
          f"body height {max(ys)-min(ys):.3f} m")

    targets = sorted((DATA / "macrodetails").rglob("*.target"))
    morphs = [(str(t.relative_to(DATA / 'macrodetails')).removesuffix(".target"), parse_target(t))
              for t in targets]

    # Detail groups (named "<dir>/<stem>"): archetype silhouettes, belly, glutes,
    # hips, and the female cup/firmness macro grid. The universal weight axis
    # alone is subtle (~3cm); these carry the real body-shape variety.
    DETAIL = {
        "bodyshapes": lambda n: True,
        "stomach": lambda n: n.startswith(("stomach-pregnant", "stomach-tone")),
        "buttocks": lambda n: n.startswith("buttocks-volume"),
        "hip": lambda n: n.startswith("hip-scale-horiz"),
        "breast": lambda n: True,
    }
    for d, keep in DETAIL.items():
        for t in sorted((DATA / d).glob("*.target")):
            stem = t.stem
            if keep(stem):
                morphs.append((f"{d}/{stem}", parse_target(t)))
    print(f"morphs: {len(morphs)}")

    skel = json.load(open(DATA / "rigs" / "default.mhskel"))
    weights = json.load(open(DATA / "rigs" / "default_weights.mhw"))["weights"]

    joint_names = sorted(skel["joints"].keys())
    joint_index = {n: i for i, n in enumerate(joint_names)}

    bone_names = sorted(skel["bones"].keys())
    bone_index = {n: i for i, n in enumerate(bone_names)}
    planes = skel.get("planes", {})

    buf = bytearray()
    buf += b"MHB1"
    buf += struct.pack("<6I", 1, len(orig), len(render_orig), len(tris), len(morphs), len(joint_names))
    buf += struct.pack("<I", len(bone_names))

    for p in orig:
        buf += struct.pack("<3f", *p)
    for i in render_orig:
        buf += struct.pack("<I", i)
    for uv in render_uv:
        buf += struct.pack("<2f", *uv)
    for i in tris:
        buf += struct.pack("<I", i)

    for name, entries in morphs:
        w_str(buf, name)
        buf += struct.pack("<I", len(entries))
        for e in entries:
            buf += struct.pack("<I3e", *e)   # f16 deltas: 0.2m max magnitude, ample precision

    for name in joint_names:
        w_str(buf, name)
        verts = skel["joints"][name]
        buf += struct.pack("<I", len(verts))
        for v in verts:
            buf += struct.pack("<I", v)

    for name in bone_names:
        bone = skel["bones"][name]
        w_str(buf, name)
        parent = bone.get("parent")
        buf += struct.pack("<i", bone_index[parent] if parent else -1)
        buf += struct.pack("<2I", joint_index[bone["head"]], joint_index[bone["tail"]])
        plane_name = bone.get("rotation_plane")
        plane = planes.get(plane_name) if isinstance(plane_name, str) else None
        if plane and len(plane) == 3 and all(j in joint_index for j in plane):
            buf += struct.pack("<3i", *(joint_index[j] for j in plane))
        else:
            buf += struct.pack("<3i", -1, -1, -1)
        bw = weights.get(name, [])
        buf += struct.pack("<I", len(bw))
        for vi, w in bw:
            buf += struct.pack("<If", vi, w)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(buf)
    print(f"wrote {OUT} ({len(buf)/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
