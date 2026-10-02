# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Make a reconstructed mesh metric by stating its size, instead of fitting it.

The reconstructed object is in normalized units; the pipeline needs metres.
tools/estimate_scale_video.py recovers metres by fitting the mesh to a depth
map, which is only as good as that depth map. When the person running the
reconstruction knows the object -- a saucepan is about 35 cm handle to rim --
that number beats any fit, and this applies it directly: the mesh's longest
axis becomes --size metres.

The whole mesh directory is copied and only the vertex lines of the OBJ are
rewritten, so the .mtl and the texture keep their names and references and
nothing about the appearance changes. Output layout matches the fit's, so the
rest of the pipeline cannot tell the difference.

Usage:
    python prep/scale_mesh_to_size.py --mesh work/<seq>/meshes-ego/<name>_rgba/<name>_align.obj \\
        --size 0.35 --out_root work/<seq>/meshes-ego-metric
"""
import argparse
import os
import os.path as osp
import shutil
import sys

import numpy as np

sys.path.append(os.getcwd())


def parse_args():
    """Parse the mesh, the metric size and the output root."""
    parser = argparse.ArgumentParser(
        description="Scale a reconstructed mesh so its longest axis is a stated length in metres")
    parser.add_argument("--mesh", required=True, help="<name>_align.obj in its own directory")
    parser.add_argument("--size", type=float, required=True,
                        help="the object's longest dimension in metres")
    parser.add_argument("--out_root", required=True,
                        help="root to write <dir>/<name>_align.obj under, e.g. <meshes root>-metric")
    return parser.parse_args()


def obj_vertices(path):
    """Return the (N, 3) vertex positions of an OBJ, in file order."""
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.array(verts)


def scale_obj_vertices(src, dst, factor):
    """Copy an OBJ, multiplying every vertex position by factor; all else verbatim."""
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith("v "):
                parts = line.split()
                xyz = [float(x) * factor for x in parts[1:4]]
                rest = parts[4:]
                fout.write("v " + " ".join(f"{v:.6f}" for v in xyz)
                           + ("" if not rest else " " + " ".join(rest)) + "\n")
            else:
                fout.write(line)


def main():
    """Scale the mesh to the stated size and write it beside its texture."""
    args = parse_args()
    src_dir = osp.dirname(osp.abspath(args.mesh))
    name = osp.basename(args.mesh)
    verts = obj_vertices(args.mesh)
    if len(verts) == 0:
        raise SystemExit(f"ERROR: no vertices in {args.mesh}")
    extents = verts.max(axis=0) - verts.min(axis=0)
    longest = float(extents.max())
    factor = args.size / longest

    dst_dir = osp.join(args.out_root, osp.basename(src_dir))
    if osp.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    dst = osp.join(dst_dir, name)
    scale_obj_vertices(args.mesh, dst, factor)
    # The fit's debug outputs are not reproduced; its scale json is, since
    # tools/estimate_scale_video.py treats one as "already done".
    import json
    seq = name.split("_")[0] if "_" not in name else "_".join(name.split("_")[:-2])
    with open(osp.join(args.out_root, f"{seq}_scale.json"), "w") as f:
        json.dump({"scale": factor, "method": "stated size", "size_m": args.size}, f)

    print(f"mesh extents {np.round(extents, 3)} units -> x{factor:.4f} -> "
          f"{np.round(extents * factor, 3)} m (longest {args.size} m)")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
