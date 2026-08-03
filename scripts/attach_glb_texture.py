# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Re-attach a Hunyuan3D GLB's texture to an already-converted OBJ.

Blender's OBJ exporter drops the GLB's packed image even with
path_mode='COPY', so prep/glb2obj.py yields a .mtl with no map_Kd and a mesh
that renders flat grey. run_hy3d_recon.py now fixes this inline, but a mesh
converted before that change needs repairing without re-running Hunyuan3D --
and run_hy3d_recon.py skips any sequence whose OBJ already exists, so simply
re-running it will not help.

Usage:
    python scripts/attach_glb_texture.py <mesh_dir>...

Each mesh_dir is a <seq>_<frame:03d>_rgba directory holding the .glb, the
.obj and the .mtl. Idempotent: a .mtl that already has map_Kd is left alone.
"""
import argparse
import os
import os.path as osp
import sys
from glob import glob

sys.path.append(os.getcwd())

from prep.run_hy3d_recon import attach_glb_texture


def parse_args():
    """Parse one or more mesh directories to repair."""
    parser = argparse.ArgumentParser(
        description="Attach a GLB's baked texture to an already-converted OBJ")
    parser.add_argument("mesh_dir", nargs="+",
                        help="<seq>_<frame:03d>_rgba directory, or a glob of them")
    return parser.parse_args()


def repair_one(mesh_dir):
    """Attach the texture for a single mesh directory.

    Locates the .glb and the *_align.obj by pattern rather than by name so it
    works for any sequence, and reports instead of raising when a directory is
    incomplete -- a batch run should skip a bad directory, not abort.

    Returns:
        True if a texture was attached, False otherwise.
    """
    glbs = sorted(glob(osp.join(mesh_dir, "*.glb")))
    objs = sorted(glob(osp.join(mesh_dir, "*_align.obj")))
    if not glbs:
        print(f"{mesh_dir}: no .glb, skipping")
        return False
    if not objs:
        print(f"{mesh_dir}: no *_align.obj, skipping")
        return False

    print(f"{mesh_dir}:")
    return attach_glb_texture(glbs[0], mesh_dir, osp.basename(objs[0])) is not None


def main():
    """Repair every directory given, reporting how many were changed."""
    args = parse_args()
    dirs = []
    for pattern in args.mesh_dir:
        dirs.extend(sorted(glob(pattern)) if any(c in pattern for c in "*?[") else [pattern])

    dirs = [d for d in dirs if osp.isdir(d)]
    if not dirs:
        raise SystemExit("ERROR: no existing directories matched")

    changed = sum(repair_one(d) for d in dirs)
    print(f"Attached textures for {changed} of {len(dirs)} directory(ies).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
