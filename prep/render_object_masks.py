# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Replace a camera's object masks with the silhouette of a tracked mesh.

When the object is tracked in the ego view (scripts/slurm_ego_object.sh), the
pipeline camera's own object masks are the one place the exo view still
speaks for the object: the depth injection writes into them, CoCoNet reads
them as its object input, and the optimizer's silhouette term pulls the mesh
toward them. If SAM3 put that mask on a spoon, every stage after the ego
track drags the pot onto the spoon. So the mask is replaced with what the ego
track says the object's silhouette in this camera IS: the metric mesh, posed
by the carried-over poses, projected through the camera's rectified pinhole.

Only the object masks change. The person masks are the exo camera's business
and stay. Frames without a pose get an empty object mask, which every reader
treats as "object not visible". The original file is kept beside the new one
as <name>.exo.h5, once.

Rasterization is the union of the mesh's projected triangles -- a silhouette
needs no depth test -- drawn with cv2.fillPoly, which handles a few thousand
triangles per frame in milliseconds.

Usage:
    python prep/render_object_masks.py --fp_pkl work/<seq>/fp/<seq>_all.pkl \\
        --mesh work/<seq>/meshes-metric/<...>_align.obj \\
        --camera_pkl work/<seq>/rect/<seq>.0.color.pkl \\
        --masks_h5 work/<seq>/rect/<seq>_masks_k0.h5
"""
import argparse
import os
import os.path as osp
import re
import shutil
import sys

import cv2
import h5py
import joblib
import numpy as np

sys.path.append(os.getcwd())


def parse_args():
    """Parse the poses, the mesh, the camera and the mask file to rewrite."""
    parser = argparse.ArgumentParser(
        description="Overwrite a camera's object masks with a tracked mesh's silhouette")
    parser.add_argument("--fp_pkl", required=True,
                        help="FP-format pickle with poses in THIS camera (prep/ego_poses_to_cam.py)")
    parser.add_argument("--mesh", required=True, help="the metric mesh those poses move")
    parser.add_argument("--camera_pkl", required=True,
                        help="<seq>.<kid>.color.pkl with fx fy cx cy H W (prep/rectify_fisheye.py)")
    parser.add_argument("--masks_h5", required=True,
                        help="<seq>_masks_k<kid>.h5 to rewrite in place (a .exo.h5 copy is kept)")
    parser.add_argument("--kid", type=int, default=0)
    parser.add_argument("--dilate", type=int, default=0,
                        help="grow the silhouette by this many px (default: 0)")
    return parser.parse_args()


def frame_key_index(key):
    """Integer frame of an FP frame key such as '000180'."""
    m = re.search(r"\d+", str(key))
    if not m:
        raise SystemExit(f"ERROR: cannot read a frame index from {key!r}")
    return int(m.group(0))


def silhouette(verts_cam, faces, K, H, W):
    """Rasterize the union of projected triangles as a boolean (H, W) mask.

    Triangles with any vertex at or behind the camera are dropped rather than
    projected through the singularity.
    """
    z = verts_cam[:, 2]
    ok = z > 1e-6
    px = np.full((len(verts_cam), 2), np.nan)
    px[ok, 0] = K[0, 0] * verts_cam[ok, 0] / z[ok] + K[0, 2]
    px[ok, 1] = K[1, 1] * verts_cam[ok, 1] / z[ok] + K[1, 2]
    tri_ok = ok[faces].all(axis=1)
    tris = px[faces[tri_ok]]
    mask = np.zeros((H, W), np.uint8)
    if len(tris) == 0:
        return mask.astype(bool)
    # Keep triangles that touch the image at all; fillPoly clips the rest.
    inside = ((tris[:, :, 0] > -W) & (tris[:, :, 0] < 2 * W)
              & (tris[:, :, 1] > -H) & (tris[:, :, 1] < 2 * H)).all(axis=1)
    polys = [np.round(t).astype(np.int32) for t in tris[inside]]
    if polys:
        cv2.fillPoly(mask, polys, 1)
    return mask.astype(bool)


def main():
    """Render every posed frame and rewrite the object masks."""
    import trimesh

    args = parse_args()
    data = joblib.load(args.fp_pkl)
    poses = np.asarray(data["fp_poses"], dtype=np.float64)[:, 0]
    frames = [frame_key_index(k) for k in data["frames"]]
    by_frame = {f: p for f, p in zip(frames, poses) if np.isfinite(p).all()}

    cam = joblib.load(args.camera_pkl)
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    H, W = int(cam["H"]), int(cam["W"])

    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts_h = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)

    backup = args.masks_h5.replace(".h5", ".exo.h5")
    if not osp.isfile(backup):
        shutil.copy2(args.masks_h5, backup)
        print(f"kept the exo object masks as {backup}")

    kernel = None
    if args.dilate > 0:
        kernel = np.ones((2 * args.dilate + 1, 2 * args.dilate + 1), np.uint8)

    suffix = f"-k{args.kid}.obj_rend_mask.png"
    n_written = n_empty = 0
    areas = []
    with h5py.File(args.masks_h5, "r+") as f:
        groups = [g for g in f if isinstance(f[g], h5py.Group)]
        grp = f[groups[0]] if groups else f
        keys = [k for k in grp if k.endswith(suffix)]
        for key in keys:
            idx = int(key[: -len(suffix)])
            if idx in by_frame:
                cam_verts = (by_frame[idx] @ verts_h.T).T[:, :3]
                mask = silhouette(cam_verts, faces, K, H, W)
                if kernel is not None:
                    mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
                n_written += 1
                areas.append(int(mask.sum()))
            else:
                mask = np.zeros((H, W), bool)
                n_empty += 1
            old = grp[key]
            if old.shape != mask.shape:
                raise SystemExit(f"ERROR: mask {key} is {old.shape}, camera says {(H, W)}")
            old[...] = mask
    print(f"rewrote {n_written} object masks from the tracked mesh, "
          f"{n_empty} frames without a pose left empty, in {args.masks_h5}")
    if areas:
        print(f"  silhouette area: median {int(np.median(areas))} px, "
              f"min {min(areas)}, max {max(areas)} (image {W}x{H})")
        if min(areas) == 0:
            print("  WARNING: some posed frames project to nothing -- the object is "
                  "outside this camera's view there, or the pose is wrong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
