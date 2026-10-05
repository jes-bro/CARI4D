# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""The human from the multi-view fit and the object from the ego track, as one bundle.

No CoCoNet, no optimizer. On the pot clip neither had evidence to add: the
cam01 object mask they read is the ego track drawn again, CoCoNet predicted no
contact so the optimizer held the object still, and both refine the pot from
exo pixels that do not show it well. What IS evidence -- the body fitted to
triangulated joints, the pot tracked in the glasses' view and carried into
cam01 -- is here written straight into the bundle shape every downstream
tool reads (tools/viz_pred.py, tools/bundle_to_interact.py):

    pr['smpl_pose']  (T, 156)   SMPL-H pose from <nlf>-opt/<seq>_params.pkl
    pr['smpl_t']     (T, 3)
    pr['betas']      (T, 10)
    pr['pose_abs']   (T, 4, 4)  the object's pose in the same camera, for the
                                mesh centred on its vertex mean (the renderer
                                re-centres the metric mesh that way, and the
                                optimizer's bundles use the same convention)

Both inputs are in the pipeline camera: the fit works in the clip's camera,
prep/ego_poses_to_cam.py carries the ego poses into it. The object pose per
frame is required for every frame of the fit; a frame the track does not
cover is an error here, not a hole in the output.

Usage:
    python prep/export_direct_bundle.py --params work/<seq>/nlf-opt/<seq>_params.pkl \\
        --fp_pkl work/<seq>/fp/<seq>_all.pkl --mesh work/<seq>/meshes-metric/<...>_align.obj \\
        --out output/direct-hy3d/<seq>.pth
"""
import argparse
import os
import os.path as osp
import re
import sys

import joblib
import numpy as np
import torch

sys.path.append(os.getcwd())


def parse_args():
    """Parse the human fit, the object track, the mesh and the output path."""
    parser = argparse.ArgumentParser(
        description="Bundle the SMPL-H fit and the ego object track without CoCoNet or the optimizer")
    parser.add_argument("--params", required=True, help="<nlf>-opt/<seq>_params.pkl from prep/fit_smplh_global.py")
    parser.add_argument("--fp_pkl", required=True,
                        help="<seq>_all.pkl with poses in the pipeline camera (prep/ego_poses_to_cam.py)")
    parser.add_argument("--mesh", required=True, help="the metric _align.obj those poses move")
    parser.add_argument("--out", required=True, help="bundle .pth to write")
    return parser.parse_args()


def frame_index(key):
    """The integer frame behind a frame key ('000180', 't0180.000', '<seq>/000180')."""
    m = re.search(r"\d+", str(key).split("/")[-1])
    if not m:
        raise SystemExit(f"ERROR: cannot read a frame index from key {key!r}")
    return int(m.group(0))


def main():
    """Join the two inputs frame by frame and write the bundle."""
    import trimesh

    args = parse_args()
    seq = osp.basename(args.params).replace("_params.pkl", "")
    params = joblib.load(args.params)
    poses = np.asarray(params["poses"])[:, 0]      # (T, 156)
    trans = np.asarray(params["transls"])[:, 0]    # (T, 3)
    betas = np.asarray(params["betas"])[:, 0]      # (T, 10)
    T = len(poses)
    fit_frames = [frame_index(k) for k in params["frames"]] if "frames" in params else list(range(T))

    fp = joblib.load(args.fp_pkl)
    fp_poses = np.asarray(fp["fp_poses"], dtype=np.float64)[:, 0]   # (N, 4, 4)
    by_frame = {frame_index(k): p for k, p in zip(fp["frames"], fp_poses) if np.isfinite(p).all()}
    missing = [f for f in fit_frames if f not in by_frame]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} of {T} fitted frames have no object pose "
                         f"(first: {missing[0]}); the track must cover the clip")

    # The renderer centres the mesh on its vertex mean and applies pose_abs to
    # that; the track's poses are for the mesh as stored. x = R v + t with
    # v = v_c + c  ->  R v_c + (R c + t): the same rotation, the translation
    # moved by R c.
    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    centre = np.append(np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0), 1.0)
    pose_abs = np.stack([by_frame[f] for f in fit_frames])
    pose_abs[:, :3, 3] = np.einsum("tij,j->ti", pose_abs, centre)[:, :3]

    pr = {
        "smpl_pose": torch.from_numpy(poses).float(),
        "smpl_t": torch.from_numpy(trans).float(),
        "betas": torch.from_numpy(betas).float(),
        "pose_abs": torch.from_numpy(pose_abs).float(),
        "contact_logits": torch.zeros(T, 2),
        "frames": [f"{seq}/{f:06d}" for f in fit_frames],
        "gender": params.get("gender", "neutral"),
        "source": {"params": osp.abspath(args.params), "fp_pkl": osp.abspath(args.fp_pkl),
                   "mesh": osp.abspath(args.mesh), "direct": True},
    }
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    torch.save({"pr": pr, "in": dict(pr), "gt": {}}, args.out)
    dist = np.linalg.norm(pose_abs[:, :3, 3] - trans, axis=1)
    print(f"wrote {args.out}: {T} frames, gender {pr['gender']}, object from the ego track")
    print(f"  object centre to body root: {dist.min():.2f}-{dist.max():.2f} m (median {np.median(dist):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
