# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Stick-figure frames straight from a reconstruction .pth -- no renderer.

When a downstream render "doesn't look like the video", every numeric check
can pass while the argument goes in circles, because each side is checking
its own data against itself. This draws the reconstruction as a colored
skeleton (LEFT limbs red, RIGHT limbs blue, reconstructed object orange dot,
triangulated object orange X when an npz is given) in world coordinates,
one subplot per frame -- the same format as a converted-clip stick render,
so the two can be compared hop by hop against the source footage. Wherever
a motion visibly disappears between two adjacent hops is where the bug is.

CPU-only (SMPL forward + matplotlib); fine on a login node in newcari4d.

Usage:
    python prep/viz_skeleton_frames.py \\
        --pth output/opt/<exp>/<seq>.pth --calib <trajectory>/gopro_calibs.csv \\
        --cam cam04 --npz bball_xyz_all.npz --lo 0 --hi 19 --out skel.png
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib_smpl import get_smpl, pose72to156
from prep.compare_ball_stages import TOLERANT_PICKLE, frame_id, read_extrinsics

# SMPL 24-body kinematic tree (parent of joint i); hands drawn as wrist stubs.
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
           13, 14, 16, 17, 18, 19, 20, 21]
LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}
RIGHT = {2, 5, 8, 11, 14, 17, 19, 21, 23}


def parse_args():
    """Parse the bundle, calibration, optional npz, frame window and view."""
    parser = argparse.ArgumentParser(
        description="Draw per-frame colored skeletons from a recon .pth")
    parser.add_argument("--pth", required=True,
                        help="bundle with pr.smpl_pose/smpl_t/betas/pose_abs")
    parser.add_argument("--calib", required=True,
                        help="gopro_calibs.csv, to map camera frame to world")
    parser.add_argument("--cam", default="cam04", help="pipeline camera uid")
    parser.add_argument("--npz", default=None,
                        help="triangulate_object npz; draws the measured object as X")
    parser.add_argument("--gender", default="male", help="SMPL gender (default male)")
    parser.add_argument("--lo", type=int, default=0, help="first frame (default 0)")
    parser.add_argument("--hi", type=int, default=19, help="last frame (default 19)")
    parser.add_argument("--azim", type=float, default=-35.0,
                        help="view azimuth in degrees (default -35)")
    parser.add_argument("--elev", type=float, default=12.0,
                        help="view elevation in degrees (default 12)")
    parser.add_argument("--out", default="skeleton_frames.png",
                        help="output PNG (default skeleton_frames.png)")
    return parser.parse_args()


def limb_color(child):
    """Red for left-side limbs, blue for right, gray for the trunk."""
    if child in LEFT:
        return "red"
    if child in RIGHT:
        return "blue"
    return "gray"


def main():
    """Render the frame grid: skeleton, recon object, measured object."""
    args = parse_args()
    d = torch.load(args.pth, map_location="cpu", weights_only=False,
                   pickle_module=TOLERANT_PICKLE)
    pr = d["pr"]
    frames = [frame_id(f) for f in pr["frames"]]

    smpl = get_smpl(args.gender, True)
    pose = pose72to156(pr["smpl_pose"].detach().float())
    with torch.no_grad():
        _, joints = smpl.forward(pose, pr["betas"].detach().float(),
                                 pr["smpl_t"].detach().float())[:2]
    joints = joints.numpy()                       # (T, 52, 3) camera frame
    obj = pr["pose_abs"][:, :3, 3].numpy()        # camera frame

    R_wc, t_wc = read_extrinsics(args.calib)[args.cam]
    to_world = lambda x: (R_wc @ x.reshape(-1, 3).T).T + t_wc
    tri = None
    if args.npz:
        n = np.load(args.npz)
        tri = {int(f): p for f, p in zip(n["frames"].astype(int), n["xyz"])}

    idxs = [i for i, f in enumerate(frames) if args.lo <= f <= args.hi]
    ncol = 5
    nrow = (len(idxs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow),
                             subplot_kw={"projection": "3d"})
    for ax, i in zip(np.atleast_1d(axes).flat, idxs):
        jw = to_world(joints[i][:24])
        for child, parent in enumerate(PARENTS):
            if parent < 0:
                continue
            seg = np.stack([jw[parent], jw[child]])
            heavy = child in (16, 17, 18, 19, 20, 21, 22, 23)
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=limb_color(child),
                    lw=3.0 if heavy else 1.4)
        ow = to_world(obj[i])[0]
        ax.scatter(*ow, color="orange", s=120)
        if tri is not None and frames[i] in tri:
            ax.scatter(*tri[frames[i]], color="orange", s=160, marker="x")
        ax.set_title(f"frame {frames[i]}", fontsize=9)
        c = jw.mean(0)
        ax.set_xlim(c[0] - 1.1, c[0] + 1.1)
        ax.set_ylim(c[1] - 1.1, c[1] + 1.1)
        ax.set_zlim(c[2] - 1.2, c[2] + 1.2)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_axis_off()
    for ax in np.atleast_1d(axes).flat[len(idxs):]:
        ax.set_visible(False)
    fig.suptitle("CARI4D bundle, world frame: LEFT arm RED, RIGHT arm BLUE, "
                 "recon ball ORANGE dot, measured ball ORANGE X", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}  ({len(idxs)} frames, view azim={args.azim} elev={args.elev})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
