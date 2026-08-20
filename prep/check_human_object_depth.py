# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Test whether the reconstructed human sits at the right depth, using the ball.

The object's position is now multi-view triangulated -- centimetre-class,
model-free. The human's absolute depth is not: it comes from monocular SMPL
fitting, whose error along the viewing ray can be tens of centimetres. When
both were wrong they could still overlap by luck; a measured ball exposes the
mismatch as "the ball floats in front of / behind the person" in the render.

While the person HOLDS the object, their hands and the object centre must be
at the same camera distance (within roughly the object's radius). So this
script runs SMPL-H forward on the reconstruction, takes the wrist closest to
the ball per frame, and compares camera distances: hand vs triangulated ball.
A systematic gap in the hold window is the human's depth error, in metres,
with its sign (positive = human placed too far, ball renders in front;
negative = human too close, ball renders behind).

Run from the repo root in an env with torch + the SMPL data (newcari4d).

Usage:
    python prep/check_human_object_depth.py \\
        --pth output/opt/<exp>/<seq>.pth --npz bball_xyz_all.npz \\
        --calib <trajectory>/gopro_calibs.csv --cam cam04 --lo 50 --hi 100
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())
from lib_smpl import get_smpl, pose72to156
from prep.compare_ball_stages import TOLERANT_PICKLE, frame_id, read_extrinsics

# SMPL joint ordering: 20 = left wrist, 21 = right wrist (same in SMPL-H).
WRISTS = (20, 21)


def parse_args():
    """Parse the reconstruction bundle, triangulation npz, calibration and window."""
    parser = argparse.ArgumentParser(
        description="Compare hand vs triangulated-object camera distance per frame")
    parser.add_argument("--pth", required=True,
                        help="CoCoNet or optimizer bundle; needs pr.smpl_pose/smpl_t/betas")
    parser.add_argument("--npz", required=True,
                        help="triangulate_object.py output (world-frame points)")
    parser.add_argument("--calib", required=True,
                        help="gopro_calibs.csv from the take's trajectory/ directory")
    parser.add_argument("--cam", default="cam04", help="camera uid (default cam04)")
    parser.add_argument("--gender", default="male", help="SMPL gender (default male)")
    parser.add_argument("--lo", type=int, default=50, help="first frame (default 50)")
    parser.add_argument("--hi", type=int, default=100, help="last frame (default 100)")
    return parser.parse_args()


def main():
    """Print per-frame hand vs ball camera distance and the systematic gap."""
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
    wrists = joints[:, WRISTS, :].numpy()          # (T, 2, 3), camera frame

    obj_cam = pr["pose_abs"][:, :3, 3].numpy()     # reconstructed object, camera frame

    R_wc, t_wc = read_extrinsics(args.calib)[args.cam]
    tri = np.load(args.npz)
    # World -> this camera's frame, so all three quantities share an origin.
    tri_cam = {int(f): R_wc.T @ (p - t_wc)
               for f, p in zip(tri["frames"].astype(int), tri["xyz"])}

    print(f"{'frame':>6} {'hand':>5} {'hand_d':>8} {'ball_d':>8} {'gap_m':>8} "
          f"{'recon_obj_d':>12} {'wrist-ball_m':>13}")
    gaps = []
    for i, f in enumerate(frames):
        if f < args.lo or f > args.hi or f not in tri_cam:
            continue
        ball = tri_cam[f]
        ball_d = float(np.linalg.norm(ball))
        # The wrist nearer the ball is the one plausibly holding it.
        w_dists = np.linalg.norm(wrists[i] - ball, axis=1)
        which = int(np.argmin(w_dists))
        hand = "L" if which == 0 else "R"   # WRISTS = (20 left, 21 right)
        w = wrists[i][which]
        hand_d = float(np.linalg.norm(w))
        gap = hand_d - ball_d
        gaps.append(gap)
        print(f"{f:>6} {hand:>5} {hand_d:>8.3f} {ball_d:>8.3f} {gap:>8.3f} "
              f"{np.linalg.norm(obj_cam[i]):>12.3f} {w_dists.min():>13.3f}")

    if not gaps:
        print("no frames with both a reconstruction and a triangulated point in the window")
        return 1
    gaps = np.array(gaps)
    print(f"\nhand-vs-ball camera-distance gap over {len(gaps)} frames: "
          f"median {np.median(gaps):+.3f} m, mean {gaps.mean():+.3f} m, "
          f"std {gaps.std():.3f} m")
    print("read: during a hold the gap should be within ~the object radius. A "
          "systematic offset is the human's depth error: positive = human too "
          "far (ball renders in front), negative = human too close (ball "
          "renders behind). wrist-ball_m is the plain 3D distance; large "
          "values mean the frame is not a hold and says nothing about depth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
