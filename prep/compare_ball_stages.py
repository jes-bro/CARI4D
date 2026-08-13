# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Compare the ball's height per pipeline stage to find where a bounce is lost.

The triangulated trajectory (bball_xyz.npz) showed the dribble bounce dipping
1.04m, but the final clip only drops ~0.8m -- some stage between the injected
depth and the exported motion flattened it. Each stage saves the object pose in
cam04's camera frame (`pose_abs`), while the triangulation lives in the take's
world frame, so this script maps every stage's translation into the world frame
via the same gopro calibration and prints the heights side by side. The column
where the dip goes shallow is the stage that ate the bounce.

Pass the stage .pth files oldest-first, e.g. CoCoNet output then optimizer
output; each is labelled by its parent directory. Needs torch (run in the
cari4d env, from the repo root).

Usage:
    python prep/compare_ball_stages.py --npz bball_xyz.npz \\
        --calib <trajectory>/gopro_calibs.csv --cam cam04 \\
        --pth output/coconet/<exp>/<seq>.pth output/opt/<exp>/<seq>.pth \\
        --lo 5 --hi 30
"""
import argparse
import csv
import os.path as osp

import numpy as np


def quaternion_to_matrix(q):
    """Convert an (x, y, z, w) quaternion to a rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_extrinsics(path):
    """Read gopro_calibs.csv and return {cam_uid: (R_wc, t_wc)}, camera-to-world.

    Deliberately not imported from triangulate_object.read_calibration: that
    pulls in cv2 (for the fisheye intrinsics this script never uses), and the
    extrinsic columns are all that matter for mapping poses between frames.
    """
    cams = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            t_wc = np.array([float(row["tx_world_cam"]), float(row["ty_world_cam"]),
                             float(row["tz_world_cam"])])
            q = np.array([float(row["qx_world_cam"]), float(row["qy_world_cam"]),
                          float(row["qz_world_cam"]), float(row["qw_world_cam"])])
            cams[row["cam_uid"]] = (quaternion_to_matrix(q), t_wc)
    return cams


def parse_args():
    """Parse the npz, calibration, camera uid, stage .pth paths and frame window."""
    parser = argparse.ArgumentParser(
        description="Print ball world-height per pipeline stage, side by side")
    parser.add_argument("--npz", required=True,
                        help="triangulate_object.py output (world-frame reference)")
    parser.add_argument("--calib", required=True,
                        help="gopro_calibs.csv from the take's trajectory/ directory")
    parser.add_argument("--cam", default="cam04",
                        help="camera uid whose frame the .pth poses live in (default cam04)")
    parser.add_argument("--pth", nargs="+", required=True,
                        help="stage .pth files (CoCoNet / optimizer output), oldest first")
    parser.add_argument("--lo", type=int, default=5,
                        help="first frame of the window of interest (default 5)")
    parser.add_argument("--hi", type=int, default=30,
                        help="last frame of the window of interest (default 30)")
    return parser.parse_args()


def load_stage_world_z(path, R_wc, t_wc):
    """Load a stage .pth and return {frame index: world z of the object centre}.

    Accepts both the CoCoNet bundle ({'pr': {...}}) and the optimizer output
    (same layout after 'pr' is overwritten); the translation is pose_abs[:,:3,3]
    in the camera frame, mapped to world via x_w = R_wc @ x_c + t_wc.
    """
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)
    pr = data["pr"] if isinstance(data, dict) and "pr" in data else data
    trans = pr["pose_abs"][:, :3, 3].numpy()
    world = (R_wc @ trans.T).T + t_wc
    # frames are stored as '<seq>/<frame_id>' in the CoCoNet bundle and as bare
    # ids elsewhere; the trailing path component is the id either way.
    return {int(str(f).split("/")[-1]): world[i, 2] for i, f in enumerate(pr["frames"])}


def summarise(label, zs):
    """Print one stage's min height in the table's rows and its drop from the first row."""
    vals = [(f, z) for f, z in zs if z is not None]
    if not vals:
        print(f"  {label}: no frames in window")
        return
    fmin, zmin = min(vals, key=lambda p: p[1])
    print(f"  {label}: min z {zmin:.3f} m at frame {fmin}, "
          f"drop from frame {vals[0][0]}: {vals[0][1] - zmin:.3f} m")


def main():
    """Print the per-frame world-z table and per-stage drop summaries."""
    args = parse_args()

    cams = read_extrinsics(args.calib)
    if args.cam not in cams:
        raise SystemExit(f"ERROR: {args.cam} not in calibration; available: {sorted(cams)}")
    R_wc, t_wc = cams[args.cam]

    tri = np.load(args.npz)
    tri_z = {int(f): z for f, z in zip(tri["frames"].astype(int), tri["xyz"][:, 2])}

    stages = [(osp.basename(osp.dirname(osp.abspath(p))) or p,
               load_stage_world_z(p, R_wc, t_wc)) for p in args.pth]

    labels = ["triangulated"] + [lab for lab, _ in stages]
    print(f"  {'frame':>6} " + " ".join(f"{lab:>18}" for lab in labels))
    rows = {lab: [] for lab in labels}
    for f in range(args.lo, args.hi + 1):
        cells = [tri_z.get(f)] + [zs.get(f) for _, zs in stages]
        for lab, z in zip(labels, cells):
            rows[lab].append((f, z))
        print(f"  {f:>6} " + " ".join(f"{z:>18.3f}" if z is not None else f"{'-':>18}"
                                      for z in cells))

    print("\n== summary (window {}-{}) ==".format(args.lo, args.hi))
    for lab in labels:
        summarise(lab, rows[lab])
    print("  read: the first column whose drop is much smaller than the "
          "triangulated drop is the stage that flattened the bounce.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
