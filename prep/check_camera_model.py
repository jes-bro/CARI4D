# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Test whether the pipeline's pinhole intrinsics explain the object's pixels.

The FP-vs-triangulation error is ~0.5m perpendicular to the viewing ray and
nearly constant across fast and slow frames alike -- the signature of a wrong
camera model, not of tracking noise. The monocular pipeline back-projects with
a pinhole K that UniDepth guessed from the image, while the footage is fisheye
(Kannala-Brandt), which the triangulation models properly.

This settles it per frame: take the triangulated 3D point (trusted, ~1-2 px
residuals), project it into the camera through (a) the pipeline's pinhole K
from the .color.pkl and (b) the calibrated fisheye model, and compare both to
the SAM3 mask centroid. Whichever projection lands on the centroid is the
model the pixels actually obey; the other one's pixel error, times depth over
focal, is the metres of lateral bias it injects into every stage that uses it.

Run in an env with cv2 (newcari4d), from the repo root.

Usage:
    python prep/check_camera_model.py --npz bball_xyz.npz \\
        --calib <trajectory>/gopro_calibs.csv --cam cam04 \\
        --video sam3masks/trimmed_vids-aligned/<seq>.0.color.mp4 \\
        --masks_root sam3masks --lo 5 --hi 30
"""
import argparse
import os
import os.path as osp
import sys

import cv2
import h5py
import joblib
import numpy as np

sys.path.append(os.getcwd())
from prep.triangulate_object import read_calibration


def parse_args():
    """Parse the npz, calibration, video (for pinhole K + masks) and window."""
    parser = argparse.ArgumentParser(
        description="Project triangulated points via pinhole vs fisheye, compare to mask centroids")
    parser.add_argument("--npz", required=True,
                        help="triangulate_object.py output (world-frame points)")
    parser.add_argument("--calib", required=True,
                        help="gopro_calibs.csv from the take's trajectory/ directory")
    parser.add_argument("--cam", default="cam04",
                        help="camera uid (default cam04)")
    parser.add_argument("--video", required=True,
                        help="the <seq>.<kid>.color.mp4 the pipeline ran on; its "
                             ".color.pkl sibling holds the pinhole intrinsics")
    parser.add_argument("--masks_root", required=True,
                        help="directory holding <seq>_masks_k<kid>.h5")
    parser.add_argument("--kid", type=int, default=0, help="camera id in mask keys (default 0)")
    parser.add_argument("--width", type=int, default=796,
                        help="mask/video width the fisheye K is scaled to (default 796)")
    parser.add_argument("--height", type=int, default=448,
                        help="mask/video height the fisheye K is scaled to (default 448)")
    parser.add_argument("--lo", type=int, default=5, help="first frame (default 5)")
    parser.add_argument("--hi", type=int, default=30, help="last frame (default 30)")
    return parser.parse_args()


def read_pinhole(video_path):
    """Read the pipeline's pinhole fx/fy/cx/cy from the video's .color.pkl."""
    pkl = video_path.replace(".color.mp4", ".color.pkl")
    if not osp.isfile(pkl):
        raise SystemExit(f"ERROR: no pinhole intrinsics at {pkl}")
    d = joblib.load(pkl)
    vals = [float(np.asarray(d[k]).ravel()[0]) for k in ("fx", "fy", "cx", "cy")]
    return vals


def mask_centroids(masks_root, seq, kid):
    """Return {frame index: (u, v)} centroids of the object masks in the H5."""
    path = osp.join(masks_root, f"{seq}_masks_k{kid}.h5")
    if not osp.isfile(path):
        raise SystemExit(f"ERROR: no mask file at {path}")
    out = {}
    with h5py.File(path, "r") as f:
        grp = f[seq] if seq in f else f
        for key in grp:
            if not key.endswith(f"-k{kid}.obj_rend_mask.png"):
                continue
            mask = grp[key][:]
            ys, xs = np.nonzero(mask)
            if len(xs):
                out[int(key.split("-")[0])] = (float(xs.mean()), float(ys.mean()))
    return out


def main():
    """Print centroid vs pinhole vs fisheye projection per frame, with pixel errors."""
    args = parse_args()
    seq = osp.basename(args.video).split(".")[0]

    cams = read_calibration(args.calib, args.width, args.height)
    if args.cam not in cams:
        raise SystemExit(f"ERROR: {args.cam} not in calibration; available: {sorted(cams)}")
    cam = cams[args.cam]
    fx, fy, cx, cy = read_pinhole(args.video)

    tri = np.load(args.npz)
    tri_xyz = {int(f): p for f, p in zip(tri["frames"].astype(int), tri["xyz"])}
    cents = mask_centroids(args.masks_root, seq, args.kid)

    print(f"pinhole K (from .color.pkl): fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print(f"fisheye K (from calib, scaled to {args.width}x{args.height}): "
          f"fx={cam['K'][0,0]:.1f} fy={cam['K'][1,1]:.1f} "
          f"cx={cam['K'][0,2]:.1f} cy={cam['K'][1,2]:.1f}  dist={np.round(cam['dist'],4)}")
    print(f"\n  {'frame':>6} {'centroid':>14} {'pinhole->px':>14} {'err':>6} "
          f"{'fisheye->px':>14} {'err':>6} {'~m/px_err':>10}")

    errs_pin, errs_fish = [], []
    for f in range(args.lo, args.hi + 1):
        if f not in tri_xyz or f not in cents:
            continue
        x_c = cam["R_cw"] @ tri_xyz[f] + cam["t_cw"]
        depth = x_c[2]
        u_pin = (fx * x_c[0] / depth + cx, fy * x_c[1] / depth + cy)
        pts = x_c.reshape(1, 1, 3).astype(np.float64)
        fish, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3),
                                            cam["K"], cam["dist"].reshape(4, 1))
        u_fish = tuple(fish.reshape(2))
        c = cents[f]
        e_pin = float(np.hypot(u_pin[0] - c[0], u_pin[1] - c[1]))
        e_fish = float(np.hypot(u_fish[0] - c[0], u_fish[1] - c[1]))
        errs_pin.append(e_pin)
        errs_fish.append(e_fish)
        m_per_px = depth / fx
        print(f"  {f:>6} {c[0]:>6.1f},{c[1]:>6.1f} {u_pin[0]:>6.1f},{u_pin[1]:>6.1f} "
              f"{e_pin:>6.1f} {u_fish[0]:>6.1f},{u_fish[1]:>6.1f} {e_fish:>6.1f} "
              f"{m_per_px:>10.3f}")

    if errs_pin:
        print(f"\n  median pixel error: pinhole {np.median(errs_pin):.1f} px, "
              f"fisheye {np.median(errs_fish):.1f} px over {len(errs_pin)} frames")
        print("  read: if fisheye sits on the centroid and pinhole misses by many px, "
              "the pipeline's camera model is the bias; px err * m/px_err "
              "approximates the lateral metres it injects.")
    else:
        print("\n  no frames with both a triangulated point and a mask centroid in the window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
