# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Triangulate named STATIC scene points from hand-clicked pixel correspondences.

A basketball hoop, a counter edge, a doorway: static scene objects need a pose
exactly once per take, so a handful of pixels clicked in two or more calibrated
views beats any inference. This intersects the sight-rays per named point with
the same Kannala-Brandt machinery that triangulated the ball, and reports the
reprojection residual per point so a mis-click is caught immediately.

Input is a JSON file of clicks:

    {
      "resolutions": {"cam01": [3840, 2160], "cam02": [3840, 2160]},
      "points": {
        "rim_center":     {"cam01": [1712, 843], "cam02": [2011, 799]},
        "board_top_left": {"cam01": [1650, 700], "cam03": [901, 512]}
      }
    }

"resolutions" states the pixel size of the images that were clicked on, per
camera (clicking on the 4K frames gives ~5x the precision of 448p). Each point
needs 2+ cameras; different points may use different camera pairs.

Output npz: names (S,), xyz (S,3) world frame -- the same world frame as the
object triangulations -- and residual (S,) in --width-scale pixels.

Usage:
    python prep/triangulate_points.py --calib <trajectory>/gopro_calibs.csv \\
        --points hoop_clicks.json --out hoop_points.npz
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from prep.triangulate_object import (pixel_to_ray, read_calibration,
                                     reprojection_error, triangulate_rays)


def parse_args():
    """Parse the calibration, the clicks JSON and the output path."""
    parser = argparse.ArgumentParser(
        description="Triangulate named static points from clicked correspondences")
    parser.add_argument("--calib", required=True,
                        help="gopro_calibs.csv from the take's trajectory/ directory")
    parser.add_argument("--points", required=True,
                        help="JSON of clicks; see the module docstring for the format")
    parser.add_argument("--width", type=int, default=796,
                        help="reference width residuals are expressed in (default: 796)")
    parser.add_argument("--max_residual", type=float, default=10.0,
                        help="warn when a point's mean reprojection error exceeds this "
                             "many reference-scale pixels (default: 10.0)")
    parser.add_argument("--out", default=None, help="write the points to this .npz")
    return parser.parse_args()


def main():
    """Intersect the rays per named point and print world position + residual."""
    args = parse_args()
    spec = json.load(open(args.points))
    res = spec["resolutions"]

    cams_by_res = {}
    def cam_for(uid):
        """Camera dict for uid at the resolution its clicks were made on."""
        w, h = res[uid]
        key = (w, h)
        if key not in cams_by_res:
            cams_by_res[key] = read_calibration(args.calib, w, h)
        if uid not in cams_by_res[key]:
            raise SystemExit(f"ERROR: {uid} not in the calibration")
        return cams_by_res[key]

    names, xyzs, residuals = [], [], []
    print(f"{'point':>18} {'views':>6} {'X':>8} {'Y':>8} {'Z':>8} {'reproj_px':>10}")
    print("-" * 64)
    for name, obs in spec["points"].items():
        if len(obs) < 2:
            print(f"{name:>18}  SKIPPED: needs 2+ views, has {len(obs)}")
            continue
        origins, dirs, checks = [], [], []
        for uid, uv in obs.items():
            cam = cam_for(uid)[uid]
            o, d = pixel_to_ray((float(uv[0]), float(uv[1])), cam)
            origins.append(o)
            dirs.append(d)
            checks.append((uv, cam, args.width / res[uid][0]))
        point, _ = triangulate_rays(origins, dirs)
        errs = [reprojection_error(point, uv, cam) * s for uv, cam, s in checks]
        r = float(np.mean(errs))
        flag = "  <-- CHECK CLICKS" if r > args.max_residual else ""
        print(f"{name:>18} {len(obs):>6} {point[0]:>8.3f} {point[1]:>8.3f} "
              f"{point[2]:>8.3f} {r:>10.2f}{flag}")
        names.append(name)
        xyzs.append(point)
        residuals.append(r)

    if not names:
        raise SystemExit("ERROR: no point had 2+ views")
    # Convenience: pairwise distances, the sanity check a rim earns against
    # known geometry (rim diameter 0.45m, board 1.8x1.05m, rim height 3.05m).
    if len(names) > 1:
        print("\npairwise distances (m):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = np.linalg.norm(np.array(xyzs[i]) - np.array(xyzs[j]))
                print(f"  {names[i]} <-> {names[j]}: {d:.3f}")
    if args.out:
        np.savez(args.out, names=np.array(names), xyz=np.array(xyzs),
                 residual=np.array(residuals))
        print(f"\nwrote {args.out} ({len(names)} points, world frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
