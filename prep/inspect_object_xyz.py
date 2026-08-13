# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Inspect a triangulate_object.py output .npz: coverage, z profile, residuals.

triangulate_object.py writes only the frames that passed its residual check,
and inject_object_depth.py silently leaves every other frame's depth untouched.
So when a reconstructed trajectory goes wrong in some window (e.g. a dribble
bounce that never reaches the floor), the first question is always: did the
triangulation actually cover those frames, and what did it say there?

This prints, per npz: total coverage, the frames covered and MISSING inside the
window of interest, a frame / z / residual table for the window, and where z
bottoms out overall. Reading the output:

  - Missing frames inside the window  -> injection fell back to monocular depth
    exactly there; fix the input (second-view masks / triangulation).
  - Window covered, z dips plausibly  -> the input was fine; whatever flattened
    the trajectory happened downstream (CoCoNet / optimizer smoothing).
  - Residuals jumping from ~1-2 px to large values -> the views disagree on
    those frames, usually a mask failing under motion blur.

Only needs numpy, so it runs in any env.

Usage:
    python3 prep/inspect_object_xyz.py bball_xyz.npz bball_xyz_ego.npz
    python3 prep/inspect_object_xyz.py bball_xyz.npz --lo 5 --hi 30
    python3 prep/inspect_object_xyz.py bball_xyz.npz --all
"""
import argparse

import numpy as np


def parse_args():
    """Parse npz paths and the frame window to report in detail."""
    parser = argparse.ArgumentParser(
        description="Inspect triangulated object trajectories (.npz from triangulate_object.py)")
    parser.add_argument("npz", nargs="+", help="one or more .npz files to inspect")
    parser.add_argument("--lo", type=int, default=5,
                        help="first frame of the window of interest (default 5)")
    parser.add_argument("--hi", type=int, default=30,
                        help="last frame of the window of interest (default 30)")
    parser.add_argument("--all", action="store_true",
                        help="print the full per-frame table, not just the window")
    return parser.parse_args()


def report(path, lo, hi, show_all):
    """Print coverage, the windowed frame/z/residual table, and z extrema for one npz."""
    data = np.load(path)
    frames = data["frames"].astype(int)
    xyz = data["xyz"]
    residual = data["residual"]

    print(f"\n=== {path} ===")
    print(f"  frames covered: {len(frames)}  "
          f"(range {frames.min()}-{frames.max()})" if len(frames) else "  EMPTY npz")
    if not len(frames):
        return

    covered = set(frames.tolist())
    missing = [f for f in range(max(lo, frames.min()), min(hi, frames.max()) + 1)
               if f not in covered]
    in_window = (frames >= lo) & (frames <= hi)
    print(f"  window {lo}-{hi}: {int(in_window.sum())} frames covered, "
          f"{len(missing)} MISSING: {missing if missing else '-'}")

    sel = np.ones_like(frames, dtype=bool) if show_all else in_window
    if sel.any():
        print(f"  {'frame':>6} {'x':>8} {'y':>8} {'z':>8} {'resid(px)':>10}")
        for f, p, r in zip(frames[sel], xyz[sel], residual[sel]):
            print(f"  {f:>6} {p[0]:>8.3f} {p[1]:>8.3f} {p[2]:>8.3f} {r:>10.1f}")
    else:
        print("  (no covered frames inside the window)")

    zmin_i = int(np.argmin(xyz[:, 2]))
    zmax_i = int(np.argmax(xyz[:, 2]))
    print(f"  z bottoms at {xyz[zmin_i, 2]:.3f} m (frame {frames[zmin_i]}), "
          f"peaks at {xyz[zmax_i, 2]:.3f} m (frame {frames[zmax_i]})")
    print(f"  residuals: median {np.median(residual):.1f} px, max {residual.max():.1f} px "
          f"(frame {frames[int(np.argmax(residual))]})")


def main():
    """Report every npz given on the command line."""
    args = parse_args()
    for path in args.npz:
        report(path, args.lo, args.hi, args.all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
