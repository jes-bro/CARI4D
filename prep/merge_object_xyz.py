# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Merge triangulate_object.py outputs into one trajectory.

Different mask sets cover different stretches of a clip -- on the basketball
take, the 448p masks track the dribble (frames 0-45) and the 4K re-runs track
the pickup and shot (46-100) -- and triangulation on each yields a partial
npz. The world-frame points are model- and resolution-free, so merging is
legitimate: take the union of frames, and where several inputs cover the same
frame keep the one with the smallest reprojection residual.

Usage:
    python prep/merge_object_xyz.py --npz bball_xyz.npz bball_xyz_4k.npz \\
        --out bball_xyz_merged.npz
"""
import argparse

import numpy as np


def parse_args():
    """Parse the input npz files and the output path."""
    parser = argparse.ArgumentParser(
        description="Merge triangulated-object npz files, keeping the lowest "
                    "residual where frames overlap")
    parser.add_argument("--npz", nargs="+", required=True,
                        help="two or more triangulate_object.py outputs, any order")
    parser.add_argument("--out", required=True, help="merged npz to write")
    return parser.parse_args()


def coverage_runs(frames):
    """Format sorted frame indices as human-readable contiguous runs."""
    runs = np.split(frames, np.where(np.diff(frames) > 1)[0] + 1)
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs)


def main():
    """Union the inputs frame-wise, preferring the smallest residual, and report."""
    args = parse_args()
    best = {}
    for path in args.npz:
        d = np.load(path)
        n_won = 0
        for f, xyz, res in zip(d["frames"].astype(int), d["xyz"], d["residual"]):
            if f not in best or res < best[f][1]:
                best[f] = (xyz, float(res), path)
                n_won += 1
        print(f"{path}: {len(d['frames'])} frames "
              f"({coverage_runs(np.sort(d['frames'].astype(int)))})")

    frames = np.array(sorted(best))
    xyz = np.array([best[f][0] for f in frames])
    residual = np.array([best[f][1] for f in frames])
    for path in args.npz:
        n = sum(1 for f in frames if best[f][2] == path)
        print(f"  kept from {path}: {n} frames")

    np.savez(args.out, frames=frames, xyz=xyz, residual=residual)
    print(f"\nwrote {args.out}: {len(frames)} frames "
          f"({coverage_runs(frames)}), residual median "
          f"{np.median(residual):.2f} px")
    lo, hi = int(frames.min()), int(frames.max())
    missing = sorted(set(range(lo, hi + 1)) - set(frames.tolist()))
    print(f"uncovered within {lo}-{hi}: {missing if missing else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
