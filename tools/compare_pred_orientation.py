# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Measure how one prediction's body is rotated against another's.

A body that renders sideways can be fixed by trying rotations until it looks
right, which is slow and teaches nothing when it fails. If a prediction of the
SAME motion already renders correctly, the difference between them is the
answer, and it can be read rather than guessed.

Both files carry the body's global orientation in smpl_pose[:, :3] as an
axis-angle vector. For one motion described twice, the per-frame residual
R_a * R_b^-1 is the transform between the two frames. A residual that is
constant across frames is a frame convention, and applying its inverse fixes the
file; one that varies is not, and means the two describe different motions.

    python tools/compare_pred_orientation.py --a <wrong>.pth --b <correct>.pth

READ ONLY. It reports; it changes nothing.
"""
import argparse
import os.path as osp
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot


def parse_args():
    """Parse the two prediction files to compare."""
    parser = argparse.ArgumentParser(
        description="Report the rotation between two predictions of one motion")
    parser.add_argument("--a", required=True, help="the one that looks wrong")
    parser.add_argument("--b", required=True, help="the one that looks right")
    parser.add_argument("--key", default="pr", choices=["pr", "gt", "in"])
    return parser.parse_args()


def load(path, key):
    """Return (root rotations, smpl_t) from a prediction file.

    Raises:
        SystemExit: if the file or its fields are missing.
    """
    if not osp.isfile(path):
        raise SystemExit(f"no prediction file at {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if key not in data:
        raise SystemExit(f"'{key}' not in {osp.basename(path)}; got {list(data)}")
    block = data[key]
    pose = block["smpl_pose"].float().numpy().astype(np.float64)
    if pose.ndim == 3:
        pose = pose.reshape(len(pose), -1)
    return sRot.from_rotvec(pose[:, :3]), block["smpl_t"].float().numpy()


def main():
    """Report the residual rotation and whether it is a constant."""
    args = parse_args()
    Ra, ta = load(args.a, args.key)
    Rb, tb = load(args.b, args.key)
    n = min(len(Ra), len(Rb))
    if len(Ra) != len(Rb):
        print(f"lengths differ ({len(Ra)} vs {len(Rb)}); comparing the first {n}")
    Ra, Rb = Ra[:n], Rb[:n]

    print(f"a: {osp.basename(args.a)}")
    print(f"b: {osp.basename(args.b)}")
    print(f"{n} frames")
    print()
    print(f"  smpl_t a: x {ta[:n, 0].min():+.2f}..{ta[:n, 0].max():+.2f}  "
          f"y {ta[:n, 1].min():+.2f}..{ta[:n, 1].max():+.2f}  "
          f"z {ta[:n, 2].min():+.2f}..{ta[:n, 2].max():+.2f}")
    print(f"  smpl_t b: x {tb[:n, 0].min():+.2f}..{tb[:n, 0].max():+.2f}  "
          f"y {tb[:n, 1].min():+.2f}..{tb[:n, 1].max():+.2f}  "
          f"z {tb[:n, 2].min():+.2f}..{tb[:n, 2].max():+.2f}")

    residual = Ra * Rb.inv()
    rotvec = residual.as_rotvec()
    angles = np.degrees(np.linalg.norm(rotvec, axis=1))

    # The mean of a set of rotations, via the mean rotation vector, is only
    # meaningful when they agree -- which is exactly the question, so the spread
    # is reported beside it rather than after it.
    mean_rot = sRot.from_rotvec(rotvec.mean(axis=0))
    spread = np.degrees((residual * mean_rot.inv()).magnitude())

    print()
    print(f"  residual rotation a*b^-1: {angles.min():.1f}..{angles.max():.1f} deg "
          f"per frame")
    print(f"  spread about the mean   : {spread.mean():.1f} deg mean, "
          f"{spread.max():.1f} max")

    axis = mean_rot.as_rotvec()
    mag = np.linalg.norm(axis)
    print()
    if spread.mean() < 5.0:
        print(f"  -> CONSTANT, so this is a frame convention, not a difference "
              f"in motion.")
        print(f"     axis {np.round(axis / (mag + 1e-12), 3)}, "
              f"angle {np.degrees(mag):.1f} deg")
        print(f"     euler xyz {np.round(mean_rot.as_euler('xyz', degrees=True), 1)}")
        print(f"     matrix:\n{np.round(mean_rot.as_matrix(), 4)}")
        print()
        print(f"     Applying the INVERSE of this to a's root orientation and "
              f"translation brings it into b's frame.")
    else:
        print(f"  -> NOT constant ({spread.mean():.1f} deg mean spread). The two "
              f"files do not describe the same motion in two frames, so no "
              f"single rotation relates them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
