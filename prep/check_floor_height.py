# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Locate the floor from the reconstructed person, independently of the ball.

Two things claim to know where the ground is and they disagree by about 24 cm.
The ball's ballistic arcs put contact at a height that implies one floor; the
person's feet, once grounded, imply another. Deciding between them from the ball
alone is circular -- taking contact height minus radius as the floor assumes the
very thing in question.

This asks the person instead. The reconstruction is in the filming camera's
frame and gopro_calibs.csv gives that camera's pose in the world, so the feet
can be placed in the same world coordinates the ball was triangulated in and
compared there. Nothing in the chain touches the ball.

    python prep/check_floor_height.py --bundle <cari4d .pth> \\
        --calib <trajectory>/gopro_calibs.csv --cam cam04 --ball_contact -1.30

READ ONLY. Nothing is written; this only reports.
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from prep.triangulate_object import read_calibration

# SMPL-H joint indices, the standard ordering used throughout this repo.
L_ANKLE, R_ANKLE, L_FOOT, R_FOOT = 7, 8, 10, 11
PELVIS = 0


def parse_args():
    """Parse the reconstruction, the camera it was reconstructed in, and the claim."""
    parser = argparse.ArgumentParser(
        description="Find the floor from the reconstructed person, not the ball")
    parser.add_argument("--bundle", required=True, help="CARI4D .pth bundle")
    parser.add_argument("--calib", required=True, help="gopro_calibs.csv")
    parser.add_argument("--cam", default="cam04",
                        help="camera the reconstruction is expressed in")
    parser.add_argument("--gender", default="male",
                        help="SMPL-H gender used during reconstruction")
    parser.add_argument("--bundle_key", default="pr", choices=["pr", "gt", "in"])
    parser.add_argument("--ball_contact", type=float, default=None,
                        help="world z the ball's arcs put contact at, for "
                             "comparison (e.g. -1.30)")
    parser.add_argument("--ball_radius", type=float, default=0.13)
    parser.add_argument("--percentile", type=float, default=10.0,
                        help="percentile of per-frame lowest foot taken as the "
                             "floor, matching rotate_pt --drop-to-floor "
                             "(default: 10)")
    return parser.parse_args()


def load_bundle(path):
    """Return the bundle dict, tolerating classes that are not importable here.

    Raises:
        SystemExit: if the file is missing.
    """
    if not osp.isfile(path):
        raise SystemExit(f"ERROR: no bundle at {path}")
    import pickle

    class _Unpickler(pickle.Unpickler):
        """Substitute a plain dict for any class this process cannot import."""

        def find_class(self, module, name):
            """Return the real class, or dict when it cannot be imported."""
            try:
                return super().find_class(module, name)
            except Exception:
                return dict

    class _PickleModule:
        """Minimal pickle-module stand-in for torch.load."""
        Unpickler = _Unpickler

        @staticmethod
        def load(fh, **kwargs):
            """Load with the permissive unpickler."""
            return _Unpickler(fh).load()

    with open(path, "rb") as fh:
        return torch.load(fh, map_location="cpu", weights_only=False,
                          pickle_module=_PickleModule)


def smplh_joints(poses, trans, betas, gender):
    """Return (T, J, 3) SMPL-H joints in the reconstruction's camera frame.

    Raises:
        SystemExit: if the body model cannot be built, since a fallback guess at
            foot height would defeat the purpose of an independent measurement.
    """
    try:
        import smplx
        from lib_smpl.const import SMPL_MODEL_ROOT, NUM_BETAS
    except Exception as exc:
        raise SystemExit(f"ERROR: cannot load the SMPL-H stack ({exc}); run this "
                         f"from the repo root in the cari4d env")

    T = len(poses)
    model = smplx.create(model_path=SMPL_MODEL_ROOT, model_type="smplh",
                         gender=gender, use_pca=False, num_betas=NUM_BETAS,
                         batch_size=T, flat_hand_mean=True)
    with torch.no_grad():
        out = model(betas=torch.as_tensor(betas[:, :NUM_BETAS], dtype=torch.float32),
                    global_orient=torch.as_tensor(poses[:, :3], dtype=torch.float32),
                    body_pose=torch.as_tensor(poses[:, 3:66], dtype=torch.float32),
                    left_hand_pose=torch.as_tensor(poses[:, 66:111], dtype=torch.float32),
                    right_hand_pose=torch.as_tensor(poses[:, 111:156], dtype=torch.float32),
                    transl=torch.as_tensor(trans, dtype=torch.float32))
    return out.joints.numpy()


def main():
    """Report the floor implied by the feet, in the ball's world frame."""
    args = parse_args()

    bundle = load_bundle(args.bundle)
    if args.bundle_key not in bundle:
        raise SystemExit(f"ERROR: bundle has no '{args.bundle_key}'; "
                         f"got {list(bundle)}")
    src = bundle[args.bundle_key]
    poses = src["smpl_pose"].detach().cpu().numpy()
    trans = src["smpl_t"].detach().cpu().numpy()
    betas = src["betas"].detach().cpu().numpy()
    if poses.shape[1] == 72:
        from lib_smpl import pose72to156
        poses = pose72to156(poses.astype(np.float64))
    print(f"{osp.basename(args.bundle)}: {len(poses)} frames, gender={args.gender}")

    joints = smplh_joints(poses, trans, betas, args.gender)
    print(f"joints: {joints.shape}")

    # The calibration's resolution does not matter here -- only the pose is used,
    # and that is resolution independent.
    cams = read_calibration(args.calib, 1, 1)
    if args.cam not in cams:
        raise SystemExit(f"ERROR: {args.cam} not in the calibration; "
                         f"found {sorted(cams)}")
    cam = cams[args.cam]
    R_wc = cam["R_cw"].T          # camera-to-world
    t_wc = cam["centre"]

    world = joints @ R_wc.T + t_wc
    feet = world[:, [L_ANKLE, R_ANKLE, L_FOOT, R_FOOT], 2]
    lowest = feet.min(axis=1)
    floor = float(np.percentile(lowest, args.percentile))

    print()
    print(f"in {args.cam}'s world frame:")
    print(f"  camera centre z   {t_wc[2]:+.3f} m")
    print(f"  pelvis z          {world[:, PELVIS, 2].min():+.3f} .. "
          f"{world[:, PELVIS, 2].max():+.3f}")
    print(f"  lowest foot z     {lowest.min():+.3f} .. {lowest.max():+.3f}")
    print(f"  floor ({args.percentile:.0f}th pct) {floor:+.3f} m")
    print(f"  camera sits {t_wc[2] - floor:.2f} m above that floor")

    if args.ball_contact is None:
        return 0

    ball_floor = args.ball_contact - args.ball_radius
    gap = ball_floor - floor
    print()
    print(f"  ball's arcs put contact at {args.ball_contact:+.3f}, so the ball "
          f"implies a floor at {ball_floor:+.3f}")
    print(f"  the feet imply {floor:+.3f} -- they disagree by {gap:+.3f} m")
    if abs(gap) < 0.06:
        print("  -> they AGREE. Both the person and the ball are right in world "
              "coordinates, so the discrepancy in the .pt was introduced "
              "downstream -- suspect rotate_pt --drop-to-floor's foot "
              "percentile.")
    elif ball_floor > floor:
        print("  -> the ball's floor is HIGHER, i.e. the ball stops short of "
              "where the feet stand. The ball is tracked high at the bounce "
              "even though it is right when held, which fits the mask riding "
              "up when the ball is behind his legs at floor level.")
    else:
        print("  -> the ball's floor is LOWER than the feet's, so the person is "
              "reconstructed above the ground the ball bounces on. Suspect the "
              "human depth alignment rather than the object.")
    print()
    print(f"  a tripod-mounted camera would sit roughly 1.2-1.6 m up; this one "
          f"is {t_wc[2] - floor:.2f} m above the feet's floor and "
          f"{t_wc[2] - ball_floor:.2f} m above the ball's, which is a further "
          f"check on which is plausible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
