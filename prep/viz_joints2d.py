# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Draw packed 2D keypoints over their video, to check Step 3 before Step 4.

run_sapiens_pose.py writes <seq>_GT-packed.pkl and prints nothing about
quality. Nothing downstream renders anything either until the very end of the
pipeline, so bad keypoints surface as a bad final result several stages later,
with no obvious attribution. This closes that gap: overlay the skeleton on the
frames and look.

Handles both packed formats the pipeline accepts, detected from the joint count
exactly as the loaders do -- 17 for COCO/Sapiens, 25 for OpenPose BODY_25.

Usage:
    python prep/viz_joints2d.py --video <clip>.mp4 --packed_root <dir>
    python prep/viz_joints2d.py --video <clip>.mp4 --packed_root <dir> --stills 8
"""
import argparse
import os
import os.path as osp
import sys

import cv2
import joblib
import numpy as np

sys.path.append(os.getcwd())

from prep.run_hy3d_recon import extract_seq_name

# (start, end) index pairs. COCO's 17 keypoints omit the neck and mid-hip that
# BODY_25 carries, so the two skeletons are not interchangeable.
COCO17_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
]
BODY25_EDGES = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10),
    (10, 11), (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16),
    (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24),
]

LEFT_RGB = (80, 160, 255)    # BGR: warm, for the subject's left
RIGHT_RGB = (80, 255, 160)   # BGR: cool, for the subject's right
LOW_CONF_RGB = (60, 60, 200)


def parse_args():
    """Parse the video, packed keypoint location and output options."""
    parser = argparse.ArgumentParser(
        description="Overlay packed 2D keypoints on their video")
    parser.add_argument("--video", required=True, help="the clip the keypoints belong to")
    parser.add_argument("--packed_root", required=True,
                        help="directory holding <seq>_GT-packed.pkl")
    parser.add_argument("--seq", default=None,
                        help="sequence name (default: derived from --video)")
    parser.add_argument("--view", type=int, default=0,
                        help="which view of the K axis to draw (default: 0)")
    parser.add_argument("--out", default=None,
                        help="output mp4 (default: <packed_root>/<seq>_joints2d.mp4)")
    parser.add_argument("--stills", type=int, default=0,
                        help="instead of a video, write this many evenly spaced PNGs")
    parser.add_argument("--conf_thres", type=float, default=0.3,
                        help="draw joints below this confidence in a warning colour "
                             "rather than hiding them, so gaps stay visible (default: 0.3)")
    parser.add_argument("--fps", type=float, default=30.0, help="output fps (default: 30)")
    return parser.parse_args()


def load_packed(packed_root, seq_name):
    """Load the packed keypoint dict and report its shape.

    Returns:
        (frames, joints2d) where joints2d is (N, K, J, 3).

    Raises:
        SystemExit: if the file or the expected keys are missing.
    """
    path = osp.join(packed_root, f"{seq_name}_GT-packed.pkl")
    if not osp.isfile(path):
        raise SystemExit(f"ERROR: no packed keypoints at {path}")
    data = joblib.load(path)
    for key in ("frames", "joints2d"):
        if key not in data:
            raise SystemExit(f"ERROR: '{key}' missing from {path}; found {list(data)}")
    joints = np.asarray(data["joints2d"])
    if joints.ndim != 4 or joints.shape[-1] != 3:
        raise SystemExit(
            f"ERROR: expected joints2d of shape (N, K, J, 3), got {joints.shape}")
    print(f"Loaded {path}")
    print(f"  frames   : {len(data['frames'])}")
    print(f"  joints2d : {joints.shape}  (N, views, joints, xyc)")
    return data["frames"], joints


def edges_for(num_joints):
    """Return the skeleton edges for a joint count, matching the loaders' detection.

    Raises:
        SystemExit: on a joint count neither format uses, which usually means the
            packing step wrote something the pipeline will silently misread.
    """
    if num_joints == 17:
        print("  format   : COCO 17 (Sapiens)")
        return COCO17_EDGES
    if num_joints == 25:
        print("  format   : OpenPose BODY_25")
        return BODY25_EDGES
    raise SystemExit(
        f"ERROR: {num_joints} joints is neither COCO 17 nor BODY_25. The pipeline "
        f"auto-detects the format from this dimension and would misread it.")


def draw_skeleton(frame, joints, edges, conf_thres):
    """Draw one frame's skeleton in place.

    Low-confidence joints are drawn in a warning colour rather than hidden --
    a missing limb and a badly-placed one look identical if you only ever draw
    what the detector was sure about.

    Returns:
        Number of joints drawn above the confidence threshold.
    """
    confident = 0
    for start, end in edges:
        if max(start, end) >= len(joints):
            continue
        (x0, y0, c0), (x1, y1, c1) = joints[start], joints[end]
        if c0 <= 0 or c1 <= 0:
            continue
        weak = c0 < conf_thres or c1 < conf_thres
        colour = LOW_CONF_RGB if weak else (LEFT_RGB if start % 2 else RIGHT_RGB)
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), colour,
                 1 if weak else 2, cv2.LINE_AA)

    for x, y, c in joints:
        if c <= 0:
            continue
        weak = c < conf_thres
        if not weak:
            confident += 1
        cv2.circle(frame, (int(x), int(y)), 2 if weak else 3,
                   LOW_CONF_RGB if weak else (240, 240, 240), -1, cv2.LINE_AA)
    return confident


def main():
    """Overlay the packed keypoints on the video and report per-frame coverage."""
    args = parse_args()
    seq_name = args.seq or extract_seq_name(args.video)
    frames_list, joints2d = load_packed(args.packed_root, seq_name)
    edges = edges_for(joints2d.shape[2])

    if args.view >= joints2d.shape[1]:
        raise SystemExit(
            f"ERROR: --view {args.view} but joints2d has {joints2d.shape[1]} view(s)")

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  video    : {W}x{H}, {total} frames")
    if total != len(frames_list):
        print(f"  WARNING  : video has {total} frames but the pkl has "
              f"{len(frames_list)} -- they may not correspond")

    still_at = set()
    writer = None
    if args.stills:
        n = min(args.stills, len(frames_list))
        still_at = {int(round(i)) for i in np.linspace(0, len(frames_list) - 1, n)}
        out_dir = args.out or osp.join(args.packed_root, f"{seq_name}_joints2d")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_path = args.out or osp.join(args.packed_root, f"{seq_name}_joints2d.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (W, H))

    drawn, empty, idx = 0, 0, 0
    while idx < len(frames_list):
        ok, frame = cap.read()
        if not ok:
            break
        joints = joints2d[idx, args.view]
        confident = draw_skeleton(frame, joints, edges, args.conf_thres)
        if confident == 0:
            empty += 1
        cv2.putText(frame, f"{frames_list[idx]}  {confident}/{len(joints)} joints",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if args.stills:
            if idx in still_at:
                path = osp.join(out_dir, f"frame{idx:06d}.png")
                cv2.imwrite(path, frame)
                print(f"  wrote {path}")
                drawn += 1
        else:
            writer.write(frame)
            drawn += 1
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Wrote {out_path} ({drawn} frames)")
    else:
        print(f"Wrote {drawn} still(s) to {out_dir}")

    print(f"Frames with no confident joints: {empty}/{idx}")
    if empty:
        print("  Those are frames the detector found nothing in. A few is normal; "
              "many means the person masks or the detection are wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
