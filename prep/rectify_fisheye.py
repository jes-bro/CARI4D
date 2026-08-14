# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Rectify a fisheye clip (and its masks) to a true pinhole for the pipeline.

The monocular pipeline assumes a pinhole camera everywhere -- NLF, alignment,
FoundationPose, silhouettes -- but egoexo4d's GoPros are Kannala-Brandt
fisheye, and UniDepth's guessed pinhole K missed the true rays by ~48 px
(~0.6 m sideways at 5 m; measured by prep/check_camera_model.py). Instead of
teaching every stage about distortion, this remaps the video once so a pinhole
model becomes *correct*, and writes the exact rectified K to the .color.pkl,
which prep/unidepth_behave.py now honors instead of guessing.

Outputs, in --out_dir, all named so the pipeline consumes them directly:
    <seq>.<kid>.color.mp4          rectified clip (same resolution)
    <seq>.<kid>.color.pkl          the true pinhole intrinsics (fx/fy/cx/cy)
    <seq>_masks_k<kid>.h5          masks warped with the same map (--masks_root)

Everything downstream then lives in rectified pixel space: 2D keypoints must
be re-detected on the rectified clip, and depth injection must target the
rectified masks. Triangulation is the exception -- it stays on the ORIGINAL
fisheye inputs (it models the distortion properly; feeding it rectified pixels
would undistort twice). Its 3D output transfers unchanged.

Run in an env with cv2 (newcari4d), from the repo root.

Usage:
    python prep/rectify_fisheye.py \\
        --video sam3masks/trimmed_vids/<seq>.0.color.mp4 \\
        --calib <trajectory>/gopro_calibs.csv --cam cam04 \\
        --masks_root sam3masks --out_dir rect-<seq>
"""
import argparse
import os
import os.path as osp
import sys

import cv2
import h5py
import imageio
import joblib
import numpy as np

sys.path.append(os.getcwd())
from prep.triangulate_object import read_calibration


def parse_args():
    """Parse the source video, calibration, camera uid, masks and output dir."""
    parser = argparse.ArgumentParser(
        description="Undistort a fisheye clip + masks to a true pinhole camera")
    parser.add_argument("--video", required=True,
                        help="source <seq>.<kid>.color.mp4 (fisheye pixels)")
    parser.add_argument("--calib", required=True,
                        help="gopro_calibs.csv from the take's trajectory/ directory")
    parser.add_argument("--cam", default="cam04", help="camera uid (default cam04)")
    parser.add_argument("--masks_root", default=None,
                        help="directory holding <seq>_masks_k<kid>.h5 to warp alongside")
    parser.add_argument("--out_dir", required=True,
                        help="output directory for the rectified clip, pkl and masks")
    parser.add_argument("--balance", type=float, default=0.0,
                        help="cv2.fisheye balance: 0 crops to fully valid pixels, "
                             "1 keeps the whole fisheye view with black borders "
                             "(default 0.0)")
    return parser.parse_args()


def build_maps(cam, width, height, balance):
    """Return (new_K, map1, map2) for remapping fisheye pixels to pinhole."""
    K, dist = cam["K"], cam["dist"].reshape(4, 1)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, dist, (width, height), np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, dist, np.eye(3), new_K, (width, height), cv2.CV_16SC2)
    return new_K, map1, map2


def rectify_video(video, out_video, map1, map2):
    """Remap every frame of the clip and return (n_frames, fps, (H, W))."""
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data().get("fps", 30)
    writer = None
    n, shape = 0, None
    for frame in reader:
        out = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        if writer is None:
            shape = out.shape[:2]
            # macro_block_size=1 or imageio pads 796 -> 800 and the clip stops
            # matching its own masks (same lesson as run_sam3_masks.save_trimmed).
            writer = imageio.get_writer(out_video, fps=fps, macro_block_size=1)
        writer.append_data(out)
        n += 1
    reader.close()
    if writer is None:
        raise SystemExit(f"ERROR: no frames in {video}")
    writer.close()
    return n, fps, shape


def rectify_masks(masks_root, seq, kid, out_path, map1, map2):
    """Warp every mask in the sequence's H5 with the same map, preserving layout."""
    src = osp.join(masks_root, f"{seq}_masks_k{kid}.h5")
    if not osp.isfile(src):
        raise SystemExit(f"ERROR: no mask file at {src}")
    n = 0
    with h5py.File(src, "r") as fin, h5py.File(out_path, "w") as fout:
        grp_in = fin[seq] if seq in fin else fin
        grp_out = fout.create_group(seq) if seq in fin else fout
        for key in grp_in:
            mask = grp_in[key][:].astype(np.uint8)
            warped = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST) > 0
            grp_out.create_dataset(key, data=warped, compression="gzip")
            n += 1
    return n


def main():
    """Rectify the clip, warp its masks, and write the true pinhole intrinsics."""
    args = parse_args()
    base = osp.basename(args.video)
    seq, kid = base.split(".")[0], int(base.split(".")[1])
    os.makedirs(args.out_dir, exist_ok=True)

    probe = imageio.get_reader(args.video)
    first = probe.get_data(0)
    probe.close()
    H, W = first.shape[:2]

    cams = read_calibration(args.calib, W, H)
    if args.cam not in cams:
        raise SystemExit(f"ERROR: {args.cam} not in calibration; available: {sorted(cams)}")
    new_K, map1, map2 = build_maps(cams[args.cam], W, H, args.balance)

    out_video = osp.join(args.out_dir, base)
    n_frames, fps, shape = rectify_video(args.video, out_video, map1, map2)
    print(f"rectified {n_frames} frames @ {fps} fps -> {out_video}  ({shape[1]}x{shape[0]})")

    fx, fy = float(new_K[0, 0]), float(new_K[1, 1])
    cx, cy = float(new_K[0, 2]), float(new_K[1, 2])
    pkl_path = out_video.replace(".color.mp4", ".color.pkl")
    joblib.dump({"fx": fx, "fy": fy, "cx": cx, "cy": cy, "focals": [fx] * n_frames,
                 "H": shape[0], "W": shape[1], "rectified": True,
                 "source_cam": args.cam, "balance": args.balance}, pkl_path)
    print(f"pinhole intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} -> {pkl_path}")

    if args.masks_root:
        out_h5 = osp.join(args.out_dir, f"{seq}_masks_k{kid}.h5")
        n_masks = rectify_masks(args.masks_root, seq, kid, out_h5, map1, map2)
        print(f"warped {n_masks} masks -> {out_h5}")
    else:
        print("no --masks_root given; masks not warped (re-run SAM3 on the "
              "rectified clip instead)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
