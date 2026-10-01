# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Rectify an Aria ego clip (and its masks) to a true pinhole camera.

The ego view is where a handheld object is large, sharp and unoccluded, so it
is the view to track the object in. But every monocular stage -- UniDepth,
FoundationPose, the silhouette losses -- assumes a pinhole camera, and Aria
RGB is a FisheyeRadTanThinPrism camera stored rotated 90 degrees from its
calibration. prep/rectify_fisheye.py does this job for the GoPros with
cv2.fisheye; this does it for the Aria model in prep/aria_camera.py.

The output is in the CALIBRATION's orientation, not the stored video's. That
makes the rectified camera's axes exactly the calibrated camera's, so a pose
FoundationPose estimates in it can be carried into any other camera through
aria_extrinsics.json with no extra rotation to get right (prep/ego_poses_to_cam.py).
The price is a sideways-looking image, which nothing downstream minds: there is
no human to be upright for, and the object is tracked by its mesh.

Outputs, in --out_dir, named for --out_seq so the monocular stages treat the
clip as a sequence of their own:
    <out_seq>.0.color.mp4       rectified clip
    <out_seq>.0.color.pkl       its pinhole intrinsics (what unidepth_behave.py
                                and BaseBehaveVideoData read for a wild video)
    <out_seq>_masks_k0.h5       the masks, warped with the same map

--out_seq should keep the Date_Sub_object_action shape the loaders parse (the
object name is split('_')[2]); scripts/slurm_ego_object.sh uses <seq>-ego.

Run in an env with cv2 and imageio (newcari4d), from the repo root.

Usage:
    python prep/rectify_aria.py --video work/<seq>/masks/trimmed_vids/aria01_214-1-ego.0.color.mp4 \\
        --calib <trajectory>/online_calibration.jsonl \\
        --masks_root work/<seq>/masks --out_dir work/<seq>/rect-ego --out_seq <seq>-ego
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
from prep.aria_camera import (load_rgb_intrinsics, project, scale_intrinsics,
                              video_to_calib_pixels)
from prep.rectify_fisheye import rectify_video


def parse_args():
    """Parse the ego clip, its calibration, the masks and where to write."""
    parser = argparse.ArgumentParser(
        description="Undistort an Aria ego clip + masks to a true pinhole camera")
    parser.add_argument("--video", required=True,
                        help="the ego clip as stored, <name>.0.color.mp4 (rotated, fisheye)")
    parser.add_argument("--calib", required=True, help="online_calibration.jsonl")
    parser.add_argument("--masks_root", default=None,
                        help="directory holding <name>_masks_k0.h5 for the clip")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_seq", default=None,
                        help="name the outputs for this sequence (default: the clip's own name)")
    parser.add_argument("--rotate", type=int, default=90,
                        help="rotation of the stored video against the calibration "
                             "(default: 90, what Aria RGB needs)")
    parser.add_argument("--focal_scale", type=float, default=1.0,
                        help="pinhole focal length as a multiple of the fisheye's f. "
                             "1.0 keeps the centre at native resolution and crops the "
                             "fisheye's outer field; below 1 keeps more field, coarser")
    return parser.parse_args()


def build_maps(params, size, rotate, focal_scale):
    """Return (K, map_x, map_y) taking pinhole pixels to stored-video pixels.

    For every target pixel: the pinhole ray in the calibration camera's frame,
    projected through the fisheye model into the calibration's pixel frame,
    then rotated into the stored video's pixel frame, which is where the
    source pixels actually are. Rays the fisheye cannot see (behind, or past
    its field) map to -1, which cv2.remap renders black.
    """
    f = float(params[0]) * focal_scale
    cx = cy = (size - 1) / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    us, vs = np.meshgrid(np.arange(size, dtype=np.float64),
                         np.arange(size, dtype=np.float64))
    rays = np.stack([(us.ravel() - cx) / f, (vs.ravel() - cy) / f,
                     np.ones(size * size)], axis=1)
    calib_px = project(rays, params)
    video_px = video_to_calib_pixels(calib_px, size, (360 - rotate) % 360)
    bad = ~np.isfinite(video_px).all(axis=1)
    bad |= (video_px < 0).any(axis=1) | (video_px > size - 1).any(axis=1)
    video_px[bad] = -1.0
    map_x = video_px[:, 0].reshape(size, size).astype(np.float32)
    map_y = video_px[:, 1].reshape(size, size).astype(np.float32)
    print(f"pinhole f={f:.1f} c=({cx:.1f}, {cy:.1f}); "
          f"{(~bad).mean() * 100:.1f}% of the target image sees source pixels")
    return K, map_x, map_y


def rectify_masks(masks_root, seq, out_seq, kid, out_path, map_x, map_y):
    """Warp every mask of the clip with the same map, under the new sequence name.

    The dataset keys carry only the frame and camera (<frame>-k<kid>.*.png),
    so renaming the group is all the renaming there is.
    """
    src = osp.join(masks_root, f"{seq}_masks_k{kid}.h5")
    if not osp.isfile(src):
        raise SystemExit(f"ERROR: no mask file at {src}")
    n = 0
    with h5py.File(src, "r") as fin, h5py.File(out_path, "w") as fout:
        grp_in = fin[seq] if seq in fin else fin
        grp_out = fout.create_group(out_seq)
        for key in grp_in:
            mask = grp_in[key][:].astype(np.uint8)
            warped = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST) > 0
            grp_out.create_dataset(key, data=warped, compression="gzip")
            n += 1
    return n


def main():
    """Rectify the clip, warp its masks, write the pinhole intrinsics."""
    args = parse_args()
    base = osp.basename(args.video)
    seq, kid = base.split(".")[0], int(base.split(".")[1])
    out_seq = args.out_seq or seq
    os.makedirs(args.out_dir, exist_ok=True)

    probe = imageio.get_reader(args.video)
    first = probe.get_data(0)
    probe.close()
    H, W = first.shape[:2]
    if H != W:
        raise SystemExit(f"ERROR: expected a square Aria frame, got {W}x{H}")

    params = scale_intrinsics(load_rgb_intrinsics(args.calib), W, H)
    K, map_x, map_y = build_maps(params, W, args.rotate, args.focal_scale)

    out_video = osp.join(args.out_dir, f"{out_seq}.{kid}.color.mp4")
    n_frames, fps, shape = rectify_video(args.video, out_video, map_x, map_y)
    print(f"rectified {n_frames} frames @ {fps} fps -> {out_video}  ({shape[1]}x{shape[0]})")

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    pkl_path = out_video.replace(".color.mp4", ".color.pkl")
    joblib.dump({"fx": fx, "fy": fy, "cx": cx, "cy": cy, "focals": [fx] * n_frames,
                 "H": shape[0], "W": shape[1], "rectified": True,
                 "source_cam": seq, "model": "fisheye624", "rotate": args.rotate,
                 "orientation": "calibration"}, pkl_path)
    print(f"pinhole intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} -> {pkl_path}")

    if args.masks_root:
        out_h5 = osp.join(args.out_dir, f"{out_seq}_masks_k{kid}.h5")
        n_masks = rectify_masks(args.masks_root, seq, out_seq, kid, out_h5, map_x, map_y)
        print(f"warped {n_masks} masks -> {out_h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
