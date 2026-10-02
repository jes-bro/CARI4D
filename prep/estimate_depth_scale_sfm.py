# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Metric scale for the ego depth map, from the Aria's own trajectory.

UniDepth on the ego clip returns depth with a free global scale, and it
varied from 0.48 to 0.91 of the truth across one clip's frames. The object's
known size can anchor it (prep/estimate_depth_scale.py) but needs a number
per object. This needs nothing: aria_extrinsics.json gives the RGB camera's
metric pose at every frame, so two frames a few centimetres apart are a
calibrated stereo pair. Static features -- stove, tiles, counter -- matched
between them triangulate to metres, and the ratio to the depth map at those
pixels is the correction, per frame.

Per frame, not global: the factor is measured on every sampled frame,
median-filtered, and interpolated to the rest, written as an .npz that
fp_hy3d_track.py --depth_scale_file applies frame by frame.

The person and object masks are excluded (dilated) from matching, since a
moving hand triangulates to nonsense. The rectified clip is in the
calibration's orientation, which is the frame the extrinsics describe, so
K [R_cw | t_cw] projects world points straight into it.

Usage:
    python prep/estimate_depth_scale_sfm.py --video work/<seq>/rect-ego/<seq>-ego.0.color.mp4 \\
        --masks_root work/<seq>/rect-ego --aria_extrinsics <trajectory>/aria_extrinsics.json \\
        --offset 5190 --out work/<seq>/rect-ego/depth_scale.npz
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
from prep.aria_camera import load_extrinsics


def parse_args():
    """Parse the clip, poses, masks, sampling and output."""
    parser = argparse.ArgumentParser(
        description="Per-frame metric scale for a monocular ego depth map, from the Aria trajectory")
    parser.add_argument("--video", required=True, help="rectified ego clip <seq>.<kid>.color.mp4")
    parser.add_argument("--aria_extrinsics", required=True, help="aria_extrinsics.json")
    parser.add_argument("--offset", type=int, required=True, help="take frame of clip frame 0")
    parser.add_argument("--masks_root", default=None,
                        help="directory holding <seq>_masks_k<kid>.h5; person and object are excluded")
    parser.add_argument("--kid", type=int, default=0)
    parser.add_argument("--stride", type=int, default=5, help="measure every Nth frame (default: 5)")
    parser.add_argument("--min_baseline", type=float, default=0.03,
                        help="smallest camera displacement to pair with, metres (default: 0.03)")
    parser.add_argument("--max_gap", type=int, default=45,
                        help="furthest frame to pair with (default: 45)")
    parser.add_argument("--max_reproj", type=float, default=2.0,
                        help="drop triangulated points reprojecting worse than this, px (default: 2)")
    parser.add_argument("--min_points", type=int, default=25,
                        help="frames with fewer good points are not measured (default: 25)")
    parser.add_argument("--no_depth", action="store_true",
                        help="only triangulate and report metric depths (no depth video needed)")
    parser.add_argument("--out", default=None, help="write per-frame factors to this .npz")
    parser.add_argument("--factor_only", action="store_true", help="print only the global factor")
    return parser.parse_args()


def read_depth_frames(video):
    """Load the depth-reg sibling of the clip as float metres, (T, H, W), or None."""
    path = video.replace(".color.mp4", ".depth-reg.mp4")
    if not osp.isfile(path):
        return None
    from behave_data.video_reader import ColorDepthController  # noqa: E402
    prefix = osp.basename(video).split(".")[0]
    ctrl = ColorDepthController(osp.join(osp.dirname(video), prefix), int(osp.basename(video).split(".")[1]))
    return ctrl


def load_exclusion(masks_root, prefix, kid, idx, shape, dilate=25):
    """A boolean mask of pixels NOT to match: person and object, grown a little."""
    if masks_root is None:
        return np.zeros(shape, bool)
    path = osp.join(masks_root, f"{prefix}_masks_k{kid}.h5")
    if not osp.isfile(path):
        return np.zeros(shape, bool)
    ex = np.zeros(shape, bool)
    with h5py.File(path, "r") as f:
        grp = f[prefix] if prefix in f else f
        for kind in ("person_mask", "obj_rend_mask"):
            key = f"{idx:06d}-k{kid}.{kind}.png"
            if key in grp:
                ex |= grp[key][()].astype(bool)
    if ex.any() and dilate > 0:
        ex = cv2.dilate(ex.astype(np.uint8), np.ones((dilate, dilate), np.uint8)) > 0
    return ex


def projection(K, R_cw, t_cw):
    """3x4 projection matrix for world points."""
    return K @ np.concatenate([R_cw, t_cw.reshape(3, 1)], axis=1)


def triangulate_pair(K, pose_a, pose_b, img_a, img_b, excl_a, excl_b, max_reproj, detector):
    """Match static features between two frames and triangulate them in metres.

    Returns (pixels in frame a, metric depth in frame a) for the points that
    reproject within max_reproj px in both frames, or empty arrays.
    """
    grey_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    grey_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    ka, da = detector.detectAndCompute(grey_a, (~excl_a).astype(np.uint8))
    kb, db = detector.detectAndCompute(grey_b, (~excl_b).astype(np.uint8))
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return np.zeros((0, 2)), np.zeros(0)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return np.zeros((0, 2)), np.zeros(0)
    pa = np.float64([ka[m.queryIdx].pt for m in good])
    pb = np.float64([kb[m.trainIdx].pt for m in good])

    Pa = projection(K, *pose_a)
    Pb = projection(K, *pose_b)
    X = cv2.triangulatePoints(Pa, Pb, pa.T, pb.T)
    X = (X[:3] / X[3]).T
    Xa = (pose_a[0] @ X.T).T + pose_a[1]
    Xb = (pose_b[0] @ X.T).T + pose_b[1]
    front = (Xa[:, 2] > 0.05) & (Xb[:, 2] > 0.05)
    ra = (K @ Xa.T).T
    rb = (K @ Xb.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        ra = ra[:, :2] / ra[:, 2:3]
        rb = rb[:, :2] / rb[:, 2:3]
    err = np.maximum(np.linalg.norm(ra - pa, axis=1), np.linalg.norm(rb - pb, axis=1))
    keep = front & np.isfinite(err) & (err < max_reproj)
    return pa[keep], Xa[keep, 2]


def main():
    """Measure the depth map's scale on sampled frames and write the per-frame factors."""
    args = parse_args()
    say = (lambda *a, **k: None) if args.factor_only else print

    prefix = osp.basename(args.video).split(".")[0]
    cam = joblib.load(args.video.replace(".mp4", ".pkl"))
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    ext = load_extrinsics(args.aria_extrinsics)

    reader = imageio.get_reader(args.video)
    frames = [np.asarray(f)[:, :, :3] for f in reader]
    reader.close()
    n = len(frames)
    H, W = frames[0].shape[:2]
    say(f"{n} frames {W}x{H}, f={K[0, 0]:.1f}")

    depth_ctrl = None if args.no_depth else read_depth_frames(args.video)
    if not args.no_depth and depth_ctrl is None:
        raise SystemExit("ERROR: no depth-reg.mp4 beside the clip; run UniDepth first or pass --no_depth")

    def pose(i):
        """World-to-camera (R, t) of clip frame i, or None."""
        return ext.get(i + args.offset)

    def centre(i):
        """Camera centre of clip frame i in the world."""
        R, t = pose(i)
        return -R.T @ t

    detector = cv2.ORB_create(nfeatures=4000)
    measured = []   # (frame, factor, n_points, metric median, depthmap median)
    for i in range(0, n, args.stride):
        if pose(i) is None:
            continue
        # The partner: the nearest later frame whose camera has moved enough
        # for the triangulation to be conditioned, within the gap allowed.
        j = None
        for gap in range(1, args.max_gap + 1):
            if i + gap >= n or pose(i + gap) is None:
                continue
            if np.linalg.norm(centre(i + gap) - centre(i)) >= args.min_baseline:
                j = i + gap
                break
        if j is None:
            continue
        excl_i = load_exclusion(args.masks_root, prefix, args.kid, i, (H, W))
        excl_j = load_exclusion(args.masks_root, prefix, args.kid, j, (H, W))
        px, z_metric = triangulate_pair(K, pose(i), pose(j), frames[i], frames[j],
                                        excl_i, excl_j, args.max_reproj, detector)
        if len(px) < args.min_points:
            continue
        if args.no_depth:
            measured.append((i, np.nan, len(px), float(np.median(z_metric)), np.nan))
            continue
        _, depth = depth_ctrl.get_closest_frame(float(i))
        depth = depth.astype(np.float32) / 1000.0
        u = np.clip(np.round(px[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(px[:, 1]).astype(int), 0, H - 1)
        z_map = depth[v, u]
        ok = z_map > 0.001
        if ok.sum() < args.min_points:
            continue
        ratio = z_metric[ok] / z_map[ok]
        measured.append((i, float(np.median(ratio)), int(ok.sum()),
                         float(np.median(z_metric[ok])), float(np.median(z_map[ok]))))

    if not measured:
        raise SystemExit("ERROR: no frame could be measured: too little camera motion, "
                         "or too few static features outside the masks")
    m = np.array(measured, dtype=np.float64)
    say(f"{len(m)} frames measured of {n // args.stride} sampled; "
        f"points per frame median {int(np.median(m[:, 2]))}")
    say(f"  triangulated static depth: median {np.nanmedian(m[:, 3]):.2f} m")
    if args.no_depth:
        return 0
    say(f"  depth map at those points: median {np.nanmedian(m[:, 4]):.2f} m")
    say(f"  per-frame factor: median {np.median(m[:, 1]):.3f}, 10th-90th pct "
        f"{np.percentile(m[:, 1], 10):.3f}-{np.percentile(m[:, 1], 90):.3f}")

    # Median-filter the measured factors against single bad pairs, then
    # interpolate to every frame; the ends hold their nearest value.
    f = m[:, 1].copy()
    if len(f) >= 5:
        f = np.array([np.median(f[max(0, k - 2):k + 3]) for k in range(len(f))])
    per_frame = np.interp(np.arange(n), m[:, 0], f)
    global_factor = float(np.median(f))
    say(f"depth scale factor: {global_factor:.4f} (global median; per-frame written)")
    if args.out:
        np.savez(args.out, frames=np.arange(n), factor=per_frame,
                 measured_frames=m[:, 0].astype(int), measured_factor=m[:, 1],
                 n_points=m[:, 2].astype(int), global_factor=global_factor)
        say(f"wrote {args.out}")
    if args.factor_only:
        print(f"{global_factor:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
