# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Write triangulated object distances into a depth video's object region.

FoundationPose seeds its pose from the median depth inside the object mask, and
on a small distant object monocular depth cannot supply that: the egoexo4d
basketball read a median of 6.47m inside a range of 6.15-15.94m, and rejecting
the background contamination still left it uninformative.

triangulate_object.py recovers the same quantity geometrically, from masks in
two or more calibrated views, to within centimetres. This puts that number
where the pipeline already looks for it, so nothing downstream needs to change:
FoundationPose keeps reading depth through the same interface, and the depth
simply stops being wrong.

Only pixels inside the object mask are touched, on frames the triangulation
covers. Everything else -- the human, the background, frames with no
triangulated position -- is left exactly as it was, so the human alignment that
stage 4 established is undisturbed.

By default the object's depth is written as a constant, the distance to the
triangulated point. That is exact for what guess_translation reads (the median)
and approximate across the object's own extent, which is what --sphere_diameter
refines when the object really is a sphere.

Usage:
    python prep/inject_object_depth.py --video <aligned clip>.mp4 \\
        --masks_root <dir> --xyz bball_xyz.npz \\
        --calib <trajectory>/gopro_calibs.csv --cam cam04
"""
import argparse
import os
import os.path as osp
import shutil
import sys

import h5py
import numpy as np

sys.path.append(os.getcwd())

from prep.run_hy3d_recon import extract_seq_name
from prep.triangulate_object import read_calibration


def parse_args():
    """Parse the clip, masks, triangulated positions and camera calibration."""
    parser = argparse.ArgumentParser(
        description="Write triangulated object distances into a depth video")
    parser.add_argument("--video", required=True,
                        help="the clip whose .depth-reg.mp4 sibling is rewritten")
    parser.add_argument("--masks_root", required=True,
                        help="directory holding <seq>_masks_k<kid>.h5")
    parser.add_argument("--xyz", required=True,
                        help=".npz from triangulate_object.py")
    parser.add_argument("--calib", required=True, help="gopro_calibs.csv")
    parser.add_argument("--cam", required=True,
                        help="cam_uid of the view being rewritten, e.g. cam04")
    parser.add_argument("--seq", default=None,
                        help="sequence name (default: derived from --video)")
    parser.add_argument("--kid", type=int, default=0, help="camera id (default: 0)")
    parser.add_argument("--sphere_diameter", type=float, default=0.0,
                        help="if the object is a sphere of this diameter in metres, "
                             "write its curved front surface instead of a constant "
                             "depth. 0 writes a constant (default: 0)")
    parser.add_argument("--dry_run", action="store_true",
                        help="report what would change without writing")
    return parser.parse_args()


def load_masks(masks_root, seq_name, kid):
    """Return {frame index: bool object mask}.

    Raises:
        SystemExit: if the mask file or its sequence group is missing.
    """
    path = osp.join(masks_root, f"{seq_name}_masks_k{kid}.h5")
    if not osp.isfile(path):
        raise SystemExit(f"ERROR: no mask file at {path}")
    masks = {}
    with h5py.File(path, "r") as f:
        if seq_name not in f:
            raise SystemExit(f"ERROR: group '{seq_name}' not in {path}; "
                             f"found {list(f.keys())}")
        group = f[seq_name]
        for key in group:
            if key.endswith(f"-k{kid}.obj_rend_mask.png"):
                masks[int(key.split("-")[0])] = group[key][:].astype(bool)
    return masks


def camera_depth(point_world, cam):
    """Return the distance along the camera's optical axis to a world point.

    This is what a depth map stores -- the Z coordinate in camera space, not the
    Euclidean range from the camera centre. Confusing the two would bias every
    value by up to a few percent toward the image periphery.
    """
    return float((cam["R_cw"] @ np.asarray(point_world) + cam["t_cw"])[2])


def sphere_front_depth(mask, centre_depth, cam, diameter):
    """Depth of a sphere's visible surface across its mask, not a flat disc.

    A sphere's near surface bulges toward the camera by up to its radius, so a
    constant depth misplaces the surface by that much at the centre. Pixel
    offsets are converted to metres at the object's distance before the
    spherical cap is evaluated, and pixels beyond the sphere's radius fall back
    to the centre depth rather than producing NaN.

    Returns:
        Float array over the mask's bounding region, in metres.
    """
    radius = diameter / 2.0
    ys, xs = np.where(mask)
    cy, cx = ys.mean(), xs.mean()
    fx, fy = cam["K"][0, 0], cam["K"][1, 1]
    dx = (xs - cx) * centre_depth / fx
    dy = (ys - cy) * centre_depth / fy
    rho2 = dx * dx + dy * dy
    bulge = np.sqrt(np.clip(radius * radius - rho2, 0.0, None))
    return centre_depth - bulge


def read_depth(path, count):
    """Read a 16-bit depth video as metres.

    Raises:
        SystemExit: if it cannot be read, since silently proceeding would write
            a depth video built from nothing.
    """
    import videoio
    try:
        return [np.asarray(d, dtype=np.float64) / 1000.0
                for _, d in zip(range(count), videoio.uint16read(path))]
    except Exception as exc:
        raise SystemExit(f"ERROR: could not read {path} ({exc})")


def main():
    """Rewrite the object's depth from the triangulated positions."""
    args = parse_args()
    seq_name = args.seq or extract_seq_name(args.video)

    data = np.load(args.xyz)
    positions = {int(f): xyz for f, xyz in zip(data["frames"], data["xyz"])}
    print(f"triangulated positions: {len(positions)} frames")

    cams = read_calibration(args.calib, 1, 1)  # scale fixed up below
    if args.cam not in cams:
        raise SystemExit(f"ERROR: {args.cam} not in the calibration; found {sorted(cams)}")

    masks = load_masks(args.masks_root, seq_name, args.kid)
    print(f"object masks           : {len(masks)} frames")
    if not masks:
        raise SystemExit("ERROR: no object masks")
    height, width = next(iter(masks.values())).shape[:2]

    # Re-read at the true resolution now that the masks have told us what it is.
    cams = read_calibration(args.calib, width, height)
    cam = cams[args.cam]
    print(f"camera {args.cam}: fx={cam['K'][0, 0]:.1f} at {width}x{height}")

    depth_path = args.video.replace(".color.mp4", ".depth-reg.mp4")
    if not osp.isfile(depth_path):
        raise SystemExit(f"ERROR: no depth video at {depth_path}")
    frames = read_depth(depth_path, max(masks) + 1)
    print(f"depth video            : {len(frames)} frames from {osp.basename(depth_path)}")

    changed, comparisons = 0, []
    for idx in sorted(positions):
        if idx not in masks or idx >= len(frames):
            continue
        mask = masks[idx]
        if mask.shape != frames[idx].shape:
            raise SystemExit(
                f"ERROR: mask is {mask.shape} but depth is {frames[idx].shape}. They "
                f"must match -- a resized clip is the usual cause.")
        z = camera_depth(positions[idx], cam)
        if z <= 0:
            print(f"  frame {idx}: triangulated point is behind the camera, skipped")
            continue

        before = frames[idx][mask & (frames[idx] >= 0.001)]
        comparisons.append((idx, float(np.median(before)) if before.size else np.nan, z))

        if not args.dry_run:
            if args.sphere_diameter > 0:
                ys, xs = np.where(mask)
                frames[idx][ys, xs] = sphere_front_depth(
                    mask, z, cam, args.sphere_diameter)
            else:
                frames[idx][mask] = z
        changed += 1

    print()
    print(f"{'frame':>6} {'was_m':>9} {'now_m':>9} {'delta_m':>9}")
    print("-" * 36)
    for idx, was, now in comparisons[:: max(1, len(comparisons) // 20)]:
        delta = now - was if np.isfinite(was) else np.nan
        print(f"{idx:>6} {was:>9.2f} {now:>9.2f} {delta:>9.2f}")

    deltas = np.array([now - was for _, was, now in comparisons if np.isfinite(was)])
    if deltas.size:
        print()
        print(f"frames rewritten: {changed}")
        print(f"  depth changed by median {np.median(np.abs(deltas)):.2f} m, "
              f"max {np.abs(deltas).max():.2f} m")
        print(f"  frames left untouched: {len(masks) - changed} "
              f"(no triangulated position)")

    if args.dry_run:
        print("\nDry run, nothing written.")
        return 0

    backup = depth_path + ".orig"
    if not osp.exists(backup):
        shutil.copy2(depth_path, backup)
        print(f"\nbacked up the original to {osp.basename(backup)}")
    # uint16save, not videosave: the latter writes 8-bit and would quantise
    # millimetre depth into 256 levels, destroying exactly what this is fixing.
    import videoio
    videoio.uint16save(depth_path,
                       np.stack([np.clip(f * 1000.0, 0, 65535).astype(np.uint16)
                                 for f in frames]))
    print(f"wrote {depth_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
