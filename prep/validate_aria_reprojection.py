# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Check the Aria camera model against ball positions the exo views already fixed.

aria_camera.py cannot verify itself. Its round trip is exact by construction --
project and unproject share one forward model, so a wrong distortion convention
inverts perfectly and still puts every ray a degree off. Rays a degree off still
intersect; they intersect in the wrong place, and the result looks like a
plausible measurement.

Three things have to be right before the ego view can contribute a ray: the
distortion convention, the extrinsics convention, and the frame offset between
the trimmed clip and the full take. This tests all three at once, against data
that does not depend on any of them -- ball positions triangulated from the
GoPros alone. Projecting those into the ego image must land on the ball in the
ego mask.

    python prep/validate_aria_reprojection.py \\
        --xyz bball_xyz.npz --masks_root sam3masks \\
        --calib <trajectory>/online_calibration.jsonl \\
        --extrinsics <trajectory>/aria_extrinsics.json --offset 354

The error's relationship to image radius is the useful part when it fails. A
constant offset across the image points at the extrinsics or the frame offset; an
error that grows toward the edges points at the distortion convention, since
that is where the models diverge.
"""
import argparse
import os
import os.path as osp
import sys

import h5py
import numpy as np

sys.path.append(os.getcwd())

from prep.aria_camera import (load_extrinsics, load_rgb_intrinsics, project,
                              scale_intrinsics)


def parse_args():
    """Parse the triangulated points, ego masks, calibration and frame offset."""
    parser = argparse.ArgumentParser(
        description="Validate the Aria model against exo-triangulated points")
    parser.add_argument("--xyz", required=True,
                        help=".npz from triangulate_object.py")
    parser.add_argument("--masks_root", required=True,
                        help="directory holding <ego>_masks_k0.h5")
    parser.add_argument("--ego_name", default="aria02_214-1",
                        help="ego stream name used for the mask file and group")
    parser.add_argument("--calib", required=True, help="online_calibration.jsonl")
    parser.add_argument("--extrinsics", required=True, help="aria_extrinsics.json")
    parser.add_argument("--offset", type=int, required=True,
                        help="index in the full take of clip frame 0 "
                             "(find_trim_offset.py)")
    parser.add_argument("--kid", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0,
                        help="stop after this many frames (0 = all)")
    return parser.parse_args()


def load_ego_centroids(masks_root, ego_name, kid, wanted):
    """Return {take frame index: (u, v) centroid, (h, w)} for the object masks.

    Only the requested frames are read. The ego mask file is several gigabytes,
    and pulling all 1922 frames to use a hundred would dominate the runtime.

    Raises:
        SystemExit: if the file or its group is missing.
    """
    path = osp.join(masks_root, f"{ego_name}_masks_k{kid}.h5")
    if not osp.isfile(path):
        raise SystemExit(f"ERROR: no ego masks at {path}")
    centroids, shape = {}, None
    with h5py.File(path, "r") as f:
        group = f[ego_name] if ego_name in f else f
        available = set(group.keys())
        for idx in sorted(wanted):
            key = f"{idx:06d}-k{kid}.obj_rend_mask.png"
            if key not in available:
                continue
            mask = group[key][:].astype(bool)
            shape = mask.shape
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                continue
            centroids[idx] = (float(xs.mean()), float(ys.mean()))
    if shape is None:
        raise SystemExit(
            f"ERROR: no object masks found in {path}. Checked keys like "
            f"'{min(wanted):06d}-k{kid}.obj_rend_mask.png'")
    return centroids, shape


def main():
    """Project triangulated points into the ego view and report the error."""
    args = parse_args()

    data = np.load(args.xyz)
    positions = {int(f): xyz for f, xyz in zip(data["frames"], data["xyz"])}
    print(f"triangulated positions: {len(positions)} clip frames")
    if args.max_frames:
        keep = sorted(positions)[:args.max_frames]
        positions = {k: positions[k] for k in keep}

    take_frames = {clip + args.offset for clip in positions}
    centroids, shape = load_ego_centroids(args.masks_root, args.ego_name,
                                          args.kid, take_frames)
    height, width = shape
    print(f"ego masks: {len(centroids)} of {len(take_frames)} frames have the "
          f"object, at {width}x{height}")
    if not centroids:
        raise SystemExit("ERROR: the ego view has no object mask on any frame "
                         "the exo views triangulated; nothing to compare")

    params = scale_intrinsics(load_rgb_intrinsics(args.calib), width, height)
    ext = load_extrinsics(args.extrinsics)

    rows = []
    for clip_idx in sorted(positions):
        take_idx = clip_idx + args.offset
        if take_idx not in centroids or take_idx not in ext:
            continue
        R_cw, t_cw = ext[take_idx]
        p_cam = R_cw @ np.asarray(positions[clip_idx]) + t_cw
        if p_cam[2] <= 0:
            continue
        uv = project(p_cam[None, :], params)[0]
        obs = np.array(centroids[take_idx])
        err = float(np.linalg.norm(uv - obs))
        radius = float(np.linalg.norm(obs - np.array([params[1], params[2]])))
        rows.append((clip_idx, obs, uv, err, radius))

    if not rows:
        raise SystemExit("ERROR: no frame had a triangulated point, an ego mask "
                         "and an extrinsic together")

    print()
    print(f"  {'clip':>5} {'observed':>17} {'projected':>17} {'err_px':>8} "
          f"{'radius':>8}")
    print("  " + "-" * 62)
    for clip_idx, obs, uv, err, radius in rows[:: max(1, len(rows) // 15)]:
        print(f"  {clip_idx:>5} ({obs[0]:>7.1f},{obs[1]:>7.1f}) "
              f"({uv[0]:>7.1f},{uv[1]:>7.1f}) {err:>8.1f} {radius:>8.1f}")

    err = np.array([r[3] for r in rows])
    rad = np.array([r[4] for r in rows])
    print()
    print(f"  {len(rows)} frames compared")
    print(f"  reprojection error: median {np.median(err):.1f} px, "
          f"mean {err.mean():.1f}, worst {err.max():.1f}")

    # A ball spanning tens of pixels means a centroid is only good to a few, so
    # single-digit error is as close as this can come. Anything much larger has
    # a structure worth reading rather than a magnitude worth quoting.
    if np.median(err) < 15:
        print("  -> PASSES. The distortion convention, the extrinsics and the "
              "frame offset are all consistent with the exo views. The ego "
              "camera can contribute rays.")
        return 0

    if len(rows) > 3:
        corr = float(np.corrcoef(rad, err)[0, 1])
        print(f"  error-vs-radius correlation: {corr:+.2f}")
        if corr > 0.5:
            print("  -> FAILS, and the error grows toward the image edges, "
                  "which is where distortion models differ. Suspect the "
                  "tangential/thin-prism composition in aria_camera.project.")
        else:
            print("  -> FAILS, but the error does not grow with radius, so it "
                  "is not the distortion model. Suspect the frame offset "
                  "(a constant lag shows up as a constant miss) or the "
                  "extrinsics convention.")
    else:
        print("  -> FAILS, with too few frames to say why.")
    print("  Do NOT add the ego view until this passes; a ray that is wrong by "
          "a degree still intersects the others and moves the answer.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
