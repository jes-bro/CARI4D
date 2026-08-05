# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Triangulate an object's 3D position from its masks in two calibrated views.

Monocular depth cannot resolve a small distant object: on the egoexo4d
basketball it read a median of 6.47m inside a range of 6.15-15.94m, and
cleaning the contamination out still left it uninformative. But the 2D
observation is good -- FoundationPose tracked the ball's image position well
while its distance jittered -- and two calibrated views turn good 2D into
precise 3D with no depth network involved.

The precision is not marginal. Depth error from triangulation is roughly
Z^2 * d / (f * B). At Z = 7m, f = 367px, a 7.1m baseline and a 1px centroid
error that is about 1.5cm, against a 24cm ball. Monocular depth was wrong by
metres.

Ego-Exo4D's exo cameras are Kannala-Brandt fisheye, so a pixel does not
unproject by the pinhole formula -- that shortcut yields plausible but wrong
geometry, silently. cv2.fisheye.undistortPoints applies the right model.

Usage:
    python prep/triangulate_object.py --calib <trajectory>/gopro_calibs.csv \\
        --view cam04:<masks_root>:<seq> --view cam03:<masks_root>:<seq> \\
        --width 796 --height 448 --out object_xyz.npz

Each --view is <cam_uid>:<masks_root>:<sequence_name>. Masks are read at the
resolution given by --width/--height and the calibration is scaled to match.
"""
import argparse
import csv
import os
import os.path as osp
import sys

import cv2
import h5py
import numpy as np

sys.path.append(os.getcwd())


def parse_args():
    """Parse the calibration, the views to triangulate and the mask resolution."""
    parser = argparse.ArgumentParser(
        description="Triangulate an object from masks in two or more calibrated views")
    parser.add_argument("--calib", required=True,
                        help="gopro_calibs.csv from the take's trajectory/ directory")
    parser.add_argument("--view", action="append", required=True,
                        help="<cam_uid>:<masks_root>:<seq_name>, repeatable; at least two")
    parser.add_argument("--width", type=int, required=True,
                        help="width of the video the masks were computed on")
    parser.add_argument("--height", type=int, required=True,
                        help="height of the video the masks were computed on")
    parser.add_argument("--kid", type=int, default=0, help="camera id in the mask keys")
    parser.add_argument("--frame_offset", action="append", default=None,
                        help="per-view frame offset, same order as --view, for clips "
                             "trimmed to different ranges (default: 0 for every view)")
    parser.add_argument("--min_px", type=int, default=4,
                        help="skip frames whose mask is smaller in any view (default: 4)")
    parser.add_argument("--max_residual", type=float, default=10.0,
                        help="drop frames whose mean reprojection error exceeds this many "
                             "pixels; large residuals mean the views are not looking at "
                             "the same object (default: 10.0)")
    parser.add_argument("--out", default=None,
                        help="write the trajectory to this .npz")
    parser.add_argument("--max_rows", type=int, default=25,
                        help="rows of per-frame detail to print (default: 25)")
    return parser.parse_args()


def read_calibration(path, width, height):
    """Read gopro_calibs.csv and scale the intrinsics to the mask resolution.

    The columns are named *_world_cam, so the pose is the camera's placement in
    the world and projection needs its inverse. Scaling is done per axis rather
    than by one factor: 3840/796 and 2160/448 differ slightly, and using one for
    both would tilt every ray a little.

    Returns:
        {cam_uid: dict with K, dist, R_cw, t_cw, and the world-space centre}.

    Raises:
        SystemExit: if the file cannot be read or a camera is not fisheye.
    """
    if not osp.isfile(path):
        raise SystemExit(f"ERROR: no calibration at {path}")
    cams = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            uid = row["cam_uid"]
            full_w, full_h = float(row["image_width"]), float(row["image_height"])
            sx, sy = width / full_w, height / full_h
            if row["intrinsics_type"] != "KANNALABRANDTK3":
                raise SystemExit(
                    f"ERROR: {uid} uses {row['intrinsics_type']}; this script implements "
                    f"Kannala-Brandt only, and treating another model as one would give "
                    f"plausible but wrong geometry")
            K = np.array([[float(row["intrinsics_0"]) * sx, 0, float(row["intrinsics_2"]) * sx],
                          [0, float(row["intrinsics_1"]) * sy, float(row["intrinsics_3"]) * sy],
                          [0, 0, 1.0]])
            dist = np.array([float(row[f"intrinsics_{i}"]) for i in range(4, 8)])

            t_wc = np.array([float(row["tx_world_cam"]), float(row["ty_world_cam"]),
                             float(row["tz_world_cam"])])
            q = np.array([float(row["qx_world_cam"]), float(row["qy_world_cam"]),
                          float(row["qz_world_cam"]), float(row["qw_world_cam"])])
            R_wc = quaternion_to_matrix(q)
            # World-to-camera is the inverse of the stored camera-in-world pose.
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc

            cams[uid] = {"K": K, "dist": dist, "R_cw": R_cw, "t_cw": t_cw,
                         "centre": t_wc, "scale": (sx, sy)}
    return cams


def quaternion_to_matrix(q):
    """Convert an (x, y, z, w) quaternion to a rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_object_centroids(masks_root, seq_name, kid, min_px):
    """Return {frame index: (u, v, pixel count)} for the object's mask centroid.

    The centroid of a sphere's silhouette is its projected centre, so for a ball
    this is the physically right point. For an asymmetric object it is merely a
    consistent one, which gives a good trajectory and a slightly biased absolute
    position.

    Raises:
        SystemExit: if the mask file or its sequence group is missing.
    """
    path = osp.join(masks_root, f"{seq_name}_masks_k{kid}.h5")
    if not osp.isfile(path):
        raise SystemExit(f"ERROR: no mask file at {path}")
    out = {}
    with h5py.File(path, "r") as f:
        if seq_name not in f:
            raise SystemExit(f"ERROR: group '{seq_name}' not in {path}; "
                             f"found {list(f.keys())}")
        group = f[seq_name]
        for key in group:
            if not key.endswith(f"-k{kid}.obj_rend_mask.png"):
                continue
            mask = group[key][:]
            ys, xs = np.where(mask)
            if len(ys) < min_px:
                continue
            out[int(key.split("-")[0])] = (float(xs.mean()), float(ys.mean()), len(ys))
    return out


def pixel_to_ray(uv, cam):
    """Unproject a pixel to a unit ray in world coordinates.

    Uses the Kannala-Brandt model rather than the pinhole inverse; at the image
    periphery the two disagree substantially, and the exo cameras are wide.

    Returns:
        (origin, direction) with direction normalised.
    """
    pts = np.array([[[uv[0], uv[1]]]], dtype=np.float64)
    undist = cv2.fisheye.undistortPoints(pts, cam["K"], cam["dist"])
    x, y = float(undist[0, 0, 0]), float(undist[0, 0, 1])
    ray_cam = np.array([x, y, 1.0])
    ray_world = cam["R_cw"].T @ ray_cam
    return cam["centre"], ray_world / np.linalg.norm(ray_world)


def triangulate_rays(origins, directions):
    """Least-squares closest point to a set of rays.

    Two rays from real observations never meet exactly, so the point minimising
    the summed squared perpendicular distance is the estimate, and that residual
    is itself the check on whether the views agree.

    Returns:
        (point, mean perpendicular distance in metres).
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, directions):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    point = np.linalg.lstsq(A, b, rcond=None)[0]
    dists = [np.linalg.norm(np.cross(d, point - o)) for o, d in zip(origins, directions)]
    return point, float(np.mean(dists))


def reprojection_error(point, uv, cam):
    """Reproject a world point into a view and return the pixel error.

    This is what catches the failure that matters: if the two views tracked
    different objects, triangulation still returns a point, and only the
    reprojection residual reveals that it explains neither observation.
    """
    p_cam = cam["R_cw"] @ point + cam["t_cw"]
    if p_cam[2] <= 1e-6:
        return np.inf
    projected, _ = cv2.fisheye.projectPoints(
        p_cam.reshape(1, 1, 3), np.zeros(3), np.zeros(3), cam["K"], cam["dist"])
    return float(np.linalg.norm(projected.ravel() - np.asarray(uv)))


def main():
    """Triangulate the object across frames and report the geometry's consistency."""
    args = parse_args()
    if len(args.view) < 2:
        raise SystemExit("ERROR: at least two --view arguments are needed")

    cams = read_calibration(args.calib, args.width, args.height)
    offsets = [int(o) for o in (args.frame_offset or [])]
    offsets += [0] * (len(args.view) - len(offsets))

    views = []
    for spec, offset in zip(args.view, offsets):
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"ERROR: --view must be <cam_uid>:<masks_root>:<seq>, got {spec}")
        uid, masks_root, seq = parts
        if uid not in cams:
            raise SystemExit(f"ERROR: {uid} not in the calibration; "
                             f"found {sorted(cams)}")
        centroids = load_object_centroids(masks_root, seq, args.kid, args.min_px)
        print(f"{uid}: {len(centroids)} frames with an object mask, offset {offset}, "
              f"centre {np.round(cams[uid]['centre'], 2)}")
        views.append({"uid": uid, "cam": cams[uid], "centroids": centroids,
                      "offset": offset})

    baseline = np.linalg.norm(views[0]["cam"]["centre"] - views[1]["cam"]["centre"])
    print(f"baseline between the first two views: {baseline:.2f} m")

    shared = set(views[0]["centroids"])
    for v in views[1:]:
        shared &= {idx - v["offset"] + views[0]["offset"] for idx in v["centroids"]}
    shared = sorted(shared)
    if not shared:
        raise SystemExit("ERROR: no frame has an object mask in every view")
    print(f"frames observed in every view: {len(shared)}")

    rows = []
    for idx in shared:
        origins, directions, obs = [], [], []
        for v in views:
            key = idx - views[0]["offset"] + v["offset"]
            uv = v["centroids"][key][:2]
            o, d = pixel_to_ray(uv, v["cam"])
            origins.append(o)
            directions.append(d)
            obs.append((uv, v["cam"]))
        point, spread = triangulate_rays(origins, directions)
        errs = [reprojection_error(point, uv, cam) for uv, cam in obs]
        rows.append({"frame": idx, "xyz": point, "spread": spread,
                     "residual": float(np.mean(errs))})

    good = [r for r in rows if r["residual"] <= args.max_residual]
    print()
    print(f"{'frame':>6} {'X':>8} {'Y':>8} {'Z':>8} {'ray_gap_m':>10} {'reproj_px':>10}")
    print("-" * 54)
    step = max(1, len(rows) // args.max_rows)
    for r in rows[::step]:
        x, y, z = r["xyz"]
        print(f"{r['frame']:>6} {x:>8.2f} {y:>8.2f} {z:>8.2f} "
              f"{r['spread']:>10.3f} {r['residual']:>10.2f}")

    residuals = np.array([r["residual"] for r in rows])
    print()
    print(f"frames triangulated : {len(rows)}")
    print(f"  reprojection error: median {np.median(residuals):.2f} px, "
          f"90th pct {np.percentile(residuals, 90):.2f} px")
    print(f"  within {args.max_residual:g} px    : {len(good)}/{len(rows)}")
    if len(good) > 1:
        xyz = np.array([r["xyz"] for r in good])
        step_m = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        print(f"  motion per frame  : median {np.median(step_m):.3f} m, "
              f"max {step_m.max():.3f} m")

    if np.median(residuals) > args.max_residual:
        print("\nThe views do not agree. Either they are tracking different objects, "
              "the frame offsets are wrong, or the extrinsics are being applied "
              "backwards. A consistent geometry gives residuals of a pixel or two.")
    else:
        print("\nThe views agree, so the geometry is consistent and the triangulated "
              "positions are usable. Motion per frame should look like plausible "
              "object movement, not jumps of metres.")

    if args.out:
        np.savez(args.out,
                 frames=np.array([r["frame"] for r in good]),
                 xyz=np.array([r["xyz"] for r in good]),
                 residual=np.array([r["residual"] for r in good]))
        print(f"\nwrote {args.out} with {len(good)} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
