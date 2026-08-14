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

Each --view is <cam_uid>:<masks_root>:<sequence_name>. Each mask set's
resolution is read from its own H5 (a stored mask is an (H, W) array) and the
calibration is scaled per view to match, so mask sets of different resolutions
are simply different observations of the same geometry. The same camera may
appear in several --view entries (say a 448p and a 4K mask set): per frame the
camera contributes exactly one of them -- two observations from one camera
share a ray origin and add no 3D information -- and the one kept is whichever
agrees best with the other cameras, measured by the same reprojection residual
used to accept or reject the frame. Residuals are always expressed in
--width-scale pixels, whatever resolution the masks came from, so
--max_residual means one thing.
"""
import argparse
import csv
import itertools
import os
import os.path as osp
import sys

import cv2
import h5py
import numpy as np

sys.path.append(os.getcwd())


from prep.aria_camera import (camera_dict, load_extrinsics, load_rgb_intrinsics,
                              project as aria_project, scale_intrinsics,
                              unproject as aria_unproject, video_to_calib_pixels)


def parse_args():
    """Parse the calibration, the views to triangulate and the mask resolution."""
    parser = argparse.ArgumentParser(
        description="Triangulate an object from masks in two or more calibrated views")
    parser.add_argument("--calib", required=True,
                        help="gopro_calibs.csv from the take's trajectory/ directory")
    parser.add_argument("--view", action="append", required=True,
                        help="<cam_uid>:<masks_root>:<seq_name>, repeatable; at least two. "
                             "Mask resolution is read from the H5 itself. The same cam_uid "
                             "may appear with several mask sets; per frame the best-agreeing "
                             "one is used")
    parser.add_argument("--width", type=int, required=True,
                        help="reference width: residuals are reported in this pixel scale "
                             "regardless of each mask set's own resolution")
    parser.add_argument("--height", type=int, required=True,
                        help="reference height, paired with --width")
    parser.add_argument("--kid", type=int, default=0, help="camera id in the mask keys")
    parser.add_argument("--frame_offset", action="append", default=None,
                        help="per-view frame offset, same order as --view, for clips "
                             "trimmed to different ranges (default: 0 for every view)")
    parser.add_argument("--min_views", type=int, default=2,
                        help="views that must see the object for a frame to be "
                             "triangulated. Two is the minimum the geometry allows; "
                             "more is stricter but reduces coverage (default: 2)")
    parser.add_argument("--min_px", type=int, default=4,
                        help="skip frames whose mask is smaller in any view (default: 4)")
    parser.add_argument("--max_residual", type=float, default=10.0,
                        help="drop frames whose mean reprojection error exceeds this many "
                             "pixels; large residuals mean the views are not looking at "
                             "the same object (default: 10.0)")
    # The ego view is added separately because its pose changes every frame,
    # while a GoPro is bolted down and has one. It is also the view most worth
    # having: metres from the ball rather than tens, with no legs between it and
    # the floor, which is where the exo views lose it.
    parser.add_argument("--aria_masks_root", default=None,
                        help="add the Aria ego view; directory holding "
                             "<aria_name>_masks_k<kid>.h5")
    parser.add_argument("--aria_name", default="aria02_214-1",
                        help="ego stream name for the mask file and group")
    parser.add_argument("--aria_calib", default=None,
                        help="online_calibration.jsonl, for the RGB intrinsics")
    parser.add_argument("--aria_extrinsics", default=None,
                        help="aria_extrinsics.json, one pose per take frame")
    parser.add_argument("--aria_offset", type=int, default=0,
                        help="index in the full take of the reference view's "
                             "frame 0 (prep/find_trim_offset.py)")
    parser.add_argument("--aria_rotate", type=int, default=90,
                        help="rotation of the stored ego video against its "
                             "calibration (default: 90, what Aria RGB needs)")
    parser.add_argument("--aria_size", type=int, default=1408,
                        help="edge of the square ego video (default: 1408)")
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
    """Return ({frame index: (u, v, pixel count)}, (H, W)) for the object masks.

    The centroid of a sphere's silhouette is its projected centre, so for a ball
    this is the physically right point. For an asymmetric object it is merely a
    consistent one, which gives a good trajectory and a slightly biased absolute
    position.

    The (H, W) is the resolution the masks were computed at, read from the
    stored arrays themselves -- it is what the caller scales the calibration to.

    Raises:
        SystemExit: if the mask file, its sequence group, or any object mask
            is missing.
    """
    path = osp.join(masks_root, f"{seq_name}_masks_k{kid}.h5")
    if not osp.isfile(path):
        raise SystemExit(f"ERROR: no mask file at {path}")
    out, shape = {}, None
    with h5py.File(path, "r") as f:
        if seq_name not in f:
            raise SystemExit(f"ERROR: group '{seq_name}' not in {path}; "
                             f"found {list(f.keys())}")
        group = f[seq_name]
        for key in group:
            if not key.endswith(f"-k{kid}.obj_rend_mask.png"):
                continue
            mask = group[key][:]
            if shape is None:
                shape = mask.shape
            ys, xs = np.where(mask)
            if len(ys) < min_px:
                continue
            out[int(key.split("-")[0])] = (float(xs.mean()), float(ys.mean()), len(ys))
    if shape is None:
        raise SystemExit(f"ERROR: no object masks for {seq_name} in {path}")
    return out, shape


def pixel_to_ray(uv, cam):
    """Unproject a pixel to a unit ray in world coordinates.

    Uses the Kannala-Brandt model rather than the pinhole inverse; at the image
    periphery the two disagree substantially, and the exo cameras are wide.

    Returns:
        (origin, direction) with direction normalised.
    """
    if cam.get("model") == "fisheye624":
        ray_cam = aria_unproject(np.asarray(uv, dtype=np.float64)[None, :],
                                 cam["params"])[0]
    else:
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
    if cam.get("model") == "fisheye624":
        projected = aria_project(p_cam[None, :], cam["params"])[0]
    else:
        proj, _ = cv2.fisheye.projectPoints(
            p_cam.reshape(1, 1, 3), np.zeros(3), np.zeros(3), cam["K"], cam["dist"])
        projected = proj.ravel()
    return float(np.linalg.norm(projected - np.asarray(uv)))


def build_aria_view(args, kid):
    """Return a view entry for the ego camera, or None if it was not requested.

    Unlike a bolted-down GoPro the Aria moves, so this carries one camera per
    take frame rather than one camera. Mask pixels are mapped into the
    calibration's frame on the way in, since the stored video is rotated
    relative to it.

    Raises:
        SystemExit: if some but not all of the Aria inputs were given, rather
            than quietly proceeding without the view that was asked for.
    """
    supplied = [args.aria_masks_root, args.aria_calib, args.aria_extrinsics]
    if not any(supplied):
        return None
    if not all(supplied):
        raise SystemExit("ERROR: --aria_masks_root, --aria_calib and "
                         "--aria_extrinsics must be given together")

    params = scale_intrinsics(load_rgb_intrinsics(args.aria_calib),
                              args.aria_size, args.aria_size)
    ext = load_extrinsics(args.aria_extrinsics)
    raw, _ = load_object_centroids(args.aria_masks_root, args.aria_name, kid,
                                   args.min_px)
    centroids, cams = {}, {}
    for take_idx, (u, v, count) in raw.items():
        if take_idx not in ext:
            continue
        uv = video_to_calib_pixels((u, v), args.aria_size, args.aria_rotate)[0]
        # Keyed by reference-view frame so the main loop indexes it like any
        # other view; the take index only matters for looking up the pose.
        ref_idx = take_idx - args.aria_offset
        centroids[ref_idx] = (float(uv[0]), float(uv[1]), count)
        cams[ref_idx] = camera_dict(params, *ext[take_idx])
    print(f"aria ({args.aria_name}): {len(centroids)} frames with an object mask "
          f"and a pose, rotated {args.aria_rotate} deg into the calibration frame")
    return {"uid": "aria", "label": "aria", "cam": None, "cams_by_frame": cams,
            "centroids": centroids, "offset": 0,
            "err_scale": args.width / args.aria_size}


def main():
    """Triangulate the object across frames and report the geometry's consistency."""
    args = parse_args()
    if len(args.view) < 2:
        raise SystemExit("ERROR: at least two --view arguments are needed")

    offsets = [int(o) for o in (args.frame_offset or [])]
    offsets += [0] * (len(args.view) - len(offsets))

    # One calibration per mask resolution encountered; each view's masks tell
    # their own resolution, so nothing on the command line has to.
    cams_by_res = {}
    views = []
    for spec, offset in zip(args.view, offsets):
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"ERROR: --view must be <cam_uid>:<masks_root>:<seq>, got {spec}")
        uid, masks_root, seq = parts
        centroids, (mask_h, mask_w) = load_object_centroids(
            masks_root, seq, args.kid, args.min_px)
        if (mask_w, mask_h) not in cams_by_res:
            cams_by_res[(mask_w, mask_h)] = read_calibration(args.calib, mask_w, mask_h)
        cams = cams_by_res[(mask_w, mask_h)]
        if uid not in cams:
            raise SystemExit(f"ERROR: {uid} not in the calibration; "
                             f"found {sorted(cams)}")
        print(f"{uid} ({seq}, {mask_w}x{mask_h}): {len(centroids)} frames with an "
              f"object mask, offset {offset}, centre {np.round(cams[uid]['centre'], 2)}")
        views.append({"uid": uid, "label": seq, "cam": cams[uid], "cams_by_frame": None,
                      "centroids": centroids, "offset": offset,
                      "err_scale": args.width / mask_w})

    aria_view = build_aria_view(args, args.kid)
    if aria_view is not None:
        views.append(aria_view)

    baseline = np.linalg.norm(views[0]["cam"]["centre"] - views[1]["cam"]["centre"])
    print(f"baseline between the first two views: {baseline:.2f} m")

    # Triangulate from whichever views see the object, not from views that all
    # do. Requiring every view makes each extra camera shrink coverage, which is
    # backwards: different sightlines fail at different moments, so the union is
    # what more cameras buy. Two rays suffice, and beyond that the reprojection
    # residual becomes a stronger consistency check rather than a weaker one.
    #
    # Frames are those the reference view sees, since the point of this is to
    # supply depth for that view's sequence -- a frame where it has no object
    # mask has no object region to write into.
    rows, skipped = [], 0
    for idx in sorted(views[0]["centroids"]):
        # Candidate observations per physical camera. A camera with several
        # mask sets contributes exactly one -- two rays from the same origin
        # add no 3D information and make the least squares degenerate.
        by_uid, uid_order = {}, []
        for v in views:
            key = idx - views[0]["offset"] + v["offset"]
            if key not in v["centroids"]:
                continue
            uv = v["centroids"][key][:2]
            cam = v["cam"] if v["cams_by_frame"] is None else v["cams_by_frame"][key]
            o, d = pixel_to_ray(uv, cam)
            if v["uid"] not in by_uid:
                uid_order.append(v["uid"])
            by_uid.setdefault(v["uid"], []).append(
                {"uv": uv, "cam": cam, "o": o, "d": d,
                 "scale": v["err_scale"], "label": v["label"]})
        if len(by_uid) < args.min_views:
            skipped += 1
            continue
        # Which mask set is best for a camera on this frame is decided by the
        # geometry: try the combinations and keep the one whose triangulation
        # the contributing views agree on most, in reference-scale pixels.
        # The combination count is tiny (mask sets per camera is 2 or 3), but
        # cap it anyway and fall back to first-listed rather than blow up.
        n_combos = int(np.prod([len(by_uid[u]) for u in uid_order]))
        if n_combos > 256:
            combos = [tuple(by_uid[u][0] for u in uid_order)]
        else:
            combos = itertools.product(*(by_uid[u] for u in uid_order))
        best = None
        for combo in combos:
            point, spread = triangulate_rays([c["o"] for c in combo],
                                             [c["d"] for c in combo])
            errs = [reprojection_error(point, c["uv"], c["cam"]) * c["scale"]
                    for c in combo]
            mean_err = float(np.mean(errs))
            if best is None or mean_err < best["residual"]:
                best = {"frame": idx, "xyz": point, "spread": spread,
                        "residual": mean_err,
                        "views": [c["label"] for c in combo]}
        rows.append(best)

    if not rows:
        raise SystemExit(
            f"ERROR: no frame of {views[0]['uid']} was seen by {args.min_views}+ views. "
            f"Check the --frame_offset values -- an offset that is wrong by even a few "
            f"frames can remove the entire overlap.")
    print(f"frames the reference view sees   : {len(views[0]['centroids'])}")
    print(f"  triangulated ({args.min_views}+ views)      : {len(rows)}")
    print(f"  skipped, too few views         : {skipped}")
    counts = {}
    for r in rows:
        counts[len(r["views"])] = counts.get(len(r["views"]), 0) + 1
    print("  views per frame                : "
          + ", ".join(f"{n} views x{c}" for n, c in sorted(counts.items())))

    good = [r for r in rows if r["residual"] <= args.max_residual]
    print()
    print(f"{'frame':>6} {'X':>8} {'Y':>8} {'Z':>8} {'ray_gap_m':>10} {'reproj_px':>10}  views")
    print("-" * 64)
    step = max(1, len(rows) // args.max_rows)
    for r in rows[::step]:
        x, y, z = r["xyz"]
        print(f"{r['frame']:>6} {x:>8.2f} {y:>8.2f} {z:>8.2f} "
              f"{r['spread']:>10.3f} {r['residual']:>10.2f}  {','.join(r['views'])}")

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
