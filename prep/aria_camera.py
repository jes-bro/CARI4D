# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Aria RGB camera model, so the ego view can join exo triangulation.

The ego camera is the one that sees a dribbled ball best: it is metres away
rather than tens, and has no player's legs between it and the floor, which is
where the exo views lose the ball and where the tracked height is worst.

Aria RGB uses FisheyeRadTanThinPrism -- 15 parameters, not the 4-coefficient
Kannala-Brandt that cv2.fisheye implements for the GoPros. Its layout, from
online_calibration.jsonl:

    [0]     f          single isotropic focal length
    [1:3]   cx, cy     principal point
    [3:9]   k1..k6     radial, in powers of theta^2
    [9:11]  p1, p2     tangential
    [11:15] s1..s4     thin prism

CORRECTNESS WARNING. The composition order of the tangential and thin-prism
terms is a convention, and a wrong one produces geometry that looks reasonable
and is not -- rays off by a degree still intersect, just in the wrong place. A
round trip only proves this file agrees with itself. Use validate_against_exo()
before trusting any ego ray: it projects points triangulated from the GoPros
into the ego image and compares against the ego mask centroids, which fails
loudly if the convention, the extrinsics or the frame offset is wrong.
"""
import json
import os
import os.path as osp
import sys

import numpy as np

sys.path.append(os.getcwd())


def load_rgb_intrinsics(calib_path):
    """Return the 15 FisheyeRadTanThinPrism parameters for the RGB camera.

    Reads the first record of online_calibration.jsonl. The file carries one
    record per timestamp because Aria re-estimates calibration online, but the
    RGB intrinsics are stable over a take; the extrinsics, which do move, come
    from aria_extrinsics.json instead.

    Raises:
        SystemExit: if the file, the camera, or the expected model is missing.
    """
    if not osp.isfile(calib_path):
        raise SystemExit(f"ERROR: no calibration at {calib_path}")
    with open(calib_path) as f:
        record = json.loads(f.readline())
    cams = record.get("CameraCalibrations", [])
    rgb = [c for c in cams if str(c.get("Label", "")).lower().endswith("rgb")]
    if not rgb:
        raise SystemExit(f"ERROR: no camera-rgb in {calib_path}; "
                         f"found {[c.get('Label') for c in cams]}")
    proj = rgb[0]["Projection"]
    if proj.get("Name") != "FisheyeRadTanThinPrism":
        raise SystemExit(
            f"ERROR: camera-rgb uses {proj.get('Name')}; this implements "
            f"FisheyeRadTanThinPrism only, and treating another model as one "
            f"would give plausible but wrong geometry")
    params = np.asarray(proj["Params"], dtype=np.float64)
    if params.size != 15:
        raise SystemExit(f"ERROR: expected 15 parameters, got {params.size}")
    return params


def scale_intrinsics(params, width, height, full_width=2880, full_height=2880):
    """Rescale focal and principal point from native resolution to the video's.

    The frame-aligned ego video is 1408x1408 against a 2880x2880 native sensor,
    so the calibration does not apply unscaled. Distortion coefficients are
    dimensionless in normalised coordinates and are left alone.

    Raises:
        SystemExit: on a non-square rescale, since a single isotropic focal
            length cannot represent one and the result would be silently skewed.
    """
    sx, sy = width / full_width, height / full_height
    if abs(sx - sy) > 1e-6:
        raise SystemExit(
            f"ERROR: {width}x{height} from {full_width}x{full_height} needs "
            f"different scales per axis ({sx:.4f} vs {sy:.4f}), which a single "
            f"focal length cannot express")
    out = params.copy()
    out[0] *= sx
    out[1] *= sx
    out[2] *= sy
    return out


def project(points_cam, params):
    """Project camera-frame points to pixels through FisheyeRadTanThinPrism.

    Args:
        points_cam: (N, 3) points in the camera's frame, +Z forward.
        params: the 15 parameters, already scaled to the image being used.

    Returns:
        (N, 2) pixel coordinates. Points at or behind the centre of projection
        return NaN rather than a wrapped-around pixel, which would otherwise
        look like a valid observation on the far side of the image.
    """
    p = np.atleast_2d(np.asarray(points_cam, dtype=np.float64))
    f, cx, cy = params[0], params[1], params[2]
    k = params[3:9]
    tan = params[9:11]
    prism = params[11:15]

    z = p[:, 2]
    ab = np.full((len(p), 2), np.nan)
    good = z > 1e-9
    ab[good] = p[good, :2] / z[good, None]

    r = np.linalg.norm(ab, axis=1)
    theta = np.arctan(r)
    th2 = theta ** 2
    radial = np.ones_like(theta)
    acc = th2.copy()
    for ki in k:
        radial += ki * acc
        acc = acc * th2
    # theta * radial / r is the fisheye compression; at r = 0 the limit is 1.
    scale = np.where(r > 1e-9, theta * radial / np.maximum(r, 1e-12), 1.0)
    xr = ab * scale[:, None]

    # Tangential, Brown-Conrady form applied to the fisheye-compressed
    # coordinates rather than the raw ones -- this is the ordering the model's
    # name implies (RadTan, then ThinPrism) and the part most worth validating.
    rsq = (xr ** 2).sum(axis=1)
    x, y = xr[:, 0], xr[:, 1]
    dx = 2 * tan[0] * x * y + tan[1] * (rsq + 2 * x ** 2)
    dy = tan[0] * (rsq + 2 * y ** 2) + 2 * tan[1] * x * y
    uv = xr + np.stack([dx, dy], axis=1)

    # Thin prism, in the same radius measured after tangential.
    rsq2 = (uv ** 2).sum(axis=1)
    uv[:, 0] += prism[0] * rsq2 + prism[1] * rsq2 ** 2
    uv[:, 1] += prism[2] * rsq2 + prism[3] * rsq2 ** 2

    return np.stack([f * uv[:, 0] + cx, f * uv[:, 1] + cy], axis=1)


def unproject(pixels, params, iters=20):
    """Invert project(), returning unit ray directions in the camera frame.

    The model has no closed-form inverse, so this runs Gauss-Newton on the
    forward model with a numerical Jacobian, starting from the pinhole guess.
    Iterating the forward model means the inverse cannot drift away from
    whatever the forward model actually is, correct or not -- which keeps the
    two consistent and leaves validation to compare against reality.

    Returns:
        (N, 3) unit directions with +Z forward.
    """
    px = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
    f, cx, cy = params[0], params[1], params[2]
    ab = np.stack([(px[:, 0] - cx) / f, (px[:, 1] - cy) / f], axis=1)

    eps = 1e-6
    for _ in range(iters):
        cur = project(np.concatenate([ab, np.ones((len(ab), 1))], axis=1), params)
        residual = cur - px
        if np.nanmax(np.abs(residual)) < 1e-6:
            break
        jac = np.zeros((len(ab), 2, 2))
        for axis in range(2):
            bumped = ab.copy()
            bumped[:, axis] += eps
            moved = project(
                np.concatenate([bumped, np.ones((len(ab), 1))], axis=1), params)
            jac[:, :, axis] = (moved - cur) / eps
        det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
        safe = np.abs(det) > 1e-12
        step = np.zeros_like(ab)
        step[safe, 0] = (jac[safe, 1, 1] * residual[safe, 0]
                         - jac[safe, 0, 1] * residual[safe, 1]) / det[safe]
        step[safe, 1] = (-jac[safe, 1, 0] * residual[safe, 0]
                         + jac[safe, 0, 0] * residual[safe, 1]) / det[safe]
        ab -= step

    rays = np.concatenate([ab, np.ones((len(ab), 1))], axis=1)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def load_extrinsics(path):
    """Return {frame index: (R_cw, t_cw)} from aria_extrinsics.json.

    The stored 3x4 is world-to-camera: its implied centre, -R^T t, tracks the
    device position in closed_loop_trajectory.csv to within a few centimetres,
    which is the RGB camera's own offset from the device origin. That is already
    the convention triangulate_object.py builds for the GoPros, so these drop
    straight in.

    Raises:
        SystemExit: if the file is missing or malformed.
    """
    if not osp.isfile(path):
        raise SystemExit(f"ERROR: no extrinsics at {path}")
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for key, mat in raw.items():
        m = np.asarray(mat, dtype=np.float64)
        if m.shape != (3, 4):
            raise SystemExit(f"ERROR: frame {key} is {m.shape}, expected (3, 4)")
        out[int(key)] = (m[:, :3], m[:, 3])
    return out


def camera_dict(params, R_cw, t_cw):
    """Package one frame's Aria pose in the form triangulate_object.py expects.

    Carries model='fisheye624' so pixel_to_ray can branch; the GoPro entries
    carry no such key and keep their Kannala-Brandt path.
    """
    return {"model": "fisheye624", "params": params, "R_cw": R_cw, "t_cw": t_cw,
            "centre": -R_cw.T @ t_cw}


def self_test(params, width=1408, height=1408, samples=200, seed=0):
    """Check that unproject inverts project across the image.

    This proves internal consistency ONLY. A wrong distortion convention
    round-trips perfectly, because both directions use the same wrong forward
    model -- which is why validate_against_exo exists.

    Returns:
        Worst round-trip error in pixels.
    """
    rng = np.random.default_rng(seed)
    px = np.stack([rng.uniform(0, width, samples),
                   rng.uniform(0, height, samples)], axis=1)
    rays = unproject(px, params)
    back = project(rays, params)
    err = np.linalg.norm(back - px, axis=1)
    finite = np.isfinite(err)
    return float(np.nanmax(err[finite])) if finite.any() else float("nan")


def main():
    """Load the calibration, scale it, and report the round-trip error."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Load and self-test the Aria RGB camera model")
    parser.add_argument("--calib", required=True, help="online_calibration.jsonl")
    parser.add_argument("--extrinsics", default=None, help="aria_extrinsics.json")
    parser.add_argument("--width", type=int, default=1408)
    parser.add_argument("--height", type=int, default=1408)
    args = parser.parse_args()

    native = load_rgb_intrinsics(args.calib)
    print(f"native params: f={native[0]:.2f} c=({native[1]:.2f}, {native[2]:.2f})")
    params = scale_intrinsics(native, args.width, args.height)
    print(f"scaled to {args.width}x{args.height}: f={params[0]:.2f} "
          f"c=({params[1]:.2f}, {params[2]:.2f})")

    worst = self_test(params, args.width, args.height)
    print(f"round trip: worst error {worst:.4f} px over 200 samples")
    if worst > 0.1:
        print("  the inverse is not converging; raise iters before using this")
    else:
        print("  project and unproject agree -- but this says nothing about "
              "whether the distortion convention is right. Run the reprojection "
              "check against exo-triangulated points before trusting a ray.")

    if args.extrinsics:
        ext = load_extrinsics(args.extrinsics)
        idx = sorted(ext)
        print(f"extrinsics: {len(idx)} frames, {idx[0]}..{idx[-1]}")
        R, t = ext[idx[0]]
        print(f"  frame {idx[0]} centre {np.round(-R.T @ t, 3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
