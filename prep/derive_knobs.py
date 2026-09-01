"""Derive the pipeline's object-dependent thresholds from the object itself.

Several knobs downstream are numbers somebody measured once, for a basketball,
at seven metres: how far to keep depth (ZFAR), how far the object may sit from
the person (DEPTH_HUMAN_BAND), how closely two views must agree to be believed
(--inlier_px). Carried to a different object they are wrong in ways that do not
announce themselves, and carried to a reviewer they read as a method that needs
hand-tuning per sequence.

None of them actually needs taste. Each is a physical quantity divided by
another physical quantity this pipeline already measures, so each can be
computed per clip:

  --inlier_px         a fraction of the object's apparent diameter. Two views
                      agreeing to within a third of the object's own width are
                      looking at the same thing; the tolerance has to scale
                      with how big the object appears, not sit at a constant
                      that happens to suit a 13-px ball.
  ZFAR                beyond the furthest the object is ever seen, with margin.
                      Its job is to cut background, and background is whatever
                      is further away than the object ever gets.
  DEPTH_HUMAN_BAND    the largest separation actually observed between the
                      object and the person, with margin. Its job is to reject
                      background bleeding into the object's mask, and the
                      object never travels further from its handler than it is
                      measured to travel.

Prints shell assignments, so a driver can `eval` it, and the reasoning to
stderr so a log records why a number was what it was.

Run from the repo root in the cari4d env.

Usage:
    # before triangulation -- needs only masks
    python prep/derive_knobs.py --masks_root work/<seq>/masks --seq <seq>

    # before tracking -- needs the triangulated object and human
    python prep/derive_knobs.py --object_xyz work/<seq>/geom/object_xyz.npz \\
        --human_j3d work/<seq>/geom/human_joints.npz \\
        --calib <take>/trajectory/gopro_calibs.csv --cam cam04
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.getcwd())

# A tolerance of a third of the object's apparent width. On the egoexo4d
# basketball -- 13 px across at the 796-px reference scale -- that reproduces
# the 4 px that was arrived at by hand.
INLIER_FRACTION = 0.3
INLIER_MIN, INLIER_MAX = 2.0, 25.0

# Margins past what was observed. Both are deliberately generous, because both
# thresholds fail asymmetrically: too loose merely keeps background that the
# object mask discards anyway, while too tight deletes the object itself and
# FoundationPose cannot initialise at all.
#
# They are also derived from TRIANGULATED frames only, and triangulation drops
# the frames where the views disagree -- which tend to be the fast ones, where
# the object is furthest or moving hardest. So the observed maximum is a lower
# bound on the true one, and the margin has to cover that too.
ZFAR_MARGIN = 2.0
ZFAR_MIN = 3.0

BAND_MARGIN = 1.5
BAND_MIN = 0.3

# Rotational symmetry, as the chamfer distance under rotation divided by the
# chamfer under a plain resample -- see symmetry_score. A symmetric object
# scores ~1: rotating it changes nothing that resampling would not have changed
# anyway. Measured on primitives: sphere 0.97, cylinder 2.46, pot with a handle
# 2.93, box 4.16, thin rod 23.9. Below the threshold the orientation is
# unobservable, so re-registering every frame costs nothing; above it,
# re-registration picks a different arbitrary orientation each time and nothing
# reports that it happened.
SYMMETRY_TOL = 1.5
SYMMETRY_ROTATIONS = 24

# Re-registration is only WORTH its cost when the object moves far enough
# between frames to leave the tracker's convergence basin. Measured as motion
# per frame in units of the object's own diameter.
MOTION_PER_DIAMETER = 0.25


def object_diameter_px(masks_root, seq, kid, width):
    """Median apparent diameter of the object, in reference-scale pixels.

    Taken from the mask's area rather than its bounding box: a bounding box
    grows when the mask sprouts a spur, while the area of the equivalent circle
    is what "how big does this look" actually means.
    """
    from prep.triangulate_object import load_object_centroids
    centroids, shape = load_object_centroids(masks_root, seq, kid, min_px=1)
    if not centroids:
        return None
    areas = np.array([c[2] for c in centroids.values()], dtype=float)
    diam_native = 2.0 * np.sqrt(np.median(areas) / np.pi)
    return float(diam_native * width / shape[1])


def camera_distances(xyz_world, calib, cam, width, height):
    """Distances from `cam` to each world point, in metres."""
    from prep.triangulate_object import read_calibration
    cams = read_calibration(calib, width, height)
    if cam not in cams:
        raise SystemExit(f"ERROR: {cam} not in {calib}")
    c = cams[cam]
    pts_cam = (c["R_cw"] @ np.asarray(xyz_world).T).T + c["t_cw"]
    return np.linalg.norm(pts_cam, axis=1)


def human_distances(npz):
    """Median distance to the person's joints per frame, in the camera frame.

    Uses joints_cam, which prep/triangulate_human.py writes when given
    --to_cam. Invalid joints are excluded rather than counted as the origin.
    """
    if "joints_cam" not in npz:
        raise SystemExit("ERROR: the human npz has no joints_cam; "
                         "re-run triangulate_human.py with --to_cam")
    jc, valid = npz["joints_cam"], npz["valid"]
    out = np.full(len(jc), np.nan)
    for i, (j, v) in enumerate(zip(jc, valid)):
        if v.any():
            out[i] = float(np.median(np.linalg.norm(j[v.astype(bool)], axis=1)))
    return out


def symmetry_score(mesh_file, n_rot=SYMMETRY_ROTATIONS, n_pts=1200, seed=0):
    """How much a rotation changes the object, relative to resampling it.

    Whether re-registering every frame is safe comes down to one question: can
    the object's orientation be observed at all? For a sphere it cannot -- every
    rotation maps the surface onto itself -- so FoundationPose picking a
    different orientation each frame costs nothing. For anything with structure
    the same setting produces a different arbitrary orientation every frame,
    silently.

    Measured by rotating the object's own surface points and asking whether they
    still land on it. The subtlety is that a rotated sample can never match
    exactly, because it is a DIFFERENT random set of points on the same surface
    -- so a perfect sphere still scores its own sampling density, and comparing
    that against an absolute threshold measures point count rather than shape.

    So the rotated chamfer is divided by the chamfer of an unrotated resample:
    the same sampling noise, without the rotation. A symmetric object lands at
    ~1.0 because rotating it changes nothing a resample would not have changed
    anyway. Anything with structure lands above.
    """
    import trimesh
    mesh = trimesh.load(mesh_file, process=False)
    rng = np.random.default_rng(seed)
    pts = mesh.sample(n_pts)
    pts = pts - pts.mean(0)

    def chamfer(a, b):
        """Mean distance from each point of `a` to the nearest point of `b`."""
        return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
                     .min(axis=1).mean())

    # The floor: a second independent sample of the same surface, unrotated.
    ref = mesh.sample(n_pts) - pts.mean(0) * 0
    ref = ref - ref.mean(0)
    floor = chamfer(ref, pts)
    if floor <= 0:
        return 0.0

    scores = []
    for _ in range(n_rot):
        # A uniformly random rotation, via QR of a Gaussian matrix.
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        scores.append(chamfer(pts @ q.T, pts))
    return float(np.median(scores)) / floor


def object_diameter_m(mesh_file):
    """The object's largest extent in metres, from its metric-scale mesh."""
    import trimesh
    return float(max(trimesh.load(mesh_file, process=False).extents))


def emit(name, value, why):
    """Print a shell assignment on stdout and its justification on stderr."""
    print(f"{name}={value}")
    print(f"  {name}={value}  <- {why}", file=sys.stderr)


def parse_args():
    """Parse whichever inputs are available; each unlocks a different knob."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--masks_root", default=None)
    p.add_argument("--seq", default=None)
    p.add_argument("--kid", type=int, default=0)
    p.add_argument("--width", type=int, default=796,
                   help="reference width residuals are expressed in (default: 796)")
    p.add_argument("--height", type=int, default=448)
    p.add_argument("--object_xyz", default=None)
    p.add_argument("--human_j3d", default=None)
    p.add_argument("--calib", default=None)
    p.add_argument("--cam", default="cam04")
    p.add_argument("--mesh", default=None,
                   help="metric-scale object mesh, for the symmetry test behind "
                        "REINIT_EVERY and for the object's true size")
    return p.parse_args()


def main():
    """Emit whichever knobs the provided inputs support."""
    args = parse_args()
    did = False

    if args.masks_root and args.seq:
        diam = object_diameter_px(args.masks_root, args.seq, args.kid, args.width)
        if diam is None:
            print(f"no object mask in {args.seq}; not deriving --inlier_px",
                  file=sys.stderr)
        else:
            px = float(np.clip(INLIER_FRACTION * diam, INLIER_MIN, INLIER_MAX))
            emit("TRI_INLIER_PX", f"{px:.2f}",
                 f"{INLIER_FRACTION:g} x {diam:.1f}px apparent diameter")
            did = True

    if args.object_xyz and args.calib:
        obj = np.load(args.object_xyz)
        d_obj = camera_distances(obj["xyz"], args.calib, args.cam,
                                 args.width, args.height)
        zfar = max(ZFAR_MIN, ZFAR_MARGIN * float(d_obj.max()))
        emit("ZFAR", f"{zfar:.1f}",
             f"{ZFAR_MARGIN:g} x {d_obj.max():.2f}m, the furthest the object is seen")
        did = True

        # Tracking's erode threshold, same Z/f as the scale step's: an object
        # of depth extent D spanning f*D/Z pixels recedes D/(f*D/Z) = Z/f per
        # pixel, and the size cancels.
        f = None
        if args.calib:
            from prep.triangulate_object import read_calibration
            cams = read_calibration(args.calib, args.width, args.height)
            K = cams[args.cam]["K"]
            f = float((K[0, 0] + K[1, 1]) / 2.0)
        if f:
            z_med = float(np.median(d_obj))
            erode = 3.0 * z_med / f
            emit("ERODE_DEPTH_THRES", f"{erode:.4f}",
                 f"3 x {z_med:.2f}m / {f:.0f}px, the object's depth change per pixel")

        # REINIT_EVERY: worth it only when the object both moves far enough
        # between frames to leave the tracker's basin AND has an orientation
        # nobody can observe anyway.
        if args.mesh and os.path.isfile(args.mesh):
            sym = symmetry_score(args.mesh)
            diam = object_diameter_m(args.mesh)
            step = float(np.median(np.linalg.norm(np.diff(obj["xyz"], axis=0), axis=1))) \
                if len(obj["xyz"]) > 1 else 0.0
            fast = diam > 0 and (step / diam) > MOTION_PER_DIAMETER
            symmetric = sym < SYMMETRY_TOL
            if symmetric and fast:
                emit("REINIT_EVERY", "1",
                     f"symmetry {sym:.3f} < {SYMMETRY_TOL} (orientation unobservable) "
                     f"and {step:.3f}m/frame over a {diam:.3f}m object")
            else:
                emit("REINIT_EVERY", "",
                     f"symmetry {sym:.3f}, motion {step:.3f}m/frame over a "
                     f"{diam:.3f}m object -- "
                     + ("not symmetric: re-registering would pick a different "
                        "orientation each frame" if not symmetric
                        else "moves little between frames; incremental tracking holds"))
                if fast and not symmetric:
                    print("  WARNING: this object moves fast AND has an observable "
                          "orientation. Neither setting is good -- incremental "
                          "tracking may lose it, re-registration may spin it. "
                          "Watch the render.", file=sys.stderr)

        if args.human_j3d:
            hum = np.load(args.human_j3d)
            d_hum = human_distances(hum)
            # Both are indexed by their own frame lists; compare only the
            # frames that appear in both, since a separation computed across
            # different instants is not a separation.
            common = np.intersect1d(obj["frames"], hum["frames"])
            if common.size:
                oi = np.searchsorted(obj["frames"], common)
                hi = np.searchsorted(hum["frames"], common)
                sep = np.abs(d_obj[oi] - d_hum[hi])
                sep = sep[np.isfinite(sep)]
                if sep.size:
                    band = max(BAND_MIN, BAND_MARGIN * float(sep.max()))
                    emit("DEPTH_HUMAN_BAND", f"{band:.2f}",
                         f"{BAND_MARGIN:g} x {sep.max():.2f}m, the largest "
                         f"object-person depth gap over {sep.size} shared frames")

    if not did:
        raise SystemExit("nothing to derive: give --masks_root/--seq, or "
                         "--object_xyz/--calib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
