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
