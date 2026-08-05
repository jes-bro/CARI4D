# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Compare the depth map's object distance against what its apparent size implies.

FoundationPose seeds its initial translation from the median depth inside the
object mask (estimater.py:205, guess_translation). If that depth is wrong the
tracker starts in the wrong place, and on a small distant object there is not
enough evidence for it to recover -- which looks like a tracking failure but is
really a depth failure.

For an object of known physical size there is a second, independent estimate:
the pinhole relation Z = f*D/d, where D is the true diameter and d the mask's
apparent width in pixels. It depends only on the intrinsics and the silhouette,
not on the depth network, so the two disagreeing localises the problem.

That relation is exact for a fronto-parallel extent and approximate for a
sphere's silhouette, which is very slightly larger than its diameter subtends;
the error is well under a percent at these distances and far below what is
being diagnosed.

Usage:
    python prep/check_object_depth.py --video <clip>.mp4 --masks_root <dir> \\
        --diameter 0.239

Reads the .depth-reg.mp4 beside the video, so point --video at the aligned clip
if you want the depth the tracker actually consumes.
"""
import argparse
import os
import os.path as osp
import sys

import cv2
import h5py
import joblib
import numpy as np

sys.path.append(os.getcwd())

from prep.run_hy3d_recon import extract_seq_name


def parse_args():
    """Parse the clip, mask location and the object's true diameter."""
    parser = argparse.ArgumentParser(
        description="Compare depth-map object distance against apparent-size distance")
    parser.add_argument("--video", required=True,
                        help="the clip; its .depth-reg.mp4 sibling is read too")
    parser.add_argument("--masks_root", required=True,
                        help="directory holding <seq>_masks_k<kid>.h5")
    parser.add_argument("--diameter", type=float, required=True,
                        help="the object's true diameter in metres, e.g. 0.239 for a "
                             "size-7 basketball")
    parser.add_argument("--seq", default=None,
                        help="sequence name (default: derived from --video)")
    parser.add_argument("--kid", type=int, default=0, help="camera id (default: 0)")
    parser.add_argument("--focal", type=float, default=None,
                        help="focal length in pixels; read from the .color.pkl "
                             "intrinsics beside the video if not given")
    parser.add_argument("--min_px", type=int, default=4,
                        help="skip frames with fewer object pixels (default: 4, the "
                             "same floor guess_translation applies)")
    parser.add_argument("--max_rows", type=int, default=40,
                        help="rows of per-frame detail to print (default: 40)")
    return parser.parse_args()


def read_focal(video_path, given):
    """Return the focal length in pixels, from --focal or the saved intrinsics.

    unidepth_behave.py:194 writes a <prefix>.<kid>.color.pkl beside the video with
    joblib.dump of a dict keyed fx/fy/cx/cy. That is the focal length the depth
    was produced under, so it is the one the apparent-size relation must use --
    a mismatch here would show up as a fake depth discrepancy.

    fx and fy differ slightly, so their mean is used rather than fx alone; the
    difference is a fraction of a percent and well below what is being measured.

    Raises:
        SystemExit: if neither source provides one.
    """
    if given is not None:
        return given
    pkl = video_path.replace(".mp4", ".pkl")
    if not osp.isfile(pkl):
        raise SystemExit(f"ERROR: no intrinsics at {pkl}; pass --focal")
    try:
        data = joblib.load(pkl)
    except Exception as exc:
        raise SystemExit(f"ERROR: could not read {pkl} ({exc}); pass --focal")

    if isinstance(data, dict) and "fx" in data:
        focal = (float(np.asarray(data["fx"]).ravel()[0])
                 + float(np.asarray(data["fy"]).ravel()[0])) / 2.0
    else:
        values = np.asarray(data, dtype=float).ravel()
        if values.size < 1:
            raise SystemExit(f"ERROR: no focal length in {pkl}; pass --focal")
        focal = float(values[0])
    print(f"focal length {focal:.1f} px, from {osp.basename(pkl)}")
    return focal


def load_object_masks(masks_root, seq_name, kid):
    """Return {frame index: bool mask} for the object.

    Raises:
        SystemExit: if the h5 or its sequence group is missing.
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


def read_depth_frames(depth_path, count):
    """Read the 16-bit depth video as metres.

    The pipeline stores depth as uint16 millimetres, which fp_behave divides by
    1000 -- mirrored here so the numbers are the ones the tracker sees.

    Returns:
        A list of float arrays in metres, empty if the video cannot be read.
    """
    try:
        import videoio
        frames = [np.asarray(d, dtype=np.float64) / 1000.0
                  for _, d in zip(range(count), videoio.uint16read(depth_path))]
        if frames:
            return frames
    except Exception as exc:
        print(f"  videoio could not read it ({exc}); falling back to OpenCV")
    cap = cv2.VideoCapture(depth_path)
    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = frame[:, :, 0]
        frames.append(frame.astype(np.float64) / 1000.0)
    cap.release()
    return frames


def measure(mask, depth, focal, diameter, min_px):
    """Return one frame's two depth estimates, or None if it cannot be measured.

    Returns:
        dict with the mask width, both distances and their ratio.
    """
    ys, xs = np.where(mask)
    if len(ys) < min_px:
        return None
    width = float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
    apparent = focal * diameter / width

    measured = np.nan
    if depth is not None and depth.shape == mask.shape:
        valid = depth[mask & (depth >= 0.001)]
        if valid.size >= min_px:
            measured = float(np.median(valid))

    return {
        "px": int(mask.sum()),
        "width": width,
        "apparent": apparent,
        "measured": measured,
        "ratio": measured / apparent if apparent > 0 and np.isfinite(measured) else np.nan,
    }


def summarise(rows, diameter):
    """Print the aggregate comparison and say what it implies."""
    ratios = np.array([r["ratio"] for r in rows if np.isfinite(r["ratio"])])
    missing = sum(1 for r in rows if not np.isfinite(r["measured"]))
    print()
    print(f"frames measured        : {len(rows)}")
    print(f"  with no valid depth  : {missing}"
          + ("   <- guess_translation fails on these" if missing else ""))
    if ratios.size == 0:
        print("\nNo frame had usable depth inside the object mask. FoundationPose "
              "cannot seed a translation from this, so the failure is upstream of "
              "tracking -- either the depth is zero over the object (check zfar) or "
              "the masks and depth disagree in size.")
        return
    print(f"  depth/apparent ratio : median {np.median(ratios):.2f}, "
          f"range {ratios.min():.2f}-{ratios.max():.2f}")
    print(f"  apparent-size range  : "
          f"{min(r['apparent'] for r in rows):.2f}-"
          f"{max(r['apparent'] for r in rows):.2f} m "
          f"(assuming {diameter:.3f} m across)")
    print()
    median = float(np.median(ratios))
    spread = float(ratios.max() - ratios.min())
    if 0.85 <= median <= 1.15 and spread < 0.5:
        print("The two agree. The depth over the object is trustworthy, so a "
              "tracking failure is not a depth-seeding problem -- look at the "
              "silhouette size and the mesh instead.")
    elif 0.85 <= median <= 1.15:
        print("They agree on average but scatter frame to frame. The depth is "
              "usable for initialisation but noisy, so expect jitter along the "
              "camera axis rather than a wrong overall trajectory.")
    else:
        print(f"They disagree by about {abs(1 - median) * 100:.0f}%. The depth map "
              f"is {'further than' if median > 1 else 'closer than'} the silhouette "
              f"implies, so FoundationPose is being seeded in the wrong place. "
              f"Constraining the object's distance by apparent size would fix more "
              f"than tuning the tracker.")


def main():
    """Compare both depth estimates across the sequence and report."""
    args = parse_args()
    seq_name = args.seq or extract_seq_name(args.video)
    focal = read_focal(args.video, args.focal)

    masks = load_object_masks(args.masks_root, seq_name, args.kid)
    print(f"sequence {seq_name}: {len(masks)} frames with object masks")

    depth_path = args.video.replace(".color.mp4", ".depth-reg.mp4")
    depths = []
    if osp.isfile(depth_path):
        print(f"depth: {depth_path}")
        depths = read_depth_frames(depth_path, len(masks))
    else:
        print(f"WARNING: no depth video at {depth_path}; reporting apparent size only")

    rows = []
    for idx in sorted(masks):
        depth = depths[idx] if idx < len(depths) else None
        row = measure(masks[idx], depth, focal, args.diameter, args.min_px)
        if row is not None:
            row["frame"] = idx
            rows.append(row)

    if not rows:
        raise SystemExit("ERROR: no frame had a large enough object mask")

    print()
    print(f"{'frame':>6} {'px':>7} {'width':>6} {'apparent_m':>11} "
          f"{'depth_m':>9} {'ratio':>6}")
    print("-" * 50)
    step = max(1, len(rows) // args.max_rows)
    for row in rows[::step]:
        measured = f"{row['measured']:9.2f}" if np.isfinite(row["measured"]) else "     none"
        ratio = f"{row['ratio']:6.2f}" if np.isfinite(row["ratio"]) else "     -"
        print(f"{row['frame']:>6} {row['px']:>7} {row['width']:>6.0f} "
              f"{row['apparent']:>11.2f} {measured} {ratio}")

    summarise(rows, args.diameter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
