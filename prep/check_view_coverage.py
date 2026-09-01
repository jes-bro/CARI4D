"""How many views can see the object, per frame, before any geometry is run.

A clip's boundaries come from the pipeline camera alone: stage 1a keeps the
frames where that one view tracks both the person and the ball. But
triangulation needs the object in TWO views, and the two criteria disagree in a
specific, recurring way -- while a player stands holding the ball, the pipeline
camera sees it perfectly and every side camera has a body between it and the
ball. Those frames enter the clip, triangulate to nothing, and fall back to the
monocular depth the whole multi-view apparatus exists to replace.

Measured on Date03_Sub01_bball_rev003: the clip was frames 0-182, the object
triangulated on 60-182, and the render was visibly wrong for exactly the first
two seconds.

This reads the mask files stage 1b already wrote and answers the question those
60 frames raise, in seconds and without a GPU: on which frames is the object
actually visible to enough cameras, and what would the clip be if that were the
criterion?

Run from the repo root, in the cari4d env (newcari4d). The work itself is only
h5py and numpy, but reading the masks goes through prep/triangulate_object,
which imports cv2 for the fisheye model it does not use here.

Usage:
    python prep/check_view_coverage.py --masks_root work/<seq>/masks --seq <seq>
    python prep/check_view_coverage.py --masks_root work/<seq>/masks --seq <seq> \\
        --aux cam01-4k --aux cam02-4k --aux cam03-4k --min_views 2
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np

sys.path.append(os.getcwd())
from prep.triangulate_object import load_object_centroids


def parse_args():
    """Parse the mask directory, the views to check and the coverage threshold."""
    parser = argparse.ArgumentParser(
        description="Per-frame count of views that see the object")
    parser.add_argument("--masks_root", required=True,
                        help="directory holding <seq>_masks_k<kid>.h5 for every view")
    parser.add_argument("--seq", required=True,
                        help="pipeline sequence name, i.e. the clip's own name")
    parser.add_argument("--aux", action="append", default=None,
                        help="aux mask-set name, repeatable (default: every cam*-4k "
                             "mask file found next to --seq)")
    parser.add_argument("--kid", type=int, default=0, help="camera id in the mask keys")
    parser.add_argument("--min_px", type=int, default=4,
                        help="object masks smaller than this do not count (default: 4, "
                             "matching triangulate_object.py)")
    parser.add_argument("--min_views", type=int, default=2,
                        help="views needed to triangulate a frame (default: 2)")
    parser.add_argument("--gap", type=int, default=0,
                        help="bridge dropouts of up to N frames when finding the longest "
                             "usable run (default: 0)")
    return parser.parse_args()


def discover_aux(masks_root, seq, kid):
    """Return the aux mask-set names present in masks_root, excluding the clip's own.

    Globbing the directory rather than requiring the list means this reports on
    what was actually masked, which is the point -- a view that was never
    masked and a view that never sees the ball are different problems.
    """
    suffix = f"_masks_k{kid}.h5"
    names = sorted(f[:-len(suffix)] for f in os.listdir(masks_root)
                   if f.endswith(suffix))
    return [n for n in names if n != seq]


def frames_seen(masks_root, name, kid, min_px):
    """Return the set of frame indices where mask set `name` holds the object."""
    centroids, _ = load_object_centroids(masks_root, name, kid, min_px)
    return set(centroids)


def longest_run(good_frames, all_frames, gap):
    """Return the longest (lo, hi) run of usable frames, bridging gaps of <= `gap`.

    Mirrors run_sam3_masks.find_tracked_runs so the window this suggests is the
    same kind of object stage 1a emits -- a contiguous range, not a scatter.
    """
    runs, start, miss, last = [], None, 0, None
    for f in all_frames:
        if f in good_frames:
            start = f if start is None else start
            miss, last = 0, f
        elif start is not None:
            miss += 1
            if miss > gap:
                runs.append((start, last))
                start = None
    if start is not None:
        runs.append((start, last))
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def main():
    """Print per-view coverage, the per-frame view count, and the usable window."""
    args = parse_args()
    aux = args.aux if args.aux else discover_aux(args.masks_root, args.seq, args.kid)
    if not aux:
        print(f"no aux mask sets in {args.masks_root} besides {args.seq}; "
              f"nothing can be triangulated", file=sys.stderr)
        return 1

    views = [args.seq] + aux
    seen = {v: frames_seen(args.masks_root, v, args.kid, args.min_px) for v in views}
    all_frames = sorted(seen[args.seq])
    if not all_frames:
        print(f"the pipeline view {args.seq} has no object mask at all", file=sys.stderr)
        return 1
    lo, hi = all_frames[0], all_frames[-1]
    n = len(all_frames)

    print(f"clip {args.seq}: frames {lo}-{hi} ({n} frames)\n")
    print(f"{'view':<28} {'frames with the object':>24}")
    print("-" * 54)
    for v in views:
        k = len(seen[v] & set(all_frames))
        print(f"{v:<28} {k:>10}/{n} ({100.0 * k / n:>5.1f}%)")

    counts = np.array([sum(f in seen[v] for v in views) for f in all_frames])
    usable = {f for f, c in zip(all_frames, counts) if c >= args.min_views}
    print(f"\nviews per frame: " + ", ".join(
        f"{c} view(s) x{int((counts == c).sum())}" for c in sorted(set(counts.tolist()))))
    print(f"triangulatable ({args.min_views}+ views): {len(usable)}/{n} "
          f"({100.0 * len(usable) / n:.1f}%)")

    run = longest_run(usable, all_frames, args.gap)
    if run is None:
        print("\nNo frame reaches the threshold. This clip cannot be triangulated: "
              "check the aux mask videos before spending anything downstream.")
        return 1
    r_lo, r_hi = run
    print(f"longest usable run: frames {r_lo}-{r_hi} "
          f"({r_hi - r_lo + 1} frames, {(r_hi - r_lo + 1) / 30.0:.1f}s)")

    lead, tail = r_lo - lo, hi - r_hi
    if lead or tail:
        print(f"\nThe clip carries {lead} frame(s) before and {tail} after that window "
              f"which no\npair of views can place in 3D. Those fall back to monocular "
              f"depth in the\ninjection step. Re-emitting the clip as {r_lo}-{r_hi} "
              f"would drop them.")
    else:
        print("\nEvery frame of this clip is triangulatable; nothing to tighten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
