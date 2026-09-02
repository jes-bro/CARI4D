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
import re
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
        print("\nNo frame reaches the threshold. This clip cannot be triangulated.")
        print_next(args.seq, args.masks_root, None, None, n, aux)
        return 1
    r_lo, r_hi = run
    print(f"longest usable run: frames {r_lo}-{r_hi} "
          f"({r_hi - r_lo + 1} frames, {(r_hi - r_lo + 1) / 30.0:.1f}s)")
    print_next(args.seq, args.masks_root, r_lo, r_hi, n, aux)
    return 0


def take_for(work, seq):
    """The take a clip came from, read off the filesystem, or "<take>".

    Stage 1a symlinks the pipeline video into the take it came from, so the
    take name is already on disk:

        work/<base>/src/<base>.0.color.mp4
            -> <takes_root>/<TAKE>/frame_aligned_videos/downscaled/448/<cam>.mp4

    Reading it there rather than from a list of sequences keeps this working for
    any capture and any object, including ones no split file has heard of. Clip
    names are <base><letter>[t], and the symlink lives in the take-level
    directory, so the trailing letters come off to find it.

    This exists only so the printed command can be pasted rather than edited; a
    missing symlink is not an error, just a placeholder.
    """
    base = re.sub(r"[a-z]+$", "", seq)
    for d in (work, osp.join(osp.dirname(osp.normpath(work)), base)):
        src = osp.join(d, "src")
        if not osp.isdir(src):
            continue
        for name in sorted(os.listdir(src)):
            target = osp.realpath(osp.join(src, name))
            if "/frame_aligned_videos/" in target:
                return osp.basename(target.split("/frame_aligned_videos/")[0])
    return "<take>"


def print_next(seq, masks_root, lo, hi, n_frames, aux, min_frames=60):
    """Say what to do about this coverage, with the numbers filled in.

    A diagnostic that stops at the diagnosis leaves the reader to work out the
    consequence, and the consequence here is a command with two frame numbers
    in it that nobody should be transcribing by eye.
    """
    # Commands here are meant to be pasted from the repo root, where the
    # relative form is what everything else prints. A work directory outside
    # the repo relativises to a stack of "..", which is worse than absolute.
    work = osp.dirname(osp.normpath(masks_root))
    rel = work
    try:
        candidate = osp.relpath(work, os.getcwd())
        if not candidate.startswith(".."):
            rel = candidate
    except ValueError:
        pass

    take = take_for(work, seq)

    print()
    print("  ============================================================")
    print("   WHAT TO DO NEXT")
    print("  ============================================================")

    # An EgoExo4D capture has four exo cameras, so one view acting as the
    # pipeline camera leaves three aux. Fewer than that means a masking job
    # failed, not that a camera was useless -- and a missing camera is the most
    # common reason coverage looks bad, so say it before suggesting a trim.
    if len(aux) < 3:
        print(f"   Only {len(aux)} aux view(s) got masked, out of the 3 a four-camera")
        print("   capture should give. A missing one is usually a failed job, not a")
        print("   useless camera, and re-running it may fix the coverage outright:")
        print()
        print("       sacct -u $USER --starttime now-1day \\")
        print("           --format=JobID%14,JobName%34,State,ExitCode \\")
        print(f"           | grep '{seq}'")
        print()

    if lo is None:
        print("   Nothing to trim to. Look at the aux mask videos:")
        print(f"       ls {rel}/masks/*_sam3_vis.mp4")
        print("   then re-run step 3 with different prompts, or drop this clip.")
        print("  ============================================================")
        return

    kept = hi - lo + 1
    if kept == n_frames:
        print("   Every frame is triangulatable. No trim needed -- go to step 5:")
        print()
        print(f"       TAKE={take} SEQ={seq} bash scripts/recon_geometry.sh")
    elif kept < min_frames:
        print(f"   The usable run is only {kept} frames ({kept / 30.0:.1f}s), under the")
        print(f"   {min_frames} this pipeline treats as reconstructable. Trimming would")
        print("   leave too little. Better options, in order:")
        print()
        print("     1. check whether a camera's masking failed (above) and re-run it")
        print("     2. drop this clip and use the others from this take")
        print()
        print("   If you want it anyway:")
        print()
        print(f"       python prep/retrim_clip.py --work {rel} --lo {lo} --hi {hi}")
    else:
        print(f"   Trim to the covered part -- {kept} of {n_frames} frames survive:")
        print()
        print(f"       python prep/retrim_clip.py --work {rel} --lo {lo} --hi {hi}")
        print()
        print(f"   That writes {seq}t. Use THAT name from step 5 on:")
        print()
        print(f"       TAKE={take} SEQ={seq}t bash scripts/recon_geometry.sh")
    print("  ============================================================")


if __name__ == "__main__":
    raise SystemExit(main())
