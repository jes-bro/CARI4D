# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""List frame count, rate and duration for every video the pipeline produced.

A pipeline that writes videos at several stages makes it easy to reason about
one file while looking at another, and nothing in a player tells you which is
which. Length is the quickest way to tell them apart, and a mismatch between the
clip's length and the length of what is on screen means the reasoning and the
evidence are about different files.

Frames are counted rather than read from the header: a container's reported
duration and its actual frame count disagree often enough that trusting the
header is how a 101-frame clip comes to be described as five seconds long.

    python prep/list_video_durations.py
    python prep/list_video_durations.py --roots /path/to/other /another
"""
import argparse
import glob
import os
import os.path as osp
import sys

sys.path.append(os.getcwd())


# Where this pipeline leaves videos, relative to a repo root. Ordered by stage
# so the listing reads as the sequence that produced them.
PATTERNS = [
    "sam3masks/trimmed_vids/*.mp4",       # the trimmed clip everything else follows
    "sam3masks/*.mp4",                    # sam3 visualisation, whole take
    "sam3masks/trimmed_vids-aligned/*.mp4",
    "output/viz/*.mp4",                   # coconet
    "output/opt/*/*.mp4",                 # optimiser, per checkpoint
    "renders/*.mp4",                      # isaac gym replays
    "data/cari4d-demo/*/*.mp4",
]


def parse_args():
    """Parse the roots to search."""
    parser = argparse.ArgumentParser(
        description="Report frames, fps and duration for pipeline videos")
    parser.add_argument("--roots", nargs="+",
                        default=["/simurgh2/projects/ret-hoi/CARI4D",
                                 "/simurgh2/projects/ret-hoi/InterMimic"],
                        help="directories to search (default: the two repos)")
    parser.add_argument("--patterns", nargs="+", default=None,
                        help="override the glob patterns searched under each root")
    return parser.parse_args()


def describe(path):
    """Return (frames, fps, seconds) for a video, counting frames directly.

    Returns:
        (None, None, None) if the file cannot be read, so one unreadable video
        does not stop the listing -- an unreadable file is itself worth seeing.
    """
    import imageio.v2 as imageio
    try:
        reader = imageio.get_reader(path)
        try:
            fps = float(reader.get_meta_data().get("fps") or 0.0)
            frames = int(reader.count_frames())
        finally:
            reader.close()
    except Exception:
        return None, None, None
    return frames, fps, (frames / fps if fps else None)


def main():
    """Print one line per video found, grouped by root."""
    args = parse_args()
    patterns = args.patterns or PATTERNS

    for root in args.roots:
        if not osp.isdir(root):
            print(f"\n{root}  (not a directory, skipped)")
            continue
        print(f"\n{root}")
        found = 0
        for pattern in patterns:
            for path in sorted(glob.glob(osp.join(root, pattern))):
                frames, fps, secs = describe(path)
                rel = osp.relpath(path, root)
                if frames is None:
                    print(f"  {'?':>6} {'?':>7} {'?':>8}   {rel}   (unreadable)")
                else:
                    secs_s = f"{secs:.2f}s" if secs else "?"
                    print(f"  {frames:>6} fr {fps:>6.1f} fps {secs_s:>8}   {rel}")
                found += 1
        if not found:
            print("  (no videos matched)")

    print("\nA clip and the render made from it should have the same frame "
          "count.\nDiffering counts mean they are not the same take.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
