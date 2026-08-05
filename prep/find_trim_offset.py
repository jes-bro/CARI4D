# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Recover where a trimmed clip sits inside the sequence it was cut from.

run_sam3_masks.py renumbers a trimmed clip's frames from zero and prints the
original range to stdout, so once that log is gone the clip no longer knows when
it happened. Everything indexed against the full take -- per-frame camera
extrinsics, another camera's frames, anything triangulated -- then lines up
against the wrong instant, silently and plausibly.

Matching the images recovers it, and unlike a remembered number it can be
checked: several probe frames must agree on the same offset, which they cannot
do by chance.

    python prep/find_trim_offset.py --clip <trimmed>.mp4 --source <full>.mp4

Comparison runs on small grayscale thumbnails. Re-encoding changes pixel values
slightly, so an exact match is not available and is not needed -- at thumbnail
scale the right frame is unmistakable and the runner-up is far behind.
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np

sys.path.append(os.getcwd())


def parse_args():
    """Parse the clip, the sequence it came from, and where to record the answer."""
    parser = argparse.ArgumentParser(
        description="Find a trimmed clip's start index in its source video")
    parser.add_argument("--clip", required=True, help="the trimmed .mp4")
    parser.add_argument("--source", required=True,
                        help="the full-length .mp4 it was cut from")
    parser.add_argument("--probes", type=int, default=5,
                        help="frames spread through the clip that must all agree "
                             "on the offset (default: 5)")
    parser.add_argument("--thumb", type=int, default=64,
                        help="thumbnail edge in pixels for comparison (default: 64)")
    parser.add_argument("--out_json", default=None,
                        help="write the offset here so it survives this shell")
    return parser.parse_args()


def thumbnail(frame, size):
    """Return a small grayscale version of a frame, as float32.

    Downsampling by strided indexing rather than interpolation: it is far
    cheaper across thousands of frames, and matching only needs the coarse
    structure that survives either way.
    """
    grey = frame[..., :3].mean(axis=2)
    rows = np.linspace(0, grey.shape[0] - 1, size).astype(int)
    cols = np.linspace(0, grey.shape[1] - 1, size).astype(int)
    small = grey[rows][:, cols].astype(np.float32)
    return (small - small.mean()) / (small.std() + 1e-6)


def read_clip_probes(path, count, size):
    """Return (probe positions within the clip, their thumbnails, clip length).

    Raises:
        SystemExit: if the clip cannot be read or is empty.
    """
    import imageio.v2 as imageio
    if not osp.isfile(path):
        raise SystemExit(f"ERROR: no video at {path}")
    frames = []
    reader = imageio.get_reader(path)
    try:
        for frame in reader:
            frames.append(np.asarray(frame))
    finally:
        reader.close()
    if not frames:
        raise SystemExit(f"ERROR: {osp.basename(path)} has no frames")
    idx = np.unique(np.linspace(0, len(frames) - 1, count).astype(int))
    return idx, [thumbnail(frames[i], size) for i in idx], len(frames)


def main():
    """Score every candidate offset and report the one the probes agree on."""
    args = parse_args()
    import imageio.v2 as imageio

    probe_idx, probe_thumbs, clip_len = read_clip_probes(
        args.clip, args.probes, args.thumb)
    print(f"clip   {osp.basename(args.clip)}: {clip_len} frames, "
          f"probing {list(probe_idx)}")

    if not osp.isfile(args.source):
        raise SystemExit(f"ERROR: no video at {args.source}")
    source = []
    reader = imageio.get_reader(args.source)
    try:
        for frame in reader:
            source.append(thumbnail(np.asarray(frame), args.thumb))
    finally:
        reader.close()
    print(f"source {osp.basename(args.source)}: {len(source)} frames")
    if len(source) < clip_len:
        raise SystemExit(
            f"ERROR: source is shorter than the clip ({len(source)} < {clip_len}); "
            f"these are not the same take")

    # Score each offset by how well EVERY probe matches at its own position.
    # One probe alone can be fooled by a repeated pose; several cannot, and the
    # margin over the runner-up is what says the answer is real.
    n = len(source) - clip_len + 1
    scores = np.zeros(n)
    for pos, thumb in zip(probe_idx, probe_thumbs):
        for off in range(n):
            scores[off] += float(np.abs(source[off + pos] - thumb).mean())
    scores /= len(probe_thumbs)

    best = int(np.argmin(scores))
    print(f"\nbest offset {best}: clip frame 0 is source frame {best}, "
          f"clip covers {best}..{best + clip_len - 1}  "
          f"({best / 30.0:.1f}s in at 30fps)")
    print(f"  score {scores[best]:.4f}")

    # Compared against the best offset that is NOT adjacent. The neighbouring
    # offsets always score well -- consecutive video frames look almost the same
    # -- so treating one of them as a rival runner-up condemns every correct
    # answer. What would signal real ambiguity is a DISTANT offset scoring
    # close, meaning the take repeats itself.
    far = np.array([i for i in range(n) if abs(i - best) > 2])
    if far.size:
        rival = int(far[np.argmin(scores[far])])
        margin = float(scores[rival] - scores[best])
        rel = margin / (scores[best] + 1e-9)
        print(f"  nearest non-adjacent rival is offset {rival} at "
              f"{scores[rival]:.4f} -- {rel * 100:.0f}% worse")
        if rel < 0.25:
            print("  WARNING: a distant offset scores nearly as well, so the "
                  "take repeats itself and this alignment is a guess.")
        else:
            print("  no distant offset comes close, so the alignment is sound.")
    else:
        rival, margin, rel = best, 0.0, 0.0
        print("  source is barely longer than the clip; only one alignment "
              "is possible.")

    if args.out_json:
        payload = {"clip": osp.abspath(args.clip),
                   "source": osp.abspath(args.source),
                   "start_frame": best,
                   "end_frame": best + clip_len - 1,
                   "clip_frames": clip_len,
                   "match_score": scores[best],
                   "margin": margin}
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
