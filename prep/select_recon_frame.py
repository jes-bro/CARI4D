# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Rank a sequence's frames by how well suited each is to object reconstruction.

run_hy3d_recon.py reconstructs from a single frame, and which frame you pick
dominates the result. Frame 0 is rarely the best one: the object may be small,
clipped by the frame edge, motion blurred, or -- in hand-object interaction,
which is the whole point of CARI4D -- mostly hidden behind the hand holding it.

This scores every frame that has masks and prints a ranked table, so the choice
is made from the data rather than by guessing. It never requires a clean frame
to exist; it ranks what is there and lets the best available win.

Scored per frame:
    area        object mask pixels. More pixels, more detail to reconstruct from.
    contact     fraction of the object's dilated boundary overlapping the person
                mask. High means a hand is on it.
    solidity    object area / convex hull area. An unoccluded convex object is
                near 1.0; something biting into it creates concavities. Catches
                occluders that are not the person, which `contact` misses.
    border      whether the object touches the frame edge -- clipped is unusable.
    sharpness   Laplacian variance inside the object box, needing --check_sharpness
                since it decodes video. Matters because an airborne, unoccluded
                object is often motion blurred.

Usage:
    python prep/select_recon_frame.py --video <clip> --masks_root <dir>
    python prep/select_recon_frame.py --video <clip> --masks_root <dir> --check_sharpness
    python prep/select_recon_frame.py --video <clip> --masks_root <dir> --preview_dir /tmp/cand

--preview_dir writes the RGBA crop for the top-ranked frames using the same
crop_rgba run_hy3d_recon.py uses, so what you look at is what Hunyuan3D gets.
"""
import argparse
import math
import os
import os.path as osp
import sys

import cv2
import h5py
import numpy as np

sys.path.append(os.getcwd())

from prep.run_hy3d_recon import crop_rgba, extract_frame, extract_seq_name


def parse_args():
    """Parse the sequence, mask location and which optional checks to run."""
    parser = argparse.ArgumentParser(
        description="Rank frames by suitability for object reconstruction")
    parser.add_argument("--video", required=True,
                        help="the clip whose frame indices match the masks")
    parser.add_argument("--masks_root", required=True,
                        help="directory holding <seq>_masks_k<kid>.h5")
    parser.add_argument("--seq", default=None,
                        help="sequence name (default: derived from --video)")
    parser.add_argument("--kid", type=int, default=0, help="camera id (default: 0)")
    parser.add_argument("--top", type=int, default=15,
                        help="how many frames to report (default: 15)")
    parser.add_argument("--dilate", type=int, default=5,
                        help="pixels to dilate the object mask when measuring "
                             "person contact (default: 5)")
    parser.add_argument("--min_area", type=int, default=1,
                        help="ignore frames whose object mask is smaller (default: 1)")
    parser.add_argument("--check_sharpness", action="store_true",
                        help="also measure focus inside the object box; decodes the "
                             "video, so it is slower")
    parser.add_argument("--preview_dir", default=None,
                        help="write RGBA crops for the top frames here")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="crop margin for previews (default: 0.2)")
    parser.add_argument("--crop_size", type=int, default=512,
                        help="preview crop size (default: 512)")
    parser.add_argument("--hires_video", default=None,
                        help="render previews from this higher-resolution copy of the "
                             "take, matching what run_hy3d_recon --hires_video would "
                             "feed Hunyuan3D. Without it previews come from --video and "
                             "understate the crop the reconstruction actually sees.")
    parser.add_argument("--hires_frame_offset", type=int, default=0,
                        help="frame in --hires_video corresponding to frame 0 of "
                             "--video; the trim start 'lo' from run_sam3_masks.py "
                             "(default: 0)")
    return parser.parse_args()


def load_all_masks(masks_root, seq_name, kid):
    """Read every frame's person and object mask from the sequence's h5.

    Returns:
        (object_masks, person_masks) as dicts of frame index -> bool array.

    Raises:
        SystemExit: if the h5 or its sequence group is missing.
    """
    h5_path = osp.join(masks_root, f"{seq_name}_masks_k{kid}.h5")
    if not osp.isfile(h5_path):
        raise SystemExit(f"ERROR: mask file not found: {h5_path}")

    objects, persons = {}, {}
    with h5py.File(h5_path, "r") as f:
        if seq_name not in f:
            raise SystemExit(
                f"ERROR: group '{seq_name}' not in {h5_path}; found {list(f.keys())}")
        group = f[seq_name]
        for key in group:
            if key.endswith(f"-k{kid}.obj_rend_mask.png"):
                idx = int(key.split("-")[0])
                objects[idx] = group[key][:].astype(bool)
            elif key.endswith(f"-k{kid}.person_mask.png"):
                idx = int(key.split("-")[0])
                persons[idx] = group[key][:].astype(bool)
    return objects, persons


def solidity(mask):
    """Return object area / convex hull area for the largest connected blob.

    An unoccluded convex object scores near 1.0. A hand biting into it leaves a
    concave silhouette and drops the score, which catches occlusion regardless of
    what is doing the occluding. Genuinely non-convex objects score low too, so
    this is a comparator between frames of one sequence, not across objects.

    It does NOT catch occlusion that removes a convex piece: an occluder taking
    exactly half a disc leaves a half-disc, which is still convex and still
    scores ~0.97. `contact` is what catches that case when the occluder is the
    person, which in hand-object interaction it usually is.

    Returns:
        Solidity in [0, 1], or 0.0 when there is no usable contour.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area <= 0:
        return 0.0
    return float(area / hull_area)


def enclosing_circle_fill(mask):
    """Return object area / area of its minimum enclosing circle.

    This is the metric solidity cannot provide: an occluder that removes a
    convex piece leaves a convex remainder, so solidity stays high, but the
    minimum enclosing circle barely shrinks while the area drops. Half a disc
    scores 0.5 here and 0.97 by solidity.

    The absolute value depends on the object -- an elongated object scores low
    even fully visible -- so it is only meaningful after normalising against the
    sequence's best frame. Since every frame shows the same rigid object, that
    constant shape bias cancels and what remains is "less complete than this
    object's best view", which is the question being asked.

    Returns:
        Fill ratio in [0, 1], or 0.0 when there is no usable contour.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    _, radius = cv2.minEnclosingCircle(contour)
    if radius <= 0:
        return 0.0
    circle_area = math.pi * radius * radius
    return float(min(1.0, cv2.contourArea(contour) / circle_area))


def person_contact(obj_mask, person_mask, dilate):
    """Fraction of the object's dilated ring that the person mask occupies.

    Dilating and looking at the added ring measures contact along the boundary
    rather than raw overlap, because SAM3 usually assigns each pixel to one mask
    or the other -- a hand gripping an object produces little overlap but heavy
    adjacency.

    Returns:
        A ratio in [0, 1]; 0.0 when there is no person mask.
    """
    if person_mask is None or not person_mask.any():
        return 0.0
    kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    dilated = cv2.dilate(obj_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & ~obj_mask
    if not ring.any():
        return 0.0
    return float((ring & person_mask).sum() / ring.sum())


def touches_border(mask, pad=2):
    """Whether the mask reaches within pad pixels of any frame edge."""
    return bool(mask[:pad].any() or mask[-pad:].any()
                or mask[:, :pad].any() or mask[:, -pad:].any())


def bbox_of(mask):
    """Return (y0, y1, x0, x1) for the mask's tight bounding box."""
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def measure_sharpness(video_path, indices, masks):
    """Laplacian variance inside each frame's object box.

    Decodes sequentially rather than seeking per frame, which is far faster on
    long videos. An object that is unoccluded because it is mid-flight is often
    motion blurred, so this stops the ranking preferring blurry frames.

    Returns:
        dict of frame index -> variance (higher is sharper).
    """
    wanted = set(indices)
    scores = {}
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            y0, y1, x0, x1 = bbox_of(masks[idx])
            patch = frame[y0:y1 + 1, x0:x1 + 1]
            if patch.size:
                grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                scores[idx] = float(cv2.Laplacian(grey, cv2.CV_64F).var())
            wanted.discard(idx)
        idx += 1
    cap.release()
    return scores


def score_frames(objects, persons, dilate, min_area):
    """Measure every frame and return the per-frame records.

    Returns:
        A list of dicts, one per frame with a large enough object mask.
    """
    records = []
    for idx in sorted(objects):
        obj = objects[idx]
        area = int(obj.sum())
        if area < min_area:
            continue
        records.append({
            "frame": idx,
            "area": area,
            "contact": person_contact(obj, persons.get(idx), dilate),
            "solidity": solidity(obj),
            "circle_fill": enclosing_circle_fill(obj),
            "border": touches_border(obj),
        })
    return records


def combine_scores(records, sharpness=None):
    """Fold the measurements into one comparable score per frame.

    Everything is normalised against the best frame in this sequence, so scores
    compare frames of one take and mean nothing across takes. That is what makes
    the shape metrics usable at all: every frame shows the same object, so its
    constant shape bias divides out and only "less complete than this object's
    best view" survives.

    Completeness is the MINIMUM of normalised solidity and normalised
    circle-fill. They fail on opposite cases -- solidity misses an occluder that
    removes a convex piece (half a disc still scores ~0.97), circle-fill misses
    little else but is noisy for elongated shapes -- so the worse of the two
    governs.

    The factors multiply rather than sum: a frame clipped by the border, or
    almost entirely occluded, should be disqualified outright rather than
    compensated for by being large.

    Note the rigid-object assumption. A deformable object genuinely changes
    silhouette between frames, and this reads that as incompleteness. For small
    deformations (a CPR mannequin's chest) that is a minor bias; for cloth it
    would be meaningless.
    """
    if not records:
        return
    max_area = max(r["area"] for r in records)
    max_solidity = max(r["solidity"] for r in records) or 1.0
    max_circle = max(r["circle_fill"] for r in records) or 1.0
    max_sharp = max(sharpness.values()) if sharpness else None
    for r in records:
        r["area_norm"] = r["area"] / max_area
        r["solidity_norm"] = min(1.0, r["solidity"] / max_solidity)
        r["circle_norm"] = min(1.0, r["circle_fill"] / max_circle)
        r["completeness"] = min(r["solidity_norm"], r["circle_norm"])
        r["sharpness"] = sharpness.get(r["frame"]) if sharpness else None
        sharp_factor = 1.0
        if max_sharp:
            sharp_factor = 0.5 + 0.5 * (r["sharpness"] or 0.0) / max_sharp
        r["score"] = (r["area_norm"]
                      * (1.0 - r["contact"])
                      * r["completeness"]
                      * (0.0 if r["border"] else 1.0)
                      * sharp_factor)


def print_table(records, top, with_sharpness):
    """Print the ranked frames with every component visible.

    The components are shown, not just the score, so a bad ranking can be
    overridden knowingly -- the weighting is a heuristic, and the person looking
    at the previews is the real judge.
    """
    header = (f"{'frame':>6} {'score':>7} {'area':>7} {'area%':>6} {'contact':>8} "
              f"{'solid':>6} {'circle':>7} {'compl':>6} {'border':>7}")
    if with_sharpness:
        header += f" {'sharp':>8}"
    print(header)
    print("-" * len(header))
    for r in records[:top]:
        line = (f"{r['frame']:>6} {r['score']:>7.3f} {r['area']:>7} "
                f"{100 * r['area_norm']:>5.0f}% {r['contact']:>8.2f} "
                f"{r['solidity']:>6.2f} {r['circle_fill']:>7.2f} "
                f"{r['completeness']:>6.2f} {str(r['border']):>7}")
        if with_sharpness:
            line += f" {r['sharpness'] or 0.0:>8.1f}"
        print(line)


def _save_preview(rgb, mask_bool, record, rank, out_dir, margin, crop_size):
    """Crop one frame and write it, returning True if it was written.

    Uses run_hy3d_recon's crop_rgba so the preview is exactly what Hunyuan3D
    would receive rather than an approximation. A frame whose mask cannot be
    cropped is reported and skipped, since one bad frame should not abandon the
    rest of the batch.
    """
    mask = mask_bool.astype(np.uint8) * 255
    if mask.shape[:2] != rgb.shape[:2]:
        H, W = rgb.shape[:2]
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
    try:
        rgba = crop_rgba(rgb, mask, margin=margin, crop_size=crop_size)
    except ValueError as exc:
        print(f"  frame {record['frame']}: {exc}")
        return False
    path = osp.join(out_dir, f"rank{rank:02d}_frame{record['frame']:06d}.png")
    rgba.save(path)
    print(f"  wrote {path}")
    return True


def write_previews(video_path, records, objects, out_dir, top, margin, crop_size,
                   hires_video=None, hires_frame_offset=0):
    """Write RGBA crops for the top-ranked frames.

    With hires_video the frames come from a higher-resolution copy of the take
    and the mask is upscaled to match, mirroring run_hy3d_recon --hires_video.
    That matters for judging: a preview built from the low-resolution clip
    understates the crop the reconstruction would actually get, which can make a
    perfectly usable frame look hopeless.

    The low-resolution path decodes sequentially because it walks most of the
    clip; the hi-res path seeks, since it only wants a handful of frames out of
    a much larger file.
    """
    os.makedirs(out_dir, exist_ok=True)
    selected = records[:top]
    written = 0

    if hires_video:
        print(f"Previewing from {hires_video} (offset {hires_frame_offset})")
        for rank, record in enumerate(selected, start=1):
            idx = record["frame"]
            try:
                rgb = extract_frame(hires_video, idx + hires_frame_offset)
            except RuntimeError as exc:
                print(f"  frame {idx}: {exc}")
                continue
            written += _save_preview(rgb, objects[idx], record, rank, out_dir,
                                     margin, crop_size)
    else:
        wanted = {r["frame"]: rank for rank, r in enumerate(selected, start=1)}
        by_frame = {r["frame"]: r for r in selected}
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while wanted and idx <= max(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                written += _save_preview(rgb, objects[idx], by_frame[idx],
                                         wanted.pop(idx), out_dir, margin, crop_size)
            idx += 1
        cap.release()

    print(f"Wrote {written} preview(s) to {out_dir}")


def main():
    """Rank the sequence's frames and report, optionally writing previews."""
    args = parse_args()
    seq_name = args.seq or extract_seq_name(args.video)
    print(f"Sequence  : {seq_name}")
    print(f"Masks     : {args.masks_root}")

    objects, persons = load_all_masks(args.masks_root, seq_name, args.kid)
    print(f"Frames    : {len(objects)} with object masks, "
          f"{sum(1 for m in objects.values() if m.any())} non-empty")

    records = score_frames(objects, persons, args.dilate, args.min_area)
    if not records:
        raise SystemExit("ERROR: no frame has a non-empty object mask")

    sharpness = None
    if args.check_sharpness:
        print("Measuring sharpness (decoding video)...")
        sharpness = measure_sharpness(
            args.video, [r["frame"] for r in records],
            {r["frame"]: objects[r["frame"]] for r in records})

    combine_scores(records, sharpness)
    records.sort(key=lambda r: -r["score"])

    print()
    print_table(records, args.top, args.check_sharpness)
    best = records[0]
    print()
    print(f"Best frame: {best['frame']}  "
          f"(area {best['area']} px, contact {best['contact']:.2f}, "
          f"completeness {best['completeness']:.2f})")
    print(f"Use it with: --frame_index {best['frame']}")

    if args.preview_dir:
        print()
        write_previews(args.video, records, objects, args.preview_dir,
                       args.top, args.margin, args.crop_size,
                       hires_video=args.hires_video,
                       hires_frame_offset=args.hires_frame_offset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
