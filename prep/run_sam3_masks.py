"""
Run SAM3 text-prompted video segmentation to produce human+object masks
in the HDF5 format expected by CARI4D (Step 2 of custom_video.md).

Usage:
    python prep/run_sam3_masks.py \
        --video data/cari4d-demo/wild/videos/Date03_Sub01_gas_wild002.0.color.mp4 \
        --human_prompt "a man with black t-shirt and black pants" \
        --object_prompt "a red gas cylinder" \
        --visualize
"""

import argparse
import os
import sys
import tempfile
import shutil

import cv2
import h5py
import imageio
import numpy as np
import torch

# Allow importing sam3 from the local sam3/ subfolder
SAM3_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sam3")
sys.path.insert(0, SAM3_ROOT)

from sam3.model_builder import build_sam3_video_predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3 masks for CARI4D")
    parser.add_argument("--video", required=True, help="Path to input MP4 video")
    parser.add_argument("--human_prompt", required=True, help="Text prompt for human")
    parser.add_argument("--object_prompt", required=True, help="Text prompt for object")
    parser.add_argument("--kid", type=int, default=0, help="Camera/kinect id (default 0)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for masks H5 (default: sibling masks/ folder)")
    # ON BY DEFAULT. The visualization and the trim are the whole point of a run
    # on a wild take -- you cannot judge whether SAM3 held without looking, and a
    # take is rarely usable end to end. Kept as an accepted no-op so existing
    # callers that pass --visualize keep working.
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Save visualization MP4 (DEFAULT ON; --no_visualize to skip)")
    parser.add_argument("--no_visualize", dest="visualize", action="store_false",
                        help="Skip the visualization MP4")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token for SAM3 checkpoint access")
    parser.add_argument("--chunk_size", type=int, default=300,
                        help="Process video in chunks of this many frames to avoid OOM (default: 300)")
    parser.add_argument("--object_select", choices=["near-person", "single", "union"],
                        default="near-person",
                        help="How to pick the object among SAM3's detections. A kitchen "
                             "has many cups and the union of all of them is not a "
                             "trajectory, so the default keeps ONE instance per frame, "
                             "seeded by proximity to the person -- the manipulated object "
                             "is the one at their hands -- then held by frame-to-frame "
                             "overlap. 'single' seeds by largest instead; 'union' is the "
                             "old merge-everything behaviour")
    parser.add_argument("--object_open", type=int, default=0,
                        help="Radius in px of a morphological opening applied to object "
                             "masks: severs thin attachments before keeping one connected "
                             "component. OFF by default -- it cannot tell a poured liquid "
                             "stream from a pot handle, and amputating real thin parts is "
                             "worse than a transient bleed, which the downstream "
                             "triangulation residual gate absorbs anyway. Guarded when on: "
                             "if cleanup would erase most of the mask, the original is "
                             "kept (default: 0)")
    parser.add_argument("--zoom", action="store_true", default=True,
                        help="Visualization: magnified inset around the object (DEFAULT ON). "
                             "A ball is ~18px in a 796x448 frame; at 1x you cannot tell "
                             "whether the mask is on it. --no_zoom to skip")
    parser.add_argument("--no_zoom", dest="zoom", action="store_false",
                        help="Skip the magnified object inset")
    parser.add_argument("--zoom_size", type=int, default=140)
    # --- trimming: cut the sequence down to a stretch where BOTH masks hold ---
    parser.add_argument("--trim_to_tracked", action="store_true", default=True,
                        help="Write <seq>_trim.0.color.mp4 + masks covering the longest run "
                             "of frames where BOTH masks are present (DEFAULT ON). SAM3 loses "
                             "small fast objects for most of a take; this salvages the usable "
                             "part. --no_trim to skip")
    parser.add_argument("--no_trim", dest="trim_to_tracked", action="store_false",
                        help="Do not write a trimmed sequence")
    parser.add_argument("--trim_gap_tolerance", type=int, default=0,
                        help="Bridge dropouts of up to N frames when finding runs (default 0, "
                             "i.e. strictly-tracked frames only). "
                             "Bridged frames keep their empty masks, so this trades mask "
                             "coverage for clip length. Measured on a basketball take: 0 -> "
                             "3.4s clip, 5 -> 5.0s at 96.7%% covered, 10 -> 10.0s at 90.3%%. "
                             "0 for strictly-tracked-only")
    parser.add_argument("--trim_min_person_px", type=int, default=1,
                        help="Min human mask area to count as tracked (default 1)")
    parser.add_argument("--trim_min_object_px", type=int, default=1,
                        help="Min object mask area to count as tracked (default 1). Raise it "
                             "to reject a few-pixel spurious blob")
    parser.add_argument("--trim_rank", type=int, default=1,
                        help="Take the Nth longest run (1 = longest)")
    return parser.parse_args()


def extract_seq_name(video_path):
    """Extract sequence name from video filename like <seq>.0.color.mp4"""
    basename = os.path.basename(video_path)
    if ".0.color.mp4" in basename:
        return basename.replace(".0.color.mp4", "")
    return os.path.splitext(basename)[0]


def load_video_frames(video_path):
    """Load all frames from an MP4 video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def merge_masks_from_output(out):
    """Union all instance masks from SAM3 output into one binary mask.

    SAM3 output format:
        out['out_binary_masks']: (N_objects, H, W) bool array
        out['out_obj_ids']: (N_objects,) int array
    """
    masks = out["out_binary_masks"]  # (N, H, W) bool
    if len(masks) == 0:
        return None
    # Union all instances
    return masks.any(axis=0)  # (H, W) bool


def select_single_mask(out, ref_mask=None, person_mask=None):
    """Pick exactly one instance from a SAM3 detection output.

    Prefers the instance overlapping ref_mask the most (keeps the same target
    across chunk boundaries, where object ids are not stable). With no usable
    ref_mask, a given person_mask breaks the tie instead: the instance nearest
    the (dilated) person wins, because the manipulated object is the one at
    their hands, not the largest look-alike in the scene. Falls back to the
    largest instance. Returns (mask, obj_id), or (None, None) if nothing was
    detected.
    """
    masks = out["out_binary_masks"]  # (N, H, W) bool
    obj_ids = [int(i) for i in out["out_obj_ids"]]
    if len(masks) == 0:
        return None, None

    if ref_mask is not None and ref_mask.any():
        overlaps = [int(np.logical_and(m, ref_mask).sum()) for m in masks]
        best = int(np.argmax(overlaps))
        if overlaps[best] > 0:
            return masks[best], obj_ids[best]

    if person_mask is not None and person_mask.any():
        # Reach scales with the frame so 448p and 4K behave alike.
        k = max(15, person_mask.shape[0] // 16)
        near = cv2.dilate(person_mask.astype(np.uint8),
                          np.ones((k, k), np.uint8)) > 0
        overlaps = [int(np.logical_and(m, near).sum()) for m in masks]
        best = int(np.argmax(overlaps))
        if overlaps[best] > 0:
            return masks[best], obj_ids[best]

    areas = [int(m.sum()) for m in masks]
    best = int(np.argmax(areas))
    return masks[best], obj_ids[best]


def clean_object_mask(mask, prev_mask, open_px):
    """Sever thin attachments from an object mask and keep one component.

    A poured liquid stream or a utensil handle joins the object as a thin
    bridge that inflates the silhouette and drags the centroid. Opening with a
    small kernel cuts such bridges; among the resulting components the one
    overlapping the previous frame's mask wins (largest as fallback), dropping
    the severed parts. Guarded twice: if the opening erases most of the mask
    -- a small or distant object can be thinner than the kernel -- the
    original mask is returned untouched.
    """
    if mask is None or open_px <= 0 or not mask.any():
        return mask
    m = mask.astype(np.uint8)
    k = 2 * open_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    if opened.sum() < 0.25 * m.sum():
        return mask
    n, labels = cv2.connectedComponents(opened)
    if n <= 1:
        return mask
    best, best_key = None, (-1, -1)
    for i in range(1, n):
        comp = labels == i
        overlap = int(np.logical_and(comp, prev_mask).sum()) if prev_mask is not None else 0
        key = (overlap, int(comp.sum()))
        if key > best_key:
            best, best_key = comp, key
    return best if best is not None else mask


def mask_for_obj_id(out, obj_id):
    """Mask of one tracked instance, or None if it is absent from this frame."""
    obj_ids = [int(i) for i in out["out_obj_ids"]]
    if obj_id not in obj_ids:
        return None
    return out["out_binary_masks"][obj_ids.index(obj_id)]


def save_chunk_as_video(frames_chunk, tmpdir, chunk_idx, fps):
    """Save a chunk of frames as a temporary MP4 for SAM3 session."""
    chunk_path = os.path.join(tmpdir, f"chunk_{chunk_idx:04d}.mp4")
    H, W = frames_chunk[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(chunk_path, fourcc, fps, (W, H))
    for frame in frames_chunk:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return chunk_path


def segment_prompt_chunked(predictor, frames, text_prompt, chunk_size, fps,
                           single_instance=False, person_masks=None, open_px=0):
    """Segment a text prompt across the video using chunked processing to avoid OOM.

    With single_instance=True, only one tracked instance is kept per frame instead
    of the union of all detections: seeded at each chunk start by overlap with the
    previous chunk's last mask, else by proximity to the person (when person_masks
    is given), else by size. open_px > 0 additionally runs clean_object_mask on
    every stored mask to cut thin attachments like a poured stream.
    """
    num_frames = len(frames)
    all_masks = {}
    prev_mask = None  # last non-empty mask, for continuity across chunks

    tmpdir = tempfile.mkdtemp(prefix="sam3_chunks_")
    try:
        for chunk_start in range(0, num_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_frames)
            chunk_frames = frames[chunk_start:chunk_end]
            chunk_len = len(chunk_frames)

            print(f"    Processing frames {chunk_start}-{chunk_end-1} ({chunk_len} frames)...")

            chunk_path = save_chunk_as_video(chunk_frames, tmpdir, chunk_start, fps)

            # Start session on chunk
            response = predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=chunk_path,
                    offload_video_to_cpu=True,
                )
            )
            session_id = response["session_id"]

            # Add text prompt on frame 0 of chunk
            resp = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=text_prompt,
                )
            )
            # Check if any objects detected
            det_out = resp["outputs"]
            n_detected = len(det_out["out_obj_ids"])
            if n_detected == 0:
                print(f"      Warning: no objects detected for prompt '{text_prompt}' in chunk starting at frame {chunk_start}")
                # Store None for all frames in this chunk
                for i in range(chunk_len):
                    all_masks[chunk_start + i] = None
                predictor.handle_request(request=dict(type="close_session", session_id=session_id))
                torch.cuda.empty_cache()
                continue

            # Store frame 0 mask from detection
            tracked_id = None
            if single_instance:
                seed_person = None if person_masks is None else person_masks.get(chunk_start)
                mask, tracked_id = select_single_mask(det_out, ref_mask=prev_mask,
                                                      person_mask=seed_person)
                print(f"      Tracking instance {tracked_id} of {n_detected} detected")
            else:
                mask = merge_masks_from_output(det_out)
            mask = clean_object_mask(mask, prev_mask, open_px)
            all_masks[chunk_start] = mask
            if mask is not None and mask.any():
                prev_mask = mask

            # Propagate through chunk
            for resp in predictor.handle_stream_request(
                request=dict(type="propagate_in_video", session_id=session_id)
            ):
                fi = resp["frame_index"]
                if single_instance:
                    mask = mask_for_obj_id(resp["outputs"], tracked_id)
                else:
                    mask = merge_masks_from_output(resp["outputs"])
                mask = clean_object_mask(mask, prev_mask, open_px)
                all_masks[chunk_start + fi] = mask
                if mask is not None and mask.any():
                    prev_mask = mask

            # Close session to free GPU memory
            predictor.handle_request(request=dict(type="close_session", session_id=session_id))
            torch.cuda.empty_cache()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Fill any missing frames with None
    for i in range(num_frames):
        if i not in all_masks:
            all_masks[i] = None

    return all_masks


def save_masks_h5(human_masks, object_masks, output_path, seq_name, kid, frame_shape):
    """Save masks to HDF5 in CARI4D format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    H, W = frame_shape[:2]

    with h5py.File(output_path, "w") as f:
        grp = f.create_group(seq_name)
        num_frames = len(human_masks)
        for frame_idx in range(num_frames):
            frame_id = f"{frame_idx:06d}"
            hm = human_masks.get(frame_idx)
            if hm is None:
                hm = np.zeros((H, W), dtype=bool)
            grp.create_dataset(
                f"{frame_id}-k{kid}.person_mask.png", data=hm.astype(bool)
            )
            om = object_masks.get(frame_idx)
            if om is None:
                om = np.zeros((H, W), dtype=bool)
            grp.create_dataset(
                f"{frame_id}-k{kid}.obj_rend_mask.png", data=om.astype(bool)
            )
    print(f"Saved masks to {output_path} ({num_frames} frames)")


PERSON_RGB = np.array([255, 0, 0])      # overlay colour for the human mask
OBJECT_RGB = np.array([0, 0, 255])      # overlay colour for the object mask
ALPHA = 0.5
STRIP_H = 18                            # height of the tracked/lost timeline bar


def _mask_or_empty(masks, idx, shape):
    """masks[idx], or an all-False mask. A dropped track is an EMPTY mask here,
    not a missing key, so downstream code can treat both the same way."""
    m = masks.get(idx)
    if m is None:
        return np.zeros(shape, dtype=bool)
    return m.astype(bool)


def mask_areas(human_masks, object_masks, num_frames, shape):
    """Per-frame pixel counts for both masks, as (person, object) int arrays."""
    per = np.array([_mask_or_empty(human_masks, i, shape).sum() for i in range(num_frames)])
    obj = np.array([_mask_or_empty(object_masks, i, shape).sum() for i in range(num_frames)])
    return per, obj


def find_tracked_runs(good, gap_tolerance=0):
    """Contiguous runs of True in `good` as inclusive (start, end) index pairs.

    gap_tolerance bridges short dropouts so a single lost frame does not split a
    long usable stretch. Bridged frames stay in the run with their empty masks,
    which is a real cost -- hence a default of 0, so bridging is always a choice
    the caller made explicitly.
    """
    runs, start, gap, last_good = [], None, 0, None
    for i, v in enumerate(good):
        if v:
            if start is None:
                start, gap = i, 0
            else:
                gap = 0
            last_good = i
        elif start is not None:
            gap += 1
            if gap > gap_tolerance:
                runs.append((start, last_good))
                start = None
    if start is not None:
        runs.append((start, last_good))
    return runs


def object_bbox(mask, pad=12):
    """Padded (x0, y0, x1, y1) around a mask, or None when it is empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(W, xs.max() + 1 + pad), min(H, ys.max() + 1 + pad))


def timeline_strip(good, width, height=STRIP_H, cursor=None):
    """Whole-sequence tracked/lost bar: green tracked, dark red lost, yellow cursor.

    A column is green only if EVERY frame it covers is tracked, so a one-frame
    dropout stays visible instead of being averaged into a healthy-looking bar.
    """
    n = len(good)
    edges = (np.arange(width + 1) * n / width).astype(int)
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        strip[:, i] = (60, 190, 90) if good[lo:hi].all() else (150, 30, 30)
    if cursor is not None and n > 0:
        c = min(width - 1, int(cursor * width / n))
        strip[:, max(0, c - 1):c + 2] = (255, 220, 0)
    return strip


def zoom_inset(img, bbox, size=140):
    """Nearest-neighbour magnification of bbox, letterboxed into size x size.

    The object can be tiny (a basketball is ~18x18px in a 796x448 frame); at 1x
    you cannot tell whether the mask is on the object, on a shadow, or on
    nothing -- which is the whole question when judging a take.
    """
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    k = max(1, int(min(size / max(w, 1), size / max(h, 1))))
    big = np.repeat(np.repeat(crop, k, axis=0), k, axis=1)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    bh, bw = min(size, big.shape[0]), min(size, big.shape[1])
    out[:bh, :bw] = big[:bh, :bw]
    return out


def _hud(img, lines):
    """Small text block with a dark plate behind it, readable over any frame."""
    for i, text in enumerate(lines):
        y = 16 + i * 16
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (4, y - th - 4), (8 + tw, y + 4), (0, 0, 0), -1)
        cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_visualization(frames, human_masks, object_masks, output_path, fps=30,
                       zoom=False, zoom_size=140, min_person_px=1, min_object_px=1):
    """Side-by-side visualization: left=RGB, right=RGB+masks overlay.

    Overlay colours are unchanged (human red, object blue, alpha 0.5). Added on
    top: a per-frame HUD (frame index, both mask areas, TRACKED/LOST), a timeline
    strip showing the whole sequence's tracked/lost pattern with a cursor, and an
    optional magnified inset around the object.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, format='FFMPEG', fps=fps)

    shape = frames[0].shape[:2]
    n = len(frames)
    per, obj = mask_areas(human_masks, object_masks, n, shape)
    good = (per >= min_person_px) & (obj >= min_object_px)
    out_w = frames[0].shape[1] * 2

    for idx, frame in enumerate(frames):
        overlay = frame.copy()
        hm = _mask_or_empty(human_masks, idx, shape)
        om = _mask_or_empty(object_masks, idx, shape)
        if hm.any():
            overlay[hm] = (overlay[hm] * (1 - ALPHA) + PERSON_RGB * ALPHA).astype(np.uint8)
        if om.any():
            overlay[om] = (overlay[om] * (1 - ALPHA) + OBJECT_RGB * ALPHA).astype(np.uint8)

        if zoom:
            bb = object_bbox(om)
            if bb is not None:
                cv2.rectangle(overlay, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 255), 1)
                ins = zoom_inset(overlay, bb, zoom_size)
                if ins is not None:
                    overlay[0:ins.shape[0], overlay.shape[1] - ins.shape[1]:] = ins

        overlay = _hud(overlay, [f"f{idx}  {'TRACKED' if good[idx] else 'LOST'}",
                                 f"person {int(per[idx])}px",
                                 f"object {int(obj[idx])}px"])
        combined = np.concatenate([frame, overlay], axis=1)
        combined = np.concatenate(
            [combined, timeline_strip(good, out_w, STRIP_H, cursor=idx)], axis=0)
        writer.append_data(combined)

    writer.close()
    print(f"Saved visualization to {output_path}")


def save_trimmed(frames, human_masks, object_masks, lo, hi, seq_name, kid,
                 out_dir, fps=30, video_subdir="trimmed_vids"):
    """Write frames [lo, hi] as a normal sequence: a clip and its masks.

    Deliberately NOT a special artifact -- no suffix, no manifest. The trimmed
    masks ARE the masks (`<seq>_masks_k<kid>.h5` in out_dir, so masks_root is
    unchanged), and the clip keeps the plain `<seq>.0.color.mp4` name. Downstream
    cannot tell a trim happened; it is just a shorter take.

    The clip goes in its own `video_subdir` because demo-custom.sh writes into
    dirname(video) -- unidepth's output lands there (`-o ${video_dir}`) and a
    sibling `<video_dir>-aligned/` is created. Pointing that at the masks folder
    would fill it with depth output. It also cannot go next to the source video,
    which is typically read-only shared storage.
    """
    vid_dir = os.path.join(out_dir, video_subdir)
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(vid_dir, f"{seq_name}.0.color.mp4")
    out_h5 = os.path.join(out_dir, f"{seq_name}_masks_k{kid}.h5")

    # macro_block_size=1 or imageio silently resizes to a multiple of 16, and
    # the clip stops matching its own masks. The egoexo4d 448 videos are 796
    # wide, which became 800 -- so stage 4 of the pipeline died on
    #   mask_gt = mask_h & (dmap_gt > 0)
    #   ValueError: operands could not be broadcast together with
    #               shapes (448,796) (448,800)
    # after UniDepth produced depth at the clip's padded width. The masks here
    # are written at the source resolution, so the clip must be too.
    writer = imageio.get_writer(out_video, fps=fps, macro_block_size=1)
    for i in range(lo, hi + 1):
        writer.append_data(frames[i])
    writer.close()

    # Renumber from 0 so the sequence is self-consistent.
    hm = {j: human_masks.get(i) for j, i in enumerate(range(lo, hi + 1))}
    om = {j: object_masks.get(i) for j, i in enumerate(range(lo, hi + 1))}
    save_masks_h5(hm, om, out_h5, seq_name, kid, frames[0].shape)

    print(f"Saved clip  -> {out_video}  ({hi - lo + 1} frames, source frames {lo}-{hi})")
    print(f"Saved masks -> {out_h5}")
    print(f"Run the pipeline on it with: bash scripts/demo-custom.sh {out_video}")
    return out_video, out_h5


def main():
    args = parse_args()

    # Set HF token
    hf_token = "" # args.hf_token or os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = ""
    assert hf_token is not None, "HF_TOKEN is not set"

    # Derive paths
    seq_name = extract_seq_name(args.video)
    if args.output_dir is None:
        video_dir = os.path.dirname(args.video)
        args.output_dir = os.path.join(os.path.dirname(video_dir), "masks")

    h5_path = os.path.join(args.output_dir, f"{seq_name}_masks_k{args.kid}.h5")

    print(f"Video: {args.video}")
    print(f"Sequence: {seq_name}")
    print(f"Human prompt: {args.human_prompt}")
    print(f"Object prompt: {args.object_prompt}")
    print(f"Output: {h5_path}")
    print(f"Chunk size: {args.chunk_size} frames")

    # Load video frames
    print("Loading video frames...")
    frames = load_video_frames(args.video)
    num_frames = len(frames)
    H, W = frames[0].shape[:2]
    print(f"Loaded {num_frames} frames ({W}x{H})")

    # Get fps
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # Build SAM3 predictor
    print("Building SAM3 video predictor...")
    gpus_to_use = list(range(torch.cuda.device_count()))
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    # Segment human (chunked) — always a single instance, CARI4D fits one body
    print(f"Segmenting human: '{args.human_prompt}'...")
    human_masks = segment_prompt_chunked(predictor, frames, args.human_prompt, args.chunk_size, fps,
                                         single_instance=True)
    human_count = sum(1 for m in human_masks.values() if m is not None and m.any())
    print(f"  Human masks found in {human_count}/{num_frames} frames")

    # Segment object (chunked). One instance near the person by default: the
    # union of every look-alike in the scene is not a trajectory.
    print(f"Segmenting object: '{args.object_prompt}' (select={args.object_select}, "
          f"open={args.object_open}px)...")
    object_masks = segment_prompt_chunked(
        predictor, frames, args.object_prompt, args.chunk_size, fps,
        single_instance=args.object_select != "union",
        person_masks=human_masks if args.object_select == "near-person" else None,
        open_px=args.object_open)
    obj_count = sum(1 for m in object_masks.values() if m is not None and m.any())
    print(f"  Object masks found in {obj_count}/{num_frames} frames")

    # Shutdown predictor
    predictor.shutdown()

    # --- Tracking report. The per-mask counts above do not say whether the two
    # hold AT THE SAME TIME, which is what decides whether the take is usable. ---
    shape = frames[0].shape[:2]
    per, obj = mask_areas(human_masks, object_masks, num_frames, shape)
    good = (per >= args.trim_min_person_px) & (obj >= args.trim_min_object_px)
    runs = sorted(find_tracked_runs(good, args.trim_gap_tolerance),
                  key=lambda r: -(r[1] - r[0] + 1))
    print(f"\nBoth masks present in {int(good.sum())}/{num_frames} frames "
          f"({100.0 * good.mean():.1f}%)")
    print(f"Longest contiguous runs (gap_tolerance={args.trim_gap_tolerance}):")
    for r, (lo, hi) in enumerate(runs[:5], start=1):
        n = hi - lo + 1
        print(f"   #{r} frames {lo}-{hi}  {n} frames  {n / fps:.1f}s")
    if not runs:
        print("   NONE -- the human and the object are never tracked together.")

    # Visualization of the WHOLE take -- it is the diagnostic for where tracking
    # holds, so trimming it first would hide exactly what you want to see.
    if args.visualize:
        vis_path = os.path.join(args.output_dir, f"{seq_name}_sam3_vis.mp4")
        save_visualization(frames, human_masks, object_masks, vis_path, fps=fps,
                           zoom=args.zoom, zoom_size=args.zoom_size,
                           min_person_px=args.trim_min_person_px,
                           min_object_px=args.trim_min_object_px)

    # --- Masks are written ONCE, and by default they are the trimmed ones: the
    # full-length masks are not wanted, only the stretch that is actually usable.
    if args.trim_to_tracked and runs:
        if args.trim_rank > len(runs):
            print(f"ERROR: --trim_rank {args.trim_rank} but only {len(runs)} runs exist.",
                  file=sys.stderr)
            sys.exit(1)
        lo, hi = runs[args.trim_rank - 1]
        covered = float(good[lo:hi + 1].mean())
        print(f"\nTrimming to run #{args.trim_rank}: source frames {lo}-{hi} "
              f"({hi - lo + 1} frames, {(hi - lo + 1) / fps:.1f}s, "
              f"both masks present in {100 * covered:.1f}% of them)")
        save_trimmed(frames, human_masks, object_masks, lo, hi, seq_name, args.kid,
                     args.output_dir, fps=fps)
    else:
        # Either trimming is off, or nothing is usable. Save the full masks
        # regardless -- an expensive SAM3 run must never end with nothing on disk.
        if args.trim_to_tracked and not runs:
            print("\n" + "!" * 72, file=sys.stderr)
            print("NOT TRIMMED: the human and the object are never tracked in the same "
                  "frame.", file=sys.stderr)
            print("Saving the FULL-length masks instead so the run is not wasted. Check "
                  "the", file=sys.stderr)
            print("visualization, then retry with different prompts or a larger "
                  "--trim_gap_tolerance.", file=sys.stderr)
            print("!" * 72 + "\n", file=sys.stderr)
        save_masks_h5(human_masks, object_masks, h5_path, seq_name, args.kid,
                      frames[0].shape)

    print("Done!")


if __name__ == "__main__":
    main()
