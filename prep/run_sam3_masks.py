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
    parser.add_argument("--visualize", action="store_true", help="Save visualization MP4")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token for SAM3 checkpoint access")
    parser.add_argument("--chunk_size", type=int, default=300,
                        help="Process video in chunks of this many frames to avoid OOM (default: 300)")
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


def select_single_mask(out, ref_mask=None):
    """Pick exactly one instance from a SAM3 detection output.

    Prefers the instance overlapping ref_mask the most (keeps the same person
    across chunk boundaries, where object ids are not stable); otherwise falls
    back to the largest instance. Returns (mask, obj_id), or (None, None) if
    nothing was detected.
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

    areas = [int(m.sum()) for m in masks]
    best = int(np.argmax(areas))
    return masks[best], obj_ids[best]


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
                           single_instance=False):
    """Segment a text prompt across the video using chunked processing to avoid OOM.

    With single_instance=True, only one tracked instance is kept per frame instead
    of the union of all detections: the largest instance at the start of the first
    chunk, then whichever instance best overlaps the previous chunk's last mask.
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
                mask, tracked_id = select_single_mask(det_out, ref_mask=prev_mask)
                print(f"      Tracking instance {tracked_id} of {n_detected} detected")
            else:
                mask = merge_masks_from_output(det_out)
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


def save_visualization(frames, human_masks, object_masks, output_path, fps=30):
    """Save side-by-side visualization: left=RGB, right=RGB+masks overlay."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)

    for idx, frame in enumerate(frames):
        overlay = frame.copy()
        hm = human_masks.get(idx)
        if hm is not None and hm.any():
            overlay[hm] = (overlay[hm] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        om = object_masks.get(idx)
        if om is not None and om.any():
            overlay[om] = (overlay[om] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
        combined = np.concatenate([frame, overlay], axis=1)
        writer.append_data(combined)

    writer.close()
    print(f"Saved visualization to {output_path}")


def main():
    args = parse_args()

    # Set HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
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

    # Segment object (chunked)
    print(f"Segmenting object: '{args.object_prompt}'...")
    object_masks = segment_prompt_chunked(predictor, frames, args.object_prompt, args.chunk_size, fps)
    obj_count = sum(1 for m in object_masks.values() if m is not None and m.any())
    print(f"  Object masks found in {obj_count}/{num_frames} frames")

    # Shutdown predictor
    predictor.shutdown()

    # Save H5
    save_masks_h5(human_masks, object_masks, h5_path, seq_name, args.kid, frames[0].shape)

    # Visualization
    if args.visualize:
        vis_path = os.path.join(args.output_dir, f"{seq_name}_sam3_vis.mp4")
        save_visualization(frames, human_masks, object_masks, vis_path, fps=fps)

    print("Done!")


if __name__ == "__main__":
    main()
