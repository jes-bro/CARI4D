# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Draw a mask set over its clip, to see where a mask actually sits.

A diagnostic with no model in it: the clip's frames with the object mask's
outline in magenta and the person mask's in green. Pointed at the rectified
pipeline-camera masks after scripts/slurm_ego_object.sh rewrote them, it
shows where the ego track puts the object in that camera BEFORE CoCoNet and
the optimizer touch anything -- which is how to tell an ego-track or
extrinsics error (outline off the object here) from a refinement error
(outline on the object here, render off it).

A second H5 can be drawn in yellow for comparison, e.g. the exo SAM3 object
masks kept as <name>.exo.h5.

Usage:
    python prep/overlay_masks.py --video work/<seq>/rect/<seq>.0.color.mp4 \\
        --masks_h5 work/<seq>/rect/<seq>_masks_k0.h5 \\
        --compare_h5 work/<seq>/rect/<seq>_masks_k0.exo.h5 \\
        --out work/<seq>/rect/<seq>_ego_object_overlay.mp4
"""
import argparse
import os
import os.path as osp
import sys

import cv2
import h5py
import imageio
import numpy as np

sys.path.append(os.getcwd())


def parse_args():
    """Parse the clip, the mask file(s) and the output video."""
    parser = argparse.ArgumentParser(description="Overlay mask outlines on a clip")
    parser.add_argument("--video", required=True, help="<seq>.<kid>.color.mp4")
    parser.add_argument("--masks_h5", required=True, help="<seq>_masks_k<kid>.h5 to draw")
    parser.add_argument("--compare_h5", default=None,
                        help="a second mask file; its object masks are drawn in yellow")
    parser.add_argument("--out", required=True, help="output .mp4")
    parser.add_argument("--kid", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def open_group(h5):
    """The group holding the masks: the sequence group if there is one, else the root."""
    groups = [g for g in h5 if isinstance(h5[g], h5py.Group)]
    return h5[groups[0]] if groups else h5


def read_mask(grp, idx, kid, kind):
    """One boolean mask, or None when the frame has none."""
    key = f"{idx:06d}-k{kid}.{kind}.png"
    return grp[key][()].astype(bool) if key in grp else None


def draw_outline(frame, mask, color, thickness=2):
    """Draw the mask's contours on the frame in place; returns the pixel count."""
    if mask is None or not mask.any():
        return 0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, thickness)
    return int(mask.sum())


def main():
    """Write the overlay video and print how many frames had each mask."""
    args = parse_args()
    reader = imageio.get_reader(args.video)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1)
    n_obj = n_cmp = n = 0
    with h5py.File(args.masks_h5, "r") as f, \
            (h5py.File(args.compare_h5, "r") if args.compare_h5 else open(os.devnull)) as g:
        grp = open_group(f)
        cmp_grp = open_group(g) if args.compare_h5 else None
        for idx, frame in enumerate(reader):
            frame = np.ascontiguousarray(frame[:, :, :3])
            draw_outline(frame, read_mask(grp, idx, args.kid, "person_mask"), (0, 255, 0), 1)
            if cmp_grp is not None:
                n_cmp += draw_outline(frame, read_mask(cmp_grp, idx, args.kid, "obj_rend_mask"),
                                      (255, 255, 0), 2) > 0
            n_obj += draw_outline(frame, read_mask(grp, idx, args.kid, "obj_rend_mask"),
                                  (255, 0, 255), 2) > 0
            cv2.putText(frame, f"{idx}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.append_data(frame)
            n += 1
    writer.close()
    print(f"{n} frames -> {args.out}")
    print(f"  magenta (object, {osp.basename(args.masks_h5)}): {n_obj} frames with a mask")
    if args.compare_h5:
        print(f"  yellow (object, {osp.basename(args.compare_h5)}): {n_cmp} frames with a mask")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
