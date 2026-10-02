# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Measure a monocular depth map's global scale from an object of known size.

UniDepth on the ego clip returns depth with a free overall scale, and the
first ego run showed what that costs: the pot, sixty centimetres from the
glasses, was placed metres away, because FoundationPose seeds its translation
from the median depth inside the mask and refines against the depth map.
Nothing in the ego view can anchor that scale -- no human of known height, no
second camera -- except the object itself, once its size is known.

So: per frame, the mask's equivalent-disc diameter in pixels against the
mesh's apparent diameter in metres gives the depth the object must be at,
z = f * D / d. The ratio of that to the depth map's median inside the mask
is the correction, and the median over frames is the global factor. The
apparent diameter is the same statistic prep/scale_object_mesh.py uses, so
the pixels and the metres describe the same thing.

Prints a short report; --factor_only prints the one number, for a shell.

Usage:
    python prep/estimate_depth_scale.py --video work/<seq>/rect-ego/<seq>-ego.0.color.mp4 \\
        --masks_root work/<seq>/rect-ego --mesh work/<seq>/meshes-ego-metric/<...>_align.obj
"""
import argparse
import os
import os.path as osp
import sys

import h5py
import joblib
import numpy as np

sys.path.append(os.getcwd())


def parse_args():
    """Parse the clip, its masks, the metric mesh and the sampling stride."""
    parser = argparse.ArgumentParser(
        description="Global scale factor for a monocular depth map, from the object's size")
    parser.add_argument("--video", required=True, help="<seq>.<kid>.color.mp4 with its .pkl and depth-reg beside it")
    parser.add_argument("--masks_root", required=True, help="directory holding <seq>_masks_k<kid>.h5")
    parser.add_argument("--mesh", required=True, help="the metric mesh (its apparent diameter is the size used)")
    parser.add_argument("--kid", type=int, default=0)
    parser.add_argument("--stride", type=int, default=5, help="sample every Nth frame (default: 5)")
    parser.add_argument("--min_px", type=int, default=200, help="skip frames with a smaller mask")
    parser.add_argument("--factor_only", action="store_true", help="print only the factor")
    return parser.parse_args()


def main():
    """Compare observed and expected object depth per frame; print the factor."""
    import trimesh
    from behave_data.video_reader import ColorDepthController
    from prep.scale_object_mesh import mesh_apparent_diameters

    args = parse_args()
    say = (lambda *a, **k: None) if args.factor_only else print

    prefix = osp.basename(args.video).split(".")[0]
    cam = joblib.load(args.video.replace(".mp4", ".pkl"))
    f = float((cam["fx"] + cam["fy"]) / 2.0)
    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    D = float(np.median(mesh_apparent_diameters(mesh)))
    say(f"mesh apparent diameter {D:.3f} m (longest axis {max(mesh.extents):.3f} m), f={f:.1f} px")

    ctrl = ColorDepthController(osp.join(osp.dirname(args.video), prefix), args.kid)
    rows = []
    with h5py.File(osp.join(args.masks_root, f"{prefix}_masks_k{args.kid}.h5"), "r") as h5:
        grp = h5[prefix] if prefix in h5 else h5
        keys = sorted(k for k in grp if k.endswith(f"-k{args.kid}.obj_rend_mask.png"))
        for key in keys[::args.stride]:
            idx = int(key.split("-")[0])
            mask = grp[key][()].astype(bool)
            area = int(mask.sum())
            if area < args.min_px:
                continue
            _, depth = ctrl.get_closest_frame(float(idx))
            depth = depth.astype(np.float32) / 1000.0
            vals = depth[mask & (depth > 0.001)]
            if vals.size < 10:
                continue
            d_px = 2.0 * np.sqrt(area / np.pi)
            z_expected = f * D / d_px
            z_observed = float(np.median(vals))
            rows.append((idx, area, z_expected, z_observed, z_expected / z_observed))

    if not rows:
        raise SystemExit("ERROR: no frame had both an object mask and depth inside it")
    r = np.array(rows)
    factor = float(np.median(r[:, 4]))
    say(f"{len(rows)} frames sampled (stride {args.stride})")
    say(f"  depth map inside the mask: median {np.median(r[:, 3]):.2f} m "
        f"(range {r[:, 3].min():.2f}-{r[:, 3].max():.2f})")
    say(f"  depth the object's size implies: median {np.median(r[:, 2]):.2f} m "
        f"(range {r[:, 2].min():.2f}-{r[:, 2].max():.2f})")
    say(f"  per-frame ratio: median {factor:.3f}, 10th-90th pct "
        f"{np.percentile(r[:, 4], 10):.3f}-{np.percentile(r[:, 4], 90):.3f}")
    if np.percentile(r[:, 4], 90) / max(np.percentile(r[:, 4], 10), 1e-6) > 2.0:
        say("  WARNING: the ratio varies a lot across frames; one global factor is a rough fit")
    say(f"depth scale factor: {factor:.4f}")
    if args.factor_only:
        print(f"{factor:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
