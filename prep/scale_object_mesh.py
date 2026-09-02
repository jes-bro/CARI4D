"""Give the reconstructed mesh its metric size, by measuring it in every view.

A single-image reconstruction has no scale in it. A photograph of a basketball
and one of a beach ball are identical up to size, so Hunyuan3D returns shape in
arbitrary units normalised to [-1, 1] and the metres have to come from
somewhere else. The released pipeline searches for them: render the mesh at
thirty candidate sizes and keep whichever best explains one image. With one
camera that is the only option, because size and distance are degenerate in a
single silhouette.

This pipeline is not single-camera. The object's 3D position is triangulated
from several calibrated views to centimetre accuracy, and a distance plus an
apparent size IS a size:

    D = Z * d_px / f

Every view and every frame gives an independent measurement of the same
physical quantity -- some hundreds of them -- so the mask noise averages out and
the disagreement between them is itself readable. That is a measurement rather
than an inference, and it is general: nothing here knows or cares what the
object is.

The shape stays the reconstruction's. Only the size is measured.

MATCHING LIKE WITH LIKE. What the cameras measure is the equivalent-disc
diameter of a silhouette AREA, from whatever direction they happen to see. That
is not a bounding-box extent, so pairing it against one -- the mesh's longest
axis, or its median extent -- compares two different kinds of quantity, and no
choice of percentile repairs that.

So the mesh is asked for the same statistic: project it from many random
directions, take each silhouette's equivalent-disc diameter, and match the
median of those against the median of the observations. Both sides are then the
same measurement of the same object and the ratio is its size.

The difference only shows on objects that are not round, which is why a
basketball cannot tell you whether the rule is right. Measured against known
shapes, recovering the longest axis:

    object                 true    silhouette   median extent   p90/longest
    sphere                 0.240      -0.0%         -0.0%          -0.0%
    box 0.30x0.20x0.15     0.300      +0.9%        +49.3%          +5.7%
    cylinder d0.24 h0.16   0.240      +0.9%        +9.6%          +14.2%
    rod 0.50x0.04x0.04     0.500      -0.3%       +315.0%         -62.8%
    pot with a handle      0.270      +1.0%           --              --

CAVEAT. The random-direction average assumes the cameras see the object from
varied angles. Four fixed cameras watching an object that barely rotates see a
restricted set of directions, and the observed median then reflects those
rather than an average over all of them. Exact for a sphere at any orientation;
good when the object turns; biased when it presents one face throughout.

Run from the repo root in the cari4d env.

Usage:
    python prep/scale_object_mesh.py --mesh work/<seq>/meshes/<d>/<n>_align.obj \\
        --object_xyz work/<seq>/geom/object_xyz.npz \\
        --calib <take>/trajectory/gopro_calibs.csv \\
        --view cam04:work/<seq>/rect:<seq> \\
        --view cam01:work/<seq>/masks:cam01-4k \\
        --out_root work/<seq>/meshes-metric
"""
import argparse
import json
import os
import os.path as osp
import shutil
import sys

import numpy as np

sys.path.append(os.getcwd())

# Masks run a few percent wide, so every apparent width is slightly inflated
# and the object comes out a little large. Reported, not corrected: the bias
# depends on the segmentation, and a fudge factor here would be exactly the
# hand-tuned constant this file exists to remove.
SIZE_PCT = 50

# Directions used to ask the mesh what it would measure. A few hundred is
# enough for a stable median and costs a second.
MESH_VIEW_DIRECTIONS = 300
MESH_VERTEX_SAMPLE = 2000


def parse_args():
    """Parse the mesh, the triangulated positions, and the views to measure in."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True, help="normalised mesh from Hunyuan3D")
    p.add_argument("--object_xyz", required=True,
                   help="triangulate_object.py output, world-frame positions")
    p.add_argument("--calib", required=True, help="gopro_calibs.csv")
    p.add_argument("--view", action="append", required=True,
                   help="<cam_uid>:<masks_root>:<seq_name>, repeatable")
    p.add_argument("--out_root", required=True,
                   help="where the metric copy goes (the -metric directory)")
    p.add_argument("--kid", type=int, default=0)
    p.add_argument("--pct", type=float, default=SIZE_PCT,
                   help="percentile of the per-view measurements to use "
                        "(default: 50, the median)")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def measure_in_view(cam_uid, masks_root, seq, kid, calib, frames, xyz):
    """Per-frame size measurements from one view, in metres.

    Reads the mask at that view's own resolution and scales the calibration to
    match, so a 448p view and a 4K one contribute on equal terms. Returns an
    empty list when the view shares no frames with the triangulation.
    """
    from prep.triangulate_object import load_object_centroids, read_calibration
    centroids, shape = load_object_centroids(masks_root, seq, kid, min_px=4)
    if not centroids:
        return []
    cams = read_calibration(calib, shape[1], shape[0])
    if cam_uid not in cams:
        raise SystemExit(f"ERROR: {cam_uid} not in {calib}")
    cam = cams[cam_uid]
    f = float((cam["K"][0, 0] + cam["K"][1, 1]) / 2.0)

    out = []
    for frame, p in zip(frames, xyz):
        if frame not in centroids:
            continue
        area = centroids[frame][2]
        d_px = 2.0 * np.sqrt(area / np.pi)
        p_cam = cam["R_cw"] @ p + cam["t_cw"]
        z = float(np.linalg.norm(p_cam))
        out.append(z * d_px / f)
    return out


def mesh_apparent_diameters(mesh, n=MESH_VIEW_DIRECTIONS, seed=0):
    """What the mesh's silhouette would measure, from many random directions.

    The same statistic the cameras produce -- equivalent-disc diameter of the
    silhouette area -- so that the observed and predicted distributions can be
    compared directly. Vertices are subsampled because the hull of 40k points,
    three hundred times, is slower than it needs to be for a median.
    """
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(seed)
    v = np.asarray(mesh.vertices)
    if len(v) > MESH_VERTEX_SAMPLE:
        v = v[rng.choice(len(v), MESH_VERTEX_SAMPLE, replace=False)]
    v = v - v.mean(0)
    out = []
    for _ in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        proj = (v @ q.T)[:, :2]
        try:
            out.append(2.0 * np.sqrt(ConvexHull(proj).volume / np.pi))
        except Exception:
            continue
    return np.array(out)


def scaled_copy(mesh_file, out_root, scale, dry_run=False):
    """Write a metric copy of the mesh, preserving its material and texture.

    The whole source directory is copied and only the vertex lines rewritten,
    so the .mtl and texture the OBJ references travel with it -- FoundationPose
    matches rendered appearance, so losing the texture would quietly cost
    tracking quality.
    """
    src_dir = osp.dirname(mesh_file)
    name = osp.basename(mesh_file)
    dst_dir = osp.join(out_root, osp.splitext(name)[0])
    if not dry_run:
        if osp.isdir(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

    dst = osp.join(dst_dir, name)
    lines_out = []
    with open(mesh_file) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                xyz = [float(v) * scale for v in parts[1:4]]
                lines_out.append("v %.6f %.6f %.6f\n" % tuple(xyz))
            else:
                lines_out.append(line)
    if not dry_run:
        with open(dst, "w") as f:
            f.writelines(lines_out)
        # A flat copy beside it: downstream globs have been seen to pick either
        # layout, and writing both costs nothing.
        shutil.copy(dst, osp.join(out_root, name))
    return dst


def main():
    """Measure the object's size across views and write the metric mesh."""
    args = parse_args()
    obj = np.load(args.object_xyz)
    frames, xyz = obj["frames"].astype(int), obj["xyz"]

    all_d, per_view = [], {}
    for spec in args.view:
        cam_uid, masks_root, seq = spec.split(":")
        d = measure_in_view(cam_uid, masks_root, seq, args.kid, args.calib,
                            frames, xyz)
        if d:
            per_view[cam_uid] = d
            all_d.extend(d)
    if not all_d:
        raise SystemExit("ERROR: no view shares frames with the triangulation; "
                         "nothing to measure")

    print(f"{'view':<10} {'frames':>7} {'median m':>10} {'p90 m':>8}")
    print("-" * 40)
    for cam_uid, d in per_view.items():
        a = np.array(d)
        print(f"{cam_uid:<10} {len(a):>7} {np.median(a):>10.3f} "
              f"{np.percentile(a, args.pct):>8.3f}")

    a = np.array(all_d)
    size_m = float(np.percentile(a, args.pct))
    print(f"\n{len(a)} measurements across {len(per_view)} view(s)")
    print(f"  median {np.median(a):.3f} m, p{args.pct:g} {size_m:.3f} m, "
          f"spread {np.percentile(a, 10):.3f}-{np.percentile(a, 90):.3f} m")
    if np.percentile(a, 90) > 2.0 * np.percentile(a, 10):
        print("  NOTE: the views disagree by more than 2x. Either the object is "
              "strongly elongated,\n        or a mask is picking up something "
              "else -- check the per-view rows above.")

    import trimesh
    mesh = trimesh.load(args.mesh, process=False)
    # Ask the mesh for the same statistic the cameras produced, then match the
    # two medians. Both sides are now the equivalent-disc diameter of a
    # silhouette, so the ratio is a size rather than a comparison of unlike
    # quantities.
    pred = mesh_apparent_diameters(mesh)
    if pred.size == 0:
        raise SystemExit("ERROR: could not project the mesh; is it degenerate?")
    pred_typical = float(np.median(pred))
    scale = size_m / pred_typical
    print(f"\nmesh extents {np.round(mesh.extents, 3)} units")
    print(f"  its silhouette would measure {pred_typical:.3f} units "
          f"(median over {len(pred)} directions)")
    print(f"observed {size_m:.3f} m -> scale {scale:.4f}, "
          f"longest axis {max(mesh.extents) * scale:.3f} m")

    os.makedirs(args.out_root, exist_ok=True)
    dst = scaled_copy(args.mesh, args.out_root, scale, args.dry_run)
    print(f"wrote {dst}")
    if not args.dry_run:
        meta = {"measured_size_m": size_m, "scale": scale,
                "percentile": args.pct, "n_measurements": int(len(a)),
                "per_view_median_m": {k: float(np.median(v))
                                      for k, v in per_view.items()},
                "source_mesh": osp.abspath(args.mesh)}
        seq_prefix = osp.basename(args.mesh).split("_")[0]
        with open(osp.join(args.out_root, "object_scale.json"), "w") as f:
            json.dump(meta, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
