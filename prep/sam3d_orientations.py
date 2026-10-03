# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""The object's orientation at keyframes, from SAM 3D Objects, for the tracker.

FoundationPose turned the pot backwards: a pot is symmetric apart from its
handle, the hand covers the handle, and texture matching cannot tell the two
ends apart. SAM 3D Objects can, because it reads the picture as a pot. Run on
a keyframe (prep/run_sam3d_recon.py --frames ... --pose_only) it returns its
own pot plus that pot's rotation in the crop's camera. This script turns
those into the rotation of OUR mesh in the clip's camera, one per keyframe,
which fp_behave.py --orient_file then holds the track to.

Two things make that a transform rather than a lookup:

  the generated pot is not our pot   Each keyframe's pot is aligned to the
                                     canonical pot (the one the mesh came
                                     from) by ICP. SAM 3D places every pot in
                                     the same canonical orientation, so this
                                     starts from identity and converges in a
                                     few degrees; the residual is reported.
  the crop is not the camera         The pose is for a square crop centred on
                                     the mask. The crop's optical axis is the
                                     ray through that centre, so the rotation
                                     is turned by the angle between that ray
                                     and the camera's axis.

The remaining conventions -- quaternion order, inverse, an axis swap between
SAM 3D's internal frame and the exported GLB, a flip of the camera axes --
were not documented and were found by trying every combination against the
ego masks on five keyframes of the pot clip; the one below matched in all
five. If SAM 3D changes, re-check with the overlay this writes.

Writes an .npz with 'frames' (clip frame indices) and 'R' (N, 3, 3): the
rotation of the mesh given by --mesh, as stored, in the camera of --video.
With --viz, an image per keyframe of the mesh at that orientation over the
frame, which is the check to look at.

Usage:
    python prep/sam3d_orientations.py --keyframes_root work/<seq>/meshes-ego-keys \\
        --canonical_dir work/<seq>/meshes-ego/<seq>-ego_180_rgba --mesh <metric _align.obj> \\
        --video work/<seq>/rect-ego/<seq>-ego.0.color.mp4 --masks_root work/<seq>/rect-ego \\
        --out work/<seq>/fp-ego/orientations.npz --viz
"""
import argparse
import os
import os.path as osp
import re
import sys
from glob import glob

import h5py
import joblib
import numpy as np

sys.path.append(os.getcwd())

# SAM 3D's pose convention, as found on the pot clip (see the module docstring).
QUAT_ORDER = "wxyz"
INVERT = True
PRE = ("x", 90.0)                 # SAM 3D internal frame -> exported GLB frame
POST = np.diag([-1.0, -1.0, 1.0])  # SAM 3D camera axes -> OpenCV camera axes
VIEW_FIX = True                    # turn by the crop centre's ray


def parse_args():
    """Parse the keyframe outputs, the canonical mesh, the clip and the output."""
    parser = argparse.ArgumentParser(
        description="Per-keyframe object orientation from SAM 3D Objects, for the tracker")
    parser.add_argument("--keyframes_root", required=True,
                        help="hy3d_root of run_sam3d_recon.py --frames: <seq>_<frame>_rgba/ per keyframe")
    parser.add_argument("--canonical_dir", required=True,
                        help="the folder the tracked mesh came from; its .glb is SAM 3D's own copy "
                             "of the object, which the keyframes are aligned to")
    parser.add_argument("--mesh", required=True,
                        help="the _align.obj the tracker uses; the rotations written are for it")
    parser.add_argument("--video", required=True, help="the clip, with its .pkl intrinsics beside it")
    parser.add_argument("--masks_root", required=True, help="directory holding <seq>_masks_k<kid>.h5")
    parser.add_argument("--kid", type=int, default=0)
    parser.add_argument("--out", required=True, help=".npz to write")
    parser.add_argument("--max_residual", type=float, default=0.01,
                        help="drop a keyframe whose pot does not align to the canonical pot "
                             "within this ICP cost (normalised units; default: 0.01)")
    parser.add_argument("--viz", action="store_true", help="write an overlay sheet beside --out")
    return parser.parse_args()


def normalised(mesh):
    """A copy centred on its bounding box with the longest axis scaled to 1."""
    import trimesh
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v = v - (v.min(0) + v.max(0)) / 2.0
    v = v / np.ptp(v, axis=0).max()
    return trimesh.Trimesh(v, mesh.faces, process=False)


def rotation_of(M):
    """The nearest rotation to the 3x3 part of a transform."""
    U, _, Vt = np.linalg.svd(M[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def angle_deg(R):
    """Rotation angle of R in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def align_from_identity(src, dst):
    """ICP of src onto dst from an identity start; returns (rotation, cost)."""
    import trimesh
    r = trimesh.registration.icp(src.sample(4000), dst, initial=np.eye(4),
                                 max_iterations=100, scale=False)
    return rotation_of(r[0]), float(r[-1])


def align_multistart(src, dst):
    """ICP of src onto dst from trimesh's principal-axis starts; returns (rotation, cost)."""
    import trimesh
    M, cost = trimesh.registration.mesh_other(src, dst, samples=2000, scale=False,
                                              icp_first=30, icp_final=100)
    return rotation_of(M), float(cost)


def view_rotation(ray):
    """The rotation taking the optical axis onto a unit ray."""
    from scipy.spatial.transform import Rotation
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, ray)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(axis / s * np.arctan2(s, float(z @ ray))).as_matrix()


def sam3d_rotation(q):
    """SAM 3D's quaternion as a rotation matrix under the convention found."""
    from scipy.spatial.transform import Rotation
    q = np.asarray(q, dtype=np.float64).reshape(4)
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]] if QUAT_ORDER == "wxyz" else q).as_matrix()
    return R.T if INVERT else R


def mask_bbox_centre(masks_root, prefix, kid, frame):
    """Centre of the object mask's bounding box, the crop's centre; None without a mask."""
    with h5py.File(osp.join(masks_root, f"{prefix}_masks_k{kid}.h5"), "r") as h5:
        grp = h5[prefix] if prefix in h5 else h5
        key = f"{frame:06d}-k{kid}.obj_rend_mask.png"
        if key not in grp:
            return None
        m = grp[key][()].astype(bool)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0


def main():
    """Align every keyframe, compose the rotations, write the file and the check."""
    import trimesh
    from scipy.spatial.transform import Rotation

    args = parse_args()
    prefix = osp.basename(args.video).split(".")[0]
    cam = joblib.load(args.video.replace(".mp4", ".pkl"))
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])

    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    glbs = sorted(glob(osp.join(args.canonical_dir, "*.glb")))
    if glbs:
        canonical = normalised(trimesh.load(glbs[0], force="mesh", process=False))
        R_canon_from_mesh, cost = align_multistart(normalised(mesh), canonical)
        print(f"tracked mesh -> canonical GLB {osp.basename(glbs[0])}: {angle_deg(R_canon_from_mesh):.1f} deg, cost {cost:.4f}")
    else:
        # No SAM 3D copy of the object (a Hunyuan3D mesh): the keyframes are
        # aligned to the tracked mesh itself, from several starts, which for a
        # near-symmetric object can settle on the wrong one.
        canonical = normalised(mesh)
        R_canon_from_mesh = np.eye(3)
        print(f"no .glb in {args.canonical_dir}: aligning keyframes to the tracked mesh from "
              f"several starts (less reliable on a symmetric object)")

    frames, rotations, rows, dropped = [], [], [], []
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)_rgba$")
    for kdir in sorted(glob(osp.join(args.keyframes_root, f"{prefix}_*_rgba"))):
        m = pattern.search(osp.basename(kdir))
        pose_files = glob(osp.join(kdir, "*_sam3d_pose.npz"))
        glb_files = glob(osp.join(kdir, "*.glb"))
        if not m or not pose_files or not glb_files:
            continue
        frame = int(m.group(1))
        keyframe = normalised(trimesh.load(glb_files[0], force="mesh", process=False))
        if glbs:
            R_key_from_canon, cost = align_from_identity(canonical, keyframe)
        else:
            R_key_from_canon, cost = align_multistart(canonical, keyframe)
        if cost > args.max_residual:
            dropped.append((frame, cost))
            continue
        q = np.load(pose_files[0])["rotation"]
        R = POST @ sam3d_rotation(q) @ Rotation.from_euler(*PRE, degrees=True).as_matrix() \
            @ R_key_from_canon @ R_canon_from_mesh
        if VIEW_FIX:
            centre = mask_bbox_centre(args.masks_root, prefix, args.kid, frame)
            if centre is None:
                dropped.append((frame, float("nan")))
                continue
            ray = np.linalg.inv(K) @ np.array([centre[0], centre[1], 1.0])
            R = view_rotation(ray / np.linalg.norm(ray)) @ R
        frames.append(frame)
        rotations.append(R)
        rows.append((frame, angle_deg(R_key_from_canon), cost))

    if not frames:
        raise SystemExit(f"ERROR: no usable keyframe under {args.keyframes_root}")
    frames, rotations = np.array(frames), np.stack(rotations)
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    np.savez(args.out, frames=frames, R=rotations, mesh=osp.abspath(args.mesh))

    print(f"{len(frames)} keyframes: {frames[0]}..{frames[-1]}")
    print("  frame  align-to-canonical  icp-cost")
    for frame, ang, cost in rows:
        print(f"  {frame:5d}  {ang:6.1f} deg          {cost:.4f}")
    if len(frames) > 1:
        steps = [angle_deg(rotations[i].T @ rotations[i + 1]) for i in range(len(frames) - 1)]
        print(f"  rotation between consecutive keyframes: median {np.median(steps):.1f} deg, "
              f"max {max(steps):.1f} deg")
    if dropped:
        print(f"  dropped {len(dropped)} keyframe(s) whose pot did not align: "
              + ", ".join(f"{f} ({c:.3f})" for f, c in dropped[:8]))
    print(f"wrote {args.out}")

    if args.viz:
        write_sheet(args, prefix, K, mesh, frames, rotations)
    return 0


def write_sheet(args, prefix, K, mesh, frames, rotations):
    """An image per keyframe: the mesh at its orientation, on the mask's ray, over the frame."""
    import cv2
    import imageio
    from prep.render_object_masks import silhouette

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    Vh = np.concatenate([V, np.ones((len(V), 1))], axis=1)
    mc = V.mean(0)
    reader = imageio.get_reader(args.video)
    tiles = []
    with h5py.File(osp.join(args.masks_root, f"{prefix}_masks_k{args.kid}.h5"), "r") as h5:
        grp = h5[prefix] if prefix in h5 else h5
        for frame, R in zip(frames, rotations):
            mask = grp[f"{frame:06d}-k{args.kid}.obj_rend_mask.png"][()].astype(bool)
            ys, xs = np.nonzero(mask)
            ray = np.linalg.inv(K) @ np.array([xs.mean(), ys.mean(), 1.0])
            # distance from the mask's size: a crude stand-in, the tracker sets the real one
            z = K[0, 0] * max(mesh.extents) / max(np.ptp(xs), np.ptp(ys), 1)
            P = np.eye(4)
            P[:3, :3] = R
            P[:3, 3] = ray / ray[2] * z - R @ mc
            img = np.ascontiguousarray(reader.get_data(int(frame))[:, :, :3]).copy()
            sil = silhouette((P @ Vh.T).T[:, :3], F, K, img.shape[0], img.shape[1])
            for m, colour in ((sil, (255, 0, 255)), (mask, (0, 255, 0))):
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, colour, 3)
            cx, cy = int(xs.mean()), int(ys.mean())
            h = max(200, int(1.5 * max(np.ptp(xs), np.ptp(ys))))
            y0, x0 = np.clip(cy - h, 0, max(img.shape[0] - 2 * h, 0)), np.clip(cx - h, 0, max(img.shape[1] - 2 * h, 0))
            tile = cv2.resize(img[y0:y0 + 2 * h, x0:x0 + 2 * h], (400, 400))
            cv2.putText(tile, str(frame), (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            tiles.append(tile)
    reader.close()
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    path = args.out.replace(".npz", "_check.jpg")
    imageio.imwrite(path, sheet)
    print(f"wrote {path} (magenta: mesh at the SAM 3D orientation; green: the mask)")


if __name__ == "__main__":
    raise SystemExit(main())
