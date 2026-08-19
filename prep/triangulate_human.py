# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Triangulate the person's COCO joints from per-view Sapiens keypoints.

The monocular SMPL fit gets the human's depth wrong by tens of centimetres
(measured: ~0.4 m at 6.5 m on the basketball opening), which no image-space
loss can see -- sliding the body along the viewing ray leaves every pixel
unchanged. Two calibrated views resolve it, exactly as they did for the
object: this triangulates each of the 17 COCO joints per frame from every
view whose detector was confident there, giving a metric, model-free human
anchor for the optimizer (w_j3d / j3d_file in opt_refineout).

Views are Sapiens outputs (<packed_root>/<seq>_GT-packed.pkl) computed on the
ORIGINAL fisheye clips -- unprojection uses the Kannala-Brandt model, so do
not feed keypoints detected on rectified videos. Each --view names the clip
the keypoints were computed on; its resolution is read from the video itself
and the calibration is scaled per view, so 448p and 4K views mix freely.

Usage:
    python prep/triangulate_human.py \\
        --calib <trajectory>/gopro_calibs.csv --packed_root packed-mv \\
        --view cam04:sam3masks/trimmed_vids/<seq>.0.color.mp4 \\
        --view cam01:sam3masks/trimmed_vids/cam01-4k.0.color.mp4 \\
        --to_cam cam04 --out human_joints.npz

The first --view is the reference: output frames use its numbering (use
--frame_offset per view when clips are trimmed to different ranges). With
--to_cam, the npz additionally carries the joints in that camera's frame,
which is what the optimizer consumes (pose_abs and smpl_t live there).
"""
import argparse
import os
import os.path as osp
import sys

import cv2
import joblib
import numpy as np

sys.path.append(os.getcwd())
from prep.triangulate_object import (pixel_to_ray, read_calibration,
                                     reprojection_error)

COCO_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder",
              "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
              "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def parse_args():
    """Parse the calibration, per-view keypoint sources and gating thresholds."""
    parser = argparse.ArgumentParser(
        description="Triangulate COCO joints from Sapiens keypoints in calibrated views")
    parser.add_argument("--calib", required=True,
                        help="gopro_calibs.csv from the take's trajectory/ directory")
    parser.add_argument("--packed_root", required=True,
                        help="directory holding <seq>_GT-packed.pkl per view")
    parser.add_argument("--view", action="append", required=True,
                        help="<cam_uid>:<video path the keypoints were computed on>, "
                             "repeatable; at least two. The sequence name and the "
                             "resolution both come from the video")
    parser.add_argument("--frame_offset", action="append", default=None,
                        help="per-view frame offset, same order as --view "
                             "(default: 0 for every view)")
    parser.add_argument("--min_conf", type=float, default=0.3,
                        help="skip a view's joint below this detector confidence "
                             "(default: 0.3)")
    parser.add_argument("--max_residual", type=float, default=10.0,
                        help="drop a joint whose mean reprojection error exceeds this "
                             "many reference-scale pixels (default: 10.0)")
    parser.add_argument("--width", type=int, default=796,
                        help="reference width residuals are expressed in (default: 796)")
    parser.add_argument("--to_cam", default=None,
                        help="also store joints in this camera's frame (what the "
                             "optimizer consumes)")
    parser.add_argument("--out", default=None, help="write the result to this .npz")
    parser.add_argument("--max_rows", type=int, default=25,
                        help="rows of per-frame detail to print (default: 25)")
    return parser.parse_args()


def load_view(spec, packed_root, calib, cams_by_res):
    """Build a view dict from '<cam_uid>:<video path>': keypoints, camera, scale.

    The packed pkl is looked up by the video's sequence name; the video itself
    supplies the pixel resolution the keypoints live in, so the calibration is
    scaled to match without anything stated on the command line.

    Raises:
        SystemExit: on a malformed spec, missing pkl/video, or unknown camera.
    """
    parts = spec.split(":")
    if len(parts) != 2:
        raise SystemExit(f"ERROR: --view must be <cam_uid>:<video path>, got {spec}")
    uid, video = parts
    seq = osp.basename(video).split(".")[0]
    pkl = osp.join(packed_root, f"{seq}_GT-packed.pkl")
    if not osp.isfile(pkl):
        raise SystemExit(f"ERROR: no keypoints at {pkl} (run Sapiens on {video} first)")
    cap = cv2.VideoCapture(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w == 0:
        raise SystemExit(f"ERROR: cannot read resolution from {video}")
    if (w, h) not in cams_by_res:
        cams_by_res[(w, h)] = read_calibration(calib, w, h)
    cams = cams_by_res[(w, h)]
    if uid not in cams:
        raise SystemExit(f"ERROR: {uid} not in the calibration; found {sorted(cams)}")
    data = joblib.load(pkl)
    joints = np.asarray(data["joints2d"])[:, 0]        # (N, 17, 3)
    frames = [int(str(f).split("/")[-1].split("-")[0]) for f in data["frames"]] \
        if "frames" in data else list(range(len(joints)))
    kps = {f: joints[i] for i, f in enumerate(frames)}
    return {"uid": uid, "seq": seq, "cam": cams[uid], "kps": kps, "res": (w, h)}


def triangulate_rays_weighted(origins, directions, weights):
    """Confidence-weighted least-squares closest point to a set of rays.

    Same construction as triangulate_object.triangulate_rays, with each ray's
    perpendicular-distance term scaled by the detector confidence, so a shaky
    detection pulls less than a sure one.

    Returns:
        (point, weighted mean perpendicular distance in metres).
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d, w in zip(origins, directions, weights):
        P = (np.eye(3) - np.outer(d, d)) * w
        A += P
        b += P @ o
    point = np.linalg.lstsq(A, b, rcond=None)[0]
    dists = [np.linalg.norm(np.cross(d, point - o))
             for o, d in zip(origins, directions)]
    return point, float(np.average(dists, weights=weights))


def main():
    """Triangulate every joint of every frame and report coverage and residuals."""
    args = parse_args()
    if len(args.view) < 2:
        raise SystemExit("ERROR: at least two --view arguments are needed")
    offsets = [int(o) for o in (args.frame_offset or [])]
    offsets += [0] * (len(args.view) - len(offsets))

    cams_by_res = {}
    views = []
    for spec, offset in zip(args.view, offsets):
        v = load_view(spec, args.packed_root, args.calib, cams_by_res)
        v["offset"] = offset
        n_conf = sum(1 for k in v["kps"].values() if (k[:, 2] >= args.min_conf).any())
        print(f"{v['uid']} ({v['seq']}, {v['res'][0]}x{v['res'][1]}): "
              f"{len(v['kps'])} frames, {n_conf} with a confident joint, offset {offset}")
        views.append(v)

    ref = views[0]
    n_joints = 17
    frames_out, joints_w, valid, residuals = [], [], [], []
    for idx in sorted(ref["kps"]):
        pts = np.zeros((n_joints, 3))
        ok = np.zeros(n_joints, dtype=bool)
        res = np.full(n_joints, np.inf)
        for j in range(n_joints):
            origins, dirs, weights, obs, seen = [], [], [], [], set()
            for v in views:
                if v["uid"] in seen:
                    continue
                key = idx - ref["offset"] + v["offset"]
                if key not in v["kps"]:
                    continue
                u, vv, conf = v["kps"][key][j]
                if conf < args.min_conf:
                    continue
                seen.add(v["uid"])
                o, d = pixel_to_ray((float(u), float(vv)), v["cam"])
                origins.append(o)
                dirs.append(d)
                weights.append(float(conf))
                obs.append(((float(u), float(vv)), v["cam"], args.width / v["res"][0]))
            if len(origins) < 2:
                continue
            point, _ = triangulate_rays_weighted(origins, dirs, weights)
            errs = [reprojection_error(point, uv, cam) * s for uv, cam, s in obs]
            res[j] = float(np.mean(errs))
            if res[j] <= args.max_residual:
                pts[j] = point
                ok[j] = True
        frames_out.append(idx)
        joints_w.append(pts)
        valid.append(ok)
        residuals.append(res)

    frames_out = np.array(frames_out)
    joints_w = np.array(joints_w)
    valid = np.array(valid)
    residuals = np.array(residuals)

    per_joint = valid.sum(axis=0)
    print(f"\nframes: {len(frames_out)}   joints triangulated per frame: "
          f"median {int(np.median(valid.sum(axis=1)))} of {n_joints}")
    print("per-joint coverage: " + ", ".join(
        f"{n}:{c}" for n, c in zip(COCO_NAMES, per_joint)))
    finite = residuals[np.isfinite(residuals) & valid]
    if finite.size:
        print(f"residuals (kept joints): median {np.median(finite):.2f} px, "
              f"90th pct {np.percentile(finite, 90):.2f} px (at {args.width}-px scale)")

    step = max(1, len(frames_out) // args.max_rows)
    print(f"\n{'frame':>6} {'#joints':>8} {'pelvis_z(m)':>12} {'nose_z(m)':>10}")
    pelvis = 0.5 * (joints_w[:, 11] + joints_w[:, 12])
    for i in range(0, len(frames_out), step):
        pz = pelvis[i][2] if valid[i, 11] and valid[i, 12] else float("nan")
        nz = joints_w[i, 0][2] if valid[i, 0] else float("nan")
        print(f"{frames_out[i]:>6} {int(valid[i].sum()):>8} {pz:>12.3f} {nz:>10.3f}")

    out = {"frames": frames_out, "joints_world": joints_w,
           "valid": valid, "residual": residuals,
           "joint_names": np.array(COCO_NAMES)}
    if args.to_cam:
        cams = next(iter(cams_by_res.values()))
        if args.to_cam not in cams:
            raise SystemExit(f"ERROR: --to_cam {args.to_cam} not in the calibration")
        cam = cams[args.to_cam]
        jc = (cam["R_cw"] @ joints_w.reshape(-1, 3).T).T + cam["t_cw"]
        out["joints_cam"] = jc.reshape(joints_w.shape)
        out["cam_uid"] = np.array(args.to_cam)
    if args.out:
        np.savez(args.out, **out)
        print(f"\nwrote {args.out} "
              f"({'world + ' + args.to_cam + ' camera frame' if args.to_cam else 'world frame'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
