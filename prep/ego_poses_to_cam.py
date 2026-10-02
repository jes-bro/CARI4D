# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Carry object poses tracked in the ego camera into the pipeline camera.

FoundationPose on the rectified ego clip (prep/rectify_aria.py) gives, per
frame, the object's pose in the Aria RGB camera -- the camera the calibration
describes, since the rectified image keeps its orientation. The Aria moves, so
that camera has a different place in the world every frame; aria_extrinsics.json
holds it, keyed by take frame, in the same world as gopro_calibs.csv. So

    pose_pipe(t) = T_pipe<-world . T_world<-ego(t) . pose_ego(t)

puts each pose into the fixed pipeline camera, where CoCoNet and the optimizer
expect FoundationPose's output. The pickle written here has the shape they
read ({'fp_poses': (T, K, 4, 4), 'frames': [...]}) so nothing downstream
knows the object was never tracked in their camera.

Also written: the object's centre per frame in world coordinates, in the file
layout of prep/triangulate_object.py, so prep/inject_object_depth.py and
prep/derive_knobs.py work unchanged from it.

The two cameras' clips must cover the same take frames (the drivers cut both
to the same window), so a frame key means the same instant in both.

Usage:
    python prep/ego_poses_to_cam.py --fp_pkl work/<seq>/fp-ego/<seq>-ego_all.pkl \\
        --calib <trajectory>/gopro_calibs.csv --cam cam01 \\
        --aria_extrinsics <trajectory>/aria_extrinsics.json --offset 5190 \\
        --mesh work/<seq>/meshes-ego-metric/<...>_align.obj \\
        --out_pkl work/<seq>/fp/<seq>_all.pkl --out_xyz work/<seq>/geom/object_xyz_ego.npz
"""
import argparse
import os
import os.path as osp
import re
import sys

import joblib
import numpy as np

sys.path.append(os.getcwd())
from prep.aria_camera import load_extrinsics
from prep.triangulate_object import read_calibration


def parse_args():
    """Parse the ego poses, both calibrations, the offset, the mesh and outputs."""
    parser = argparse.ArgumentParser(
        description="Transform FoundationPose output from the ego camera to the pipeline camera")
    parser.add_argument("--fp_pkl", required=True, help="<ego seq>_all.pkl from fp_hy3d_track.py")
    parser.add_argument("--calib", required=True, help="gopro_calibs.csv")
    parser.add_argument("--cam", required=True, help="pipeline camera uid, e.g. cam01")
    parser.add_argument("--aria_extrinsics", required=True, help="aria_extrinsics.json")
    parser.add_argument("--offset", type=int, required=True,
                        help="take frame of the clip's frame 0 (window.json chosen.lo)")
    parser.add_argument("--mesh", required=True,
                        help="the metric mesh the poses were tracked with; its vertex "
                             "mean is the object centre written to --out_xyz")
    parser.add_argument("--out_pkl", required=True, help="FP-format pickle for the pipeline sequence")
    parser.add_argument("--out_xyz", required=True, help="object_xyz-format .npz, world coordinates")
    return parser.parse_args()


def to_homogeneous(R, t):
    """4x4 from a 3x3 rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def frame_index(key):
    """The integer frame behind an FP frame key ('000180', or 't0180.000')."""
    m = re.search(r"\d+", str(key))
    if not m:
        raise SystemExit(f"ERROR: cannot read a frame index from key {key!r}")
    return int(m.group(0))


def main():
    """Transform every tracked pose and write both outputs."""
    import trimesh

    args = parse_args()
    data = joblib.load(args.fp_pkl)
    poses = np.asarray(data["fp_poses"], dtype=np.float64)  # (T, K, 4, 4)
    frames = list(data["frames"])
    if poses.ndim != 4 or poses.shape[1] != 1:
        raise SystemExit(f"ERROR: expected (T, 1, 4, 4) poses, got {poses.shape}")

    # Any resolution works for the extrinsics; the intrinsics are not used.
    cams = read_calibration(args.calib, 796, 448)
    if args.cam not in cams:
        raise SystemExit(f"ERROR: {args.cam} not in {args.calib}; have {sorted(cams)}")
    T_pipe_world = to_homogeneous(cams[args.cam]["R_cw"], cams[args.cam]["t_cw"])
    ext = load_extrinsics(args.aria_extrinsics)

    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    centre_mesh = np.append(mesh.vertices.mean(axis=0), 1.0)

    out_poses, out_frames, xyz_frames, xyz = [], [], [], []
    missing_pose, no_track = [], []
    for i, key in enumerate(frames):
        pose_ego = poses[i, 0]
        if not np.isfinite(pose_ego).all():
            no_track.append(key)
            continue
        take_idx = frame_index(key) + args.offset
        if take_idx not in ext:
            missing_pose.append(take_idx)
            continue
        T_ego_world = to_homogeneous(*ext[take_idx])
        T_world_ego = np.linalg.inv(T_ego_world)
        pose_pipe = T_pipe_world @ T_world_ego @ pose_ego
        out_poses.append(pose_pipe[None])
        out_frames.append(key)
        xyz_frames.append(frame_index(key))
        xyz.append((T_world_ego @ pose_ego @ centre_mesh)[:3])

    if not out_poses:
        raise SystemExit("ERROR: no frame had both a tracked pose and an Aria pose")
    out_poses = np.stack(out_poses)
    xyz = np.array(xyz)

    os.makedirs(osp.dirname(osp.abspath(args.out_pkl)), exist_ok=True)
    os.makedirs(osp.dirname(osp.abspath(args.out_xyz)), exist_ok=True)
    out = dict(data)
    out["fp_poses"] = out_poses
    out["frames"] = out_frames
    out["source"] = {"ego_fp_pkl": osp.abspath(args.fp_pkl), "pipeline_cam": args.cam,
                     "offset": args.offset}
    joblib.dump(out, args.out_pkl)
    np.savez(args.out_xyz, frames=np.array(xyz_frames), xyz=xyz,
             residual=np.zeros(len(xyz_frames)))

    depth_pipe = out_poses[:, 0, :3, 3] @ np.array([0, 0, 1.0])
    kept = [i for i, k in enumerate(frames) if k in out_frames]
    depth_ego = poses[kept, 0, 2, 3]
    print(f"{len(out_frames)}/{len(frames)} frames carried into {args.cam}")
    print(f"  object distance from the ego camera: {depth_ego.min():.2f}-{depth_ego.max():.2f} m "
          f"(median {np.median(depth_ego):.2f}) -- a handheld object should be well under 1 m")
    if no_track:
        print(f"  {len(no_track)} frame(s) had no ego pose (FoundationPose lost the object)")
    if missing_pose:
        print(f"  {len(missing_pose)} frame(s) had no Aria pose, e.g. take frame {missing_pose[0]}")
    print(f"  object distance from {args.cam}: {depth_pipe.min():.2f}-{depth_pipe.max():.2f} m "
          f"(median {np.median(depth_pipe):.2f})")
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(xyz) > 1 else np.zeros(1)
    print(f"  object motion per frame: median {np.median(step):.3f} m, max {step.max():.3f} m")
    print(f"wrote {args.out_pkl}")
    print(f"wrote {args.out_xyz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
