# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Drive tools/pyt3d_wrapper.py to get the blue-body / salmon-object figure.

That look -- the one in the BEHAVE and CARI4D papers -- comes from
SMPL_OBJ_COLOR_LIST and MeshRendererWrapper in tools/pyt3d_wrapper.py. Both are
already in this repo and nothing calls them. viz_pred.py cannot produce it: it
draws a UV-textured body composited over the footage, which is a different
figure for a different purpose.

This is a driver, not a renderer. Every decision about lighting, shading and
rasterisation stays in pyt3d_wrapper; this only poses the meshes, colours them
from that list, and hands them over.

    python tools/render_behave_style.py -pf <pred>.pth --out figure.mp4 \\
        --hy3d_meshes_root data/cari4d-demo/meshes-metric

The camera is the one the reconstruction was made in -- the same intrinsics
viz_pred's working front panel uses, with the axis flip get_kinect_camera
applies between camera coordinates and pytorch3d's. Nothing about the viewpoint
is invented here, which is the point: every earlier attempt at a fresh viewpoint
got an axis wrong and produced a body lying down or sinking.
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())


def parse_args():
    """Parse the prediction file, the meshes and the output."""
    parser = argparse.ArgumentParser(
        description="Render a prediction in the BEHAVE paper style")
    parser.add_argument("-pf", "--pred_file", required=True)
    parser.add_argument("--key", default="pr", choices=["pr", "gt", "in"])
    parser.add_argument("--out", required=True, help="output .mp4")
    parser.add_argument("--seq", default=None,
                        help="sequence name (default: the file's stem)")
    parser.add_argument("--gender", default="male", choices=["male", "female"])
    parser.add_argument("--hy3d_meshes_root", default=None)
    parser.add_argument("--width", type=int, default=796,
                        help="video width the intrinsics belong to")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--render_scale", type=float, default=2.0,
                        help="multiple of the source resolution to render at "
                             "(default: 2, i.e. 1592x896 for a 796x448 video). "
                             "The intrinsics are scaled to match, so the "
                             "projection stays the camera's. Rendering square "
                             "instead puts the principal point in the wrong "
                             "place and the subject drifts off frame.")
    parser.add_argument("--bg", default="1,1,1",
                        help="background r,g,b in 0-1 (default: white)")
    parser.add_argument("--intrinsics", default="401.74728,401.15918,401.4052,228.35431",
                        help="fx,fy,cx,cy for the video the reconstruction was "
                             "made in. The default is what the pipeline printed "
                             "for this take; check your own run's log.")
    parser.add_argument("--max_faces_per_bin", type=int, default=300000,
                        help="pytorch3d coarse-rasterisation budget (default: "
                             "300000). MeshRendererWrapper defaults to 50000, "
                             "which a reconstructed object overflows -- and an "
                             "overflow silently drops faces rather than failing.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def vec3(text):
    """Parse 'a,b,c' into three floats.

    Raises:
        SystemExit: on anything that is not three numbers.
    """
    parts = [p for p in str(text).replace(" ", "").split(",") if p]
    if len(parts) != 3:
        raise SystemExit(f"expected 'r,g,b', got {text!r}")
    return [float(p) for p in parts]


def load_prediction(path, key):
    """Return (smpl_pose, betas, smpl_t, pose_abs).

    Raises:
        SystemExit: if the file or any required field is missing.
    """
    if not osp.isfile(path):
        raise SystemExit(f"no prediction file at {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if key not in data:
        raise SystemExit(f"'{key}' not in {osp.basename(path)}; got {list(data)}")
    block = data[key]
    for field in ("smpl_pose", "betas", "smpl_t", "pose_abs"):
        if field not in block:
            raise SystemExit(f"'{key}' has no '{field}'; got {list(block)}")
    return (block["smpl_pose"].float(), block["betas"].float(),
            block["smpl_t"].float(), block["pose_abs"].float())


def object_verts(seq, meshes_root):
    """Return the object's centroid-centred vertices and faces.

    CARI4D subtracts the mesh centroid before applying pose_abs
    (viz_pred.py:153), so pose_abs locates the centroid. Reproducing that here
    is the difference between an object in the subject's hands and one beside
    them.

    Raises:
        SystemExit: if no mesh is found.
    """
    import trimesh
    from behave_data.const import get_hy3d_mesh_file

    path = get_hy3d_mesh_file(seq, meshes_root=meshes_root)
    if path is None:
        raise SystemExit(f"no aligned Hy3D mesh for {seq}; set --hy3d_meshes_root")
    mesh = trimesh.load(path, force="mesh", process=False)
    v = np.asarray(mesh.vertices, dtype=np.float32)
    return v - v.mean(axis=0), np.asarray(mesh.faces, dtype=np.int64)


def reconstruction_camera(K, width, height, device):
    """Return the camera the reconstruction was made in, as pytorch3d wants it.

    Modelled on pyt3d_wrapper.get_kinect_camera, including its R: pytorch3d's
    conventions differ from a vision camera's by a flip of the first two axes,
    which is what R[0,0] = R[1,1] = -1 expresses. Taking that from their code
    rather than deriving it is deliberate -- deriving it is what produced a
    subject lying on their side.
    """
    from pytorch3d.renderer import PerspectiveCameras

    R = torch.eye(3)
    R[0, 0] = R[1, 1] = -1
    focal = torch.tensor((float(K[0, 0]), float(K[1, 1])),
                         dtype=torch.float32).unsqueeze(0)
    centre = torch.tensor((float(K[0, 2]), float(K[1, 2])),
                          dtype=torch.float32).unsqueeze(0)
    return PerspectiveCameras(focal_length=focal, principal_point=centre,
                              image_size=((height, width),), device=device,
                              R=R.unsqueeze(0), T=torch.zeros(3).unsqueeze(0),
                              in_ndc=False)


def main():
    """Pose each frame and render it through pyt3d_wrapper."""
    args = parse_args()
    if args.hy3d_meshes_root:
        os.environ["HY3D_MESHES_ROOT"] = args.hy3d_meshes_root

    import imageio
    from pytorch3d.renderer import PointLights, TexturesVertex
    from pytorch3d.structures import Meshes
    from tools.pyt3d_wrapper import MeshRendererWrapper, SMPL_OBJ_COLOR_LIST
    from lib_smpl import get_smpl

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seq = args.seq or osp.splitext(osp.basename(args.pred_file))[0]
    pose, betas, trans, pose_abs = load_prediction(args.pred_file, args.key)
    pose, betas, trans, pose_abs = (pose[::args.stride], betas[::args.stride],
                                    trans[::args.stride], pose_abs[::args.stride])
    T = len(pose)
    print(f"{seq}: {T} frames, key={args.key}, device={device}")

    # The bundle may store 72-dim SMPL or 156-dim SMPL-H, and get_smpl(.., True)
    # builds SMPL-H. viz_pred.py:196 makes the same conversion; without it the
    # blend-shape term is 207 wide against the model's 459.
    from lib_smpl import pose72to156
    if pose.shape[1] == 72:
        pose = pose72to156(pose)
        print(f"converted 72-dim SMPL pose to {pose.shape[1]}-dim SMPL-H")
    elif pose.shape[1] != 156:
        raise SystemExit(f"smpl_pose is {pose.shape[1]} wide; expected 72 or 156")

    body = get_smpl(args.gender, True).to(device)
    with torch.no_grad():
        verts_b, _, _, _ = body(pose.to(device), betas.to(device), trans.to(device))
    faces_b = torch.as_tensor(np.asarray(body.faces, dtype=np.int64), device=device)

    ov, of = object_verts(seq, args.hy3d_meshes_root)
    ov_t = torch.as_tensor(ov, device=device)
    verts_o = (torch.matmul(ov_t[None], pose_abs[:, :3, :3].to(device).permute(0, 2, 1))
               + pose_abs[:, :3, 3].to(device)[:, None])
    faces_o = torch.as_tensor(of, device=device)
    print(f"body {verts_b.shape[1]} verts, object {ov.shape[0]} verts")

    parts = [p for p in args.intrinsics.replace(" ", "").split(",") if p]
    if len(parts) != 4:
        raise SystemExit(f"--intrinsics wants fx,fy,cx,cy; got {args.intrinsics!r}")
    fx, fy, cx, cy = (float(p) for p in parts)
    # Scaled with the render, so the projection remains the camera's own.
    sc = float(args.render_scale)
    out_w, out_h = int(round(args.width * sc)), int(round(args.height * sc))
    K = np.array([[fx * sc, 0.0, cx * sc], [0.0, fy * sc, cy * sc], [0.0, 0.0, 1.0]])
    print(f"source {args.width}x{args.height}, rendering {out_w}x{out_h} "
          f"(x{sc:g})")
    print(f"intrinsics fx={fx * sc:.1f} fy={fy * sc:.1f} "
          f"c=({cx * sc:.1f}, {cy * sc:.1f})")
    cameras = reconstruction_camera(K, out_w, out_h, device)

    smpl_rgb = torch.tensor(SMPL_OBJ_COLOR_LIST[0], dtype=torch.float32,
                            device=device)
    obj_rgb = torch.tensor(SMPL_OBJ_COLOR_LIST[1], dtype=torch.float32,
                           device=device)
    print(f"colours: body {SMPL_OBJ_COLOR_LIST[0]}, object {SMPL_OBJ_COLOR_LIST[1]}")

    lights = PointLights(((0.5, 0.5, 0.5),), ((0.5, 0.5, 0.5),),
                         ((0.05, 0.05, 0.05),), ((0, -2, 0),), device)
    n_faces = len(faces_b) + len(faces_o)
    print(f"scene has {n_faces} faces; bin budget {args.max_faces_per_bin}")
    if n_faces > args.max_faces_per_bin:
        print("  WARNING: the budget is below the face count. pytorch3d will "
              "warn about coarse-rasterisation overflow and DROP faces, which "
              "shows as holes rather than an error. Raise --max_faces_per_bin.")
    # (H, W), not a square: RasterizationSettings takes a tuple, and a square
    # render with non-square intrinsics misplaces the principal point.
    renderer = MeshRendererWrapper(image_size=(out_h, out_w), device=device,
                                   lights=lights,
                                   max_faces_per_bin=args.max_faces_per_bin)
    bg = np.array(vec3(args.bg), dtype=np.float32)

    os.makedirs(osp.dirname(osp.abspath(args.out)) or ".", exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1)
    for i in range(T):
        # One mesh, so a single rasterisation resolves body against object.
        v = torch.cat([verts_b[i], verts_o[i]], 0)
        f = torch.cat([faces_b, faces_o + len(verts_b[i])], 0)
        c = torch.cat([smpl_rgb.expand(len(verts_b[i]), 3),
                       obj_rgb.expand(len(verts_o[i]), 3)], 0)
        mesh = Meshes(verts=[v], faces=[f], textures=TexturesVertex([c]))
        rgb, mask = renderer.render(mesh, cameras, ret_mask=True)
        # Composited onto a flat colour rather than left on the renderer's
        # default, so the figure has the plain ground those papers use.
        img = np.where(mask[..., None], rgb, bg[None, None, :])
        writer.append_data((np.clip(img, 0, 1) * 255).astype(np.uint8))
        if i % 20 == 0:
            print(f"  frame {i + 1}/{T}", flush=True)
    writer.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
