# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Render the body and object as clean 3D geometry, for figures.

viz_pred.py composites meshes over the footage and pairs each panel with a side
view. That answers "does the reconstruction explain the pixels", and it is not
what a paper figure wants: the side view is empty on sequences whose geometry
falls outside its fixed framing, and the overlay hides the shape it is meant to
show.

This renders the posed SMPL body and the object alone -- lit, shaded, on a plain
ground, from a viewpoint you choose. No video behind them and no second panel.

    python tools/render_figure.py -pf <pred>.pth --out figure.mp4 \\
        --hy3d_meshes_root data/cari4d-demo/meshes-metric --azim 45 --elev 12

Conventions follow viz_pred exactly, so the geometry is the same one it draws:
the body from get_smpl()(pose, betas, trans), the object as its centroid-centred
vertices rotated and translated by pose_abs.
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())


def parse_args():
    """Parse the prediction file, the viewpoint and the output."""
    parser = argparse.ArgumentParser(
        description="Render posed body + object as clean 3D geometry")
    parser.add_argument("-pf", "--pred_file", required=True)
    parser.add_argument("--key", default="pr", choices=["pr", "gt", "in"],
                        help="which reconstruction to draw (default: pr)")
    parser.add_argument("--out", required=True, help="output .mp4")
    parser.add_argument("--gender", default="male",
                        choices=["male", "female"])
    parser.add_argument("--hy3d_meshes_root", default=None,
                        help="directory holding <seq>*/<seq>*_align.obj")
    parser.add_argument("--seq", default=None,
                        help="sequence name (default: the file's stem)")
    parser.add_argument("--azim", type=float, default=45.0,
                        help="camera azimuth in degrees (default: 45)")
    parser.add_argument("--elev", type=float, default=10.0,
                        help="camera elevation in degrees (default: 10). Low "
                             "reads body orientation better than looking down.")
    parser.add_argument("--orbit", type=float, default=0.0,
                        help="degrees of azimuth to sweep across the clip "
                             "(default: 0, a fixed camera)")
    parser.add_argument("--margin", type=float, default=1.25,
                        help="fraction of extra room around the motion")
    parser.add_argument("--size", type=int, default=1024, help="output edge, px")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--body_color", default="0.75,0.76,0.80")
    parser.add_argument("--object_color", default="0.85,0.45,0.15")
    parser.add_argument("--bg", default="1,1,1",
                        help="background as r,g,b in 0-1 (default: white)")
    parser.add_argument("--no_ground", action="store_true",
                        help="omit the ground plane")
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def vec3(text):
    """Parse 'a,b,c' into a float array.

    Raises:
        SystemExit: on anything that is not three numbers.
    """
    parts = [p for p in str(text).replace(" ", "").split(",") if p]
    if len(parts) != 3:
        raise SystemExit(f"expected 'r,g,b', got {text!r}")
    return np.array([float(p) for p in parts], dtype=np.float32)


def load_prediction(path, key):
    """Return (smpl_pose, betas, smpl_t, pose_abs) as tensors.

    Raises:
        SystemExit: if the file or the requested sub-dict is missing.
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


def object_base_verts(seq, meshes_root):
    """Return the object's vertices and faces, centred as CARI4D poses them.

    CARI4D subtracts the mesh centroid before applying pose_abs, so pose_abs
    locates the centroid rather than the mesh origin. Reproduced here or the
    object sits displaced by the difference.

    Raises:
        SystemExit: if no mesh is found.
    """
    import trimesh
    from behave_data.const import get_hy3d_mesh_file

    path = get_hy3d_mesh_file(seq, meshes_root=meshes_root)
    if path is None:
        raise SystemExit(
            f"no aligned Hy3D mesh for {seq} under "
            f"{meshes_root or os.environ.get('HY3D_MESHES_ROOT', 'the default')}")
    mesh = trimesh.load(path, force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    verts = verts - verts.mean(axis=0)
    return verts, np.asarray(mesh.faces, dtype=np.int64)


def build_renderer(size, bg, device):
    """Return a pytorch3d renderer with soft shading and a key light.

    Soft Phong rather than a hard rasteriser: the point of this render is that
    shape reads clearly, and flat shading on a grey body does not.
    """
    from pytorch3d.renderer import (BlendParams, MeshRasterizer, MeshRenderer,
                                    PointLights, RasterizationSettings,
                                    SoftPhongShader)

    raster = RasterizationSettings(image_size=size, blur_radius=0.0,
                                   faces_per_pixel=1, bin_size=0)
    lights = PointLights(device=device,
                         location=[[2.0, 3.0, 2.0]],
                         ambient_color=[[0.45, 0.45, 0.48]],
                         diffuse_color=[[0.65, 0.65, 0.62]],
                         specular_color=[[0.15, 0.15, 0.15]])
    blend = BlendParams(background_color=tuple(float(c) for c in bg))
    return MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster),
        shader=SoftPhongShader(device=device, lights=lights, blend_params=blend))


def ground_quad(centre, radius, height):
    """Return vertices and faces for a square ground plane under the motion."""
    r = radius * 2.0
    v = np.array([[centre[0] - r, centre[1] - r, height],
                  [centre[0] + r, centre[1] - r, height],
                  [centre[0] + r, centre[1] + r, height],
                  [centre[0] - r, centre[1] + r, height]], dtype=np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return v, f


def main():
    """Pose every frame, render it, and write the video."""
    args = parse_args()
    if args.hy3d_meshes_root:
        os.environ["HY3D_MESHES_ROOT"] = args.hy3d_meshes_root

    import imageio
    from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from lib_smpl import get_smpl

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq = args.seq or osp.splitext(osp.basename(args.pred_file))[0]
    print(f"sequence {seq}, key={args.key}, device={device}")

    pose, betas, trans, pose_abs = load_prediction(args.pred_file, args.key)
    pose, betas, trans, pose_abs = (pose[::args.stride], betas[::args.stride],
                                    trans[::args.stride], pose_abs[::args.stride])
    T = len(pose)
    print(f"{T} frames, pose {tuple(pose.shape)}")

    body = get_smpl(args.gender, True).to(device)
    with torch.no_grad():
        verts_b, _, _, _ = body(pose.to(device), betas.to(device), trans.to(device))
    faces_b = torch.as_tensor(np.asarray(body.faces, dtype=np.int64), device=device)
    print(f"body: {verts_b.shape[1]} verts, {len(faces_b)} faces")

    obj_v, obj_f = object_base_verts(seq, args.hy3d_meshes_root)
    obj_v_t = torch.as_tensor(obj_v, device=device)
    R_o = pose_abs[:, :3, :3].to(device)
    t_o = pose_abs[:, :3, 3].to(device)
    verts_o = torch.matmul(obj_v_t[None], R_o.permute(0, 2, 1)) + t_o[:, None]
    faces_o = torch.as_tensor(obj_f, device=device)
    print(f"object: {obj_v.shape[0]} verts, {len(obj_f)} faces")

    all_v = torch.cat([verts_b.reshape(-1, 3), verts_o.reshape(-1, 3)], 0)
    lo = all_v.min(0).values.cpu().numpy()
    hi = all_v.max(0).values.cpu().numpy()
    centre = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo) / 2.0)
    print(f"extent x {lo[0]:+.2f}..{hi[0]:+.2f}  y {lo[1]:+.2f}..{hi[1]:+.2f}  "
          f"z {lo[2]:+.2f}..{hi[2]:+.2f}")

    body_rgb = torch.as_tensor(vec3(args.body_color), device=device)
    obj_rgb = torch.as_tensor(vec3(args.object_color), device=device)
    ground_rgb = torch.as_tensor(np.array([0.93, 0.93, 0.94], dtype=np.float32),
                                 device=device)

    gv, gf = (None, None)
    if not args.no_ground:
        gv_np, gf_np = ground_quad(centre, radius, float(lo[2]))
        gv = torch.as_tensor(gv_np, device=device)
        gf = torch.as_tensor(gf_np, device=device)

    renderer = build_renderer(args.size, vec3(args.bg), device)
    os.makedirs(osp.dirname(osp.abspath(args.out)) or ".", exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1)

    dist = radius * args.margin / np.tan(np.radians(30.0))
    print(f"camera {dist:.2f} m out, azim {args.azim}, elev {args.elev}, "
          f"orbit {args.orbit}")

    for i in range(T):
        # Concatenated into one mesh per frame rather than rendered separately:
        # a single rasterisation resolves body against object correctly, which
        # separate passes composited afterwards would not.
        vs = [verts_b[i], verts_o[i]]
        fs = [faces_b, faces_o + len(verts_b[i])]
        cs = [body_rgb.expand(len(verts_b[i]), 3),
              obj_rgb.expand(len(verts_o[i]), 3)]
        if gv is not None:
            fs.append(gf + len(verts_b[i]) + len(verts_o[i]))
            vs.append(gv)
            cs.append(ground_rgb.expand(len(gv), 3))
        v = torch.cat(vs, 0)
        f = torch.cat(fs, 0)
        c = torch.cat(cs, 0)

        mesh = Meshes(verts=[v], faces=[f], textures=TexturesVertex([c]))
        azim = args.azim + args.orbit * (i / max(1, T - 1))
        R, Tv = look_at_view_transform(dist=dist, elev=args.elev, azim=azim,
                                       at=((float(centre[0]), float(centre[1]),
                                            float(centre[2])),),
                                       up=((0.0, 0.0, 1.0),), device=device)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=Tv, fov=60.0)
        with torch.no_grad():
            img = renderer(mesh, cameras=cameras)[0, ..., :3]
        writer.append_data((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        if i % 20 == 0:
            print(f"  frame {i + 1}/{T}", flush=True)

    writer.close()
    print(f"wrote {args.out} -- {T} frames at {args.size}x{args.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
