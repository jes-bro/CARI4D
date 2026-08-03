# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Generate a ball object template, bypassing image-based reconstruction.

Hunyuan3D exists to recover unknown geometry. A ball's geometry is known
exactly, so reconstructing it from pixels only adds error -- and for a distant
ball there may be no usable pixels at all (a basketball 12x7 px in a 448-wide
egoexo4d frame carries no shape information whatsoever).

This writes the same artifacts prep/run_hy3d_recon.py would, in the layout
fp_hy3d_track.py globs for:

    <hy3d_root>/<seq>_<frame:03d>_rgba/<seq>_<frame:03d>_align.obj
                                       <...>_align.mtl
                                       <...>_align_texture.png
                                       <...>_align_ball.json

The mesh is normalised so its longest axis spans [-1, 1], matching direct
Hunyuan3D output, because tools/estimate_scale_video.py assumes it. True
metric size and mass go in the JSON sidecar -- OBJ carries neither, and the
figures vary by ball type, which is the whole reason this is parameterised.

Usage:
    python scripts/make_ball_mesh.py --seq cam04 --hy3d_root data/cari4d-demo/meshes
    python scripts/make_ball_mesh.py --seq cam04 --ball size6 --frame_index 12
    python scripts/make_ball_mesh.py --seq cam04 --diameter 0.235 --mass 0.60
"""
import argparse
import json
import math
import os
import os.path as osp
import sys

import numpy as np

# Diameter in metres and mass in kilograms, from the governing bodies' specs.
# Where a range is allowed the midpoint is used; the range itself is kept so
# --list can report the tolerance rather than implying false precision.
BALL_SPECS = {
    "size7": {
        "diameter_m": 0.239, "mass_kg": 0.624,
        "diameter_range_m": (0.238, 0.243), "mass_range_kg": (0.567, 0.624),
        "description": "Basketball size 7 -- men's (NBA, FIBA, NCAA men's)",
    },
    "size6": {
        "diameter_m": 0.231, "mass_kg": 0.567,
        "diameter_range_m": (0.229, 0.232), "mass_range_kg": (0.510, 0.567),
        "description": "Basketball size 6 -- women's (WNBA, NCAA women's, FIBA women's)",
    },
    "size5": {
        "diameter_m": 0.222, "mass_kg": 0.482,
        "diameter_range_m": (0.220, 0.224), "mass_range_kg": (0.470, 0.500),
        "description": "Basketball size 5 -- youth",
    },
    "size3": {
        "diameter_m": 0.178, "mass_kg": 0.300,
        "diameter_range_m": (0.176, 0.180), "mass_range_kg": (0.280, 0.310),
        "description": "Basketball size 3 -- mini",
    },
    "soccer5": {
        "diameter_m": 0.220, "mass_kg": 0.430,
        "diameter_range_m": (0.217, 0.226), "mass_range_kg": (0.410, 0.450),
        "description": "Football/soccer size 5 -- adult",
    },
    "volleyball": {
        "diameter_m": 0.209, "mass_kg": 0.270,
        "diameter_range_m": (0.205, 0.213), "mass_range_kg": (0.260, 0.280),
        "description": "Volleyball -- FIVB indoor",
    },
}

DEFAULT_BALL = "size7"
BASKETBALL_ORANGE = (214, 106, 44)
SEAM_BLACK = (26, 22, 20)


def parse_args():
    """Parse the sequence, ball type and any explicit size/mass overrides."""
    parser = argparse.ArgumentParser(
        description="Generate a normalised ball mesh as an object template")
    parser.add_argument("--seq", help="sequence name, e.g. cam04")
    parser.add_argument("--hy3d_root", default="data/cari4d-demo/meshes",
                        help="mesh output root (default: data/cari4d-demo/meshes)")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="frame index used in the directory name (default: 0)")
    parser.add_argument("--ball", default=DEFAULT_BALL, choices=sorted(BALL_SPECS),
                        help=f"ball type (default: {DEFAULT_BALL})")
    parser.add_argument("--diameter", type=float, default=None,
                        help="override the true diameter in metres")
    parser.add_argument("--mass", type=float, default=None,
                        help="override the true mass in kilograms")
    parser.add_argument("--subdivisions", type=int, default=4,
                        help="icosphere subdivisions; 4 gives 5120 faces (default: 4)")
    parser.add_argument("--texture_size", type=int, default=1024,
                        help="procedural texture resolution (default: 1024)")
    parser.add_argument("--texture_from", default=None,
                        help="RGBA crop of the real object (the <seq>_<frame>_rgba.png "
                             "run_hy3d_recon writes). Projected onto the sphere so the "
                             "template carries this ball's actual markings instead of a "
                             "generic pattern. Falls back to procedural if omitted.")
    parser.add_argument("--no_texture", action="store_true",
                        help="write an untextured mesh")
    parser.add_argument("--list", action="store_true",
                        help="print the known ball specs and exit")
    return parser.parse_args()


def print_specs():
    """Print every known ball spec with its tolerance range."""
    print(f"{'name':<12} {'diameter (m)':<24} {'mass (kg)':<22} description")
    for name, spec in sorted(BALL_SPECS.items()):
        dlo, dhi = spec["diameter_range_m"]
        mlo, mhi = spec["mass_range_kg"]
        print(f"{name:<12} {spec['diameter_m']:.3f} ({dlo:.3f}-{dhi:.3f})   "
              f"{spec['mass_kg']:.3f} ({mlo:.3f}-{mhi:.3f})   {spec['description']}")


def resolve_spec(args):
    """Return the ball spec, applying any --diameter/--mass overrides.

    Overrides are recorded in the returned dict so the JSON sidecar shows the
    value actually used rather than the table default -- otherwise a later
    reader cannot tell a measured ball from an assumed one.
    """
    spec = dict(BALL_SPECS[args.ball])
    spec["ball"] = args.ball
    spec["overridden"] = []
    if args.diameter is not None:
        spec["diameter_m"] = args.diameter
        spec["overridden"].append("diameter_m")
    if args.mass is not None:
        spec["mass_kg"] = args.mass
        spec["overridden"].append("mass_kg")
    return spec


def sphere_uv(vertices):
    """Compute an equirectangular UV per vertex from its direction.

    u wraps around the Y axis, v runs pole to pole. Faces spanning the u=0/1
    seam interpolate the long way round and show a visible band there; that is
    inherent to per-vertex UVs on a closed sphere without duplicating the seam
    vertices, and it does not affect geometry or pose tracking.
    """
    directions = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    u = 0.5 + np.arctan2(z, x) / (2 * math.pi)
    v = 0.5 - np.arcsin(np.clip(y, -1.0, 1.0)) / math.pi
    return np.column_stack([u, v])


def find_ball_disc(alpha):
    """Locate the object's disc in an RGBA crop from its alpha channel.

    crop_rgba centres the object with a margin, so the object does not fill the
    image and the unit sphere must be mapped to the object's own extent rather
    than to the image bounds.

    Returns:
        (cx, cy, radius) in pixels.

    Raises:
        SystemExit: if the alpha channel is empty.
    """
    ys, xs = np.where(alpha > 127)
    if len(ys) == 0:
        raise SystemExit("ERROR: --texture_from image has an empty alpha channel")
    cx = (float(xs.min()) + float(xs.max())) / 2.0
    cy = (float(ys.min()) + float(ys.max())) / 2.0
    radius = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) / 2.0
    return cx, cy, max(radius, 1.0)


def project_crop_to_texture(crop_path, size):
    """Project an orthographic photo of a ball into equirectangular UV space.

    Treats the crop as an orthographic view down -Z: a surface point in
    direction (x, y, z) with z > 0 is visible and lands at image position
    (cx + x*r, cy - y*r). Inverting sphere_uv per texel gives that direction, so
    each texel can be sampled straight from the photo.

    The camera only ever saw one hemisphere, so the far side has no data. It is
    filled by mirroring through the image plane -- sampling |z| rather than z --
    which wraps the visible markings around the back. That is a fabrication, but
    a plausible and clearly-bounded one: for a ball the invented half looks like
    the real half, which is truer than a flat colour and far truer than the
    flattened blob single-image reconstruction produced.

    Grazing texels near the silhouette (|z| small) stretch badly, since an
    orthographic view compresses them into almost no pixels. Unavoidable from
    one view -- it is exactly the coverage that multi-view would fix.

    Args:
        crop_path: RGBA crop of the object.
        size: output texture resolution.

    Returns:
        A PIL RGB image, size x size.
    """
    from PIL import Image

    crop = Image.open(crop_path).convert("RGBA")
    pixels = np.asarray(crop)
    rgb, alpha = pixels[:, :, :3], pixels[:, :, 3]
    cx, cy, radius = find_ball_disc(alpha)
    print(f"  source disc: centre ({cx:.0f}, {cy:.0f}) radius {radius:.0f} px "
          f"in a {crop.width}x{crop.height} crop")

    # Invert sphere_uv: u wraps around Y, v runs pole to pole.
    us = (np.arange(size) + 0.5) / size
    vs = (np.arange(size) + 0.5) / size
    uu, vv = np.meshgrid(us, vs)
    y = np.sin(math.pi * (0.5 - vv))
    ring = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    angle = 2 * math.pi * (uu - 0.5)
    x = ring * np.cos(angle)

    # Mirror the far hemisphere: sample by |z|, so the back reuses the front.
    px = np.clip(np.round(cx + x * radius), 0, crop.width - 1).astype(np.int32)
    py = np.clip(np.round(cy - y * radius), 0, crop.height - 1).astype(np.int32)
    texture = rgb[py, px]

    # Anything sampled from outside the silhouette gets the object's mean colour
    # rather than background, which would otherwise ring the poles and seams.
    sampled_alpha = alpha[py, px]
    inside = alpha > 127
    mean_colour = rgb[inside].mean(axis=0).astype(np.uint8)
    texture[sampled_alpha <= 127] = mean_colour
    filled = int((sampled_alpha <= 127).sum())
    if filled:
        print(f"  filled {filled} texel(s) outside the silhouette with the mean colour "
              f"{tuple(int(c) for c in mean_colour)}")

    return Image.fromarray(texture, "RGB")


def make_basketball_texture(size):
    """Render an approximate basketball texture in equirectangular UV space.

    Real basketballs carry two great-circle seams plus two curved seams. This
    approximates them well enough to give FoundationPose surface features to
    match against -- it is not a claim about a specific ball's markings.

    Returns:
        A PIL RGB image, size x size.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), BASKETBALL_ORANGE)
    draw = ImageDraw.Draw(img)
    width = max(2, size // 128)

    # Horizontal great circle (the equator) and two vertical ones, which in
    # equirectangular space are a horizontal line and two vertical lines.
    draw.line([(0, size // 2), (size, size // 2)], fill=SEAM_BLACK, width=width)
    for u in (0.25, 0.75):
        draw.line([(int(u * size), 0), (int(u * size), size)], fill=SEAM_BLACK, width=width)

    # The two curved seams, as sinusoids either side of the equator.
    for phase, amplitude in ((0.0, 0.22), (math.pi, 0.22)):
        points = []
        for i in range(size + 1):
            u = i / size
            v = 0.5 + amplitude * math.sin(2 * math.pi * u + phase)
            points.append((i, int(v * size)))
        draw.line(points, fill=SEAM_BLACK, width=width, joint="curve")

    return img


def build_mesh(subdivisions):
    """Build a unit-radius icosphere with per-vertex UVs.

    Radius 1 means the longest axis spans [-1, 1] with no rescaling, matching
    the normalisation direct Hunyuan3D output uses.
    """
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.visual = trimesh.visual.TextureVisuals(uv=sphere_uv(mesh.vertices))
    return mesh


def write_outputs(mesh, texture, spec, outdir, base_name):
    """Write the .obj, .mtl, texture and JSON sidecar into outdir.

    The OBJ is written by trimesh so it round-trips through the same loader
    fp_hy3d_track.py and estimate_scale_video.py use.

    Returns:
        Path to the written .obj.
    """
    import trimesh

    os.makedirs(outdir, exist_ok=True)
    obj_path = osp.join(outdir, f"{base_name}.obj")
    mtl_name = f"{base_name}.mtl"

    if texture is not None:
        tex_name = f"{base_name}_texture.png"
        texture.save(osp.join(outdir, tex_name))
        material = trimesh.visual.material.SimpleMaterial(image=texture)
        mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=material)

    obj_data = trimesh.exchange.obj.export_obj(
        mesh, include_texture=texture is not None, mtl_name=mtl_name)
    # export_obj returns the OBJ text and stashes the .mtl in the return of
    # export_scene for scenes; for a single mesh we write the .mtl ourselves so
    # map_Kd names the file we actually saved.
    with open(obj_path, "w") as f:
        f.write(obj_data if isinstance(obj_data, str) else obj_data[0])

    if texture is not None:
        with open(osp.join(outdir, mtl_name), "w") as f:
            f.write("# generated by scripts/make_ball_mesh.py\n\n"
                    "newmtl material_0\n"
                    "Ka 0.200000 0.200000 0.200000\n"
                    "Kd 0.800000 0.800000 0.800000\n"
                    "Ks 0.100000 0.100000 0.100000\n"
                    "Ns 10.000000\n"
                    f"map_Kd {base_name}_texture.png\n")

    meta = dict(spec)
    meta.update({
        "source": "scripts/make_ball_mesh.py",
        "normalised_radius": 1.0,
        "metres_per_normalised_unit": spec["diameter_m"] / 2.0,
        "faces": int(len(mesh.faces)),
        "note": ("Mesh is normalised to [-1, 1]; multiply by "
                 "metres_per_normalised_unit for true scale."),
    })
    with open(osp.join(outdir, f"{base_name}_ball.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return obj_path


def main():
    """Generate the ball template and report what was written."""
    args = parse_args()
    if args.list:
        print_specs()
        return 0
    if not args.seq:
        raise SystemExit("ERROR: --seq is required (or use --list)")

    spec = resolve_spec(args)
    dir_name = f"{args.seq}_{args.frame_index:03d}_rgba"
    base_name = f"{args.seq}_{args.frame_index:03d}_align"
    outdir = osp.join(args.hy3d_root, dir_name)

    mesh = build_mesh(args.subdivisions)
    if args.no_texture:
        texture = None
    elif args.texture_from:
        print(f"Projecting texture from {args.texture_from}")
        texture = project_crop_to_texture(args.texture_from, args.texture_size)
    else:
        texture = make_basketball_texture(args.texture_size)
    obj_path = write_outputs(mesh, texture, spec, outdir, base_name)

    print(f"Ball      : {spec['ball']} -- {spec['description']}")
    if spec["overridden"]:
        print(f"Overridden: {', '.join(spec['overridden'])}")
    print(f"Diameter  : {spec['diameter_m']:.3f} m")
    print(f"Mass      : {spec['mass_kg']:.3f} kg")
    print(f"Faces     : {len(mesh.faces)}")
    print(f"Wrote     : {obj_path}")
    return 0


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    sys.exit(main())
