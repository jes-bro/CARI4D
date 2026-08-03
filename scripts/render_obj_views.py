# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Blender script: render orbit views of a textured OBJ, for eyeballing it.

Run inside Blender, not the conda env:

    blender -b -P scripts/render_obj_views.py -- <obj_path> <out_dir> [n_views] [res]

Renders n_views images orbiting the object and writes view_000.png ... to
out_dir. Uses the Workbench engine in TEXTURE colour mode: no ray tracing, no
light rig, no GPU compositing, so it runs headless in seconds and shows the
texture exactly as the UVs address it.

The point is to confirm the texture actually lines up with the geometry.
Hunyuan3D's image is baked against the full-resolution mesh, and prep/glb2obj.py
decimates 676k faces down to 40k before export -- if that resampled the UV
layout rather than preserving it, the mesh renders with the texture smeared or
scrambled, and no numeric check catches that. A person looking at four views
catches it immediately.
"""
import math
import os
import os.path as osp
import sys

import bpy
from mathutils import Vector

USAGE = ("usage: blender -b -P render_obj_views.py -- "
         "<obj_path> <out_dir> [n_views] [resolution]")


def parse_argv():
    """Read the script arguments Blender passes after a bare '--'."""
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit(USAGE)
    argv = argv[argv.index("--") + 1:]
    if len(argv) < 2:
        raise SystemExit(USAGE)
    n_views = int(argv[2]) if len(argv) > 2 else 4
    resolution = int(argv[3]) if len(argv) > 3 else 512
    return argv[0], argv[1], n_views, resolution


def clear_scene():
    """Empty the scene so Blender's startup cube and light are not rendered."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_obj(obj_path):
    """Import an OBJ, handling both Blender operator generations.

    Blender 4.0 replaced bpy.ops.import_scene.obj with bpy.ops.wm.obj_import.

    Raises:
        RuntimeError: if the file imported no mesh objects.
    """
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=obj_path)
    else:
        bpy.ops.import_scene.obj(filepath=obj_path)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh objects imported from {obj_path}")
    return meshes


def scene_bounds(meshes):
    """Return the world-space centre and radius enclosing every mesh.

    Used to frame the camera, so the object fills the view regardless of the
    mesh's scale -- these templates are normalised to [-1, 1] but the script
    should not depend on that.
    """
    corners = [obj.matrix_world @ Vector(c)
               for obj in meshes for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    centre = Vector(((min(xs) + max(xs)) / 2,
                     (min(ys) + max(ys)) / 2,
                     (min(zs) + max(zs)) / 2))
    radius = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2
    return centre, max(radius, 1e-6)


def setup_camera(centre, radius):
    """Add a camera aimed at centre via a TRACK_TO constraint.

    Constraining to an empty at the object's centre means orbiting only has to
    move the camera's location -- Blender keeps it pointed correctly, which
    avoids hand-computing look-at rotations.

    Returns:
        (camera, target_empty, orbit_distance)
    """
    bpy.ops.object.empty_add(location=centre)
    target = bpy.context.object

    bpy.ops.object.camera_add(location=(centre.x, centre.y - radius * 3, centre.z))
    camera = bpy.context.object
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    bpy.context.scene.camera = camera
    return camera, target, radius * 3


def setup_render(resolution):
    """Configure Workbench in TEXTURE mode at the requested square resolution.

    Workbench rasterises without ray tracing or a light rig, so a headless
    render takes well under a second and shows the texture as the UVs address
    it -- which is the only thing being checked here.
    """
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False


def render_views(camera, centre, distance, out_dir, n_views):
    """Orbit the camera and render n_views images into out_dir.

    Returns:
        The list of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for i in range(n_views):
        angle = 2 * math.pi * i / n_views
        camera.location = (
            centre.x + distance * math.sin(angle),
            centre.y - distance * math.cos(angle),
            centre.z + distance * 0.35,
        )
        path = osp.join(out_dir, f"view_{i:03d}.png")
        bpy.context.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"[render] wrote {path}")
        written.append(path)
    return written


def main():
    """Import the OBJ, frame it, and render the orbit."""
    obj_path, out_dir, n_views, resolution = parse_argv()
    if not osp.isfile(obj_path):
        raise SystemExit(f"ERROR: no such file: {obj_path}")

    print(f"[render] blender {bpy.app.version_string}")
    print(f"[render] obj={obj_path} out={out_dir} views={n_views} res={resolution}")

    clear_scene()
    meshes = import_obj(obj_path)
    faces = sum(len(m.data.polygons) for m in meshes)
    materials = {s.material.name for m in meshes for s in m.material_slots if s.material}
    print(f"[render] {len(meshes)} mesh(es), {faces} faces, materials={sorted(materials)}")

    centre, radius = scene_bounds(meshes)
    print(f"[render] centre={tuple(round(v, 4) for v in centre)} radius={radius:.4f}")

    camera, _, distance = setup_camera(centre, radius)
    setup_render(resolution)
    written = render_views(camera, centre, distance, out_dir, n_views)
    print(f"[render] done, {len(written)} image(s) in {out_dir}")


if __name__ == "__main__":
    main()
