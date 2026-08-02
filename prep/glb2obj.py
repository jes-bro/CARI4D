# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Blender script: convert every .glb in a directory to a textured .obj.

Run inside Blender, not the conda env:

    blender -b -P prep/glb2obj.py -- <glb_dir> <out_dir> [max_faces]

For each ``<glb_dir>/<name>.glb`` this writes ``<out_dir>/<name>/<name>.obj``
plus its ``.mtl`` and copied textures -- the layout ``run_hy3d_recon.run_glb2obj``
expects before it flattens the result into ``<out_dir>``.

Meshes above ``max_faces`` triangles are decimated first; raw Hunyuan3D output
runs 200k+ faces, which makes FoundationPose rendering crawl.

Upstream references this file from run_hy3d_recon.py but never shipped it
(commit 53b1d5e adds the caller only), so this is a reimplementation.
"""
import os
import os.path as osp
import sys

import bpy

DEFAULT_MAX_FACES = 40000
USAGE = "usage: blender -b -P glb2obj.py -- <glb_dir> <out_dir> [max_faces]"


def parse_argv():
    """Blender passes script arguments after a bare '--'."""
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit(USAGE)
    argv = argv[argv.index("--") + 1:]
    if len(argv) < 2:
        raise SystemExit(USAGE)
    max_faces = int(argv[2]) if len(argv) > 2 else DEFAULT_MAX_FACES
    return argv[0], argv[1], max_faces


def clear_scene():
    """Start from a genuinely empty scene.

    Blender's startup file ships a cube, camera and light; exporting those
    alongside the imported mesh would corrupt the object template.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)


def mesh_objects():
    """Return every MESH object in the current scene.

    GLB files often import as several meshes (one per material), so callers
    must not assume a single object.
    """
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def total_faces():
    """Return the face count summed across every mesh in the scene.

    Called after triangulation, so this is the triangle count that drives the
    decimation ratio.
    """
    return sum(len(o.data.polygons) for o in mesh_objects())


def apply_modifier(obj, mod):
    """modifier_apply acts on the active object, so select and activate first."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def triangulate_and_decimate(max_faces):
    """Triangulate every mesh, then collapse-decimate down to max_faces.

    Triangulation runs unconditionally so the face count is a true triangle
    count and FoundationPose gets the triangle soup it expects. Decimation is
    skipped when the mesh is already within budget. The ratio is computed
    across the whole scene but applied per object, so multi-mesh imports keep
    their relative density.

    Args:
        max_faces: triangle budget for the whole scene.

    Raises:
        RuntimeError: if the GLB imported no mesh objects.
    """
    objs = mesh_objects()
    if not objs:
        raise RuntimeError("no mesh objects were imported from the GLB")

    for obj in objs:
        mod = obj.modifiers.new(name="triangulate", type="TRIANGULATE")
        mod.quad_method = "BEAUTY"
        mod.ngon_method = "BEAUTY"
        apply_modifier(obj, mod)

    faces = total_faces()
    if faces <= max_faces:
        print(f"[glb2obj] {faces} faces, within the {max_faces} budget -- no decimation")
        return

    ratio = max_faces / float(faces)
    print(f"[glb2obj] decimating {faces} -> ~{max_faces} faces (ratio={ratio:.4f})")
    for obj in mesh_objects():
        mod = obj.modifiers.new(name="decimate", type="DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = ratio
        apply_modifier(obj, mod)
    print(f"[glb2obj] now {total_faces()} faces")


def export_obj(obj_path):
    """Export the scene to OBJ, handling both Blender operator generations.

    Blender 4.0 removed bpy.ops.export_scene.obj in favour of
    bpy.ops.wm.obj_export. Both default to forward=-Z / up=Y, so the branches
    agree on orientation.

    path_mode='COPY' writes textures next to the .obj so the .mtl still
    resolves after run_hy3d_recon moves files around.
    """
    if hasattr(bpy.ops.wm, "obj_export"):
        try:
            bpy.ops.wm.obj_export(
                filepath=obj_path,
                export_selected_objects=False,
                export_materials=True,
                export_triangulated_mesh=True,
                path_mode="COPY",
                forward_axis="NEGATIVE_Z",
                up_axis="Y",
            )
        except TypeError as exc:
            # Guard against keyword drift in newer Blender releases.
            print(f"[glb2obj] full obj_export signature rejected ({exc}); retrying minimally")
            bpy.ops.wm.obj_export(filepath=obj_path, path_mode="COPY")
    else:
        bpy.ops.export_scene.obj(
            filepath=obj_path,
            use_selection=False,
            use_materials=True,
            use_triangles=True,
            path_mode="COPY",
            axis_forward="-Z",
            axis_up="Y",
        )


def process_glb_file_with_decimation(glb_path, out_dir, max_faces=DEFAULT_MAX_FACES):
    """Import one GLB, decimate it, export to <out_dir>/<name>/<name>.obj."""
    name = osp.splitext(osp.basename(glb_path))[0]
    dest_dir = osp.join(out_dir, name)
    os.makedirs(dest_dir, exist_ok=True)
    obj_path = osp.join(dest_dir, f"{name}.obj")

    print(f"[glb2obj] importing {glb_path}")
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=glb_path)
    print(f"[glb2obj] imported {len(mesh_objects())} mesh object(s), {total_faces()} faces")

    triangulate_and_decimate(max_faces)

    print(f"[glb2obj] exporting {obj_path}")
    export_obj(obj_path)

    if not osp.isfile(obj_path):
        raise RuntimeError(f"Blender exited cleanly but {obj_path} was not written")
    print(f"[glb2obj] wrote {obj_path} ({osp.getsize(obj_path)} bytes)")
    return obj_path


def main():
    """Convert every .glb in glb_dir, exiting non-zero if there is nothing to do.

    Fails loudly on a missing directory or an empty glob rather than exiting
    clean, so run_hy3d_recon's subprocess check=True surfaces the problem
    instead of reporting a silent success with no OBJ written.
    """
    glb_dir, out_dir, max_faces = parse_argv()
    print(f"[glb2obj] blender {bpy.app.version_string}")
    print(f"[glb2obj] glb_dir={glb_dir} out_dir={out_dir} max_faces={max_faces}")

    if not osp.isdir(glb_dir):
        raise SystemExit(f"glb_dir does not exist: {glb_dir}")

    glbs = sorted(f for f in os.listdir(glb_dir) if f.lower().endswith(".glb"))
    if not glbs:
        raise SystemExit(f"no .glb files found in {glb_dir}")

    os.makedirs(out_dir, exist_ok=True)
    for fname in glbs:
        process_glb_file_with_decimation(osp.join(glb_dir, fname), out_dir, max_faces)

    print(f"[glb2obj] done, converted {len(glbs)} file(s)")


if __name__ == "__main__":
    main()
