# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
Object reconstruction with SAM 3D Objects instead of Hunyuan3D.

Same contract as run_hy3d_recon.py, so nothing downstream changes: an RGBA crop
from one video frame plus its object mask, one textured mesh, and the OBJ at

    <hy3d_root>/<seq>_<frame:03d>_rgba/<seq>_<frame:03d>_align.obj

that fp_hy3d_track.py globs for and estimate_scale_video.py parses. The crop,
the mask loading, the Blender conversion and the texture re-attachment are
imported from run_hy3d_recon.py rather than copied, so a fix there is a fix
here.

Why a second reconstructor: Hunyuan3D hallucinates the far side of a pan or a
pot from one view, and the result is a pretty but wrong shape that
FoundationPose then has to track. SAM 3D Objects was trained on occluded,
cluttered real photographs and also predicts the object's pose in the camera
frame, which this script keeps beside the mesh for later use.

Usage (inside the sam3d-objects env, from the repo root):

    python prep/run_sam3d_recon.py \\
        --video <masks_root>/trimmed_vids/<seq>.0.color.mp4 \\
        --masks_root <masks_root> \\
        --hy3d_root <work>/meshes-sam3d \\
        --frame_index 0 \\
        --sam3d_root sam-3d-objects \\
        --blender_path Hunyuan3D-2/blender-3.6.0-linux-x64/blender

The --hires_video / --out_seq / --out_frame_index flags mean exactly what they
mean in run_hy3d_recon.py: reconstruct from the camera where the object is
closest (the Aria ego view, for a pot on a stove) but name the mesh for the
sequence the pipeline tracks in.
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np

sys.path.append(os.getcwd())

from prep.run_hy3d_recon import (attach_glb_texture, crop_rgba, extract_frame,
                                 extract_seq_name, load_hires_frame_and_mask,
                                 load_object_mask, run_glb2obj)

# Hunyuan3D emits meshes whose bounding box fits in [-1, 1]; SAM 3D emits a
# [-0.5, 0.5] cube. The metric-scale step fits a free scale either way, but
# the docs promise the [-1, 1] convention, so the mesh is renormalised to it
# and every mesh under <hy3d_root> means the same thing regardless of source.
TARGET_HALF_EXTENT = 1.0
# Above this FoundationPose's render-and-compare crawls; the same ceiling
# glb2obj.py decimates to. SAM 3D's own 95% simplification usually lands
# below it already.
MAX_FACES = 40000


def parse_args():
    """Parse the CLI arguments. Mirrors run_hy3d_recon.py, plus --sam3d_*.

    --skip_sam3d stops after the RGBA crop, which is the sanity check to run
    before spending a 32 GB GPU allocation: it needs no model and no GPU.
    """
    parser = argparse.ArgumentParser(
        description="Extract RGBA, run SAM 3D Objects, write the OBJ the tracker expects")
    parser.add_argument("--video", required=True, help="Path to input video, e.g. <seq>.0.color.mp4")
    parser.add_argument("--masks_root", required=True, help="Directory containing HDF5 mask files")
    parser.add_argument("--hy3d_root", required=True,
                        help="Output root for the meshes. Named for the Hunyuan3D convention "
                             "because that is the key every downstream script reads.")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Video frame index to use for reconstruction (default: 0)")
    parser.add_argument("--kid", type=int, default=0, help="Camera/kinect ID (default: 0)")
    parser.add_argument("--sam3d_root", default="sam-3d-objects",
                        help="Clone of facebookresearch/sam-3d-objects with checkpoints/<tag>/ "
                             "downloaded (default: sam-3d-objects under the repo root)")
    parser.add_argument("--checkpoint_tag", default="hf",
                        help="checkpoints/<tag>/pipeline.yaml inside --sam3d_root (default: hf)")
    parser.add_argument("--blender_path", default="blender",
                        help="Path to Blender executable (default: 'blender')")
    parser.add_argument("--no_blender", action="store_true",
                        help="Write the OBJ with trimesh instead of Blender. Use when no "
                             "Blender is installed in this env; the texture still comes along.")
    parser.add_argument("--vertex_color", action="store_true",
                        help="Skip texture baking and colour vertices instead. Faster and "
                             "lighter, but FoundationPose tracks a flat-grey-ish mesh worse "
                             "on objects whose shape alone is ambiguous.")
    parser.add_argument("--fill_mask_holes", action="store_true",
                        help="Fill enclosed holes in the object mask before cropping. "
                             "SAM3 segments a pot seen from above as a ring, leaving the "
                             "liquid inside unmasked; the model then reads the inside as "
                             "empty and builds a tube with no bottom. The silhouette a "
                             "single-image reconstructor needs is the outer contour with "
                             "everything inside it, which is what this restores.")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Total border margin ratio for cropping (default: 0.2)")
    parser.add_argument("--crop_size", type=int, default=512,
                        help="Output RGBA image size (default: 512)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--skip_sam3d", action="store_true",
                        help="Skip SAM 3D inference, only do RGBA extraction")
    parser.add_argument("--hires_video", default=None,
                        help="Take the reconstruction frame from this higher-resolution copy "
                             "of the same take, upscaling the mask to match")
    parser.add_argument("--hires_frame_offset", type=int, default=0,
                        help="Frame index in --hires_video corresponding to frame 0 of --video")
    parser.add_argument("--out_seq", default=None,
                        help="Name the output for this sequence instead of the one derived "
                             "from --video (reconstruct from one camera, track in another)")
    parser.add_argument("--out_frame_index", type=int, default=None,
                        help="Frame index to encode in the output name instead of --frame_index")
    return parser.parse_args()


def fill_mask_holes(mask):
    """Return the mask with every enclosed hole filled, as uint8 0/255.

    Flood-fills the background from the border of a padded copy; whatever the
    flood does not reach and the mask does not cover is a hole. Pure cv2, so
    it needs nothing the reconstruction env lacks.
    """
    import cv2

    fg = (mask > 127).astype(np.uint8)
    padded = np.pad(fg, 1)
    flood = padded.copy()
    h, w = flood.shape
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    holes = (flood == 0) & (padded == 0)
    filled = (padded | holes.astype(np.uint8))[1:-1, 1:-1]
    n_holes = int(holes.sum())
    print(f"Filled {n_holes} enclosed hole px in the object mask "
          f"({int(fg.sum())} -> {int(filled.sum())} px)")
    return filled * 255


def load_sam3d(sam3d_root, tag):
    """Import and construct the SAM 3D Objects notebook wrapper.

    The repo is not a pip package with a stable top-level API: the README
    drives it through notebook/inference.py, which is what this imports. Both
    the repo root (for the sam3d_objects package) and notebook/ go on sys.path.

    Raises:
        SystemExit: when the clone or the checkpoint config is missing, with
            the setup command to run, since the ImportError alone is cryptic.
    """
    root = osp.abspath(sam3d_root)
    config = osp.join(root, "checkpoints", tag, "pipeline.yaml")
    if not osp.isdir(osp.join(root, "notebook")):
        raise SystemExit(f"ERROR: no SAM 3D Objects clone at {root}; "
                         f"git clone https://github.com/facebookresearch/sam-3d-objects {root}")
    if not osp.isfile(config):
        raise SystemExit(f"ERROR: no checkpoint config at {config}; download with\n"
                         f"  hf download --repo-type model --local-dir {root}/checkpoints/hf-download "
                         f"facebook/sam-3d-objects && mv {root}/checkpoints/hf-download/checkpoints "
                         f"{root}/checkpoints/{tag}")
    sys.path.insert(0, root)
    sys.path.insert(0, osp.join(root, "notebook"))
    # sam3d_objects/__init__.py imports a sam3d_objects.init module that is not
    # in the repository, unless this is set; the notebook wrapper sets it too,
    # but only at its own import, and the order matters. CUDA_HOME is what
    # their JIT-built extensions read, and the conda env ships the toolkit.
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    if "CONDA_PREFIX" in os.environ:
        os.environ.setdefault("CUDA_HOME", os.environ["CONDA_PREFIX"])
    from inference import Inference  # noqa: E402  (sam-3d-objects/notebook/inference.py)
    print(f"Loading SAM 3D Objects from {config}")
    return Inference(config, compile=False)


def run_sam3d(inference, rgba_img, seed, vertex_color):
    """Run the full SAM 3D pipeline and return its output dict with a mesh.

    The notebook wrapper's __call__ switches mesh post-processing and texture
    baking OFF, because the demo only wants the Gaussian splat. The tracker
    needs a mesh, so this calls the underlying pipeline directly with the mesh
    options on. The RGBA crop goes in as the image with alpha as the mask,
    which is exactly what the wrapper would have built from a separate mask.

    Args:
        inference: the notebook Inference object.
        rgba_img: PIL RGBA crop, alpha = object mask.
        seed: sampling seed.
        vertex_color: True to skip texture baking and colour vertices instead.

    Returns:
        The pipeline's dict. Keys used here: 'glb' (trimesh.Trimesh),
        'rotation', 'translation', 'scale' (the predicted camera-frame pose).
    """
    rgba = np.asarray(rgba_img.convert("RGBA"), dtype=np.uint8)
    output = inference._pipeline.run(
        rgba, None, seed,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=not vertex_color,
        with_layout_postprocess=False,
        use_vertex_color=vertex_color,
    )
    if output.get("glb") is None:
        raise RuntimeError("SAM 3D returned no mesh; the pipeline's decode_formats must "
                           "include 'mesh' (it does in the released pipeline.yaml)")
    print(f"SAM 3D done: {len(output['glb'].faces)} faces")
    return output


def normalize_mesh(mesh, half_extent=TARGET_HALF_EXTENT):
    """Centre the mesh and scale its longest axis to [-half_extent, half_extent].

    Done in place on the trimesh. Returns the (centre, scale) applied so the
    predicted pose can be carried along in the same units.
    """
    lo, hi = mesh.bounds
    centre = (lo + hi) / 2.0
    longest = float(np.max(hi - lo))
    scale = (2.0 * half_extent) / longest
    mesh.vertices = (mesh.vertices - centre) * scale
    print(f"Normalised mesh: extents {np.round(hi - lo, 3)} -> "
          f"longest axis [-{half_extent}, {half_extent}] (x{scale:.3f})")
    return centre, scale


def decimate_if_needed(mesh, max_faces=MAX_FACES):
    """Reduce the face count when it exceeds what FoundationPose renders quickly.

    Blender does this itself in glb2obj.py; the trimesh path (--no_blender)
    has to do it here. Texture UVs survive quadric decimation in trimesh only
    on recent versions, so if the call fails the mesh is left as is and the
    message says so rather than dropping the texture silently.
    """
    n = len(mesh.faces)
    if n <= max_faces:
        return mesh
    try:
        out = mesh.simplify_quadric_decimation(face_count=max_faces)
        print(f"Decimated {n} -> {len(out.faces)} faces")
        return out
    except Exception as e:  # noqa: BLE001  (any backend failure is non-fatal)
        print(f"Could not decimate {n} faces ({e}); leaving the mesh as is")
        return mesh


def export_obj_trimesh(mesh, outdir, obj_name):
    """Write <outdir>/<obj_name> plus .mtl and texture with trimesh, no Blender.

    trimesh writes the material and texture image next to the OBJ when given
    a path, under its own names (material.mtl, material_0.png). The .obj
    references the .mtl by bare filename, so everything stays in outdir and
    the tracker finds the texture the way it does after the Blender path.
    """
    mesh = decimate_if_needed(mesh)
    obj_path = osp.join(outdir, obj_name)
    mesh.export(obj_path, include_texture=True)
    print(f"Saved OBJ via trimesh: {obj_path}")
    if not any(f.endswith(".mtl") for f in os.listdir(outdir)):
        print("trimesh wrote no .mtl; the mesh will render untextured")


def save_pose(output, outdir, out_name, centre, scale):
    """Keep SAM 3D's predicted camera-frame pose beside the mesh.

    rotation is a quaternion, translation and scale are in the crop's inferred
    camera, so they are relative to the 512x512 RGBA crop rather than the
    video's intrinsics. Stored for a later FoundationPose initialisation or a
    sanity check against the tracker, not consumed by anything yet. The mesh
    normalisation applied above is recorded so the two can be related.
    """
    def as_np(x):
        """Tensor or array to a float64 numpy array on the CPU."""
        return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x,
                          dtype=np.float64)
    keys = {k: as_np(output[k]) for k in ("rotation", "translation", "scale") if k in output}
    if not keys:
        print("No pose keys in the SAM 3D output; nothing to save")
        return
    path = osp.join(outdir, f"{out_name}_sam3d_pose.npz")
    np.savez(path, mesh_centre=centre, mesh_scale=scale, **keys)
    print(f"Saved predicted pose: {path} ({', '.join(keys)})")


def main():
    """Run the reconstruction for one video frame.

    Extracts the frame, loads its object mask, writes the cropped RGBA, runs
    SAM 3D Objects, normalises the mesh, exports the GLB and converts it to
    the <seq>_<frame:03d>_align.obj that fp_hy3d_track.py expects. Returns
    early if that OBJ already exists, so re-running is cheap.
    """
    args = parse_args()

    seq_name = extract_seq_name(args.video)
    frame_idx = args.frame_index
    out_seq = args.out_seq or seq_name
    out_frame = args.out_frame_index if args.out_frame_index is not None else frame_idx
    if args.out_seq or args.out_frame_index is not None:
        print(f"Reconstructing from {seq_name} frame {frame_idx}, "
              f"naming output {out_seq} frame {out_frame}")

    out_name = f"{out_seq}_{out_frame:03d}_rgba"
    outdir = osp.join(args.hy3d_root, out_name)
    obj_name = f"{out_name.replace('_rgba', '')}_align.obj"
    obj_path = osp.join(outdir, obj_name)
    rgba_path = osp.join(outdir, f"{out_name}.png")
    glb_path = osp.join(outdir, f"{out_name}.glb")
    os.makedirs(outdir, exist_ok=True)

    if osp.isfile(obj_path) and not args.skip_sam3d:
        print(f"Output already exists: {obj_path}, skipping.")
        return

    if args.hires_video:
        rgb, mask = load_hires_frame_and_mask(
            args.hires_video, args.masks_root, seq_name, frame_idx,
            args.hires_frame_offset, args.kid)
    else:
        print(f"Extracting frame {frame_idx} from {args.video}")
        rgb = extract_frame(args.video, frame_idx)
        print(f"Loading object mask from {args.masks_root}")
        mask = load_object_mask(args.masks_root, seq_name, frame_idx, args.kid)

    if args.fill_mask_holes:
        mask = fill_mask_holes(mask)
    rgba_img = crop_rgba(rgb, mask, margin=args.margin, crop_size=args.crop_size)
    rgba_img.save(rgba_path)
    print(f"Saved RGBA: {rgba_path}")

    if args.skip_sam3d:
        print("Skipping SAM 3D inference (--skip_sam3d)")
        return

    inference = load_sam3d(args.sam3d_root, args.checkpoint_tag)
    output = run_sam3d(inference, rgba_img, args.seed, args.vertex_color)

    mesh = output["glb"]
    centre, scale = normalize_mesh(mesh)
    mesh.export(glb_path)
    print(f"Saved GLB: {glb_path}")
    save_pose(output, outdir, out_name, centre, scale)
    if "gs" in output:
        try:
            output["gs"].save_ply(osp.join(outdir, f"{out_name}_splat.ply"))
        except Exception as e:  # noqa: BLE001  (the splat is a viewing aid only)
            print(f"Could not save the Gaussian splat ({e}); the mesh is unaffected")

    if args.no_blender:
        export_obj_trimesh(mesh, outdir, obj_name)
    else:
        run_glb2obj(glb_path, outdir, obj_name, args.blender_path)
        attach_glb_texture(glb_path, outdir, obj_name)

    if not osp.isfile(obj_path):
        raise RuntimeError(f"No OBJ at {obj_path} after conversion")
    print(f"Done. Output: {obj_path}")


if __name__ == "__main__":
    main()
