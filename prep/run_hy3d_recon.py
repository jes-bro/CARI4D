# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
End-to-end object reconstruction: extract RGBA from video + masks, run Hunyuan3D, convert GLB to OBJ.

Usage:
    python prep/run_hy3d_recon.py \
        --video data/cari4d-demo/wild/videos/<seq>.0.color.mp4 \
        --masks_root data/cari4d-demo/wild/masks \
        --hy3d_root data/cari4d-demo/meshes \
        --frame_index 0 \
        --blender_path /path/to/blender \
        --kid 0
"""
import argparse
import os
import os.path as osp
import shutil
import subprocess

import cv2
import h5py
import numpy as np
from PIL import Image


def parse_args():
    """Parse the CLI arguments for the object-reconstruction pipeline.

    The --skip_* flags exist so the expensive stages can be bypassed
    independently: --skip_hy3d for a fast check that the video, mask and crop
    are sane, --skip_glb2obj to stop at the GLB when Blender is unavailable.
    """
    parser = argparse.ArgumentParser(description="Extract RGBA, run Hunyuan3D, convert GLB to OBJ")
    parser.add_argument("--video", required=True, help="Path to input video, e.g. <seq>.0.color.mp4")
    parser.add_argument("--masks_root", required=True, help="Directory containing HDF5 mask files")
    parser.add_argument("--hy3d_root", required=True, help="Output root for Hunyuan3D meshes")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Video frame index to use for reconstruction (default: 0)")
    parser.add_argument("--kid", type=int, default=0, help="Camera/kinect ID (default: 0)")
    parser.add_argument("--blender_path", default="blender",
                        help="Path to Blender executable (default: 'blender')")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Total border margin ratio for cropping (default: 0.2)")
    parser.add_argument("--crop_size", type=int, default=512,
                        help="Output RGBA image size (default: 512)")
    parser.add_argument("--seed", type=int, default=600, help="Random seed (default: 600)")
    parser.add_argument("--skip_hy3d", action="store_true",
                        help="Skip Hunyuan3D inference, only do RGBA extraction")
    parser.add_argument("--skip_glb2obj", action="store_true",
                        help="Skip GLB to OBJ conversion")
    parser.add_argument("--hires_video", default=None,
                        help="Take the reconstruction frame from this higher-resolution "
                             "copy of the same take, upscaling the mask to match. Only "
                             "this one frame needs resolution, so the masks can stay at "
                             "whatever resolution SAM3 ran at.")
    parser.add_argument("--hires_frame_offset", type=int, default=0,
                        help="Frame index in --hires_video corresponding to frame 0 of "
                             "--video. Set this to the trim start 'lo' reported by "
                             "run_sam3_masks.py when --video is a trimmed clip and "
                             "--hires_video is the untrimmed original (default: 0)")
    return parser.parse_args()


def extract_seq_name(video_path):
    """Extract sequence name from video filename like <seq>.0.color.mp4"""
    basename = osp.basename(video_path)
    if ".0.color.mp4" in basename:
        return basename.replace(".0.color.mp4", "")
    return osp.splitext(basename)[0]


def extract_frame(video_path, frame_index):
    """Extract a single RGB frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_object_mask(masks_root, seq_name, frame_index, kid):
    """Load object mask from HDF5 file."""
    h5_path = osp.join(masks_root, f"{seq_name}_masks_k{kid}.h5")
    if not osp.isfile(h5_path):
        raise FileNotFoundError(f"Mask file not found: {h5_path}")
    frame_id = f"{frame_index:06d}"
    key = f"{seq_name}/{frame_id}-k{kid}.obj_rend_mask.png"
    with h5py.File(h5_path, 'r') as f:
        if key not in f:
            raise KeyError(f"Mask key '{key}' not found in {h5_path}")
        mask = f[key][:].astype(np.uint8) * 255
    return mask


def load_hires_frame_and_mask(hires_video, masks_root, seq_name, frame_index,
                              frame_offset, kid):
    """Read the reconstruction frame at full resolution, with the mask upscaled.

    Only the single reconstruction frame benefits from resolution -- everything
    downstream is derived from the RGBA crop -- so running SAM3 over the whole
    take at full resolution to serve one frame is wasted GPU time. This reads
    that frame from a larger copy of the same take and resamples the existing
    mask up to match.

    The mask is upscaled with linear interpolation rather than nearest, which
    leaves a soft edge. That is deliberate: crop_rgba thresholds at >127 to find
    the bounding box but keeps the raw values as alpha, so a soft edge gives
    Hunyuan3D an antialiased silhouette instead of the staircase a 4-5x nearest
    upscale would produce.

    Args:
        hires_video: higher-resolution copy of the same take.
        masks_root: directory holding <seq>_masks_k<kid>.h5.
        seq_name: sequence name.
        frame_index: frame index in the low-resolution/trimmed sequence.
        frame_offset: index in hires_video corresponding to frame 0 of the
            low-resolution sequence. Non-zero when --video is a trimmed clip and
            hires_video is the untrimmed original.
        kid: camera/kinect id.

    Returns:
        (rgb, mask) at the high-resolution frame's dimensions.
    """
    hires_index = frame_index + frame_offset
    print(f'Extracting frame {hires_index} from {hires_video} '
          f'(frame {frame_index} + offset {frame_offset})')
    rgb = extract_frame(hires_video, hires_index)

    print(f'Loading object mask from {masks_root}')
    mask = load_object_mask(masks_root, seq_name, frame_index, kid)

    H, W = rgb.shape[:2]
    if mask.shape[:2] != (H, W):
        mh, mw = mask.shape[:2]
        print(f'Upscaling mask {mw}x{mh} -> {W}x{H} '
              f'({W / float(mw):.2f}x) to match the hi-res frame')
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        print('Mask already matches the hi-res frame, no upscaling needed')

    return rgb, mask


def crop_rgba(rgb, mask, margin=0.2, crop_size=512):
    """Apply mask as alpha, crop square around object with margin, resize.

    Args:
        rgb: (H, W, 3) uint8
        mask: (H, W) uint8, 255=object
        margin: total margin ratio (crop_size = 1.2 * bbox_size)
        crop_size: output image size
    Returns:
        RGBA PIL Image of size (crop_size, crop_size)
    """
    H, W = mask.shape
    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        raise ValueError("Object mask is empty, cannot crop")

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    bh = y_max - y_min
    bw = x_max - x_min
    bbox_size = max(bh, bw)

    # Total margin: crop_size = (1 + margin) * bbox_size
    crop_len = int(bbox_size * (1.0 + margin))
    # Center of bbox
    cy = (y_min + y_max) / 2.0
    cx = (x_min + x_max) / 2.0

    # Square crop coordinates
    y1 = int(cy - crop_len / 2.0)
    x1 = int(cx - crop_len / 2.0)
    y2 = y1 + crop_len
    x2 = x1 + crop_len

    # Compute padding if crop extends beyond image
    pad_top = max(0, -y1)
    pad_left = max(0, -x1)
    pad_bottom = max(0, y2 - H)
    pad_right = max(0, x2 - W)

    # Clamp to image bounds
    y1_c = max(0, y1)
    x1_c = max(0, x1)
    y2_c = min(H, y2)
    x2_c = min(W, x2)

    # Crop and pad
    rgb_crop = rgb[y1_c:y2_c, x1_c:x2_c]
    mask_crop = mask[y1_c:y2_c, x1_c:x2_c]

    if pad_top or pad_bottom or pad_left or pad_right:
        rgb_crop = np.pad(rgb_crop,
                          ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                          mode='constant', constant_values=0)
        mask_crop = np.pad(mask_crop,
                           ((pad_top, pad_bottom), (pad_left, pad_right)),
                           mode='constant', constant_values=0)

    # Compose RGBA
    rgba = np.concatenate([rgb_crop, mask_crop[..., None]], axis=-1)
    rgba_img = Image.fromarray(rgba, 'RGBA')
    rgba_img = rgba_img.resize((crop_size, crop_size), Image.LANCZOS)
    return rgba_img


def run_hunyuan3d(rgba_img, outdir, glb_name, seed=600):
    """Run Hunyuan3D shape + texture generation."""
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from hy3dgen.text2image import seed_everything

    seed_everything(seed)

    model_path = 'tencent/Hunyuan3D-2'
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(
        model_path, subfolder='hunyuan3d-paint-v2-0-turbo'
    )

    mesh = pipeline_shapegen(image=rgba_img)[0]
    print('Shape generation done')
    mesh = pipeline_texgen(mesh, image=rgba_img)
    print('Texture generation done')

    glb_path = osp.join(outdir, glb_name)
    mesh.export(glb_path)
    print(f'Saved GLB: {glb_path}')
    return glb_path


def run_glb2obj(glb_path, outdir, obj_name, blender_path):
    """Convert GLB to OBJ with Blender and flatten the result into outdir.

    glb2obj.py writes a subdirectory <outdir>/<glb_basename>/ holding the .obj,
    its .mtl and the copied texture images. All of it has to move up into
    <outdir> together: fp_hy3d_track.py globs for <seq>*/*_align.obj exactly one
    level down, and the .obj references its .mtl by bare filename, so moving the
    .obj alone would leave a dangling mtllib line and cost the texture that
    FoundationPose renders with.

    Only the .obj is renamed to obj_name; the .mtl keeps its original name so
    the mtllib reference inside the .obj stays valid after the move.

    Args:
        glb_path: the .glb Blender should convert.
        outdir: destination directory, also where the .glb already lives.
        obj_name: final basename, e.g. '<seq>_000_align.obj'.
        blender_path: Blender executable.

    Raises:
        subprocess.CalledProcessError: if Blender exits non-zero.
        RuntimeError: if Blender succeeds but writes no OBJ.
    """
    script_path = osp.join(osp.dirname(osp.abspath(__file__)), 'glb2obj.py')
    glb_dir = osp.dirname(glb_path)
    cmd = [blender_path, '-b', '-P', script_path, '--', glb_dir, outdir]
    print(f'Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)

    glb_basename = osp.splitext(osp.basename(glb_path))[0]
    produced_dir = osp.join(outdir, glb_basename)
    produced_obj = osp.join(produced_dir, f'{glb_basename}.obj')
    target_obj = osp.join(outdir, obj_name)

    if not osp.isdir(produced_dir) and osp.isfile(target_obj):
        print(f'OBJ already exists: {target_obj}')
        return

    if not osp.isfile(produced_obj):
        raise RuntimeError(
            f'Blender exited 0 but wrote no OBJ at {produced_obj}. '
            f'Check the [glb2obj] lines above for the real failure.')

    # Move the .obj plus every sidecar (.mtl, textures) up one level.
    for fname in sorted(os.listdir(produced_dir)):
        src = osp.join(produced_dir, fname)
        dst = target_obj if fname == f'{glb_basename}.obj' else osp.join(outdir, fname)
        if osp.abspath(src) == osp.abspath(dst):
            continue
        if osp.isdir(dst):
            shutil.rmtree(dst)
        elif osp.isfile(dst):
            os.remove(dst)
        shutil.move(src, dst)
        print(f'  {fname} -> {osp.relpath(dst, outdir)}')

    if not os.listdir(produced_dir):
        os.rmdir(produced_dir)
    print(f'Flattened Blender output into {outdir}')


def find_glb_texture(glb_path):
    """Return the baked texture image from a GLB, or None if it has none.

    trimesh exposes the image as ``material.image`` for a SimpleMaterial and
    ``material.baseColorTexture`` for a PBRMaterial; Hunyuan3D writes the
    latter, but both are checked so this does not depend on which loader path
    trimesh takes.

    Args:
        glb_path: the .glb written by Hunyuan3D.

    Returns:
        A PIL image, or None when the GLB carries no texture.
    """
    import trimesh

    mesh = trimesh.load(glb_path, force='mesh')
    material = getattr(mesh.visual, 'material', None)
    if material is None:
        return None
    return getattr(material, 'image', None) or getattr(material, 'baseColorTexture', None)


def attach_glb_texture(glb_path, outdir, obj_name):
    """Extract the GLB's texture and point the converted OBJ's .mtl at it.

    Blender's OBJ exporter drops the GLB's packed image even with
    path_mode='COPY', leaving a .mtl with no map_Kd and a mesh that renders
    flat grey. The UVs survive both the decimation and the export, and the
    texture image is unchanged by either, so re-attaching the original image
    is enough -- no re-conversion, and the decimated geometry is kept.

    This matters because FoundationPose does render-and-compare against the
    video frames; an untextured mesh weakens tracking on objects whose shape
    alone is ambiguous.

    Args:
        glb_path: the .glb Hunyuan3D produced.
        outdir: directory holding the flattened .obj and .mtl.
        obj_name: the final OBJ basename, used to name the texture file.

    Returns:
        Path to the written texture, or None if there was nothing to attach.
    """
    image = find_glb_texture(glb_path)
    if image is None:
        print('No texture found in the GLB, leaving the .mtl unchanged.')
        return None

    mtl_files = sorted(f for f in os.listdir(outdir) if f.endswith('.mtl'))
    if not mtl_files:
        print(f'No .mtl in {outdir}, cannot attach the texture.')
        return None
    if len(mtl_files) > 1:
        print(f'Several .mtl files in {outdir} ({mtl_files}); using {mtl_files[0]}.')
    mtl_path = osp.join(outdir, mtl_files[0])

    with open(mtl_path) as f:
        mtl = f.read()
    if 'map_Kd' in mtl:
        print(f'{mtl_files[0]} already references a texture, leaving it alone.')
        return None

    tex_name = f"{osp.splitext(obj_name)[0]}_texture.png"
    tex_path = osp.join(outdir, tex_name)
    image.save(tex_path)
    print(f'Saved texture: {tex_path} ({image.size[0]}x{image.size[1]})')

    # map_Kd binds to the material block it follows. These meshes carry a
    # single newmtl, so appending is correct; with several the texture would
    # attach only to the last, hence the warning above.
    if mtl and not mtl.endswith('\n'):
        mtl += '\n'
    with open(mtl_path, 'w') as f:
        f.write(f'{mtl}map_Kd {tex_name}\n')
    print(f'Added map_Kd {tex_name} to {mtl_files[0]}')
    return tex_path


def main():
    """Run the six reconstruction steps for one video frame.

    Extracts the frame, loads its object mask, writes the cropped RGBA, runs
    Hunyuan3D shape and texture generation, then converts the GLB to the
    <seq>_<frame:03d>_align.obj that fp_hy3d_track.py expects. Returns early if
    that OBJ already exists, so re-running is cheap.
    """
    args = parse_args()

    seq_name = extract_seq_name(args.video)
    frame_idx = args.frame_index

    # Output directory and file names following the convention:
    # <hy3d_root>/<seq>_<frame_index:03d>_rgba/<seq>_<frame_index:03d>_align.obj
    out_name = f"{seq_name}_{frame_idx:03d}_rgba"
    outdir = osp.join(args.hy3d_root, out_name)
    obj_name = f"{out_name.replace('_rgba', '')}_align.obj"
    obj_path = osp.join(outdir, obj_name)
    rgba_path = osp.join(outdir, f"{out_name}.png")
    glb_name = f"{out_name}.glb"

    os.makedirs(outdir, exist_ok=True)

    # Check if final output exists
    if osp.isfile(obj_path) and not args.skip_hy3d:
        print(f'Output already exists: {obj_path}, skipping.')
        return

    # Step 1-2: Extract the RGB frame and its object mask. Only this one frame
    # needs resolution -- everything downstream works off the RGBA crop -- so
    # --hires_video lets it come from a larger copy of the same take while the
    # masks stay at whatever resolution SAM3 ran at.
    if args.hires_video:
        rgb, mask = load_hires_frame_and_mask(
            args.hires_video, args.masks_root, seq_name, frame_idx,
            args.hires_frame_offset, args.kid)
    else:
        print(f'Extracting frame {frame_idx} from {args.video}')
        rgb = extract_frame(args.video, frame_idx)
        print(f'Loading object mask from {args.masks_root}')
        mask = load_object_mask(args.masks_root, seq_name, frame_idx, args.kid)

    # Step 3-4: Apply mask, crop, save RGBA
    rgba_img = crop_rgba(rgb, mask, margin=args.margin, crop_size=args.crop_size)
    rgba_img.save(rgba_path)
    print(f'Saved RGBA: {rgba_path}')

    if args.skip_hy3d:
        print('Skipping Hunyuan3D inference (--skip_hy3d)')
        return

    # Step 5: Run Hunyuan3D
    glb_path = run_hunyuan3d(rgba_img, outdir, glb_name, seed=args.seed)

    if args.skip_glb2obj:
        print('Skipping GLB to OBJ conversion (--skip_glb2obj)')
        return

    # Step 6: Convert GLB to OBJ
    run_glb2obj(glb_path, outdir, obj_name, args.blender_path)

    # Step 7: Re-attach the texture Blender dropped on export.
    attach_glb_texture(glb_path, outdir, obj_name)

    print(f'Done. Output: {obj_path}')


if __name__ == '__main__':
    main()
