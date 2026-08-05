# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Rename a prepared sequence to the Date_Sub_object_action convention.

Several stages parse meaning out of the sequence name rather than treating it
as an opaque id:

    run_nlf_sepK.py:41        _sub_gender[prefix.split('_')[1]]   -> stage 2
    fp_hy3d_track.py:28       prefix.split('_')[2]                -> stage 5.2
    run_horefine.py:94,146,355                                    -> stage 6
    opt_refineout.py:168,180,340                                  -> stage 7

So part [1] must be a key in behave_data.const._sub_gender, which also selects
the SMPL body model's gender, and part [2] is taken as the object name. A name
like Ego-Exo4D's `cam04` has no underscores at all and fails at stage 2.

Renaming by hand is error-prone because two of the artifacts carry the sequence
name *inside* them: the mask HDF5 keys everything under a group named for the
sequence, and the OBJ references its material by filename, which the material
in turn references the texture by. Missing either fails much later and less
obviously than a missing file.

Usage:
    python prep/rename_sequence.py --old cam04 --new Date03_Sub01_bball_dribble \\
        --video_dir sam3masks/trimmed_vids --masks_root sam3masks \\
        --packed_root data/cari4d-demo/wild/packed-coco \\
        --hy3d_root data/cari4d-demo/meshes --dry_run

Copies rather than moves by default, so the originals survive a mistake.
"""
import argparse
import os
import os.path as osp
import shutil
import sys
from glob import glob

import h5py

sys.path.append(os.getcwd())

try:
    from behave_data.const import _sub_gender
except Exception:  # keep the script usable outside a configured checkout
    _sub_gender = {}


def parse_args():
    """Parse the old and new names plus the roots holding each artifact."""
    parser = argparse.ArgumentParser(
        description="Rename a sequence's video, masks, keypoints and mesh together")
    parser.add_argument("--old", required=True, help="current sequence name, e.g. cam04")
    parser.add_argument("--new", required=True,
                        help="new name, e.g. Date03_Sub01_bball_dribble")
    parser.add_argument("--video_dir", default=None,
                        help="directory holding <seq>.0.color.mp4")
    parser.add_argument("--masks_root", default=None,
                        help="directory holding <seq>_masks_k<kid>.h5")
    parser.add_argument("--packed_root", default=None,
                        help="directory holding <seq>_GT-packed.pkl")
    parser.add_argument("--hy3d_root", default=None,
                        help="directory holding <seq>_<frame>_rgba/")
    parser.add_argument("--move", action="store_true",
                        help="move instead of copy; the originals are not kept")
    parser.add_argument("--dry_run", action="store_true",
                        help="report what would happen without touching anything")
    return parser.parse_args()


def check_new_name(new):
    """Warn if the new name will not satisfy the stages that parse it.

    Reported rather than enforced: the convention is not documented anywhere as
    a rule, and a caller may know something this check does not.
    """
    parts = new.split("_")
    if len(parts) < 3:
        print(f"  WARNING: '{new}' has {len(parts)} underscore-separated part(s); "
              f"stages index [1] and [2] and will raise IndexError")
        return
    if _sub_gender and parts[1] not in _sub_gender:
        print(f"  WARNING: part [1] is '{parts[1]}', which is not a key in "
              f"_sub_gender. Stage 2 looks up the SMPL gender with it and will "
              f"raise KeyError. Known keys include: "
              f"{', '.join(sorted(_sub_gender)[:6])}...")
        return
    gender = _sub_gender.get(parts[1], "?") if _sub_gender else "?"
    print(f"  name ok: subject '{parts[1]}' (gender {gender}), object '{parts[2]}'")


def transfer(src, dst, move, dry_run):
    """Copy or move one path, reporting it. Returns True if it existed."""
    if not osp.exists(src):
        return False
    verb = "move" if move else "copy"
    print(f"  {verb}: {src}\n      -> {dst}")
    if dry_run:
        return True
    os.makedirs(osp.dirname(dst), exist_ok=True)
    if move:
        shutil.move(src, dst)
    elif osp.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return True


def rename_video(video_dir, old, new, move, dry_run):
    """Transfer every <old>.<kid>.* file, e.g. the clip and any depth beside it."""
    print("video:")
    found = 0
    for src in sorted(glob(osp.join(video_dir, f"{old}.*"))):
        dst = osp.join(video_dir, osp.basename(src).replace(old, new, 1))
        found += transfer(src, dst, move, dry_run)
    if not found:
        print(f"  none found matching {osp.join(video_dir, old + '.*')}")
    return found


def rename_masks(masks_root, old, new, move, dry_run):
    """Transfer the mask h5 and rewrite its internal group name.

    The group is named for the sequence and every frame's datasets live under
    it, so a file renamed without this loads as an empty sequence -- which
    surfaces as "mask not found" much later.
    """
    print("masks:")
    found = 0
    for src in sorted(glob(osp.join(masks_root, f"{old}_masks_k*.h5"))):
        dst = osp.join(masks_root, osp.basename(src).replace(old, new, 1))
        if not transfer(src, dst, move, dry_run):
            continue
        found += 1
        if dry_run:
            print(f"      and rename the internal HDF5 group '{old}' -> '{new}'")
            continue
        with h5py.File(dst, "a") as f:
            if old in f:
                f.move(old, new)
                print(f"      renamed internal group '{old}' -> '{new}'")
            elif new in f:
                print(f"      internal group already '{new}'")
            else:
                print(f"      WARNING: no group '{old}' inside; found {list(f.keys())}")
    if not found:
        print(f"  none found matching {osp.join(masks_root, old + '_masks_k*.h5')}")
    return found


def rename_packed(packed_root, old, new, move, dry_run):
    """Transfer the packed keypoints, which carry no sequence name internally."""
    print("keypoints:")
    src = osp.join(packed_root, f"{old}_GT-packed.pkl")
    dst = osp.join(packed_root, f"{new}_GT-packed.pkl")
    if not transfer(src, dst, move, dry_run):
        print(f"  none found at {src}")
        return 0
    return 1


def rewrite_references(path, old, new, dry_run):
    """Rewrite <old> to <new> inside a text file, reporting affected lines.

    Used for the OBJ's mtllib and the MTL's map_Kd. Both name their target by
    bare filename, so renaming the files without this leaves the mesh untextured
    -- and nothing errors, which is the worst version of this failure.
    """
    if not osp.isfile(path):
        return
    with open(path) as f:
        lines = f.readlines()
    hits = [i for i, l in enumerate(lines) if old in l]
    if not hits:
        return
    for i in hits:
        print(f"      {osp.basename(path)}: {lines[i].strip()}")
    if dry_run:
        return
    with open(path, "w") as f:
        f.writelines(l.replace(old, new) for l in lines)


def rename_meshes(hy3d_root, old, new, move, dry_run):
    """Transfer each <old>_<frame>_rgba/ directory, its files and its references."""
    print("meshes:")
    found = 0
    for src_dir in sorted(glob(osp.join(hy3d_root, f"{old}_*_rgba"))):
        dst_dir = osp.join(hy3d_root, osp.basename(src_dir).replace(old, new, 1))
        if not transfer(src_dir, dst_dir, move, dry_run):
            continue
        found += 1
        if dry_run:
            for name in sorted(os.listdir(src_dir)):
                if old in name:
                    print(f"      rename {name} -> {name.replace(old, new, 1)}")
            for name in sorted(os.listdir(src_dir)):
                if name.endswith((".obj", ".mtl")):
                    rewrite_references(osp.join(src_dir, name), old, new, dry_run)
            continue
        for name in sorted(os.listdir(dst_dir)):
            if old in name:
                os.rename(osp.join(dst_dir, name),
                          osp.join(dst_dir, name.replace(old, new, 1)))
        for name in sorted(os.listdir(dst_dir)):
            if name.endswith((".obj", ".mtl")):
                rewrite_references(osp.join(dst_dir, name), old, new, dry_run)
    if not found:
        print(f"  none found matching {osp.join(hy3d_root, old + '_*_rgba')}")
    return found


def main():
    """Rename every artifact of one sequence, or report what would change."""
    args = parse_args()
    if args.old == args.new:
        raise SystemExit("ERROR: --old and --new are the same")

    print(f"{args.old}  ->  {args.new}")
    check_new_name(args.new)
    if args.dry_run:
        print("DRY RUN -- nothing will be written\n")
    else:
        print(f"mode: {'move' if args.move else 'copy'}\n")

    total = 0
    if args.video_dir:
        total += rename_video(args.video_dir, args.old, args.new, args.move, args.dry_run)
    if args.masks_root:
        total += rename_masks(args.masks_root, args.old, args.new, args.move, args.dry_run)
    if args.packed_root:
        total += rename_packed(args.packed_root, args.old, args.new, args.move, args.dry_run)
    if args.hy3d_root:
        total += rename_meshes(args.hy3d_root, args.old, args.new, args.move, args.dry_run)

    if not any([args.video_dir, args.masks_root, args.packed_root, args.hy3d_root]):
        raise SystemExit("ERROR: give at least one of --video_dir, --masks_root, "
                         "--packed_root, --hy3d_root")

    print(f"\n{total} artifact(s) {'would be ' if args.dry_run else ''}handled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
