# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Patch a local Hunyuan3D-2 clone so texgen loads under modern diffusers.

Hunyuan3D-2 loads its multiview paint model through a custom pipeline:

    DiffusionPipeline.from_pretrained(
        multiview_ckpt_path,
        custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16)

diffusers made ``trust_remote_code=True`` mandatory for custom pipelines after
the version Hunyuan3D-2 targeted, so on diffusers 0.39 this raises::

    ValueError: The directory .../hunyuanpaint contains custom code in
    pipeline.py which must be executed to correctly load the model.

Hunyuan3D-2's requirements.txt pins nothing, and downgrading diffusers to the
0.31.0 that docs/custom_video.md mentions drags transformers back far enough to
break the import chain (``cannot import name 'FLAX_WEIGHTS_NAME'``). Adding the
keyword is the smaller change.

Usage:
    python scripts/patch_hy3d_trust_remote_code.py [hunyuan3d_root]

Defaults to ./Hunyuan3D-2. Idempotent -- safe to re-run. Writes a .bak beside
each file it changes. Pass --revert to restore from those backups.
"""
import argparse
import os.path as osp
import shutil
import sys

# Files to patch, as (path relative to the Hunyuan3D-2 root, needle, replacement).
# Each needle must appear exactly once or the patch aborts rather than guess.
PATCHES = [
    (
        "hy3dgen/texgen/utils/multiview_utils.py",
        "custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16)",
        "custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16,\n"
        "            trust_remote_code=True)",
    ),
]

MARKER = "trust_remote_code=True"


def parse_args():
    """Parse the Hunyuan3D-2 root and the --revert flag."""
    parser = argparse.ArgumentParser(
        description="Add trust_remote_code=True to Hunyuan3D-2's custom pipeline loads")
    parser.add_argument("root", nargs="?", default="Hunyuan3D-2",
                        help="path to the Hunyuan3D-2 clone (default: ./Hunyuan3D-2)")
    parser.add_argument("--revert", action="store_true",
                        help="restore each patched file from its .bak")
    return parser.parse_args()


def revert_one(path):
    """Restore a single file from its .bak, returning True if anything changed."""
    backup = f"{path}.bak"
    if not osp.isfile(backup):
        print(f"  no backup for {path}, leaving as-is")
        return False
    shutil.move(backup, path)
    print(f"  reverted {path}")
    return True


def patch_one(path, needle, replacement):
    """Apply one substitution, skipping files that already carry the marker.

    Backs the file up to <path>.bak before writing. Aborts rather than guess if
    the needle is missing or appears more than once, since a silent no-op here
    surfaces much later as the same confusing diffusers ValueError.

    Returns:
        True if the file was modified, False if it was already patched.

    Raises:
        SystemExit: if the file is missing or the needle count is not 1.
    """
    if not osp.isfile(path):
        raise SystemExit(f"ERROR: not found: {path}\n"
                         f"Is the Hunyuan3D-2 root correct?")

    with open(path) as f:
        source = f.read()

    if MARKER in source:
        print(f"  already patched: {path}")
        return False

    count = source.count(needle)
    if count != 1:
        raise SystemExit(
            f"ERROR: expected exactly 1 occurrence of the target call in {path}, "
            f"found {count}. Hunyuan3D-2 may have changed; patch it by hand.")

    shutil.copy2(path, f"{path}.bak")
    with open(path, "w") as f:
        f.write(source.replace(needle, replacement))
    print(f"  patched {path} (backup at {path}.bak)")
    return True


def main():
    """Patch or revert every entry in PATCHES, then report what changed."""
    args = parse_args()
    root = osp.abspath(args.root)
    if not osp.isdir(root):
        raise SystemExit(f"ERROR: no such directory: {root}")

    print(f"Hunyuan3D-2 root: {root}")
    changed = 0
    for rel, needle, replacement in PATCHES:
        path = osp.join(root, rel)
        if args.revert:
            changed += revert_one(path)
        else:
            changed += patch_one(path, needle, replacement)

    verb = "reverted" if args.revert else "patched"
    print(f"{changed} file(s) {verb}.")
    if not args.revert and changed:
        print("Re-run the reconstruction; texgen should now load the multiview model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
