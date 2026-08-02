# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Convert HuggingFace .bin checkpoints to .safetensors, in place.

Recent transformers refuses to torch.load a .bin unless torch >= 2.6::

    ValueError: Due to a serious vulnerability issue in `torch.load`, even
    with `weights_only=True`, we now require users to upgrade torch to at
    least v2.6 ... This version restriction does not apply when loading
    files with safetensors.

Hunyuan3D-2's paint pipeline ships text_encoder and vae as .bin only, so
texgen dies on torch 2.5.1 even though every other component has safetensors.

Upgrading torch would force a rebuild of custom_rasterizer, and downgrading
transformers drags diffusers back with it. Converting the two files sidesteps
both: the restriction is a transformers policy, not a torch defect, so
torch.load(weights_only=True) still works here. Once a .safetensors sits
beside the .bin, both transformers and diffusers prefer it and the check
never fires.

Usage:
    python scripts/bin_to_safetensors.py <model_dir> [--dry-run]

Walks model_dir, converting every .bin that has no .safetensors sibling.
Idempotent -- existing safetensors are left alone. Adds files only; the .bin
is never deleted or modified, so this is safe to run against an HF cache.
"""
import argparse
import os
import os.path as osp
import sys

# HF's naming convention: transformers models use model.safetensors, diffusers
# submodules use diffusion_pytorch_model.safetensors. from_pretrained looks for
# these exact names, so a conversion under any other name is silently ignored.
BIN_TO_SAFETENSORS = {
    "pytorch_model.bin": "model.safetensors",
    "diffusion_pytorch_model.bin": "diffusion_pytorch_model.safetensors",
}


def parse_args():
    """Parse the model directory and the --dry-run flag."""
    parser = argparse.ArgumentParser(
        description="Convert .bin checkpoints to .safetensors beside them")
    parser.add_argument("model_dir",
                        help="directory to walk, e.g. an HF snapshot subfolder")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be converted without writing")
    return parser.parse_args()


def find_candidates(model_dir):
    """Return (bin_path, safetensors_path) for every .bin lacking a sibling.

    Only filenames in BIN_TO_SAFETENSORS are considered -- an arbitrarily named
    .bin would convert to a name from_pretrained never looks for, which is
    worse than leaving it alone because it looks like the job succeeded.
    """
    candidates = []
    for dirpath, _, filenames in os.walk(model_dir):
        for fname in sorted(filenames):
            target = BIN_TO_SAFETENSORS.get(fname)
            if target is None:
                continue
            st_path = osp.join(dirpath, target)
            if osp.exists(st_path):
                continue
            candidates.append((osp.join(dirpath, fname), st_path))
    return candidates


def convert(bin_path, st_path):
    """Load one .bin and write it as safetensors.

    Tensors are cloned before saving because safetensors rejects tensors that
    share storage, and tied weights (a text encoder's input and output
    embeddings, typically) do exactly that. Cloning costs a little memory and
    makes the shared copies independent, which is what the format requires.

    Non-tensor entries are dropped -- safetensors holds tensors only. Nothing
    in these checkpoints needs them, but the count is reported so a surprise
    is visible rather than silent.
    """
    import torch
    from safetensors.torch import save_file

    print(f"  loading {bin_path}")
    state = torch.load(bin_path, map_location="cpu", weights_only=True)

    tensors, skipped = {}, []
    for key, value in state.items():
        if hasattr(value, "clone"):
            tensors[key] = value.clone().contiguous()
        else:
            skipped.append(key)

    if not tensors:
        raise SystemExit(f"ERROR: no tensors found in {bin_path}")
    if skipped:
        print(f"  skipped {len(skipped)} non-tensor entries: {skipped[:5]}")

    # Write to a temp name first so an interrupted run cannot leave a
    # truncated .safetensors that later loads look like a valid checkpoint.
    tmp_path = f"{st_path}.tmp"
    save_file(tensors, tmp_path, metadata={"format": "pt"})
    os.replace(tmp_path, st_path)
    size_mb = osp.getsize(st_path) / (1024 * 1024)
    print(f"  wrote {st_path} ({len(tensors)} tensors, {size_mb:.1f} MB)")


def main():
    """Convert every eligible .bin under model_dir, reporting what changed."""
    args = parse_args()
    model_dir = osp.abspath(args.model_dir)
    if not osp.isdir(model_dir):
        raise SystemExit(f"ERROR: no such directory: {model_dir}")

    print(f"Scanning {model_dir}")
    candidates = find_candidates(model_dir)
    if not candidates:
        print("Nothing to convert -- every .bin already has a .safetensors sibling.")
        return 0

    print(f"{len(candidates)} file(s) to convert:")
    for bin_path, st_path in candidates:
        print(f"  {osp.relpath(bin_path, model_dir)} -> {osp.basename(st_path)}")

    if args.dry_run:
        print("Dry run, nothing written.")
        return 0

    for bin_path, st_path in candidates:
        convert(bin_path, st_path)

    print(f"Converted {len(candidates)} file(s). Re-run the reconstruction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
