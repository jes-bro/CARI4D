#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="cari4d-opt-only"
#SBATCH --output=cari4d-opt-only-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Re-run ONLY the joint optimizer (demo-custom.sh step 7) on an existing
# CoCoNet bundle. For optimizer config experiments -- loss weights, what gets
# optimized -- where re-running FP and CoCoNet (slurm_fp_onward.sh) would
# recompute identical inputs just to reach the stage under test.
#
# EXTRA is appended verbatim after the standard arguments; OmegaConf's
# last-key-wins CLI makes it override anything, so e.g. EXTRA can flip
# opt_smpl_trans=True. Give SAVE_NAME a fresh value per experiment or the
# output directory overwrites the previous run's.
#
#   MASKS_ROOT=rect-bball PACKED_ROOT=rect-bball HY3D_ROOT=data/cari4d-demo/meshes \
#   VIDEO_ROOT=rect-bball-aligned SAVE_NAME=optv2strans EXTRA="opt_smpl_trans=True" \
#       sbatch scripts/slurm_opt_only.sh output/coconet/<exp>+<step><id>/<seq>.pth
#
# Output lands in output/opt/<exp>+<step><id>-hy3d3-<SAVE_NAME>/.

set -euo pipefail

PTH_FILE=${1:?usage: sbatch scripts/slurm_opt_only.sh <coconet .pth>}

# REPO defaults to the directory sbatch was invoked from, which the drivers
# guarantee is the repo root -- they cd there before submitting. That makes the
# whole pipeline work from any checkout without editing a line, instead of every
# job cd'ing into one person's home. An explicit REPO still wins, and the
# literal remains the last resort for a bare sbatch from somewhere else.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
CACHE_ROOT="${CACHE_ROOT:-/simurgh2/projects/ret-hoi}"

log() { echo "[opt-only $(date -u +%H:%M:%S)] $*"; }

export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch_cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton_cache}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CARI4D_ENV:-newcari4d}"

cd "$REPO"

masks_root="${MASKS_ROOT:-data/cari4d-demo/videogen/masks/}"
packed_root="${PACKED_ROOT:-data/cari4d-demo/videogen/packed/}"
hy3d_root="${HY3D_ROOT:-data/cari4d-demo/videogen/meshes}"
video_root="${VIDEO_ROOT:?set VIDEO_ROOT to the aligned clip directory}"
save_name="${SAVE_NAME:-optv2}"
extra="${EXTRA:-}"

log "host=$(hostname) job=${SLURM_JOB_ID:-none} env=${CONDA_DEFAULT_ENV:-none}"
log "code=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo +dirty)"
log "pth_file=${PTH_FILE}"
log "video_root=${video_root}  masks_root=${masks_root}  packed_root=${packed_root}  hy3d_root=${hy3d_root}"
log "save_name=${save_name}  extra='${extra}'"

for required in "$PTH_FILE" "$video_root" "$masks_root" "$packed_root" "${hy3d_root}-metric"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: missing required input: $required" >&2
        exit 1
    fi
done

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('cuda ok:', torch.cuda.get_device_name(0))"

# Step 7 verbatim from demo-custom.sh; ${extra} last so it overrides.
python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300  save_name=${save_name} batch_size=64 opt_rot=True \
opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=2.0 w_j2d=0.006 opt_smpl_trans=False opt_betas=False  \
pth_file=${PTH_FILE}  wild_video=True use_input=True \
video_root=${video_root} \
packed_root=${packed_root} \
masks_root=${masks_root}  \
hy3d_meshes_root=${hy3d_root}-metric outpath=output/opt ${extra}

log "done; results:"
ls -la output/opt/*"${save_name}"* 2>/dev/null || true
