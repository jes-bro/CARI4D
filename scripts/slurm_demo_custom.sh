#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

#SBATCH --job-name="cari4d-demo"
#SBATCH --output=cari4d-demo-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Step 4 of docs/custom_video.md: the whole pipeline, via scripts/demo-custom.sh.
# Seven stages -- UniDepth, NLF, SMPL-H fitting, depth/human alignment, metric
# scale, FoundationPose tracking, CoCoNet, then 3000 optimisation steps.
#
#   MASKS_ROOT=data/cari4d-demo/wild/masks \
#   PACKED_ROOT=data/cari4d-demo/wild/packed-coco \
#   HY3D_ROOT=data/cari4d-demo/meshes \
#       sbatch scripts/slurm_demo_custom.sh data/cari4d-demo/wild/videos/<seq>.0.color.mp4
#
# sbatch's default --export=ALL carries those through to demo-custom.sh, which
# reads them directly -- nothing needs forwarding here.
#
# 12h is deliberately generous: the run is long and a time-limit kill loses
# every stage completed so far, since demo-custom.sh has no resume. Lower it
# with `sbatch --time=` once you know what a sequence actually costs -- shorter
# requests are far more likely to backfill into a free slot.

set -euo pipefail

VIDEO=${1:?usage: sbatch scripts/slurm_demo_custom.sh <video.mp4>}

# REPO defaults to the directory sbatch was invoked from, which the drivers
# guarantee is the repo root -- they cd there before submitting. That makes the
# whole pipeline work from any checkout without editing a line, instead of every
# job cd'ing into one person's home. An explicit REPO still wins, and the
# literal remains the last resort for a bare sbatch from somewhere else.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
CACHE_ROOT="${CACHE_ROOT:-/simurgh2/projects/ret-hoi}"

log() { echo "[cari4d $(date -u +%H:%M:%S)] $*"; }

# Caches to project space -- /sailhome is over quota, and these stages pull
# UniDepth and NLF weights and JIT-build CUDA extensions.
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch_cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton_cache}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

# Unbuffered, so a job killed at the time limit keeps the log of what finished.
export PYTHONUNBUFFERED=1

# conda may not be on PATH at all: sbatch --export=ALL hands the job the
# SUBMITTING shell's environment, so a terminal that never initialised conda
# produces jobs that cannot find it -- and the failure is "conda: command not
# found" from inside a queued job, an hour after the mistake.
if ! command -v conda >/dev/null 2>&1; then
    for _rc in "$HOME/.bashrc" "$HOME/.bash_profile" /etc/profile.d/conda.sh; do
        [ -f "$_rc" ] && . "$_rc" >/dev/null 2>&1 || true
        command -v conda >/dev/null 2>&1 && break
    done
fi
command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda not found. It is not on PATH here and sourcing your shell" >&2
    echo "       profile did not provide it. Run 'source ~/.bashrc' in the shell" >&2
    echo "       you submit from, or set CONDA_EXE." >&2
    exit 1
}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CARI4D_ENV:-newcari4d}"

cd "$REPO"

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "video=$VIDEO"
log "env=${CONDA_DEFAULT_ENV:-none} cwd=$(pwd)"
log "masks_root=${MASKS_ROOT:-<default>} packed_root=${PACKED_ROOT:-<default>} hy3d_root=${HY3D_ROOT:-<default>}"
free -g | head -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The pipeline dies at stage 1 without a GPU, but only after loading UniDepth,
# and the message is a warning buried in the log rather than an error.
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('cuda ok:', torch.cuda.get_device_name(0))"

# videoio shells out to ffprobe to read video metadata; without it stage 1
# fails with a FileNotFoundError that does not name the pipeline stage.
command -v ffprobe >/dev/null || { echo "ERROR: ffprobe not found. conda install -c conda-forge ffmpeg" >&2; exit 1; }

log "launching demo-custom.sh"
rc=0
bash scripts/demo-custom.sh "$VIDEO" || rc=$?
log "demo-custom.sh exited rc=$rc"

VIDEO_PREFIX=$(basename "$VIDEO" | cut -d. -f1)
log "results:"
ls -la "output/opt"/*"${VIDEO_PREFIX}"* 2>/dev/null || ls -la output/opt 2>/dev/null || true
exit $rc
