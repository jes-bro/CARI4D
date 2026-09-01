#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

#SBATCH --job-name="hy3d-recon"
#SBATCH --output=hy3d-recon-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Object reconstruction (Step 2 of docs/custom_video.md): RGBA crop from the
# video + object mask, Hunyuan3D shape and texture, GLB -> OBJ via Blender.
#
# Usage:
#   sbatch scripts/slurm_hy3d_recon.sh <video.mp4> [frame_index] [extra args...]
#
# Example:
#   sbatch scripts/slurm_hy3d_recon.sh \
#       data/cari4d-demo/wild/videos/Date03_Sub01_gas_wild002.0.color.mp4
#
# Smoke test first -- extracts the frame, loads the mask and writes the RGBA
# crop in ~a second, with no model download and no GPU work. Run this before
# spending another allocation:
#   sbatch scripts/slurm_hy3d_recon.sh <video.mp4> 0 --skip_hy3d

set -euo pipefail

VIDEO=${1:?usage: sbatch scripts/slurm_hy3d_recon.sh <video.mp4> [frame_index] [extra args...]}
FRAME_INDEX=${2:-0}
shift $(( $# > 2 ? 2 : $# ))
EXTRA_ARGS=("$@")   # e.g. --skip_hy3d for a fast smoke test

# REPO defaults to the directory sbatch was invoked from, which the drivers
# guarantee is the repo root -- they cd there before submitting. That makes the
# whole pipeline work from any checkout without editing a line, instead of every
# job cd'ing into one person's home. An explicit REPO still wins, and the
# literal remains the last resort for a bare sbatch from somewhere else.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
# Per-clip work directories need their own mesh root; the flat default is the
# released demo's layout and would put every clip's object in one place under
# the same frame-indexed name.
HY3D_ROOT=${HY3D_ROOT:-$REPO/data/cari4d-demo/meshes}
CACHE_ROOT="${CACHE_ROOT:-/simurgh2/projects/ret-hoi}"

# run_sam3_masks.py writes its trimmed clip to <masks_root>/trimmed_vids/, so a
# video sitting in a trimmed_vids/ directory tells us its own masks_root. That
# makes `sbatch ... <clip>` work for any sequence with no path editing -- which
# matters because the trimmed clip is the ONLY video whose frame indices line up
# with the trimmed masks. An explicit MASKS_ROOT in the environment still wins.
VIDEO_DIR=$(dirname "$VIDEO")
if [[ -n "${MASKS_ROOT:-}" ]]; then
    :
elif [[ $(basename "$VIDEO_DIR") == "trimmed_vids" ]]; then
    MASKS_ROOT=$(dirname "$VIDEO_DIR")
else
    MASKS_ROOT=$REPO/data/cari4d-demo/wild/masks
fi

log() { echo "[hy3d $(date -u +%H:%M:%S)] $*"; }

# Every cache must live in project space -- /sailhome is over quota, which is
# what produced "OSError: [Errno 122] Disk quota exceeded". HF_HOME/HY3DGEN
# alone are not enough: hy3dgen's texgen JIT-builds CUDA extensions
# (custom_rasterizer, differentiable_renderer) into TORCH_EXTENSIONS_DIR, and
# that build silently stalls or retries when the target filesystem is full.
export HF_HOME=$CACHE_ROOT/hf_cache
export HY3DGEN_MODELS=$CACHE_ROOT/hy3dgen_cache
export TORCH_HOME=$CACHE_ROOT/torch_cache
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch_extensions
export XDG_CACHE_HOME=$CACHE_ROOT/xdg_cache
export TRITON_CACHE_DIR=$CACHE_ROOT/triton_cache
mkdir -p "$HF_HOME" "$HY3DGEN_MODELS" "$TORCH_HOME" \
         "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

# Without this Python block-buffers stdout into the .out file, so a job killed
# by the time limit loses every print() it made -- which is exactly why job
# 16400341 showed the banner below but not one line from the script itself.
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate "${HY3D_ENV:-hy3d}"

cd "$REPO"

# Pick whichever Blender tarball is actually extracted.
BLENDER=$(ls -d "$REPO"/Hunyuan3D-2/blender-*/blender 2>/dev/null | sort -V | head -1)
if [[ -z "$BLENDER" ]]; then
    echo "ERROR: no blender found under $REPO/Hunyuan3D-2/blender-*/" >&2
    exit 1
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "video=$VIDEO frame=$FRAME_INDEX extra=${EXTRA_ARGS[*]:-none}"
log "masks_root=$MASKS_ROOT"
if [[ ! -f $VIDEO ]]; then
    echo "ERROR: no such video: $VIDEO" >&2
    exit 1
fi
log "blender=$BLENDER"
log "mem=${SLURM_MEM_PER_NODE:-?}MB cpus=${SLURM_CPUS_PER_TASK:-?} timelimit=${SBATCH_TIMELIMIT:-see header}"
free -g | head -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Confirm the caches are writable and have room -- a silent failure here is the
# most likely explanation for a job that burns its whole allocation doing
# nothing visible.
log "cache root usage:"
df -h "$CACHE_ROOT" | tail -1
for d in "$HF_HOME" "$TORCH_EXTENSIONS_DIR"; do
    touch "$d/.writetest" 2>/dev/null && rm -f "$d/.writetest" \
        && log "  writable: $d" \
        || { echo "ERROR: not writable: $d" >&2; exit 1; }
done

# Watchdog: every 2 min report whether the GPU is busy and whether the HF cache
# is still growing. This is what distinguishes "downloading weights slowly"
# from "hung" from "actually generating" -- all three look identical otherwise.
watchdog() {
    while true; do
        sleep 120
        log "watchdog: gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
             --format=csv,noheader | tr '\n' ' ') hf_cache=$(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
    done
}
watchdog &
WATCHDOG_PID=$!
trap 'kill $WATCHDOG_PID 2>/dev/null || true' EXIT

log "launching run_hy3d_recon.py"
# Capture rc explicitly: under `set -e` a bare failure would exit before we
# could log it, which is how the previous run told us nothing.
rc=0
python -u prep/run_hy3d_recon.py \
    --video "$VIDEO" \
    --masks_root "$MASKS_ROOT" \
    --hy3d_root "$HY3D_ROOT" \
    --frame_index "$FRAME_INDEX" \
    --blender_path "$BLENDER" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} || rc=$?
log "run_hy3d_recon.py exited rc=$rc"

SEQ=$(basename "$VIDEO" | sed 's/\.0\.color\.mp4$//')
echo "[hy3d] done. Expected mesh:"
ls -la "$HY3D_ROOT/${SEQ}_$(printf '%03d' "$FRAME_INDEX")_rgba/" || true

# Exit with the reconstruction's status, not the listing's. Without this the
# job reports COMPLETED however the python went -- `ls ... || true` returns 0 --
# so a run killed by a bad GPU looked successful in sacct while writing no mesh,
# and any --dependency=afterok behind it would have started on nothing.
exit $rc
