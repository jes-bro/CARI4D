#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G

#SBATCH --job-name="sam3d-recon"
#SBATCH --output=sam3d-recon-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Object reconstruction with SAM 3D Objects, the drop-in alternative to
# slurm_hy3d_recon.sh. Same arguments, same output layout, so the mesh root it
# writes goes straight into slurm_scale_object.sh / slurm_fp_onward.sh as
# HY3D_ROOT.
#
# Usage:
#   sbatch scripts/slurm_sam3d_recon.sh <video.mp4> [frame_index] [extra args...]
#
# Reconstruct from the Aria ego clip (object close and unoccluded) but name the
# mesh for the exo sequence the pipeline tracks in:
#   HY3D_ROOT=<work>/meshes-sam3d MASKS_ROOT=<ego masks_root> \
#   sbatch scripts/slurm_sam3d_recon.sh <ego masks_root>/trimmed_vids/<ego>.0.color.mp4 120 \
#       --out_seq <exo seq> --out_frame_index 0
#
# Smoke test first (RGBA crop only, no model, no GPU work):
#   sbatch scripts/slurm_sam3d_recon.sh <video.mp4> 0 --skip_sam3d
#
# The 48G constraint is SAM 3D's own requirement ("at least 32 Gb of VRAM");
# drop it if the partition names its big cards differently.

set -euo pipefail

VIDEO=${1:?usage: sbatch scripts/slurm_sam3d_recon.sh <video.mp4> [frame_index] [extra args...]}
FRAME_INDEX=${2:-0}
shift $(( $# > 2 ? 2 : $# ))
EXTRA_ARGS=("$@")

REPO="${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
HY3D_ROOT=${HY3D_ROOT:-$REPO/data/cari4d-demo/meshes-sam3d}
SAM3D_ROOT=${SAM3D_ROOT:-$REPO/sam-3d-objects}
CACHE_ROOT="${CACHE_ROOT:-/simurgh2/projects/ret-hoi}"

# Same rule as the Hunyuan3D job: a clip inside trimmed_vids/ names its own
# masks_root, because that is the only video whose frames line up with them.
VIDEO_DIR=$(dirname "$VIDEO")
if [[ -n "${MASKS_ROOT:-}" ]]; then
    :
elif [[ $(basename "$VIDEO_DIR") == "trimmed_vids" ]]; then
    MASKS_ROOT=$(dirname "$VIDEO_DIR")
else
    MASKS_ROOT=$REPO/data/cari4d-demo/wild/masks
fi

log() { echo "[sam3d $(date -u +%H:%M:%S)] $*"; }

export HF_HOME=$CACHE_ROOT/hf_cache
export TORCH_HOME=$CACHE_ROOT/torch_cache
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch_extensions
export XDG_CACHE_HOME=$CACHE_ROOT/xdg_cache
export TRITON_CACHE_DIR=$CACHE_ROOT/triton_cache
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SAM3D_ENV:-sam3d-objects}"

cd "$REPO"

# Blender is optional here: without one the script falls back to trimesh for
# the OBJ, which keeps the texture but skips Blender's decimation pass.
BLENDER=$(ls -d "$REPO"/Hunyuan3D-2/blender-*/blender "$REPO"/blender-*/blender 2>/dev/null | sort -V | head -1 || true)
BLENDER_ARGS=()
if [[ -n "$BLENDER" ]]; then
    BLENDER_ARGS=(--blender_path "$BLENDER")
else
    BLENDER_ARGS=(--no_blender)
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO sam3d_root=$SAM3D_ROOT"
log "video=$VIDEO frame=$FRAME_INDEX extra=${EXTRA_ARGS[*]:-none}"
log "masks_root=$MASKS_ROOT hy3d_root=$HY3D_ROOT"
log "blender=${BLENDER:-none, using trimesh}"
if [[ ! -f $VIDEO ]]; then
    echo "ERROR: no such video: $VIDEO" >&2
    exit 1
fi
if [[ ! -f $SAM3D_ROOT/checkpoints/${SAM3D_TAG:-hf}/pipeline.yaml ]]; then
    echo "ERROR: no checkpoints under $SAM3D_ROOT/checkpoints/${SAM3D_TAG:-hf}; see docs/custom_video.md" >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

watchdog() {
    while true; do
        sleep 120
        log "watchdog: gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
             --format=csv,noheader | tr '\n' ' ')"
    done
}
watchdog &
WATCHDOG_PID=$!
trap 'kill $WATCHDOG_PID 2>/dev/null || true' EXIT

log "launching run_sam3d_recon.py"
rc=0
python -u prep/run_sam3d_recon.py \
    --video "$VIDEO" \
    --masks_root "$MASKS_ROOT" \
    --hy3d_root "$HY3D_ROOT" \
    --frame_index "$FRAME_INDEX" \
    --sam3d_root "$SAM3D_ROOT" \
    --checkpoint_tag "${SAM3D_TAG:-hf}" \
    "${BLENDER_ARGS[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} || rc=$?
log "run_sam3d_recon.py exited rc=$rc"

SEQ=$(basename "$VIDEO" | sed 's/\.0\.color\.mp4$//')
OUT_FRAME=$FRAME_INDEX
for i in "${!EXTRA_ARGS[@]}"; do
    [ "${EXTRA_ARGS[$i]}" = "--out_seq" ] && SEQ="${EXTRA_ARGS[$((i+1))]}"
    [ "${EXTRA_ARGS[$i]}" = "--out_frame_index" ] && OUT_FRAME="${EXTRA_ARGS[$((i+1))]}"
done
echo "[sam3d] done. Expected mesh:"
ls -la "$HY3D_ROOT/${SEQ}_$(printf '%03d' "$OUT_FRAME")_rgba/" || true

exit $rc
