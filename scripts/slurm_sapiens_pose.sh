#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="sapiens-pose"
#SBATCH --output=sapiens-pose-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Step 3 of docs/custom_video.md: 2D human keypoints via Sapiens, packed into
# the <seq>_GT-packed.pkl the pipeline reads.
#
#   sbatch scripts/slurm_sapiens_pose.sh <video.mp4>
#   MASKS_ROOT=... PACKED_ROOT=... sbatch scripts/slurm_sapiens_pose.sh <video.mp4>
#
# The 1h default is generous for a short clip and keeps the job backfill-eligible.
# Override for a long take with `sbatch --time=04:00:00 ...`.

set -euo pipefail

VIDEO=${1:?usage: sbatch scripts/slurm_sapiens_pose.sh <video.mp4>}

REPO=/simurgh2/projects/ret-hoi/CARI4D
MASKS_ROOT="${MASKS_ROOT:-$REPO/data/cari4d-demo/wild/masks}"
PACKED_ROOT="${PACKED_ROOT:-$REPO/data/cari4d-demo/wild/packed-coco}"
CHECKPOINT="${CHECKPOINT:-/simurgh2/projects/ret-hoi/sapiens_ckpts/sapiens_host/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth}"

log() { echo "[sapiens $(date -u +%H:%M:%S)] $*"; }

# Caches to project space -- /sailhome is over quota. mmengine and torch.hub
# both write there by default and fail late and confusingly when it is full.
export HF_HOME="${HF_HOME:-/simurgh2/projects/ret-hoi/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/simurgh2/projects/ret-hoi/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/simurgh2/projects/ret-hoi/xdg_cache}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"

# Unbuffered, so a job killed at the time limit does not lose its output.
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CARI4D_ENV:-newcari4d}"

# run_sapiens_pose.py builds its sapiens paths from os.getcwd(), inserting
# sapiens/pose and sapiens/pretrain into sys.path -- mmpose and mmpretrain come
# from the clone, not from pip. So the working directory is not optional.
cd "$REPO"

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "video=$VIDEO"
log "masks_root=$MASKS_ROOT"
log "packed_root=$PACKED_ROOT"
log "checkpoint=$CHECKPOINT"
log "env=${CONDA_DEFAULT_ENV:-none} cwd=$(pwd)"

for path in "$VIDEO" "$CHECKPOINT"; do
    if [[ ! -f $path ]]; then
        echo "ERROR: no such file: $path" >&2
        exit 1
    fi
done
for dir in "$MASKS_ROOT" "$REPO/sapiens/pose/mmpose" "$REPO/sapiens/pretrain/mmpretrain"; do
    if [[ ! -d $dir ]]; then
        echo "ERROR: no such directory: $dir" >&2
        exit 1
    fi
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

log "launching run_sapiens_pose.py"
rc=0
python -u prep/run_sapiens_pose.py \
    --video "$VIDEO" \
    --masks_root "$MASKS_ROOT" \
    --packed_root "$PACKED_ROOT" \
    --checkpoint "$CHECKPOINT" || rc=$?
log "run_sapiens_pose.py exited rc=$rc"

SEQ=$(basename "$VIDEO" | sed 's/\.0\.color\.mp4$//;s/\.mp4$//')
log "expected output:"
ls -la "$PACKED_ROOT/${SEQ}_GT-packed.pkl" || true
exit $rc
