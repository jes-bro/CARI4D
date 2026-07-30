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
#   sbatch scripts/slurm_hy3d_recon.sh <video.mp4> [frame_index]
#
# Example:
#   sbatch scripts/slurm_hy3d_recon.sh \
#       data/cari4d-demo/wild/videos/Date03_Sub01_gas_wild002.0.color.mp4

set -euo pipefail

VIDEO=${1:?usage: sbatch scripts/slurm_hy3d_recon.sh <video.mp4> [frame_index]}
FRAME_INDEX=${2:-0}

REPO=/simurgh2/projects/ret-hoi/CARI4D
MASKS_ROOT=$REPO/data/cari4d-demo/wild/masks
HY3D_ROOT=$REPO/data/cari4d-demo/meshes

# Caches must live in project space -- /sailhome is over quota and the weights
# are ~10GB (this is what produced "OSError: [Errno 122] Disk quota exceeded").
export HF_HOME=/simurgh2/projects/ret-hoi/hf_cache
export HY3DGEN_MODELS=/simurgh2/projects/ret-hoi/hy3dgen_cache
mkdir -p "$HF_HOME" "$HY3DGEN_MODELS"

source ~/.bashrc
conda activate hy3d

cd "$REPO"

# Pick whichever Blender tarball is actually extracted.
BLENDER=$(ls -d "$REPO"/Hunyuan3D-2/blender-*/blender 2>/dev/null | sort -V | head -1)
if [[ -z "$BLENDER" ]]; then
    echo "ERROR: no blender found under $REPO/Hunyuan3D-2/blender-*/" >&2
    exit 1
fi

echo "[hy3d] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[hy3d] video=$VIDEO frame=$FRAME_INDEX"
echo "[hy3d] blender=$BLENDER"
echo "[hy3d] mem=${SLURM_MEM_PER_NODE:-?}MB cpus=${SLURM_CPUS_PER_TASK:-?}"
free -g | head -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python prep/run_hy3d_recon.py \
    --video "$VIDEO" \
    --masks_root "$MASKS_ROOT" \
    --hy3d_root "$HY3D_ROOT" \
    --frame_index "$FRAME_INDEX" \
    --blender_path "$BLENDER"

SEQ=$(basename "$VIDEO" | sed 's/\.0\.color\.mp4$//')
echo "[hy3d] done. Expected mesh:"
ls -la "$HY3D_ROOT/${SEQ}_$(printf '%03d' "$FRAME_INDEX")_rgba/" || true
