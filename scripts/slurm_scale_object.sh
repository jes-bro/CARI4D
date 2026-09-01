#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="recon-scale"
#SBATCH --output=recon-scale-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# demo-custom.sh step 5.1 -- the object's metric scale -- moved to AFTER the
# depth injection.
#
# Why it moved. The scale is recovered by rendering the mesh at a range of
# scales and asking which one explains the image. But a silhouette cannot
# separate scale from distance: a small object close and a large object far
# project identically. Only depth breaks that tie, and before the injection the
# depth inside the object's mask is the monocular estimate -- the one that read
# 6.15m to 15.94m for a ball genuinely at 6.5m, and the reason the injection
# step exists at all.
#
# Run against that, the search walked to the smallest scale it was offered:
# 0.03, giving a basketball 5.8cm across instead of 24cm. Position was right to
# a few centimetres, size was a quarter of true, and the ball rendered about
# three pixels wide -- invisible, while every number upstream looked healthy.
#
# After the injection the object's depth is the triangulated one, centimetre
# class, and the degeneracy is broken.
#
#   MASKS_ROOT=work/<seq>/rect HY3D_ROOT=work/<seq>/meshes \
#       sbatch scripts/slurm_scale_object.sh work/<seq>/rect-aligned/<seq>.0.color.mp4

set -euo pipefail

VIDEO=${1:?usage: sbatch scripts/slurm_scale_object.sh <injected aligned video.mp4>}

REPO="${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
CACHE_ROOT="${CACHE_ROOT:-/simurgh2/projects/ret-hoi}"

log() { echo "[scale $(date -u +%H:%M:%S)] $*"; }

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

masks_root="${MASKS_ROOT:?set MASKS_ROOT}"
hy3d_root="${HY3D_ROOT:?set HY3D_ROOT}"
# 'auto' derives the erode threshold from Z/f -- this object's distance over the
# focal length, the object's own size cancelling out.
SCALE_ERODE="${SCALE_ERODE:-auto}"

log "host=$(hostname) job=${SLURM_JOB_ID:-none} env=${CONDA_DEFAULT_ENV:-none}"
log "repo=$REPO"
log "video=$VIDEO (injected)"
log "masks_root=$masks_root  hy3d_root=$hy3d_root  scale_erode=$SCALE_ERODE"

depth_video=${VIDEO/.color.mp4/.depth-reg.mp4}
for required in "$VIDEO" "$depth_video" "$masks_root" "$hy3d_root"; do
    [ -e "$required" ] || { echo "ERROR: missing required input: $required" >&2; exit 1; }
done

python tools/estimate_scale_video.py --wild_video --video "$VIDEO" \
    --masks_root "$masks_root" --hy3d_root "$hy3d_root" -o "$hy3d_root-metric" \
    --erode_depth_thres "$SCALE_ERODE"

# The scale is the whole output, so report it rather than leaving it to be
# discovered three jobs later as an object too small to see.
log "resulting object size:"
find "$hy3d_root-metric" -name "*_align.obj" -exec python -c "
import trimesh, sys
m = trimesh.load(sys.argv[1], process=False)
e = m.extents
print(f'  {sys.argv[1]}')
print(f'  extents {e} m, largest {max(e):.3f} m')
" {} \;
