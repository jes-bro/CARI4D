#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --job-name="recon-prop"
#SBATCH --output=recon-prop-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# One subcommand of prep/static_prop.py on the cluster.
#
#   sbatch scripts/slurm_static_prop.sh geometry     # CPU: triangulate, pool, pick the mesh frame
#   sbatch --gres=gpu:1 scripts/slurm_static_prop.sh register   # GPU: FoundationPose, medoid, sidecar
#
# Driven by scripts/recon_prop.sh, which exports every variable read below and
# adds --gres for the register step. Standalone use means exporting them.
#
# Runs in the pipeline's own env (newcari4d): triangulate_object.py, the
# rectifier and FoundationPose all live there. The Hunyuan3D mesh is NOT built
# here -- that needs the hy3d env and goes through slurm_hy3d_recon.sh with the
# prop's own HY3D_ROOT, exactly as recon_object.sh does for the tracked object.

set -euo pipefail

CMD="${1:?usage: sbatch scripts/slurm_static_prop.sh <geometry|register|check>}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
CACHE_ROOT="${CACHE_ROOT:-/simurgh2/projects/ret-hoi}"

log() { echo "[prop $(date -u +%H:%M:%S)] $*"; }

export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch_cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg_cache}"
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CARI4D_ENV:-newcari4d}"
cd "$REPO"

: "${SEQ:?set SEQ}" "${PROP:?set PROP}" "${CALIB:?set CALIB}" "${PIPE_CAM:?set PIPE_CAM}"
: "${PROP_DIR:?set PROP_DIR}" "${PROP_MASKS_DIR:?set PROP_MASKS_DIR}"

log "host=$(hostname) job=${SLURM_JOB_ID:-none} env=${CONDA_DEFAULT_ENV:-none} gpu=${CUDA_VISIBLE_DEVICES:-none}"
log "repo=$REPO code=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo +dirty)"
log "seq=$SEQ prop=$PROP cmd=$CMD pipe_cam=$PIPE_CAM aux=${AUX_CAMS:-}"

# Register needs the metric mesh; found the same way slurm_scale_object.sh
# leaves it, under the prop's own roots -- so two props never share a glob.
extra=()
if [ "$CMD" = register ]; then
    : "${ALIGNED_CLIP:?set ALIGNED_CLIP}"
    MESH="${PROP_MESH:-$(ls "$PROP_DIR"/meshes-metric/*/*_align.obj 2>/dev/null | head -1 || true)}"
    [ -n "$MESH" ] || { echo "ERROR: no metric mesh under $PROP_DIR/meshes-metric -- run the scale step" >&2; exit 1; }
    extra=(--aligned_clip "$ALIGNED_CLIP" --mesh "$MESH"
           ${PROP_K:+--k "$PROP_K"} ${PROP_DEPTH_MODE:+--depth_mode "$PROP_DEPTH_MODE"}
           ${ERODE_DEPTH_THRES:+--erode_depth_thres "$ERODE_DEPTH_THRES"})
fi

# shellcheck disable=SC2086
python prep/static_prop.py "$CMD" --prop "$PROP" --seq "$SEQ" --take "${TAKE:-}" \
    --calib "$CALIB" --pipe_cam "$PIPE_CAM" --aux_cams ${AUX_CAMS:-} \
    --masks_dir "$PROP_MASKS_DIR" --prop_dir "$PROP_DIR" \
    ${TRI_INLIER_PX:+--inlier_px "$TRI_INLIER_PX"} \
    ${extra[@]+"${extra[@]}"}

log "done."
