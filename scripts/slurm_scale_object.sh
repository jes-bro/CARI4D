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

# The object's metric size, MEASURED across the calibrated views rather than
# searched for.
#
# A single-image reconstruction has no scale in it: a photograph of a
# basketball and one of a beach ball are identical up to size, so Hunyuan3D
# returns shape in arbitrary units and the metres must come from elsewhere. The
# released pipeline searches for them -- render at thirty candidate sizes, keep
# whichever best explains one image -- because with one camera size and distance
# are degenerate and fitting is the only option.
#
# That search produced a 5.8cm basketball here, four times running: it kept
# selecting the smallest candidate it was offered while every number upstream
# looked healthy, and the ball rendered three pixels wide, which reads as "no
# ball" rather than as a scale error.
#
# This pipeline is not single-camera. The object's position is triangulated
# from several calibrated views to centimetre accuracy, and a distance plus an
# apparent size IS a size: D = Z * d_px / f, one measurement per view per
# frame, hundreds of them. Nothing in it knows what the object is.
#
# SCALE_METHOD=search restores the old behaviour for comparison.
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
log "masks_root=$masks_root  hy3d_root=$hy3d_root  method=${SCALE_METHOD:-measure}"

depth_video=${VIDEO/.color.mp4/.depth-reg.mp4}
for required in "$VIDEO" "$depth_video" "$masks_root" "$hy3d_root"; do
    [ -e "$required" ] || { echo "ERROR: missing required input: $required" >&2; exit 1; }
done

if [ "${SCALE_METHOD:-measure}" = "measure" ]; then
    : "${OBJECT_XYZ:?set OBJECT_XYZ}" "${CALIB:?set CALIB}" "${MASKS_DIR:?set MASKS_DIR}"
    MESH_SRC="$(ls "$hy3d_root"/*/*_align.obj 2>/dev/null | head -1 || true)"
    [ -n "$MESH_SRC" ] || { echo "ERROR: no mesh under $hy3d_root" >&2; exit 1; }
    # The same --view list triangulation used: fisheye masks with the fisheye
    # calibration, each at its own resolution.
    views=(--view "${PIPE_CAM:-cam04}:$MASKS_DIR:$SEQ")
    for c in ${AUX_CAMS:-}; do views+=(--view "$c:$MASKS_DIR:$c-4k"); done
    log "measuring the object across $(( ${#views[@]} / 2 )) view(s)"
    python prep/scale_object_mesh.py --mesh "$MESH_SRC" \
        --object_xyz "$OBJECT_XYZ" --calib "$CALIB" "${views[@]}" \
        --out_root "$hy3d_root-metric"
else
    log "SCALE_METHOD=search: the render-and-fit path"
    python tools/estimate_scale_video.py --wild_video --video "$VIDEO" \
        --masks_root "$masks_root" --hy3d_root "$hy3d_root" -o "$hy3d_root-metric" \
        --erode_depth_thres "$SCALE_ERODE"
fi

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
