#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="ego-object"
#SBATCH --output=ego-object-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# The object, tracked monocularly in the EGO view, delivered to the pipeline
# camera. Replaces triangulation + the pipeline-camera FoundationPose run for
# objects the exo cameras see badly: a pot a metre from the wearer's glasses
# is large, sharp and unoccluded there and a few dozen pixels anywhere else.
#
#   1  rectify_aria.py       ego clip + masks -> pinhole, named <seq>-ego
#   2  unidepth_behave.py    monocular metric depth for that clip
#   3  the mesh, copied in under the ego sequence's name
#   4  estimate_scale_video  metric scale, fitted in the ego view
#   5  fp_hy3d_track.py      FoundationPose in the ego camera
#   6  ego_poses_to_cam.py   poses into the pipeline camera (FP pickle) and the
#                            object centre per frame (object_xyz format), so
#                            inject_object_depth, CoCoNet and the optimizer run
#                            unchanged; the metric mesh copied in for them too
#
# Driven by scripts/recon_solve.sh with OBJECT_FROM=ego, which exports every
# variable below. The pipeline-camera stages that follow (unidepth, NLF, SMPL-H,
# alignment, inject, CoCoNet, opt) are untouched; slurm_fp_onward.sh runs with
# SKIP_FP=1 so it reads the pickle written here instead of tracking again.
#
# Knobs, all for the ego camera's geometry (an object about a metre away):
#   EGO_ZFAR   (3)      furthest depth kept, metres
#   EGO_ERODE  (0.005)  depth erosion threshold, metres: Z/f with margin
#   EGO_BAND   (1.0)    object-to-hands depth band, metres

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
CACHE_ROOT="${CACHE_ROOT:-/simurgh2/projects/ret-hoi}"
log() { echo "[ego-object $(date -u +%H:%M:%S)] $*"; }

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

: "${SEQ:?set SEQ}" "${PIPE_CAM:?set PIPE_CAM}" "${CALIB:?set CALIB}"
: "${EGO_CLIP:?set EGO_CLIP}" "${EGO_MASKS:?set EGO_MASKS}" "${MASKS_DIR:?set MASKS_DIR}"
: "${ARIA_CALIB:?set ARIA_CALIB}" "${ARIA_EXTRINSICS:?set ARIA_EXTRINSICS}"
: "${WINDOW_JSON:?set WINDOW_JSON}" "${MESH_DIR:?set MESH_DIR}"
: "${EGO_RECT_DIR:?set EGO_RECT_DIR}" "${EGO_MESH_DIR:?set EGO_MESH_DIR}" "${EGO_FP_DIR:?set EGO_FP_DIR}"
: "${EGO_PIPE_SEQ:?set EGO_PIPE_SEQ}" "${FP_DIR:?set FP_DIR}" "${OBJECT_XYZ_EGO:?set OBJECT_XYZ_EGO}"

EGO_ZFAR="${EGO_ZFAR:-3}"
EGO_ERODE="${EGO_ERODE:-0.005}"
EGO_BAND="${EGO_BAND:-1.0}"
DEPTH_MAD_K="${DEPTH_MAD_K:-3.0}"

log "host=$(hostname) job=${SLURM_JOB_ID:-none} env=${CONDA_DEFAULT_ENV:-none}"
log "code=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo +dirty)"
log "seq=$SEQ  ego seq=$EGO_PIPE_SEQ  pipeline cam=$PIPE_CAM"
log "ego clip=$EGO_CLIP"
log "knobs: zfar=$EGO_ZFAR erode=$EGO_ERODE band=$EGO_BAND mad_k=$DEPTH_MAD_K"

for required in "$EGO_CLIP" "$EGO_MASKS" "$ARIA_CALIB" "$ARIA_EXTRINSICS" "$WINDOW_JSON" "$CALIB"; do
    [ -e "$required" ] || { echo "ERROR: missing input: $required" >&2; exit 1; }
done
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('cuda ok:', torch.cuda.get_device_name(0))"

RECT_CLIP="$EGO_RECT_DIR/$EGO_PIPE_SEQ.0.color.mp4"

# --- 1: pinhole ------------------------------------------------------------------
if [ -f "$RECT_CLIP" ] && [ -f "$EGO_RECT_DIR/${EGO_PIPE_SEQ}_masks_k0.h5" ]; then
    log "1 rectify: already done ($RECT_CLIP)"
else
    log "1 rectify the ego clip and its masks"
    python prep/rectify_aria.py --video "$EGO_CLIP" --calib "$ARIA_CALIB" \
        --masks_root "$MASKS_DIR" --out_dir "$EGO_RECT_DIR" --out_seq "$EGO_PIPE_SEQ"
fi

# --- 2: depth ----------------------------------------------------------------------
log "2 unidepth on the rectified ego clip"
python prep/unidepth_behave.py --wild_video --video "$RECT_CLIP" -o "$EGO_RECT_DIR"

# --- 3: the mesh, under the ego sequence's name ----------------------------------
# The reconstructed object sits in $MESH_DIR named for the pipeline sequence.
# estimate_scale_video.py and fp_hy3d_track.py glob by the video's prefix, so
# the ego run needs a copy named for its own sequence. Directory and OBJ are
# renamed; the .mtl and texture keep their names, which the OBJ refers to.
src_mesh_dir=$(ls -d "$MESH_DIR"/*_rgba 2>/dev/null | head -1 || true)
[ -n "$src_mesh_dir" ] || { echo "ERROR: no reconstructed mesh under $MESH_DIR" >&2; exit 1; }
frame=$(basename "$src_mesh_dir" | sed -E 's/.*_([0-9]{3})_rgba$/\1/')
ego_mesh_dir="$EGO_MESH_DIR/${EGO_PIPE_SEQ}_${frame}_rgba"
if [ ! -d "$ego_mesh_dir" ]; then
    mkdir -p "$EGO_MESH_DIR" && cp -r "$src_mesh_dir" "$ego_mesh_dir"
    for f in "$ego_mesh_dir"/*_align.obj; do
        [ -f "$f" ] || continue
        [ "$(basename "$f")" = "${EGO_PIPE_SEQ}_${frame}_align.obj" ] || mv "$f" "$ego_mesh_dir/${EGO_PIPE_SEQ}_${frame}_align.obj"
    done
fi
log "3 mesh for the ego run: $ego_mesh_dir"

# --- 4: metric scale, fitted in the ego view -------------------------------------
# Render-and-fit against this clip's depth on the frame where the object mask
# is largest. The erosion threshold derives from Z/f for this camera.
log "4 metric scale"
python tools/estimate_scale_video.py --wild_video --video "$RECT_CLIP" \
    --masks_root "$EGO_RECT_DIR" --hy3d_root "$EGO_MESH_DIR" -o "$EGO_MESH_DIR-metric" \
    --erode_depth_thres auto
metric_obj=$(ls "$EGO_MESH_DIR-metric"/*/*_align.obj 2>/dev/null | head -1 || true)
[ -n "$metric_obj" ] || { echo "ERROR: scale step wrote no mesh under $EGO_MESH_DIR-metric" >&2; exit 1; }
python -c "
import trimesh, sys
m = trimesh.load(sys.argv[1], process=False)
print(f'  metric mesh {sys.argv[1]}: extents {m.extents} m, largest {max(m.extents):.3f} m')
" "$metric_obj"

# --- 5: FoundationPose in the ego camera -----------------------------------------
# No --reinit_every: a pot's orientation is observable, and re-registering
# every frame would spin it. The hands stand in for the person in the depth
# band test, which is what the ego masks' person channel holds.
log "5 foundationpose on the ego clip"
python prep/fp_hy3d_track.py --viz_path x --wild_video --kid 0 \
    --masks_root "$EGO_RECT_DIR" --hy3d_root="$EGO_MESH_DIR-metric" \
    --video "$RECT_CLIP" -o "$EGO_FP_DIR" --zfar "$EGO_ZFAR" -tstart 0 \
    --erode_depth_thres "$EGO_ERODE" \
    --depth_human_band "$EGO_BAND" --depth_mad_k "$DEPTH_MAD_K"
ego_pkl="$EGO_FP_DIR/${EGO_PIPE_SEQ}_all.pkl"
[ -f "$ego_pkl" ] || { echo "ERROR: FoundationPose wrote no $ego_pkl" >&2; exit 1; }

# --- 6: into the pipeline camera ----------------------------------------------------
lo=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['chosen']['lo'])" "$WINDOW_JSON")
log "6 ego poses -> $PIPE_CAM (clip frame 0 = take frame $lo)"
mkdir -p "$FP_DIR" "$(dirname "$OBJECT_XYZ_EGO")"
python prep/ego_poses_to_cam.py --fp_pkl "$ego_pkl" --calib "$CALIB" --cam "$PIPE_CAM" \
    --aria_extrinsics "$ARIA_EXTRINSICS" --offset "$lo" --mesh "$metric_obj" \
    --out_pkl "$FP_DIR/${SEQ}_all.pkl" --out_xyz "$OBJECT_XYZ_EGO"

# The metric mesh for the pipeline stages, renamed for the pipeline sequence.
# Everything the scale step wrote is carried over, including debug outputs.
mkdir -p "$MESH_DIR-metric"
for d in "$EGO_MESH_DIR-metric"/*; do
    [ -e "$d" ] || continue
    dst="$MESH_DIR-metric/$(basename "$d" | sed "s/${EGO_PIPE_SEQ}/${SEQ}/g")"
    rm -rf "$dst"; cp -r "$d" "$dst"
    if [ -d "$dst" ]; then
        for f in "$dst"/*"${EGO_PIPE_SEQ}"*; do
            [ -e "$f" ] || continue
            mv "$f" "$dst/$(basename "$f" | sed "s/${EGO_PIPE_SEQ}/${SEQ}/g")"
        done
        # The OBJ names its .mtl (mtllib) and the .mtl names its texture
        # (map_Kd), both by the old name; CoCoNet renders with that texture,
        # so a dangling reference would cost it.
        for f in "$dst"/*.obj "$dst"/*.mtl; do
            [ -f "$f" ] && sed -i "s/${EGO_PIPE_SEQ}/${SEQ}/g" "$f"
        done
    fi
done
log "metric mesh for the pipeline: $(ls "$MESH_DIR-metric"/*/*_align.obj 2>/dev/null | head -1)"

log "done. outputs:"
ls -la "$EGO_RECT_DIR" "$EGO_FP_DIR" "$FP_DIR" "$(dirname "$OBJECT_XYZ_EGO")"
