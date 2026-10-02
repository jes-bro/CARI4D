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
#   EGO_OBJECT_SIZE     the object's longest dimension in metres, when known;
#                       replaces the depth fit (a saucepan with handle: ~0.35)
#                       and anchors the depth map's scale to the object
#   EGO_DEPTH_SCALE     that depth factor typed by hand, instead of measured
#   EGO_RGB_ONLY=1      FoundationPose refines on appearance only; the depth
#                       map seeds the first frame and is otherwise ignored.
#                       For when the pot lands a constant distance too far
#                       along the ego's line of sight: the depth was wrong
#                       and the refiner followed it

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
if [ -n "${EGO_OBJECT_SIZE:-}" ]; then
    # The person knows the object. A stated longest dimension in metres beats
    # a fit against monocular depth, and is what to reach for when the fit
    # comes out visibly wrong.
    log "4 metric scale: stated, longest axis $EGO_OBJECT_SIZE m"
    rm -rf "$EGO_MESH_DIR-metric"
    python prep/scale_mesh_to_size.py --mesh "$ego_mesh_dir/${EGO_PIPE_SEQ}_${frame}_align.obj" \
        --size "$EGO_OBJECT_SIZE" --out_root "$EGO_MESH_DIR-metric"
else
    log "4 metric scale: fitted to the ego depth"
    python tools/estimate_scale_video.py --wild_video --video "$RECT_CLIP" \
        --masks_root "$EGO_RECT_DIR" --hy3d_root "$EGO_MESH_DIR" -o "$EGO_MESH_DIR-metric" \
        --erode_depth_thres auto
fi
metric_obj=$(ls "$EGO_MESH_DIR-metric"/*/*_align.obj 2>/dev/null | head -1 || true)
[ -n "$metric_obj" ] || { echo "ERROR: scale step wrote no mesh under $EGO_MESH_DIR-metric" >&2; exit 1; }
python -c "
import trimesh, sys
m = trimesh.load(sys.argv[1], process=False)
print(f'  metric mesh {sys.argv[1]}: extents {m.extents} m, largest {max(m.extents):.3f} m')
" "$metric_obj"

# --- 4b: the depth map's scale, from the object's size --------------------------
# UniDepth's ego depth has a free global scale, and the first ego run put the
# pot metres away for want of an anchor. With the size stated, the object is
# its own anchor: its apparent size in the mask says how far it must be.
DEPTH_SCALE="${EGO_DEPTH_SCALE:-1.0}"
if [ -n "${EGO_OBJECT_SIZE:-}" ] && [ -z "${EGO_DEPTH_SCALE:-}" ]; then
    log "4b depth scale from the object's size"
    python prep/estimate_depth_scale.py --video "$RECT_CLIP" --masks_root "$EGO_RECT_DIR" --mesh "$metric_obj"
    # tail -1: the video reader prints a warning on stdout before the number.
    DEPTH_SCALE=$(python prep/estimate_depth_scale.py --video "$RECT_CLIP" --masks_root "$EGO_RECT_DIR" \
        --mesh "$metric_obj" --factor_only 2>/dev/null | tail -1)
    case "$DEPTH_SCALE" in
        ''|*[!0-9.]*) echo "ERROR: depth scale came out as '$DEPTH_SCALE'" >&2; exit 1 ;;
    esac
fi
log "depth scale for the ego track: $DEPTH_SCALE"

# --- 5: FoundationPose in the ego camera -----------------------------------------
# No --reinit_every: a pot's orientation is observable, and re-registering
# every frame would spin it. The hands stand in for the person in the depth
# band test, which is what the ego masks' person channel holds.
ego_pkl="$EGO_FP_DIR/${EGO_PIPE_SEQ}_all.pkl"
# A stated size changes the mesh the poses were fitted with, so the track is
# redone then as well; a cached one would carry the old scale's translations.
if [ -f "$ego_pkl" ] && [ -z "${FORCE_FP:-}" ] && [ -z "${EGO_OBJECT_SIZE:-}" ]; then
    log "5 foundationpose: already done ($ego_pkl); FORCE_FP=1 to redo"
else
    log "5 foundationpose on the ego clip"
    python prep/fp_hy3d_track.py --viz_path x --wild_video --kid 0 \
        --masks_root "$EGO_RECT_DIR" --hy3d_root="$EGO_MESH_DIR-metric" \
        --video "$RECT_CLIP" -o "$EGO_FP_DIR" --zfar "$EGO_ZFAR" -tstart 0 \
        --erode_depth_thres "$EGO_ERODE" --depth_scale "$DEPTH_SCALE" \
        --depth_human_band "$EGO_BAND" --depth_mad_k "$DEPTH_MAD_K" \
        ${EGO_RGB_ONLY:+--rgb_only}
fi
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
# The directory is cleared first: a mesh left there by an earlier run under
# another name sorts ahead of this one, and every reader takes the first
# match -- which is how cam01 rendered a mesh a thirtieth of the pot's size
# while the ego track itself was right.
rm -rf "$MESH_DIR-metric"
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
pipe_metric_obj=$(ls "$MESH_DIR-metric"/*/*_align.obj 2>/dev/null | head -1)
log "metric mesh for the pipeline: $pipe_metric_obj"

# --- 7: the pipeline camera's object masks, from the ego track ------------------
# Until here cam01's own SAM3 object mask still spoke for the object in three
# places: the depth injection writes into it, CoCoNet reads it, the optimizer
# pulls the mesh toward it. On a mask that sits on the wrong thing, all three
# drag a correct ego pose onto the wrong thing. So it is replaced by the
# silhouette of the tracked mesh in this camera. The exo copy is kept as
# <name>.exo.h5. The person masks are untouched.
: "${RECT_DIR:?set RECT_DIR}"
rect_masks="$RECT_DIR/${SEQ}_masks_k0.h5"
rect_pkl="$RECT_DIR/$SEQ.0.color.pkl"
for required in "$rect_masks" "$rect_pkl"; do
    [ -e "$required" ] || { echo "ERROR: stage 2 output missing: $required" >&2; exit 1; }
done
log "7 object masks in $PIPE_CAM from the ego track"
python prep/render_object_masks.py --fp_pkl "$FP_DIR/${SEQ}_all.pkl" --mesh "$pipe_metric_obj" \
    --camera_pkl "$rect_pkl" --masks_h5 "$rect_masks"

log "done. outputs:"
ls -la "$EGO_RECT_DIR" "$EGO_FP_DIR" "$FP_DIR" "$(dirname "$OBJECT_XYZ_EGO")"
