#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --job-name="recon-geom"
#SBATCH --output=recon-geom-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# The geometry step: everything that turns masks and keypoints into metric 3D,
# plus the rectification the monocular pipeline needs. CPU only -- triangulation
# is numpy and rectification is a cv2 remap.
#
# Four things, in this order because each needs the last:
#
#   1  triangulate_object.py   ball position per frame, from >=2 fisheye views
#   2  inspect_object_xyz.py   the coverage/residual report you actually read
#   3  triangulate_human.py    17 COCO joints per frame, same views, same model
#   4  rectify_fisheye.py      pipeline clip + masks -> a true pinhole camera
#   5  make_ball_mesh.py       the object template (a ball's shape is known;
#                              Hunyuan3D on a 13px ball only adds error)
#
# Triangulation runs on the ORIGINAL fisheye pixels and rectification happens
# after it, deliberately: the Kannala-Brandt model is applied once, and feeding
# rectified pixels to triangulate_* would undistort twice.
#
# Driven by scripts/recon_geometry.sh, which exports every variable below.
# Standalone use means exporting them yourself.

set -euo pipefail

REPO=/simurgh2/projects/ret-hoi/CARI4D
CACHE_ROOT=/simurgh2/projects/ret-hoi

log() { echo "[geom $(date -u +%H:%M:%S)] $*"; }

export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg_cache}"
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CARI4D_ENV:-newcari4d}"

cd "$REPO"

: "${SEQ:?set SEQ}"
: "${CALIB:?set CALIB}"
: "${MASKS_DIR:?set MASKS_DIR}"
: "${CLIPS_DIR:?set CLIPS_DIR}"
: "${PIPE_CLIP:?set PIPE_CLIP}"
: "${PACKED_MV:?set PACKED_MV}"
: "${GEOM_DIR:?set GEOM_DIR}"
: "${OBJECT_XYZ:?set OBJECT_XYZ}"
: "${HUMAN_J3D:?set HUMAN_J3D}"
: "${RECT_DIR:?set RECT_DIR}"
: "${MESH_DIR:?set MESH_DIR}"
PIPE_CAM="${PIPE_CAM:-cam04}"
AUX_CAMS="${AUX_CAMS:-cam01 cam03}"

log "host=$(hostname) job=${SLURM_JOB_ID:-none} env=${CONDA_DEFAULT_ENV:-none}"
log "code=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo +dirty)"
log "seq=$SEQ  pipe_cam=$PIPE_CAM  aux=$AUX_CAMS"

mkdir -p "$GEOM_DIR" "$RECT_DIR" "$MESH_DIR"

# --view arguments, built once. Object views name mask sets (<cam>:<root>:<seq>);
# human views name the clip the keypoints were computed on (<cam>:<clip>). The
# pipeline camera is first in both, so output frames use its numbering.
obj_views=(--view "$PIPE_CAM:$MASKS_DIR:$SEQ")
hum_views=(--view "$PIPE_CAM:$PIPE_CLIP")
for c in $AUX_CAMS; do
    obj_views+=(--view "$c:$MASKS_DIR:$c-4k")
    hum_views+=(--view "$c:$CLIPS_DIR/$c-4k.0.color.mp4")
done

# --- 1: the ball, in metres -------------------------------------------------
log "triangulating the object"
python prep/triangulate_object.py --calib "$CALIB" "${obj_views[@]}" \
    --width "${TRI_WIDTH:-796}" --height "${TRI_HEIGHT:-448}" \
    ${TRI_MAX_RESIDUAL:+--max_residual "$TRI_MAX_RESIDUAL"} \
    --out "$OBJECT_XYZ"

# --- 2: the report. Printed here so it is in this job's log, next to the run
#        that produced it, rather than needing a separate invocation later. ---
log "triangulation coverage"
python3 prep/inspect_object_xyz.py "$OBJECT_XYZ"

# --- 3: the human anchor. Without it the body slides along the viewing ray by
#        tens of centimetres and no image-space loss can see it. --------------
log "triangulating the human"
python prep/triangulate_human.py --calib "$CALIB" --packed_root "$PACKED_MV" \
    "${hum_views[@]}" --to_cam "$PIPE_CAM" --out "$HUMAN_J3D"

# --- 4: fisheye -> pinhole, for the clip and its masks ----------------------
log "rectifying the pipeline clip"
python prep/rectify_fisheye.py --video "$PIPE_CLIP" --calib "$CALIB" \
    --cam "$PIPE_CAM" --masks_root "$MASKS_DIR" --out_dir "$RECT_DIR"

# --- 5: the object template -------------------------------------------------
log "writing the ball template"
python scripts/make_ball_mesh.py --seq "$SEQ" --hy3d_root "$MESH_DIR" \
    ${BALL:+--ball "$BALL"}

log "done. outputs:"
ls -la "$GEOM_DIR" "$RECT_DIR" "$MESH_DIR"
