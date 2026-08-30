#!/bin/bash
# Stage 1b: mask the aux views of ONE clip.
#
# Stage 1a (scripts/recon_clips.sh) already masked the pipeline camera over the
# whole take and cut it into clip directories. This takes one of those clips and
# gives it the extra calibrated views triangulation needs:
#
#   B  cut every aux camera to exactly this clip's frames, at 4K. Frame
#      accuracy is the point -- a +-1 slip between views corrupts the geometry
#      silently, so the trim counts frames and fails the job on a mismatch.
#   C  SAM3 on each aux clip with --no_trim. It is already trimmed; letting each
#      view pick its own best run would de-synchronise them.
#
# Every exo camera except the pipeline one is masked, not a chosen pair.
# Masking is the expensive one-time step, while deciding which views to believe
# is arithmetic over centroids that already exist -- triangulate_object.py picks
# the agreeing subset per frame, so a view that is only sometimes good still
# contributes on the frames where it is good.
#
#   TAKE=unc_basketball_03-31-23_02_3 SEQ=Date03_Sub01_bball_rev003a \
#       bash scripts/recon_masks.sh
#
# SEQ is a CLIP name -- the trailing letter matters, and `ls work/` after stage
# 1a lists them. DRY_RUN=1 prints without submitting.
#
# STOP HERE AND LOOK, per view:
#     $WORK/masks/<view>_sam3_vis.mp4     is the mask on the right ball?
#     the log's "Object masks found in N/M frames" -- the number that decides
#     how many frames can be triangulated, since the ball needs two views

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/recon_common.sh

recon_require_env
recon_paths

log() { echo "[recon-masks] $*" >&2; }

# This clip has to exist, which means stage 1a has to have run. The error names
# the missing file rather than the missing stage, because the usual cause is a
# letter that was never emitted -- asking for `c` when only a and b came out.
if [ -z "$DRY_RUN" ]; then
    for required in "$WINDOW_JSON" "$PIPE_CLIP"; do
        [ -e "$required" ] || {
            echo "ERROR: no clip at $required" >&2
            echo "       Stage 1a (scripts/recon_clips.sh) writes it. Clips that exist:" >&2
            ls -d "${WORK_ROOT}/${SEQ%?}"* 2>/dev/null | sed 's/^/         /' >&2 || true
            exit 1
        }
    done
fi
for required in "$TAKE_DIR" "$CALIB"; do
    [ -e "$required" ] || { echo "ERROR: missing input: $required" >&2; exit 1; }
done
for c in $AUX_CAMS; do
    [ -f "$FAV_DIR/$c.mp4" ] || { echo "ERROR: no aux video $FAV_DIR/$c.mp4" >&2; exit 1; }
done

n_frames=$(recon_window_frames)
log "take=$TAKE  clip=$SEQ${n_frames:+  ($n_frames frames)}"
log "pipeline cam=$PIPE_CAM  aux=$AUX_CAMS"

recon_run mkdir -p "$CLIPS_DIR"

# --- B: cut the aux views to this clip's frames -----------------------------
# The window comes from the clip's own window.json, written by stage 1a, so the
# aux views land on exactly the frames the pipeline camera kept.
export SRC_DIR="$FAV_DIR" OUT_DIR="$CLIPS_DIR" SUFFIX="-4k" CAMS="$AUX_CAMS"
unset START END
job_b=$(recon_sbatch --job-name="m1-$SEQ" scripts/slurm_trim_clips.sh)
log "B  trim aux views to this clip        job $job_b"

# --- C: SAM3 on each aux clip, 4K, no further trimming ----------------------
# CHUNK drops from 300 to 60: these frames are 3840x2160 against the pipeline
# camera's 796x448, and the chunk is what has to fit in GPU memory.
export OUT_DIR="$MASKS_DIR" NO_TRIM=1 CHUNK="${AUX_CHUNK:-60}"
export HUMAN="$HUMAN_PROMPT" OBJECT="$OBJECT_PROMPT"
unset WINDOW_JSON EMIT_ROOT CLIPS_JSON  # --no_trim picks no window, emits no clips
for c in $AUX_CAMS; do
    export VIDEO="$(recon_aux_clip "$c")"
    job_c=$(recon_sbatch $(recon_dep "$job_b") \
        --time="${SAM3_AUX_TIME:-04:00:00}" \
        --job-name="m2-$c-$SEQ" scripts/slurm_sam3_masks.sh)
    log "C  sam3 aux $c (4K, no trim)      job $job_c"
done

log ""
log "when these finish, CHECK before stage 2:"
log "  grep -H 'Object masks found' sam3-masks-*.out"
log "  ls $MASKS_DIR/*_sam3_vis.mp4     # watch each one"
log "then: TAKE=$TAKE SEQ=$SEQ bash scripts/recon_geometry.sh"
