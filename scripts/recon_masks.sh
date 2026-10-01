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

# --- B'/C': the same two steps for the ego view ------------------------------
# Its own trim job because the suffix differs (-ego, not -4k: the ego clip is
# 1408x1408, not 4K, and the name is what tells the readers which model to
# unproject with). SAM3's person prompt is "hands" here -- the wearer is
# otherwise invisible to their own glasses -- and the chunk is smaller than the
# exo default because the frames are larger. The ego masks are OPTIONAL
# downstream: geometry adds the ray when the file exists and carries on
# without it, so this job failing costs the ego view, not the clip.
if recon_ego_ready; then
    export SRC_DIR="$FAV_DIR" OUT_DIR="$CLIPS_DIR" SUFFIX="-ego" CAMS="$EGO_CAM" \
           WINDOW_JSON="$WORK/window.json"
    unset NO_TRIM CHUNK HUMAN OBJECT
    job_be=$(recon_sbatch --job-name="m1-ego-$SEQ" scripts/slurm_trim_clips.sh)
    log "B' trim ego view to this clip        job $job_be"
    export OUT_DIR="$MASKS_DIR" NO_TRIM=1 CHUNK="${EGO_CHUNK:-100}"
    export HUMAN="$EGO_HUMAN_PROMPT" OBJECT="$OBJECT_PROMPT" VIDEO="$EGO_CLIP"
    unset WINDOW_JSON
    job_ce=$(recon_sbatch $(recon_dep "$job_be") \
        --time="${SAM3_EGO_TIME:-06:00:00}" \
        --job-name="m2-ego-$SEQ" scripts/slurm_sam3_masks.sh)
    log "C' sam3 ego $EGO_CAM (no trim)   job $job_ce"
else
    log "no ego view for this take (EGO=$EGO, cam='${EGO_CAM:-none}')"
fi

recon_check \
    ${EGO_SEQ:+"# the ego masks: $MASKS_DIR/${EGO_SEQ}_sam3_vis.mp4 -- is the object the right one?"} \
    "python prep/check_view_coverage.py --masks_root $MASKS_DIR --seq $SEQ" \
    "# if it reports a shorter usable run than the clip, cut the clip down:" \
    "#   python prep/retrim_clip.py --work $WORK --lo <first> --hi <last>" \
    "#   then use the NEW name (${SEQ}t) in every command after that"
recon_next \
    "TAKE=$TAKE SEQ=$SEQ bash scripts/recon_geometry.sh" \
    "" \
    "and this one too -- they are independent, run both:" \
    "" \
    "   python prep/pick_object_frame.py --work $WORK" \
    "" \
    "That one writes a sheet of candidate crops and prints the exact" \
    "recon_object.sh command for the tile you choose."
