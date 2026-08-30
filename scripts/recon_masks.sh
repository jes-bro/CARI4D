#!/bin/bash
# Stage 1 of 3: masks, and the frame window they imply.
#
# Run on the login node -- it submits jobs, it does not compute. Three chained
# steps, each queued immediately with --dependency=afterok:
#
#   A  SAM3 over the WHOLE take on the pipeline camera at 448, which trims
#      itself to the longest run of frames where the person AND the ball are
#      both masked. That run IS the clip: no attempt-by-attempt segmentation,
#      no hand-picked frame numbers.
#   B  cut the aux views to exactly those frames, at 4K. Frame-accurate,
#      because a +-1 slip between views corrupts triangulation silently.
#   C  SAM3 on each aux clip with --no_trim (already trimmed; trimming each
#      view to its own best run would de-synchronise them).
#
# STOP HERE AND LOOK. This is the first of the two checkpoints. For each view:
#     $WORK/masks/<seq>_sam3_vis.mp4      is the mask on the right thing?
#     the job log's "Both masks present in N/M frames"
#     $WORK/window.json                   how long a clip survived
# The failure this catches is the expensive one: masks that track a bystander,
# the wrong ball, or nothing, which every later stage will happily consume.
#
#   TAKE=unc_basketball_03-31-23_02_3 SEQ=Date03_Sub01_bball_rev003 \
#       bash scripts/recon_masks.sh
#
#   DRY_RUN=1 TAKE=... SEQ=... bash scripts/recon_masks.sh    # print, submit nothing
#
# Knobs come from scripts/recon_common.sh: PIPE_CAM, AUX_CAMS, HUMAN_PROMPT,
# OBJECT_PROMPT, MIN_FRAMES, WORK_ROOT.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/recon_common.sh

recon_require_env
recon_paths

log() { echo "[recon-masks] $*" >&2; }

# Inputs are read-only shared storage; a missing one should say so now, not
# inside a queued job an hour from now.
for required in "$TAKE_DIR" "$SRC_448" "$CALIB"; do
    [ -e "$required" ] || { echo "ERROR: missing input: $required" >&2; exit 1; }
done
for c in $AUX_CAMS; do
    [ -f "$FAV_DIR/$c.mp4" ] || { echo "ERROR: no aux video $FAV_DIR/$c.mp4" >&2; exit 1; }
done

log "take=$TAKE  seq=$SEQ"
log "work=$WORK"
log "pipeline cam=$PIPE_CAM  aux=$AUX_CAMS"
log "prompts: human='$HUMAN_PROMPT' object='$OBJECT_PROMPT'"

recon_run mkdir -p "$WORK/src" "$MASKS_DIR" "$CLIPS_DIR"

# The symlink is what gives the sequence its name: run_sam3_masks.py reads the
# sequence out of the filename, so masking the take's own cam04.mp4 would name
# every take's masks "cam04". Symlink, not copy -- these are 4K-source files.
recon_run ln -sf "$SRC_448" "$PIPE_SRC"

# --- A: full-take SAM3 on the pipeline camera, which chooses the window ------
# The take is ~2000 frames rather than the ~100 a trimmed clip has, so this
# wants more wall time than the script's 2h default.
export VIDEO="$PIPE_SRC" OUT_DIR="$MASKS_DIR" WINDOW_JSON="$WINDOW_JSON"
export HUMAN="$HUMAN_PROMPT" OBJECT="$OBJECT_PROMPT"
unset NO_TRIM CHUNK
job_a=$(recon_sbatch --time="${SAM3_FULL_TIME:-06:00:00}" \
    --job-name="m1-$SEQ" scripts/slurm_sam3_masks.sh)
log "A  sam3 full take (pipeline cam)      job $job_a"

# --- B: cut the aux views to the same frames --------------------------------
# WINDOW_JSON instead of START/END because the window does not exist yet at
# submission time. MIN_FRAMES makes this the fail-fast for a take whose masks
# never held: the aux SAM3 jobs below are the expensive ones.
export SRC_DIR="$FAV_DIR" OUT_DIR="$CLIPS_DIR" SUFFIX="-4k" CAMS="$AUX_CAMS"
unset START END
job_b=$(recon_sbatch $(recon_dep "$job_a") \
    --job-name="m2-$SEQ" scripts/slurm_trim_clips.sh)
log "B  trim aux views to that window      job $job_b"

# --- C: SAM3 on each aux clip, 4K, no further trimming ----------------------
# CHUNK drops from 300 to 60: these frames are 3840x2160 against the pipeline
# camera's 796x448, and the chunk is what has to fit in GPU memory.
export OUT_DIR="$MASKS_DIR" NO_TRIM=1 CHUNK="${AUX_CHUNK:-60}"
window_json_path="$WINDOW_JSON"
unset WINDOW_JSON   # --no_trim chooses no window, so there is none to record
for c in $AUX_CAMS; do
    export VIDEO="$(recon_aux_clip "$c")"
    job_c=$(recon_sbatch $(recon_dep "$job_b") \
        --time="${SAM3_AUX_TIME:-04:00:00}" \
        --job-name="m3-$c-$SEQ" scripts/slurm_sam3_masks.sh)
    log "C  sam3 aux $c (4K, no trim)      job $job_c"
done

log ""
log "when these finish, CHECK before stage 2:"
log "  cat $window_json_path"
log "  ls $MASKS_DIR/*_sam3_vis.mp4     # watch each one"
log "  grep -h 'Both masks present' sam3-masks-*.out"
log "then: TAKE=$TAKE SEQ=$SEQ bash scripts/recon_geometry.sh"
