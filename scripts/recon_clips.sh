#!/bin/bash
# Stage 1a: cut one take into clips.
#
# Runs SAM3 over the WHOLE take on the pipeline camera and writes every stretch
# where the person AND the ball are both tracked, long enough to reconstruct, as
# its own clip directory. One 68s layup take is several attempts; the masks for
# all of them are computed in this pass either way, so keeping only the longest
# throws away work already done.
#
#   TAKE=unc_basketball_03-31-23_02_3 SEQ=Date03_Sub01_bball_rev003 \
#       bash scripts/recon_clips.sh
#
# SEQ here is the TAKE's base name. The clips get a letter, longest first:
#
#   work/Date03_Sub01_bball_rev003/      this pass: source symlink, clips.json,
#                                        the full-take mask video
#   work/Date03_Sub01_bball_rev003a/     a clip -- and a sequence in its own right
#   work/Date03_Sub01_bball_rev003b/
#   work/Date03_Sub01_bball_rev003c/
#
# A clip directory is what every later stage takes, so from here on the unit of
# work is a folder name and `ls work/` is the worklist.
#
# STOP HERE AND LOOK, before spending aux GPU on clips that may not be layups:
#     work/<seq>/clips.json                     what came out, and how long
#     work/<seq>/masks/<seq>_sam3_vis.mp4       the whole take, masks overlaid
# Then run stage 1b (scripts/recon_masks.sh) per clip you want to keep.
#
# EMIT_MIN_FRAMES (default 60) is the shortest run worth a reconstruction;
# EMIT_MAX_CLIPS (default 4) caps a badly tracked take. TRIM_GAP bridges short
# mask dropouts, merging runs that a single lost frame split apart.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/recon_common.sh

recon_require_env
recon_paths

log() { echo "[recon-clips] $*" >&2; }

for required in "$TAKE_DIR" "$SRC_448" "$CALIB"; do
    [ -e "$required" ] || { echo "ERROR: missing input: $required" >&2; exit 1; }
done

log "take=$TAKE  base seq=$SEQ"
log "pipeline cam=$PIPE_CAM"
log "prompts: human='$HUMAN_PROMPT' object='$OBJECT_PROMPT'"
log "emit: min_frames=${EMIT_MIN_FRAMES:-60} max_clips=${EMIT_MAX_CLIPS:-4} gap=${TRIM_GAP:-0}"

recon_run mkdir -p "$WORK/src" "$MASKS_DIR"

# The symlink is what names the sequence: run_sam3_masks.py reads it out of the
# filename, so masking the take's own cam04.mp4 would name every take "cam04".
recon_run ln -sf "$SRC_448" "$PIPE_SRC"

# EMIT_ROOT is WORK_ROOT, so the clips land as SIBLINGS of this take's
# directory rather than inside it -- a clip directory has to sit where
# recon_paths expects a sequence to live, which is $WORK_ROOT/$SEQ.
export VIDEO="$PIPE_SRC" OUT_DIR="$MASKS_DIR" WINDOW_JSON="$WINDOW_JSON"
export HUMAN="$HUMAN_PROMPT" OBJECT="$OBJECT_PROMPT"
export EMIT_ROOT="$WORK_ROOT" CLIPS_JSON="$CLIPS_JSON"
unset NO_TRIM CHUNK

job=$(recon_sbatch --time="${SAM3_FULL_TIME:-01:00:00}" \
    --job-name="c1-$SEQ" scripts/slurm_sam3_masks.sh)
log "sam3 over the whole take            job $job"

log ""
log "when it finishes, CHECK before stage 1b:"
log "  cat $CLIPS_JSON"
log "  ls $MASKS_DIR/${SEQ}_sam3_vis.mp4     # watch it"
log "then, per clip you want:"
log "  TAKE=$TAKE SEQ=${SEQ}a bash scripts/recon_masks.sh"
