#!/bin/bash
# One command, one take: run every stage up to the checkpoint that matters,
# waiting for the cluster in between, then say what to look at.
#
#   TAKE=iiith_cooking_57_2 SEQ=Date03_Sub06_mpot_pour \
#   HUMAN_PROMPT=person OBJECT_PROMPT=pot \
#       nohup bash scripts/recon_pilot.sh > pilot-mpot.log 2>&1 &
#
#   tail -f pilot-mpot.log        # or: cat work/<seq>/NEXT.txt when it is done
#
# What it runs, in order, and why it waits where it waits:
#
#   0  recon_check.sh      the inputs exist (seconds, no job)
#   1a recon_clips.sh      SAM3 over the whole take on the pipeline camera
#      -- WAIT: which clips came out is not known until this finishes --
#   1b recon_masks.sh      per clip: aux views at 4K, and the EGO view, masked
#      -- WAIT: geometry and the mesh both read these masks --
#   2  recon_geometry.sh   triangulate (ego ray included), rectify
#      recon_object.sh     the mesh, from the best (view, frame) the picker finds
#      -- WAIT, then STOP: the next stage is hours, this is the cheap look --
#
# It stops before stage 3 on purpose. The three-stage split exists so a bad mask
# is seen before the GPU hours are spent; this script keeps that one checkpoint
# and removes the babysitting between the others. THROUGH=solve runs stage 3
# too, unattended, when you already trust this take.
#
# CLIPS=a (default) takes only the longest clip stage 1a emitted; CLIPS=all
# takes every one. Re-running is safe: a stage whose main output already exists
# is skipped, so a pilot that died waiting picks up where it was.
#
# Everything recon_common.sh reads applies here -- PIPE_CAM, EGO=0, MESH_CAM,
# MESH_FRAME, MESH_BACKEND, EXCLUDE_NODES, DRY_RUN=1 for the plan.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/recon_common.sh

recon_require_env
recon_paths

CLIPS="${CLIPS:-a}"
THROUGH="${THROUGH:-object}"
case "$THROUGH" in object|solve) ;; *) echo "ERROR: THROUGH must be object or solve" >&2; exit 1 ;; esac

BASE_SEQ="$SEQ"
BASE_WORK="$WORK"
PILOT_LOG="$BASE_WORK/pilot.log"
[ -n "$DRY_RUN" ] || mkdir -p "$BASE_WORK"

log() {
    # Timestamped, to stderr and to the pilot log: this runs for hours under
    # nohup, and "what was it doing at 03:40" is the question that gets asked.
    local line="[pilot $(date -u '+%m-%d %H:%M')] $*"
    echo "$line" >&2
    [ -n "$DRY_RUN" ] || echo "$line" >> "$PILOT_LOG"
}

run_stage() {
    # Run one driver stage, show its output, and set `ids` to the job ids it
    # submitted (appending to whatever is there).
    #
    # The stages log "<what>   job <id>" on stderr; those ids are what the
    # wait needs. Output is captured and re-echoed rather than piped so the
    # stage's exit status survives -- a stage that refuses to run (missing
    # input, basketball prompts on a kitchen) stops the pilot right here.
    local out
    out=$(bash "scripts/recon_$1.sh" 2>&1) || { echo "$out" >&2; return 1; }
    echo "$out" >&2
    [ -n "$DRY_RUN" ] || echo "$out" >> "$PILOT_LOG"
    local id
    while read -r id; do ids+=("$id"); done < <(echo "$out" | grep -oE ' job [A-Z0-9]+$' | awk '{print $2}')
}

# --- 0: inputs -----------------------------------------------------------------
verdict=$(bash scripts/recon_check.sh)
log "$verdict"
case "$verdict" in MISSING*) echo "ERROR: fix the inputs above first" >&2; exit 1 ;; esac
if recon_ego_ready; then
    log "ego view: $EGO_CAM (masked and triangulated alongside the exo views)"
else
    log "ego view: none for this take"
fi
log "prompts: human='$HUMAN_PROMPT' object='$OBJECT_PROMPT'  pipeline cam=$PIPE_CAM"

# --- 1a: cut the take into clips --------------------------------------------
if [ -f "$CLIPS_JSON" ] && [ -z "${FORCE:-}" ]; then
    log "1a clips: already done ($CLIPS_JSON exists)"
else
    log "1a clips: submitting"
    ids=()
    run_stage clips
    recon_wait "${ids[@]}"
fi

# Which clips to carry on with. Under DRY_RUN nothing was written, so the plan
# is shown for the clip stage 1a would most likely emit.
all_clips=()
if [ -n "$DRY_RUN" ] && [ ! -f "$CLIPS_JSON" ]; then
    clips=("${BASE_SEQ}a")
else
    mapfile -t all_clips < <(python3 -c "import json,sys
print('\n'.join(c['seq'] for c in json.load(open(sys.argv[1]))['clips']))" "$CLIPS_JSON")
    [ ${#all_clips[@]} -gt 0 ] || {
        echo "ERROR: stage 1a emitted no clip: SAM3 never held both masks for $MIN_FRAMES frames." >&2
        echo "       Watch $MASKS_DIR/${BASE_SEQ}_sam3_vis.mp4 and retry with other prompts." >&2
        exit 1; }
    if [ "$CLIPS" = all ]; then
        clips=("${all_clips[@]}")
    else
        clips=("${all_clips[0]}")
    fi
fi
log "clips: ${clips[*]}  (of ${#all_clips[@]} emitted; CLIPS=$CLIPS)"

# --- 1b: aux + ego masks, every chosen clip at once ---------------------------
ids=()
for clip in "${clips[@]}"; do
    export SEQ="$clip"; recon_paths
    done_masks=1
    for c in $AUX_CAMS; do [ -f "$MASKS_DIR/$c-4k_masks_k0.h5" ] || done_masks=""; done
    if recon_ego_ready 2>/dev/null && [ ! -f "$EGO_MASKS" ]; then done_masks=""; fi
    if [ -n "$done_masks" ] && [ -z "${FORCE:-}" ]; then
        log "1b masks $clip: already done"
        continue
    fi
    log "1b masks $clip: submitting"
    run_stage masks
done
recon_wait "${ids[@]}"

# --- 2: geometry, and the mesh from the frame the picker likes best -----------
ids=()
for clip in "${clips[@]}"; do
    export SEQ="$clip"; recon_paths
    if [ -f "$OBJECT_XYZ" ] && [ -z "${FORCE:-}" ]; then
        log "2  geometry $clip: already done"
    else
        log "2  geometry $clip: submitting"
        run_stage geometry
    fi
    if compgen -G "$MESH_DIR/*/*_align.obj" >/dev/null && [ -z "${FORCE:-}" ]; then
        log "2  object $clip: already done"
        continue
    fi
    # The picker scores every aux and ego frame and writes the contact sheet;
    # its last line is the exact MESH_CAM/MESH_FRAME it recommends. A typed
    # MESH_CAM or MESH_FRAME still wins. If the picker cannot run here (no
    # cv2 in this shell), the stage falls back to its own midpoint default.
    if [ -z "${MESH_CAM:-}" ] && [ -z "${MESH_FRAME:-}" ] && [ -z "$DRY_RUN" ]; then
        pick=$(python prep/pick_object_frame.py --work "$WORK" 2>/dev/null \
               | grep -oE 'MESH_CAM=\S+ MESH_FRAME=[0-9]+' | tail -1 || true)
        if [ -n "$pick" ]; then
            log "2  object $clip: picker says $pick  (sheet: $WORK/object_candidates.png)"
            export MESH_CAM="${pick#MESH_CAM=}"; MESH_CAM="${MESH_CAM%% *}"
            export MESH_FRAME="${pick##*MESH_FRAME=}"
        else
            log "2  object $clip: picker unavailable, using the clip midpoint"
        fi
    fi
    log "2  object $clip: submitting"
    run_stage object
    unset MESH_CAM MESH_FRAME
done
recon_wait "${ids[@]}"

# --- 3: only on request ---------------------------------------------------------
if [ "$THROUGH" = solve ]; then
    ids=()
    for clip in "${clips[@]}"; do
        export SEQ="$clip"; recon_paths
        log "3  solve $clip: submitting"
        run_stage solve
    done
    recon_wait "${ids[@]}"
fi

# --- the look -------------------------------------------------------------------
export SEQ="$BASE_SEQ"; recon_paths
lines=()
for clip in "${clips[@]}"; do
    cw="$WORK_ROOT/$clip"
    lines+=("# $clip")
    lines+=("python prep/inspect_object_xyz.py $cw/geom/object_xyz.npz")
    lines+=("#   coverage and residuals; 'aria' in the per-view lines means the ego ray was used")
    lines+=("ls $cw/meshes/*/")
    lines+=("#   the _rgba.png is the crop, the _align.obj the mesh; $cw/object_candidates.png is the sheet")
    [ -n "$EGO_SEQ" ] && lines+=("#   ego masks: $cw/masks/${EGO_SEQ}_sam3_vis.mp4")
    lines+=("")
done
recon_check "${lines[@]}"
if [ "$THROUGH" = solve ]; then
    recon_next "cat $WORK_ROOT/${clips[0]}/NEXT.txt     # stage 3 wrote where the result landed"
else
    nxt=()
    for clip in "${clips[@]}"; do nxt+=("TAKE=$TAKE SEQ=$clip bash scripts/recon_solve.sh"); done
    recon_next "${nxt[@]}"
fi
log "done"
