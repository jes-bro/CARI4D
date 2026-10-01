#!/bin/bash
# One command, one take: run every stage up to the checkpoint that matters,
# waiting for the cluster in between, then say what to look at.
#
#   TAKE=iiith_cooking_57_2 SEQ=Date03_Sub06_mpot_pour CLIP_LO=5190 CLIP_HI=5550 \
#   HUMAN_PROMPT=person OBJECT_PROMPT="metal saucepan with handle" \
#       nohup bash scripts/recon_pilot.sh > pilot-mpot.log 2>&1 &
#
#   tail -f pilot-mpot.log        # or: cat work/<seq>/NEXT.txt when it is done
#
# Or, inside an interactive GPU allocation while a new kind of take is still
# being debugged, with every job running in this terminal instead of the queue
# (scripts/recon_local.sh):
#
#   RECON_BACKEND=local TAKE=... SEQ=... bash scripts/recon_pilot.sh 2>&1 | tee pilot.log
#
# What it runs, in order, and why it waits where it waits:
#
#   0  recon_check.sh      the inputs exist (seconds, no job)
#   1a the window          pipeline camera cut to CLIP_LO-CLIP_HI and masked
#      -- WAIT: everything after reads those masks --
#   1b recon_masks.sh      aux views at 4K, and the EGO view, masked
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
# CLIP_LO / CLIP_HI, in TAKE frames, are REQUIRED: the window to reconstruct.
# The person running this has watched the take and knows where the action is,
# so the pipeline camera is cut to exactly those frames and masked there, the
# clip is SEQ itself (no letter), and nothing is computed on the rest of the
# take. Letting SAM3 find a window instead (scripts/recon_clips.sh) is for a
# batch nobody has watched; on a kitchen take where person and pot are in
# frame throughout it finds the whole take.
#
# Re-running is safe: a stage whose main output already exists is skipped, so
# a pilot that died waiting picks up where it was.
#
# OBJECT_FROM=ego tracks the object in the ego view only (scripts/
# slurm_ego_object.sh, run by stage 3) and carries the poses into the pipeline
# camera; the exo cameras then contribute nothing to the object. The default,
# tri, is the triangulate-and-inject path.
#
# MESH_FROM=<mesh root> drops an already reconstructed object in instead of
# running the object stage: the <seq>_<frame>_rgba/ directory there is copied
# into the clip's meshes/ and renamed for the clip. The frame in its name must
# be a frame of the clip, since the scale step reads depth there.
#
# Everything recon_common.sh reads applies here -- PIPE_CAM, EGO=0, MESH_CAM,
# MESH_FRAME, MESH_BACKEND, MESH_EXTRA, EXCLUDE_NODES, DRY_RUN=1 for the plan.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/recon_common.sh

recon_require_env
recon_paths

: "${CLIP_LO:?set CLIP_LO, the first take frame of the window}"
: "${CLIP_HI:?set CLIP_HI, the last take frame of the window (inclusive)}"
[ "$CLIP_HI" -gt "$CLIP_LO" ] || { echo "ERROR: CLIP_HI must exceed CLIP_LO" >&2; exit 1; }
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
    # wait needs. Output streams through tee as it happens -- under the local
    # backend the jobs themselves run inside the stage, and watching them is
    # the point -- and is parsed from the copy afterwards. The stage's exit
    # status is kept through the pipe, so a stage that refuses to run
    # (missing input, basketball prompts on a kitchen) stops the pilot here.
    local tmp rc
    tmp=$(mktemp)
    bash "scripts/recon_$1.sh" 2>&1 | tee "$tmp" >&2
    rc=${PIPESTATUS[0]}
    [ -n "$DRY_RUN" ] || cat "$tmp" >> "$PILOT_LOG"
    local id
    while read -r id; do ids+=("$id"); done < <(grep -oE ' job [A-Z0-9]+$' "$tmp" | awk '{print $2}')
    rm -f "$tmp"
    return "$rc"
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

# --- 1a: the window. The pipeline camera cut to it and masked. --------------
clips=("$BASE_SEQ")
n_win=$((CLIP_HI - CLIP_LO + 1))
if [ -f "$WINDOW_JSON" ] && [ -f "$MASKS_DIR/${SEQ}_masks_k0.h5" ] && [ -z "${FORCE:-}" ]; then
    log "1a window: already done ($WINDOW_JSON and the $PIPE_CAM masks exist)"
else
    log "1a window: $PIPE_CAM (448) cut to take frames $CLIP_LO-$CLIP_HI ($n_win frames) as $SEQ"
    recon_run mkdir -p "$CLIPS_DIR"
    # The window file every later stage reads: the aux and ego trims cut to
    # it, geometry takes the ego offset from it, the object stage defaults its
    # frame from it. Written here, not by SAM3, because with --no_trim SAM3
    # chooses nothing.
    if [ -z "$DRY_RUN" ]; then
        python3 -c "import json,sys; json.dump({'seq': sys.argv[1], 'source': 'recon_pilot CLIP_LO/CLIP_HI', 'num_frames': int(sys.argv[4]), 'runs': [], 'chosen': {'lo': int(sys.argv[2]), 'hi': int(sys.argv[3]), 'n_frames': int(sys.argv[4]), 'covered_frac': 1.0}}, open(sys.argv[5], 'w'), indent=1)" "$SEQ" "$CLIP_LO" "$CLIP_HI" "$n_win" "$WINDOW_JSON"
    fi
    ids=()
    export SRC_DIR="$FAV_DIR/downscaled/448" OUT_DIR="$CLIPS_DIR" SUFFIX="" CAMS="$PIPE_CAM" \
           OUT_NAME="$SEQ" START="$CLIP_LO" END="$CLIP_HI"
    job=$(recon_sbatch --job-name="c0-$SEQ" scripts/slurm_trim_clips.sh)
    ids+=("$job"); log "1a window: trim $PIPE_CAM                job $job"
    unset OUT_NAME SUFFIX START END
    export VIDEO="$PIPE_CLIP" OUT_DIR="$MASKS_DIR" NO_TRIM=1
    export HUMAN="$HUMAN_PROMPT" OBJECT="$OBJECT_PROMPT"
    unset CHUNK WINDOW_JSON EMIT_ROOT CLIPS_JSON
    job=$(recon_sbatch $(recon_dep "${ids[@]}") --job-name="c1-$SEQ" scripts/slurm_sam3_masks.sh)
    ids+=("$job"); log "1a window: sam3 $PIPE_CAM (no trim)      job $job"
    unset NO_TRIM VIDEO HUMAN OBJECT
    recon_paths   # restore WINDOW_JSON and friends for the stages below
    recon_wait "${ids[@]}"
fi

place_mesh() {
    # Copy the reconstructed object under $MESH_FROM into this clip's meshes/,
    # renamed for the clip. The tracker globs <seq>*/*<seq>*_align.obj, so
    # both the directory and the OBJ carry the clip's name; the .mtl and the
    # texture keep theirs, which is what the OBJ's mtllib line refers to.
    local src dst frame
    for src in "$MESH_FROM"/*_rgba; do
        [ -d "$src" ] || continue
        frame=$(basename "$src" | sed -E 's/.*_([0-9]{3})_rgba$/\1/')
        dst="$MESH_DIR/${SEQ}_${frame}_rgba"
        if [ -n "$DRY_RUN" ]; then echo "  would copy $src -> $dst" >&2; continue; fi
        mkdir -p "$MESH_DIR" && cp -r "$src" "$dst"
        for f in "$dst"/*_align.obj; do
            [ -f "$f" ] || continue
            [ "$(basename "$f")" = "${SEQ}_${frame}_align.obj" ] || mv "$f" "$dst/${SEQ}_${frame}_align.obj"
        done
        log "placed mesh from $src as $dst"
    done
}

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
    if [ -n "${MESH_FROM:-}" ]; then
        log "2  object $clip: taking the mesh from $MESH_FROM"
        place_mesh
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
