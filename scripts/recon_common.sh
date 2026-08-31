#!/bin/bash
# Shared layout and helpers for the three recon_* driver stages.
#
# Sourced, never run. Everything here is either a path convention or a small
# function; nothing is submitted from this file.
#
# WHY A PER-SEQUENCE DIRECTORY. The pipeline's artifact names carry no sequence
# of their own past the pipeline camera: run_sam3_masks.py derives the sequence
# from the video filename, so the aux clips are `cam01-4k`, `cam03-4k` for
# EVERY take. One flat sam3masks/ was right for one reconstruction and silently
# destroys the previous one on the second. So each sequence owns a directory
# and the flat names live inside it.
#
# Layout under $WORK (= $WORK_ROOT/$SEQ):
#
#   src/$SEQ.0.color.mp4        symlink to the take's 448 pipeline-camera video,
#                               renamed so SAM3 names the sequence correctly
#   window.json                 the frame window SAM3's trim chose
#   masks/                      $SEQ_masks_k0.h5 and cam0N-4k_masks_k0.h5
#   masks/trimmed_vids/         $SEQ.0.color.mp4 (trimmed) + cam0N-4k.0.color.mp4
#   packed-mv/                  Sapiens keypoints per view, on FISHEYE clips
#   geom/                       object_xyz.npz, human_joints.npz
#   rect/                       rectified clip + true K .pkl + rectified masks
#                               + Sapiens keypoints re-detected on the rectified clip
#   rect-aligned/               depth-aligned clip (written by stage 3, sibling of rect/)
#   meshes/                     ball template, and -metric after scale estimation
#   nlf/, fp/                   pipeline intermediates
#
# SEQUENCE NAMING is not free-form. prep/rename_sequence.py documents why:
# stages parse `Date_Sub_object_action`, part[1] must be a key in
# behave_data.const._sub_gender (which also picks the SMPL body gender --
# Sub01-Sub05 male, Sub06-Sub08 female) and part[2] is the object name. So a
# layup by a female subject is e.g. Date04_Sub06_bball_rev017, and getting the
# Sub wrong silently fits the wrong-gender body model.

REPO="${REPO:-/simurgh2/projects/ret-hoi/CARI4D}"
TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
WORK_ROOT="${WORK_ROOT:-$REPO/work}"

# The camera the pipeline reconstructs from. It is a property of the RIG
# PLACEMENT, not of the task: the exo cameras are fixed within a capture session
# and re-placed between them, so this belongs in the manifest per capture, not
# here. The value below is only the fallback for a take run by hand.
#
# EgoExo4D's own `best_exo` field is a starting hypothesis, not the answer --
# it says cam01 for the UNC basketball capture, while the verified dribble
# reconstruction was built from cam04. It is labelling a different question.
# Captured before anything derives a default, so recon_paths() can tell an
# explicit choice from one it computed. A batch iterates over takes whose
# pipeline camera differs, so both have to be re-derived per sequence -- but a
# value the caller actually typed must survive that.
PIPE_CAM_EXPLICIT="${PIPE_CAM:-}"
AUX_CAMS_EXPLICIT="${AUX_CAMS:-}"
PIPE_CAM="${PIPE_CAM:-cam04}"

# Every exo camera in an EgoExo4D capture. AUX_CAMS derives to all of them
# except the pipeline camera: mask everything. Masking is the expensive,
# one-time GPU step, while choosing which views to believe is arithmetic over
# centroids that already exist -- and triangulate_object.py now does that per
# frame by consensus, so an aux view that is only sometimes good contributes
# on the frames where it is good instead of being excluded up front.
ALL_CAMS="${ALL_CAMS:-cam01 cam02 cam03 cam04}"

# SAM3 prompts. These are the basketball ones; a different object needs both
# overridden, and the roadmap's 0/507-mask incident was exactly this default
# being left in place for a kitchen sequence.
HUMAN_PROMPT="${HUMAN_PROMPT:-one basketball player playing basketball}"
OBJECT_PROMPT="${OBJECT_PROMPT:-ball}"

# Shortest window worth reconstructing, in frames. Enforced by the trim job so
# a take whose masks never hold stops the chain before the 4K SAM3 runs, and by
# stage 1a as the cutoff for which runs become clips at all.
MIN_FRAMES="${MIN_FRAMES:-60}"
EMIT_MIN_FRAMES="${EMIT_MIN_FRAMES:-$MIN_FRAMES}"
EMIT_MAX_CLIPS="${EMIT_MAX_CLIPS:-4}"
export EMIT_MIN_FRAMES EMIT_MAX_CLIPS

# DRY_RUN=1 prints every sbatch instead of submitting it. The whole chain is
# only checkable this way -- each stage depends on files the previous one has
# not written yet, so there is nothing else to test against on a login node.
DRY_RUN="${DRY_RUN:-}"

# Nodes to keep off. A GPU with failing memory reports "uncorrectable ECC
# error" and kills whatever lands on it, which looks exactly like a pipeline
# bug from the log -- three aux SAM3 jobs died this way, two of them within ten
# seconds. Set EXCLUDE_NODES once and every job in every stage carries it:
#
#   EXCLUDE_NODES=simurgh6 TAKE=... SEQ=... bash scripts/recon_masks.sh
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

recon_require_env() {
    # Fail early and by name when TAKE or SEQ is missing.
    #
    # Both are required by every stage and neither has a defensible default:
    # guessing the take would read someone else's footage, guessing the
    # sequence name would write over another reconstruction's directory.
    : "${TAKE:?set TAKE to the egoexo4d take name, e.g. unc_basketball_03-31-23_02_3}"
    : "${SEQ:?set SEQ to the sequence name, e.g. Date03_Sub01_bball_rev003}"
}

recon_paths() {
    # Export the per-sequence paths every stage refers to.
    #
    # Derived rather than passed so the three stages cannot disagree about
    # where an artifact lives -- the failure mode being a later stage quietly
    # reading nothing and reconstructing from defaults.
    # Derived here rather than once at source time, because a batch changes
    # PIPE_CAM between rows and the aux list has to follow it.
    if [ -n "$AUX_CAMS_EXPLICIT" ]; then
        AUX_CAMS="$AUX_CAMS_EXPLICIT"
    else
        AUX_CAMS=""
        for _c in $ALL_CAMS; do
            [ "$_c" = "$PIPE_CAM" ] || AUX_CAMS="${AUX_CAMS:+$AUX_CAMS }$_c"
        done
    fi

    # Exported, not just set: the drivers hand these to sbatch through the
    # environment rather than --export=ALL,K=V, so anything a job reads has to
    # be exported here or at the call site.
    export TAKE SEQ PIPE_CAM AUX_CAMS MIN_FRAMES HUMAN_PROMPT OBJECT_PROMPT
    export WORK="$WORK_ROOT/$SEQ"
    export TAKE_DIR="$TAKES_ROOT/$TAKE"
    export FAV_DIR="$TAKE_DIR/frame_aligned_videos"
    export CALIB="$TAKE_DIR/trajectory/gopro_calibs.csv"
    export SRC_448="$FAV_DIR/downscaled/448/$PIPE_CAM.mp4"
    export PIPE_SRC="$WORK/src/$SEQ.0.color.mp4"
    export WINDOW_JSON="$WORK/window.json"
    # Only meaningful for the take-level pass (stage 1a), where SEQ is the take
    # base rather than a clip: the list of clips that pass produced.
    export CLIPS_JSON="$WORK/clips.json"
    export MASKS_DIR="$WORK/masks"
    export CLIPS_DIR="$WORK/masks/trimmed_vids"
    export PIPE_CLIP="$CLIPS_DIR/$SEQ.0.color.mp4"
    export PACKED_MV="$WORK/packed-mv"
    export GEOM_DIR="$WORK/geom"
    export OBJECT_XYZ="$GEOM_DIR/object_xyz.npz"
    export HUMAN_J3D="$GEOM_DIR/human_joints.npz"
    export RECT_DIR="$WORK/rect"
    export RECT_CLIP="$RECT_DIR/$SEQ.0.color.mp4"
    export ALIGNED_DIR="$WORK/rect-aligned"
    export ALIGNED_CLIP="$ALIGNED_DIR/$SEQ.0.color.mp4"
    export MESH_DIR="$WORK/meshes"
    export NLF_DIR="$WORK/nlf"
    export FP_DIR="$WORK/fp"
}

recon_aux_clip() {
    # Path of the trimmed 4K clip for aux camera $1.
    #
    # The -4k suffix is slurm_trim_clips.sh's naming and is also the sequence
    # name SAM3 gives that view's masks, which triangulate_object.py then names
    # in its --view arguments. Changing it here changes it in three places.
    echo "$CLIPS_DIR/$1-4k.0.color.mp4"
}

recon_run() {
    # Run a command, or print it under DRY_RUN.
    #
    # Used for the non-sbatch work the drivers do on the login node (mkdir,
    # symlinks) so a dry run really touches nothing.
    if [ -n "$DRY_RUN" ]; then
        echo "  would run: $*"
    else
        "$@"
    fi
}

recon_sbatch() {
    # Submit an sbatch and echo its job id, or print the command under DRY_RUN.
    #
    # The id goes to stdout because the callers chain on it with
    # --dependency=afterok; every human-readable line in the drivers therefore
    # goes to stderr, not stdout.
    #
    # Job variables are passed by exporting them before the call, not through
    # --export=ALL,K=V: sbatch splits that list on commas, so a value with a
    # space in it -- the SAM3 prompts, the optimizer's override string -- comes
    # out truncated. sbatch's default --export=ALL carries the exported
    # environment through instead, with no parsing involved.
    # Prepended, so a caller passing its own --exclude still wins (last one
    # on the sbatch line takes effect).
    local excl=()
    [ -n "$EXCLUDE_NODES" ] && excl=(--exclude="$EXCLUDE_NODES")

    if [ -n "$DRY_RUN" ]; then
        echo "  would submit: sbatch ${excl[*]} $*" >&2
        # The environment IS the argument list now, so a dry run that showed
        # only the sbatch line would hide everything worth checking.
        for v in VIDEO OUT_DIR WINDOW_JSON EMIT_ROOT CLIPS_JSON \
                 EMIT_MIN_FRAMES EMIT_MAX_CLIPS TRIM_GAP NO_TRIM CHUNK HUMAN OBJECT \
                 SRC_DIR CAMS SUFFIX START END MIN_FRAMES \
                 MASKS_ROOT PACKED_ROOT HY3D_ROOT NLF_PATH FP_ROOT \
                 HY3D_MESHES_ROOT IDENTIFIER SAVE_NAME OPT_EXTRA \
                 TSTART ZFAR ERODE_DEPTH_THRES REINIT_EVERY \
                 DEPTH_HUMAN_BAND DEPTH_MAD_K; do
            [ -n "${!v:-}" ] && echo "         $v=${!v}" >&2
        done
        echo "DRYRUN"
        return
    fi
    local out
    out=$(sbatch "${excl[@]}" "$@") || { echo "sbatch failed: $*" >&2; exit 1; }
    echo "$out" | awk '{print $NF}'
}

recon_dep() {
    # Build a --dependency=afterok:... argument from job ids, or nothing.
    #
    # Returns an empty string under DRY_RUN (the ids are fake) and when no ids
    # were given, so the caller can always interpolate it unquoted.
    local ids=""
    for id in "$@"; do
        [ "$id" = "DRYRUN" ] && continue
        [ -z "$id" ] && continue
        ids="${ids:+$ids:}$id"
    done
    [ -n "$ids" ] && echo "--dependency=afterok:$ids"
}

recon_window_frames() {
    # Print the chosen window's frame count, or nothing when there is none.
    #
    # Reads window.json rather than a job log, which is the point of writing it.
    python3 -c "import json,sys
try:
    w = json.load(open(sys.argv[1]))['chosen']
except Exception:
    sys.exit(0)
if w:
    print(w['n_frames'])" "$WINDOW_JSON" 2>/dev/null
}
