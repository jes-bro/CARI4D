#!/bin/bash
# A static prop beside the reconstruction: chair, bench, CPR manikin, piano.
#
#   PROP=chair PROP_PROMPT="chair" TAKE=<take> SEQ=<clip> bash scripts/recon_prop.sh masks
#   ...                                                    bash scripts/recon_prop.sh geometry
#   ...                                                    bash scripts/recon_prop.sh mesh
#   ...                                                    bash scripts/recon_prop.sh scale
#   ...                                                    bash scripts/recon_prop.sh register
#   ...                                                    bash scripts/recon_prop.sh check
#
# WHY A SIDECAR. The pipeline has one object, tracked frame by frame, and the
# optimizer's contact term is wrists-only while its penetration term pushes
# the body out of anything it touches. Putting a chair in there would unseat
# the sitter. So a prop gets its own masks, its own triangulation, its own
# Hunyuan3D mesh and ONE FoundationPose registration held for the whole clip,
# written to <work>/props/<prop>/<seq>_<prop>.json in the same camera frame as
# the bundle's pose_abs -- for the simulator to place as static collision
# geometry, the way the basketball hoop was. Nothing in the main pipeline
# changes; see prep/static_prop.py for the file format.
#
# WHERE IT SITS IN THE FLOW. `masks` needs stage 1b (the trimmed 4K aux clips).
# `geometry` and `mesh` need only that. `scale` and `register` need the main
# stage 3 to have produced rect-aligned/ (the depth-aligned clip), so run them
# after recon_solve.sh has got that far. A second prop (bench AND piano) is the
# same six commands with another PROP; nothing collides, every path carries
# the prop's name.
#
# The prompts: PROP_PROMPT is the SAM3 text for the prop ("chair", "piano
# bench", "CPR manikin"). HUMAN_PROMPT defaults to "person" here -- the person
# masks are needed too, both for the frame pick (a chair frame with nobody on
# it beats one with) and because the mask readers expect them.

set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:?usage: PROP=... PROP_PROMPT=... TAKE=... SEQ=... bash scripts/recon_prop.sh <masks|geometry|mesh|scale|register|check>}"
: "${PROP:?set PROP to a short name: chair, bench, manikin, piano}"
case "$STAGE" in masks) : "${PROP_PROMPT:?set PROP_PROMPT, the SAM3 text for the prop}" ;; esac

# The object prompt the guard in recon_common.sh looks at is the prop's here:
# this driver never masks the tracked object, and the guard exists to stop
# basketball defaults reaching a non-basketball take, which a set PROP_PROMPT
# already rules out.
export HUMAN_PROMPT="${HUMAN_PROMPT:-person}"
export OBJECT_PROMPT="${OBJECT_PROMPT:-${PROP_PROMPT:-prop}}"
source scripts/recon_common.sh
recon_require_env
recon_paths

export PROP PROP_PROMPT="${PROP_PROMPT:-}"
export PROP_DIR="$WORK/props/$PROP"
export PROP_MASKS_DIR="$PROP_DIR/masks"
export PROP_MESHES="$PROP_DIR/meshes"
SIDECAR="$PROP_DIR/${SEQ}_${PROP}.json"

log() { echo "[recon-prop] $*" >&2; }
log "take=$TAKE  clip=$SEQ  prop=$PROP  pipe_cam=$PIPE_CAM  aux=$AUX_CAMS"
log "prop dir=$PROP_DIR"

case "$STAGE" in
# --- masks: SAM3 with the prop prompt, own directory so nothing collides -----
masks)
    [ -n "$DRY_RUN" ] || [ -f "$PIPE_CLIP" ] || { echo "ERROR: no trimmed pipeline clip at $PIPE_CLIP -- has stage 1 run?" >&2; exit 1; }
    for c in $AUX_CAMS; do
        [ -n "$DRY_RUN" ] || [ -f "$(recon_aux_clip "$c")" ] || { echo "ERROR: no trimmed aux clip for $c -- has stage 1b run?" >&2; exit 1; }
    done
    recon_run mkdir -p "$PROP_MASKS_DIR"
    # Same seq names as the main masks (the clip's, and <cam>-4k), in a
    # different directory: that is what lets triangulate_object.py and
    # scale_object_mesh.py take the prop's sets through the same --view syntax.
    export OUT_DIR="$PROP_MASKS_DIR" NO_TRIM=1 HUMAN="$HUMAN_PROMPT" OBJECT="$PROP_PROMPT"
    unset WINDOW_JSON EMIT_ROOT CLIPS_JSON
    export VIDEO="$PIPE_CLIP" CHUNK=300
    job=$(recon_sbatch --job-name="p1-$PROP-$SEQ" scripts/slurm_sam3_masks.sh)
    log "P1 sam3 '$PROP_PROMPT' on the pipeline clip   job $job"
    export CHUNK="${AUX_CHUNK:-60}"
    for c in $AUX_CAMS; do
        export VIDEO="$(recon_aux_clip "$c")"
        job=$(recon_sbatch --time="${SAM3_AUX_TIME:-04:00:00}" --job-name="p2-$PROP-$c-$SEQ" scripts/slurm_sam3_masks.sh)
        log "P2 sam3 '$PROP_PROMPT' on $c (4K)          job $job"
    done
    recon_check \
        "ls $PROP_MASKS_DIR/*_sam3_vis.mp4" \
        "# WATCH THEM. Is the blue mask on the $PROP and only the $PROP -- not the" \
        "# person, not a second $PROP in the room? If it grabs the wrong thing, re-run" \
        "# this stage with a more specific PROP_PROMPT (\"black office chair\")." \
        "python prep/check_view_coverage.py --masks_root $PROP_MASKS_DIR --seq $SEQ"
    recon_next "PROP=$PROP TAKE=$TAKE SEQ=$SEQ bash scripts/recon_prop.sh geometry"
    ;;

# --- geometry: triangulate + pool + pick the mesh frame (CPU) --------------
geometry)
    export CALIB PIPE_CAM AUX_CAMS TAKE
    job=$(recon_sbatch --job-name="p3-$PROP-$SEQ" scripts/slurm_static_prop.sh geometry)
    log "P3 triangulate, pool, pick frame        job $job"
    recon_check \
        "cat $PROP_DIR/prop_world.json" \
        "# tri_spread_median_m is how far a STATIC thing wandered between frames." \
        "# Centimetres: fine. Tens of centimetres: a mask caught the person or" \
        "# another $PROP on some frames -- look at the overlay videos again." \
        "cat $PROP_DIR/mesh_pick.json" \
        "# cam + frame Hunyuan3D will see. contact is the fraction of the prop's" \
        "# outline the person touches there; low is good."
    recon_next "PROP=$PROP TAKE=$TAKE SEQ=$SEQ bash scripts/recon_prop.sh mesh"
    ;;

# --- mesh: Hunyuan3D from the picked aux frame, prop's own mesh root --------
mesh)
    pick="$PROP_DIR/mesh_pick.json"
    if [ -z "${MESH_CAM:-}" ] || [ -z "${MESH_FRAME:-}" ]; then
        [ -n "$DRY_RUN" ] || [ -f "$pick" ] || { echo "ERROR: no $pick -- run geometry first, or set MESH_CAM and MESH_FRAME" >&2; exit 1; }
        if [ -f "$pick" ]; then
            MESH_CAM="${MESH_CAM:-$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['cam'])" "$pick")}"
            MESH_FRAME="${MESH_FRAME:-$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['frame'])" "$pick")}"
        else
            MESH_CAM="${MESH_CAM:-<cam>}"; MESH_FRAME="${MESH_FRAME:-<frame>}"
        fi
    fi
    log "mesh from $MESH_CAM (4K) frame $MESH_FRAME"
    recon_run mkdir -p "$PROP_MESHES"
    # --out_seq names the mesh for the clip, as recon_object.sh does; the prop's
    # own HY3D_ROOT is what keeps it from landing on the tracked object's mesh.
    export HY3D_ROOT="$PROP_MESHES" MASKS_ROOT="$PROP_MASKS_DIR"
    job=$(recon_sbatch --job-name="p4-$PROP-$SEQ" \
        scripts/slurm_hy3d_recon.sh "$(recon_aux_clip "$MESH_CAM")" "$MESH_FRAME" \
        --out_seq "$SEQ" ${SKIP_HY3D:+--skip_hy3d})
    log "P4 hunyuan3d $PROP mesh                  job $job"
    recon_check \
        "ls $PROP_MESHES/${SEQ}_$(printf '%03d' "$MESH_FRAME" 2>/dev/null || echo "$MESH_FRAME")_rgba" \
        "# The _rgba.png is the crop it was given -- if the person is in it, the" \
        "# mesh has a body-shaped bite filled with invented geometry. Acceptable" \
        "# for a seat the simulator only has to stand on; not for a piano keyboard." \
        "# Another frame: rm -rf that directory, then MESH_CAM=... MESH_FRAME=... rerun." \
        "# Two meshes side by side and the scale step takes the first."
    recon_next \
        "# needs the main stage 3 to have produced $ALIGNED_DIR first, then:" \
        "PROP=$PROP TAKE=$TAKE SEQ=$SEQ bash scripts/recon_prop.sh scale"
    ;;

# --- scale: measure the mesh across the views (same tool, prop's roots) -----
scale)
    [ -n "$DRY_RUN" ] || [ -f "$ALIGNED_CLIP" ] || { echo "ERROR: no $ALIGNED_CLIP -- run recon_solve.sh far enough to align depth first" >&2; exit 1; }
    export MASKS_ROOT="$PROP_MASKS_DIR" HY3D_ROOT="$PROP_MESHES" OBJECT_XYZ="$PROP_DIR/prop_xyz.npz"
    export CALIB MASKS_DIR="$PROP_MASKS_DIR" SEQ PIPE_CAM AUX_CAMS SCALE_METHOD="${SCALE_METHOD:-measure}"
    job=$(recon_sbatch --job-name="p5-$PROP-$SEQ" scripts/slurm_scale_object.sh "$ALIGNED_CLIP")
    log "P5 scale the $PROP mesh                   job $job"
    recon_check \
        "cat $PROP_MESHES-metric/object_scale.json" \
        "# measured_size_m is the prop's apparent diameter. A chair is ~0.5-1.0 m," \
        "# a CPR manikin ~0.6-1.0 m, a bench ~1 m, an upright piano ~1.5 m."
    recon_next "PROP=$PROP TAKE=$TAKE SEQ=$SEQ bash scripts/recon_prop.sh register"
    ;;

# --- register: FoundationPose on the best frames, medoid, sidecar (GPU) ----
register)
    [ -n "$DRY_RUN" ] || [ -f "$ALIGNED_CLIP" ] || { echo "ERROR: no $ALIGNED_CLIP" >&2; exit 1; }
    export CALIB PIPE_CAM AUX_CAMS TAKE ALIGNED_CLIP
    job=$(recon_sbatch --gres=gpu:1 --time="${PROP_REG_TIME:-01:00:00}" \
        --job-name="p6-$PROP-$SEQ" scripts/slurm_static_prop.sh register)
    log "P6 register + sidecar                     job $job"
    recon_check \
        "python -c \"import json; d=json.load(open('$SIDECAR')); print(json.dumps(d['checks'], indent=1))\"" \
        "# register_spread_deg: the K registrations should agree to a few degrees." \
        "# Tens of degrees means the $PROP looks symmetric to FoundationPose -- check" \
        "# the pose against the footage before shipping it." \
        "# tri_vs_register_m compares the mesh's bbox centre with the mask centroid;" \
        "# they are different points on a big prop, so flag metres, not centimetres."
    recon_next \
        "# the sidecar for the simulator:" \
        "cat $SIDECAR" \
        "# same camera frame as pose_abs in the bundle; format in prep/static_prop.py"
    ;;

check)
    python prep/static_prop.py check --prop "$PROP" --seq "$SEQ" --calib "$CALIB" --pipe_cam "$PIPE_CAM" \
        --masks_dir "$PROP_MASKS_DIR" --prop_dir "$PROP_DIR" \
        --mesh "$(ls "$PROP_MESHES"-metric/*/*_align.obj 2>/dev/null | head -1 || true)"
    ;;

*)
    echo "ERROR: stage must be masks, geometry, mesh, scale, register or check (got '$STAGE')" >&2; exit 1 ;;
esac
