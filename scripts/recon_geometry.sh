#!/bin/bash
# Stage 2 of 3: turn the masks into metric 3D, and rectify the camera.
#
# Run on the login node after you have looked at stage 1's masks. Submits:
#
#   D  Sapiens 2D keypoints per view, on the FISHEYE clips, into packed-mv.
#      Fisheye on purpose: triangulate_human.py unprojects with the
#      Kannala-Brandt model, so keypoints detected on a rectified clip would be
#      undistorted twice.
#   E  the geometry job (scripts/slurm_geometry.sh): triangulate the ball,
#      triangulate the human, rectify the pipeline clip and its masks.
#   F  Sapiens again, this time on the RECTIFIED clip, into rect/. That is the
#      set the monocular pipeline and the optimizer's 2D-joint loss read;
#      everything downstream of rectification lives in rectified pixels.
#
# STOP HERE AND LOOK. Second checkpoint, and the cheap one to act on -- these
# jobs are minutes, the stage after them is hours. In the recon-geom log:
#     triangulation coverage    frames covered, and where it is missing
#     residuals                 ~1-2px is healthy; large means the views
#                               disagree, usually a mask lost to motion blur
#     human joints residual     the basketball reconstruction ran 1.8px median
# Bad geometry here is what produced the "bent gravity" and flattened-bounce
# investigations; it is visible in these numbers before anything is rendered.
#
#   TAKE=unc_basketball_03-31-23_02_3 SEQ=Date03_Sub01_bball_rev003 \
#       bash scripts/recon_geometry.sh
#
#   DRY_RUN=1 ... to print the chain without submitting.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/recon_common.sh

recon_require_env
recon_paths

log() { echo "[recon-geom] $*" >&2; }

# --- refuse to build on a window stage 1 did not actually find --------------
if [ -z "$DRY_RUN" ]; then
    n_frames=$(recon_window_frames)
    if [ -z "$n_frames" ]; then
        echo "ERROR: no usable window in $WINDOW_JSON -- SAM3 never held both masks." >&2
        echo "       Retry stage 1 with different prompts, or a larger" >&2
        echo "       --trim_gap_tolerance, or --trim_rank 2 for the next-longest run." >&2
        exit 1
    fi
    if [ "$n_frames" -lt "$MIN_FRAMES" ]; then
        echo "ERROR: window is $n_frames frames, under MIN_FRAMES=$MIN_FRAMES." >&2
        exit 1
    fi
    log "window: $n_frames frames"

    for required in "$PIPE_CLIP" "$MASKS_DIR/${SEQ}_masks_k0.h5" "$CALIB"; do
        [ -e "$required" ] || { echo "ERROR: missing stage-1 output: $required" >&2; exit 1; }
    done
    for c in $AUX_CAMS; do
        for required in "$(recon_aux_clip "$c")" "$MASKS_DIR/$c-4k_masks_k0.h5"; do
            [ -e "$required" ] || { echo "ERROR: missing aux output: $required" >&2; exit 1; }
        done
    done
fi

log "take=$TAKE  seq=$SEQ  work=$WORK"

recon_run mkdir -p "$PACKED_MV" "$GEOM_DIR" "$RECT_DIR"

# --- D: Sapiens per view, fisheye pixels, one packed root -------------------
export MASKS_ROOT="$MASKS_DIR" PACKED_ROOT="$PACKED_MV"

pose_jobs=()
job=$(recon_sbatch --job-name="g1-$PIPE_CAM-$SEQ" \
    scripts/slurm_sapiens_pose.sh "$PIPE_CLIP")
pose_jobs+=("$job")
log "D  sapiens $PIPE_CAM (fisheye)        job $job"

for c in $AUX_CAMS; do
    # 4K frames, so more wall time than the script's 1h default.
    job=$(recon_sbatch --time="${SAPIENS_AUX_TIME:-00:30:00}" \
        --job-name="g1-$c-$SEQ" \
        scripts/slurm_sapiens_pose.sh "$(recon_aux_clip "$c")")
    pose_jobs+=("$job")
    log "D  sapiens $c (fisheye, 4K)       job $job"
done

# --- E: triangulate + rectify + template ------------------------------------
# Every variable slurm_geometry.sh reads was exported by recon_paths().
job_e=$(recon_sbatch $(recon_dep "${pose_jobs[@]}") \
    --job-name="g2-$SEQ" scripts/slurm_geometry.sh)
log "E  triangulate + rectify                job $job_e"

# --- F: Sapiens on the rectified clip, for the pipeline itself --------------
# MASKS_ROOT and PACKED_ROOT are both RECT_DIR: rectify_fisheye.py wrote the
# warped masks there, and demo-custom.sh reads masks and keypoints from the
# same place (the basketball run's MASKS_ROOT=rect-bball PACKED_ROOT=rect-bball).
export MASKS_ROOT="$RECT_DIR" PACKED_ROOT="$RECT_DIR"
job_f=$(recon_sbatch $(recon_dep "$job_e") \
    --job-name="g3-rect-$SEQ" \
    scripts/slurm_sapiens_pose.sh "$RECT_CLIP")
log "F  sapiens on the rectified clip      job $job_f"

log ""
log "when these finish, CHECK before stage 3:"
log "  grep -A20 'triangulation coverage' recon-geom-*.out"
log "  python3 prep/inspect_object_xyz.py $OBJECT_XYZ"
log "then, if the object mesh is not made yet:"
log "  TAKE=$TAKE SEQ=$SEQ bash scripts/recon_object.sh"
log "then: TAKE=$TAKE SEQ=$SEQ bash scripts/recon_solve.sh"
