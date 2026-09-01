#!/bin/bash
# Reconstruct the object's mesh from the scene, for ONE clip.
#
# Hunyuan3D on an RGBA crop of the object taken from the clip itself: the
# object's real shape, recovered from the footage, not a template. That is the
# point -- the pipeline is reconstructing this scene's object, and substituting
# a primitive throws away the thing being measured.
#
# Runs in the hy3d conda env with Blender for the GLB->OBJ conversion, which is
# why it is its own submission rather than a step inside slurm_geometry.sh.
# It only needs the clip and its masks, so it can run alongside stage 2 -- but
# it must finish before stage 3, which consumes the mesh.
#
#   TAKE=<take> SEQ=<seq><letter> bash scripts/recon_object.sh
#   MESH_FRAME=64 TAKE=... SEQ=... bash scripts/recon_object.sh
#
# WHICH FRAME. The crop comes from one frame, so pick one where the object is
# big, sharp and unoccluded -- single-image reconstruction cannot recover what
# it cannot see. The default is the clip's midpoint rather than frame 0, since
# a clip often starts at motion onset with the object far away or still in a
# hand. The dribble reconstruction used frame 64 of 101.
#
# The choice matters twice over: tools/estimate_scale_video.py reads the frame
# index back out of the mesh filename and estimates metric scale from THAT
# frame's depth, so a frame with no usable depth inside the object mask fails
# there with "valid is empty" -- one stage later, in a different job.
#
# SMOKE TEST FIRST. --skip_hy3d writes the RGBA crop and stops, in about a
# second, with no model download and no GPU:
#
#   SKIP_HY3D=1 TAKE=... SEQ=... bash scripts/recon_object.sh
#
# Then look at the crop before spending an allocation on it -- it is the entire
# input to the reconstruction.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/recon_common.sh

recon_require_env
recon_paths

log() { echo "[recon-object] $*" >&2; }

if [ -z "$DRY_RUN" ]; then
    for required in "$PIPE_CLIP" "$MASKS_DIR/${SEQ}_masks_k0.h5"; do
        [ -e "$required" ] || {
            echo "ERROR: missing stage-1 output: $required" >&2
            echo "       The object crop comes from the clip and its masks." >&2
            exit 1
        }
    done
fi

# Midpoint by default: a clip's first frame is the worst systematic choice,
# being where the action starts and the object is least likely to be well seen.
frames=$(recon_window_frames)
MESH_FRAME="${MESH_FRAME:-$(( ${frames:-100} / 2 ))}"

log "take=$TAKE  clip=$SEQ${frames:+  ($frames frames)}"
log "reconstructing the object from frame $MESH_FRAME"
log "mesh root=$MESH_DIR"

recon_run mkdir -p "$MESH_DIR"

# MASKS_ROOT is set explicitly rather than left to slurm_hy3d_recon.sh's
# trimmed_vids/ inference -- it would infer the same thing, but a path this
# script already knows should not be re-derived by string surgery downstream.
export HY3D_ROOT="$MESH_DIR" MASKS_ROOT="$MASKS_DIR"
job=$(recon_sbatch --job-name="o1-$SEQ" \
    scripts/slurm_hy3d_recon.sh "$PIPE_CLIP" "$MESH_FRAME" \
    ${SKIP_HY3D:+--skip_hy3d})
log "hunyuan3d object reconstruction     job $job"

log ""
log "when it finishes, LOOK AT IT before stage 3:"
log "  ls $MESH_DIR/${SEQ}_$(printf '%03d' "$MESH_FRAME")_rgba/"
log "  # the _rgba.png is the input crop; the _align.obj is the mesh"
log "then: TAKE=$TAKE SEQ=$SEQ bash scripts/recon_solve.sh"
