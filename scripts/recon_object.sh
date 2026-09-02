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
# WHICH CAMERA. The crop comes from a 4K AUX view, not the 448 pipeline view.
# The object is ~13 px across in the pipeline camera and ~110 px in the aux
# ones -- eight times the linear resolution, for geometry that is the same
# regardless of which camera saw it. run_hy3d_recon's --out_seq names the
# result for the tracking sequence, so nothing downstream can tell the
# difference. MESH_CAM picks the view; unset, the one whose object mask is
# largest on the chosen frame wins.
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
# PICK THE FRAME FIRST, do not guess:
#
#   python prep/pick_object_frame.py --work work/<seq>
#
# That scores every frame the object is masked in and writes ONE labelled
# contact sheet of the best candidates. Look at it, read off a frame number,
# pass it as MESH_FRAME. Nothing else in this pipeline rests on a single frame
# the way this does.

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

# Pick the aux view with the most object pixels on that frame. The aux clips
# are frame-aligned with the pipeline clip by construction -- that is what the
# frame-accurate trim in stage 1b buys -- so MESH_FRAME means the same instant
# in all of them.
if [ -z "${MESH_CAM:-}" ]; then
    MESH_CAM=$(python3 -c "
import h5py, sys
frame, masks, cams = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
best, best_n = None, -1
for cam in cams:
    try:
        with h5py.File(f'{masks}/{cam}-4k_masks_k0.h5', 'r') as f:
            n = int(f[f'{cam}-4k/{frame:06d}-k0.obj_rend_mask.png'][()].sum())
    except Exception:
        continue
    if n > best_n:
        best, best_n = cam, n
print(best or '')
" "$MESH_FRAME" "$MASKS_DIR" $AUX_CAMS 2>/dev/null || true)
fi
[ -n "${MESH_CAM:-}" ] || { echo "ERROR: no aux view has an object mask on frame $MESH_FRAME." >&2
    echo "       Pick another frame, or MESH_CAM=<cam> to force one." >&2; exit 1; }

MESH_CLIP="$(recon_aux_clip "$MESH_CAM")"
[ -n "$DRY_RUN" ] || [ -f "$MESH_CLIP" ] || {
    echo "ERROR: no 4K clip at $MESH_CLIP -- has stage 1b run?" >&2; exit 1; }

log "take=$TAKE  clip=$SEQ${frames:+  ($frames frames)}"
log "reconstructing the object from $MESH_CAM (4K) frame $MESH_FRAME"
log "mesh root=$MESH_DIR"

recon_run mkdir -p "$MESH_DIR"

# MASKS_ROOT is set explicitly rather than left to slurm_hy3d_recon.sh's
# trimmed_vids/ inference -- it would infer the same thing, but a path this
# script already knows should not be re-derived by string surgery downstream.
export HY3D_ROOT="$MESH_DIR" MASKS_ROOT="$MASKS_DIR"
# --out_seq names the result for the tracking sequence even though the pixels
# came from another camera: fp_hy3d_track globs the tracking prefix, and
# estimate_scale_video parses the frame index back out of the filename to fetch
# depth from the TRACKING video.
job=$(recon_sbatch --job-name="o1-$SEQ" \
    scripts/slurm_hy3d_recon.sh "$MESH_CLIP" "$MESH_FRAME" \
    --out_seq "$SEQ" ${SKIP_HY3D:+--skip_hy3d})
log "hunyuan3d object reconstruction     job $job"

MESH_DIR_F="$MESH_DIR/${SEQ}_$(printf '%03d' "$MESH_FRAME")_rgba"
recon_check \
    "ls $MESH_DIR_F" \
    "# LOOK AT THE MESH. The _rgba.png is the crop it was given; the _align.obj" \
    "# is what came out. Orbit renders, if you want them:" \
    "#   \$(ls -d Hunyuan3D-2/blender-*/blender) -b -P scripts/render_obj_views.py \\" \
    "#       -- $MESH_DIR_F/${SEQ}_$(printf '%03d' "$MESH_FRAME")_align.obj /tmp/objviews 4 512" \
    "#" \
    "# Judge whether it looks like the object. Its SIZE is not set yet and is not" \
    "# yours to judge here -- the next stage measures that across all the views" \
    "# and reports it." \
    "#" \
    "# If it is wrong, delete this mesh before trying another frame, or two will" \
    "# sit side by side and the pipeline will silently take the first:" \
    "#   rm -rf $MESH_DIR_F"
recon_next \
    "TAKE=$TAKE SEQ=$SEQ bash scripts/recon_solve.sh"
