#!/bin/bash
# Stage 3 of 3: the reconstruction itself.
#
# Run on the login node after stage 2's geometry numbers look right. Everything
# here is GPU hours; nothing in it needs a decision from you until the render.
#
#   G  demo-custom.sh steps 1-4 on the rectified clip: UniDepth, NLF, global
#      SMPL-H, depth/human alignment.
#   H  inject the triangulated ball distances into the aligned depth. FP seeds
#      its pose from the median depth inside the object mask, and monocular
#      depth cannot supply that for a small distant ball.
#   S  the object's metric scale, AFTER the injection -- it separates scale from
#      distance using depth, and before H that depth is the monocular one.
#   I  FoundationPose -> CoCoNet -> joint optimization, with the multi-view
#      human anchor (w_j3d against the triangulated joints) applied in the same
#      job rather than re-optimizing afterwards.
#   J  render the result over the source video.
#
# The FoundationPose knobs below are the basketball reconstruction's, carried
# from slurm_fp_onward.sh's own documented invocation -- not defaults, and not
# guesses. They are basketball-specific: ZFAR=20 because the ball is 6-7m out,
# ERODE_DEPTH_THRES=0.05 because a 24cm sphere at 6m recedes ~18mm per pixel and
# the 1mm default erases it, REINIT_EVERY=1 because a thrown ball leaves the
# tracker's convergence basin every frame and a sphere's orientation is
# unobservable anyway. A different object needs all three revisited.
#
#   TAKE=unc_basketball_03-31-23_02_3 SEQ=Date03_Sub01_bball_rev003 \
#       bash scripts/recon_solve.sh
#
#   DRY_RUN=1 ... to print the chain without submitting.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/recon_common.sh

recon_require_env
recon_paths

log() { echo "[recon-solve] $*" >&2; }

# --- FoundationPose knobs ----------------------------------------------------
# ZFAR and DEPTH_HUMAN_BAND are DERIVED from this clip's own triangulated
# geometry, not carried over from a basketball at seven metres: the first is
# beyond the furthest the object is ever seen, the second is the largest
# object-to-person depth gap actually observed. Both with generous margin,
# since both fail asymmetrically -- too loose keeps background the mask
# discards anyway, too tight deletes the object.
# Every object-dependent knob below is DERIVED from this clip: how far the
# object is seen (ZFAR), how far it gets from the person (DEPTH_HUMAN_BAND),
# how much its surface recedes per pixel (ERODE_DEPTH_THRES = Z/f), and whether
# its orientation is observable at all (REINIT_EVERY, from a symmetry test on
# the reconstructed mesh). Each reproduces the value that was originally
# arrived at by hand for the basketball, which is the point: same method, no
# constant to retype for a different object.
# `|| true` because under `set -o pipefail` a no-match ls fails the whole
# pipeline, and the assignment failing exits the script with ls's status --
# which is what a dry run on a machine with no mesh did, silently and with
# exit 2.
MESH_OBJ="$(ls "$MESH_DIR"/*/*_align.obj 2>/dev/null | head -1 || true)"
if [ -z "$DRY_RUN" ] && [ -f "$OBJECT_XYZ" ]; then
    eval "$(python prep/derive_knobs.py --object_xyz "$OBJECT_XYZ" \
        ${HUMAN_J3D:+--human_j3d "$HUMAN_J3D"} --calib "$CALIB" --cam "$PIPE_CAM" \
        --masks_root "$RECT_DIR" --seq "$SEQ" \
        ${MESH_OBJ:+--mesh "$MESH_OBJ"})"
fi
TSTART="${TSTART:-0}"
ZFAR="${ZFAR:-20}"
DEPTH_HUMAN_BAND="${DEPTH_HUMAN_BAND:-3.0}"
ERODE_DEPTH_THRES="${ERODE_DEPTH_THRES:-0.05}"
REINIT_EVERY="${REINIT_EVERY-1}"
DEPTH_MAD_K="${DEPTH_MAD_K:-3.0}"

# --- optimizer variant ------------------------------------------------------
# IDENTIFIER and SAVE_NAME together name the output directory, and both must be
# fresh per variant or output/opt and the FP pickle overwrite the previous run.
# Defaults reproduce the exported basketball recipe (optj3d): the multi-view
# human anchor on, the monocular init's height prior off.
IDENTIFIER="${IDENTIFIER:-_rectinj}"
SAVE_NAME="${SAVE_NAME:-optj3d}"
OPT_EXTRA="${OPT_EXTRA:-opt_smpl_trans=True w_init_ht=0 w_j3d=500 j3d_file=$HUMAN_J3D}"

if [ -z "$DRY_RUN" ]; then
    for required in "$RECT_CLIP" "$OBJECT_XYZ" "$HUMAN_J3D" "$CALIB" \
                    "$RECT_DIR/${SEQ}_masks_k0.h5" "$MESH_DIR"; do
        [ -e "$required" ] || { echo "ERROR: missing stage-2 output: $required" >&2; exit 1; }
    done
fi

log "take=$TAKE  seq=$SEQ  work=$WORK"
log "fp: tstart=$TSTART zfar=$ZFAR erode=$ERODE_DEPTH_THRES reinit=$REINIT_EVERY band=$DEPTH_HUMAN_BAND mad_k=$DEPTH_MAD_K"
log "opt: identifier=$IDENTIFIER save_name=$SAVE_NAME extra='$OPT_EXTRA'"

recon_run mkdir -p "$NLF_DIR" "$FP_DIR"

# --- G: steps 1-5.1 ---------------------------------------------------------
# One masks/packed root for every job below: rectify_fisheye.py put the warped
# masks in rect/, and stage 2's job F put the rectified-clip keypoints there too.
export MASKS_ROOT="$RECT_DIR" PACKED_ROOT="$RECT_DIR"
export HY3D_ROOT="$MESH_DIR" NLF_PATH="$NLF_DIR" FP_ROOT="$FP_DIR"

# Exported before job G, not just before job I: step 5.1 inside the prep job
# reads depth through the same eroding code path FoundationPose tracks with, so
# it needs the same threshold. Exporting it only for the tracking job left the
# scale step on FoundationPose's 1mm default, which erases a small distant
# object entirely.
export TSTART ZFAR ERODE_DEPTH_THRES REINIT_EVERY DEPTH_HUMAN_BAND DEPTH_MAD_K

job_g=$(recon_sbatch --job-name="s1-$SEQ" \
    scripts/slurm_prep_aligned.sh "$RECT_CLIP")
log "G  unidepth -> nlf -> smplh -> align  job $job_g"

# --- H: triangulated depth into the object mask -----------------------------
job_h=$(recon_sbatch $(recon_dep "$job_g") \
    --job-name="s2-$SEQ" \
    scripts/slurm_inject_depth.sh "$ALIGNED_CLIP")
log "H  inject triangulated ball depth     job $job_h"

# --- I: FP -> CoCoNet -> optimization ---------------------------------------
export IDENTIFIER SAVE_NAME OPT_EXTRA
job_s=$(recon_sbatch $(recon_dep "$job_h") \
    --job-name="s2b-$SEQ" \
    scripts/slurm_scale_object.sh "$ALIGNED_CLIP")
log "S  object metric scale (post-inject)  job $job_s"

job_i=$(recon_sbatch $(recon_dep "$job_s") \
    --job-name="s3-$SEQ" \
    scripts/slurm_fp_onward.sh "$ALIGNED_CLIP")
log "I  foundationpose -> coconet -> opt   job $job_i"

# --- J: the render you actually judge it by ---------------------------------
RESULT_DIR="output/opt/${EXP_NAME:-cari4d-release}+${EXP_STEP:-step031397}${IDENTIFIER}-hy3d3-${SAVE_NAME}"
export HY3D_MESHES_ROOT="$MESH_DIR-metric"
job_j=$(recon_sbatch $(recon_dep "$job_i") \
    --job-name="s4-$SEQ" \
    scripts/slurm_viz_pred.sh "$RESULT_DIR/$SEQ.pth" "$ALIGNED_CLIP")
log "J  render                             job $job_j"

recon_check \
    "tail -18 recon-scale-${job_s}.out" \
    "# the object's MEASURED size, and whether the views agreed on it." \
    "# Views disagreeing badly means a mask is on something else." \
    "" \
    "ls -lat output/viz-pred/ | head -3" \
    "# watch the newest mp4 (the filename is timestamped, so it is the fresh one)." \
    "# The object is drawn with its own texture, so a correct render LOOKS LIKE" \
    "# the real object -- judge whether it tracks, not whether it stands out." \
    "" \
    "result: $RESULT_DIR/$SEQ.pth" \
    "" \
    "# Re-running this stage? Delete the scaled mesh first, or the scale step" \
    "# reuses its cached result:   rm -rf $MESH_DIR-metric"
recon_next \
    "Nothing -- this clip is done." \
    "" \
    "Start the next clip at:" \
    "" \
    "   TAKE=$TAKE SEQ=<next clip> bash scripts/recon_masks.sh"
