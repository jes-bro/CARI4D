#!/bin/bash
# Stage 3 of 3: the reconstruction itself.
#
# Run on the login node after stage 2's geometry numbers look right. Everything
# here is GPU hours; nothing in it needs a decision from you until the render.
#
#   G  demo-custom.sh steps 1-5.1 on the rectified clip: UniDepth, NLF, global
#      SMPL-H, depth/human alignment, object scale.
#   H  inject the triangulated ball distances into the aligned depth. FP seeds
#      its pose from the median depth inside the object mask, and monocular
#      depth cannot supply that for a small distant ball.
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

# --- FoundationPose knobs, from the verified basketball run ------------------
TSTART="${TSTART:-0}"
ZFAR="${ZFAR:-20}"
ERODE_DEPTH_THRES="${ERODE_DEPTH_THRES:-0.05}"
REINIT_EVERY="${REINIT_EVERY:-1}"
DEPTH_HUMAN_BAND="${DEPTH_HUMAN_BAND:-3.0}"
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
export TSTART ZFAR ERODE_DEPTH_THRES REINIT_EVERY DEPTH_HUMAN_BAND DEPTH_MAD_K
job_i=$(recon_sbatch $(recon_dep "$job_h") \
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

log ""
log "result:   $RESULT_DIR/$SEQ.pth"
log "render:   output/viz-pred/"
