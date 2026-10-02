#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="cari4d-fp-onward"
#SBATCH --output=cari4d-fp-onward-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Re-run the pipeline from FoundationPose onward (demo-custom.sh steps 5.2, 6,
# 7: FP tracking -> CoCoNet -> joint optimisation) on an ALREADY-ALIGNED clip.
#
# Why this exists: demo-custom.sh always starts at step 1, and its step 4
# (align_monod2hum) REWRITES the aligned depth video. Any manual edit to that
# depth -- e.g. prep/inject_object_depth.py writing triangulated distances into
# the object mask -- would be wiped by a full re-run. This script enters the
# pipeline after alignment, so the depth it finds is the depth it uses.
#
# The argument is the clip inside the -aligned directory (not the original):
#
#   MASKS_ROOT=rect-bball PACKED_ROOT=rect-bball \
#   HY3D_ROOT=data/cari4d-demo/meshes \
#   FP_ROOT=data/cari4d-demo/videogen/fp-hy3d3-track-inject \
#   IDENTIFIER=_rectinj TSTART=0 ZFAR=20 ERODE_DEPTH_THRES=0.05 \
#   REINIT_EVERY=1 DEPTH_HUMAN_BAND=3.0 DEPTH_MAD_K=3.0 \
#       sbatch scripts/slurm_fp_onward.sh rect-bball-aligned/<seq>.0.color.mp4
#
# Give FP_ROOT and IDENTIFIER fresh values per variant, or the FP pickle and
# the CoCoNet/opt output directories silently overwrite the previous run's
# (which is how the original run's FP track was lost to the _rect one).
#
# Knobs and defaults are demo-custom.sh's, applied identically; sbatch's
# default --export=ALL carries the environment through.

set -euo pipefail

VIDEO=${1:?usage: sbatch scripts/slurm_fp_onward.sh <aligned video.mp4>}

# REPO defaults to the directory sbatch was invoked from, which the drivers
# guarantee is the repo root -- they cd there before submitting. That makes the
# whole pipeline work from any checkout without editing a line, instead of every
# job cd'ing into one person's home. An explicit REPO still wins, and the
# literal remains the last resort for a bare sbatch from somewhere else.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
CACHE_ROOT="${CACHE_ROOT:-/simurgh2/projects/ret-hoi}"

log() { echo "[fp-onward $(date -u +%H:%M:%S)] $*"; }

# Caches to project space -- /sailhome is over quota (same as slurm_demo_custom).
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch_cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton_cache}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CARI4D_ENV:-newcari4d}"

cd "$REPO"

video=$VIDEO
video_prefix=$(basename "$video" | cut -d. -f1)

# Same knobs as demo-custom.sh, same defaults, same env overrides.
masks_root="${MASKS_ROOT:-data/cari4d-demo/videogen/masks/}"
packed_root="${PACKED_ROOT:-data/cari4d-demo/videogen/packed/}"
hy3d_root="${HY3D_ROOT:-data/cari4d-demo/videogen/meshes}"
nlf_path="${NLF_PATH:-data/cari4d-demo/videogen/nlf}"
fp_root="${FP_ROOT:-data/cari4d-demo/videogen/fp-hy3d3-track}"
coconet_out="${COCONET_OUT:-output/coconet}"
exp_name="${EXP_NAME:-cari4d-release}"
exp_step="${EXP_STEP:-step031397}"
identifier="${IDENTIFIER:-_demo}"
tstart="${TSTART:-3.0}"
zfar="${ZFAR:-8.0}"
erode_depth_thres="${ERODE_DEPTH_THRES:-0.001}"
reinit_every="${REINIT_EVERY:-}"
depth_human_band="${DEPTH_HUMAN_BAND:-0.0}"
depth_mad_k="${DEPTH_MAD_K:-0.0}"

# Step 7's own knobs. Defaults reproduce demo-custom.sh exactly, so an existing
# invocation is unaffected. They exist so the multi-view human anchor
# (w_j3d / j3d_file from prep/triangulate_human.py) can be applied in this job
# instead of re-running the optimizer afterwards through slurm_opt_only.sh --
# on a batch of takes that second optimization is the single largest piece of
# duplicated GPU time. Give SAVE_NAME a fresh value per variant or output/opt
# overwrites the previous run.
save_name="${SAVE_NAME:-optv2}"
opt_extra="${OPT_EXTRA:-}"

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "code=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo +dirty)"
log "video=$video (aligned)  env=${CONDA_DEFAULT_ENV:-none}"
log "masks_root=${masks_root}  packed_root=${packed_root}  hy3d_root=${hy3d_root}"
log "nlf_path=${nlf_path}  fp_root=${fp_root}  coconet_out=${coconet_out}"
log "exp=${exp_name}+${exp_step}${identifier}  zfar=${zfar}  tstart=${tstart}  erode_depth_thres=${erode_depth_thres}  reinit_every=${reinit_every:-never}"
log "depth_human_band=${depth_human_band}  depth_mad_k=${depth_mad_k}"
log "save_name=${save_name}  opt_extra='${opt_extra}'"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('cuda ok:', torch.cuda.get_device_name(0))"

# The aligned clip's depth sibling is the whole point -- refuse to run without it.
depth_video=${video/.color.mp4/.depth-reg.mp4}
for required in "$video" "$depth_video" "$masks_root" "$packed_root" "${hy3d_root}-metric" "${nlf_path}-opt"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: missing required input: $required (did steps 1-5.1 run?)" >&2
        exit 1
    fi
done

# Step 5.2: FoundationPose in tracking mode (verbatim from demo-custom.sh).
# SKIP_FP=1 when the pickle under fp_root was produced elsewhere -- the object
# tracked in the ego camera and carried over by scripts/slurm_ego_object.sh --
# so CoCoNet reads that instead of a track made in this camera.
if [ -n "${SKIP_FP:-}" ]; then
    fp_pkl="${fp_root}/${video_prefix}_all.pkl"
    [ -f "$fp_pkl" ] || { echo "ERROR: SKIP_FP set but no $fp_pkl" >&2; exit 1; }
    log "step 5.2 skipped (SKIP_FP): using $fp_pkl"
else
python prep/fp_hy3d_track.py --viz_path x --wild_video --kid 0 \
--masks_root ${masks_root} --hy3d_root=${hy3d_root}-metric \
--video ${video} -o ${fp_root} --zfar ${zfar} -tstart ${tstart} \
--erode_depth_thres ${erode_depth_thres} ${reinit_every:+--reinit_every ${reinit_every}} \
--depth_human_band ${depth_human_band} --depth_mad_k ${depth_mad_k}
fi

# Step 6: CoCoNet refinement (verbatim from demo-custom.sh).
# run_horefine.py returns early when its output already exists. This job's
# whole purpose is to redo everything from FoundationPose onward, so a cached
# CoCoNet result is never what is wanted here: it handed the optimizer the
# previous run's object track -- the exo one -- on every ego rerun. Removed
# first, with a line in the log saying so.
# Moved into PREVIOUS_DIR when the driver set one, deleted otherwise.
stash_previous() {
    # Move $1 into $PREVIOUS_DIR, or remove it; either way it will not be reused.
    [ -e "$1" ] || return 0
    if [ -n "${PREVIOUS_DIR:-}" ]; then
        mkdir -p "$PREVIOUS_DIR"
        mv "$1" "$PREVIOUS_DIR/$(basename "$(dirname "$1")")__$(basename "$1")"
        log "moved previous $1 -> $PREVIOUS_DIR/"
    else
        rm -f "$1"
        log "removed previous $1"
    fi
}
coconet_pth="${coconet_out}/${exp_name}+${exp_step}${identifier}/${video_prefix}.pth"
stash_previous "$coconet_pth"
python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-behave.json \
use_sel_view=True render_video=True identifier=${identifier} use_intermediate=False data_name=test-only \
hy3d_meshes_root=${hy3d_root}-metric \
masks_root=${masks_root} \
fp_root=${fp_root} \
nlf_root=${nlf_path}-opt \
video=${video}  cam_id=0 wild_video=True \
outpath=${coconet_out}

# The optimizer RESUMES from its own previous result: opt_refineout.py globs
# <exp_dir>/<seq>*.pth, loads the last one, and when its step count has
# reached num_steps prints "skipping" and returns. So a finished result from
# an earlier run is handed back unchanged by every later run, whatever CoCoNet
# fed in. Same reasoning as for the CoCoNet file above: this job exists to
# redo the solve, so the old result goes first. KEEP_OPT=1 keeps the resume
# behaviour, for continuing an optimisation that was cut short.
opt_dir="output/opt/${exp_name}+${exp_step}${identifier}-hy3d3-${save_name}"
if [ -z "${KEEP_OPT:-}" ]; then
    for f in "${opt_dir}/${video_prefix}"*.pth; do
        [ -e "$f" ] && stash_previous "$f"
    done
fi

# Step 7: joint optimisation (demo-custom.sh's, with save_name and a trailing
# override slot). OmegaConf's CLI is last-key-wins, so ${opt_extra} unquoted at
# the end overrides any weight above it.
python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300  save_name=${save_name} batch_size=64 opt_rot=True \
opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=2.0 w_j2d=0.006 opt_smpl_trans=False opt_betas=False  \
pth_file=${coconet_out}/${exp_name}+${exp_step}${identifier}/${video_prefix}.pth  wild_video=True use_input=True \
video_root=$(dirname "$video") \
packed_root=${packed_root} \
masks_root=${masks_root}  \
hy3d_meshes_root=${hy3d_root}-metric outpath=output/opt ${opt_extra}

log "results:"
ls -la "output/opt"/*"${identifier}"* 2>/dev/null || ls -la output/opt 2>/dev/null || true
