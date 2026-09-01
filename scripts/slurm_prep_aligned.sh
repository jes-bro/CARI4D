#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

#SBATCH --job-name="recon-prep"
#SBATCH --output=recon-prep-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# demo-custom.sh steps 1 through 5.1 only: UniDepth, NLF, global SMPL-H fitting,
# depth/human alignment, and object scale. Stops there, leaving the aligned clip
# on disk.
#
# Why it stops: prep/inject_object_depth.py has to write triangulated distances
# into that clip's depth BEFORE FoundationPose reads it, and step 4 rewrites the
# aligned depth on every run -- so a full demo-custom.sh pass would either wipe
# the injection or, run afterwards, waste steps 5.2-7 on depth it is about to
# overwrite. This is the front half; scripts/slurm_fp_onward.sh is the back half
# (steps 5.2-7), and the injection goes between them.
#
# The four commands are verbatim from demo-custom.sh, same knobs, same
# environment overrides -- the same relationship slurm_fp_onward.sh has to the
# back half.
#
#   MASKS_ROOT=work/<seq>/rect PACKED_ROOT=work/<seq>/rect \
#   HY3D_ROOT=work/<seq>/meshes NLF_PATH=work/<seq>/nlf \
#       sbatch scripts/slurm_prep_aligned.sh work/<seq>/rect/<seq>.0.color.mp4

set -euo pipefail

VIDEO=${1:?usage: sbatch scripts/slurm_prep_aligned.sh <rectified video.mp4>}

# REPO defaults to the directory sbatch was invoked from, which the drivers
# guarantee is the repo root -- they cd there before submitting. That makes the
# whole pipeline work from any checkout without editing a line, instead of every
# job cd'ing into one person's home. An explicit REPO still wins, and the
# literal remains the last resort for a bare sbatch from somewhere else.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
CACHE_ROOT="${CACHE_ROOT:-/simurgh2/projects/ret-hoi}"

log() { echo "[prep $(date -u +%H:%M:%S)] $*"; }

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
video_dir=$(dirname "$video")

masks_root="${MASKS_ROOT:?set MASKS_ROOT}"
packed_root="${PACKED_ROOT:?set PACKED_ROOT}"
hy3d_root="${HY3D_ROOT:?set HY3D_ROOT}"
nlf_path="${NLF_PATH:?set NLF_PATH}"

# behave_data.const.get_hy3d_mesh_file falls back to $HY3D_MESHES_ROOT when no
# root is threaded through, and step 4 goes through it: align_monod2hum calls
# load_smpl_obj_uvmap(use_hy3d=True), which SystemExits if no mesh is found --
# before `hum_only=True` discards the object mesh it just insisted on. So the
# lookup has to succeed even though the result is unused.
#
# demo-custom.sh never sets this because the released demo's meshes sit under
# the default root. A per-sequence mesh directory does not, so it is set here.
# The non-metric root is right: step 5.1 has not run yet, so -metric does not
# exist, and the mesh is discarded anyway.
export HY3D_MESHES_ROOT="${HY3D_MESHES_ROOT:-$HY3D_ROOT}"

log "host=$(hostname) job=${SLURM_JOB_ID:-none} env=${CONDA_DEFAULT_ENV:-none}"
log "code=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo +dirty)"
log "video=$video"
log "masks_root=$masks_root  packed_root=$packed_root  hy3d_root=$hy3d_root  nlf_path=$nlf_path"
log "hy3d_meshes_root=$HY3D_MESHES_ROOT"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('cuda ok:', torch.cuda.get_device_name(0))"
command -v ffprobe >/dev/null || { echo "ERROR: ffprobe not found" >&2; exit 1; }

for required in "$video" "$masks_root" "$packed_root" "$hy3d_root"; do
    [ -e "$required" ] || { echo "ERROR: missing required input: $required" >&2; exit 1; }
done

# Step 1: metric depth. Honors the rectified K written next to the clip by
# prep/rectify_fisheye.py instead of guessing a pinhole.
log "step 1: unidepth"
python prep/unidepth_behave.py --wild_video --video "$video" -o "$video_dir"

# Step 2: NLF human pose initialisation.
log "step 2: nlf"
python prep/run_nlf_sepK.py -o "$nlf_path" --masks_root "$masks_root" --video "$video" --wild_video

# Step 3: globally consistent SMPL-H. The explicit -o is not optional; without
# it BaseBehaveVideoData mkdir's the original author's home and dies.
log "step 3: smplh fitting"
python prep/fit_smplh_global.py --wild_video --video "$video" --packed_root "$packed_root" \
    --masks_root "$masks_root" --nlf_path="$nlf_path" -o "$nlf_path-opt"

# Step 4: put UniDepth into the human's metric scale. Writes <video_dir>-aligned/.
log "step 4: depth/human alignment"
python prep/align_monod2hum.py --wild_video --nlf_path "$nlf_path-opt" \
    --masks_root "$masks_root" --video "$video" -o "$nlf_path-opt"

aligned="${video_dir}-aligned/${video_prefix}.0.color.mp4"
[ -f "$aligned" ] || { echo "ERROR: alignment produced no $aligned" >&2; exit 1; }

# Step 5.1: metric scale for the object template. Ahead of the injection
# because it reads the object mask and the aligned depth, not the injected one,
# and slurm_fp_onward.sh requires <hy3d_root>-metric to already exist.
log "step 5.1: object scale"
python tools/estimate_scale_video.py --wild_video --video "$aligned" \
    --masks_root "$masks_root" --hy3d_root "$hy3d_root" -o "$hy3d_root-metric"

log "done. aligned clip: $aligned"
ls -la "${video_dir}-aligned" "$hy3d_root-metric"
