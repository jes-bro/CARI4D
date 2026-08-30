#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1

#SBATCH --job-name="sam3-masks"
#SBATCH --output=sam3-masks-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Run SAM3 human+object mask extraction on one clip, one GPU.
#
#   sbatch slurm_sam3_masks.sh                                   # uses the defaults below
#   VIDEO=data/cari4d-demo/wild/videos/other.mp4 sbatch slurm_sam3_masks.sh
#   HUMAN="man" OBJECT="a blue chair" VIDEO=path.mp4 sbatch slurm_sam3_masks.sh
#
# Egocentric (Aria) sources need NO_TRIM=1: the camera wearer is invisible in
# their own view, so the trim criterion -- longest run where BOTH masks are
# present -- has no person mask to work with. Those runs also want a smaller
# CHUNK, since Aria RGB is 1408x1408 against the exo cameras' 448x796:
#
#   NO_TRIM=1 CHUNK=100 VIDEO=<aria_rgb>.mp4 OBJECT="basketball" HUMAN="hands" \
#       sbatch --time=06:00:00 slurm_sam3_masks.sh
#
# NOTE: run this from the CARI4D repo root -- run_sam3_masks.py lives in prep/ and
# the --video path is relative to the repo root (same convention as your InterMimic
# slurm scripts). #SBATCH lines are ignored if you invoke it with `bash` instead.

set -euo pipefail
cd "$(dirname "$0")"

# ---- knobs (override on the sbatch line: VAR=... sbatch slurm_sam3_masks.sh) ----
VIDEO="${VIDEO:-/vision/group/egoexo4d/takes/unc_basketball_03-31-23_02_9/frame_aligned_videos/downscaled/448/cam04.mp4}"
HUMAN="${HUMAN:-one basketball player playing basketball}"
OBJECT="${OBJECT:-ball}"

# Where the masks land. The flat default is fine for one sequence at a time;
# a batch of takes needs a directory each, because the aux clips are named for
# their camera (cam01-4k) and would otherwise collide between takes.
OUT_DIR="${OUT_DIR:-/simurgh2/projects/ret-hoi/CARI4D/sam3masks}"

# Optional: record the trimmed window as JSON so the aux views can be cut to
# the same frames by a job queued before the window is known.
WINDOW_JSON="${WINDOW_JSON:-}"

# ---- HF cache -> project space, NOT quota'd home. Also where sam3.pt lands:
#      $HF_HOME/hub/models--facebook--sam3/snapshots/<sha>/sam3.pt
export HF_HOME="${HF_HOME:-/simurgh2/projects/ret-hoi/hf_cache}"

# ---- token: must be in the *submitting* env (sbatch doesn't source ~/.bashrc).
#      Pre-authenticate ONCE on the login node so this job needs no network login:
#          HF_HOME=/simurgh2/projects/ret-hoi/hf_cache huggingface-cli login --token <tok>
#      That writes $HF_HOME/token, which this job reads. If instead you export
#      HF_TOKEN in your shell before sbatch, --export=ALL (SLURM default) carries it.
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HF_HOME/token" ]; then
    echo "[sam3] ERROR: no HF auth found -- neither \$HF_TOKEN nor $HF_HOME/token." >&2
    echo "[sam3] Pre-login on the login node (see comment above) or export HF_TOKEN before sbatch." >&2
    exit 1
fi

# ---- conda env with Python 3.12+ where you `pip install -e sam3`.
#      EDIT this to your actual env name.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SAM3_ENV:-sam3}"

echo "[sam3] host=$(hostname) job=${SLURM_JOB_ID:-<none>} gpu=$CUDA_VISIBLE_DEVICES"
echo "[sam3] video=$VIDEO"
echo "[sam3] out_dir=$OUT_DIR  window_json=${WINDOW_JSON:-<none>}"
echo "[sam3] human='$HUMAN' object='$OBJECT'  HF_HOME=$HF_HOME"
echo "[sam3] chunk=${CHUNK:-300} trim=$([ -n "${NO_TRIM:-}" ] && echo off || echo on)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd /simurgh2/projects/ret-hoi/CARI4D

# python -u so prints survive a kill: a job cancelled at the time limit
# otherwise loses everything still sitting in the stdout buffer.
# ${NO_TRIM:+--no_trim} is unquoted on purpose -- it expands to nothing when
# NO_TRIM is unset, and quoting it would pass an empty argument argparse rejects.
# The :+ form is safe under `set -u`.
python -u /simurgh2/projects/ret-hoi/CARI4D/prep/run_sam3_masks.py \
    --video "$VIDEO" \
    --human_prompt "$HUMAN" \
    --object_prompt "$OBJECT" \
    --output_dir "$OUT_DIR" \
    --chunk_size "${CHUNK:-300}" \
    ${NO_TRIM:+--no_trim} \
    ${WINDOW_JSON:+--window_json "$WINDOW_JSON"} \
    --visualize

echo "[sam3] done."
