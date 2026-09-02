#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --job-name="cari4d-viz"
#SBATCH --output=cari4d-viz-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Render a reconstruction .pth overlaid on its original video via
# tools/viz_pred.py -- the CARI4D-native visualization, no InterMimic
# conversion anywhere in the path. For judging where recon ends and
# conversion artifacts begin.
#
#   sbatch scripts/slurm_viz_pred.sh <pred .pth> <video.mp4>
#   EXTRA="--no_sphere" sbatch scripts/slurm_viz_pred.sh <pred .pth> <video.mp4>
#
# Defaults are the wild-clip case: --wild_video, --no_side (the side camera's
# fixed look-at is wrong on wild sequences), -tstart 0. Output lands in
# output/viz-pred/ (override with OUT_ROOT).

set -euo pipefail

PTH=${1:?usage: sbatch scripts/slurm_viz_pred.sh <pred .pth> <video.mp4>}
VIDEO=${2:?usage: sbatch scripts/slurm_viz_pred.sh <pred .pth> <video.mp4>}

# REPO defaults to the directory sbatch was invoked from, which the drivers
# guarantee is the repo root -- they cd there before submitting. That makes the
# whole pipeline work from any checkout without editing a line, instead of every
# job cd'ing into one person's home. An explicit REPO still wins, and the
# literal remains the last resort for a bare sbatch from somewhere else.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
CACHE_ROOT="${CACHE_ROOT:-/simurgh2/projects/ret-hoi}"

export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch_cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg_cache}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CARI4D_ENV:-newcari4d}"
cd "$REPO"

OUT_ROOT="${OUT_ROOT:-output/viz-pred}"
HY3D="${HY3D_MESHES_ROOT:-data/cari4d-demo/meshes-metric}"

echo "[viz] pth=$PTH"
echo "[viz] video=$VIDEO  out_root=$OUT_ROOT  hy3d=$HY3D  extra='${EXTRA:-}'"
for f in "$PTH" "$VIDEO"; do
    [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done

python tools/viz_pred.py -pf "$PTH" --video "$VIDEO" \
    --wild_video --no_side -tstart 0 \
    --hy3d_meshes_root "$HY3D" \
    --out_root "$OUT_ROOT" -o "$OUT_ROOT" ${EXTRA:-}

echo "[viz] done; results:"
ls -la "$OUT_ROOT" | tail -5
