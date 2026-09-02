#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --job-name="recon-inject"
#SBATCH --output=recon-inject-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Write the triangulated object distances into the aligned clip's depth video.
# CPU only, minutes.
#
# Its own job rather than a step inside the one before it because this is what
# gets re-run: a better mask set or a different --max_residual changes the
# triangulation, and re-injecting must not mean recomputing UniDepth and NLF.
#
#   SEQ=<seq> CALIB=<take>/trajectory/gopro_calibs.csv PIPE_CAM=cam04 \
#   MASKS_ROOT=work/<seq>/rect OBJECT_XYZ=work/<seq>/geom/object_xyz.npz \
#       sbatch scripts/slurm_inject_depth.sh work/<seq>/rect-aligned/<seq>.0.color.mp4
#
# MASKS_ROOT is the RECTIFIED mask set: the clip being edited is in rectified
# pixels, so the mask marking which pixels to touch must be too. The
# triangulated positions themselves are camera-frame 3D and need no such care.

set -euo pipefail

VIDEO=${1:?usage: sbatch scripts/slurm_inject_depth.sh <aligned video.mp4>}

# REPO defaults to the directory sbatch was invoked from, which the drivers
# guarantee is the repo root -- they cd there before submitting. That makes the
# whole pipeline work from any checkout without editing a line, instead of every
# job cd'ing into one person's home. An explicit REPO still wins, and the
# literal remains the last resort for a bare sbatch from somewhere else.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
export PYTHONUNBUFFERED=1

# conda may not be on PATH at all: sbatch --export=ALL hands the job the
# SUBMITTING shell's environment, so a terminal that never initialised conda
# produces jobs that cannot find it -- and the failure is "conda: command not
# found" from inside a queued job, an hour after the mistake.
if ! command -v conda >/dev/null 2>&1; then
    for _rc in "$HOME/.bashrc" "$HOME/.bash_profile" /etc/profile.d/conda.sh; do
        [ -f "$_rc" ] && . "$_rc" >/dev/null 2>&1 || true
        command -v conda >/dev/null 2>&1 && break
    done
fi
command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda not found. It is not on PATH here and sourcing your shell" >&2
    echo "       profile did not provide it. Run 'source ~/.bashrc' in the shell" >&2
    echo "       you submit from, or set CONDA_EXE." >&2
    exit 1
}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CARI4D_ENV:-newcari4d}"
cd "$REPO"

: "${MASKS_ROOT:?set MASKS_ROOT to the rectified mask directory}"
: "${OBJECT_XYZ:?set OBJECT_XYZ to the triangulation .npz}"
: "${CALIB:?set CALIB}"
PIPE_CAM="${PIPE_CAM:-cam04}"

echo "[inject] video=$VIDEO"
echo "[inject] masks_root=$MASKS_ROOT  xyz=$OBJECT_XYZ  cam=$PIPE_CAM"
echo "[inject] code=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo +dirty)"

for required in "$VIDEO" "$MASKS_ROOT" "$OBJECT_XYZ" "$CALIB"; do
    [ -e "$required" ] || { echo "ERROR: missing required input: $required" >&2; exit 1; }
done

python prep/inject_object_depth.py --video "$VIDEO" --masks_root "$MASKS_ROOT" \
    --xyz "$OBJECT_XYZ" --calib "$CALIB" --cam "$PIPE_CAM" \
    ${SEQ:+--seq "$SEQ"} \
    ${SPHERE_DIAMETER:+--sphere_diameter "$SPHERE_DIAMETER"}

echo "[inject] done."
