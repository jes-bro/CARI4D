#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --job-name="direct-bundle"
#SBATCH --output=direct-bundle-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# The result WITHOUT CoCoNet or the optimizer: the SMPL-H fit from the
# triangulated joints (stage 3 job G) and the object track carried into the
# pipeline camera (job E), written as the bundle the renderer and the
# InterAct converter read. prep/export_direct_bundle.py says why.
#
# Driven by scripts/recon_solve.sh with DIRECT=1; needs PARAMS, FP_PKL, OUT
# and either MESH_ROOT (globbed here, since job E writes the mesh after the
# driver submits this) or METRIC_MESH.

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CARI4D_ENV:-newcari4d}"
cd "$REPO"
export PYTHONUNBUFFERED=1

: "${PARAMS:?set PARAMS}" "${FP_PKL:?set FP_PKL}" "${OUT:?set OUT}"
if [ -z "${METRIC_MESH:-}" ]; then
    : "${MESH_ROOT:?set METRIC_MESH or MESH_ROOT}"
    METRIC_MESH=$(ls "$MESH_ROOT"/*/*_align.obj 2>/dev/null | head -1 || true)
    [ -n "$METRIC_MESH" ] || { echo "ERROR: no *_align.obj under $MESH_ROOT" >&2; exit 1; }
fi
echo "[direct] code=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo +dirty)"
echo "[direct] params=$PARAMS"
echo "[direct] fp_pkl=$FP_PKL"
echo "[direct] mesh=$METRIC_MESH"
for required in "$PARAMS" "$FP_PKL" "$METRIC_MESH"; do
    [ -e "$required" ] || { echo "ERROR: missing input: $required" >&2; exit 1; }
done
if [ -e "$OUT" ] && [ -n "${PREVIOUS_DIR:-}" ]; then
    mkdir -p "$PREVIOUS_DIR" && mv "$OUT" "$PREVIOUS_DIR/$(basename "$OUT")"
    echo "[direct] moved the previous bundle -> $PREVIOUS_DIR/"
fi
python prep/export_direct_bundle.py --params "$PARAMS" --fp_pkl "$FP_PKL" --mesh "$METRIC_MESH" --out "$OUT"
