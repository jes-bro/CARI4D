#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="behave-figure"
#SBATCH --output=behave-figure-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# The blue-body / salmon-object figure, through the repo's own renderer
# (tools/pyt3d_wrapper.py, as used for the BEHAVE and CARI4D papers).
#
#   sbatch scripts/slurm_behave_figure.sh                  # the simulator rollout
#   PRED=output/opt/<...>/<seq>.pth TAG=recon sbatch scripts/slurm_behave_figure.sh
#   BG=0,0,0 TAG=dark sbatch scripts/slurm_behave_figure.sh
#
# Renders from the camera the reconstruction was made in. That is not a
# limitation of the renderer but a deliberate default: it is the one viewpoint
# known to be correct, and inventing another is what produced a body lying on
# its side earlier.

set -euo pipefail

CARI4D=/simurgh2/projects/ret-hoi/CARI4D

SEQ="${SEQ:-Date03_Sub01_bball_dribble}"

# The simulator rollout, converted by InterMimic's sim_to_cari4d_bundle.py.
# That is the thing this exists to show: what the physics engine did, rendered
# the way the papers render a reconstruction. PRED takes any CARI4D prediction
# file, so pointing it at output/opt/ draws the reconstruction for comparison.
PRED="${PRED:-$CARI4D/output/sim/reference-hy3d/${SEQ}.pth}"
KEY="${KEY:-pr}"

HY3D_MESHES_ROOT="${HY3D_MESHES_ROOT:-$CARI4D/data/cari4d-demo/meshes-metric}"
GENDER="${GENDER:-male}"

# fx,fy,cx,cy of the video the reconstruction was made in, and that video's
# size. The pipeline prints this K; check your own run's log if it differs.
INTRINSICS="${INTRINSICS:-401.74728,401.15918,401.4052,228.35431}"
WIDTH="${WIDTH:-796}"
HEIGHT="${HEIGHT:-448}"

IMAGE_SIZE="${IMAGE_SIZE:-1024}"

# pytorch3d's coarse-rasterisation budget. The wrapper's own default of 50000 is
# below a reconstructed object's face count, and an overflow drops faces quietly
# -- holes in the render, no error.
MAX_FACES_PER_BIN="${MAX_FACES_PER_BIN:-300000}"
BG="${BG:-1,1,1}"
FPS="${FPS:-30}"
STRIDE="${STRIDE:-1}"

TAG="${TAG:-sim}"
OUT="${OUT:-$CARI4D/output/figures/${SEQ}_${TAG}.mp4}"

log() { echo "[behave-fig $(date -u +%H:%M:%S)] $*"; }

export PYTHONUNBUFFERED=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CARI4D_ENV:-newcari4d}"
cd "$CARI4D"

log "host=$(hostname) job=${SLURM_JOB_ID:-none} env=$CONDA_DEFAULT_ENV"
log "pred=$PRED key=$KEY"
log "out=$OUT"

for required in "$PRED" "$HY3D_MESHES_ROOT"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: missing required input: $required" >&2
        exit 1
    fi
done

mkdir -p "$(dirname "$OUT")"
python -u tools/render_behave_style.py \
    -pf "$PRED" \
    --key "$KEY" \
    --seq "$SEQ" \
    --out "$OUT" \
    --gender "$GENDER" \
    --hy3d_meshes_root "$HY3D_MESHES_ROOT" \
    --intrinsics "$INTRINSICS" \
    --width "$WIDTH" --height "$HEIGHT" \
    --image_size "$IMAGE_SIZE" \
    --max_faces_per_bin "$MAX_FACES_PER_BIN" \
    --bg "$BG" --fps "$FPS" --stride "$STRIDE"

log "done"
ls -lh "$OUT"
