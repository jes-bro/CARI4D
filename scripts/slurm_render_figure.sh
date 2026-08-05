#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="render-figure"
#SBATCH --output=render-figure-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Render a reconstruction as clean 3D geometry: the posed body and the object,
# shaded, on a plain ground, from a viewpoint of your choosing. No footage
# behind them and no side panel.
#
#   sbatch scripts/slurm_render_figure.sh                    # the sim rollout
#   PRED=output/opt/.../<seq>.pth TAG=recon sbatch scripts/slurm_render_figure.sh
#   ORBIT=90 TAG=orbit sbatch scripts/slurm_render_figure.sh
#   BG=0.08,0.09,0.10 TAG=dark sbatch scripts/slurm_render_figure.sh
#
# Pointing PRED at the original bundle draws the reconstruction instead, through
# the identical renderer -- so a reference figure and a simulator figure differ
# only in what they show.

set -euo pipefail

CARI4D=/simurgh2/projects/ret-hoi/CARI4D

SEQ="${SEQ:-Date03_Sub01_bball_dribble}"

# Default is the simulator rollout converted by InterMimic's
# scripts/sim_to_cari4d_bundle.py. Any CARI4D prediction file works.
PRED="${PRED:-$CARI4D/output/sim/reference-hy3d/${SEQ}.pth}"
KEY="${KEY:-pr}"

HY3D_MESHES_ROOT="${HY3D_MESHES_ROOT:-$CARI4D/data/cari4d-demo/meshes-metric}"
GENDER="${GENDER:-male}"

# Viewpoint. A low elevation reads body orientation better than looking down;
# ORBIT sweeps the azimuth across the clip for a turntable.
AZIM="${AZIM:-45}"
ELEV="${ELEV:-10}"
ORBIT="${ORBIT:-0}"
MARGIN="${MARGIN:-1.25}"

SIZE="${SIZE:-1024}"
FPS="${FPS:-30}"
STRIDE="${STRIDE:-1}"
BG="${BG:-1,1,1}"
BODY_COLOR="${BODY_COLOR:-0.75,0.76,0.80}"
OBJECT_COLOR="${OBJECT_COLOR:-0.85,0.45,0.15}"
NO_GROUND="${NO_GROUND:-0}"

# CARI4D writes predictions in the camera's frame, where +Y is down and +Z is
# depth. 'zup' skips the conversion for data already gravity-aligned.
FRAME="${FRAME:-camera}"

TAG="${TAG:-sim}"
OUT="${OUT:-$CARI4D/output/figures/${SEQ}_${TAG}.mp4}"

log() { echo "[figure $(date -u +%H:%M:%S)] $*"; }

export PYTHONUNBUFFERED=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CARI4D_ENV:-newcari4d}"
cd "$CARI4D"

log "host=$(hostname) job=${SLURM_JOB_ID:-none} env=$CONDA_DEFAULT_ENV"
log "pred=$PRED key=$KEY"
log "out=$OUT"
log "view azim=$AZIM elev=$ELEV orbit=$ORBIT  size=$SIZE"

for required in "$PRED" "$HY3D_MESHES_ROOT"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: missing required input: $required" >&2
        exit 1
    fi
done

GROUND_FLAG=""
if [ "$NO_GROUND" = "1" ]; then GROUND_FLAG="--no_ground"; fi

mkdir -p "$(dirname "$OUT")"
python -u tools/render_figure.py \
    -pf "$PRED" \
    --key "$KEY" \
    --seq "$SEQ" \
    --out "$OUT" \
    --gender "$GENDER" \
    --hy3d_meshes_root "$HY3D_MESHES_ROOT" \
    --azim "$AZIM" --elev "$ELEV" --orbit "$ORBIT" --margin "$MARGIN" \
    --size "$SIZE" --fps "$FPS" --stride "$STRIDE" \
    --bg "$BG" --body_color "$BODY_COLOR" --object_color "$OBJECT_COLOR" \
    --frame "$FRAME" \
    ${GROUND_FLAG:+$GROUND_FLAG}

log "done"
ls -lh "$OUT"
