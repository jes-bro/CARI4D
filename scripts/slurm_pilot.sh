#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#SBATCH --job-name="recon-pilot"
#SBATCH --output=recon-pilot-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# The whole pilot as ONE queued job: every stage runs in this allocation, in
# order, through the local backend (scripts/recon_local.sh) -- the same thing
# as running recon_pilot.sh in an interactive shell, for when the interactive
# partition is slow to give one out. Every knob the pilot reads comes through
# the environment of the sbatch line (sbatch exports it by default):
#
#   OBJECT_FROM=ego EGO_OBJECT_SIZE=0.35 EGO_RGB_ONLY=1 VIZ_EXTRA="--obj_color 255,0,255" \
#   TAKE=iiith_cooking_57_2 SEQ=Date03_Sub06_mpot_pour CLIP_LO=5190 CLIP_HI=5550 \
#   HUMAN_PROMPT=person OBJECT_PROMPT="metal saucepan with handle" \
#   MESH_FROM=$PWD/work/ego-test/meshes THROUGH=solve \
#       sbatch --exclude=simurgh2,simurgh6 scripts/slurm_pilot.sh
#
#   tail -f recon-pilot-<jobid>.out
#
# Submit from base: an active conda env with compiler hooks leaks into the
# job and breaks the job scripts' own activation.

set -euo pipefail
cd "${REPO:-${SLURM_SUBMIT_DIR:-/simurgh2/projects/ret-hoi/CARI4D}}"
export REPO="$PWD"
export RECON_BACKEND=local
echo "[pilot-job] host=$(hostname) job=${SLURM_JOB_ID:-none} repo=$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec bash scripts/recon_pilot.sh
