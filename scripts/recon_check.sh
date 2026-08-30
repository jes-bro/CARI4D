#!/bin/bash
# Stage 0: does this take have everything the drivers need, before any job runs?
#
# Read-only, seconds, no slurm. Checks the three inputs stage 1 opens, which are
# not the same three that a take listing reports present:
#
#   downscaled/448/<pipe cam>.mp4   the clip SAM3 masks the whole take on
#   frame_aligned_videos/<aux>.mp4  the 4K sources the aux views are cut from
#   trajectory/gopro_calibs.csv     the fisheye calibration every geometry step
#                                   needs -- triangulation, rectification,
#                                   injection all read it, and it is per-take
#
# A take can have all four exo videos mirrored and still be missing the 448
# downscale or the trajectory, which is why this exists separately from
# tools/list_layup_takes.py --takes_root.
#
#   TAKE=unc_basketball_03-31-23_02_3 SEQ=Date03_Sub01_bball_rev003 \
#       bash scripts/recon_check.sh
#   bash scripts/recon_batch.sh check          # every take in the manifest
#
# Always exits 0: in a batch the point is the report, not stopping at the first
# gap. Grep the output for MISSING.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/recon_common.sh

recon_require_env
recon_paths

missing=()

[ -d "$TAKE_DIR" ] || missing+=("take_dir")
[ -f "$SRC_448" ] || missing+=("448/$PIPE_CAM")
[ -f "$CALIB" ] || missing+=("gopro_calibs.csv")
for c in $AUX_CAMS; do
    [ -f "$FAV_DIR/$c.mp4" ] || missing+=("4k/$c")
done

if [ ${#missing[@]} -eq 0 ]; then
    echo "OK      $SEQ  ($TAKE)"
else
    echo "MISSING $SEQ  ($TAKE): ${missing[*]}"
fi
