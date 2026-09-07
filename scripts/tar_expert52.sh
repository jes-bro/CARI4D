#!/bin/bash
# Pack the pan cooking takes of EgoExo4D participant 52, plus each take's
# fisheye calibration. Sibling of tar_expert50.sh -- read that header too, for
# the pan-specific pipeline notes.
#
# WHO THIS IS. Participant 52: female, iiith, 2 expert-tier takes, both
# omelets, both Early Expert, 10.5 minutes. Captures iiith_cooking_11 and _12,
# one take each, both at physical_setting_uid 8.
#
# ONLY TWO TAKES -- the thinnest of the five cooking people. Both are the same
# dish, so there is no task variety within this participant either. She is in
# the set for gender balance and because the two takes are long (346 s and
# 282 s, so 10.5 minutes from two videos). If the clip target per person is
# tight, this is the participant most likely to miss it, and the replacement
# with the same profile is 530 (female, iiith, 2 omelet takes, 10.1 minutes,
# setting 45).
#
# THE OBJECT IS A PAN. See tar_expert50.sh's header: OBJECT_PROMPT must change
# from the basketball default, and REINIT_EVERY=1 is wrong for a pan.
#
# CAMERAS: cam01 through cam04 -- matches the pipeline default.
#
# Verify availability on the mirror BEFORE running this:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-24s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in T if t['take_name'] in ['iiith_cooking_11_1','iiith_cooking_12_1']]"
#
#   bash scripts/tar_expert52.sh                        # -> <repo>/expert52_takes.tar
#
# The archive is ~4.5 GB.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert52_takes.tar}"

[ -d "$TAKES_ROOT" ] || { echo "ERROR: no takes root at $TAKES_ROOT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

NEED_KB=5500000
AVAIL_KB=$(df -Pk "$(dirname "$OUT")" | awk 'NR==2 {print $4}')
if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
    echo "ERROR: need ~$((NEED_KB / 1024 / 1024)) GB at $(dirname "$OUT"), have $((AVAIL_KB / 1024 / 1024)) GB" >&2
    exit 1
fi

cd "$TAKES_ROOT"

tar -chvf "$OUT" \
    iiith_cooking_11_1/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_11_1/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_11_1/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_11_1/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_11_1/frame_aligned_videos/downscaled/448 \
    iiith_cooking_11_1/trajectory/gopro_calibs.csv \
    iiith_cooking_12_1/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_12_1/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_12_1/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_12_1/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_12_1/frame_aligned_videos/downscaled/448 \
    iiith_cooking_12_1/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "2 takes x (4 cams + the 448 dir record + its files + calib)."
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
