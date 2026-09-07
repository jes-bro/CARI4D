#!/bin/bash
# Pack the saucepan cooking takes of EgoExo4D participant 56, plus each take's
# fisheye calibration. Sibling of tar_expert50.sh -- read that header too, for
# the pan-specific pipeline notes.
#
# WHO THIS IS. Participant 56: male, iiith, 2 expert-tier takes, both milk tea,
# both Intermediate Expert, 8.1 minutes. Captures iiith_cooking_29 and _31, one
# take each, both at physical_setting_uid 10.
#
# THE ONE MAN IN THE COOKING SET, which is otherwise 50, 58, 55 and 52 -- all
# female. Swap him for 530 (female, iiith, 2 omelet takes, 10.1 min, setting
# 45) if a 5/5 female cooking set is wanted instead of 4/1.
#
# SHARES A KITCHEN WITH 55, who is also at setting 10 (captures
# iiith_cooking_25, _26, _28). Same room, separate captures, so the scene
# carries over even though the calibration does not.
#
# MILK TEA IS A SAUCEPAN, not a frying pan. Both his takes are the same dish,
# so this participant contributes saucepan interaction only -- the omelet
# people (52, and half of 50 and 55) cover the frying pan.
#
# SMALLEST OF THE SET: 2 takes, 269 s and 216 s. If the clip target per person
# is tight this participant and 52 are the two at risk.
#
# THE OBJECT IS A PAN. See tar_expert50.sh's header: OBJECT_PROMPT must change
# from the basketball default, and REINIT_EVERY=1 is wrong for a pan.
#
# CAMERAS: cam01 through cam04 -- matches the pipeline default.
#
# Verify availability on the mirror BEFORE running this:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-24s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in T if t['take_name'] in ['iiith_cooking_29_1','iiith_cooking_31_1']]"
#
#   bash scripts/tar_expert56.sh                        # -> <repo>/expert56_takes.tar
#
# The archive is ~3.5 GB.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert56_takes.tar}"

[ -d "$TAKES_ROOT" ] || { echo "ERROR: no takes root at $TAKES_ROOT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

NEED_KB=4500000
AVAIL_KB=$(df -Pk "$(dirname "$OUT")" | awk 'NR==2 {print $4}')
if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
    echo "ERROR: need ~$((NEED_KB / 1024 / 1024)) GB at $(dirname "$OUT"), have $((AVAIL_KB / 1024 / 1024)) GB" >&2
    exit 1
fi

cd "$TAKES_ROOT"

tar -chvf "$OUT" \
    iiith_cooking_29_1/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_29_1/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_29_1/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_29_1/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_29_1/frame_aligned_videos/downscaled/448 \
    iiith_cooking_29_1/trajectory/gopro_calibs.csv \
    iiith_cooking_31_1/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_31_1/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_31_1/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_31_1/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_31_1/frame_aligned_videos/downscaled/448 \
    iiith_cooking_31_1/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "2 takes x (4 cams + the 448 dir record + its files + calib)."
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
