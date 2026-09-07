#!/bin/bash
# Pack the pan/saucepan cooking takes of EgoExo4D participant 58, plus each
# take's fisheye calibration. Sibling of tar_expert50.sh -- read that header
# too, for the pan-specific pipeline notes.
#
# WHO THIS IS. Participant 58: female, iiith, 6 expert-tier takes on
# pan-and-stove tasks -- three omelets and three milk teas, ALL Intermediate
# Expert, 19.5 minutes. Uniform tier across every take, which none of the other
# cooking people have. All at physical_setting_uid 11, one kitchen.
#
# SIX CAPTURES FOR SIX TAKES: iiith_cooking_40 through _45, one take each.
# Every take carries its own trajectory/gopro_calibs.csv, so the geometry side
# is covered, but confirm the pipeline camera per take rather than assuming it
# holds across the set.
#
# THE MOST EVEN TAKE LENGTHS in the cooking selection -- 169 to 212 seconds,
# where participant 50 ranges 189 to 411. Easier to reason about masking cost.
#
# THE OBJECT IS A PAN. See tar_expert50.sh's header: OBJECT_PROMPT must change
# from the basketball default, and REINIT_EVERY=1 (which exists because a
# sphere's orientation is unobservable) is wrong for a pan.
#
# CAMERAS: cam01 through cam04 -- matches the pipeline default.
#
# Verify availability on the mirror BEFORE running this:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-24s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in T if t.get('participant_uid')==58 and t['take_name'].startswith('iiith_cooking_4')]"
#
#   bash scripts/tar_expert58.sh                        # -> <repo>/expert58_takes.tar
#
# The archive is ~8 GB. Run it detached:
#   nohup bash scripts/tar_expert58.sh > tar58.log 2>&1 &

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert58_takes.tar}"

[ -d "$TAKES_ROOT" ] || { echo "ERROR: no takes root at $TAKES_ROOT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

NEED_KB=9000000
AVAIL_KB=$(df -Pk "$(dirname "$OUT")" | awk 'NR==2 {print $4}')
if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
    echo "ERROR: need ~$((NEED_KB / 1024 / 1024)) GB at $(dirname "$OUT"), have $((AVAIL_KB / 1024 / 1024)) GB" >&2
    exit 1
fi

cd "$TAKES_ROOT"

tar -chvf "$OUT" \
    iiith_cooking_40_1/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_40_1/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_40_1/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_40_1/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_40_1/frame_aligned_videos/downscaled/448 \
    iiith_cooking_40_1/trajectory/gopro_calibs.csv \
    iiith_cooking_41_1/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_41_1/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_41_1/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_41_1/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_41_1/frame_aligned_videos/downscaled/448 \
    iiith_cooking_41_1/trajectory/gopro_calibs.csv \
    iiith_cooking_42_1/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_42_1/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_42_1/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_42_1/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_42_1/frame_aligned_videos/downscaled/448 \
    iiith_cooking_42_1/trajectory/gopro_calibs.csv \
    iiith_cooking_43_1/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_43_1/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_43_1/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_43_1/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_43_1/frame_aligned_videos/downscaled/448 \
    iiith_cooking_43_1/trajectory/gopro_calibs.csv \
    iiith_cooking_44_1/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_44_1/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_44_1/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_44_1/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_44_1/frame_aligned_videos/downscaled/448 \
    iiith_cooking_44_1/trajectory/gopro_calibs.csv \
    iiith_cooking_45_1/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_45_1/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_45_1/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_45_1/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_45_1/frame_aligned_videos/downscaled/448 \
    iiith_cooking_45_1/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "6 takes x (4 cams + the 448 dir record + its files + calib)."
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
