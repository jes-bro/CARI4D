#!/bin/bash
# Pack the pan/saucepan cooking takes of EgoExo4D participant 50, plus each
# take's fisheye calibration. Sibling of tar_expert387.sh and the others.
#
# WHO THIS IS. Participant 50: female, iiith, 6 expert-tier takes on
# pan-and-stove tasks -- three omelets (Early Expert) and three milk teas
# (Intermediate Expert), 28.9 minutes total. The most footage of any cooking
# participant at an expert tier. All at physical_setting_uid 7, one kitchen.
#
# WHY COOKING IS THE GENDER-DIVERSE SET. Every one of the 31 labelled soccer
# participants is male, and the basketball experts are overwhelmingly male --
# but iiith's cooking participants are largely women. Of the five cooking
# people selected (50, 58, 55, 52, 56) four are female.
#
# NO LATE EXPERTS EXIST IN COOKING. The Cooking scenario tops out at
# Intermediate Expert; there is no Late Expert anywhere in it. These takes are
# Early and Intermediate, which is the ceiling, not a compromise on the
# available data.
#
# ONE CAPTURE PER TAKE. Unlike basketball, iiith cooking records each dish as
# its own capture -- these six takes come from five: iiith_cooking_03, _04,
# _05 (two takes), _56 and _57. Each take carries its own
# trajectory/gopro_calibs.csv so the geometry side is covered automatically,
# but the pipeline-camera choice should be confirmed per take rather than
# assumed to hold across the set.
#
# LONG VIDEOS. 189 to 411 seconds each against basketball's 64. The masking
# bill scales with that, and SAM3 over a 7-minute 4K take is a different
# proposition from a 1-minute one -- budget accordingly.
#
# THE OBJECT IS A PAN, not a ball. Every object-dependent knob the solve stage
# derives (ZFAR, DEPTH_HUMAN_BAND, ERODE_DEPTH_THRES, REINIT_EVERY) was tuned
# for a 24 cm sphere at 6 m. prep/derive_knobs.py re-derives them from this
# clip's own geometry, which is the point, but OBJECT_PROMPT must change:
#
#   OBJECT_PROMPT="frying pan" HUMAN_PROMPT="a person cooking at a stove" ...
#
# and a pan's orientation IS observable, unlike a sphere's, so REINIT_EVERY
# should not stay at the basketball value of 1.
#
# CAMERAS: cam01 through cam04 -- matches the pipeline default, unlike the
# soccer captures.
#
# Verify availability on the mirror BEFORE running this:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-24s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in T if t['take_name'] in ['iiith_cooking_03_3','iiith_cooking_04_1','iiith_cooking_05_1','iiith_cooking_05_5','iiith_cooking_56_2','iiith_cooking_57_2']]"
#
#   bash scripts/tar_expert50.sh                        # -> <repo>/expert50_takes.tar
#
# The archive is ~12 GB, the largest in the set. Run it detached:
#   nohup bash scripts/tar_expert50.sh > tar50.log 2>&1 &

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert50_takes.tar}"

[ -d "$TAKES_ROOT" ] || { echo "ERROR: no takes root at $TAKES_ROOT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

NEED_KB=13000000
AVAIL_KB=$(df -Pk "$(dirname "$OUT")" | awk 'NR==2 {print $4}')
if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
    echo "ERROR: need ~$((NEED_KB / 1024 / 1024)) GB at $(dirname "$OUT"), have $((AVAIL_KB / 1024 / 1024)) GB" >&2
    exit 1
fi

cd "$TAKES_ROOT"

tar -chvf "$OUT" \
    iiith_cooking_03_3/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_03_3/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_03_3/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_03_3/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_03_3/frame_aligned_videos/downscaled/448 \
    iiith_cooking_03_3/trajectory/gopro_calibs.csv \
    iiith_cooking_04_1/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_04_1/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_04_1/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_04_1/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_04_1/frame_aligned_videos/downscaled/448 \
    iiith_cooking_04_1/trajectory/gopro_calibs.csv \
    iiith_cooking_05_1/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_05_1/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_05_1/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_05_1/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_05_1/frame_aligned_videos/downscaled/448 \
    iiith_cooking_05_1/trajectory/gopro_calibs.csv \
    iiith_cooking_05_5/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_05_5/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_05_5/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_05_5/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_05_5/frame_aligned_videos/downscaled/448 \
    iiith_cooking_05_5/trajectory/gopro_calibs.csv \
    iiith_cooking_56_2/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_56_2/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_56_2/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_56_2/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_56_2/frame_aligned_videos/downscaled/448 \
    iiith_cooking_56_2/trajectory/gopro_calibs.csv \
    iiith_cooking_57_2/frame_aligned_videos/cam01.mp4 \
    iiith_cooking_57_2/frame_aligned_videos/cam02.mp4 \
    iiith_cooking_57_2/frame_aligned_videos/cam03.mp4 \
    iiith_cooking_57_2/frame_aligned_videos/cam04.mp4 \
    iiith_cooking_57_2/frame_aligned_videos/downscaled/448 \
    iiith_cooking_57_2/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "6 takes x (4 cams + the 448 dir record + its files + calib)."
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
