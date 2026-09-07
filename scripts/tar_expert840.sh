#!/bin/bash
# Pack every soccer take of EgoExo4D participant 840, plus each take's fisheye
# calibration. Sibling of tar_expert725.sh -- read that one's header too.
#
# WHO THIS IS. Participant 840: Late Expert, male, iiith, 3 takes -- one of
# each soccer drill (Dribbling 215.1s, Juggling 178.2s, Penalty Kick 130.9s),
# 8.7 minutes total, the most footage of the five soccer people selected.
# Capture iiith_soccer_030, capture_uid b747986d-c148-448c-912f-ae23093f3c65,
# physical_setting_uid 46.
#
# ONE PARTICIPANT PER CAPTURE at iiith, unlike utokyo_soccer_8000_46_47 where
# 725 and 726 share one. So this take set costs a calibration of its own, as do
# 841 (iiith_soccer_031) and 196 (iiith_soccer_002). All three are at
# physical_setting_uid 46 -- the same pitch, re-rigged between sessions, so the
# scene carries over but the calibration does not.
#
# FIVE EXO CAMERAS: cam01 through cam05. Override the pipeline's UNC default:
#
#   ALL_CAMS="cam01 cam02 cam03 cam04 cam05" TAKE=... SEQ=... bash scripts/recon_masks.sh
#
# Note this differs from the OTHER iiith soccer captures in this batch:
# iiith_soccer_031 has only cam01-cam04, and iiith_soccer_002 has no cam02.
# The camera set is per capture even within one university -- do not copy an
# ALL_CAMS value between them.
#
# WHICH CAMERA: unknown. best_exo says cam04 for two of this capture's three
# takes and cam01 for the third, which cannot both describe a fixed rig. Pick
# from the mask videos.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the five 4K exo views
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# Verify availability on the mirror BEFORE running this:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-32s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04','cam05']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in sorted(T,key=lambda x:x['take_idx']) if t.get('participant_uid')==840]"
#
#   bash scripts/tar_expert840.sh                       # -> <repo>/expert840_takes.tar
#
# The archive is ~4.5 GB.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert840_takes.tar}"

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
    iiith_soccer_030_2/frame_aligned_videos/cam01.mp4 \
    iiith_soccer_030_2/frame_aligned_videos/cam02.mp4 \
    iiith_soccer_030_2/frame_aligned_videos/cam03.mp4 \
    iiith_soccer_030_2/frame_aligned_videos/cam04.mp4 \
    iiith_soccer_030_2/frame_aligned_videos/cam05.mp4 \
    iiith_soccer_030_2/frame_aligned_videos/downscaled/448 \
    iiith_soccer_030_2/trajectory/gopro_calibs.csv \
    iiith_soccer_030_4/frame_aligned_videos/cam01.mp4 \
    iiith_soccer_030_4/frame_aligned_videos/cam02.mp4 \
    iiith_soccer_030_4/frame_aligned_videos/cam03.mp4 \
    iiith_soccer_030_4/frame_aligned_videos/cam04.mp4 \
    iiith_soccer_030_4/frame_aligned_videos/cam05.mp4 \
    iiith_soccer_030_4/frame_aligned_videos/downscaled/448 \
    iiith_soccer_030_4/trajectory/gopro_calibs.csv \
    iiith_soccer_030_6/frame_aligned_videos/cam01.mp4 \
    iiith_soccer_030_6/frame_aligned_videos/cam02.mp4 \
    iiith_soccer_030_6/frame_aligned_videos/cam03.mp4 \
    iiith_soccer_030_6/frame_aligned_videos/cam04.mp4 \
    iiith_soccer_030_6/frame_aligned_videos/cam05.mp4 \
    iiith_soccer_030_6/frame_aligned_videos/downscaled/448 \
    iiith_soccer_030_6/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "3 takes x (5 cams + the 448 dir record + its files + calib)."
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
