#!/bin/bash
# Pack every soccer take of EgoExo4D participant 841, plus each take's fisheye
# calibration. Sibling of tar_expert725.sh -- read that one's header too.
#
# WHO THIS IS. Participant 841: Late Expert, male, iiith, 3 takes -- one of
# each soccer drill (Dribbling 183.2s, Juggling 105.9s, Penalty Kick 221.0s),
# 8.5 minutes total. Capture iiith_soccer_031, capture_uid
# 0a78873e-acfb-4f11-a262-a87c5671004f, physical_setting_uid 46.
#
# ONE PARTICIPANT PER CAPTURE at iiith, so this costs its own calibration.
# Same pitch (setting 46) as 840 and 196, re-rigged between sessions.
#
# FOUR EXO CAMERAS HERE: cam01 through cam04 -- NOT five. The sibling captures
# in this same batch differ: iiith_soccer_030 has cam01-cam05, and
# iiith_soccer_002 has cam01, cam03, cam04, cam05 with no cam02. The camera set
# is a property of the capture, not the university, so do not copy an ALL_CAMS
# value across them. For this one:
#
#   ALL_CAMS="cam01 cam02 cam03 cam04" TAKE=... SEQ=... bash scripts/recon_masks.sh
#
# which happens to match the pipeline's UNC default -- the only capture in this
# soccer batch that does.
#
# WHICH CAMERA: unknown. best_exo says cam02 for two takes and cam03 for the
# third, within a capture whose rig did not move. Pick from the mask videos.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the four 4K exo views
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# Verify availability on the mirror BEFORE running this:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-32s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in sorted(T,key=lambda x:x['take_idx']) if t.get('participant_uid')==841]"
#
#   bash scripts/tar_expert841.sh                       # -> <repo>/expert841_takes.tar
#
# The archive is ~3.5 GB.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert841_takes.tar}"

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
    iiith_soccer_031_2/frame_aligned_videos/cam01.mp4 \
    iiith_soccer_031_2/frame_aligned_videos/cam02.mp4 \
    iiith_soccer_031_2/frame_aligned_videos/cam03.mp4 \
    iiith_soccer_031_2/frame_aligned_videos/cam04.mp4 \
    iiith_soccer_031_2/frame_aligned_videos/downscaled/448 \
    iiith_soccer_031_2/trajectory/gopro_calibs.csv \
    iiith_soccer_031_4/frame_aligned_videos/cam01.mp4 \
    iiith_soccer_031_4/frame_aligned_videos/cam02.mp4 \
    iiith_soccer_031_4/frame_aligned_videos/cam03.mp4 \
    iiith_soccer_031_4/frame_aligned_videos/cam04.mp4 \
    iiith_soccer_031_4/frame_aligned_videos/downscaled/448 \
    iiith_soccer_031_4/trajectory/gopro_calibs.csv \
    iiith_soccer_031_6/frame_aligned_videos/cam01.mp4 \
    iiith_soccer_031_6/frame_aligned_videos/cam02.mp4 \
    iiith_soccer_031_6/frame_aligned_videos/cam03.mp4 \
    iiith_soccer_031_6/frame_aligned_videos/cam04.mp4 \
    iiith_soccer_031_6/frame_aligned_videos/downscaled/448 \
    iiith_soccer_031_6/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "3 takes x (4 cams + the 448 dir record + its files + calib)."
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
