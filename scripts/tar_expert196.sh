#!/bin/bash
# Pack every soccer take of EgoExo4D participant 196, plus each take's fisheye
# calibration. Sibling of tar_expert725.sh -- read that one's header too.
#
# WHO THIS IS. Participant 196: Late Expert, male, iiith, 3 takes -- one of
# each soccer drill (Dribbling 259.4s, Juggling 176.1s, Penalty Kick 45.8s),
# 8.0 minutes total. Capture iiith_soccer_002, capture_uid
# 831ce766-12fc-4705-a715-0937bec214f7, physical_setting_uid 46.
#
# ONE PARTICIPANT PER CAPTURE at iiith, so this costs its own calibration.
# Same pitch (setting 46) as 840 and 841, re-rigged between sessions.
#
# NO cam02 IN THIS CAPTURE. The exo cameras are cam01, cam03, cam04, cam05 --
# the numbering has a hole in it, which is why every path below is written out
# rather than generated from a range. Override the pipeline accordingly:
#
#   ALL_CAMS="cam01 cam03 cam04 cam05" TAKE=... SEQ=... bash scripts/recon_masks.sh
#
# The UNC default of cam01-cam04 would look for a cam02 that does not exist and
# fail, and the sibling captures in this batch each differ again:
# iiith_soccer_030 has cam01-cam05, iiith_soccer_031 has cam01-cam04.
#
# THE PENALTY KICK TAKE IS SHORT -- 45.8 s against 259.4 s for the dribbling
# one. It is the shortest take in the whole soccer selection, so if a clip
# target is tight this is the take most likely to come up short.
#
# WHICH CAMERA: unknown. best_exo says cam03 for two takes and cam01 for the
# third. Pick from the mask videos.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the four 4K exo views (no cam02)
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# Verify availability on the mirror BEFORE running this -- note the camera list
# in the check matches this capture's, not the UNC one:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-32s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam03','cam04','cam05']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in sorted(T,key=lambda x:x['take_idx']) if t.get('participant_uid')==196]"
#
#   bash scripts/tar_expert196.sh                       # -> <repo>/expert196_takes.tar
#
# The archive is ~4 GB.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert196_takes.tar}"

[ -d "$TAKES_ROOT" ] || { echo "ERROR: no takes root at $TAKES_ROOT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

NEED_KB=5000000
AVAIL_KB=$(df -Pk "$(dirname "$OUT")" | awk 'NR==2 {print $4}')
if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
    echo "ERROR: need ~$((NEED_KB / 1024 / 1024)) GB at $(dirname "$OUT"), have $((AVAIL_KB / 1024 / 1024)) GB" >&2
    exit 1
fi

cd "$TAKES_ROOT"

tar -chvf "$OUT" \
    iiith_soccer_002_2/frame_aligned_videos/cam01.mp4 \
    iiith_soccer_002_2/frame_aligned_videos/cam03.mp4 \
    iiith_soccer_002_2/frame_aligned_videos/cam04.mp4 \
    iiith_soccer_002_2/frame_aligned_videos/cam05.mp4 \
    iiith_soccer_002_2/frame_aligned_videos/downscaled/448 \
    iiith_soccer_002_2/trajectory/gopro_calibs.csv \
    iiith_soccer_002_4/frame_aligned_videos/cam01.mp4 \
    iiith_soccer_002_4/frame_aligned_videos/cam03.mp4 \
    iiith_soccer_002_4/frame_aligned_videos/cam04.mp4 \
    iiith_soccer_002_4/frame_aligned_videos/cam05.mp4 \
    iiith_soccer_002_4/frame_aligned_videos/downscaled/448 \
    iiith_soccer_002_4/trajectory/gopro_calibs.csv \
    iiith_soccer_002_6/frame_aligned_videos/cam01.mp4 \
    iiith_soccer_002_6/frame_aligned_videos/cam03.mp4 \
    iiith_soccer_002_6/frame_aligned_videos/cam04.mp4 \
    iiith_soccer_002_6/frame_aligned_videos/cam05.mp4 \
    iiith_soccer_002_6/frame_aligned_videos/downscaled/448 \
    iiith_soccer_002_6/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "3 takes x (4 cams -- no cam02 -- + the 448 dir record + its files + calib)."
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
