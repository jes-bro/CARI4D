#!/bin/bash
# Pack every soccer take of EgoExo4D participant 726, plus each take's fisheye
# calibration. Sibling of tar_expert725.sh -- read that one's header too.
#
# WHO THIS IS. Participant 726: Intermediate Expert, male, utokyo, 3 takes --
# one of each soccer drill (Dribbling, Juggling, Penalty Kick), 6.0 minutes.
# Capture utokyo_soccer_8000_46_47, capture_uid
# 881b61f3-02aa-4e88-bf66-892be04d4a2c, physical_setting_uid 87.
#
# SAME CAPTURE AS 725, who is Late Expert with takes _2, _4, _6. The rig never
# moved between them, so 726's camera work IS 725's -- one calibration for two
# people, the only such pair in the entire soccer scenario. Reconstruct 725
# first (higher tier, marginally longer takes) and 726 inherits the setup.
#
# FIVE EXO CAMERAS: cam01 through cam05. Override the pipeline's UNC default:
#
#   ALL_CAMS="cam01 cam02 cam03 cam04 cam05" TAKE=... SEQ=... bash scripts/recon_masks.sh
#
# WHICH CAMERA: unknown. best_exo disagrees with itself across this capture
# (cam03 for four takes, cam02 for two) even though the rig is fixed, so it is
# not reporting a rig property. Pick from the mask videos, and use the same
# camera for 725 and 726.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the five 4K exo views
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# Verify availability on the mirror BEFORE running this:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-32s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04','cam05']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in sorted(T,key=lambda x:x['take_idx']) if t.get('participant_uid')==726]"
#
#   bash scripts/tar_expert726.sh                       # -> <repo>/expert726_takes.tar
#
# The archive is ~3 GB.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert726_takes.tar}"

[ -d "$TAKES_ROOT" ] || { echo "ERROR: no takes root at $TAKES_ROOT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

NEED_KB=4000000
AVAIL_KB=$(df -Pk "$(dirname "$OUT")" | awk 'NR==2 {print $4}')
if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
    echo "ERROR: need ~$((NEED_KB / 1024 / 1024)) GB at $(dirname "$OUT"), have $((AVAIL_KB / 1024 / 1024)) GB" >&2
    exit 1
fi

cd "$TAKES_ROOT"

tar -chvf "$OUT" \
    utokyo_soccer_8000_46_47_8/frame_aligned_videos/cam01.mp4 \
    utokyo_soccer_8000_46_47_8/frame_aligned_videos/cam02.mp4 \
    utokyo_soccer_8000_46_47_8/frame_aligned_videos/cam03.mp4 \
    utokyo_soccer_8000_46_47_8/frame_aligned_videos/cam04.mp4 \
    utokyo_soccer_8000_46_47_8/frame_aligned_videos/cam05.mp4 \
    utokyo_soccer_8000_46_47_8/frame_aligned_videos/downscaled/448 \
    utokyo_soccer_8000_46_47_8/trajectory/gopro_calibs.csv \
    utokyo_soccer_8000_46_47_10/frame_aligned_videos/cam01.mp4 \
    utokyo_soccer_8000_46_47_10/frame_aligned_videos/cam02.mp4 \
    utokyo_soccer_8000_46_47_10/frame_aligned_videos/cam03.mp4 \
    utokyo_soccer_8000_46_47_10/frame_aligned_videos/cam04.mp4 \
    utokyo_soccer_8000_46_47_10/frame_aligned_videos/cam05.mp4 \
    utokyo_soccer_8000_46_47_10/frame_aligned_videos/downscaled/448 \
    utokyo_soccer_8000_46_47_10/trajectory/gopro_calibs.csv \
    utokyo_soccer_8000_46_47_12/frame_aligned_videos/cam01.mp4 \
    utokyo_soccer_8000_46_47_12/frame_aligned_videos/cam02.mp4 \
    utokyo_soccer_8000_46_47_12/frame_aligned_videos/cam03.mp4 \
    utokyo_soccer_8000_46_47_12/frame_aligned_videos/cam04.mp4 \
    utokyo_soccer_8000_46_47_12/frame_aligned_videos/cam05.mp4 \
    utokyo_soccer_8000_46_47_12/frame_aligned_videos/downscaled/448 \
    utokyo_soccer_8000_46_47_12/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "3 takes x (5 cams + the 448 dir record + its files + calib)."
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
