#!/bin/bash
# Pack every soccer take of EgoExo4D participant 725, plus each take's fisheye
# calibration. Sibling of tar_expert387.sh and the other tar_expert*.sh.
#
# WHO THIS IS. Participant 725: Late Expert, male, utokyo, 3 takes -- one of
# each soccer drill (Dribbling, Juggling, Penalty Kick), 6.3 minutes total.
# Capture utokyo_soccer_8000_46_47, capture_uid
# 881b61f3-02aa-4e88-bf66-892be04d4a2c, physical_setting_uid 87.
#
# SHARES A CAPTURE WITH 726, who is Intermediate Expert and has the other three
# takes (_8, _10, _12). Two people for one calibration -- the only such pair in
# the whole soccer scenario, every other soccer capture holds one participant.
# Do 725 and 726 together.
#
# THREE TAKES, NOT EIGHTEEN. Soccer participants get one take per drill, so
# three is the ceiling here and no soccer participant has more than six. If the
# target is clips rather than takes, the takes are long -- 113 to 134 seconds
# against basketball's 64 -- so there is more inside each one.
#
# ALL SOCCER EXPERTS ARE MALE. All 31 proficiency-labelled soccer participants
# in EgoExo4D are male; there is no gender-diverse alternative to pick. The
# cooking set is where the women are.
#
# FIVE EXO CAMERAS, NOT FOUR. This capture has cam01 through cam05. The
# pipeline's ALL_CAMS in scripts/recon_common.sh is hardcoded to the UNC
# basketball set of four, so it must be overridden for anything here:
#
#   ALL_CAMS="cam01 cam02 cam03 cam04 cam05" TAKE=... SEQ=... bash scripts/recon_masks.sh
#
# Camera sets are NOT uniform across EgoExo4D -- iiith_soccer_002 has no cam02,
# iiith_soccer_031 has no cam05, upenn cooking uses gp01-gp06. Check per capture.
#
# WHICH CAMERA: unknown, and best_exo does not even agree with itself here --
# it says cam03 for four takes in this capture and cam02 for the other two.
# Since the exo rig is fixed within a capture, that disagreement is evidence
# best_exo is answering "which view looks best for this take", not "where is
# the rig". Pick from the mask videos.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the five 4K exo views
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# Verify availability on the mirror BEFORE running this -- no soccer take has
# been reconstructed from, and the downscaled/448 file count varies by capture:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-32s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04','cam05']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in sorted(T,key=lambda x:x['take_idx']) if t.get('participant_uid')==725]"
#
#   bash scripts/tar_expert725.sh                       # -> <repo>/expert725_takes.tar
#   OUT=/scratch/handoff/expert725.tar bash scripts/tar_expert725.sh
#
# The archive is ~3 GB.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert725_takes.tar}"

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
    utokyo_soccer_8000_46_47_2/frame_aligned_videos/cam01.mp4 \
    utokyo_soccer_8000_46_47_2/frame_aligned_videos/cam02.mp4 \
    utokyo_soccer_8000_46_47_2/frame_aligned_videos/cam03.mp4 \
    utokyo_soccer_8000_46_47_2/frame_aligned_videos/cam04.mp4 \
    utokyo_soccer_8000_46_47_2/frame_aligned_videos/cam05.mp4 \
    utokyo_soccer_8000_46_47_2/frame_aligned_videos/downscaled/448 \
    utokyo_soccer_8000_46_47_2/trajectory/gopro_calibs.csv \
    utokyo_soccer_8000_46_47_4/frame_aligned_videos/cam01.mp4 \
    utokyo_soccer_8000_46_47_4/frame_aligned_videos/cam02.mp4 \
    utokyo_soccer_8000_46_47_4/frame_aligned_videos/cam03.mp4 \
    utokyo_soccer_8000_46_47_4/frame_aligned_videos/cam04.mp4 \
    utokyo_soccer_8000_46_47_4/frame_aligned_videos/cam05.mp4 \
    utokyo_soccer_8000_46_47_4/frame_aligned_videos/downscaled/448 \
    utokyo_soccer_8000_46_47_4/trajectory/gopro_calibs.csv \
    utokyo_soccer_8000_46_47_6/frame_aligned_videos/cam01.mp4 \
    utokyo_soccer_8000_46_47_6/frame_aligned_videos/cam02.mp4 \
    utokyo_soccer_8000_46_47_6/frame_aligned_videos/cam03.mp4 \
    utokyo_soccer_8000_46_47_6/frame_aligned_videos/cam04.mp4 \
    utokyo_soccer_8000_46_47_6/frame_aligned_videos/cam05.mp4 \
    utokyo_soccer_8000_46_47_6/frame_aligned_videos/downscaled/448 \
    utokyo_soccer_8000_46_47_6/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "3 takes x (5 cams + the 448 dir record + its files + calib)."
echo "The availability check in this script's header prints the 448 count."
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
