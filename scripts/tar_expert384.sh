#!/bin/bash
# Pack every video take of EgoExo4D participant 384, plus each take's fisheye
# calibration. Sibling of tar_expert387.sh and the other tar_expert*.sh.
#
# WHO THIS IS. Participant 384: Late Expert, male, unc, 18 takes at contiguous
# indices _2 through _19, none dropped, 14.8 minutes. Capture
# unc_basketball_03-31-23_01, capture_uid d3ee5b95-cc0c-42f3-b34f-1de8c52af46a,
# physical_setting_uid 51.
#
# WHY THIS CAPTURE MATTERS MOST. Eighteen takes is the MAXIMUM for any
# basketball participant in EgoExo4D -- nobody has twenty, and only six people
# have eighteen. Three of those six live in this one capture: 384, 385 and 386.
# So this is three maximum-length participants for a single calibration, the
# best remaining block by any measure. (The other three eighteen-take people
# are 382, 383 and 387, already packed.)
#
# The capture is 54 takes, exactly 18 each for the three of them, and NO takes
# with a null participant_uid -- the only capture packed so far with nothing
# ambiguous in it.
#
# CAMERA: NOT cam04. best_exo says cam03 for all 54 takes here, where it said
# cam01 for unc_basketball_03-31-23_02 and the verified reconstruction there
# came from cam04. Different rig placement, different answer: do not carry
# cam04 over from 387/388. Treat cam03 as the hypothesis and confirm it against
# the first mask video before committing the capture to it.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the four 4K exo views
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# Verify availability on the mirror BEFORE running this -- nothing has been
# reconstructed from this capture:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-33s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in sorted(T,key=lambda x:x['take_idx']) if t.get('participant_uid')==384]"
#
# Paths inside the archive are relative to the takes root, so the recipient
# runs `tar -xf expert384_takes.tar -C /their/egoexo4d/takes` and their
# TAKES_ROOT works unchanged.
#
#   bash scripts/tar_expert384.sh                       # -> <repo>/expert384_takes.tar
#   OUT=/scratch/handoff/expert384.tar bash scripts/tar_expert384.sh
#
# The archive is ~7 GB. Run it detached -- the -v output floods a terminal and
# a Ctrl-C leaves a truncated archive with no resume:
#   nohup bash scripts/tar_expert384.sh > tar384.log 2>&1 &

set -euo pipefail

# Resolved from the script's own location, not the working directory: the tar
# runs from inside TAKES_ROOT (that is what makes the archive paths relative),
# so anything derived from `pwd` would land the archive in the middle of the
# read-only mirror.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert384_takes.tar}"

[ -d "$TAKES_ROOT" ] || { echo "ERROR: no takes root at $TAKES_ROOT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

# Checked before a byte is written. Filling a quota mid-tar leaves a truncated
# archive that looks finished, which is a worse failure than refusing to start.
NEED_KB=7500000
AVAIL_KB=$(df -Pk "$(dirname "$OUT")" | awk 'NR==2 {print $4}')
if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
    echo "ERROR: need ~$((NEED_KB / 1024 / 1024)) GB at $(dirname "$OUT"), have $((AVAIL_KB / 1024 / 1024)) GB" >&2
    echo "       set OUT to somewhere with room" >&2
    exit 1
fi

cd "$TAKES_ROOT"

# -c create, -h follow symlinks, -v so the log shows every file that went in,
# -f the archive. No -z: the payload is already-encoded H.264.
tar -chvf "$OUT" \
    unc_basketball_03-31-23_01_2/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_2/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_2/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_2/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_2/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_2/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_3/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_3/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_3/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_3/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_3/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_3/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_4/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_4/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_4/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_4/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_4/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_4/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_5/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_5/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_5/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_5/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_5/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_5/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_6/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_6/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_6/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_6/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_6/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_6/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_7/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_7/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_7/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_7/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_7/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_7/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_8/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_8/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_8/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_8/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_8/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_8/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_9/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_9/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_9/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_9/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_9/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_9/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_10/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_10/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_10/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_10/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_10/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_10/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_11/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_11/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_11/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_11/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_11/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_11/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_12/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_12/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_12/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_12/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_12/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_12/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_13/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_13/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_13/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_13/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_13/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_13/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_14/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_14/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_14/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_14/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_14/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_14/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_15/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_15/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_15/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_15/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_15/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_15/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_16/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_16/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_16/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_16/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_16/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_16/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_17/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_17/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_17/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_17/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_17/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_17/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_18/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_18/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_18/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_18/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_18/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_18/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_19/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_19/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_19/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_19/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_19/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_19/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "expected: 270 if each downscaled/448 holds 9 files (18 takes x 15)."
echo "          The availability check in this script's header prints the real count."
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
