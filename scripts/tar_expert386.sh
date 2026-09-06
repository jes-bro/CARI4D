#!/bin/bash
# Pack every video take of EgoExo4D participant 386, plus each take's fisheye
# calibration. Sibling of tar_expert387.sh and the other tar_expert*.sh.
#
# WHO THIS IS. Participant 386: Late Expert, male, unc, 18 takes at contiguous
# indices _40 through _57, none dropped, 12.2 minutes. Capture
# unc_basketball_03-31-23_01, capture_uid d3ee5b95-cc0c-42f3-b34f-1de8c52af46a,
# physical_setting_uid 51.
#
# THIRD OF THREE IN THIS CAPTURE. Eighteen takes is the MAXIMUM for any
# basketball participant in EgoExo4D -- nobody has twenty, and only six people
# have eighteen. Three of them are here: 384, 385 and 386, 54 takes between
# them with no null participant_uid anywhere in the capture. One calibration
# serves all three, so do them together.
#
# SHORTEST TAKES OF THE THREE. 12.2 minutes against 384's 14.8 and 385's 16.4,
# same eighteen takes -- so his individual takes are shorter and will yield
# fewer clips each if the pipeline is emitting more than one per take. Sequence
# him last of the three if clip count is the binding constraint.
#
# CAMERA: NOT cam04. best_exo says cam03 for all 54 takes here, where it said
# cam01 for unc_basketball_03-31-23_02 and the verified reconstruction there
# came from cam04. Different rig placement, different answer: confirm against
# the first mask video rather than carrying cam04 over from 387/388.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the four 4K exo views
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# Verify availability on the mirror BEFORE running this:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-33s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in sorted(T,key=lambda x:x['take_idx']) if t.get('participant_uid')==386]"
#
# Paths inside the archive are relative to the takes root, so the recipient
# runs `tar -xf expert386_takes.tar -C /their/egoexo4d/takes` and their
# TAKES_ROOT works unchanged.
#
#   bash scripts/tar_expert386.sh                       # -> <repo>/expert386_takes.tar
#   OUT=/scratch/handoff/expert386.tar bash scripts/tar_expert386.sh
#
# The archive is ~6 GB. Run it detached:
#   nohup bash scripts/tar_expert386.sh > tar386.log 2>&1 &

set -euo pipefail

# Resolved from the script's own location, not the working directory: the tar
# runs from inside TAKES_ROOT (that is what makes the archive paths relative),
# so anything derived from `pwd` would land the archive in the middle of the
# read-only mirror.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert386_takes.tar}"

[ -d "$TAKES_ROOT" ] || { echo "ERROR: no takes root at $TAKES_ROOT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

# Checked before a byte is written. Filling a quota mid-tar leaves a truncated
# archive that looks finished, which is a worse failure than refusing to start.
NEED_KB=6500000
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
    unc_basketball_03-31-23_01_40/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_40/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_40/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_40/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_40/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_40/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_41/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_41/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_41/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_41/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_41/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_41/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_42/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_42/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_42/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_42/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_42/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_42/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_43/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_43/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_43/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_43/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_43/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_43/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_44/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_44/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_44/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_44/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_44/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_44/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_45/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_45/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_45/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_45/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_45/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_45/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_46/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_46/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_46/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_46/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_46/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_46/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_47/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_47/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_47/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_47/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_47/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_47/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_48/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_48/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_48/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_48/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_48/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_48/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_49/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_49/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_49/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_49/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_49/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_49/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_50/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_50/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_50/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_50/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_50/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_50/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_51/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_51/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_51/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_51/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_51/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_51/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_52/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_52/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_52/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_52/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_52/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_52/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_53/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_53/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_53/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_53/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_53/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_53/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_54/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_54/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_54/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_54/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_54/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_54/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_55/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_55/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_55/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_55/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_55/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_55/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_56/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_56/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_56/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_56/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_56/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_56/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_01_57/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_01_57/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_01_57/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_01_57/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_01_57/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_01_57/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "expected: 270 if each downscaled/448 holds 9 files (18 takes x 15)."
echo "          The availability check in this script's header prints the real count."
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
