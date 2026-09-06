#!/bin/bash
# Pack every video take of EgoExo4D participant 315, plus each take's fisheye
# calibration. Sibling of tar_expert387.sh and the other tar_expert*.sh.
#
# WHO THIS IS. Participant 315 is the second of only TWO female Late Experts in
# all of EgoExo4D basketball; 318 (uniandes_basketball_001) is the other. Both
# uniandes. Labels come from proficiency_demonstrator_{train,val}.json joined to
# takes.json on take_uid; gender from participants.json.
#
# 6 takes, contiguous _2 through _7, none dropped, 6.6 minutes. Capture
# uniandes_basketball_002, capture_uid 8d7756bd-e59c-4c98-af99-14b15112af12,
# physical_setting_uid 49.
#
# A SEPARATE CALIBRATION FROM 318. Same gym as uniandes_basketball_001 but a
# different capture, so the rig was re-placed: this one needs its own
# calibration and its own pipeline-camera confirmation. The two female Late
# Experts unavoidably cost two calibrations -- they were recorded in different
# sessions and there is no way around it.
#
# THIN CLIP MARGIN. Six takes, ~24 estimated clips (list_layup_takes.py's
# estimator capped at EMIT_MAX_CLIPS=4) against a 20-clip target. This capture
# holds four participants -- 315, 320, 316, 323, the last three Intermediate
# Expert and all four female -- at 6 takes each, sharing this calibration. If
# 315's masks come up short, a replacement from the same capture costs no new
# camera work.
#
# One take in this capture has a null participant_uid. It is not packed: only
# 315's six are, listed individually.
#
# CAMERA: cam03. best_exo says cam03 for all 25 takes here, and
# splits/layup-batch.tsv already pins uniandes to cam03 -- but confirm against
# the first mask video. At UNC best_exo said cam01 for the whole capture and
# the verified reconstruction came from cam04.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the four 4K exo views
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# Verify availability on the mirror BEFORE running this -- no uniandes capture
# has been reconstructed from, and the file count under downscaled/448 follows
# the capture's aria stream count, so do not assume UNC's nine:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-32s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in sorted(T,key=lambda x:x['take_idx']) if t.get('participant_uid')==315]"
#
# Paths inside the archive are relative to the takes root, so the recipient
# runs `tar -xf expert315_takes.tar -C /their/egoexo4d/takes` and their
# TAKES_ROOT works unchanged.
#
#   bash scripts/tar_expert315.sh                       # -> <repo>/expert315_takes.tar
#   OUT=/scratch/handoff/expert315.tar bash scripts/tar_expert315.sh
#
# The archive is ~3 GB.

set -euo pipefail

# Resolved from the script's own location, not the working directory: the tar
# runs from inside TAKES_ROOT (that is what makes the archive paths relative),
# so anything derived from `pwd` would land the archive in the middle of the
# read-only mirror.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert315_takes.tar}"

[ -d "$TAKES_ROOT" ] || { echo "ERROR: no takes root at $TAKES_ROOT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

# Checked before a byte is written. Filling a quota mid-tar leaves a truncated
# archive that looks finished, which is a worse failure than refusing to start.
NEED_KB=4500000
AVAIL_KB=$(df -Pk "$(dirname "$OUT")" | awk 'NR==2 {print $4}')
if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
    echo "ERROR: need ~$((NEED_KB / 1024 / 1024)) GB at $(dirname "$OUT"), have $((AVAIL_KB / 1024 / 1024)) GB" >&2
    echo "       set OUT to somewhere with room" >&2
    exit 1
fi

cd "$TAKES_ROOT"

# -c create, -h follow symlinks (the mirror uses them in places), -v so the
# terminal shows every file that went in, -f the archive. No -z: the payload is
# already-encoded H.264, so compression costs minutes and saves nothing.
#
#   nohup bash scripts/tar_expert315.sh > tar315.log 2>&1 &
tar -chvf "$OUT" \
    uniandes_basketball_002_2/frame_aligned_videos/cam01.mp4 \
    uniandes_basketball_002_2/frame_aligned_videos/cam02.mp4 \
    uniandes_basketball_002_2/frame_aligned_videos/cam03.mp4 \
    uniandes_basketball_002_2/frame_aligned_videos/cam04.mp4 \
    uniandes_basketball_002_2/frame_aligned_videos/downscaled/448 \
    uniandes_basketball_002_2/trajectory/gopro_calibs.csv \
    uniandes_basketball_002_3/frame_aligned_videos/cam01.mp4 \
    uniandes_basketball_002_3/frame_aligned_videos/cam02.mp4 \
    uniandes_basketball_002_3/frame_aligned_videos/cam03.mp4 \
    uniandes_basketball_002_3/frame_aligned_videos/cam04.mp4 \
    uniandes_basketball_002_3/frame_aligned_videos/downscaled/448 \
    uniandes_basketball_002_3/trajectory/gopro_calibs.csv \
    uniandes_basketball_002_4/frame_aligned_videos/cam01.mp4 \
    uniandes_basketball_002_4/frame_aligned_videos/cam02.mp4 \
    uniandes_basketball_002_4/frame_aligned_videos/cam03.mp4 \
    uniandes_basketball_002_4/frame_aligned_videos/cam04.mp4 \
    uniandes_basketball_002_4/frame_aligned_videos/downscaled/448 \
    uniandes_basketball_002_4/trajectory/gopro_calibs.csv \
    uniandes_basketball_002_5/frame_aligned_videos/cam01.mp4 \
    uniandes_basketball_002_5/frame_aligned_videos/cam02.mp4 \
    uniandes_basketball_002_5/frame_aligned_videos/cam03.mp4 \
    uniandes_basketball_002_5/frame_aligned_videos/cam04.mp4 \
    uniandes_basketball_002_5/frame_aligned_videos/downscaled/448 \
    uniandes_basketball_002_5/trajectory/gopro_calibs.csv \
    uniandes_basketball_002_6/frame_aligned_videos/cam01.mp4 \
    uniandes_basketball_002_6/frame_aligned_videos/cam02.mp4 \
    uniandes_basketball_002_6/frame_aligned_videos/cam03.mp4 \
    uniandes_basketball_002_6/frame_aligned_videos/cam04.mp4 \
    uniandes_basketball_002_6/frame_aligned_videos/downscaled/448 \
    uniandes_basketball_002_6/trajectory/gopro_calibs.csv \
    uniandes_basketball_002_7/frame_aligned_videos/cam01.mp4 \
    uniandes_basketball_002_7/frame_aligned_videos/cam02.mp4 \
    uniandes_basketball_002_7/frame_aligned_videos/cam03.mp4 \
    uniandes_basketball_002_7/frame_aligned_videos/cam04.mp4 \
    uniandes_basketball_002_7/frame_aligned_videos/downscaled/448 \
    uniandes_basketball_002_7/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "expected: 6 takes x (4 cams + the 448 dir record + its files + calib)."
echo "          90 if each downscaled/448 holds 9 files, as UNC's do -- the"
echo "          availability check in this script's header prints the real count."
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
