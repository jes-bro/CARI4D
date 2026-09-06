#!/bin/bash
# Pack every video take of EgoExo4D participant 318, plus each take's fisheye
# calibration. Sibling of tar_expert387.sh and the other tar_expert*.sh.
#
# WHO THIS IS. Participant 318 is one of only TWO female Late Experts in all of
# EgoExo4D basketball -- the other is 315, in uniandes_basketball_002. Both are
# uniandes. Every other Late Expert in the scenario is male or has no recorded
# gender, so if the reconstruction set is to contain a woman at the top
# proficiency tier at all, it contains these two. Labels come from
# proficiency_demonstrator_{train,val}.json joined to takes.json on take_uid;
# gender from participants.json.
#
# 6 takes, contiguous _30 through _35, none dropped, 6.8 minutes. Capture
# uniandes_basketball_001, capture_uid c813f8a9-d0ae-4c19-bba1-a99fdfb22af7,
# physical_setting_uid 49 -- a different country and gym from the UNC captures
# already sent, so this is the place-diversity half as well.
#
# THIN CLIP MARGIN, unlike the UNC participants. Six takes against UNC's
# eighteen: tools/list_layup_takes.py's estimator, capped at EMIT_MAX_CLIPS=4,
# puts this at roughly 24 clips against a 20-clip target. That needs most takes
# to yield 3-4 clips, so two takes lost to bad masks is the difference between
# hitting the target and missing it. Over-recruit rather than re-run: this
# capture holds SIX participants at 6 takes each, including three more labelled
# women (314, 317, 319 -- Intermediate Expert), and they all share this
# calibration. Swapping in a replacement costs no new camera work.
#
# CAMERA: cam03. best_exo says cam03 for all 36 takes in this capture,
# unanimously, and splits/layup-batch.tsv already pins uniandes to cam03. That
# is a stronger signal than it was at UNC -- where best_exo said cam01 for
# everything and the verified reconstruction came from cam04 -- but it is still
# a starting hypothesis to check against the first mask video, not a result.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the four 4K exo views
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# Verify availability on the mirror BEFORE running this. No uniandes capture
# has been reconstructed from, and the number of files under downscaled/448
# is per-capture (it follows the aria stream count), so confirm it rather than
# assuming UNC's nine:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-32s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in sorted(T,key=lambda x:x['take_idx']) if t.get('participant_uid')==318]"
#
# Paths inside the archive are relative to the takes root, so the recipient
# runs `tar -xf expert318_takes.tar -C /their/egoexo4d/takes` and their
# TAKES_ROOT works unchanged.
#
#   bash scripts/tar_expert318.sh                       # -> <repo>/expert318_takes.tar
#   OUT=/scratch/handoff/expert318.tar bash scripts/tar_expert318.sh
#
# The archive is ~3 GB -- much smaller than the UNC ones, because six takes.

set -euo pipefail

# Resolved from the script's own location, not the working directory: the tar
# runs from inside TAKES_ROOT (that is what makes the archive paths relative),
# so anything derived from `pwd` would land the archive in the middle of the
# read-only mirror.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert318_takes.tar}"

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
# Small enough to watch directly, but detaching still costs nothing:
#   nohup bash scripts/tar_expert318.sh > tar318.log 2>&1 &
tar -chvf "$OUT" \
    uniandes_basketball_001_30/frame_aligned_videos/cam01.mp4 \
    uniandes_basketball_001_30/frame_aligned_videos/cam02.mp4 \
    uniandes_basketball_001_30/frame_aligned_videos/cam03.mp4 \
    uniandes_basketball_001_30/frame_aligned_videos/cam04.mp4 \
    uniandes_basketball_001_30/frame_aligned_videos/downscaled/448 \
    uniandes_basketball_001_30/trajectory/gopro_calibs.csv \
    uniandes_basketball_001_31/frame_aligned_videos/cam01.mp4 \
    uniandes_basketball_001_31/frame_aligned_videos/cam02.mp4 \
    uniandes_basketball_001_31/frame_aligned_videos/cam03.mp4 \
    uniandes_basketball_001_31/frame_aligned_videos/cam04.mp4 \
    uniandes_basketball_001_31/frame_aligned_videos/downscaled/448 \
    uniandes_basketball_001_31/trajectory/gopro_calibs.csv \
    uniandes_basketball_001_32/frame_aligned_videos/cam01.mp4 \
    uniandes_basketball_001_32/frame_aligned_videos/cam02.mp4 \
    uniandes_basketball_001_32/frame_aligned_videos/cam03.mp4 \
    uniandes_basketball_001_32/frame_aligned_videos/cam04.mp4 \
    uniandes_basketball_001_32/frame_aligned_videos/downscaled/448 \
    uniandes_basketball_001_32/trajectory/gopro_calibs.csv \
    uniandes_basketball_001_33/frame_aligned_videos/cam01.mp4 \
    uniandes_basketball_001_33/frame_aligned_videos/cam02.mp4 \
    uniandes_basketball_001_33/frame_aligned_videos/cam03.mp4 \
    uniandes_basketball_001_33/frame_aligned_videos/cam04.mp4 \
    uniandes_basketball_001_33/frame_aligned_videos/downscaled/448 \
    uniandes_basketball_001_33/trajectory/gopro_calibs.csv \
    uniandes_basketball_001_34/frame_aligned_videos/cam01.mp4 \
    uniandes_basketball_001_34/frame_aligned_videos/cam02.mp4 \
    uniandes_basketball_001_34/frame_aligned_videos/cam03.mp4 \
    uniandes_basketball_001_34/frame_aligned_videos/cam04.mp4 \
    uniandes_basketball_001_34/frame_aligned_videos/downscaled/448 \
    uniandes_basketball_001_34/trajectory/gopro_calibs.csv \
    uniandes_basketball_001_35/frame_aligned_videos/cam01.mp4 \
    uniandes_basketball_001_35/frame_aligned_videos/cam02.mp4 \
    uniandes_basketball_001_35/frame_aligned_videos/cam03.mp4 \
    uniandes_basketball_001_35/frame_aligned_videos/cam04.mp4 \
    uniandes_basketball_001_35/frame_aligned_videos/downscaled/448 \
    uniandes_basketball_001_35/trajectory/gopro_calibs.csv

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
