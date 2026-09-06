#!/bin/bash
# Pack every video take of EgoExo4D participant 383, plus each take's fisheye
# calibration. Sibling of scripts/tar_expert387.sh and tar_expert388.sh.
#
# WHO THIS IS. Participant 383 is a Late Expert (EgoExo4D
# proficiency_demonstrator_{train,val}.json, joined to takes.json on take_uid)
# with 18 takes at contiguous indices _21 through _38 -- 18.8 minutes, the most
# footage of any Late Expert basketball participant. Capture
# unc_basketball_03-30-23_02, capture_uid 5f22b4cd-938e-419d-9e52-2b070fa81d90.
#
# NOT THE SAME CAPTURE AS 387/388. Same gym (physical_setting_uid 51), so the
# room, the hoop and the lighting carry over -- but the exo rig was re-placed
# between sessions, so this capture needs its OWN calibration, its own pipeline
# camera choice and its own rectification. Budget that work once; it then
# covers 382 as well (see below).
#
# WHICH CAMERA IS UNKNOWN. EgoExo4D's best_exo says cam01 for 34 of this
# capture's 36 takes, but that field answered a different question for 387's
# capture -- it said cam01 there too, and the verified reconstruction came from
# cam04. So cam01 here is a hypothesis to check against the mask videos, not an
# answer. All four exo views ship at 4K precisely so that choice stays open.
#
# 382 COMES NEARLY FREE. This capture holds exactly two people, both Late
# Expert: 383 (_21-_38) and 382 (_2-_19), 18 takes each and no unattributed
# takes at all. Once this capture's camera work is done for 383, 382 reuses it
# outright -- the same relationship 388 has to 387.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the four 4K exo views
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# Verify availability on the mirror BEFORE running this -- these takes are from
# a capture nothing has been reconstructed from yet, so unlike 387's and 388's
# they have never been checked:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-33s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in sorted(T,key=lambda x:x['take_idx']) if t.get('participant_uid')==383]"
#
# Paths inside the archive are relative to the takes root, so the recipient
# runs `tar -xf expert383_takes.tar -C /their/egoexo4d/takes` and their
# TAKES_ROOT works unchanged.
#
#   bash scripts/tar_expert383.sh                       # -> <repo>/expert383_takes.tar
#   OUT=/scratch/handoff/expert383.tar bash scripts/tar_expert383.sh
#
# The archive is ~9 GB -- bigger than 387's or 388's, because this participant
# has the longest takes (_34 alone is 120.7 s). It lands in the checkout, is
# untracked, and the repo has no .gitignore: do not `git add -A` while it is
# there, and point OUT at scratch if the checkout is on a quota.

set -euo pipefail

# Resolved from the script's own location, not the working directory: the tar
# runs from inside TAKES_ROOT (that is what makes the archive paths relative),
# so anything derived from `pwd` would land the archive in the middle of the
# read-only mirror.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert383_takes.tar}"

[ -d "$TAKES_ROOT" ] || { echo "ERROR: no takes root at $TAKES_ROOT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

# Checked before a byte is written. Filling a quota mid-tar leaves a truncated
# archive that looks finished, which is a worse failure than refusing to start.
NEED_KB=9500000
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
# Redirect to a log and run under nohup or tmux for this one -- at ~9 GB the -v
# output floods a terminal, and a Ctrl-C leaves a truncated archive with no
# resume:
#   nohup bash scripts/tar_expert383.sh > tar383.log 2>&1 &
tar -chvf "$OUT" \
    unc_basketball_03-30-23_02_21/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_21/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_21/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_21/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_21/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_21/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_22/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_22/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_22/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_22/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_22/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_22/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_23/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_23/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_23/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_23/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_23/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_23/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_24/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_24/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_24/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_24/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_24/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_24/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_25/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_25/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_25/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_25/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_25/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_25/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_26/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_26/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_26/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_26/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_26/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_26/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_27/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_27/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_27/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_27/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_27/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_27/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_28/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_28/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_28/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_28/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_28/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_28/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_29/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_29/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_29/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_29/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_29/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_29/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_30/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_30/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_30/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_30/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_30/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_30/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_31/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_31/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_31/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_31/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_31/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_31/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_32/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_32/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_32/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_32/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_32/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_32/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_33/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_33/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_33/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_33/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_33/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_33/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_34/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_34/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_34/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_34/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_34/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_34/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_35/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_35/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_35/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_35/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_35/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_35/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_36/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_36/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_36/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_36/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_36/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_36/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_37/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_37/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_37/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_37/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_37/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_37/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_38/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_38/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_38/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_38/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_38/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_38/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "expected: 270 entries (18 takes x 15: four 4K cams, the 448 dir record,"
echo "          nine files inside it, and gopro_calibs.csv)"
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
