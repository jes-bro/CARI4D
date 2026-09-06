#!/bin/bash
# Pack every video take of EgoExo4D participant 382, plus each take's fisheye
# calibration. Sibling of tar_expert387.sh, tar_expert388.sh, tar_expert383.sh.
#
# WHO THIS IS. Participant 382 is a Late Expert (EgoExo4D
# proficiency_demonstrator_{train,val}.json, joined to takes.json on take_uid)
# with 18 takes at contiguous indices _2 through _19, none dropped -- 14.7
# minutes. Capture unc_basketball_03-30-23_02, capture_uid
# 5f22b4cd-938e-419d-9e52-2b070fa81d90.
#
# THE SECOND PERSON IN 383's CAPTURE. 382 and 383 are the only two people in
# it, both Late Expert, 18 takes each, and between them they account for every
# take in the capture -- there are no unattributed takes to worry about here.
# The exo rig was fixed across both, so once 383's calibration and pipeline
# camera are settled, 382 reuses them outright with no new camera work. That is
# the same relationship 388 has to 387, and it is why these two were packed
# back to back.
#
# So: do NOT reconstruct this one first. Its camera work is 383's camera work,
# and 383 is the one with more footage to justify doing it.
#
# WHICH CAMERA IS UNKNOWN, same as 383. EgoExo4D's best_exo says cam01 for 34
# of this capture's 36 takes, but that field said cam01 for 387's capture too
# and the verified reconstruction there came from cam04. All four exo views
# ship at 4K so the choice stays open.
#
# NO NAME COLLISION with the takes already sent, despite the overlapping
# indices: this capture is 03-30-23_02 while 387's and 388's is 03-31-23_02, so
# `unc_basketball_03-30-23_02_9` and `unc_basketball_03-31-23_02_9` are
# different directories and extracting one cannot clobber the other.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the four 4K exo views
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# Verify availability on the mirror BEFORE running this -- nothing has been
# reconstructed from this capture, so its completeness has never been checked:
#
#   python3 -c "import json,os; R='/vision/group/egoexo4d/takes'; T=json.load(open('/vision/group/egoexo4d/takes.json')); [print('%-33s cams=%d 448=%d calib=%s' % (t['take_name'], sum(os.path.isfile('%s/%s/frame_aligned_videos/%s.mp4'%(R,t['take_name'],c)) for c in ['cam01','cam02','cam03','cam04']), len(os.listdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name']))) if os.path.isdir('%s/%s/frame_aligned_videos/downscaled/448'%(R,t['take_name'])) else 0, os.path.isfile('%s/%s/trajectory/gopro_calibs.csv'%(R,t['take_name'])))) for t in sorted(T,key=lambda x:x['take_idx']) if t.get('participant_uid')==382]"
#
# Paths inside the archive are relative to the takes root, so the recipient
# runs `tar -xf expert382_takes.tar -C /their/egoexo4d/takes` and their
# TAKES_ROOT works unchanged.
#
#   bash scripts/tar_expert382.sh                       # -> <repo>/expert382_takes.tar
#   OUT=/scratch/handoff/expert382.tar bash scripts/tar_expert382.sh
#
# The archive is ~7 GB. It lands in the checkout, is untracked, and the repo
# has no .gitignore: do not `git add -A` while it is there, and point OUT at
# scratch if the checkout is on a quota.

set -euo pipefail

# Resolved from the script's own location, not the working directory: the tar
# runs from inside TAKES_ROOT (that is what makes the archive paths relative),
# so anything derived from `pwd` would land the archive in the middle of the
# read-only mirror.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert382_takes.tar}"

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

# -c create, -h follow symlinks (the mirror uses them in places), -v so the
# terminal shows every file that went in, -f the archive. No -z: the payload is
# already-encoded H.264, so compression costs minutes and saves nothing.
#
# Run it detached -- the -v output floods a terminal and a Ctrl-C leaves a
# truncated archive with no resume:
#   nohup bash scripts/tar_expert382.sh > tar382.log 2>&1 &
tar -chvf "$OUT" \
    unc_basketball_03-30-23_02_2/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_2/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_2/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_2/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_2/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_2/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_3/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_3/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_3/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_3/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_3/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_3/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_4/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_4/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_4/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_4/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_4/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_4/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_5/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_5/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_5/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_5/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_5/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_5/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_6/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_6/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_6/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_6/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_6/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_6/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_7/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_7/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_7/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_7/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_7/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_7/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_8/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_8/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_8/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_8/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_8/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_8/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_9/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_9/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_9/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_9/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_9/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_9/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_10/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_10/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_10/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_10/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_10/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_10/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_11/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_11/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_11/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_11/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_11/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_11/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_12/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_12/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_12/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_12/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_12/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_12/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_13/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_13/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_13/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_13/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_13/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_13/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_14/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_14/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_14/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_14/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_14/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_14/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_15/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_15/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_15/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_15/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_15/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_15/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_16/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_16/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_16/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_16/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_16/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_16/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_17/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_17/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_17/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_17/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_17/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_17/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_18/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_18/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_18/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_18/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_18/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_18/trajectory/gopro_calibs.csv \
    unc_basketball_03-30-23_02_19/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-30-23_02_19/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-30-23_02_19/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-30-23_02_19/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-30-23_02_19/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-30-23_02_19/trajectory/gopro_calibs.csv

echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "expected: 270 entries (18 takes x 15: four 4K cams, the 448 dir record,"
echo "          nine files inside it, and gopro_calibs.csv)"
echo ""
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
