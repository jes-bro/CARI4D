#!/bin/bash
# Pack every video take of EgoExo4D participant 387 -- the person in the
# reconstruction we already have -- plus each take's fisheye calibration.
#
# WHO THIS IS. Date03_Sub01_bball_dribble was cut from take
# unc_basketball_03-31-23_02_9 (cam04, frames 354-454). That take's record in
# takes.json says participant_uid 387, capture_uid
# 77c153c0-b4e0-4d98-aa7d-9bfc28db53c7. Filtering takes.json on that
# participant returns exactly the 18 takes below -- indices _2 through _19, no
# gaps, all in that one capture. _20 onward is participant 388, and _21, _42,
# _43, _45 have a null participant_uid (unattributed, so left out).
#
# The paths are written out one by one rather than generated, so what gets
# packed is readable here and cannot drift from the list that was checked.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the four 4K exo views
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# Paths inside the archive are relative to the takes root, so the recipient
# runs `tar -xf expert387_takes.tar -C /their/egoexo4d/takes` and their
# TAKES_ROOT works unchanged.
#
# NOTE FOR WHOEVER RECONSTRUCTS THESE: this capture uses cam04, not the cam01
# that EgoExo4D's best_exo field reports. The exo rig is fixed within a
# capture, so one camera choice and one calibration cover all 18 takes.
#
#   bash scripts/tar_expert387.sh
#   OUT=/scratch/handoff/expert387.tar bash scripts/tar_expert387.sh

set -euo pipefail

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$HOME/expert387/expert387_takes.tar}"

[ -d "$TAKES_ROOT" ] || { echo "ERROR: no takes root at $TAKES_ROOT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
cd "$TAKES_ROOT"

# -c create, -h follow symlinks (the mirror uses them in places), -v so the
# terminal shows every file that went in, -f the archive. No -z: the payload is
# already-encoded H.264, so compression costs minutes and saves nothing.
tar -chvf "$OUT" \
    unc_basketball_03-31-23_02_2/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_2/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_2/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_2/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_2/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_2/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_3/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_3/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_3/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_3/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_3/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_3/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_4/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_4/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_4/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_4/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_4/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_4/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_5/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_5/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_5/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_5/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_5/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_5/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_6/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_6/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_6/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_6/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_6/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_6/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_7/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_7/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_7/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_7/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_7/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_7/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_8/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_8/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_8/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_8/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_8/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_8/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_9/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_9/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_9/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_9/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_9/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_9/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_10/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_10/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_10/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_10/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_10/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_10/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_11/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_11/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_11/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_11/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_11/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_11/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_12/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_12/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_12/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_12/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_12/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_12/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_13/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_13/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_13/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_13/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_13/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_13/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_14/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_14/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_14/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_14/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_14/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_14/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_15/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_15/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_15/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_15/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_15/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_15/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_16/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_16/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_16/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_16/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_16/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_16/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_17/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_17/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_17/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_17/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_17/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_17/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_18/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_18/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_18/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_18/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_18/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_18/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_19/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_19/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_19/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_19/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_19/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_19/trajectory/gopro_calibs.csv


echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "checksum it before sending:"
echo "  sha256sum $OUT | tee $OUT.sha256"
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
