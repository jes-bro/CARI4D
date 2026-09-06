#!/bin/bash
# Pack every video take of EgoExo4D participant 388, plus each take's fisheye
# calibration. Sibling of scripts/tar_expert387.sh; same shape, different person.
#
# WHO THIS IS. Participant 388 is the second Late Expert in the SAME capture as
# 387 -- unc_basketball_03-31-23_02, capture_uid
# 77c153c0-b4e0-4d98-aa7d-9bfc28db53c7. Both labels come from EgoExo4D's
# proficiency_demonstrator_{train,val}.json joined to takes.json on take_uid.
# Same capture means the exo rig was never moved between them, so 388 reuses
# 387's calibration and pipeline camera outright: no new camera work.
#
# WHY THE INDICES JUMP. 388 has 16 takes at _23, _25 through _27, and _29
# through _40 -- NOT a contiguous range. _24, _28 and the capture's _21, _42,
# _43, _45 carry a null participant_uid in takes.json: unattributed, so they
# are not his and are not packed. A seq range would have swept them in.
#
# _23 IS FOUR MINUTES (250.6 s) against ~50 s for every other take -- 29% of
# his footage in one file. Expect it to dominate both the archive and whatever
# masking it later.
#
# WHAT EACH TAKE CONTRIBUTES:
#   frame_aligned_videos/cam0N.mp4         the four 4K exo views
#   frame_aligned_videos/downscaled/448    the pipeline-resolution copies (dir)
#   trajectory/gopro_calibs.csv            per-take fisheye calibration
#
# All 16 were verified complete on the mirror before this was written: four exo
# cams, nine files under downscaled/448, calibration present.
#
# Paths inside the archive are relative to the takes root, so the recipient
# runs `tar -xf expert388_takes.tar -C /their/egoexo4d/takes` and their
# TAKES_ROOT works unchanged.
#
# NOTE FOR WHOEVER RECONSTRUCTS THESE: cam04, same as 387 -- not the cam01 that
# EgoExo4D's best_exo field reports. One calibration covers the whole capture,
# both people included.
#
#   bash scripts/tar_expert388.sh                       # -> <repo>/expert388_takes.tar
#   OUT=/scratch/handoff/expert388.tar bash scripts/tar_expert388.sh
#
# The archive is ~7 GB. It lands in the checkout, is untracked, and the repo
# has no .gitignore -- do not `git add -A` while it is there, and point OUT at
# scratch if the checkout is on a quota.

set -euo pipefail

# Resolved from the script's own location, not the working directory: the tar
# runs from inside TAKES_ROOT (that is what makes the archive paths relative),
# so anything derived from `pwd` would land the archive in the middle of the
# read-only mirror.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAKES_ROOT="${TAKES_ROOT:-/vision/group/egoexo4d/takes}"
OUT="${OUT:-$REPO/expert388_takes.tar}"

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
tar -chvf "$OUT" \
    unc_basketball_03-31-23_02_23/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_23/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_23/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_23/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_23/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_23/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_25/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_25/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_25/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_25/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_25/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_25/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_26/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_26/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_26/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_26/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_26/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_26/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_27/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_27/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_27/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_27/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_27/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_27/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_29/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_29/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_29/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_29/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_29/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_29/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_30/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_30/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_30/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_30/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_30/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_30/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_31/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_31/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_31/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_31/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_31/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_31/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_32/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_32/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_32/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_32/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_32/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_32/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_33/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_33/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_33/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_33/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_33/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_33/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_34/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_34/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_34/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_34/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_34/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_34/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_35/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_35/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_35/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_35/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_35/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_35/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_36/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_36/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_36/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_36/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_36/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_36/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_37/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_37/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_37/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_37/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_37/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_37/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_38/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_38/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_38/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_38/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_38/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_38/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_39/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_39/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_39/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_39/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_39/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_39/trajectory/gopro_calibs.csv \
    unc_basketball_03-31-23_02_40/frame_aligned_videos/cam01.mp4 \
    unc_basketball_03-31-23_02_40/frame_aligned_videos/cam02.mp4 \
    unc_basketball_03-31-23_02_40/frame_aligned_videos/cam03.mp4 \
    unc_basketball_03-31-23_02_40/frame_aligned_videos/cam04.mp4 \
    unc_basketball_03-31-23_02_40/frame_aligned_videos/downscaled/448 \
    unc_basketball_03-31-23_02_40/trajectory/gopro_calibs.csv


echo ""
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "entries: $(tar -tf "$OUT" | wc -l)"
echo ""
echo "checksum it before sending:"
echo "  sha256sum $OUT | tee $OUT.sha256"
echo "extract on the other side with:"
echo "  tar -xf $(basename "$OUT") -C /their/egoexo4d/takes"
