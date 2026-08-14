#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

#SBATCH --job-name="trim-clips"
#SBATCH --output=trim-clips-%j.out

#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=ALL

# Cut frame-accurate window clips from a take's exo videos, for SAM3 masking
# and triangulation. CPU-only; exists because a 4K decode gets OOM-killed on
# the login node.
#
# Frame accuracy is the entire point: the clips are consumed by
# prep/triangulate_object.py, where a +-1 frame slip between views silently
# corrupts the geometry. So frames are selected by index with the select
# filter (no keyframe seeking), and every output is frame-counted afterwards
# against the expected window length -- the job FAILS loudly on a mismatch
# rather than leaving a subtly wrong clip behind.
#
#   START=354 END=454 \
#   SRC_DIR=/vision/group/egoexo4d/takes/<take>/frame_aligned_videos \
#   OUT_DIR=sam3masks/trimmed_vids SUFFIX=-4k CAMS="cam01 cam02 cam03 cam04" \
#       sbatch scripts/slurm_trim_clips.sh
#
# Output naming: <OUT_DIR>/<cam><SUFFIX>.0.color.mp4 -- the .0.color.mp4
# convention is what run_sam3_masks.py parses the sequence name from.

set -euo pipefail

SRC_DIR=${SRC_DIR:?set SRC_DIR to the take frame_aligned_videos dir}
START=${START:?set START to the first take frame of the window}
END=${END:?set END to the last take frame of the window (inclusive)}
OUT_DIR=${OUT_DIR:-sam3masks/trimmed_vids}
SUFFIX=${SUFFIX:--4k}
CAMS=${CAMS:-cam01 cam02 cam03 cam04}

REPO=/simurgh2/projects/ret-hoi/CARI4D
cd "$REPO"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CARI4D_ENV:-newcari4d}"

mkdir -p "$OUT_DIR"
n_expect=$((END - START + 1))
echo "[trim] window ${START}-${END} (${n_expect} frames) from ${SRC_DIR}"

rc=0
for c in $CAMS; do
    src="$SRC_DIR/$c.mp4"
    out="$OUT_DIR/$c$SUFFIX.0.color.mp4"
    if [ ! -f "$src" ]; then
        echo "[trim] ERROR: no source video at $src" >&2
        rc=1
        continue
    fi
    # -frames:v stops the decode as soon as the window is written instead of
    # chewing through the rest of the take.
    ffmpeg -hide_banner -loglevel error -y -i "$src" \
        -vf "select='between(n\,${START}\,${END})',setpts=N/FRAME_RATE/TB" \
        -fps_mode vfr -frames:v "$n_expect" -crf 18 -an "$out"
    n_got=$(ffprobe -v error -count_frames -select_streams v \
        -show_entries stream=nb_read_frames -of csv=p=0 "$out")
    if [ "$n_got" != "$n_expect" ]; then
        echo "[trim] ERROR: $out has $n_got frames, expected $n_expect" >&2
        rc=1
    else
        echo "[trim] $out  ok ($n_got frames)"
    fi
done
exit $rc
