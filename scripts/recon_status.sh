#!/bin/bash
# Where every clip stands, and the one command each needs next.
#
#   bash scripts/recon_status.sh
#   bash scripts/recon_status.sh rev009        # only clips matching a pattern
#
# State is read off the filesystem rather than remembered, so it is right after
# a cancelled job, a closed terminal or a week away. Each stage also leaves its
# own notes in <work>/NEXT.txt, but nobody wants to hunt for a path per clip --
# this prints the lot.
#
# The columns are the artifacts each stage produces:
#
#   clip     the clip's own masks and trimmed video, from recon_clips.sh
#   aux      the other cameras masked, from recon_masks.sh
#   geom     triangulated object and human, plus the rectified clip
#   mesh     the object's reconstructed shape
#   metric   that mesh at its measured size
#   result   the reconstruction itself

set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/recon_common.sh 2>/dev/null || true

FILTER="${1:-}"
WORK_ROOT="${WORK_ROOT:-$(pwd)/work}"
MANIFEST="${MANIFEST:-splits/layup-batch.tsv}"

take_for() {
    # The take a clip came from, by stripping the clip suffix and looking the
    # base sequence up in the manifest. Clip names are <base><letter>[t], and
    # the base is what the manifest knows.
    local seq="$1" base
    base="$(echo "$seq" | sed -E 's/[a-z]+$//')"
    [ -f "$MANIFEST" ] || { echo "<take>"; return; }
    awk -F'\t' -v b="$base" '!/^#/ && $2==b {print $1; found=1; exit}
                             END {if (!found) print "<take>"}' "$MANIFEST"
}

mark() { [ -e "$1" ] && printf '  yes ' || printf '   -  '; }

printf '%-34s %-6s %-6s %-6s %-6s %-7s %-7s\n' \
    "clip" "clip" "aux" "geom" "mesh" "metric" "result"
printf '%s\n' "--------------------------------------------------------------------------------"

shopt -s nullglob
for d in "$WORK_ROOT"/*/; do
    d="${d%/}"          # no trailing slash: these paths get copy-pasted
    seq="$(basename "$d")"
    [ -n "$FILTER" ] && [[ "$seq" != *"$FILTER"* ]] && continue
    # A take-level directory holds clips.json and is not itself a clip.
    [ -f "$d/clips.json" ] && continue

    has_clip="$d/masks/${seq}_masks_k0.h5"
    has_aux="$(echo "$d"/masks/cam*-4k_masks_k0.h5 | cut -d' ' -f1)"
    has_geom="$d/geom/object_xyz.npz"
    has_mesh="$(echo "$d"/meshes/*/*_align.obj | cut -d' ' -f1)"
    has_metric="$(echo "$d"/meshes-metric/*/*_align.obj | cut -d' ' -f1)"
    has_result="$(echo output/opt/*/"$seq".pth | cut -d' ' -f1)"

    printf '%-34s' "$seq"
    for p in "$has_clip" "$has_aux" "$has_geom" "$has_mesh" "$has_metric" "$has_result"; do
        mark "$p"
    done
    echo
done

echo
echo
echo "WHAT TO RUN NEXT"

for d in "$WORK_ROOT"/*/; do
    d="${d%/}"          # no trailing slash: these paths get copy-pasted
    seq="$(basename "$d")"
    [ -n "$FILTER" ] && [[ "$seq" != *"$FILTER"* ]] && continue
    [ -f "$d/clips.json" ] && continue
    take="$(take_for "$seq")"
    # Relative, because every command here is meant to be run from the repo
    # root and the absolute form wraps across two lines.
    rel="${d#$(pwd)/}"

    echo
    echo "──────────────────────────────────────────────────────────────────"
    printf '  %s\n' "$seq"
    printf '  take: %s\n' "$take"
    echo

    if [ ! -e "$d/masks/${seq}_masks_k0.h5" ]; then
        echo "  incomplete -- this clip has no masks of its own."
        echo "  Re-run recon_clips.sh for its take."
    elif [ ! -e "$(echo "$d"/masks/cam*-4k_masks_k0.h5 | cut -d' ' -f1)" ]; then
        echo "  needs: the other cameras masked"
        echo
        echo "      TAKE=$take SEQ=$seq bash scripts/recon_masks.sh"
    elif [ ! -e "$d/geom/object_xyz.npz" ]; then
        echo "  needs: geometry -- but check coverage first"
        echo
        echo "      python prep/check_view_coverage.py \\"
        echo "          --masks_root $rel/masks --seq $seq"
        echo
        echo "  then, if it says every frame is triangulatable:"
        echo
        echo "      TAKE=$take SEQ=$seq bash scripts/recon_geometry.sh"
        echo
        echo "  if it reports a shorter usable run, cut the clip down first:"
        echo
        echo "      python prep/retrim_clip.py --work $rel --lo <first> --hi <last>"
        echo "      (that makes ${seq}t -- use the new name from then on)"
    elif [ ! -e "$(echo "$d"/meshes/*/*_align.obj | cut -d' ' -f1)" ]; then
        echo "  needs: the object reconstructed"
        echo
        echo "      python prep/pick_object_frame.py --work $rel"
        echo
        echo "  then run the MESH_CAM/MESH_FRAME command it prints."
    elif [ ! -e "$(echo output/opt/*/"$seq".pth | cut -d' ' -f1)" ]; then
        echo "  needs: the reconstruction"
        echo
        echo "      TAKE=$take SEQ=$seq bash scripts/recon_solve.sh"
    else
        echo "  DONE."
        echo
        echo "  Watch it:      ls -lat output/viz-pred/ | head -3"
        echo "  Object size:   grep -A4 'resulting object size' \$(ls -t recon-scale-*.out | head -1)"
    fi
done
echo
