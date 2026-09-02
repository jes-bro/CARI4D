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
# Step numbers match the written instructions:
#   1 recon_clips   2 look at the clips     3 recon_masks    4 check coverage
#   5 recon_geometry  6 pick_object_frame   7 recon_object   8 check both
#   9 recon_solve   10 watch the render    11 replay in isaacgym
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

clip_window() {
    # "take frames 193-299  (107 frames, 3.6 s)" from the clip's own window.json.
    # A clip name says which take and which drill but nothing about WHICH three
    # seconds, and that is the thing you need when deciding whether to keep it.
    python3 -c "
import json, sys
try:
    w = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
c = w.get('chosen')
if not c:
    sys.exit(0)
n, fps = c['n_frames'], w.get('fps', 30.0) or 30.0
print(f\"{c['lo']}-{c['hi']} of the take  ({n} frames, {n/fps:.1f} s)\")
" "$1" 2>/dev/null
}

take_dir_for() {
    # The take directory a clip came from, resolved off the filesystem.
    #
    # Stage 1a symlinks the pipeline video into the take, so the answer is
    # already on disk and does not depend on the clip appearing in any split
    # file -- which matters the moment this is used on something that is not a
    # basketball. The symlink lives in the take-level directory, so a clip name
    # <base><letter>[t] has its trailing letters stripped to find it.
    local seq="$1" d="$2" base t f
    base="$(echo "$seq" | sed -E 's/[a-z]+$//')"
    for cand in "$d/src" "$WORK_ROOT/$base/src"; do
        [ -d "$cand" ] || continue
        for f in "$cand"/*; do
            t="$(readlink -f "$f" 2>/dev/null)" || continue
            case "$t" in
                */frame_aligned_videos/*) echo "${t%%/frame_aligned_videos/*}"; return ;;
            esac
        done
    done
}

take_for() {
    # The take's name: from the symlink if it is there, from the manifest if
    # not. The manifest only knows this batch, so it is the fallback.
    local seq="$1" d="$2" base tdir
    tdir="$(take_dir_for "$seq" "$d")"
    [ -n "$tdir" ] && { basename "$tdir"; return; }
    base="$(echo "$seq" | sed -E 's/[a-z]+$//')"
    [ -f "$MANIFEST" ] || { echo "<take>"; return; }
    awk -F'\t' -v b="$base" '!/^#/ && $2==b {print $1; found=1; exit}
                             END {if (!found) print "<take>"}' "$MANIFEST"
}

aux_expected() {
    # How many aux views this take should yield: every exo camera it has, minus
    # the one driving the pipeline. Counted from the take rather than assumed,
    # because not every capture has four.
    local tdir="$1" n
    [ -n "$tdir" ] || { echo 0; return; }
    n="$(ls "$tdir"/frame_aligned_videos/cam*.mp4 2>/dev/null | wc -l)"
    [ "$n" -gt 0 ] && echo $((n - 1)) || echo 0
}

pipe_cam_for() {
    # Which camera drives the pipeline, from the same symlink that names the
    # take: its target is .../downscaled/448/<cam>.mp4. Resolved rather than
    # assumed, since PIPE_CAM varies by capture.
    local seq="$1" d="$2" base t f
    base="$(echo "$seq" | sed -E 's/[a-z]+$//')"
    for cand in "$d/src" "$WORK_ROOT/$base/src"; do
        [ -d "$cand" ] || continue
        for f in "$cand"/*; do
            t="$(readlink -f "$f" 2>/dev/null)" || continue
            case "$t" in
                */frame_aligned_videos/*) basename "$t" .mp4; return ;;
            esac
        done
    done
}

aux_missing() {
    # Which aux cameras have no mask file. A partially-masked clip used to read
    # as fully masked here, because the check was "does any cam*-4k mask
    # exist" -- and a silently missing camera is the usual reason coverage
    # later looks bad.
    local seq="$1" tdir="$2" d="$3" pipe c out=""
    [ -n "$tdir" ] || return
    pipe="$(pipe_cam_for "$seq" "$d")"
    for f in "$tdir"/frame_aligned_videos/cam*.mp4; do
        c="$(basename "$f" .mp4)"
        [ "$c" = "$pipe" ] && continue
        [ -e "$d/masks/${c}-4k_masks_k0.h5" ] || out="${out:+$out }$c"
    done
    echo "$out"
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
    has_geom="$d/geom/object_xyz.npz"
    has_mesh="$(echo "$d"/meshes/*/*_align.obj | cut -d' ' -f1)"
    has_metric="$(echo "$d"/meshes-metric/*/*_align.obj | cut -d' ' -f1)"
    has_result="$(echo output/opt/*/"$seq".pth | cut -d' ' -f1)"

    # A count, not a yes: "2/3" is the case worth seeing, and it used to print
    # as "yes" because one existing mask satisfied the check.
    tdir="$(take_dir_for "$seq" "$d")"
    n_aux="$(ls "$d"/masks/cam*-4k_masks_k0.h5 2>/dev/null | wc -l)"
    n_want="$(aux_expected "$tdir")"
    if [ "$n_aux" -eq 0 ]; then aux_col="  -  "
    elif [ "$n_want" -gt 0 ]; then aux_col=" $n_aux/$n_want "
    else aux_col="  $n_aux  "; fi

    printf '%-34s' "$seq"
    mark "$has_clip"
    printf '%-6s' "$aux_col"
    for p in "$has_geom" "$has_mesh" "$has_metric" "$has_result"; do
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
    take="$(take_for "$seq" "$d")"
    # Relative, because every command here is meant to be run from the repo
    # root and the absolute form wraps across two lines.
    rel="${d#$(pwd)/}"
    missing="$(aux_missing "$seq" "$(take_dir_for "$seq" "$d")" "$d")"

    echo
    echo "──────────────────────────────────────────────────────────────────"
    # Every line labelled, including the name -- and the name is labelled "seq"
    # because that is the variable it gets pasted into.
    printf '  seq:    %s\n' "$seq"
    printf '  take:   %s\n' "$take"
    win="$(clip_window "$d/window.json")"
    [ -n "$win" ] && printf '  frames: %s\n' "$win"
    [ -f "$d/masks/trimmed_vids/${seq}.0.color.mp4" ] && \
        printf '  watch:  %s\n' "$rel/masks/trimmed_vids/${seq}.0.color.mp4"
    echo

    if [ ! -e "$d/masks/${seq}_masks_k0.h5" ]; then
        echo "  incomplete -- this clip has no masks of its own."
        echo "  Re-run step 1 (recon_clips.sh) for its take."
    elif [ ! -e "$(echo "$d"/masks/cam*-4k_masks_k0.h5 | cut -d' ' -f1)" ]; then
        echo "  NEXT: step 3 -- mask the other cameras"
        echo
        echo "      TAKE=$take SEQ=$seq bash scripts/recon_masks.sh"
    elif [ ! -e "$d/geom/object_xyz.npz" ]; then
        # Called out before step 4, because a missing camera is the usual
        # reason step 4 reports poor coverage, and re-masking it fixes that
        # outright instead of shortening the clip to fit.
        if [ -n "$missing" ]; then
            echo "  WARNING: no mask for $missing -- that camera's job failed, or"
            echo "  it was never submitted. Fewer views means worse coverage in"
            echo "  step 4, so fix this first:"
            echo
            echo "      sacct -u \$USER --starttime now-2days --format=JobID%14,JobName%40,State,ExitCode | grep $seq"
            echo
            echo "  If it failed, re-mask just that camera (the others are kept):"
            echo
            echo "      AUX_CAMS=\"$missing\" TAKE=$take SEQ=$seq bash scripts/recon_masks.sh"
            echo
        fi
        echo "  NEXT: step 4 -- can two cameras see the object?"
        echo
        echo "      python prep/check_view_coverage.py \\"
        echo "          --masks_root $rel/masks --seq $seq"
        echo
        echo "  then step 5 -- geometry, if every frame is triangulatable:"
        echo
        echo "      TAKE=$take SEQ=$seq bash scripts/recon_geometry.sh"
        echo
        echo "  or, if step 4 reports a shorter usable run, cut the clip first:"
        echo
        echo "      python prep/retrim_clip.py --work $rel --lo <first> --hi <last>"
        echo "      (that makes ${seq}t -- use the new name from step 5 on)"
    elif [ ! -e "$(echo "$d"/meshes/*/*_align.obj | cut -d' ' -f1)" ]; then
        echo "  NEXT: step 6 -- pick the frame to reconstruct the object from"
        echo
        echo "      python prep/pick_object_frame.py --work $rel"
        echo
        echo "  then step 7 -- the MESH_CAM/MESH_FRAME command it prints."
    elif [ ! -e "$(echo output/opt/*/"$seq".pth | cut -d' ' -f1)" ]; then
        echo "  NEXT: step 9 -- the reconstruction itself"
        echo "  (step 8, the geometry and mesh checks, comes first if you have"
        echo "   not looked at them)"
        echo
        echo "      TAKE=$take SEQ=$seq bash scripts/recon_solve.sh"
    else
        echo "  DONE through step 9. Step 10 is to watch it."
        echo
        echo "  Watch it:      ls -lat output/viz-pred/ | head -3"
        echo "  Object size:   grep -A4 'resulting object size' \$(ls -t recon-scale-*.out | head -1)"
    fi
done
echo
