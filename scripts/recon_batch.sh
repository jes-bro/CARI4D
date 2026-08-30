#!/bin/bash
# Run one recon stage over every take in a manifest.
#
# The stages are deliberately not chained across the batch: you run `masks` for
# all of them, look at the mask videos, run `geometry` for the ones that
# survived, look at the residuals, then run `solve`. That is the whole point of
# the split -- a batch that ran end to end would spend its GPU hours before
# anyone saw a mask.
#
#   bash scripts/recon_batch.sh masks
#   bash scripts/recon_batch.sh geometry splits/layup-batch.tsv
#   ONLY=387 bash scripts/recon_batch.sh solve
#   DRY_RUN=1 bash scripts/recon_batch.sh masks        # print the whole plan
#
# ONLY filters by participant column; SKIP_EXISTING=1 skips a sequence whose
# work directory already has the stage's main output, so a partly-failed batch
# can be re-run without redoing what worked.
#
# Manifest format: tab-separated `take  seq  participant  drill  duration`,
# `#` comments ignored. splits/layup-batch.tsv is the layup set.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/recon_common.sh

STAGE=${1:?usage: bash scripts/recon_batch.sh <masks|geometry|solve> [manifest]}
MANIFEST=${2:-splits/layup-batch.tsv}

case "$STAGE" in
    check|clips|masks|geometry|solve) ;;
    *) echo "ERROR: stage must be check, clips, masks, geometry or solve (got '$STAGE')" >&2; exit 1 ;;
esac

# check and clips act on a TAKE, which is what a manifest row is. Everything
# after stage 1a acts on a CLIP, and one take yields several -- so those stages
# expand each row into the clips that take actually produced.
case "$STAGE" in
    check|clips) PER_CLIP="" ;;
    *)           PER_CLIP=1 ;;
esac

clips_of() {
    # Print the clip sequence names stage 1a emitted for the take in $WORK.
    #
    # Reads the clip list rather than globbing directory names, so a half-built
    # or hand-made directory is never mistaken for an emitted clip.
    local list="$WORK/clips.json"
    [ -f "$list" ] || return 0
    python3 -c "import json,sys; print('\n'.join(c['seq'] for c in json.load(open(sys.argv[1]))['clips']))" "$list"
}
[ -f "$MANIFEST" ] || { echo "ERROR: no manifest at $MANIFEST" >&2; exit 1; }

stage_done() {
    # Whether stage $1 already has its main output for the sequence in $WORK.
    #
    # One representative artifact each, not a full check: the goal is to skip
    # work that plainly succeeded, and anything ambiguous should be re-run.
    case "$1" in
        check)    return 1 ;;   # read-only and cheap; never worth skipping
        masks)    [ -f "$WORK/window.json" ] ;;
        geometry) [ -f "$WORK/geom/object_xyz.npz" ] && [ -f "$WORK/rect/$SEQ.0.color.mp4" ] ;;
        solve)    compgen -G "output/opt/*/$SEQ.pth" >/dev/null ;;
    esac
}

n=0
while IFS=$'\t' read -r take seq participant drill duration pipe_cam; do
    case "$take" in ''|'#'*) continue ;; esac
    if [ -n "${ONLY:-}" ] && [ "$participant" != "$ONLY" ]; then
        continue
    fi
    export TAKE="$take" SEQ="$seq"
    # The manifest's pipeline camera wins, since it is a per-capture property;
    # a PIPE_CAM the caller actually typed still overrides it. recon_paths then
    # re-derives the aux list from whichever won.
    if [ -z "$PIPE_CAM_EXPLICIT" ] && [ -n "$pipe_cam" ]; then
        PIPE_CAM="$pipe_cam"
    fi
    recon_paths
    if [ -n "${SKIP_EXISTING:-}" ] && stage_done "$STAGE"; then
        echo "== skip $seq ($STAGE already done)"
        continue
    fi
    if [ -z "$PER_CLIP" ]; then
        # check prints its own one-line verdict; a header would bury it.
        [ "$STAGE" = check ] || echo "== $seq  <- $take  (participant $participant,$drill,${duration}s)"
        bash "scripts/recon_$STAGE.sh"
        n=$((n + 1))
        continue
    fi

    base_seq="$seq"
    mapfile -t clips < <(clips_of)
    if [ ${#clips[@]} -eq 0 ]; then
        echo "== skip $base_seq (no clips yet -- run stage 'clips' first)"
        continue
    fi
    for clip in "${clips[@]}"; do
        export SEQ="$clip"
        recon_paths
        if [ -n "${SKIP_EXISTING:-}" ] && stage_done "$STAGE"; then
            echo "== skip $clip ($STAGE already done)"
            continue
        fi
        echo "== $clip  <- $take  (participant $participant,$drill)"
        bash "scripts/recon_$STAGE.sh"
        n=$((n + 1))
    done
done < "$MANIFEST"

echo
if [ "$STAGE" = check ]; then
    echo "checked $n sequence(s) from $MANIFEST"
else
    echo "submitted stage '$STAGE' for $n sequence(s) from $MANIFEST"
fi
