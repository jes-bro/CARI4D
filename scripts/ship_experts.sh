#!/bin/bash
# Pack, ship and delete each subject's archive in turn.
#
# The 173 tar_expert*.sh scripts come to ~495 GB. Packing them all and then
# transferring is not a plan: it needs half a terabyte of scratch that nobody
# has, and it front-loads hours of tar before a single byte moves. This does
# one subject at a time -- pack, rsync, verify, delete -- so peak local disk is
# ONE archive (the largest is participant 50 at ~19 GB) no matter how many
# subjects you ship.
#
#   DEST=kk@ikura:/data/egoexo/handoff bash scripts/ship_experts.sh 387 388 383
#   DEST=... bash scripts/ship_experts.sh --file splits/to-ship.txt
#   DEST=... KEEP=1 bash scripts/ship_experts.sh 387      # keep the local tar
#   DRY_RUN=1 DEST=... bash scripts/ship_experts.sh 387   # print the plan
#
# RESUMABLE IN TWO SENSES. rsync --partial --append-verify picks a killed
# transfer back up mid-file rather than restarting it, which is the whole
# reason to prefer this over wormhole at these sizes. And a subject whose
# marker file exists is skipped, so re-running the same list after an
# interruption continues where it stopped instead of redoing what landed.
#
# The local archive is deleted ONLY after rsync exits 0. A failed transfer
# leaves the tar in place so the next run resumes it rather than re-packing.

set -uo pipefail
cd "$(dirname "$0")/.."

DEST="${DEST:?set DEST to an rsync target, e.g. user@host:/path/to/handoff}"
KEEP="${KEEP:-}"
DRY_RUN="${DRY_RUN:-}"
STATE="${STATE:-.shipped}"
mkdir -p "$STATE"

if [ "${1:-}" = "--file" ]; then
    [ -f "${2:-}" ] || { echo "ERROR: no such list: ${2:-}" >&2; exit 1; }
    # '#' comments and blank lines ignored, so a list can carry notes.
    IDS=$(sed 's/#.*//' "$2" | tr -s '[:space:]' '\n' | grep -E '^[0-9]+$' || true)
else
    IDS="$*"
fi
[ -n "${IDS// /}" ] || { echo "usage: DEST=... bash scripts/ship_experts.sh <pid>... | --file <list>" >&2; exit 1; }

total=0; done_n=0; failed=()
for pid in $IDS; do total=$((total + 1)); done
echo "shipping $total subject(s) to $DEST"
echo ""

for pid in $IDS; do
    script="scripts/tar_expert${pid}.sh"
    tarball="expert${pid}_takes.tar"
    marker="$STATE/$pid"

    if [ ! -f "$script" ]; then
        echo "== $pid  SKIP: no $script"; failed+=("$pid:no-script"); continue
    fi
    if [ -f "$marker" ]; then
        echo "== $pid  already shipped ($(cat "$marker"))"; done_n=$((done_n + 1)); continue
    fi

    echo "== $pid"
    if [ -n "$DRY_RUN" ]; then
        echo "   would: bash $script"
        echo "   would: rsync -aP --partial --append-verify $tarball $DEST/"
        echo "   would: rm $tarball  (unless KEEP=1)"
        continue
    fi

    # Re-pack only when there is no archive to resume. A tar left by a failed
    # transfer is complete -- it was written before rsync ran -- so re-making
    # it would waste the one expensive step this script exists to do once.
    if [ -f "$tarball" ]; then
        echo "   $tarball already exists, reusing it"
    else
        echo "   packing..."
        if ! bash "$script" > "tar${pid}.log" 2>&1; then
            echo "   PACK FAILED, see tar${pid}.log:"; tail -3 "tar${pid}.log"
            failed+=("$pid:pack"); continue
        fi
    fi

    entries=$(tar -tf "$tarball" 2>/dev/null | wc -l)
    if [ "$entries" -lt 1 ]; then
        echo "   EMPTY OR UNREADABLE ARCHIVE ($entries entries), not shipping"
        failed+=("$pid:empty"); continue
    fi
    echo "   $entries entries, $(du -h "$tarball" | cut -f1)"

    echo "   rsync..."
    if rsync -aP --partial --append-verify "$tarball" "$DEST/"; then
        # Recorded before the delete, so an interrupted run never re-ships.
        echo "$(date -u '+%Y-%m-%d %H:%M UTC')  $entries entries" > "$marker"
        [ -n "$KEEP" ] || rm -f "$tarball"
        echo "   done${KEEP:+ (kept locally)}"
        done_n=$((done_n + 1))
    else
        echo "   RSYNC FAILED (rc=$?). The tar is kept so the next run resumes it."
        failed+=("$pid:rsync")
    fi
    echo ""
done

echo "shipped $done_n of $total"
if [ ${#failed[@]} -gt 0 ]; then
    echo "failed: ${failed[*]}"
    echo "re-run the same command -- finished subjects are skipped, partial"
    echo "transfers resume mid-file."
    exit 1
fi
