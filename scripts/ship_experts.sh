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
#   XFER=rclone DEST=gdrive:cari4d-handoff bash scripts/ship_experts.sh 387
#   DEST=... bash scripts/ship_experts.sh --file splits/to-ship.txt
#   DEST=... DELETE_AFTER=1 bash scripts/ship_experts.sh 387   # free disk as it goes
#   DRY_RUN=1 DEST=... bash scripts/ship_experts.sh 387   # print the plan
#
# RESUMABLE IN TWO SENSES. rsync --partial --append-verify picks a killed
# transfer back up mid-file rather than restarting it, which is the whole
# reason to prefer this over wormhole at these sizes. And a subject whose
# marker file exists is skipped, so re-running the same list after an
# interruption continues where it stopped instead of redoing what landed.
#
# With DELETE_AFTER=1 the local archive is removed ONLY after the transfer
# exits 0, and only that archive. A failed transfer always leaves the tar in
# place so the next run resumes it rather than re-packing.
#
# WATCH YOUR DISK without it: 173 subjects is ~495 GB, and by default they all
# stay. Ship in batches, or pass DELETE_AFTER=1 once you trust the uploads.

set -uo pipefail
cd "$(dirname "$0")/.."

DEST="${DEST:?set DEST to an rsync target (user@host:/path) or an rclone remote (gdrive:folder)}"
# rsync for a machine you can reach, rclone for object storage -- Google Drive
# most likely, where the web UI and Drive for Desktop both give up on a 19 GB
# file and rclone does chunked, resumable, verified uploads instead.
XFER="${XFER:-rsync}"
case "$XFER" in rsync|rclone) ;; *) echo "ERROR: XFER must be rsync or rclone" >&2; exit 1 ;; esac
command -v "$XFER" >/dev/null 2>&1 || { echo "ERROR: $XFER is not installed" >&2; exit 1; }
# Grouped by scenario, because the object changes with it and so do the SAM3
# prompts and the FoundationPose knobs -- a folder per scenario is the unit
# somebody actually works through. Read from the script's own header rather
# than a table here, so it cannot disagree with what was generated.
GROUP="${GROUP:-1}"
# NOTHING IS DELETED unless you ask. This used to default the other way --
# pack, ship, remove -- which kept peak disk at one archive but meant the
# script quietly destroyed something after every upload. A tool that deletes
# by default is a tool you have to remember a flag to make safe, and that is
# backwards. DELETE_AFTER=1 restores the old behaviour when disk is the
# binding constraint, and even then only ever removes the archive it just
# finished uploading.
DELETE_AFTER="${DELETE_AFTER:-}"
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

    # Scenario from takes.json via the first take the script names, not from
    # the script's own prose: the hand-written ones say "video take" where the
    # generated ones say "basketball take", and a folder called video/ helps
    # nobody. Empty when it cannot be resolved, which just means no subfolder.
    sub=""
    if [ -n "$GROUP" ]; then
        sub=$(python3 -c "
import json, os, re, sys
txt = open(sys.argv[1]).read()
m = re.search(r'^    (\S+?)/', txt, re.M)
if not m: raise SystemExit
for p in (os.environ.get('EGOEXO_TAKES_JSON'),
          os.path.join(os.path.dirname(os.environ.get('TAKES_ROOT','').rstrip('/')), 'takes.json'),
          os.path.expanduser('~/egoexo4d/takes.json')):
    if p and os.path.isfile(p):
        for t in json.load(open(p)):
            if t['take_name'] == m.group(1):
                s = (t.get('parent_task_name') or '').lower().replace(' ', '-')
                print({'health': 'cpr'}.get(s, s))
                raise SystemExit
" "$script" 2>/dev/null)
        [ -n "$sub" ] && sub="${sub}/"
    fi

    echo "== $pid  ${sub:-(ungrouped)}"
    if [ -n "$DRY_RUN" ]; then
        echo "   would: bash $script"
        if [ "$XFER" = rclone ]; then
            echo "   would: rclone copyto $tarball $DEST/${sub}$tarball"
        else
            echo "   would: rsync -aP --partial --append-verify $tarball $DEST/$sub"
        fi
        if [ -n "$DELETE_AFTER" ]; then
            echo "   would: rm $tarball   (DELETE_AFTER=1)"
        else
            echo "   would KEEP $tarball  (pass DELETE_AFTER=1 to remove it)"
        fi
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

    echo "   uploading with $XFER..."
    if [ "$XFER" = rclone ]; then
        # --drive-chunk-size trades memory for throughput; 128M is the usual
        # sweet spot. rclone verifies the upload itself, so a clean exit is
        # the check -- there is no partial-file state to resume on Drive, but
        # a re-run re-uploads only what is missing.
        xfer_cmd=(rclone copyto --progress --drive-chunk-size 128M
                  --retries 5 --low-level-retries 20 "$tarball" "$DEST/${sub}$tarball")
    else
        xfer_cmd=(rsync -aP --partial --append-verify --mkpath "$tarball" "$DEST/$sub")
    fi
    if "${xfer_cmd[@]}"; then
        # Recorded before the delete, so an interrupted run never re-ships.
        echo "$(date -u '+%Y-%m-%d %H:%M UTC')  $entries entries" > "$marker"
        if [ -n "$DELETE_AFTER" ]; then
            rm -f "$tarball"
            echo "   done (local archive deleted, DELETE_AFTER=1)"
        else
            echo "   done (local archive kept at $tarball)"
        fi
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
