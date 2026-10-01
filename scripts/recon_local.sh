#!/bin/bash
# The no-queue backend for the recon drivers: run each job script right here.
#
# Sourced by recon_common.sh when RECON_BACKEND=local, or when there is no
# sbatch on the machine. recon_sbatch() then hands its arguments to
# recon_local_submit() instead of sbatch, and every job runs in the FOREGROUND,
# in submission order, with its output in this terminal. That is the mode for
# debugging a new kind of take inside an interactive GPU allocation
# (srun --pty bash): you watch SAM3 find or lose the object as it happens,
# rather than reading a .out file an hour later.
#
# Jobs that Slurm would run in parallel -- the aux views' SAM3, Sapiens per
# view -- run one after another. On one GPU that is what would happen anyway.
# --dependency arguments are satisfied by construction and ignored, as are
# --time, --job-name, --exclude and the rest of the sbatch vocabulary.
#
# The "job id" echoed is the word LOCAL. recon_dep() and recon_wait() both
# skip it, so the drivers chain exactly as they do under Slurm, minus the wait.
#
#   RECON_BACKEND=local TAKE=... SEQ=... bash scripts/recon_pilot.sh
#
# A job that fails stops the driver at that point with the script's own error
# on screen, which under set -e is the same place a --dependency=afterok chain
# would have stopped, only immediately.

recon_local_submit() {
    # Run the job script among $@ in the foreground; echo LOCAL as its id.
    #
    # The arguments are an sbatch command line: options first, then the
    # script, then the script's own arguments. Options are dropped, the rest
    # is executed with bash. Output goes to stderr so the id on stdout stays
    # parseable by the callers that capture it.
    local script="" args=()
    for a in "$@"; do
        if [ -z "$script" ]; then
            case "$a" in
                --*) continue ;;
                *.sh) script="$a" ;;
                *) echo "[local] unexpected argument before the script: $a" >&2; return 1 ;;
            esac
        else
            args+=("$a")
        fi
    done
    [ -n "$script" ] || { echo "[local] no job script in: $*" >&2; return 1; }
    echo "[local] running $script ${args[*]:-}" >&2
    # REPO is exported by recon_common.sh, so the job's own fallback to
    # SLURM_SUBMIT_DIR never fires. The script's exit status is the job's.
    bash "$script" ${args[@]+"${args[@]}"} >&2 || {
        echo "[local] $script exited $?" >&2
        return 1
    }
    echo "LOCAL"
}
