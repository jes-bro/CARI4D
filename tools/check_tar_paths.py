"""Check that every path a tar_expert script names exists on the takes mirror.

Run BEFORE packing a batch. tar -c with a missing path exits non-zero under
`set -e`, so one absent gopro_calibs.csv fails the whole subject after tens of
gigabytes were already written. Cheaper to ask first.

    python tools/check_tar_paths.py --file splits/ship-guitar.txt
    python tools/check_tar_paths.py 400 401 405
    TAKES_ROOT=/vision/group/egoexo4d/takes python tools/check_tar_paths.py --file ...

Prints one line per subject: OK with the path count, or MISSING with the
first few absent paths. Exit status is the number of subjects with a problem.
"""
import argparse
import os
import re
import sys

sys.path.append(os.getcwd())


def pids_from_list(path):
    """Participant ids from a ship list: '#' comments and blank lines ignored."""
    ids = []
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line.isdigit():
                ids.append(int(line))
    return ids


def script_paths(script):
    """The literal tar arguments in a tar_expert script, in order.

    Every argument is a relative path containing a slash; the slash is what
    separates them from the script's other 4-space-indented lines (echo, exit).
    """
    with open(script) as f:
        return re.findall(r'^    (\S+/\S+)', f.read(), re.M)


def check(pid, root):
    """Return the list of paths for `pid` that do not exist under `root`."""
    script = f'scripts/tar_expert{pid}.sh'
    if not os.path.isfile(script):
        return [f'(no {script})']
    return [p for p in script_paths(script) if not os.path.exists(os.path.join(root, p))]


def main():
    """Check each requested subject and report."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pids', nargs='*', type=int)
    ap.add_argument('--file', default=None, help='a splits/ship-*.txt list')
    ap.add_argument('--takes_root',
                    default=os.environ.get('TAKES_ROOT', '/vision/group/egoexo4d/takes'))
    args = ap.parse_args()

    pids = list(args.pids) + (pids_from_list(args.file) if args.file else [])
    if not pids:
        ap.error('give pids or --file')
    if not os.path.isdir(args.takes_root):
        raise SystemExit(f'no takes root at {args.takes_root}; set TAKES_ROOT')

    n_bad = 0
    for pid in pids:
        missing = check(pid, args.takes_root)
        if missing:
            n_bad += 1
            more = f' (+{len(missing) - 3} more)' if len(missing) > 3 else ''
            print(f'MISSING {pid}: {", ".join(missing[:3])}{more}')
        else:
            print(f'OK      {pid}: {len(script_paths(f"scripts/tar_expert{pid}.sh"))} paths')
    sys.exit(n_bad)


if __name__ == '__main__':
    main()
