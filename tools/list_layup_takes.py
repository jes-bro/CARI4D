"""List EgoExo4D basketball layup takes, ranked by how much of an existing
reconstruction's setup they can reuse.

The expensive, manual part of a new reconstruction is not the pipeline -- it is
the camera work: rectifying a new fisheye view, re-deriving the pipeline
camera's intrinsics/extrinsics, re-locating the hoop, re-tuning the depth band.
None of that has to be redone for a take shot in the *same capture session* as
one already reconstructed, because the exo rig is not moved between takes of a
session. So the takes are grouped into tiers by what they share with an anchor
take:

  tier 0  same participant, same capture   -- more motions from the same person,
                                              zero new camera work
  tier 1  other participants, same capture -- other people, still zero new
                                              camera work
  tier 2  same gym, different capture      -- same room and hoop, but the rig
                                              was re-placed: new calibration
  tier 3  different gym                    -- everything new

Usage:

    python tools/list_layup_takes.py                       # default anchor
    python tools/list_layup_takes.py --tier 0 --tier 1
    python tools/list_layup_takes.py --task all --json out.json
    python tools/list_layup_takes.py --takes_root /vision/group/egoexo4d/takes

`--takes_root` turns on an on-disk check: each take is marked present when its
frame_aligned_videos directory holds the exo videos, so the download list is
whatever comes back missing.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.append(os.getcwd())

DEFAULT_TAKES_JSON = os.path.expanduser('~/egoexo4d/takes.json')
# The take already reconstructed: Date03_Sub01_bball_dribble came out of this
# one (cam04, frames 354-454). Everything is ranked relative to it.
DEFAULT_ANCHOR = 'unc_basketball_03-31-23_02_9'


def load_takes(takes_json):
    """Read takes.json and return the list of take records."""
    with open(takes_json) as f:
        return json.load(f)


def capture_key(take):
    """Return the capture session a take belongs to.

    Takes of one session share a camera rig placement, hence a calibration.
    `capture_uid` is the authoritative field; the take-name prefix is the
    human-readable form of the same thing and is used for display.
    """
    return take['capture_uid']


def capture_name(take):
    """Return the readable capture session name, e.g. unc_basketball_03-31-23_02."""
    return take['take_name'].rsplit('_', 1)[0]


def is_layup(take):
    """Return whether a take is one of the two layup drills.

    EgoExo4D's basketball parent task has three children: Mikan Layup, Reverse
    Layup, and Mid-Range Jump Shooting. Only the first two are layups.
    """
    return 'Layup' in (take.get('task_name') or '')


def is_basketball(take):
    """Return whether a take belongs to the Basketball parent task."""
    return (take.get('parent_task_name') or '') == 'Basketball'


def drill_name(take):
    """Return the drill name with the shared 'Basketball Drills - ' prefix removed."""
    name = take.get('task_name') or ''
    prefix = 'Basketball Drills - '
    return name[len(prefix):] if name.startswith(prefix) else name


def tier_of(take, anchor):
    """Return the reuse tier of `take` relative to the `anchor` take.

    0 = same participant and capture, 1 = same capture, 2 = same gym,
    3 = anything else. See the module docstring for what each tier costs.
    """
    if capture_key(take) == capture_key(anchor):
        if take['participant_uid'] == anchor['participant_uid']:
            return 0
        return 1
    if take.get('physical_setting_uid') == anchor.get('physical_setting_uid'):
        return 2
    return 3


def exo_videos_present(takes_root, take, cams):
    """Return the exo cameras of `take` whose frame-aligned video exists on disk.

    Returns an empty list when `takes_root` is None, i.e. when no on-disk check
    was requested.
    """
    if not takes_root:
        return []
    vid_dir = os.path.join(takes_root, take['take_name'], 'frame_aligned_videos')
    return [c for c in cams if os.path.isfile(os.path.join(vid_dir, f'{c}.mp4'))]


def estimate_attempts(take):
    """Estimate how many distinct layup attempts a take contains.

    Measured off a 0.5 s contact sheet of the anchor take (a 64 s Reverse Layup
    take, ~13 attempts): the reverse drill cycles approach -> layup -> rebound
    -> walk back on a ~4.5 s period, while the Mikan drill is continuous
    alternating layups under the rim at roughly 2.2 s per rep. This is a
    planning number, not a segmentation -- see --json and the window finder for
    actual frame ranges.
    """
    period = 2.2 if 'Mikan' in drill_name(take) else 4.5
    return max(1, int(round(take['duration_sec'] / period)))


def collect(takes, anchor, tiers, layups_only):
    """Group the takes of interest by tier, then by participant.

    Returns {tier: {participant_uid: [take, ...]}}, each take list sorted by
    take index.
    """
    keep = is_layup if layups_only else is_basketball
    out = defaultdict(lambda: defaultdict(list))
    for take in takes:
        if not is_basketball(take) or not keep(take):
            continue
        if take.get('is_dropped'):
            continue
        tier = tier_of(take, anchor)
        if tier not in tiers:
            continue
        out[tier][take['participant_uid']].append(take)
    for by_part in out.values():
        for lst in by_part.values():
            lst.sort(key=lambda t: t['take_idx'])
    return out


def print_report(grouped, anchor, takes_root, cams):
    """Print the tier tables and a per-tier total of takes, minutes and attempts."""
    tier_label = {
        0: 'tier 0  same participant, same capture -- no new camera work',
        1: 'tier 1  other people, same capture -- no new camera work',
        2: 'tier 2  same gym, new capture -- needs its own calibration',
        3: 'tier 3  different gym -- everything new',
    }
    print(f'anchor: {anchor["take_name"]}  participant {anchor["participant_uid"]}  '
          f'capture {capture_key(anchor)}')
    for tier in sorted(grouped):
        by_part = grouped[tier]
        n_takes = sum(len(v) for v in by_part.values())
        secs = sum(t['duration_sec'] for v in by_part.values() for t in v)
        attempts = sum(estimate_attempts(t) for v in by_part.values() for t in v)
        print()
        print(f'=== {tier_label[tier]}')
        print(f'    {len(by_part)} participant(s), {n_takes} takes, '
              f'{secs / 60:.1f} min, ~{attempts} attempts')
        for part in sorted(by_part, key=lambda p: (p is None, p)):
            lst = by_part[part]
            print(f'  participant {part}:')
            for t in lst:
                disk = ''
                if takes_root:
                    have = exo_videos_present(takes_root, t, cams)
                    disk = '  [' + (','.join(have) if have else 'MISSING') + ']'
                print(f'    {t["take_name"]:34} {drill_name(t):24} '
                      f'{t["duration_sec"]:6.1f}s  ~{estimate_attempts(t):3d} attempts  '
                      f'{t["take_uid"]}{disk}')


def write_json(grouped, path, takes_root, cams):
    """Write the selected takes as a flat JSON list, for scripted downloads and staging."""
    rows = []
    for tier in sorted(grouped):
        for part in sorted(grouped[tier], key=lambda p: (p is None, p)):
            for t in grouped[tier][part]:
                rows.append({
                    'take_name': t['take_name'],
                    'take_uid': t['take_uid'],
                    'capture_uid': t['capture_uid'],
                    'participant_uid': t['participant_uid'],
                    'drill': drill_name(t),
                    'duration_sec': t['duration_sec'],
                    'best_exo': t['best_exo'],
                    'tier': tier,
                    'est_attempts': estimate_attempts(t),
                    'exo_on_disk': exo_videos_present(takes_root, t, cams),
                })
    with open(path, 'w') as f:
        json.dump(rows, f, indent=1)
    print(f'\nwrote {len(rows)} takes to {path}')


def main():
    """Parse arguments, group the takes and print (optionally dump) the report."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--takes_json', default=DEFAULT_TAKES_JSON,
                        help='EgoExo4D takes.json (default: %(default)s)')
    parser.add_argument('--anchor', default=DEFAULT_ANCHOR,
                        help='take name the tiers are measured against (default: %(default)s)')
    parser.add_argument('--tier', type=int, action='append', dest='tiers',
                        help='tier to include, repeatable (default: 0 1 2)')
    parser.add_argument('--task', choices=['layup', 'all'], default='layup',
                        help='layup drills only, or every basketball drill')
    parser.add_argument('--takes_root', default=None,
                        help='EgoExo4D takes/ dir; enables the on-disk video check')
    parser.add_argument('--cams', default='cam01,cam02,cam03,cam04',
                        help='exo cameras to check for (default: %(default)s)')
    parser.add_argument('--json', default=None, help='also write the selection here')
    args = parser.parse_args()

    takes = load_takes(args.takes_json)
    by_name = {t['take_name']: t for t in takes}
    if args.anchor not in by_name:
        parser.error(f'anchor take {args.anchor} not in {args.takes_json}')
    anchor = by_name[args.anchor]

    tiers = set(args.tiers) if args.tiers else {0, 1, 2}
    cams = [c for c in args.cams.split(',') if c]
    grouped = collect(takes, anchor, tiers, args.task == 'layup')
    print_report(grouped, anchor, args.takes_root, cams)
    if args.json:
        write_json(grouped, args.json, args.takes_root, cams)


if __name__ == '__main__':
    main()
