#!/usr/bin/env python3
"""Pick N EgoExo4D participants per activity that are NOT already in the
InterMimic training set, for the novice-side handoff.

The trained subjects are the 35 sub4xx/sub5xx ids in the InterMimic g3 configs
(bball7 + soccer15 + cpr13). Those ids are CARI4D Sub slots, not EgoExo
participant ids, so the mapping back to EgoExo comes from the conversion
manifests that produced them (~/cari4d_{bball7,soccer,cpr}/manifest.csv, the
`take` column). Every used take resolves to exactly one participant, and the
35 subjects resolve to 35 distinct participants.

EXCLUSION IS PER PARTICIPANT, not per take. A different take by a trained
person is still that person's body and motion, so excluding only the exact
take would leak the subject across the split.

PROFICIENCY is EgoExo4D's proficiency_demonstrator label, which does not exist
everywhere:
  * Basketball  -- has a Novice tier; exactly 10 unused Novice participants.
  * Soccer      -- has NO Novice tier anywhere in the dataset (only
                   Intermediate/Late Expert), so `--tier unlabeled` is the
                   closest available thing to a novice.
  * CPR         -- no proficiency_demonstrator labels at all; every candidate
                   is unlabeled and selection is by exclusion only.

    python tools/pick_novice_handoff.py basketball --tier Novice
    python tools/pick_novice_handoff.py cpr --tier unlabeled
    python tools/pick_novice_handoff.py soccer --tier unlabeled --emit-files
"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.append(os.getcwd())

# activity -> (parent_task_name, substring the task_name must contain or None)
ACTIVITIES = {
    'basketball': ('Basketball', None),
    'soccer': ('Soccer', None),
    'cpr': ('Health', 'CPR'),
}

# the conversion manifests that name every take already converted for InterMimic
MANIFESTS = [
    '~/cari4d_bball7/manifest.csv',
    '~/cari4d_soccer/manifest.csv',
    '~/cari4d_cpr/manifest.csv',
]

TAKES_JSON = '~/Downloads/takes.json'
ANNOT_DIR = '~/Downloads/annotations-egoexo4d'

# exo cameras the CARI4D pipeline can actually read; scripts/recon_common.sh
# globs cam*.mp4, so a gp01-gp06 rig take is not reconstructable yet.
USABLE_PREFIX = 'cam'


def load_takes(path=TAKES_JSON):
    """Return the EgoExo4D takes.json list."""
    p = os.path.expanduser(path)
    if not os.path.isfile(p):
        raise SystemExit(f'no takes.json at {p}')
    with open(p) as f:
        return json.load(f)


def load_proficiency(annot_dir=ANNOT_DIR):
    """take_uid -> proficiency tier, from the train and val demonstrator files."""
    out = {}
    for split in ('train', 'val'):
        p = os.path.join(os.path.expanduser(annot_dir),
                         f'proficiency_demonstrator_{split}.json')
        if os.path.isfile(p):
            with open(p) as f:
                for a in json.load(f)['annotations']:
                    out[a['take_uid']] = a['proficiency_score']
    return out


def used_participants(takes, manifests=MANIFESTS):
    """Participants already converted into the InterMimic training set.

    Returns (participant_uids, unresolved_take_names). A take name that is not
    in takes.json is reported rather than silently dropped: it would otherwise
    quietly shrink the exclusion set and leak a trained subject.
    """
    by_name = {t['take_name']: t for t in takes}
    pids, missing = set(), []
    for m in manifests:
        p = os.path.expanduser(m)
        if not os.path.isfile(p):
            raise SystemExit(f'missing conversion manifest {p}')
        with open(p) as f:
            for row in csv.DictReader(f):
                t = by_name.get(row['take'])
                if t is None:
                    missing.append(row['take'])
                elif t.get('participant_uid') is not None:
                    pids.add(t['participant_uid'])
    return pids, sorted(set(missing))


def candidates(takes, activity, used, prof):
    """participant_uid -> their non-dropped takes in this activity, unused only.

    Participants with no recorded participant_uid are dropped: they cannot be
    checked against the exclusion set, so they are not safe to hand off.
    """
    scenario, needle = ACTIVITIES[activity]
    out = collections.defaultdict(list)
    for t in takes:
        if t.get('is_dropped') or t['parent_task_name'] != scenario:
            continue
        if needle is not None and needle not in t['task_name']:
            continue
        pid = t.get('participant_uid')
        if pid is None or pid in used:
            continue
        out[pid].append(t)
    return {p: sorted(v, key=lambda t: t['take_idx']) for p, v in out.items()}


def tier_of(takes_of_pid, prof):
    """The single proficiency tier for a participant, or a marker string.

    'unlabeled' when none of their takes carry a label, 'MIXED' when their
    takes disagree -- a mixed participant is never a clean novice, so callers
    filtering on a tier will exclude them.
    """
    tiers = {prof.get(t['take_uid']) for t in takes_of_pid} - {None}
    if not tiers:
        return 'unlabeled'
    return tiers.pop() if len(tiers) == 1 else 'MIXED'


def usable_takes(takes_of_pid):
    """The participant's takes that have at least one cam*.mp4 exo view."""
    return [t for t in takes_of_pid
            if any(k.startswith(USABLE_PREFIX) for k in t['frame_aligned_videos'])]


def main():
    """Print the chosen participants and, with --emit-files, their file list."""
    ap = argparse.ArgumentParser()
    ap.add_argument('activity', choices=sorted(ACTIVITIES))
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--tier', default='Novice',
                    help="proficiency tier to require, or 'unlabeled', or 'any'")
    ap.add_argument('--emit-files', action='store_true',
                    help='print take-relative paths for the transfer list')
    ap.add_argument('--emit-list', action='store_true',
                    help='print a splits/ship-*.txt list (one pid per line)')
    ap.add_argument('--emit-gen', action='store_true',
                    help='print the tools/gen_tar_expert.py commands')
    args = ap.parse_args()

    takes = load_takes()
    prof = load_proficiency()
    used, missing = used_participants(takes)
    if missing:
        raise SystemExit(f'takes not found in takes.json: {missing}')

    pool = candidates(takes, args.activity, used, prof)
    rows = []
    for pid, ts in pool.items():
        ok = usable_takes(ts)
        if not ok:
            continue
        tier = tier_of(ts, prof)
        if args.tier != 'any' and tier != args.tier:
            continue
        rows.append((pid, tier, ok))
    # most footage first: a handoff subject with one 8s take is not much use
    rows.sort(key=lambda r: (-sum(t['duration_sec'] for t in r[2]), r[0]))

    sys.stderr.write(f'# {args.activity}: {len(used)} trained participants excluded, '
                     f'{len(pool)} unused, {len(rows)} match tier={args.tier}\n')
    chosen = rows[:args.n]
    if args.emit_files:
        for _, _, ts in chosen:
            for t in ts:
                for cam in sorted(k for k in t['frame_aligned_videos']
                                  if k.startswith(USABLE_PREFIX)):
                    print(f"{t['take_name']}/frame_aligned_videos/{cam}.mp4")
                print(f"{t['take_name']}/trajectory/gopro_calibs.csv")
        return
    if args.emit_list:
        print(f'# {len(chosen)} {args.activity} participants for the novice handoff, from')
        print('#   tools/pick_novice_handoff.py '
              f'{args.activity} --tier {args.tier} --emit-list')
        print('# None is in the InterMimic training set (bball7 + soccer15 + cpr13).')
        print('# Ship with:')
        print('#   XFER=rclone DEST=gdrive:cari4d-handoff bash scripts/ship_experts.sh '
              f'--file splits/ship-novice-{args.activity}.txt')
        for pid, tier, ts in chosen:
            mins = sum(t['duration_sec'] for t in ts) / 60.0
            print(f"{pid}  # {tier}, {len(ts)} takes, {mins:.1f} min, "
                  f"{ts[0]['university_name']}")
        return
    if args.emit_gen:
        for pid, tier, ts in chosen:
            note = (f"{tier}, {len(ts)} takes in {ts[0]['take_name'].rsplit('_', 1)[0]}; "
                    f"novice-side handoff, not in the training set.")
            print(f"python tools/gen_tar_expert.py {pid} --task \"{args.activity}\" "
                  f"--note \"{note}\"")
        return
    print(f'# {args.activity}: {len(chosen)} participants, tier={args.tier}')
    print('# participant\ttier\ttakes\tminutes\tuniversity\tcapture')
    for pid, tier, ts in chosen:
        mins = sum(t['duration_sec'] for t in ts) / 60.0
        print(f"{pid}\t{tier}\t{len(ts)}\t{mins:.1f}\t{ts[0]['university_name']}\t"
              f"{ts[0]['take_name'].rsplit('_', 1)[0]}")


if __name__ == '__main__':
    main()
