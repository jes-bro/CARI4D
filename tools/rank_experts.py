"""Rank EgoExo4D participants by proficiency for one activity and pick the top N.

The partner wants "10 experts each" for guitar, piano and partner dancing. This
turns that into a checkable list rather than a hand-picked one: every
participant with a take in the activity is scored, the top N are chosen, and
the exact gen_tar_expert.py commands for them are printed.

    python tools/rank_experts.py                       # rank all three, pick 10 each
    python tools/rank_experts.py --n 10 --only guitar  # one activity, full table
    python tools/rank_experts.py --emit-gen > /tmp/gen.sh   # the generator commands
    python tools/rank_experts.py --emit-lists          # splits/ship-<activity>.txt

SCORING. Each take's proficiency_demonstrator label is worth Late Expert 3,
Intermediate Expert 2, Early Expert 1, Novice 0, unlabelled 0, and a
participant's score is the sum over their takes in the activity. So four Late
Expert takes (12) outrank one Late Expert take plus three unlabelled (3), and a
fully-labelled Intermediate (8) outranks a half-labelled Late (6). Ties break
on the best single tier, then on take count.

ONE SUBJECT, ONE ARCHIVE. Archives are named expert<pid>_takes.tar, so a
participant can only be packed once. Activities are assigned in the order
given -- thinnest first, guitar before piano -- and a participant already
taken by an earlier activity is skipped by the later one. Participant 493 is a
Late Expert at both guitar and piano; guitar has fewer candidates, so guitar
gets them.

WHY SALSA IS THE DANCE. "Dancing, ideally with a partner": the uniandes
LosAndes_*_Salsa takes are partner dances; the "Performing the choreography"
takes at uniandes and upenn are solo. Participants who did both are filtered
to their salsa takes.

The upenn Music takes use a gp01-gp06 GoPro rig rather than cam01-cam05.
tools/gen_tar_expert.py packs them since it learned that prefix; note that
scripts/recon_common.sh still globs cam*.mp4, so those archives extract fine
but are not yet reconstructable without teaching it the second prefix.
"""
import argparse
import collections
import json
import os
import sys

sys.path.append(os.getcwd())

from tools.gen_tar_expert import cams_of, load_takes  # noqa: E402

TIER_SCORE = {'Late Expert': 3, 'Intermediate Expert': 2, 'Early Expert': 1, 'Novice': 0}

# activity -> (parent_task_name, substring every wanted task_name contains)
ACTIVITIES = {
    'guitar': ('Music', 'Guitar'),
    'piano': ('Music', 'Piano'),
    'salsa': ('Dance', 'Salsa'),
}

ANNOT_DIR = os.path.expanduser('~/Downloads/annotations-egoexo4d')


def load_proficiency(annot_dir=ANNOT_DIR):
    """take_uid -> proficiency_score, from the train and val demonstrator files."""
    out = {}
    for split in ('train', 'val'):
        p = os.path.join(annot_dir, f'proficiency_demonstrator_{split}.json')
        if os.path.isfile(p):
            with open(p) as f:
                for a in json.load(f)['annotations']:
                    out[a['take_uid']] = a['proficiency_score']
    if not out:
        raise SystemExit(f'no proficiency_demonstrator_*.json under {annot_dir}')
    return out


def load_people():
    """participant_uid -> participant record, for the gender field in notes."""
    for d in (os.environ.get('EGOEXO_META'), os.path.expanduser('~/egoexo4d')):
        p = d and os.path.join(d, 'participants.json')
        if p and os.path.isfile(p):
            with open(p) as f:
                return {r['participant_uid']: r for r in json.load(f)}
    return {}


def gender_of(rec):
    """Recorded gender, lowercased, or 'gender unrecorded'."""
    try:
        g = json.loads(rec['metadata']).get('gender')
        return g.lower() if g else 'gender unrecorded'
    except Exception:
        return 'gender unrecorded'


def rig_of(take):
    """'gp' for the upenn GoPro naming, 'cam' otherwise, '?' if neither."""
    cams = cams_of(take)
    return cams[0][:2] if cams and cams[0].startswith('gp') else ('cam' if cams else '?')


def candidates(takes, prof, activity):
    """Every participant with a non-dropped take in `activity`, scored.

    Returns a list of dicts sorted best-first. `takes` on each is the
    participant's takes in this activity only, sorted by take_idx.
    """
    scenario, needle = ACTIVITIES[activity]
    by = collections.defaultdict(list)
    for t in takes:
        if (t.get('parent_task_name') == scenario and needle in t['task_name']
                and not t.get('is_dropped') and t.get('participant_uid') is not None):
            by[t['participant_uid']].append(t)
    rows = []
    for pid, tk in by.items():
        tk.sort(key=lambda t: t['take_idx'])
        tiers = collections.Counter(prof.get(t['take_uid'], 'unlabelled') for t in tk)
        score = sum(TIER_SCORE.get(prof.get(t['take_uid']), 0) for t in tk)
        best = max((TIER_SCORE[x] for x in tiers if x in TIER_SCORE), default=-1)
        rows.append({'pid': pid, 'takes': tk, 'tiers': tiers, 'score': score,
                     'best': best, 'minutes': sum(t['duration_sec'] for t in tk) / 60,
                     'university': tk[0]['university_name'], 'rig': rig_of(tk[0])})
    rows.sort(key=lambda r: (-r['score'], -r['best'], -len(r['takes']), r['pid']))
    return rows


def pick(takes, prof, n, order):
    """Top-n per activity, each participant used at most once, in `order`."""
    taken, chosen = set(), {}
    for act in order:
        rows = [r for r in candidates(takes, prof, act) if r['pid'] not in taken]
        chosen[act] = rows[:n]
        taken.update(r['pid'] for r in chosen[act])
    return chosen


def tier_text(tiers):
    """'Late Expert x4' or 'Late Expert x1, unlabelled x3', best tier first."""
    order = sorted(tiers, key=lambda k: -TIER_SCORE.get(k, -1))
    return ', '.join(f'{k} x{tiers[k]}' for k in order)


def note_for(row, activity, people, all_takes):
    """The WHO THIS IS sentence written into the generated tar script header."""
    caps = sorted({t['take_name'].rsplit('_', 1)[0] for t in row['takes']})
    others = collections.Counter(
        t['task_name'] for t in all_takes
        if t.get('participant_uid') == row['pid'] and not t.get('is_dropped')
        and t not in row['takes'])
    s = (f"{tier_text(row['tiers'])}; {gender_of(people.get(row['pid'], {}))}, "
         f"{row['university']}. {len(row['takes'])} {activity} takes in {', '.join(caps)}.")
    if others:
        s += (f" Their other {sum(others.values())} take(s) "
              f"({', '.join(sorted(others))}) are filtered out.")
    if row['rig'] == 'gp':
        s += ' upenn gp01-gp06 rig: packs fine, recon_common.sh cannot mask it yet.'
    return s


def gen_command(row, activity, people, all_takes):
    """One tools/gen_tar_expert.py invocation, with the task filter spelled out."""
    tasks = '|'.join(sorted({t['task_name'] for t in row['takes']}))
    scenario = ACTIVITIES[activity][0]
    note = note_for(row, activity, people, all_takes).replace("'", "'\\''")
    return (f"python tools/gen_tar_expert.py {row['pid']} --task {activity} "
            f"--scenario '{scenario}' --tasks '{tasks}' --note '{note}'")


def print_table(activity, rows, chosen_pids):
    """Human-readable ranking, with a marker on the chosen ones."""
    print(f'== {activity}: {len(rows)} candidates')
    for r in rows:
        mark = '*' if r['pid'] in chosen_pids else ' '
        print(f" {mark} pid={r['pid']:<4} score={r['score']:<3} n={len(r['takes']):<2} "
              f"{r['minutes']:5.1f}min {r['university']:<8} {r['rig']:<3} {tier_text(r['tiers'])}")


def main():
    """Rank, pick, and print either the table, the gen commands, or the lists."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--only', choices=list(ACTIVITIES), default=None)
    ap.add_argument('--emit-gen', action='store_true',
                    help='print the gen_tar_expert.py commands for the chosen')
    ap.add_argument('--emit-lists', action='store_true',
                    help='write splits/ship-<activity>.txt for scripts/ship_experts.sh --file')
    ap.add_argument('--takes_json', default=None)
    args = ap.parse_args()

    takes = load_takes(args.takes_json)
    prof = load_proficiency()
    people = load_people()
    order = [args.only] if args.only else list(ACTIVITIES)
    chosen = pick(takes, prof, args.n, order)

    if args.emit_gen:
        print('set -e')
        for act in order:
            for r in chosen[act]:
                print(gen_command(r, act, people, takes))
        return
    if args.emit_lists:
        os.makedirs('splits', exist_ok=True)
        for act in order:
            path = f'splits/ship-{act}.txt'
            with open(path, 'w') as f:
                f.write(f'# top {args.n} {act} participants by proficiency, from '
                        f'tools/rank_experts.py. One pid per line; ship with\n'
                        f'#   XFER=rclone DEST=gdrive:cari4d-handoff '
                        f'bash scripts/ship_experts.sh --file {path}\n')
                for r in chosen[act]:
                    f.write(f"{r['pid']}  # {tier_text(r['tiers'])}, {len(r['takes'])} takes, "
                            f"{r['minutes']:.1f} min, {r['university']}\n")
            print(f'wrote {path}')
        return
    for act in order:
        print_table(act, candidates(takes, prof, act), {r['pid'] for r in chosen[act]})
        print()


if __name__ == '__main__':
    main()
