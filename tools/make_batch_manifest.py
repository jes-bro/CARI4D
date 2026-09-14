"""Write a recon_batch.sh manifest for one activity's shipped subjects.

The layup manifest was written by hand from tools/list_layup_takes.py output.
Every activity after it needs the same eight columns, and two of them are
easy to get wrong by hand: the sequence name, whose Sub slot silently picks
the SMPL body gender, and the SAM3 prompts, whose basketball defaults produce
zero masks anywhere else. So the manifest is generated from the same ship list
that packed the archives, and the prompts travel in the row.

    python tools/make_batch_manifest.py guitar                 # splits/guitar-batch.tsv
    python tools/make_batch_manifest.py guitar --pids 400      # one subject
    python tools/make_batch_manifest.py piano --object_prompt 'upright piano'

Columns (tab-separated, read by scripts/recon_batch.sh):

    take  seq  participant  drill  duration_sec  pipe_cam  human_prompt  object_prompt

pipe_cam is written as '-' (recon_batch.sh reads that as empty): recon_common.sh
resolves it from the take's own best_exo at run time, and a value typed here
would only outrank that when someone has checked it against a mask video. It
is '-' and not blank because bash's tab-separated read collapses consecutive
tabs, which would shift the prompts one column left.

SEQUENCE NAMES are Date_Sub_object_action, parsed downstream (see
scripts/recon_common.sh, "SEQUENCE NAMING"). Sub01 is a male body, Sub06
female; a participant whose gender is unrecorded -- every upenn musician --
gets Sub01 and a comment saying so, because the body model has no neutral.
The action slot ends in the take index so recon_status.sh can strip the
clip letters stage 1a appends (Date05_Sub01_guitar_free400x2 -> ...x2a).

PROMPTS default per activity to the short forms docs/custom_video.md
recommends. Salsa is deliberately absent: there is no object, and the second
dancer is what the pipeline would need to learn to see. See rank_experts.py.
"""
import argparse
import os
import re
import sys

sys.path.append(os.getcwd())

from tools.check_tar_paths import pids_from_list  # noqa: E402
from tools.gen_tar_expert import load_takes  # noqa: E402
from tools.rank_experts import ACTIVITIES, gender_of, load_people  # noqa: E402

# object slot of the sequence name, and the SAM3 prompts, per activity
OBJECT_SLOT = {'guitar': 'guitar', 'piano': 'piano', 'cpr': 'manikin'}
PROMPTS = {
    'guitar': ('person', 'guitar'),
    'piano': ('person', 'piano'),
    'cpr': ('person', 'CPR manikin'),
}
# a Date slot per activity, so two activities never produce the same name
DATE_SLOT = {'guitar': 'Date05', 'piano': 'Date06', 'cpr': 'Date07'}

# task_name -> short drill token for the action slot
DRILL_WORDS = {'Freeplaying': 'free', 'Scales and Arpeggios': 'scales', 'Suzuki Books': 'suzuki'}


def drill_of(take):
    """A short alphanumeric drill token from the take's task_name.

    'Playing Guitar - Freeplaying' -> 'free', 'Playing Guitar' -> 'play',
    anything else -> its last ' - ' segment, lowercased, letters only, <= 6.
    """
    name = take['task_name']
    tail = name.split(' - ')[-1] if ' - ' in name else ''
    if tail in DRILL_WORDS:
        return DRILL_WORDS[tail]
    if not tail:
        return 'play' if name.startswith('Playing') else re.sub(r'[^a-z]', '', name.lower())[:6]
    return re.sub(r'[^a-z]', '', tail.lower())[:6] or 'take'


def sub_of(pid, people):
    """Sub slot for the body gender: Sub06 for female, Sub01 otherwise."""
    return 'Sub06' if gender_of(people.get(pid, {})) == 'female' else 'Sub01'


def seq_name(activity, take, people):
    """Date_Sub_object_action, unique per take, ending in a digit."""
    pid = take['participant_uid']
    return (f"{DATE_SLOT[activity]}_{sub_of(pid, people)}_{OBJECT_SLOT[activity]}_"
            f"{drill_of(take)}{pid}x{take['take_idx']}")


def rows_for(activity, pids, takes, people, human_prompt, object_prompt):
    """One manifest row per non-dropped take of each pid in the activity."""
    scenario, needle = ACTIVITIES[activity]
    out = []
    for pid in pids:
        mine = sorted((t for t in takes
                       if t.get('participant_uid') == pid and not t.get('is_dropped')
                       and t.get('parent_task_name') == scenario and needle in t['task_name']),
                      key=lambda t: t['take_idx'])
        for t in mine:
            # '-' rather than empty: bash's tab-separated read collapses
            # consecutive tabs, so an empty column shifts every later one left.
            out.append([t['take_name'], seq_name(activity, t, people), str(pid), drill_of(t),
                        f"{t['duration_sec']:.1f}", '-', human_prompt, object_prompt])
    return out


def main():
    """Generate splits/<activity>-batch.tsv from splits/ship-<activity>.txt."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('activity', choices=sorted(PROMPTS))
    ap.add_argument('--pids', nargs='*', type=int, default=None,
                    help='subjects to include; default: every pid in splits/ship-<activity>.txt')
    ap.add_argument('--human_prompt', default=None)
    ap.add_argument('--object_prompt', default=None)
    ap.add_argument('--out', default=None, help='default splits/<activity>-batch.tsv')
    ap.add_argument('--takes_json', default=None)
    args = ap.parse_args()

    if args.activity not in ACTIVITIES:
        raise SystemExit(f"{args.activity} is not in rank_experts.ACTIVITIES; add it there first")
    pids = args.pids or pids_from_list(f'splits/ship-{args.activity}.txt')
    takes = load_takes(args.takes_json)
    people = load_people()
    hp = args.human_prompt or PROMPTS[args.activity][0]
    op = args.object_prompt or PROMPTS[args.activity][1]
    rows = rows_for(args.activity, pids, takes, people, hp, op)
    if not rows:
        raise SystemExit('no takes matched')

    unrecorded = sorted({int(r[2]) for r in rows
                         if gender_of(people.get(int(r[2]), {})) == 'gender unrecorded'})
    out = args.out or f'splits/{args.activity}-batch.tsv'
    with open(out, 'w') as f:
        f.write('# take\tseq\tparticipant\tdrill\tduration_sec\tpipe_cam\thuman_prompt\tobject_prompt\n')
        f.write(f'# generated by tools/make_batch_manifest.py {args.activity}; '
                f'{len(rows)} takes, {len(pids)} subjects.\n')
        f.write('# pipe_cam is "-" on purpose (= unset): resolved from best_exo at run time.\n')
        if unrecorded:
            f.write(f'# gender unrecorded, so Sub01 (male body) assumed for: '
                    f'{" ".join(map(str, unrecorded))}\n')
        f.write(f"# run one subject first:  ONLY={rows[0][2]} bash scripts/recon_batch.sh clips {out}\n")
        for r in rows:
            f.write('\t'.join(r) + '\n')
    print(f'wrote {out}: {len(rows)} takes from {len(pids)} subjects; prompts {hp!r} / {op!r}')


if __name__ == '__main__':
    main()
