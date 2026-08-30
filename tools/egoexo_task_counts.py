"""Count EgoExo4D takes and unique participants per task.

Written to be checkable rather than trusted: every number quoted about how much
footage a task has, and how deep it goes per person, comes out of here.

The distinction that matters for planning is takes-per-person, not takes. A task
with many people and few takes each (basketball: 110 people, median 8 takes)
supports breadth; one with few people and many takes each supports depth. Which
you need depends on whether a policy is trained per subject or across a pool,
and the totals alone do not tell you.

    python tools/egoexo_task_counts.py                       # every parent task
    python tools/egoexo_task_counts.py --level child         # every drill/recipe
    python tools/egoexo_task_counts.py --task Basketball     # its children, then
                                                             #   every participant
    python tools/egoexo_task_counts.py --task Basketball --level child

NOTE ON SPLITS: takes.json carries no train/val/test field -- the only related
key is `validated`, a data-quality flag. So these counts are the WHOLE release,
all splits pooled. Comparing against another dataset's training set needs the
split file, which is a separate download.

--takes_json is resolved the same way tools/list_layup_takes.py resolves it:
explicit path, then $EGOEXO_TAKES_JSON, then beside --takes_root, then $HOME.
"""
import argparse
import collections
import os
import sys

sys.path.append(os.getcwd())

from tools.list_layup_takes import load_takes, resolve_takes_json


def task_of(take, level):
    """Return a take's task name at 'parent' (e.g. Basketball) or 'child' level."""
    key = 'parent_task_name' if level == 'parent' else 'task_name'
    return take.get(key) or '(none)'


def summarize(takes):
    """Return (people, n_takes, per-person take counts sorted, total hours).

    Takes with no participant_uid are counted in n_takes but cannot be
    attributed to a person, so they are excluded from the per-person figures
    rather than silently treated as one extra participant.
    """
    per = collections.Counter(t['participant_uid'] for t in takes
                              if t.get('participant_uid') is not None)
    hours = sum(t.get('duration_sec') or 0 for t in takes) / 3600.0
    return len(per), len(takes), sorted(per.values()), hours


def median(values):
    """Return the median of a sorted list, or 0 when it is empty."""
    if not values:
        return 0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def print_table(groups, title):
    """Print one row per task: people, takes, takes-per-person spread, hours."""
    print(f'\n{title}')
    print(f"{'task':<44} {'people':>7} {'takes':>7} {'takes/person min/med/max':>26} {'hours':>7}")
    print('-' * 95)
    for name in sorted(groups, key=lambda k: -summarize(groups[k])[0]):
        people, n, per, hours = summarize(groups[name])
        spread = f'{min(per)}/{median(per):g}/{max(per)}' if per else '-'
        print(f'{name:<44} {people:>7} {n:>7} {spread:>26} {hours:>7.1f}')


def print_participants(takes, level):
    """Print each participant's take count for a single task, deepest first.

    This is the listing that verifies a per-person claim -- that 387 has 12
    layup takes against a median of 5, say -- rather than asking anyone to
    believe an aggregate.
    """
    by_person = collections.defaultdict(list)
    for t in takes:
        by_person[t.get('participant_uid')].append(t)
    print(f"\n{'participant':>12} {'takes':>6} {'minutes':>8}  breakdown")
    print('-' * 76)
    for uid in sorted(by_person, key=lambda u: (-len(by_person[u]), (u is None, u))):
        v = by_person[uid]
        mins = sum(t.get('duration_sec') or 0 for t in v) / 60.0
        kinds = collections.Counter(task_of(t, 'child') for t in v)
        detail = ', '.join(f'{k.split(" - ")[-1]}={n}' for k, n in sorted(kinds.items()))
        print(f'{str(uid):>12} {len(v):>6} {mins:>8.1f}  {detail}')


def main():
    """Resolve takes.json, group by task, and print the requested breakdown."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--takes_json', default=None,
                        help='EgoExo4D takes.json (see module docstring for the search order)')
    parser.add_argument('--takes_root', default=None,
                        help='EgoExo4D takes/ dir, used only to locate takes.json')
    parser.add_argument('--level', choices=['parent', 'child'], default='parent',
                        help="'parent' groups by Basketball/Cooking/...; 'child' by the "
                             'individual drill or recipe (default: parent)')
    parser.add_argument('--task', default=None,
                        help='restrict to one task, matched against either level; also '
                             'prints the per-participant listing')
    args = parser.parse_args()

    try:
        takes_json = resolve_takes_json(args.takes_json, args.takes_root)
    except FileNotFoundError as err:
        parser.error(str(err))
    takes = load_takes(takes_json)
    print(f'metadata: {takes_json}')

    people = {t['participant_uid'] for t in takes if t.get('participant_uid') is not None}
    print(f'ALL of EgoExo4D: {len(takes)} takes, {len(people)} people')

    if args.task:
        sel = [t for t in takes
               if args.task.lower() in (task_of(t, 'parent') + ' ' + task_of(t, 'child')).lower()]
        if not sel:
            parser.error(f'no takes match --task {args.task!r}')
        groups = collections.defaultdict(list)
        for t in sel:
            groups[task_of(t, 'child')].append(t)
        print_table(groups, f'{args.task}: by child task')
        print_participants(sel, args.level)
        return

    groups = collections.defaultdict(list)
    for t in takes:
        groups[task_of(t, args.level)].append(t)
    print_table(groups, f'by {args.level} task')


if __name__ == '__main__':
    main()
