"""Write the MANIFEST.tsv and README that travel with the handoff archives.

An archive named expert50_takes.tar says nothing about who 50 is, what they
did, how many takes are inside or which cameras they used. Whoever unpacks
these will not have takes.json in front of them, so the manifest carries what
they need to decide what to reconstruct and in what order.

    python tools/make_handoff_manifest.py --out handoff/

Reads the tar_expert*.sh scripts as the source of truth for what is actually
packed, rather than re-deriving it, so the manifest cannot disagree with the
archives.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.append(os.getcwd())


def load_json(*names):
    """Load the first of `names` that exists, from the usual metadata places."""
    for n in names:
        for d in (os.environ.get('EGOEXO_META'), os.path.expanduser('~/egoexo4d')):
            if d and os.path.isfile(os.path.join(d, n)):
                with open(os.path.join(d, n)) as f:
                    return json.load(f)
    raise SystemExit(f'could not find {names[0]}')


def scripted_takes(path):
    """The take names one tar_expert script packs, in the order it packs them."""
    seen, out = set(), []
    for n in re.findall(r'^    (\S+?)/', open(path).read(), re.M):
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def gender_of(rec):
    """A participant's recorded gender, or 'unrecorded' when there is none."""
    try:
        return json.loads(rec['metadata']).get('gender') or 'unrecorded'
    except Exception:
        return 'unrecorded'


def rows(takes_by_name, people, prof):
    """One manifest row per tar_expert script, sorted by scenario then subject."""
    out = []
    for f in sorted(glob.glob('scripts/tar_expert*.sh')):
        pid = int(re.search(r'tar_expert(\d+)', f).group(1))
        names = scripted_takes(f)
        tk = [takes_by_name[n] for n in names]
        cams = sorted({c for t in tk for c in t.get('frame_aligned_videos', {})
                       if c.startswith('cam')})
        tiers = sorted({prof[t['take_uid']] for t in tk if t['take_uid'] in prof})
        out.append({
            # 'health' is EgoExo4D's name for the scenario; everyone here calls
            # it CPR, and scripts/ship_experts.sh names the folder that too.
            'scenario': {'health': 'cpr'}.get(
                (tk[0].get('parent_task_name') or '').lower().replace(' ', '-'),
                (tk[0].get('parent_task_name') or '').lower().replace(' ', '-')),
            'participant': pid,
            'archive': f'expert{pid}_takes.tar',
            'takes': len(tk),
            'minutes': round(sum(t['duration_sec'] for t in tk) / 60, 1),
            'proficiency': '/'.join(tiers) or 'unlabelled',
            'gender': gender_of(people.get(pid, {})),
            'university': tk[0]['university_name'],
            'captures': ';'.join(sorted({n.rsplit('_', 1)[0] for n in names})),
            'cameras': ';'.join(cams),
            'tasks': ';'.join(sorted({t['task_name'] for t in tk})),
        })
    out.sort(key=lambda r: (r['scenario'], -r['takes'], r['participant']))
    return out


README = """CARI4D handoff archives
=======================

One tar per subject, grouped by scenario. Paths inside each archive are
relative to an EgoExo4D takes/ root, so:

    tar -xf expert387_takes.tar -C /your/egoexo4d/takes

puts every file where the pipeline scripts look for it. Each archive holds,
per take: the 4K exo videos, the whole frame_aligned_videos/downscaled/448
directory, and trajectory/gopro_calibs.csv.

MANIFEST.tsv lists every archive with its subject, take count, duration,
proficiency tier, gender, university, captures, cameras and tasks.

THINGS THAT WILL BITE YOU

Cameras are per capture, not per university. Some captures have five exo
cameras, some are missing cam02 entirely. scripts/recon_common.sh reads each
take's list off disk; do not copy an ALL_CAMS value between captures.

The pipeline camera comes from each take's own best_exo now, not a cam04
default. Confirm it against the stage-1 mask overlay before committing a
capture to a view.

The SAM3 prompts default to basketball. A pan, a soccer ball and a CPR
manikin all need OBJECT_PROMPT changed, and leaving the default in place is
how a sequence comes back with no masks at all.

REINIT_EVERY is derived from the object's symmetry. A ball gets
re-registration and cannot drift; a pan and a manikin have observable
orientation and must not be forced to 1, or re-registration will spin them.

CPR carries no proficiency tier. The Health scenario has no proficiency
annotations at all -- that is the dataset, not an omission.
"""


def main():
    """Write MANIFEST.tsv and README.txt describing every archive."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='handoff', help='directory to write into')
    args = ap.parse_args()

    takes = load_json('takes.json')
    people = {p['participant_uid']: p for p in load_json('participants.json')}
    prof = {}
    for n in ('proficiency_demonstrator_train.json', 'proficiency_demonstrator_val.json'):
        p = os.path.expanduser(f'~/Downloads/annotations-egoexo4d/{n}')
        if os.path.isfile(p):
            for a in json.load(open(p))['annotations']:
                prof[a['take_uid']] = a['proficiency_score']

    r = rows({t['take_name']: t for t in takes}, people, prof)
    os.makedirs(args.out, exist_ok=True)
    cols = list(r[0])
    with open(os.path.join(args.out, 'MANIFEST.tsv'), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for row in r:
            f.write('\t'.join(str(row[c]) for c in cols) + '\n')
    with open(os.path.join(args.out, 'README.txt'), 'w') as f:
        f.write(README)

    print(f'{len(r)} archives described in {args.out}/MANIFEST.tsv')
    from collections import Counter
    for scen, n in Counter(x['scenario'] for x in r).most_common():
        mins = sum(x['minutes'] for x in r if x['scenario'] == scen)
        print(f'  {scen:<12} {n:3d} subjects  {mins:7.1f} min')


if __name__ == '__main__':
    main()
