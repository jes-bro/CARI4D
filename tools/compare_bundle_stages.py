"""Say which stage changed the human, and whether it changed shape or depth.

A reconstruction that "looks right at init and shrinks by the end" has two
completely different causes, and they are distinguished by numbers already
sitting in the output file:

  SHAPE   the betas changed, so the body really is a different size
  DEPTH   the betas are identical and the body moved away from the camera, so
          it only LOOKS smaller -- apparent size goes as 1/z

This reads both, for both stages, out of one bundle.

WHY ONE FILE IS ENOUGH. learning/training/opt_refineout.py copies its input to
`pth_data['in']` before optimizing (the `pth_data['in'] = pr` line) and writes
its own result to `pth_data['pr']`. So the final output/opt/<...>/<seq>.pth
holds CoCoNet's answer AND the optimizer's answer side by side. Comparing them
localizes the change without re-rendering anything.

  python tools/compare_bundle_stages.py output/opt/<...>/<seq>.pth

Pass the CoCoNet bundle too, to reach one stage further back:

  python tools/compare_bundle_stages.py output/opt/<...>/<seq>.pth \\
      --coconet output/coconet/<...>/<seq>.pth

CAVEAT ON RESUMED RUNS: `pth_data['in'] = pr` only runs when the optimizer
starts from scratch. A run resumed from a checkpoint leaves 'in' as whatever
the first run wrote, which may be stale. This script says so when it cannot
tell.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())


def load_bundle(path):
    """Load a CARI4D .pth bundle, or exit with a message naming the path."""
    if not os.path.isfile(path):
        raise SystemExit(f'no such bundle: {path}')
    return torch.load(path, map_location='cpu', weights_only=False)


def as_numpy(x):
    """Return `x` as a numpy array, whether it arrived as a tensor or an array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def stage_summary(name, stage):
    """Print frame count, betas and translation statistics for one stage dict."""
    betas = as_numpy(stage['betas'])
    smpl_t = as_numpy(stage['smpl_t'])
    n = smpl_t.shape[0]
    drift = float(np.max(np.abs(betas - betas[0:1]))) if betas.ndim == 2 else 0.0
    print(f'  {name:<10} {n:4d} frames')
    print(f'    betas[0]   {np.array2string(betas[0], precision=3, suppress_small=True)}')
    print(f'    betas drift across frames  {drift:.4g}')
    print(f'    smpl_t mean  x={smpl_t[:, 0].mean():+.3f}  '
          f'y={smpl_t[:, 1].mean():+.3f}  z={smpl_t[:, 2].mean():+.3f}')
    print(f'    smpl_t z     min={smpl_t[:, 2].min():.3f}  max={smpl_t[:, 2].max():.3f}')
    if 'pose_abs' in stage:
        obj_t = as_numpy(stage['pose_abs'])[:, :3, 3]
        print(f'    object z     mean={obj_t[:, 2].mean():.3f}  '
              f'min={obj_t[:, 2].min():.3f}  max={obj_t[:, 2].max():.3f}')


def compare(before_name, before, after_name, after):
    """Print what changed between two stages, and name the likely mechanism.

    The verdict rests on two numbers: how far the betas moved (a real change of
    body size) and how the camera-space depth changed (an apparent one, since
    projected size goes as 1/z).
    """
    b0, b1 = as_numpy(before['betas']), as_numpy(after['betas'])
    t0, t1 = as_numpy(before['smpl_t']), as_numpy(after['smpl_t'])
    n = min(len(t0), len(t1))
    if len(t0) != len(t1):
        print(f'  NOTE: frame counts differ ({len(t0)} vs {len(t1)}); '
              f'comparing the first {n}')
    b0, b1, t0, t1 = b0[:n], b1[:n], t0[:n], t1[:n]

    beta_diff = float(np.max(np.abs(b1 - b0)))
    z0, z1 = t0[:, 2], t1[:, 2]
    z_ratio = float(np.mean(z1) / np.mean(z0)) if np.mean(z0) != 0 else float('nan')
    shift = (t1 - t0)

    print(f'\n=== {before_name}  ->  {after_name} ===')
    print(f'  betas max |change|        {beta_diff:.4g}')
    print(f'  smpl_t mean shift         x={shift[:, 0].mean():+.4f}  '
          f'y={shift[:, 1].mean():+.4f}  z={shift[:, 2].mean():+.4f}')
    print(f'  depth z mean              {np.mean(z0):.3f} -> {np.mean(z1):.3f}  '
          f'(ratio {z_ratio:.3f})')
    print(f'  => apparent size scales by {1.0 / z_ratio:.3f}x from depth alone')

    print('\n  VERDICT')
    if beta_diff > 1e-4:
        print(f'    SHAPE CHANGED. betas moved by {beta_diff:.4g} in this stage.')
        print('    The optimizer only touches betas when opt_betas=True, which')
        print('    defaults to False -- so check the invocation for it, and if')
        print('    this is the coconet->opt comparison, that flag is the cause.')
    else:
        print('    Body shape is UNCHANGED (betas identical to 1e-4).')
    if abs(1.0 - z_ratio) > 0.02:
        direction = 'AWAY FROM' if z_ratio > 1 else 'TOWARD'
        print(f'    Human moved {direction} the camera: depth x{z_ratio:.3f}, so it')
        print(f'    renders {1.0 / z_ratio:.3f}x its previous size with no shape change.')
        print('    smpl_t is anchored by w_init_ht (default 8000) and is only free')
        print('    when opt_smpl_trans=True -- check the invocation for that too.')
    else:
        print('    Depth is essentially unchanged (within 2%).')
    if beta_diff <= 1e-4 and abs(1.0 - z_ratio) <= 0.02:
        print('    Neither shape nor depth moved here. If the render still shrinks')
        print('    between these two stages, the cause is elsewhere -- pose, or the')
        print('    renderer camera. Compare the earlier stage instead.')


def main():
    """Parse arguments, summarise each stage, and compare the ones available."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bundle', help='output/opt/<...>/<seq>.pth (holds both in and pr)')
    ap.add_argument('--coconet', default=None,
                    help='optional output/coconet/<...>/<seq>.pth, to reach one stage back')
    args = ap.parse_args()

    data = load_bundle(args.bundle)
    print(f'bundle: {args.bundle}')
    print(f'  keys: {sorted(k for k in data)}')
    print()

    stages = {}
    for key in ('in', 'pr', 'gt'):
        if key in data and isinstance(data[key], dict) and 'betas' in data[key]:
            stages[key] = data[key]
    if not stages:
        raise SystemExit("no stage dicts with betas found; is this a CARI4D bundle?")

    if args.coconet:
        cc = load_bundle(args.coconet)
        if 'pr' in cc:
            stages['coconet'] = cc['pr']

    print('stage summaries:')
    for name in ('coconet', 'in', 'pr', 'gt'):
        if name in stages:
            stage_summary(name, stages[name])

    if 'in' not in stages:
        print("\nNOTE: no 'in' key. The optimizer only writes it on a fresh run,")
        print("      so this bundle is probably from a resumed run. Use --coconet")
        print("      to supply the pre-optimization state instead.")

    if 'coconet' in stages and 'in' in stages:
        compare('coconet', stages['coconet'], 'in (opt input)', stages['in'])
    if 'in' in stages and 'pr' in stages:
        compare('in (coconet out)', stages['in'], 'pr (final)', stages['pr'])
    elif 'coconet' in stages and 'pr' in stages:
        compare('coconet', stages['coconet'], 'pr (final)', stages['pr'])


if __name__ == '__main__':
    main()
