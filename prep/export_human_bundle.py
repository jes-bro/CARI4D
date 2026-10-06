"""Write a human-only sequence as a CARI4D bundle, so the usual tools can read it.

The object pipeline ends in output/opt/<...>/<seq>.pth, a torch file holding
{'gt', 'pr', 'in'} dicts with smpl_pose, smpl_t, betas, frames and pose_abs.
Everything downstream -- tools/viz_pred.py, tools/compare_bundle_stages.py,
InterMimic's cari4d_to_interact.py -- reads that shape. A human-only fit (a
dancer: no object, so no CoCoNet and no optimizer) ends in
<nlf>-opt/<seq>_params.pkl from prep/fit_smplh_global.py instead. This
rewrites it in the bundle shape:

    python prep/export_human_bundle.py --params <nlf>-opt/<seq>_params.pkl \\
        --out output/human/<seq>.pth

pose_abs is the identity for every frame and `human_only` is set, so a reader
can tell there is no object rather than an object at the camera origin.
smpl_pose is the full 156-dim SMPL-H vector (the fit keeps the NLF hands);
the optimizer's bundles carry 72 and downstream converters accept either.
"""
import argparse
import os
import os.path as osp
import sys

import joblib
import numpy as np
import torch

sys.path.append(os.getcwd())


def params_to_bundle(params, seq):
    """The {'pr','in','gt'} dict a CARI4D bundle holds, from fit_smplh_global's pkl."""
    poses = np.asarray(params['poses'])[:, 0]        # (T, 156)
    trans = np.asarray(params['transls'])[:, 0]      # (T, 3)
    betas = np.asarray(params['betas'])[:, 0]        # (T, 10)
    T = len(poses)
    frames = [f"{seq}/{i:06d}" for i in range(T)]
    pr = {
        'smpl_pose': torch.from_numpy(poses).float(),
        'smpl_t': torch.from_numpy(trans).float(),
        'betas': torch.from_numpy(betas).float(),
        'pose_abs': torch.eye(4).repeat(T, 1, 1),
        'contact_logits': torch.zeros(T, 2),
        'frames': frames,
        'gender': params.get('gender', 'neutral'),
        'human_only': True,
    }
    # 'gt' is a copy rather than {}: tools/viz_pred.py reads smpl_pose, smpl_t,
    # betas and pose_abs out of it unconditionally. There is no ground truth
    # for a wild clip; see prep/export_direct_bundle.py for why the readers
    # expect one anyway.
    return {'pr': pr, 'in': dict(pr), 'gt': dict(pr)}


def main():
    """Convert one params pkl to a bundle .pth."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', required=True, help='<nlf>-opt/<seq>_params.pkl')
    ap.add_argument('--out', required=True, help='bundle .pth to write')
    args = ap.parse_args()
    seq = osp.basename(args.params).replace('_params.pkl', '')
    params = joblib.load(args.params)
    bundle = params_to_bundle(params, seq)
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    torch.save(bundle, args.out)
    T = len(bundle['pr']['frames'])
    print(f"wrote {args.out}: {T} frames, gender {bundle['pr']['gender']}, human only")


if __name__ == '__main__':
    main()
