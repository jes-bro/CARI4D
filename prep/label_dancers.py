"""Say which of two tracked people is the EgoExo4D participant, from the Aria glasses.

A partner-dance take has two people in every exo view and SAM3 (run with
--human_instances 2) tracks both, but its slot order is by size at the first
chunk, which says nothing about who is who. The participant is the one wearing
the Aria glasses, and the take records exactly where those glasses were:
trajectory/aria_extrinsics.json holds the Aria RGB camera's world-to-camera
pose per take frame, in the same world frame as gopro_calibs.csv. Projecting
the camera centre into an exo view puts a pixel on the wearer's head; whichever
person mask that pixel lands on, or is nearest to, is the participant.

    python prep/label_dancers.py --take_dir <takes>/uniandes_dance_022_41 \\
        --view cam01:work/<seqA>/masks/<seqA>_masks_k0.h5:work/<seqA>/masks/<seqB>_masks_k0.h5 \\
        --frame_offset <lo from window.json>            # take frame of clip frame 0
        [--view cam03:<A 4k h5>:<B 4k h5> ...] [--apply]

Every --view votes on every frame; the report is the fraction of votes for
each slot and the median pixel margin between the two, per view and overall.
--apply then renames the two mask files so the FIRST sequence is the
participant, when the vote says they are the wrong way round, and writes
dancers.json beside them recording the decision. Without --apply nothing is
touched.

WHY THE HEAD, NOT THE WHOLE MASK. In a close hold the two bodies overlap and
their centroids are 30 cm apart; the glasses are on one head and heads are
apart even when hips are not. The score is the distance from the projected
point to the nearest pixel in the top fifth of each mask, so a mask whose head
is under the point wins even if the other mask's torso is nearer overall.

FRAME OFFSET. aria_extrinsics.json is keyed by TAKE frame; a clip cut by stage
1 starts at window.json's `lo`. Pass that as --frame_offset. The Aria stream
and the exo streams are frame-aligned by EgoExo4D, so no further sync is
needed; prep/validate_aria_reprojection.py checks that assumption on a take.
"""
import argparse
import json
import os
import os.path as osp
import sys

import cv2
import h5py
import numpy as np

sys.path.append(os.getcwd())

from prep.aria_camera import load_extrinsics  # noqa: E402
from prep.triangulate_object import read_calibration  # noqa: E402


def aria_centres(take_dir):
    """{take frame: Aria RGB camera centre in world} from aria_extrinsics.json."""
    ext = load_extrinsics(osp.join(take_dir, "trajectory", "aria_extrinsics.json"))
    return {f: -R.T @ t for f, (R, t) in ext.items()}


def project(cam, point_world):
    """Pixel of a world point in a Kannala-Brandt exo camera, or None if behind it."""
    p = cam["R_cw"] @ point_world + cam["t_cw"]
    if p[2] <= 1e-6:
        return None
    uv, _ = cv2.fisheye.projectPoints(p.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                      cam["K"], cam["dist"])
    return uv.ravel()


def h5_frames(path):
    """(group, sorted frame indices) for a mask H5 opened read-only."""
    f = h5py.File(path, "r")
    seq = osp.basename(path).split("_masks_k")[0]
    g = f[seq] if seq in f else f
    frames = sorted(int(k.split("-")[0]) for k in g if k.endswith(".person_mask.png"))
    return f, g, frames


def head_distance(mask, uv, top_frac=0.2):
    """Distance in px from uv to the nearest pixel of the mask's top fifth, or inf."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.inf
    y_cut = ys.min() + top_frac * (ys.max() - ys.min() + 1)
    keep = ys <= y_cut
    d = np.hypot(xs[keep] - uv[0], ys[keep] - uv[1])
    return float(d.min())


def vote_view(spec, cams_by_res, centres, offset, stride):
    """Votes for one exo view: per frame, which slot's head is nearer the wearer.

    Returns a dict with the per-view tally and the median margin, where margin
    is (distance to the loser) - (distance to the winner) in that view's pixels.
    """
    cam_uid, h5a, h5b = spec.split(":")
    fa, ga, frames_a = h5_frames(h5a)
    fb, gb, frames_b = h5_frames(h5b)
    frames = sorted(set(frames_a) & set(frames_b))[::max(1, stride)]
    if not frames:
        raise SystemExit(f"{spec}: the two mask files share no frames")
    sample = ga[f"{frames[0]:06d}-k0.person_mask.png"]
    H, W = sample.shape
    cams = cams_by_res.setdefault((W, H), None)
    votes = {"a": 0, "b": 0, "skipped": 0}
    margins = []
    for fr in frames:
        c = centres.get(fr + offset)
        if c is None:
            votes["skipped"] += 1
            continue
        uv = project(cams[cam_uid], c)
        if uv is None or not (0 <= uv[0] < W and 0 <= uv[1] < H):
            votes["skipped"] += 1
            continue
        ma = ga[f"{fr:06d}-k0.person_mask.png"][()].astype(bool)
        mb = gb[f"{fr:06d}-k0.person_mask.png"][()].astype(bool)
        da, db = head_distance(ma, uv), head_distance(mb, uv)
        if not np.isfinite(da) and not np.isfinite(db):
            votes["skipped"] += 1
            continue
        if da <= db:
            votes["a"] += 1
            margins.append(db - da if np.isfinite(db) else np.inf)
        else:
            votes["b"] += 1
            margins.append(da - db if np.isfinite(da) else np.inf)
    fa.close()
    fb.close()
    finite = [m for m in margins if np.isfinite(m)]
    return {"view": cam_uid, "frames": len(frames), **votes,
            "median_margin_px": float(np.median(finite)) if finite else None,
            "width": W, "height": H}


def main():
    """Vote across views, report, and optionally swap the files so A is the participant."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take_dir", required=True, help="the take: trajectory/ holds calibs + aria extrinsics")
    ap.add_argument("--view", action="append", required=True,
                    help="<cam_uid>:<h5 of person A>:<h5 of person B>; repeat per view")
    ap.add_argument("--frame_offset", type=int, required=True,
                    help="take frame index of the clip's frame 0 (window.json lo)")
    ap.add_argument("--stride", type=int, default=3, help="vote every Nth frame (default 3)")
    ap.add_argument("--min_agreement", type=float, default=0.75,
                    help="fraction of votes one slot needs before --apply will act")
    ap.add_argument("--apply", action="store_true", help="rename the files so the FIRST is the participant")
    args = ap.parse_args()

    calib = osp.join(args.take_dir, "trajectory", "gopro_calibs.csv")
    centres = aria_centres(args.take_dir)
    # calibration is per resolution: read once per (W, H) the mask sets come in
    cams_by_res = {}
    results = []
    for spec in args.view:
        cam_uid, h5a, _ = spec.split(":")
        with h5py.File(h5a, "r") as f:
            seq = osp.basename(h5a).split("_masks_k")[0]
            g = f[seq] if seq in f else f
            k = next(k for k in g if k.endswith(".person_mask.png"))
            H, W = g[k].shape
        if cams_by_res.get((W, H)) is None:
            cams_by_res[(W, H)] = read_calibration(calib, W, H)
        r = vote_view(spec, cams_by_res, centres, args.frame_offset, args.stride)
        results.append(r)
        margin = "-" if r["median_margin_px"] is None else f"{r['median_margin_px']:.0f} px"
        print(f"{r['view']:6s} {r['width']}x{r['height']}: A {r['a']:4d}  B {r['b']:4d}  "
              f"skipped {r['skipped']:3d}  median margin {margin}")

    a = sum(r["a"] for r in results)
    b = sum(r["b"] for r in results)
    total = a + b
    if total == 0:
        raise SystemExit("no frame could be voted on: is --frame_offset right, and is the "
                         "wearer inside these views?")
    winner = "a" if a >= b else "b"
    frac = max(a, b) / total
    print(f"\nparticipant = person {winner.upper()}  ({frac * 100:.0f}% of {total} votes)")
    if frac < args.min_agreement:
        print(f"WARNING: below --min_agreement {args.min_agreement:.2f}; the two people may swap "
              "slots mid-clip (a chunk boundary where SAM3 re-seeded). Look at the overlay "
              "video around the chunk edges before trusting either sequence.")

    decision = {"participant_slot": winner, "votes_a": a, "votes_b": b, "agreement": frac,
                "frame_offset": args.frame_offset, "views": results, "applied": False}
    first_a = args.view[0].split(":")[1]
    first_b = args.view[0].split(":")[2]
    if args.apply and winner == "b" and frac >= args.min_agreement:
        # Swap every pair, via a temporary name so neither overwrites the other.
        for spec in args.view:
            _, ha, hb = spec.split(":")
            tmp = ha + ".swap"
            os.rename(ha, tmp)
            os.rename(hb, ha)
            os.rename(tmp, hb)
            print(f"swapped {osp.basename(ha)} <-> {osp.basename(hb)}")
        decision["applied"] = True
        print("the FIRST sequence of every view now holds the participant")
    elif args.apply:
        print("no swap needed" if winner == "a" else "not applied: agreement too low")
        decision["applied"] = winner == "a"
    out = osp.join(osp.dirname(first_a), "dancers.json")
    with open(out, "w") as f:
        json.dump(decision, f, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
