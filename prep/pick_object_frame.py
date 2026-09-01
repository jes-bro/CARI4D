"""Rank a clip's frames as candidates for object reconstruction, on one sheet.

Hunyuan3D sees exactly one frame. Everything the object's reconstructed shape
can ever be is decided by that crop, and a bad choice fails in ways that do not
announce themselves: on this project it once produced a flattened teardrop
instead of a basketball, from a crop that looked fine at a glance.

Picking that frame should not be a judgement made blind, once per clip, across
a hundred clips. So this scores every frame the object is masked in, and writes
ONE labelled contact sheet of the best candidates. A person looks at a single
image, reads off a frame number, and passes it as MESH_FRAME.

Candidates are scored across every 4K aux view, not the pipeline view, because
that is what scripts/recon_object.sh reconstructs from: the object is ~13 px
across in the 448 pipeline camera and ~110 px in the aux ones, and the geometry
is the same whichever camera saw it. Different cameras also see different
occlusions on the same frame, so the choice is a (view, frame) pair rather than
a frame.

The crops are produced by run_hy3d_recon's own crop_rgba, so what you are
looking at is exactly what the reconstruction would receive -- not an
approximation of it.

What the score rewards, and why:

  area        more pixels on the object is more shape information. A ball 13px
              across carries none at all.
  sharpness   variance of the Laplacian inside the crop. Motion blur destroys
              the surface detail single-image reconstruction depends on, and a
              layup is mostly fast motion.
  clipping    a mask touching the frame edge is a partial object; the missing
              part is unrecoverable and gets invented.
  roundness   reported, not scored. For a ball a low value means something is
              in front of it -- usually a hand. For other objects it means
              nothing, so it does not drive the ranking.

Run from the repo root in the cari4d env (newcari4d).

Usage:
    python prep/pick_object_frame.py --work work/<seq>
    python prep/pick_object_frame.py --work work/<seq> --top 12 --out sheet.png
"""
import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np

sys.path.append(os.getcwd())
from prep.run_hy3d_recon import crop_rgba, extract_frame, load_object_mask


def parse_args():
    """Parse the clip directory, how many candidates to show and where."""
    parser = argparse.ArgumentParser(
        description="Score and preview frames as object-reconstruction candidates")
    parser.add_argument("--work", required=True,
                        help="the clip's work directory, e.g. work/<seq>")
    parser.add_argument("--top", type=int, default=9,
                        help="candidates to put on the sheet (default: 9)")
    parser.add_argument("--stride", type=int, default=3,
                        help="score every Nth frame (default: 3). These are 4K clips "
                             "and every candidate costs a seek and a decode, so "
                             "scoring all of them in every view is slow for little "
                             "gain -- you are choosing between candidates, not "
                             "optimising to the frame")
    parser.add_argument("--views", default=None,
                        help="mask-set names to score, comma separated "
                             "(default: every cam*-4k set present)")
    parser.add_argument("--min_px", type=int, default=16,
                        help="ignore frames whose object mask is smaller (default: 16)")
    parser.add_argument("--tile", type=int, default=256, help="tile size on the sheet")
    parser.add_argument("--out", default=None,
                        help="contact sheet path (default: <work>/object_candidates.png)")
    parser.add_argument("--kid", type=int, default=0)
    return parser.parse_args()


def mask_metrics(mask):
    """Return (area, touches_border, roundness) for one object mask.

    Roundness is 4*pi*area / perimeter^2 -- 1.0 for a perfect disc, lower for a
    silhouette that is notched or elongated. For a ball that is a direct read on
    whether something is occluding it.
    """
    area = int(mask.sum())
    if area == 0:
        return 0, True, 0.0
    border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    roundness = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        if peri > 0:
            roundness = float(4.0 * np.pi * cv2.contourArea(c) / (peri * peri))
    return area, border, roundness


def sharpness(rgb, mask):
    """Variance of the Laplacian over the object's pixels only.

    Restricted to the mask because the background is usually a flat gym floor,
    which would dominate the statistic and rank a blurred ball on a busy
    background above a sharp one on a plain floor.
    """
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(grey, cv2.CV_64F)
    vals = lap[mask]
    return float(vals.var()) if vals.size else 0.0


def aux_views(masks_root, kid):
    """Return the 4K aux mask-set names present, which are what get scored."""
    suffix = f"_masks_k{kid}.h5"
    return sorted(f[:-len(suffix)] for f in os.listdir(masks_root)
                  if f.endswith(suffix) and f.startswith("cam") and "-4k" in f)


def score_frames(video, masks_root, seq, kid, stride, min_px):
    """Score every frame whose object mask is usable, best first.

    Returns a list of dicts. area and sharpness are normalised against the best
    frame in this clip rather than an absolute scale, because both depend
    entirely on how far away this particular camera was.
    """
    cap = cv2.VideoCapture(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    rows = []
    for idx in range(0, n_frames, stride):
        try:
            mask = load_object_mask(masks_root, seq, idx, kid)
        except Exception:
            continue
        if mask is None:
            continue
        mask = mask.astype(bool)
        area, border, roundness = mask_metrics(mask)
        if area < min_px:
            continue
        try:
            rgb = extract_frame(video, idx)
        except RuntimeError:
            continue    # a frame the decoder cannot seek to is not a candidate
        rows.append({"frame": idx, "area": area, "border": border,
                     "roundness": roundness, "sharp": sharpness(rgb, mask)})
    if not rows:
        return []

    max_area = max(r["area"] for r in rows)
    max_sharp = max(r["sharp"] for r in rows) or 1.0
    for r in rows:
        # Multiplicative: a frame needs BOTH size and focus, and being excellent
        # at one cannot buy its way out of being useless at the other.
        r["score"] = (r["area"] / max_area) * (r["sharp"] / max_sharp)
        if r["border"]:
            r["score"] *= 0.25   # heavily penalised, not excluded -- sometimes
                                 # every frame touches an edge and you still
                                 # have to pick one
    rows.sort(key=lambda r: -r["score"])
    return rows


def build_sheet(masks_root, kid, picks, tile, out_path):
    """Write one labelled contact sheet of the candidate crops.

    Each tile is the crop the reconstruction would actually be given, captioned
    with its frame number -- the number the reader is going to type back in.
    """
    tiles = []
    for r in picks:
        rgb = extract_frame(r["video"], r["frame"])
        # crop_rgba wants the mask as uint8 with 255 for the object, which is
        # what load_object_mask returns -- do not convert it to bool here.
        mask = load_object_mask(masks_root, r["view"], r["frame"], kid)
        rgba = np.array(crop_rgba(rgb, mask, crop_size=tile))
        img = rgba[:, :, :3].copy()
        # Composite onto grey: the crop is transparent outside the object, and
        # black or white backing hides exactly the silhouette edges worth
        # judging.
        alpha = (rgba[:, :, 3:4] / 255.0) if rgba.shape[2] == 4 else 1.0
        img = (img * alpha + 128 * (1 - alpha)).astype(np.uint8)
        cv2.rectangle(img, (0, 0), (tile - 1, 22), (0, 0, 0), -1)
        cv2.putText(img, f"{r['view'].replace('-4k','')} f{r['frame']}  "
                    f"round {r['roundness']:.2f}",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(img)

    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows_n = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows_n * tile, cols * tile, 3), 40, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = t
    cv2.imwrite(out_path, sheet[:, :, ::-1])
    return out_path


def main():
    """Score every aux view's frames, print the ranking and write the sheet."""
    args = parse_args()
    work = osp.normpath(args.work)
    seq = osp.basename(work)
    masks_root = osp.join(work, "masks")
    if not osp.isdir(masks_root):
        raise SystemExit(f"ERROR: no {masks_root}; has this clip been through stage 1?")

    views = args.views.split(",") if args.views else aux_views(masks_root, args.kid)
    if not views:
        raise SystemExit(
            f"ERROR: no cam*-4k mask sets in {masks_root}. Stage 1b (recon_masks.sh) "
            f"writes them, and the object is reconstructed from a 4K aux view "
            f"because it is ~8x larger there than in the pipeline camera.")

    rows = []
    for view in views:
        video = osp.join(masks_root, "trimmed_vids", f"{view}.0.color.mp4")
        if not osp.isfile(video):
            print(f"  (no clip for {view}, skipping)", file=sys.stderr)
            continue
        got = score_frames(video, masks_root, view, args.kid, args.stride, args.min_px)
        for r in got:
            r["view"] = view
            r["video"] = video
        rows.extend(got)
    if not rows:
        raise SystemExit(f"ERROR: no frame in any aux view has an object mask of "
                         f"{args.min_px}+ px")

    # Re-normalise across views: score_frames normalised within each view, so a
    # view that never sees the object well would otherwise field a "best" frame
    # scoring 1.0 alongside a genuinely good one.
    max_area = max(r["area"] for r in rows)
    max_sharp = max(r["sharp"] for r in rows) or 1.0
    for r in rows:
        r["score"] = (r["area"] / max_area) * (r["sharp"] / max_sharp)
        if r["border"]:
            r["score"] *= 0.25
    rows.sort(key=lambda r: -r["score"])

    picks = rows[:args.top]
    print(f"{seq}: {len(rows)} candidates across {len(views)} view(s)\n")
    print(f"{'view':<12} {'frame':>6} {'score':>7} {'area_px':>8} {'sharp':>9} "
          f"{'round':>6}  note")
    print("-" * 70)
    for r in picks:
        note = "TOUCHES EDGE" if r["border"] else ""
        if r["roundness"] and r["roundness"] < 0.7:
            note = (note + " occluded?").strip()
        print(f"{r['view']:<12} {r['frame']:>6} {r['score']:>7.3f} {r['area']:>8} "
              f"{r['sharp']:>9.1f} {r['roundness']:>6.2f}  {note}")

    out = args.out or osp.join(work, "object_candidates.png")
    build_sheet(masks_root, args.kid, picks, args.tile, out)
    print(f"\nsheet: {osp.abspath(out)}")
    best = picks[0]
    print(f"\nLook at the sheet, then reconstruct from the tile you like:")
    print(f"  MESH_CAM={best['view'].replace('-4k', '')} MESH_FRAME={best['frame']} "
          f"TAKE=<take> SEQ={seq} bash scripts/recon_object.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
