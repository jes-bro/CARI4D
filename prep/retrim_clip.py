"""Cut an existing clip down to a sub-window, every view at once.

A clip's boundaries come from the pipeline camera, which sees the ball in
frames no other camera can. Those frames enter the clip, triangulate to
nothing, and fall back to monocular depth -- on
Date03_Sub01_bball_rev003 that was the first two seconds, and it was visibly
wrong in the replay while the rest was fine.

prep/check_view_coverage.py finds where two cameras can actually place the
object. This carves that window out as a new clip, so a clip with a dead head
becomes a shorter usable one instead of a discard.

Nothing is recomputed. The masks and clips for every view already exist and are
already frame-aligned, so this is slicing and renumbering -- CPU only, seconds.
That also means you are not forced to decide before seeing a reconstruction:
run the full clip, watch it, and retrim afterwards for the cost of one stage 3,
not a re-mask.

Frame numbers are indices into THIS clip (0-based), which is what
check_view_coverage reports. The new clip's window.json records the
corresponding range in the original take, so provenance survives the cut.

Run from the repo root in an env with h5py and imageio (newcari4d).

Usage:
    python prep/check_view_coverage.py --masks_root work/<seq>/masks --seq <seq>
    python prep/retrim_clip.py --work work/<seq> --lo 60 --hi 182
    python prep/retrim_clip.py --work work/<seq> --lo 60 --hi 182 --new_seq <name>

Writes work/<new_seq>/ with the same shape stage 1a emits, so the later stages
take it unchanged -- pass the new name as SEQ and carry on from stage 2.
"""
import argparse
import json
import os
import os.path as osp
import shutil
import sys

import h5py
import imageio
import numpy as np

sys.path.append(os.getcwd())

# The mask layout this reads and writes is run_sam3_masks.save_masks_h5's:
# one group per sequence, datasets "<frame:06d>-k<kid>.person_mask.png" and
# ".obj_rend_mask.png". It is duplicated rather than imported because
# run_sam3_masks imports sam3 at module level, which needs a different Python
# than the env this runs in.
PERSON = "person_mask.png"
OBJECT = "obj_rend_mask.png"


def parse_args():
    """Parse the clip directory, the sub-window and the new clip's name."""
    parser = argparse.ArgumentParser(
        description="Slice a clip and all its views down to a sub-window")
    parser.add_argument("--work", required=True,
                        help="the clip's work directory, e.g. work/<seq>")
    parser.add_argument("--lo", type=int, required=True,
                        help="first frame to keep, indexed within this clip")
    parser.add_argument("--hi", type=int, required=True,
                        help="last frame to keep, inclusive")
    parser.add_argument("--new_seq", default=None,
                        help="name for the result (default: the clip's name + 't'). "
                             "Must keep the Date_Sub_object_action shape -- part 2 is "
                             "read as the SMPL gender key and part 3 as the object")
    parser.add_argument("--kid", type=int, default=0, help="camera id in the mask keys")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--dry_run", action="store_true",
                        help="report what would be written and stop")
    return parser.parse_args()


def clip_seq_name(work):
    """Return the sequence name a clip directory belongs to."""
    return osp.basename(osp.normpath(work))


def mask_sets(masks_dir, kid):
    """Return every mask-set name in a clip's masks directory."""
    suffix = f"_masks_k{kid}.h5"
    return sorted(f[:-len(suffix)] for f in os.listdir(masks_dir)
                  if f.endswith(suffix))


def slice_masks(src_h5, src_group, dst_h5, dst_group, lo, hi, kid):
    """Copy frames [lo, hi] of one mask set into a new file, renumbered from 0.

    Renumbering matters: a clip's frames start at 0 by convention, and every
    reader indexes them that way. Leaving the original numbers would make the
    masks silently disagree with the video beside them.
    """
    kept = 0
    with h5py.File(src_h5, "r") as fin, h5py.File(dst_h5, "w") as fout:
        gin = fin[src_group]
        gout = fout.create_group(dst_group)
        for i, frame in enumerate(range(lo, hi + 1)):
            for kind in (PERSON, OBJECT):
                key = f"{frame:06d}-k{kid}.{kind}"
                if key not in gin:
                    continue
                gout.create_dataset(f"{i:06d}-k{kid}.{kind}", data=gin[key][()])
            kept += 1
    return kept


def slice_video(src, dst, lo, hi, fps):
    """Write frames [lo, hi] of a video to a new file.

    macro_block_size=1 because the default silently resizes to a multiple of
    16 -- egoexo4d's 796-wide clips became 800 and stopped matching their own
    masks, which fails much later and much less clearly.
    """
    reader = imageio.get_reader(src)
    writer = imageio.get_writer(dst, fps=fps, macro_block_size=1)
    n = 0
    for i, frame in enumerate(reader):
        if i < lo:
            continue
        if i > hi:
            break
        writer.append_data(frame)
        n += 1
    writer.close()
    reader.close()
    return n


def take_range(work, lo, hi):
    """Map a clip-relative window back to frame numbers in the original take.

    Reads the clip's own window.json, so a retrimmed clip still records where
    in the source take it came from -- otherwise the second cut would erase the
    provenance the first one recorded.
    """
    path = osp.join(work, "window.json")
    if not osp.isfile(path):
        return None, None
    with open(path) as f:
        chosen = json.load(f).get("chosen")
    if not chosen:
        return None, None
    return chosen["lo"] + lo, chosen["lo"] + hi


def main():
    """Slice every view of a clip to the window and write a new clip directory."""
    args = parse_args()
    work = osp.normpath(args.work)
    seq = clip_seq_name(work)
    new_seq = args.new_seq or f"{seq}t"
    if args.hi < args.lo:
        raise SystemExit(f"ERROR: --hi {args.hi} is before --lo {args.lo}")

    masks_dir = osp.join(work, "masks")
    clips_dir = osp.join(masks_dir, "trimmed_vids")
    for d in (masks_dir, clips_dir):
        if not osp.isdir(d):
            raise SystemExit(f"ERROR: {d} does not exist; is {work} a clip directory?")

    new_work = osp.join(osp.dirname(work), new_seq)
    new_masks = osp.join(new_work, "masks")
    new_clips = osp.join(new_masks, "trimmed_vids")
    if osp.exists(new_work) and not args.dry_run:
        raise SystemExit(f"ERROR: {new_work} already exists; remove it or pass "
                         f"a different --new_seq")

    views = mask_sets(masks_dir, args.kid)
    n_new = args.hi - args.lo + 1
    t_lo, t_hi = take_range(work, args.lo, args.hi)
    print(f"{seq} -> {new_seq}")
    print(f"  keeping clip frames {args.lo}-{args.hi} ({n_new} frames, "
          f"{n_new / args.fps:.1f}s)")
    if t_lo is not None:
        print(f"  which is take frames {t_lo}-{t_hi}")
    print(f"  views: {', '.join(views)}")
    if args.dry_run:
        print(f"  would write {new_work}")
        return 0

    os.makedirs(new_clips, exist_ok=True)
    for view in views:
        # The pipeline view is named for the clip, so it is renamed with it;
        # aux views are named for their camera and keep their names.
        dst_view = new_seq if view == seq else view
        kept = slice_masks(
            osp.join(masks_dir, f"{view}_masks_k{args.kid}.h5"), view,
            osp.join(new_masks, f"{dst_view}_masks_k{args.kid}.h5"), dst_view,
            args.lo, args.hi, args.kid)
        src_vid = osp.join(clips_dir, f"{view}.0.color.mp4")
        n_vid = 0
        if osp.isfile(src_vid):
            n_vid = slice_video(src_vid, osp.join(new_clips, f"{dst_view}.0.color.mp4"),
                                args.lo, args.hi, args.fps)
        print(f"    {view:<28} -> {dst_view:<28} {kept} masks, {n_vid} frames")
        if n_vid and n_vid != kept:
            print(f"      WARNING: {n_vid} video frames against {kept} mask frames",
                  file=sys.stderr)

    with open(osp.join(new_work, "window.json"), "w") as f:
        json.dump({
            "seq": new_seq,
            "source_video": osp.abspath(osp.join(clips_dir, f"{seq}.0.color.mp4")),
            "num_frames": n_new,
            "fps": args.fps,
            "both_masks_frames": n_new,
            "both_masks_frac": 1.0,
            "runs": [],
            "chosen": {"lo": t_lo if t_lo is not None else args.lo,
                       "hi": t_hi if t_hi is not None else args.hi,
                       "n_frames": n_new, "covered_frac": 1.0},
            "retrimmed_from": {"seq": seq, "lo": args.lo, "hi": args.hi},
        }, f, indent=1)

    print(f"\nwrote {new_work}")
    # Which stage comes next depends on what there was to slice. Retrimming
    # after stage 1b carries every view across and geometry can run; retrimming
    # a clip that has only ever seen stage 1a carries one view, and geometry
    # would fail on it for want of anything to triangulate against. Doing it in
    # that order is legitimate and cheaper -- the aux views then get cut to the
    # tightened window and the dead frames are never masked at all -- so this
    # says which case you are in rather than assuming.
    if len(views) > 1:
        print(f"next: TAKE=<take> SEQ={new_seq} bash scripts/recon_geometry.sh")
    else:
        print(f"only the pipeline view was present, so this clip has no aux masks yet.")
        print(f"next: TAKE=<take> SEQ={new_seq} bash scripts/recon_masks.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
