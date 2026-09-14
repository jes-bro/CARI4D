"""Reconstruct a STATIC prop (chair, bench, CPR manikin, piano) as one held pose.

The pipeline tracks one moving object per sequence and has no notion of a
thing that does not move: a chair under a seated guitarist would be tracked
frame by frame through heavy occlusion, and inside the optimizer the
penetration loss would push the sitter off it (contact is wrist-only). So a
prop is reconstructed BESIDE the pipeline, not inside it, the way the
basketball hoop was handled: one mesh, one pose, handed to the simulator as
static collision geometry. The human reconstruction is untouched.

Three subcommands, run in order, each reading the previous one's files under
<work>/props/<prop>/ (all paths come from scripts/recon_prop.sh):

  geometry   triangulate the prop's mask centroid across the calibrated views
             (prep/triangulate_object.py, unchanged), pool the per-frame
             positions to ONE world point, and pick the aux view + frame to
             build the mesh from -- the frame where the prop is largest, least
             covered by the person, and not cut by the image border.
             CPU only. Writes prop_xyz.npz, prop_world.json, mesh_pick.json.

  register   warp the prop masks into the rectified pipeline camera, run
             FoundationPose registration (not tracking) on the K best frames of
             the depth-aligned clip, take the medoid pose, and write the
             sidecar. GPU. Needs the metric mesh (scripts/slurm_scale_object.sh
             on the prop's own roots) and the main pipeline's rect-aligned/
             clip, since that is the depth it registers against.

  check      print what exists and the agreement numbers, no GPU.

THE SIDECAR: <work>/props/<prop>/<seq>_<prop>.json

  {
    "prop": "chair", "seq": ..., "take": ..., "camera": "<pipe_cam>",
    "frame": "rectified pipeline camera; rectification uses R=I so the
              extrinsics of <pipe_cam> in gopro_calibs.csv apply unchanged",
    "mesh": "<path to the metric _align.obj>",
    "mesh_convention": "pose applies to the mesh's vertices exactly as stored:
                        X_cam = R @ X_obj + t (FoundationPose register output,
                        which already folds in its own bbox-centring)",
    "mesh_centroid_obj": [..],      # so a consumer that centres by centroid can re-centre
    "pose_cam": [[4x4]],             # the held pose, same frame as the bundle's pose_abs
    "world_xyz": [..],              # pooled triangulated centroid, world frame
    "frames_registered": [..], "n_poses": K,
    "checks": {"register_spread_m": ..., "register_spread_deg": ...,
               "tri_vs_register_m": ..., "tri_spread_m": ..., "tri_frames": N}
  }

pose_cam is in the SAME frame as pose_abs in output/opt/<...>/<seq>.pth, so a
consumer already placing the tracked object can place the prop with the same
transform. world_xyz is in the world frame of the take (the frame
object_xyz.npz and triangulate_points.py use) for anything that works there.

DEPTH FOR REGISTRATION. The aligned depth is UniDepth, scaled to the human;
a prop metres from the person may sit at the wrong absolute distance in it.
--depth_mode shift (default) keeps UniDepth's relative structure inside the
prop mask but moves its median to the triangulated Z, so registration sees a
plausibly shaped surface at the right distance. 'inject' flattens the mask to
the triangulated Z (what prep/inject_object_depth.py does for the tracked
object); 'aligned' uses the depth as is.

WHAT TO EXPECT. The Hunyuan3D mesh comes from one RGBA crop, and a chair with
someone on it has a body-shaped hole in its silhouette that the model fills
with invented geometry. geometry's frame pick prefers frames before the person
sits; if the clip has none, the mesh will be rough where the body was. For the
simulator that is usually fine -- the seat and legs are what carry weight --
but look at the _rgba.png the way recon_object.sh's CHECK block says to.
"""
import argparse
import json
import os
import os.path as osp
import subprocess
import sys

import cv2
import h5py
import numpy as np

sys.path.append(os.getcwd())

from prep.select_recon_frame import (object_extent, person_contact, solidity,  # noqa: E402
                                     touches_border)
from prep.triangulate_object import read_calibration  # noqa: E402

MASK_SUFFIX_OBJ = "obj_rend_mask.png"
MASK_SUFFIX_PERSON = "person_mask.png"


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------

def log(msg):
    """Print one prefixed line, flushed, so a killed job keeps what it said."""
    print(f"[prop] {msg}", flush=True)


def h5_group(f, seq):
    """The group holding a sequence's masks: `<seq>/` if present, else the root."""
    return f[seq] if seq in f else f


def mask_frames(h5_path, seq, kid=0):
    """Sorted frame indices that have an object mask in the H5."""
    tail = f"-k{kid}.{MASK_SUFFIX_OBJ}"
    with h5py.File(h5_path, "r") as f:
        g = h5_group(f, seq)
        return sorted(int(k.split("-")[0]) for k in g if k.endswith(tail))


def read_mask(g, frame, kid, suffix):
    """One boolean mask from an open H5 group, or None when the key is absent."""
    key = f"{frame:06d}-k{kid}.{suffix}"
    if key not in g:
        return None
    return g[key][()].astype(bool)


def world_to_cam(cam, xyz_world):
    """x_cam = R_cw @ x_world + t_cw, with `cam` from read_calibration."""
    return cam["R_cw"] @ np.asarray(xyz_world, float) + cam["t_cw"]


def rotation_angle_deg(R1, R2):
    """Geodesic angle between two rotation matrices, in degrees."""
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def medoid_pose(poses):
    """The pose closest to all the others, plus the spread around it.

    A medoid rather than a mean: averaging rotations that disagree by 180
    degrees (a symmetric-looking chair registered back to front) produces a
    rotation that matches none of them. Distance is translation in metres plus
    rotation in radians, which weights 1 m like 57 degrees -- crude, but both
    terms are tiny for poses that agree and large for the one that does not.
    """
    P = np.asarray(poses, float)
    n = len(P)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                dt = np.linalg.norm(P[i, :3, 3] - P[j, :3, 3])
                dr = np.radians(rotation_angle_deg(P[i, :3, :3], P[j, :3, :3]))
                D[i, j] = D[j, i] = dt + dr
    k = int(np.argmin(D.sum(1)))
    others = [i for i in range(n) if i != k]
    spread_m = float(np.median([np.linalg.norm(P[i, :3, 3] - P[k, :3, 3]) for i in others])) if others else 0.0
    spread_deg = float(np.median([rotation_angle_deg(P[i, :3, :3], P[k, :3, :3]) for i in others])) if others else 0.0
    return P[k], k, spread_m, spread_deg


def pool_positions(xyz):
    """Median world position of a static prop, and how far the frames stray.

    The per-frame triangulation of a static object should not move; when it
    does, that is a mask or calibration problem, so the spread is reported as
    a check rather than smoothed away. Median, not mean: a few frames where the
    mask caught the person's leg pull a mean but not a median.
    """
    xyz = np.asarray(xyz, float)
    centre = np.median(xyz, axis=0)
    dev = np.linalg.norm(xyz - centre, axis=1)
    return centre, float(np.median(dev)), float(dev.max())


# ----------------------------------------------------------------------------
# geometry: triangulate, pool, pick the mesh frame
# ----------------------------------------------------------------------------

def run_triangulation(args):
    """Shell out to prep/triangulate_object.py on the prop's own mask sets.

    The same tool and the same --view syntax the main object uses, pointed at
    the prop's masks directory. The pipeline camera is first so the npz frame
    numbers are the pipeline clip's.
    """
    views = ["--view", f"{args.pipe_cam}:{args.masks_dir}:{args.seq}"]
    for c in args.aux_cams:
        if osp.isfile(osp.join(args.masks_dir, f"{c}-4k_masks_k0.h5")):
            views += ["--view", f"{c}:{args.masks_dir}:{c}-4k"]
        else:
            log(f"no prop masks for {c}, skipping that view")
    cmd = [sys.executable, "prep/triangulate_object.py", "--calib", args.calib, *views,
           "--width", str(args.width), "--height", str(args.height),
           "--min_px", str(args.min_px), "--out", args.xyz_out]
    if args.inlier_px:
        cmd += ["--inlier_px", str(args.inlier_px)]
    log("triangulating: " + " ".join(cmd[1:]))
    subprocess.run(cmd, check=True)


def score_view_frames(h5_path, seq, kid, stride, min_px, dilate):
    """Score every stride-th frame of one aux view for mesh suitability.

    Streams the 4K masks one at a time -- a whole view's masks would be
    gigabytes -- and measures each with prep/select_recon_frame's helpers:
    extent (bigger is better), person contact (the fraction of the prop's
    outline the person touches, worse), solidity (a chair with a body-shaped
    bite is less solid than the same chair empty), and the image border.
    """
    rows = []
    with h5py.File(h5_path, "r") as f:
        g = h5_group(f, seq)
        frames = sorted(int(k.split("-")[0]) for k in g if k.endswith(f"-k{kid}.{MASK_SUFFIX_OBJ}"))
        for fr in frames[::max(1, stride)]:
            obj = read_mask(g, fr, kid, MASK_SUFFIX_OBJ)
            if obj is None or obj.sum() < min_px:
                continue
            per = read_mask(g, fr, kid, MASK_SUFFIX_PERSON)
            per = per if per is not None else np.zeros_like(obj)
            rows.append({"frame": fr, "area": int(obj.sum()), "extent": float(object_extent(obj)),
                         "contact": float(person_contact(obj, per, dilate)),
                         "solidity": float(solidity(obj)), "border": bool(touches_border(obj))})
    return rows


def pick_mesh_frame(args):
    """Choose (aux view, frame) for Hunyuan3D across every aux view.

    Score = extent (normalised over all views, so a camera that sees the prop
    big wins) x (1 - contact) x solidity x (0.25 if the prop touches the image
    border). Written to mesh_pick.json with the runner-up per view, so a bad
    top pick can be swapped by hand.
    """
    per_view = {}
    for c in args.aux_cams:
        h5 = osp.join(args.masks_dir, f"{c}-4k_masks_k0.h5")
        if not osp.isfile(h5):
            continue
        rows = score_view_frames(h5, f"{c}-4k", 0, args.pick_stride, args.min_px, args.pick_dilate)
        if rows:
            per_view[c] = rows
    if not per_view:
        raise SystemExit("no aux view has prop masks; did the masks stage run?")
    max_extent = max(r["extent"] for rows in per_view.values() for r in rows) or 1.0
    ranked = []
    for c, rows in per_view.items():
        for r in rows:
            s = (r["extent"] / max_extent) * (1.0 - r["contact"]) * max(r["solidity"], 1e-3)
            s *= 0.25 if r["border"] else 1.0
            ranked.append(dict(r, cam=c, score=float(s)))
    ranked.sort(key=lambda r: -r["score"])
    best = ranked[0]
    pick = {"cam": best["cam"], "frame": best["frame"], "score": best["score"],
            "contact": best["contact"], "solidity": best["solidity"], "extent_px": best["extent"],
            "runners_up": ranked[1:6],
            "per_view_best": {c: max(rows, key=lambda r: (1 - r["contact"]) * r["extent"])
                              for c, rows in per_view.items()}}
    with open(args.pick_out, "w") as f:
        json.dump(pick, f, indent=1)
    log(f"mesh frame: {best['cam']} frame {best['frame']}  contact={best['contact']:.2f} "
        f"solidity={best['solidity']:.2f} extent={best['extent']:.0f}px  -> {args.pick_out}")
    if best["contact"] > 0.3:
        log("WARNING: the person touches >30% of the prop outline in the best frame; "
            "expect Hunyuan3D to invent geometry where the body was")
    return pick


def cmd_geometry(args):
    """Triangulate, pool to one world point, pick the mesh frame."""
    os.makedirs(osp.dirname(args.xyz_out), exist_ok=True)
    run_triangulation(args)
    d = np.load(args.xyz_out)
    if len(d["frames"]) == 0:
        raise SystemExit("triangulation covered no frames; check the prop masks per view")
    centre, spread_med, spread_max = pool_positions(d["xyz"])
    cams = read_calibration(args.calib, 1, 1)
    pipe = cams[args.pipe_cam]
    out = {"prop": args.prop, "seq": args.seq, "world_xyz": centre.tolist(),
           "cam_xyz": world_to_cam(pipe, centre).tolist(), "camera": args.pipe_cam,
           "tri_frames": int(len(d["frames"])), "tri_spread_median_m": spread_med,
           "tri_spread_max_m": spread_max, "tri_residual_median_px": float(np.median(d["residual"]))}
    with open(args.world_out, "w") as f:
        json.dump(out, f, indent=1)
    log(f"pooled {out['tri_frames']} frames -> world {np.round(centre, 3).tolist()}  "
        f"spread median {spread_med:.3f} m, max {spread_max:.3f} m")
    if spread_med > args.static_tol:
        log(f"WARNING: a static prop should not wander {spread_med:.2f} m between frames; "
            "look at the prop masks (person's leg? second chair?) before trusting this")
    pick_mesh_frame(args)


# ----------------------------------------------------------------------------
# register: FoundationPose on the K best frames, medoid, sidecar
# ----------------------------------------------------------------------------

def rectify_prop_masks(args, W, H):
    """Warp the pipeline-camera prop masks into the rectified frame.

    Same maps prep/rectify_fisheye.py built for the clip: recomputed here from
    the calibration and the balance the true-K pkl recorded, rather than read
    from disk, because the pipeline does not keep them.
    """
    import joblib
    from prep.rectify_fisheye import build_maps, rectify_masks
    pkl = args.aligned_clip.replace(".color.mp4", ".color.pkl")
    meta = joblib.load(pkl)
    cams = read_calibration(args.calib, W, H)
    new_K, map1, map2 = build_maps(cams[args.pipe_cam], W, H, meta.get("balance", 0.0))
    if abs(new_K[0, 0] - meta["fx"]) > 1e-3:
        log(f"WARNING: recomputed fx {new_K[0, 0]:.2f} != pkl fx {meta['fx']:.2f}; "
            "was the clip rectified with another balance?")
    out = osp.join(args.rect_dir, f"{args.seq}_masks_k0.h5")
    os.makedirs(args.rect_dir, exist_ok=True)
    n = rectify_masks(args.masks_dir, args.seq, 0, out, map1, map2)
    log(f"warped {n} prop masks -> {out}")
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1.0]])
    return out, K


def read_depth_frames(depth_video):
    """All frames of a uint16 millimetre depth video as float32 metres (T,H,W)."""
    import videoio
    frames = [np.asarray(f) for f in videoio.uint16read(depth_video)]   # a generator
    return np.stack(frames).astype(np.float32) / 1000.0


def read_color_frames(video):
    """All RGB frames of a clip (T,H,W,3) uint8."""
    import imageio
    reader = imageio.get_reader(video)
    frames = [fr for fr in reader]
    reader.close()
    return np.stack(frames)


def choose_register_frames(h5_path, seq, depth, k, min_px):
    """Up to k frames spread across the clip where the prop mask is large and has depth.

    Ranked by mask area with valid depth, then thinned to at most k frames at
    least (T / k) apart so a burst of good frames at one instant does not
    supply every vote.
    """
    T = depth.shape[0]
    cands = []
    with h5py.File(h5_path, "r") as f:
        g = h5_group(f, seq)
        for fr in mask_frames(h5_path, seq):
            if fr >= T:
                continue
            m = read_mask(g, fr, 0, MASK_SUFFIX_OBJ)
            if m is None or m.sum() < min_px:
                continue
            valid = int(((depth[fr] > 0.05) & m).sum())
            if valid >= min_px:
                cands.append((valid, fr))
    cands.sort(reverse=True)
    chosen, gap = [], max(1, T // max(k, 1))
    for _, fr in cands:
        if all(abs(fr - c) >= gap for c in chosen):
            chosen.append(fr)
        if len(chosen) >= k:
            break
    if len(chosen) < k and cands:                      # fall back to the plain top-k
        for _, fr in cands:
            if fr not in chosen:
                chosen.append(fr)
            if len(chosen) >= k:
                break
    return sorted(chosen)


def prop_depth(depth, mask, z_tri, mode, context_px):
    """Depth handed to register(): the prop's pixels, at the triangulated distance.

    Returns (depth_for_register, note). Outside a dilated mask the depth is
    zeroed (fp_behave's --depth_context reasoning: register() erodes at
    radius 2 and would eat an un-dilated island).
    """
    d = depth.copy()
    inside = d[mask] > 0.05
    if mode == "inject":
        d[mask] = z_tri
        note = f"inject: mask set to Z={z_tri:.3f}"
    elif mode == "shift" and inside.any():
        med = float(np.median(d[mask][inside]))
        d[mask & (d > 0.05)] += z_tri - med
        note = f"shift: UniDepth median {med:.3f} -> triangulated Z {z_tri:.3f}"
    else:
        note = "aligned depth as is"
    ksize = 2 * context_px + 1
    keep = cv2.dilate(mask.astype(np.uint8), np.ones((ksize, ksize), np.uint8)) > 0
    d[~keep] = 0
    return d, note


def cmd_register(args):
    """Register the metric mesh on the best frames, take the medoid, write the sidecar."""
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

    depth_video = args.aligned_clip.replace(".color.mp4", ".depth-reg.mp4")
    for p in (args.aligned_clip, depth_video, args.mesh, args.world_json):
        if not osp.exists(p):
            raise SystemExit(f"missing input: {p}")
    with open(args.world_json) as f:
        world = json.load(f)
    color = read_color_frames(args.aligned_clip)
    depth = read_depth_frames(depth_video)
    T, H, W = depth.shape
    log(f"clip: {T} frames {W}x{H}")
    rect_h5, K = rectify_prop_masks(args, W, H)
    frames = choose_register_frames(rect_h5, args.seq, depth, args.k, args.min_px)
    if not frames:
        raise SystemExit("no frame has a prop mask with valid depth in the rectified clip")
    log(f"registering on frames {frames}")

    mesh = trimesh.load(args.mesh, process=False)
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                         debug_dir="data/debug", debug=0, glctx=dr.RasterizeCudaContext())
    est.erode_depth_diff_thres = args.erode_depth_thres
    log(f"mesh {args.mesh}: diameter {est.diameter:.3f} m")

    z_tri = float(world["cam_xyz"][2])
    poses, used = [], []
    with h5py.File(rect_h5, "r") as f:
        g = h5_group(f, args.seq)
        for fr in frames:
            mask = read_mask(g, fr, 0, MASK_SUFFIX_OBJ)
            d, note = prop_depth(depth[fr], mask, z_tri, args.depth_mode, args.depth_context)
            try:
                pose = est.register(K=K, rgb=color[fr], depth=d, ob_mask=mask, iteration=args.iterations)
            except TypeError:
                log(f"frame {fr}: register found too little depth ({note}); skipped")
                continue
            poses.append(np.asarray(pose, float))
            used.append(int(fr))
            log(f"frame {fr}: t={np.round(pose[:3, 3], 3).tolist()}  ({note})")
    if not poses:
        raise SystemExit("every registration failed; see the depth notes above")

    pose, k_idx, spread_m, spread_deg = medoid_pose(poses)
    tri_cam = np.asarray(world["cam_xyz"], float)
    # register() places the mesh's bbox centre; compare that to the triangulated
    # mask centroid, which is a different point on a big asymmetric prop -- so
    # this check flags metres, not centimetres.
    bbox_centre_obj = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2.0
    centre_cam = pose[:3, :3] @ bbox_centre_obj + pose[:3, 3]
    tri_vs_reg = float(np.linalg.norm(centre_cam - tri_cam))
    log(f"medoid = frame {used[k_idx]}; spread over {len(poses)} poses: "
        f"{spread_m:.3f} m / {spread_deg:.1f} deg; bbox centre vs triangulated centroid {tri_vs_reg:.3f} m")
    if spread_deg > 30:
        log("WARNING: registrations disagree by >30 deg -- the prop may look symmetric to "
            "FoundationPose (a bench, a box); check the pose against the footage")
    if tri_vs_reg > max(0.5, 0.5 * est.diameter):
        log("WARNING: registered position is far from the triangulated one; one of them is wrong")

    side = {
        "prop": args.prop, "seq": args.seq, "take": args.take, "camera": args.pipe_cam,
        "frame": (f"rectified pipeline camera {args.pipe_cam}; rectification uses R=I so the "
                  f"extrinsics of {args.pipe_cam} in gopro_calibs.csv apply unchanged"),
        "mesh": osp.abspath(args.mesh),
        "mesh_convention": ("pose applies to the mesh's vertices exactly as stored: "
                            "X_cam = R @ X_obj + t (FoundationPose register output)"),
        "mesh_centroid_obj": mesh.vertices.mean(0).tolist(),
        "mesh_bbox_centre_obj": bbox_centre_obj.tolist(),
        "pose_cam": pose.tolist(),
        "world_xyz": world["world_xyz"],
        "frames_registered": used, "medoid_frame": used[k_idx], "n_poses": len(poses),
        "depth_mode": args.depth_mode,
        "checks": {"register_spread_m": spread_m, "register_spread_deg": spread_deg,
                   "tri_vs_register_m": tri_vs_reg,
                   "tri_spread_m": world["tri_spread_median_m"], "tri_frames": world["tri_frames"]},
    }
    os.makedirs(osp.dirname(args.sidecar), exist_ok=True)
    with open(args.sidecar, "w") as f:
        json.dump(side, f, indent=1)
    np.save(args.sidecar.replace(".json", "_pose_cam.npy"), pose)
    log(f"wrote {args.sidecar}")


# ----------------------------------------------------------------------------
# check
# ----------------------------------------------------------------------------

def cmd_check(args):
    """Say what exists for this prop and print the agreement numbers."""
    for label, p in (("prop masks (pipeline cam)", osp.join(args.masks_dir, f"{args.seq}_masks_k0.h5")),
                     ("triangulation", args.xyz_out), ("pooled position", args.world_out),
                     ("mesh frame pick", args.pick_out), ("metric mesh", args.mesh or "<unset>"),
                     ("sidecar", args.sidecar)):
        print(f"  {'yes' if p and osp.exists(p) else ' - ':>3}  {label:28s} {p}")
    for p in (args.world_out, args.pick_out, args.sidecar):
        if osp.exists(p):
            with open(p) as f:
                d = json.load(f)
            print(f"\n{osp.basename(p)}:")
            print(json.dumps({k: v for k, v in d.items() if k not in ("runners_up", "per_view_best", "pose_cam")},
                             indent=1))


# ----------------------------------------------------------------------------

def parse_args():
    """Subcommand + the paths scripts/recon_prop.sh passes explicitly."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["geometry", "register", "check"])
    ap.add_argument("--prop", required=True, help="short name: chair, bench, manikin, piano")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--take", default="")
    ap.add_argument("--calib", required=True, help="the take's gopro_calibs.csv")
    ap.add_argument("--pipe_cam", required=True)
    ap.add_argument("--aux_cams", nargs="*", default=[])
    ap.add_argument("--masks_dir", required=True, help="the PROP's masks dir (<seq>_masks_k0.h5, <cam>-4k_masks_k0.h5)")
    ap.add_argument("--prop_dir", required=True, help="<work>/props/<prop>; outputs land here")
    # geometry
    ap.add_argument("--width", type=int, default=796)
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--min_px", type=int, default=16)
    ap.add_argument("--inlier_px", type=float, default=None)
    ap.add_argument("--static_tol", type=float, default=0.15, help="warn if frames stray more (m)")
    ap.add_argument("--pick_stride", type=int, default=5)
    ap.add_argument("--pick_dilate", type=int, default=5)
    # register
    ap.add_argument("--aligned_clip", default=None, help="<work>/rect-aligned/<seq>.0.color.mp4")
    ap.add_argument("--mesh", default=None, help="the metric _align.obj")
    ap.add_argument("--k", type=int, default=5, help="frames to register")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--depth_mode", choices=["shift", "inject", "aligned"], default="shift")
    ap.add_argument("--depth_context", type=int, default=4)
    ap.add_argument("--erode_depth_thres", type=float, default=0.05)
    args = ap.parse_args()
    args.xyz_out = osp.join(args.prop_dir, "prop_xyz.npz")
    args.world_out = osp.join(args.prop_dir, "prop_world.json")
    args.world_json = args.world_out
    args.pick_out = osp.join(args.prop_dir, "mesh_pick.json")
    args.rect_dir = osp.join(args.prop_dir, "rect")
    args.sidecar = osp.join(args.prop_dir, f"{args.seq}_{args.prop}.json")
    if args.cmd == "register" and not (args.aligned_clip and args.mesh):
        ap.error("register needs --aligned_clip and --mesh")
    return args


def main():
    """Dispatch the subcommand."""
    args = parse_args()
    {"geometry": cmd_geometry, "register": cmd_register, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
