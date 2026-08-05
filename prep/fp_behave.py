# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
run foundationpose on behave
"""
import json
import joblib
import sys, os
import os.path as osp
from glob import glob
from os import path as osp

import imageio
from h5py import File

sys.path.append(os.getcwd())
import cv2
import h5py, hdf5plugin
import torch
from tqdm import tqdm
import numpy as np
import tarfile
from os.path import join, basename, dirname
import io, trimesh
from PIL import Image
from behave_data.utils import get_intrinsics_unified
from behave_data.behave_video import BaseBehaveVideoData, load_masks


def report_depth_coverage(depth, mask_o, frame_time, zfar):
    """Print what register() will see: object mask size and its usable depth.

    guess_translation needs at least 4 pixels that are both inside the object
    mask and carry depth >= 0.001. When it does not get them it prints 'valid is
    empty' and returns zeros, register falls back to self.pose_last, and on the
    first frame that is None -- so the whole thing surfaces as

        TypeError: 'NoneType' object is not subscriptable

    which names neither the depth nor the mask. Printing the deciding numbers on
    every registration frame turns that into an obvious diagnosis, and costs one
    line per sequence.

    Args:
        depth: depth map in metres, already masked and zfar-clipped.
        mask_o: boolean object mask.
        frame_time: frame identifier, for the message.
        zfar: the clipping threshold in use, for the message.
    """
    mask_px = int(mask_o.sum())
    inside = depth[mask_o] if mask_px else np.array([])
    valid = inside[inside >= 0.001] if inside.size else np.array([])
    print(f'[register] frame {frame_time}: object mask {mask_px} px, '
          f'{valid.size} with depth >= 0.001'
          + (f', median {np.median(valid):.2f}m, range '
             f'{valid.min():.2f}-{valid.max():.2f}m' if valid.size else ''))
    if valid.size >= 4:
        return
    if mask_px < 4:
        print(f'[register] WARNING: the object mask is {mask_px} px, so there is '
              f'nothing to register against. Pick a frame where the object is '
              f'visible -- prep/select_recon_frame.py ranks them.')
    else:
        print(f'[register] WARNING: {mask_px} px of object mask but only '
              f'{valid.size} carry depth, so registration cannot seed a '
              f'translation and will fail. The mask is fine; the depth over the '
              f'object is zero or beyond zfar={zfar}. '
              f'prep/check_object_depth.py compares the depth map against the '
              f'distance the silhouette implies.')
import nvdiffrast.torch as dr
import signal
from scipy.spatial.transform import Rotation as R
import Utils
import imageio.v3 as iio
import imageio
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

class FPBehaveVideoProcessor(BaseBehaveVideoData):
    def process_depth(self, depth):
        "input and output depth should be float"
        return depth

    def process_video(self, kid_to_run, refiner=None, mesh=None, glctx=None, est=None):
        "init once and then run tracking mode"
        args = self.args
        output_path = self.output_path.replace('.pkl', f'_k{kid_to_run}.pkl')
        if osp.isfile(output_path):
            print('Already exists {}, all done'.format(output_path))
            return
        if est is None:
            est, glctx, mesh, refiner = self.init_pose_estimator(debug=0)
        else:
            assert refiner is not None
            assert mesh is not None
            assert glctx is not None

        self.crop_ratio_default = refiner.cfg.crop_ratio
        kids = self.kids
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        if not self.args.wild_video:
            K_all = [get_intrinsics_unified(self.args.data_source, self.video_prefix, kid, self.args.wild_video) for kid
                     in self.kids]
            K_all = np.stack(K_all)
            K_all[:, :2] /= self.scale_ratio  # make sure the resolution matches
        else:
            K_all = [self.camera_K]  # this already take into account the scale ratio

        pose_dict, pose_hist_dict = {}, {}
        reinit_every = args.reinit_every if args.reinit_every is not None else len(self.times) + 10

        for enum_idx, k in enumerate(kids):
            tar_mask = h5py.File(self.tar_path.replace('_masks_k0.h5', f'_masks_k{k}.h5'), 'r')
            if kid_to_run is not None and k != kid_to_run:
                continue
            print(f'Processing view {k}')
            loop = tqdm(self.times)
            loop.set_description(f"{self.video_prefix}-k{k}")

            if args.viz_path is not None:
                viz_file = f"{output_path.replace('.pkl', f'_k{k}.mp4')}"
                vw = imageio.get_writer(viz_file, 'ffmpeg', fps=2)
            is_first_frame = True
            for i, t in enumerate(loop):
                color, depth = self.load_color_depth(enum_idx, kids, t)
                frame_time = self.get_time_str(t)

                h, w = color.shape[:2]
                # 8m suits BEHAVE's indoor capture volume, but clips everything
                # further away to zero -- and FoundationPose needs valid depth
                # inside the object mask to guess an initial translation. On an
                # egoexo4d basketball take the human alone sits at 6.65m and the
                # ball beyond that, so register() found no valid pixels and died
                # on frame 0 with 'NoneType' object is not subscriptable, having
                # tried to fall back to a previous pose that does not yet exist.
                zfar = self.args.zfar
                color = cv2.resize(color, (int(w / self.scale_ratio), int(h / self.scale_ratio)))
                depth = cv2.resize(depth, (int(w / self.scale_ratio), int(h / self.scale_ratio)),
                                   cv2.INTER_NEAREST) / 1000.
                depth = self.process_depth(depth)

                depth[(depth < 0.001) | (depth >= zfar)] = 0
                mask_h, mask_o = load_masks(self.video_prefix, frame_time, k, tar_mask)
                if mask_h is None:
                    continue
                mask_h = cv2.resize(mask_h, (int(w / self.scale_ratio), int(h / self.scale_ratio))) > 127
                mask_o = cv2.resize(mask_o, (int(w / self.scale_ratio), int(h / self.scale_ratio))) > 127
                # remove depth due to human mask or background mask
                depth_scene = depth.copy()  # kept for registration, see below
                depth[~mask_o] = 0

                if is_first_frame or i % reinit_every == 0:  # reinit does not work well, especially for symmetric objects.
                    mname_o = f'{self.video_prefix}/{frame_time}-k{k}.obj_rend_mask.png'
                    mask_o = tar_mask[mname_o][:]  # this is 3-4 it/s
                    mask_o = mask_o.astype(np.uint8) * 255
                    mask_o = cv2.resize(mask_o, (int(w / self.scale_ratio), int(h / self.scale_ratio))) > 127
                    # register() needs valid depth inside the object mask, and its
                    # failure mode is opaque: guess_translation prints 'valid is
                    # empty', register falls back to a previous pose, and on the
                    # first frame that is None -- so it surfaces as a TypeError
                    # naming neither depth nor the mask. Print the numbers that
                    # actually decide it.
                    # register() erodes and bilateral-filters the depth before
                    # using it (estimater.py:238-239, radius 2 each). Zeroing
                    # depth outside the object leaves it an isolated island, and
                    # on a small object the erosion consumes the whole thing:
                    # 85 valid pixels went in and guess_translation found fewer
                    # than 4, then register fell back to a pose that does not
                    # exist yet. Dilating the mask used for zeroing gives the
                    # erosion a margin to eat. The extra pixels do not pollute
                    # the result -- register restricts to (depth>=0.001) &
                    # (ob_mask>0) with the true mask, and tracking below still
                    # gets the tightly masked depth.
                    depth_reg = depth
                    if self.args.depth_context > 0:
                        ksize = 2 * self.args.depth_context + 1
                        keep = cv2.dilate(mask_o.astype(np.uint8),
                                          np.ones((ksize, ksize), np.uint8)) > 0
                        depth_reg = depth_scene.copy()
                        depth_reg[~keep] = 0
                    report_depth_coverage(depth_reg, mask_o, frame_time, zfar)
                    pose = est.register(K=K_all[k], rgb=color, depth=depth_reg, ob_mask=mask_o.astype(bool),
                                        iteration=5,
                                        vis_score_path=output_path.replace('.pkl', f'_{t:06f}_k{k}_score.png'),
                                        vis_refine_path=output_path.replace('.pkl', f'_{t:06f}_k{k}_refine.png'),
                                        rgb_only=False, both_depth_and_rgb=False
                                        )
                    is_first_frame = False
                else:
                    # run tracking mode
                    pose = est.track_one(rgb=color, depth=depth, K=K_all[k], iteration=5)
                if frame_time not in pose_hist_dict:
                    pose_hist_dict[frame_time] = []
                if frame_time not in pose_dict:
                    pose_dict[frame_time] = []
                pose_dict[frame_time].append(pose)

                # visualize the result
                if args.viz_path is not None and i % 15 == 0:
                    center_pose = pose @ np.linalg.inv(to_origin)
                    vis = color.copy()
                    vis = Utils.draw_posed_3d_box(K_all[k], img=vis, ob_in_cam=center_pose, bbox=bbox)
                    vis = Utils.draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K_all[k], thickness=3,
                                              transparency=0, is_input_rgb=True)
                    comb = np.concatenate((color, vis), 1)
                    cv2.putText(comb, f'{frame_time} ', (vis.shape[1], 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                                (0, 255, 255), 2)

                    # add mask viz
                    viz_mask = color.copy()
                    viz_mask[mask_h] = 0
                    comb = np.concatenate((comb, viz_mask), 1)
                    comb = cv2.resize(comb, (comb.shape[1] // 3, comb.shape[0] // 3))
                    vw.append_data(comb)

        # pack results and save
        kids = [kid_to_run]
        poses_all = [np.stack(v) for k, v in sorted(pose_dict.items()) if len(v) == len(kids)]
        frames = [k for k, v in sorted(pose_dict.items()) if len(v) == len(kids)]
        pose_all = np.stack(poses_all)  # T, K, 4, 4
        out_dict = {"fp_poses": pose_all, "frames": frames}
        joblib.dump(out_dict, output_path)
        print('all done, saved to', output_path, 'pose_all:', pose_all.shape)
        if args.viz_path is not None:
            vw.close()
            print(f'visualization saved to {viz_file}')

    def load_template_mesh(self, ret_file=False):
        from behave_data.utils import load_template
        if self.args.data_source == 'behave':
            mesh_file = self.get_template_file()
            print('Using template mesh from {}'.format(mesh_file))
            mesh = trimesh.load(mesh_file, process=False) # this mesh should have already been centered at origin
            # get behave mesh template center
            print("Not loading any behave template!")
        elif self.args.data_source == 'hodome':
            obj_name = self.video_prefix.split('_')[2]
            mesh_file = f'/home/xianghuix/datasets/HODome/obj-newtex/{obj_name}/{obj_name}.obj'
            print('Using template mesh from {}'.format(mesh_file))
            mesh = trimesh.load(mesh_file, process=False)
            # center by mean of vertices
            center = np.mean(mesh.vertices, 0)
            mesh.vertices = mesh.vertices - center
        elif self.args.data_source == 'imhd':
            obj_name = self.video_prefix.split('_')[2]
            mesh_file = f'/home/xianghuix/datasets/IMHD2/hy3d-texgen-simp/{obj_name}/{obj_name}_simplified_transformed.obj'
            print('Using template mesh from {}'.format(mesh_file))
            mesh = trimesh.load(mesh_file, process=False)
            # center by mean of vertices
            center = np.mean(mesh.vertices, 0)
            mesh.vertices = mesh.vertices - center

        elif self.args.data_source == 'intercap':
            # get from HY3D
            obj_name = self.video_prefix.split('_')[2]
            files = sorted(glob(f'/home/xianghuix/datasets/behave/selected-views/hy3d-aligned/{self.video_prefix}*/*{obj_name}*_align.obj'))
            if len(files) == 0:
                print(f'no aligned hy3d template found for {self.video_prefix}, existing...')
                return
            mesh_file = files[0]
            print('using object template:', mesh_file)
            mesh = trimesh.load(mesh_file, process=False)
            # need to subtract center, otherwise not aligned with GT pose
            from behave_data.utils import get_template_path
            icap_path = get_template_path(None, obj_name)
            icap_mesh = trimesh.load(icap_path, process=False)
            center = np.mean(icap_mesh.vertices, 0)
            mesh.vertices = mesh.vertices - center
        elif self.args.data_source == 'procigen':
            # get the template mesh from ShapeNet, rescale
            from behave_data.const import shapenet_root
            packed_file = f'/home/xianghuix/datasets/behave/behave-packed/{self.video_prefix}_GT-packed.pkl'
            packed_data = joblib.load(packed_file)
            # load objaverse uids and check if it is objaverse
            objav_uids = json.load(open('splits/objaverse_ids.json', 'r'))
            ins_name = packed_data["ins_names"][0]
            if ins_name in objav_uids:
                from behave_data.const import objav_root
                template_file = f'{objav_root}/{ins_name}/model.obj'
                mesh = trimesh.load(template_file, process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    print("Failed to load mesh with consistent texture, loading without texture")
                    mesh = trimesh.load(template_file, process=False, force='mesh')
                # for objaverse, need to rotate around x axis by 90 degree: y->z, -z->y
                rot_x90 = np.array([
                    [1, 0, 0.],
                    [0, 0, -1],
                    [0, 1, 0],
                ])
                mesh.vertices = np.matmul(mesh.vertices, rot_x90.T)
            else:
                template_file = f'{shapenet_root}/{packed_data["synsets"][0]}/{packed_data["ins_names"][0]}/models/model_normalized.obj'
                mesh = trimesh.load(template_file, process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    print("Failed to load mesh with consistent texture, loading without texture")
                    mesh = trimesh.load(template_file, process=False, force='mesh')
            # compute a scale
            u, s, vt = np.linalg.svd(packed_data['obj_rot_orig'][0])
            mesh.vertices = mesh.vertices * s[0]
            print(f"loading mesh template from {template_file}, scaled by {s[0]:.3f}")
        if ret_file:
            # only works for hodome and imhd
            return mesh, mesh_file
        return mesh

    def init_pose_estimator(self, debug=0):
        mesh = self.load_template_mesh()
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        # init fp
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        debug_dir = 'data/debug'
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer,
                             refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
        # erode_depth keeps a pixel only when 20% of its 5x5 neighbourhood agrees
        # to within this. It has to exceed the object's own depth change between
        # adjacent pixels, which for a small curved object seen at distance is
        # large: a 24cm sphere at 6m spanning 13px recedes ~18mm per pixel, so
        # the 1mm default erases it entirely before registration.
        est.erode_depth_diff_thres = self.args.erode_depth_thres
        print(f'[fp] erode_depth_diff_thres={est.erode_depth_diff_thres} m, '
              f'mesh diameter {est.diameter:.3f} m')
        return est, glctx, mesh, refiner


    def load_color_depth(self, enum_idx, kids, t):
        if self.args.wild_video:
            actual_time = t
        else:
            actual_times = np.array([self.controllers[x].get_closest_time(t) for x, _ in enumerate(kids)])
            best_kid = np.argmin(np.abs(actual_times - t))
            actual_time = actual_times[best_kid]
        if self.args.nodepth:
            return self.controllers[enum_idx].get_closest_frame(actual_time), None 
        else:
            color, depth = self.controllers[enum_idx].get_closest_frame(actual_time)
            return color, depth

    def get_template_file(self):
        obj_name = self.video_prefix.split('_')[2]
        mesh_file = f'/home/xianghuix/datasets/behave/objects/{obj_name}/{obj_name}.obj'
        if self.args.wild_video:
            files = sorted(glob(f'/home/xianghuix/datasets/behave/hy3d/manual-icp-out-miny-nocent/{obj_name}/*{obj_name}*_rgba.obj'))
            mesh_file = files[0]
        print('using object template:', mesh_file)
        return mesh_file

    @staticmethod
    def get_parser():
        parser = BaseBehaveVideoData.get_parser()
        parser.add_argument('--run_backwards', default=False, action='store_true')
        parser.add_argument('--vis_thres', default=0.7, type=float)

        # 1 for reinit every frame, None for not reinit
        parser.add_argument("--reinit_every", default=None, type=int)
        parser.add_argument("--erode_depth_thres", default=0.001, type=float,
                            help="metres of depth difference erode_depth treats as "
                                 "consistent between neighbouring pixels. Must exceed "
                                 "the object's own depth change per pixel, or the whole "
                                 "object is erased before registration -- a 24cm sphere "
                                 "at 6m spanning 13px recedes ~18mm per pixel. The 1mm "
                                 "default suits large close objects on a depth sensor "
                                 "(default: 0.001)")
        parser.add_argument("--depth_context", default=4, type=int,
                            help="pixels of depth kept around the object mask when "
                                 "seeding registration. register() erodes the depth at "
                                 "radius 2 and bilateral-filters it at radius 2, which "
                                 "consumes a small object entirely if its depth is an "
                                 "isolated island. 0 restores the previous behaviour "
                                 "(default: 4)")
        parser.add_argument("--zfar", default=8.0, type=float,
                            help="depth beyond this many metres is discarded. The 8m "
                                 "default matches BEHAVE's indoor capture volume; raise "
                                 "it for scenes shot at distance, or FoundationPose finds "
                                 "no valid depth inside the object mask and cannot "
                                 "initialise (default: 8.0)")
        return parser


def process_video(args):
    processor = FPBehaveVideoProcessor(args)
    processor.process_video()



if __name__ == '__main__':
    parser = FPBehaveVideoProcessor.get_parser()
    args = parser.parse_args()

    try:
        process_video(args)
    except Exception as e:
        import traceback
        traceback.print_exc()


def merge_pickles(videos, args):
    for video in videos:
        video_prefix = osp.basename(video).split('.')[0]
        files = sorted(glob(f'{args.outpath}/{video_prefix}_*k*.pkl'))  # for InterCap, no 000 prefix
        dnew = {}
        for file in files:
            d = joblib.load(file)
            for k, v in d.items():
                if k not in dnew:
                    dnew[k] = []
                if k == 'frames':
                    dnew[k] = v
                elif k in ['backward', 'vis_thres']:
                    continue
                else:
                    dnew[k].append(v)
            # in the end the poses should be in shape (T, K, 4, 4), where T is the number of frames, K is the number of cameras, 4, 4 is the pose matrix
            outfile = osp.join(args.outpath, f'{video_prefix}_all.pkl')
            for k, v in dnew.items():
                if k in ['frames', 'backward', 'vis_thres']:
                    continue
                dnew[k] = np.concatenate(v, axis=1)
                print(k, dnew[k].shape)
            joblib.dump(dnew, outfile)
            print('saved packed results to', outfile)
