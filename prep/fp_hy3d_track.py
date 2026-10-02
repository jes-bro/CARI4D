# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
use the FP track mode to run simple videos 
"""

import sys, os

import cv2
import imageio
import trimesh
sys.path.append(os.getcwd())
from glob import glob
import json
import os.path as osp
from prep.fp_behave import FPBehaveVideoProcessor, merge_pickles


class BehaveHy3DTrackFPRunner(FPBehaveVideoProcessor):
    def get_template_file(self):
        "load from hy3d"
        obj_name = self.video_prefix.split('_')[2]
        files = sorted(glob(f'{self.args.hy3d_root}/{self.video_prefix}*/*{self.video_prefix}*_align.obj'))
        if len(files) == 0:
            raise ValueError(f'no aligned hy3d template found for {self.video_prefix}')
        mesh_file = files[0]
        print('using object template from file:', mesh_file)
        return mesh_file

    def process_depth(self, depth):
        """Depth in metres, in; depth in metres, out, times --depth_scale.

        Monocular depth has a free global scale, and FoundationPose seeds and
        refines its translation from it: a depth map twice too far puts the
        object twice too far. prep/estimate_depth_scale.py measures that
        factor from the object's known size; this is where it is applied,
        after the mm->m conversion and before the zfar clip.
        """
        scale = float(getattr(self.args, 'depth_scale', 1.0) or 1.0)
        per_frame = getattr(self.args, 'depth_scale_file', None)
        if per_frame:
            if not hasattr(self, '_depth_factors'):
                import numpy as np
                self._depth_factors = np.load(per_frame)['factor']
                print(f'per-frame depth scale from {per_frame}: '
                      f'{len(self._depth_factors)} frames, median {float(np.median(self._depth_factors)):.4f}')
            import re
            m = re.search(r'\d+', str(getattr(self, 'current_frame_time', '0')))
            idx = min(int(m.group(0)) if m else 0, len(self._depth_factors) - 1)
            scale = float(self._depth_factors[idx])
        elif scale != 1.0 and not getattr(self, '_depth_scale_said', False):
            print(f'scaling depth by {scale:.4f}')
            self._depth_scale_said = True
        return depth * scale

if __name__ == '__main__':
    parser = BehaveHy3DTrackFPRunner.get_parser()
    parser.add_argument('--depth_scale', type=float, default=1.0,
                        help='multiply the depth map by this before use; the factor from '
                             'prep/estimate_depth_scale.py for a monocular depth of unknown scale')
    parser.add_argument('--depth_scale_file', default=None,
                        help='.npz with a per-frame "factor" array (prep/estimate_depth_scale_sfm.py); '
                             'overrides --depth_scale frame by frame')
    args = parser.parse_args()

    if osp.isfile(args.video):
        videos = [args.video]
    else:
        videos = sorted(glob(args.video))
    
    print(f"In total {len(videos)} video files")
    selected_views = json.load(open('splits/selected-views-map.json'))
    video_prefix = osp.basename(args.video).split('.')[0]
    if args.index is not None:
        chunk_size = len(videos) // 8 + 1  # for easy parallel 
        videos = videos[args.index * chunk_size:(args.index + 1) * chunk_size]
    print(f"Processing {len(videos)} video files, first video: {videos[0]}, last video: {videos[-1]}")
    for video in videos:
        args.video = video
        video_prefix = osp.basename(video).split('.')[0]
        kid_to_run = int(selected_views[video_prefix][1]) if video_prefix in selected_views else args.kid
        kids = [kid_to_run]
        for kid in kids:
            print(f'Processing view {kid} for {video}')
            processor = BehaveHy3DTrackFPRunner(args)
            processor.process_video(kid)

    merge_pickles(videos, args)
