
"""
the image and masks are loaded from the video data format
"""

import sys, os
sys.path.append(os.getcwd())
import torch, os, json 
import trimesh
import numpy as np
from tqdm import tqdm
import cv2 
import os.path as osp
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
import h5py

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
import nvdiffrast.torch as dr
from glob import glob
import Utils

from tools.estimate_scale import estimate_metric_scale
from behave_data.behave_video import BaseBehaveVideoData

def auto_erode_thres(depth, mask, camera_K, safety=3.0):
    """Depth-erosion threshold for THIS object at THIS distance, in metres.

    erode_depth marks a pixel inconsistent when a neighbour's depth differs by
    more than the threshold, so the threshold has to exceed how much the
    object's own surface recedes from one pixel to the next -- otherwise the
    object erodes away completely and guess_translation finds nothing.

    For an object of depth extent D spanning f*D/Z pixels at distance Z, that
    per-pixel change is D / (f*D/Z) = Z/f. The object's SIZE cancels: only its
    distance and the focal length matter, which is what makes this work for a
    basketball at 7m and a pot at 1.5m without anyone tuning a number per
    object. On the egoexo4d basketball, Z=6m and f=328px gives 18mm -- the
    figure demo-custom.sh quotes from measuring it.

    Z is the median raw depth inside the mask. It is monocular and unreliable
    in absolute terms, which is the entire reason the injection step exists --
    but a threshold only needs the right order of magnitude, and `safety`
    covers the rest.

    Returns None when the mask holds no depth at all, since a guess from
    nothing is worse than the caller's default.
    """
    vals = depth[(mask > 127) & (depth > 0.001)]
    if vals.size == 0:
        return None
    Z = float(np.median(vals))
    f = float((camera_K[0, 0] + camera_K[1, 1]) / 2.0)
    return safety * Z / f


def get_specific_frame(video_prefix, frame_time, kid=1):
    from behave_data.video_reader import ColorDepthController
    ctrl = ColorDepthController(video_prefix, kid)
    color, depth = ctrl.get_closest_frame(float(frame_time[1:]))
    return color, depth

class MetricScaleEstimator(BaseBehaveVideoData):
    def estimate_scale(self, args):
        "estimate the metric scale for the video, and save to a json file"
        video_prefix = osp.basename(args.video).split('.')[0]
        out_file = osp.join(args.outpath, f'{video_prefix}_scale.json')
        if osp.isfile(out_file) and not args.redo:
            print(f'{out_file} already exists, skipping...')
            return 
        
        # get the frame index used to reconstruct the mesh, filename format: <video_prefix>*_<frame_index>_rgba.obj
        hy_file = sorted(glob(f'{self.args.hy3d_root}/{video_prefix}*/*{video_prefix}*.obj'))[0]
        frame_time = '000' + osp.basename(hy_file).split('_')[-2]

        # Get RGB and depth
        color, depth = get_specific_frame(f'{osp.dirname(args.video)}/{video_prefix}', frame_time, kid=0)

        # The depth video stores millimetres. prep/fp_behave.py divides by 1000
        # before handing depth to FoundationPose; this never did, so every
        # threshold downstream -- erode_depth's, register's depth>=0.001 test,
        # and the mesh's own metre-scale geometry -- was being compared against
        # numbers a thousand times too large. The symptom was
        # "guess_translation() valid is empty" and a crash on pose_last, which
        # read like a bad frame or a bad mesh and was neither.
        depth = depth.astype(np.float32) / 1000.0
        # Get mask

        assert self.scale_ratio == 1.0, "the camera should not be rescaled"
        camera_K = self.camera_K.copy() 

        h5_file = f'{args.masks_root}/{video_prefix}_masks_k0.h5'
        h5_data = h5py.File(h5_file, 'r')
        mname_o = f'{video_prefix}/{frame_time}-k0.obj_rend_mask.png'
        mask_o = h5_data[mname_o][:] 
        mask_o = mask_o.astype(np.uint8) * 255

        # Init foundationpose
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

        thres = args.erode_depth_thres
        if isinstance(thres, str) and thres == 'auto':
            thres = auto_erode_thres(depth, mask_o, camera_K, safety=args.erode_safety)
            if thres is None:
                thres = 0.001
                print('auto erode threshold: no depth inside the object mask, '
                      f'falling back to {thres}')
            else:
                print(f'auto erode threshold: {thres * 1000:.1f} mm '
                      f'(safety {args.erode_safety} x Z/f)')
        estimate_metric_scale(scorer, refiner, glctx, args.outpath, hy_file, color, depth, mask_o, camera_K,
                              erode_depth_thres=float(thres))



        
    
    

if __name__ == '__main__':
    import argparse
    parser = BaseBehaveVideoData.get_parser()
    # Same knob and same default as prep/fp_behave.py's. Without it this step
    # keeps FoundationPose's 1mm default while the tracking step right after it
    # uses whatever the pipeline passes, so the two disagree about the same
    # depth map -- and a small distant object survives only one of them.
    parser.add_argument('--erode_depth_thres', default='auto',
                        help="metres of depth difference erode_depth treats as "
                             "consistent between neighbouring pixels, or 'auto' "
                             "(default) to derive it from this object's distance and "
                             "the focal length. Must exceed the object's own depth "
                             "change per pixel or the object is erased before its "
                             "scale can be measured")
    parser.add_argument('--erode_safety', default=3.0, type=float,
                        help='multiple of Z/f used by --erode_depth_thres auto '
                             '(default: 3.0, which reproduces the 0.05 hand-tuned for '
                             'the basketball)')
    args = parser.parse_args()

    estimator = MetricScaleEstimator(args)
    estimator.estimate_scale(args)