import sys
import os
import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image, ImageOps
import math

Z_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if Z_PATH not in sys.path:
    sys.path.insert(0, Z_PATH)
from utils import path_to_dust3r_and_mast3r

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.utils.misc import mkdir_for

from dust3r.inference import inference
#from dust3r.utils.image import load_images
import cv2
import tqdm
import sys

from transforms3d.quaternions import mat2quat

from utils.utils import scale_intrinsics, visualize_all_features
from utils.file_io import load_images

#from sklearn.linear_model import RANSACRegressor

import poselib



def printer(q, t):
    return f"{q[0]} {q[1]} {q[2]} {q[3]} {t[0][0]} {t[1][0]} {t[2][0]}"

def get_prediction(model_name, device, intrinsics, IMAGES, image_name, visualize=False): 

    model = AsymmetricMASt3R.from_pretrained(model_name, verbose=False).to(device)
    images = load_images(IMAGES, size=512, verbose=False)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw mast3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = (int(x) for x in view1['true_shape'][0])
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0 - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0 - 3)

    H1, W1 = (int(x) for x in view2['true_shape'][0])
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < W1 - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < H1 - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    # Convert the other outputs to numpy arrays
    pts3d_im0 = pred1['pts3d'].squeeze(0).detach().cpu().numpy()
    pts3d_im1 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy()

    conf_im0 = pred1['conf'].squeeze(0).detach().cpu().numpy()
    conf_im1 = pred2['conf'].squeeze(0).detach().cpu().numpy()

    desc_conf_im0 = pred1['desc_conf'].squeeze(0).detach().cpu().numpy()
    desc_conf_im1 = pred2['desc_conf'].squeeze(0).detach().cpu().numpy()


    intrinsics_0 = intrinsics[0]
    intrinsics_1 = intrinsics[1]

    # Convert matches to numpy float32 arrays
    points1 = matches_im0.astype(np.float32)
    points2 = matches_im1.astype(np.float32)

    orig_W0, orig_H0 = (int(x) for x in view1['original_shape'])
    orig_W1, orig_H1 = (int(x) for x in view1['original_shape'])

    # Intrinsic matrices
    K1 = scale_intrinsics(intrinsics_0, orig_W0, orig_H0, W0, H0)
    K2 = scale_intrinsics(intrinsics_1, orig_W1, orig_H1, W1, H1)

    # If not enough matches, return empty string (this will skip the line in the result file, which is okay)
    if (len(points1) < 5) or (len(points2) < 5):
        return ""

    # Prepare cameras for poselib
    camera1 = {
        'model': 'PINHOLE',
        'width': W0,  
        'height': H0, 
        'params': [K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]]  # fx, fy, cx, cy
    }

    camera2 = {
        'model': 'PINHOLE',
        'width': W1,   
        'height': H1, 
        'params': [K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]]  # fx, fy, cx, cy
    }
  

    pose, info = poselib.estimate_absolute_pose(matches_im1.astype(np.float32), pts3d_im0[matches_im0[:, 1], matches_im0[:, 0], :], camera1, {'max_reproj_error': 5, 'max_iterations': 10_000, 'success_prob': 0.9999}, {})

    Rt = pose.Rt  # (3x4)


    R = Rt[:,:3]
    
    t = Rt[:,3:]
    
    quat = mat2quat(R)

    # Prepare result output string
    res = printer(quat, t)

    # Visualize matches if flag is set
    if visualize:# Extract inlier matches
        mask = np.array(info["inliers"])

        mask_E = mask.ravel().astype(bool)
        points1_inliers = points1[mask_E]
        points2_inliers = points2[mask_E]

        # Convert inlier matches to integer indices
        indices_query = points1_inliers.astype(int)
        indices_map = points2_inliers.astype(int)

        visualize_all_features(points1_inliers, points2_inliers, IMAGES[0], IMAGES[1], 384, 512)

    return f"{image_name} {res} {len(valid_matches)}"

"""SIZE L
{
  "Average Median Translation Error": 0.5283644664948984,
  "Average Median Rotation Error": 4.087547500110743,
  "Average Median Reprojection Error": 66.1819391876078,
  "Precision @ Pose Error < (25.0cm, 5deg)": 0.4250374812593703,
  "AUC @ Pose Error < (25.0cm, 5deg)": 0.029336014743627518,
  "Precision @ VCRE < 90px": 0.7256371814092953,
  "AUC @ VCRE < 90px": 0.03952820999839224,
  "Estimates for % of frames": 1.0
}
"""