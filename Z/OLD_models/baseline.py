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

#FIXME: Improve???
def compute_scale_factor_BAD(pts3d_query, pts3d_map, R, t):
    """
    Computes the scale factor for aligning 3D points from two images in absolute pose estimation.
    :param pts3d_query: np.ndarray, triangulated 3D points from the query image (N x 3).
    :param pts3d_map: np.ndarray, corresponding 3D points in the map (N x 3).
    :param R: np.ndarray, estimated rotation matrix (3 x 3).
    :param t: np.ndarray, estimated translation vector (3,).
    :return: float, scale factor.
    """
    assert pts3d_query.shape == pts3d_map.shape, "Point arrays must have the same shape"
    
    # Transform the query points using the estimated rotation
    pts3d_query_transformed = (R @ pts3d_query.T).T + t.ravel()  # Shape: (N, 3)
    
    # Compute pairwise distances between query and map points
    distances_query = np.linalg.norm(pts3d_query_transformed, axis=1)
    distances_map = np.linalg.norm(pts3d_map, axis=1)
    
    # Scale factor is the median ratio of distances
    scale_factors = distances_map / distances_query
    s = np.median(scale_factors)
    
    return s

def compute_scale_factor(pts3d_query, pts3d_map, R, t):
    """
    Computes the scale factor to align the translation vector using 3D point clouds.
    """
    # Normalize the translation vector
    t_norm = t / np.linalg.norm(t)
    
    # Apply rotation to the query points
    pts3d_query_transformed = (R @ pts3d_query.T).T # Shape: (N, 3)
    
    # Compute the difference
    diff = pts3d_map - pts3d_query_transformed  # Shape: (N, 3)
    
    # Project differences onto the normalized translation vector
    projections = np.dot(diff, t_norm)  # Shape: (N,)
    
    # Compute the scale factor
    s = np.median(projections)
    
    return s

def printer(q, t):
    return f"{q[0]} {q[1]} {q[2]} {q[3]} {t[0][0]} {t[1][0]} {t[2][0]}"

def get_prediction(model_name, device, intrinsics, IMAGES, image_name, visualize=False): 

    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
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

    # Normalize 2d points
    norm_points1 = np.linalg.inv(K1) @ np.column_stack((points1, np.ones(points1.shape[0]))).T
    norm_points1 = norm_points1[:2] / norm_points1[2]

    norm_points2 = np.linalg.inv(K2) @ np.column_stack((points2, np.ones(points2.shape[0]))).T
    norm_points2 = norm_points2[:2] / norm_points2[2]


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

    cameraa = {
        'model': 'PINHOLE',
        'width': W0,
        'height': H0,
        'params': [1, 1, 0, 0]  # fx = fy = 1, cx = cy = 0 for normalized points
    }

    # Initialize RANSAC and Bundle Adjustment options as dictionaries
    ransac_options = poselib.RansacOptions()
    bundle_options = poselib.BundleOptions()

    # Normalize threshold
    threshold = 0.75
    avg_diagonal = (K1[0][0] + K1[1][1] + K2[0][0] + K2[1][1]) / 4
    normalized_threshold = threshold / avg_diagonal

    # Set RANSAC options
    ransac_options['max_epipolar_error'] = normalized_threshold
    ransac_options['success_prob'] = 0.999

    # Estimate the relative pose
    if True:
        # NEW, but worse???
        pose, info = poselib.estimate_relative_pose(
            norm_points1.T, norm_points2.T, cameraa, cameraa, ransac_options, bundle_options
        )
    else:
        # OLD
        pose, info = poselib.estimate_relative_pose(
            points1, points2, camera1, camera2, ransac_options, bundle_options
        )

    # Extract rotation (R) and translation (t) from the pose
    R = pose.R
    t = pose.t

    t = t.reshape(-1, 1)

    # Get inliers from info dict
    inliers = info['inliers']  # List of boolean values indicating inliers

    # Convert the inliers list to a mask array similar to OpenCV's output
    mask = np.array(inliers, dtype=np.uint8).reshape(-1, 1)

    quat = mat2quat(R)
    
    # Extract inlier matches
    mask_E = mask.ravel().astype(bool)
    points1_inliers = points1[mask_E]
    points2_inliers = points2[mask_E]

    # Convert inlier matches to integer indices
    indices_query = points1_inliers.astype(int)
    indices_map = points2_inliers.astype(int)

    # Ensure indices are within bounds
    indices_query[:, 0] = np.clip(indices_query[:, 0], 0, pts3d_im0.shape[1] - 1)
    indices_query[:, 1] = np.clip(indices_query[:, 1], 0, pts3d_im0.shape[0] - 1)
    indices_map[:, 0] = np.clip(indices_map[:, 0], 0, pts3d_im1.shape[1] - 1)
    indices_map[:, 1] = np.clip(indices_map[:, 1], 0, pts3d_im1.shape[0] - 1)

    # Extract corresponding 3D points
    pts3d_query_matched = pts3d_im0[indices_query[:, 1], indices_query[:, 0]]  # Shape: (N, 3)
    pts3d_map_matched = pts3d_im1[indices_map[:, 1], indices_map[:, 0]]        # Shape: (N, 3)

    # Compute the scale factor
    s = compute_scale_factor(pts3d_query_matched, pts3d_map_matched, R, t)

    # Scale translation vector accordingly
    t_scaled = t * s

    # Prepare result output string
    res = printer(quat, t_scaled)

    # Visualize matches if flag is set
    if visualize:
        visualize_all_features(points1_inliers, points2_inliers, IMAGES[0], IMAGES[1], 384, 512)

    return f"{image_name} {res} {len(valid_matches)}"



"""
{
  "Average Median Translation Error": 1.6023888366104968,
  "Average Median Rotation Error": 7.733649716229225,
  "Average Median Reprojection Error": 205.68479301054873,
  "Precision @ Pose Error < (25.0cm, 5deg)": 0.08995502248875563,
  "AUC @ Pose Error < (25.0cm, 5deg)": 0.005011464179288525,
  "Precision @ VCRE < 90px": 0.2436281859070465,
  "AUC @ VCRE < 90px": 0.012066982342166476,
  "Estimates for % of frames": 1.0
}
"""