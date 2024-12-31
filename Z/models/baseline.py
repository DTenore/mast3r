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
from dust3r.utils.image import load_images
import cv2
import tqdm
import sys

from transforms3d.quaternions import mat2quat

from utils.utils import scale_intrinsics, visualize_all_features

from sklearn.linear_model import RANSACRegressor

import poselib

def compute_scale_factor(pts3d_query, pts3d_map, R, t):
    """
    Computes the scale factor to align the translation vector using 3D point clouds.
    """
    # Normalize the translation vector
    t_norm = t / np.linalg.norm(t)
    
    # Apply rotation to the query points
    pts3d_query_transformed = (R @ pts3d_query.T).T  # Shape: (N, 3)
    
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
    SCALE_MODE = "median"

    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    images = load_images(IMAGES, size=512, verbose=False)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

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

    # Intrinsic matrices
    K1 = scale_intrinsics(intrinsics_0, 540, 720, 384, 512)
    K2 = scale_intrinsics(intrinsics_1, 540, 720, 384, 512)

    # If not enough matches, return empty string (this will skip the line in the result file, which is okay)
    if (len(points1) < 5) or (len(points2) < 5):
        return ""

    # FIXME: This is wrong
    # Normalize 3d points
    norm_points1 = np.linalg.inv(K1) @ np.column_stack((points1, np.ones(points1.shape[0]))).T
    norm_points1 = norm_points1[:2] / norm_points1[2]

    norm_points2 = np.linalg.inv(K2) @ np.column_stack((points2, np.ones(points2.shape[0]))).T
    norm_points2 = norm_points2[:2] / norm_points2[2]


    # Prepare cameras for poselib
    camera1 = {
        'model': 'PINHOLE',
        'width': 384,  
        'height': 512, 
        'params': [K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]]  # fx, fy, cx, cy
    }

    camera2 = {
        'model': 'PINHOLE',
        'width': 384,   
        'height': 512, 
        'params': [K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]]  # fx, fy, cx, cy
    }

    # Initialize RANSAC and Bundle Adjustment options as dictionaries
    ransac_options = poselib.RansacOptions()
    bundle_options = poselib.BundleOptions()

    # Normalize threshold
    threshold = 0.75
    avg_diagonal = (K1[0][0] + K1[1][1] + K2[0][0] + K2[1][1]) / 4
    normalized_threshold = threshold / avg_diagonal

    # Set RANSAC options by assigning key-value pairs
    ransac_options['max_epipolar_error'] = normalized_threshold
    ransac_options['success_prob'] = 0.999


    # Convert normalized points into a list of column vectors
    # FIXME: normalization is wrong
    #points1_list = [np.array([[x], [y]], dtype=np.float64) for x, y in norm_points1.T]
    #points2_list = [np.array([[x], [y]], dtype=np.float64) for x, y in norm_points2.T]

    # Estimate the relative pose
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