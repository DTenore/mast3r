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

from utils.utils import visualize_matches_on_images, scale_intrinsics

def compute_scale_factor(pts3d_query, pts3d_map, R, t, use_median=False):
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
    if use_median:
        s = np.median(projections)
    else:
        s = np.mean(projections)
    
    return s

def normalize_points(points, K):
    """
    Normalize 2D points using the intrinsic matrix.

    Args:
        points (np.ndarray): Nx2 array of 2D points.
        K (np.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        np.ndarray: Nx2 array of normalized 2D points.
    """
    # Convert to homogeneous coordinates
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])  # Nx3

    # Apply inverse of intrinsic matrix
    K_inv = np.linalg.inv(K)
    points_norm_h = (K_inv @ points_h.T).T  # Nx3

    # Convert back to Euclidean coordinates
    points_norm = points_norm_h[:, :2] / points_norm_h[:, 2][:, np.newaxis]

    return points_norm

def get_prediction(model_name, device, intrinsics, imgs, image_name):
    K0 = intrinsics[0]
    K1 = intrinsics[1]

    # Scale intrinsics based on image resizing or cropping
    # Replace (540, 720, 384, 512) with actual parameters if different
    scale_K0 = scale_intrinsics(np.array(K0), 540, 720, 384, 512)
    scale_K1 = scale_intrinsics(np.array(K1), 540, 720, 384, 512)

    reference_image_path, query_image_path = imgs
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    conf_thr = 1.001

    images = load_images(imgs, size=512, verbose=False)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # Raw predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1 = pred1['desc'].squeeze(0).detach()
    desc2 = pred2['desc'].squeeze(0).detach()

    # Find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1, desc2, subsample_or_initxy1=8,
        device=device, dist='dot', block_size=2**13
    )

    BORDER = 3

    # Ignore small borders around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (
        (matches_im0[:, 0] >= BORDER) &
        (matches_im0[:, 0] < int(W0) - BORDER) &
        (matches_im0[:, 1] >= BORDER) &
        (matches_im0[:, 1] < int(H0) - BORDER)
    )

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (
        (matches_im1[:, 0] >= BORDER) &
        (matches_im1[:, 0] < int(W1) - BORDER) &
        (matches_im1[:, 1] >= BORDER) &
        (matches_im1[:, 1] < int(H1) - BORDER)
    )

    valid_matches = valid_matches_im0 & valid_matches_im1

    # Apply the mask to matches
    matches_im0 = matches_im0[valid_matches]
    matches_im1 = matches_im1[valid_matches]

    if False:
        # Step 3: Normalize the matched points using their respective intrinsic matrices
        points_map_norm = normalize_points(matches_im0, scale_K1)
        points_query_norm = normalize_points(matches_im1, scale_K2)

        # Step 4: Estimate the Essential Matrix using normalized points
        # Since the points are normalized, set focal=1 and pp=(0,0)
        E, mask_E = cv2.findEssentialMat(
            points_map_norm,
            points_query_norm,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0  # Adjust based on your data quality
        )

    matches_im0 = matches_im0.astype(np.float32)
    matches_im1 = matches_im1.astype(np.float32)

    matches_im0_norm = cv2.undistortPoints(np.expand_dims(matches_im0, axis=1), cameraMatrix=K0, distCoeffs=None)
    matches_im1_norm = cv2.undistortPoints(np.expand_dims(matches_im1, axis=1), cameraMatrix=K0, distCoeffs=None)

    E, mask_E = cv2.findEssentialMat(matches_im0_norm, matches_im1_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
    

    # Check if Essential Matrix was found
    if E is None:
        raise ValueError("Essential Matrix estimation failed.")

    # Step 5: Select inlier matches based on mask_E
    inliers_map_norm = matches_im0_norm[mask_E.ravel() == 1]
    inliers_query_norm = matches_im1_norm[mask_E.ravel() == 1]

    # Also get the original inlier image coordinates for 3D point mapping
    inliers_map = matches_im0[mask_E.ravel() == 1]
    inliers_query = matches_im1[mask_E.ravel() == 1]

    # Define the identity camera matrix since points are normalized
    camera_matrix_identity = np.eye(3)

    # Step 6: Recover the relative pose from the Essential Matrix
    retval, R, t, mask_pose = cv2.recoverPose(
        E,
        inliers_map_norm,
        inliers_query_norm,
        camera_matrix_identity
    )

    # Get 3D points from model's output
    pts3d_query = pred1['pts3d'].squeeze(0).cpu().numpy()
    pts3d_map = pred2['pts3d_in_other_view'].squeeze(0).cpu().numpy()

    # Convert inlier image coordinates to integer indices
    indices_query = inliers_query.astype(int)
    indices_map = inliers_map.astype(int)

    # Ensure indices are within bounds
    indices_query[:, 0] = np.clip(indices_query[:, 0], 0, pts3d_query.shape[1] - 1)
    indices_query[:, 1] = np.clip(indices_query[:, 1], 0, pts3d_query.shape[0] - 1)
    indices_map[:, 0] = np.clip(indices_map[:, 0], 0, pts3d_map.shape[1] - 1)
    indices_map[:, 1] = np.clip(indices_map[:, 1], 0, pts3d_map.shape[0] - 1)

    # Extract corresponding 3D points
    pts3d_query_matched = pts3d_query[indices_query[:, 1], indices_query[:, 0]]  # [N, 3]
    pts3d_map_matched = pts3d_map[indices_map[:, 1], indices_map[:, 0]]        # [N, 3]

    # Compute the scale factor to align translation vector using 3D point clouds
    scale_factor = compute_scale_factor(pts3d_query_matched, pts3d_map_matched, R, t, use_median=True)

    # Scale the translation vector
    t_scaled = scale_factor * (t / np.linalg.norm(t))

    # Construct the relative pose matrix [R | t_scaled]
    pose_rel = np.hstack((R, t_scaled))
    pose_rel = np.vstack((pose_rel, np.array([0, 0, 0, 1])))

    # Convert rotation matrix to quaternion
    rot_quat = mat2quat(pose_rel[:3, :3])

    # Prepare the prediction string
    prediction = f"{image_name} {rot_quat[0]} {rot_quat[1]} {rot_quat[2]} {rot_quat[3]} {pose_rel[0, 3]} {pose_rel[1, 3]} {pose_rel[2, 3]} {len(inliers_map)}"

    return prediction


#{
#  "Average Median Translation Error": 6.173638470725831,
#  "Average Median Rotation Error": 109.31547771325396,
#  "Average Median Reprojection Error": 344.2323912720817,
#  "Precision @ Pose Error < (25.0cm, 5deg)": 0.0,
#  "AUC @ Pose Error < (25.0cm, 5deg)": 0.0,
#  "Precision @ VCRE < 90px": 2.7414535186555912e-05,
#  "AUC @ VCRE < 90px": 2.7794433119176553e-05,
#  "Estimates for % of frames": 0.003125257011267374
#}