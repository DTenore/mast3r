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

def compute_scale_factor(pts3d_query, pts3d_map, R, t, use_median=True):
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

def compute_scale_factor_ransac(pts3d_query, pts3d_map, R, t):
    """
    Computes the scale factor using RANSAC to mitigate the effect of outliers.
    """
    # Normalize the translation vector
    t_norm = t.flatten() / np.linalg.norm(t)

    # Apply rotation to the query points
    pts3d_query_transformed = (R @ pts3d_query.T).T  # Shape: (N, 3)

    # Compute the difference
    diff = pts3d_map - pts3d_query_transformed  # Shape: (N, 3)

    # Project differences onto the normalized translation vector
    projections = np.dot(diff, t_norm)  # Shape: (N,)

    # Reshape for sklearn
    projections = projections.reshape(-1, 1)
    targets = projections.copy()

    # RANSAC regression to find the best scale factor
    ransac = RANSACRegressor()
    ransac.fit(projections, targets)
    s = ransac.estimator_.coef_[0][0]

    return s

def compute_scale_factor_ransac_corrected(pts3d_query, pts3d_map, R, t):
    """
    Computes the scale factor using RANSAC to handle outliers in scale estimation.
    """
    # Normalize the translation vector
    t_norm = t.flatten() / np.linalg.norm(t)
    
    # Apply rotation to the query points
    pts3d_query_transformed = (R @ pts3d_query.T).T
    
    # Compute the difference
    diff = pts3d_map - pts3d_query_transformed
    
    # Project differences onto the normalized translation vector
    projections = np.dot(diff, t_norm)
    
    # Reshape for sklearn (features, targets)
    X = np.ones_like(projections).reshape(-1, 1)  # Intercept term
    y = projections
    
    # RANSACRegressor to fit s * 1 = y
    ransac = RANSACRegressor()
    ransac.fit(X, y)
    
    s = ransac.estimator_.intercept_
    
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

    if (len(points1) < 5) or (len(points2) < 5):
        return ""

    #TODO: normalize and use both intrinsics, also adjust threshold!!!

    if False:    
        # Compute the Essential Matrix
        #WIP: replace with poselib
        E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=K1, method=cv2.USAC_MAGSAC, prob=0.999, threshold=1.0)
        # Recover the pose from the Essential Matrix
        #WIP: replace with poselib
        _, R, t, mask_pose = cv2.recoverPose(E, points1, points2, cameraMatrix=K1)

    else:
        camera1 = {
            'model': 'PINHOLE',
            'width': 384,   # Replace with the actual image width
            'height': 512, # Replace with the actual image height
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

        # Set RANSAC options by assigning key-value pairs
        ransac_options['max_epipolar_error'] = 1.0  # Set threshold as needed
        ransac_options['success_prob'] = 0.999

        # Estimate the relative pose
        pose, info = poselib.estimate_relative_pose(
            points1, points2, camera1, camera2, ransac_options, bundle_options
        )

        # Extract rotation (R) and translation (t) from the pose
        R = pose.R
        t = pose.t

        t = t.reshape(-1, 1)

        # 'info' contains additional information about the estimation process, such as inliers
        inliers = info['inliers']  # List of boolean values indicating inliers

        # Convert the inliers list to a mask array similar to OpenCV's output
        mask = np.array(inliers, dtype=np.uint8).reshape(-1, 1)


    quat = mat2quat(R)
    
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
    if SCALE_MODE == "median":
        s = compute_scale_factor(pts3d_query_matched, pts3d_map_matched, R, t, use_median=True)
    elif SCALE_MODE == "ransac":
        s = compute_scale_factor_ransac(pts3d_query_matched, pts3d_map_matched, R, t)
    elif SCALE_MODE == "ransac_2":
        s = compute_scale_factor_ransac_corrected(pts3d_query_matched, pts3d_map_matched, R, t)

    t_scaled = t * s

    res = printer(quat, t_scaled)

    if visualize:
        visualize_all_features(points1_inliers, points2_inliers, IMAGES[0], IMAGES[1], 384, 512)

    return f"{image_name} {res} {len(valid_matches)}"

# Mode median
#{
#  "Average Median Translation Error": 0.28868831764709924,
#  "Average Median Rotation Error": 2.7497896384005243,
#  "Average Median Reprojection Error": 46.52166842150194,
#  "Precision @ Pose Error < (25.0cm, 5deg)": 0.0010691668722756806,
#  "AUC @ Pose Error < (25.0cm, 5deg)": 0.0015285793190010124,
#  "Precision @ VCRE < 90px": 0.0023302354908572524,
#  "AUC @ VCRE < 90px": 0.0029029277450884836,
#  "Estimates for % of frames": 0.003125257011267374
#}


# Mode ransac
#{
#  "Average Median Translation Error": 1.4690356688515631,
#  "Average Median Rotation Error": 2.7497896384005243,
  #"Average Median Reprojection Error": 296.8197906816472,
#  "Precision @ Pose Error < (25.0cm, 5deg)": 0.0002741453518655591,
#  "AUC @ Pose Error < (25.0cm, 5deg)": 0.0003337958372204225,
#  "Precision @ VCRE < 90px": 0.0005482907037311182,
#  "AUC @ VCRE < 90px": 0.0007122507237050054,
#  "Estimates for % of frames": 0.003125257011267374
#}

# Mode ransac_2
#{
#  "Average Median Translation Error": 0.3230706364406601,
#  "Average Median Rotation Error": 2.7497896384005243,
 # "Average Median Reprojection Error": 51.31979967965075,
#  "Precision @ Pose Error < (25.0cm, 5deg)": 0.0010965814074622364,
#  "AUC @ Pose Error < (25.0cm, 5deg)": 0.0016430161840121226,
#  "Precision @ VCRE < 90px": 0.0023302354908572524,
#  "AUC @ VCRE < 90px": 0.0029029277450884836,
#  "Estimates for % of frames": 0.003125257011267374
#}


########################
#BIG
# MEDIAN
#{
#  "Average Median Translation Error": 1.2515250019950364,
#  "Average Median Rotation Error": 2.4989702904610938,
#  "Average Median Reprojection Error": 182.31448277346698,
#  "Precision @ Pose Error < (25.0cm, 5deg)": 0.004093497864261984,
#  "AUC @ Pose Error < (25.0cm, 5deg)": 0.005031412281144657,
#  "Precision @ VCRE < 90px": 0.008246321784527765,
#  "AUC @ VCRE < 90px": 0.008917280715749269,
#  "Estimates for % of frames": 0.02215828191741813
#}

# BIIIIIIIG and Median
#{
#  "Average Median Translation Error": 1.4327915169474361,
#  "Average Median Rotation Error": 5.8156609131032,
#  "Average Median Reprojection Error": 192.6193295486678,
#  "Precision @ Pose Error < (25.0cm, 5deg)": 0.00531863765979943,
#  "AUC @ Pose Error < (25.0cm, 5deg)": 0.006499850060766222,
#  "Precision @ VCRE < 90px": 0.012014994713402326,
#  "AUC @ VCRE < 90px": 0.01366252807065593,
#  "Estimates for % of frames": 0.04274134119380987
#}


# Poselib
#{
#  "Average Median Translation Error": 1.4123356789362238,
#  "Average Median Rotation Error": 5.53879148295968,
#  "Average Median Reprojection Error": 193.06656720150448,
#  "Precision @ Pose Error < (25.0cm, 5deg)": 0.13493253373313344,
#  "AUC @ Pose Error < (25.0cm, 5deg)": 0.006898831888214502,
#  "Precision @ VCRE < 90px": 0.2848575712143928,
#  "AUC @ VCRE < 90px": 0.013756940338455849,
#  "Estimates for % of frames": 1.0
#}