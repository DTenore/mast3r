#!/usr/bin/env python3
# --------------------------------------------------------
# Idea: Use Visloc code for mapfree
# --------------------------------------------------------
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
from dust3r.utils.geometry import geotrf
import cv2
import tqdm
import sys

from transforms3d.quaternions import mat2quat

from utils.utils import visualize_matches_on_images

#sys.stderr = open(os.devnull, 'w')

def coarse_matching(query_view, map_view, model, device, pixel_tol, fast_nn_params):
    """
    Perform coarse matching between query and map views.
    Returns matches_im_query, matches_im_map, matches_conf, and model output.
    """
    # Prepare batch
    imgs = []
    for idx, img in enumerate([query_view['rgb_rescaled'], map_view['rgb_rescaled']]):
        imgs.append({
            'img': img.unsqueeze(0).to(device),
            'true_shape': np.int32([img.shape[1:]]),
            'idx': idx,
            'instance': str(idx)
        })

    # Run inference
    output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']

    # Extract descriptors and confidence maps
    conf_list = [pred1['desc_conf'].squeeze(0).cpu().numpy(), pred2['desc_conf'].squeeze(0).cpu().numpy()]
    desc_list = [pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()]

    # Find 2D-2D matches between the two images
    PQ, PM = desc_list[0], desc_list[1]
    if len(PQ) == 0 or len(PM) == 0:
        return [], [], [], [], output

    # Find matches using fast_reciprocal_NNs
    matches_im_map, matches_im_query = fast_reciprocal_NNs(
        PM, PQ, subsample_or_initxy1=8, **fast_nn_params)

    # Apply confidence threshold
    HM, WM = map_view['rgb_rescaled'].shape[1:]
    HQ, WQ = query_view['rgb_rescaled'].shape[1:]
    # Ignore small border around the edge
    valid_matches_map = (matches_im_map[:, 0] >= 3) & (matches_im_map[:, 0] < WM - 3) & (
        matches_im_map[:, 1] >= 3) & (matches_im_map[:, 1] < HM - 3)
    valid_matches_query = (matches_im_query[:, 0] >= 3) & (matches_im_query[:, 0] < WQ - 3) & (
        matches_im_query[:, 1] >= 3) & (matches_im_query[:, 1] < HQ - 3)
    valid_matches = valid_matches_map & valid_matches_query
    matches_im_map = matches_im_map[valid_matches]
    matches_im_query = matches_im_query[valid_matches]
    
    matches_confs = np.minimum(
        conf_list[1][matches_im_map[:, 1], matches_im_map[:, 0]],
        conf_list[0][matches_im_query[:, 1], matches_im_query[:, 0]]
    )

    # Adjust coordinates (from cv2 to colmap and back)
    matches_im_query = matches_im_query.astype(np.float64)
    matches_im_map = matches_im_map.astype(np.float64)
    matches_im_query[:, 0] += 0.5
    matches_im_query[:, 1] += 0.5
    matches_im_map[:, 0] += 0.5
    matches_im_map[:, 1] += 0.5
    # Rescale coordinates (assuming no scaling)
    # From colmap back to cv2
    matches_im_query[:, 0] -= 0.5
    matches_im_query[:, 1] -= 0.5
    matches_im_map[:, 0] -= 0.5
    matches_im_map[:, 1] -= 0.5
    return [], matches_im_query, matches_im_map, matches_confs, output  # Return output for pointmaps

def resize_and_pad_image(img, max_size, patch_size=16):
    """
    Resize the image so that the largest dimension does not exceed max_size.
    Then pad the image so that both height and width are multiples of patch_size.
    Returns the resized and padded image and the padding applied.
    """
    W, H = img.size
    scaling_factor = max_size / max(W, H)
    if scaling_factor < 1.0:
        new_W = int(W * scaling_factor)
        new_H = int(H * scaling_factor)
        img = img.resize((new_W, new_H), Image.BICUBIC)
    else:
        new_W, new_H = W, H

    # Calculate padding to make new_W and new_H multiples of patch_size
    pad_W = (patch_size - new_W % patch_size) if new_W % patch_size != 0 else 0
    pad_H = (patch_size - new_H % patch_size) if new_H % patch_size != 0 else 0

    if pad_W != 0 or pad_H != 0:
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom)  # left, top, right, bottom
        img = ImageOps.expand(img, padding)
    else:
        padding = (0, 0, 0, 0)

    return img, padding

def scale_intrinsics(K, original_size, new_size, padding):
    """
    Scale the camera intrinsics based on the resizing and padding applied to the image.
    """
    W_orig, H_orig = original_size
    W_new, H_new = new_size
    pad_left, pad_top, pad_right, pad_bottom = padding

    scaling_factor_x = W_new / W_orig
    scaling_factor_y = H_new / H_orig

    K_scaled = K.copy()
    K_scaled[0, :] *= scaling_factor_x
    K_scaled[1, :] *= scaling_factor_y

    # Adjust the principal point based on padding
    K_scaled[0, 2] += pad_left
    K_scaled[1, 2] += pad_top

    return K_scaled

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

def get_prediction(model_name, device, intrinsics, images, image_name):
    reference_image_path,query_image_path = images
    conf_thr = 0.8
    max_image_size = 512
    
    # Load Model
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    fast_nn_params = dict(device=device, dist='dot', block_size=2**13)

    
    # Load Images
    query_rgb = Image.open(query_image_path).convert('RGB')
    reference_rgb = Image.open(reference_image_path).convert('RGB')

    # Store original sizes
    WQ_original, HQ_original = query_rgb.size
    WM_original, HM_original = reference_rgb.size

    # Resize and pad images
    patch_size = 16  # As per model's requirement
    query_rgb_resized, query_padding = resize_and_pad_image(query_rgb, max_image_size, patch_size)
    reference_rgb_resized, reference_padding = resize_and_pad_image(reference_rgb, max_image_size, patch_size)

    # Update the image sizes after resizing and padding
    WQ, HQ = query_rgb_resized.size
    WM, HM = reference_rgb_resized.size

    
    # Set Default Intrinsics using Ground Truth
    def default_intrinsics():
        f = 590.1284
        cx = 270.2328
        cy = 352.1782
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]], dtype=np.float64)
        return K


    query_intrinsics = default_intrinsics()
    reference_intrinsics = default_intrinsics()

    # Scale intrinsics based on resizing and padding
    query_intrinsics = scale_intrinsics(
        query_intrinsics,
        (WQ_original, HQ_original),
        (WQ, HQ),
        query_padding
    )
    reference_intrinsics = scale_intrinsics(
        reference_intrinsics,
        (WM_original, HM_original),
        (WM, HM),
        reference_padding
    )

    
    # Set Default Poses
    query_pose = None  
    reference_pose = None

    # Replace ImgNorm with standard normalization
    transform = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
    ])

    query_rgb_tensor = transform(query_rgb_resized)  # [C, H, W]
    reference_rgb_tensor = transform(reference_rgb_resized)  # [C, H, W]

    
    # Prepare the Views
    query_view = {
        'rgb_rescaled': query_rgb_tensor,
        'intrinsics': query_intrinsics,
        'image_name': os.path.basename(query_image_path),
        'cam_to_world': query_pose
    }

    map_view = {
        'rgb_rescaled': reference_rgb_tensor,
        'intrinsics': reference_intrinsics,
        'image_name': os.path.basename(reference_image_path),
        'cam_to_world': reference_pose
    }

    
    # Perform Coarse Matching
    valid_pts3d, matches_im_query, matches_im_map, matches_conf, output = coarse_matching(
        query_view, map_view, model, device, pixel_tol=5, fast_nn_params=fast_nn_params
    )

    
    # Apply Confidence Threshold
    if len(matches_conf) > 0:
        mask = matches_conf >= conf_thr
        matches_im_query = matches_im_query[mask]
        matches_im_map = matches_im_map[mask]
        matches_conf = matches_conf[mask]
        
    # Prepare Matched Points
    points_query = matches_im_query.astype(np.float64)
    points_map = matches_im_map.astype(np.float64)

    
    # Estimate Essential Matrix using RANSAC
    E, mask_E = cv2.findEssentialMat(
        points_map,
        points_query,
        focal=query_intrinsics[0, 0],
        pp=(query_intrinsics[0, 2], query_intrinsics[1, 2]),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0 # Adjust threshold as needed
    )

    if E is None:
        prediction = f"{IMAGE_I} 0 0 0 0 0 0 0 42"

        with open("pose_s00460.txt", "a") as file:
            file.write(prediction + "\n")
        # It fails here on MapFree
    
    # Select Inlier Matches
    mask_E = mask_E.astype(bool).ravel()
    inliers = mask_E
    points_query_inliers = points_query[inliers]
    points_map_inliers = points_map[inliers]
    
    # Recover Pose from Essential Matrix
    retval, R, t, mask_pose = cv2.recoverPose(
        E,
        points_map_inliers,
        points_query_inliers,
        focal=query_intrinsics[0, 0],
        pp=(query_intrinsics[0, 2], query_intrinsics[1, 2])
    )

    # Get 3D points from model's output
    pred1 = output['pred1']
    pred2 = output['pred2']

    pts3d_query = pred1['pts3d'].squeeze(0).cpu().numpy()
    pts3d_map = pred2['pts3d_in_other_view'].squeeze(0).cpu().numpy()

    # Convert matches_im_query and matches_im_map to integer indices
    indices_query = points_query_inliers.astype(int)
    indices_map = points_map_inliers.astype(int)

    # Ensure indices are within bounds
    indices_query[:, 0] = np.clip(indices_query[:, 0], 0, pts3d_query.shape[1] - 1)
    indices_query[:, 1] = np.clip(indices_query[:, 1], 0, pts3d_query.shape[0] - 1)
    indices_map[:, 0] = np.clip(indices_map[:, 0], 0, pts3d_map.shape[1] - 1)
    indices_map[:, 1] = np.clip(indices_map[:, 1], 0, pts3d_map.shape[0] - 1)

    # Extract corresponding 3D points
    pts3d_query_matched = pts3d_query[indices_query[:, 1], indices_query[:, 0]]  # [N, 3]
    pts3d_map_matched = pts3d_map[indices_map[:, 1], indices_map[:, 0]]        # [N, 3]

    scale_factor = compute_scale_factor(pts3d_query_matched, pts3d_map_matched, R, t, use_median=True)

    # Scale the translation vector
    t_scaled = scale_factor * (t / np.linalg.norm(t))

    # Construct the relative pose matrix [R | t_scaled]
    pose_rel = np.hstack((R, t_scaled))
    pose_rel = np.vstack((pose_rel, np.array([0, 0, 0, 1])))

    
    # Output the Estimated Relative Pose
    #print("Estimated Relative Pose [R | t_scaled]:")
    #np.set_printoptions(suppress=True)
    #print(pose_rel)

    rot_quat = mat2quat(pose_rel[:3, :3])
    prediction = f"{image_name} {rot_quat[0]} {rot_quat[1]} {rot_quat[2]} {rot_quat[3]} {pose_rel[0, 3]} {pose_rel[1, 3]} {pose_rel[2, 3]} {len(inliers)}"

    return prediction