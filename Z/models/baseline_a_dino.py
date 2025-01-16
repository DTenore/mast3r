import sys
import os
import numpy as np
import torch
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

import poselib

from transformers import Dinov2Model, AutoImageProcessor
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#FIXME: consistnenlty call img0 and img1 everywhere!!!!

def get_dino_patch_grids(device, processor, dino_model, images, heights, widths): 
    # TODO: add batches?

    # Load the DINOv2 model
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
    dino_model.eval()  # Set the model to evaluation mode

    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results = []

    for i, image in enumerate(images):
        # Load & resize the image
        img = Image.open(image).convert('RGB')
        new_width = (widths[i] // 14) * 14
        new_height = (heights[i] // 14) * 14
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)

        # Preprocess the image
        img_tensor = preprocess(img_resized).unsqueeze(0)  # Add batch dimension

        # Extract per-patch features
        with torch.no_grad():  # Disable gradient computation
            outputs = dino_model.forward_features(img_tensor)

        # Extract per-patch features (excluding [CLS] token)
        patch_features = outputs['x_norm_patchtokens']  # Shape: (1, num_patches, hidden_size)

        # Calculate number of patches along each dimension
        num_patches_height = new_height // 14
        num_patches_width = new_width // 14

        # Reshape to grid
        patch_features_grid = patch_features.reshape(1, num_patches_height, num_patches_width, -1)[0]

        results.append(patch_features_grid)

    if False:
        plt.figure(figsize=(16, 8))
        for idx, (image_path, patch_features_grid) in enumerate(zip(images, results)):
            # Flatten patch features
            patch_features_flat = patch_features_grid.reshape(-1, patch_features_grid.shape[-1])

            # Perform PCA to reduce to 3 components
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(patch_features_flat)

            # Reshape PCA features back to grid
            num_patches_height, num_patches_width, _ = patch_features_grid.shape
            pca_features_grid = pca_features.reshape(num_patches_height, num_patches_width, 3)

            # Normalize the PCA components to [0, 255] for visualization
            pca_features_normalized = (pca_features_grid - pca_features_grid.min()) / (
                pca_features_grid.max() - pca_features_grid.min()
            )
            pca_features_rgb = (pca_features_normalized * 255).astype(np.uint8)

            # Create PCA-based RGB overlay image
            upsample_factor = 14
            pca_overlay = Image.fromarray(pca_features_rgb, 'RGB').resize(
                (num_patches_width * upsample_factor, num_patches_height * upsample_factor), Image.NEAREST
            )

            # Load the original image (resized to match PCA overlay size)
            original_image = Image.open(image_path).convert('RGB')
            original_image_resized = original_image.resize(
                (num_patches_width * 14, num_patches_height * 14), Image.LANCZOS
            )

            # Display side-by-side images
            plt.subplot(1, len(images), idx + 1)
            plt.imshow(Image.blend(original_image_resized, pca_overlay, alpha=0.5))
            plt.title(f"Image {idx + 1} with PCA Overlay")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    return results


def get_similarity(patches, match0, match1, H0, W0, H1, W1):
    H0 = float(H0)
    W0 = float(W0)
    H1 = float(H1)
    W1 = float(W1)

    # Match shape: w x h
    patches0, patches1 = patches
    p0_h, p0_w, _ = patches0.shape
    p1_h, p1_w, _ = patches1.shape


    features0 = patches0[
        (np.floor(match0[:, 1] * (p0_h / H0))).astype(int),
        (np.floor(match0[:, 0] * (p0_w / W0))).astype(int)
    ]

    features1 = patches1[
        (np.floor(match1[:, 1] * (p1_h / H1))).astype(int),
        (np.floor(match1[:, 0] * (p1_w / W1))).astype(int)
    ]

    # Normalize for cosine similarity
    features0_normalized = torch.nn.functional.normalize(features0, p=2, dim=1)
    features1_normalized = torch.nn.functional.normalize(features1, p=2, dim=1)

    # Compute row-wise cosine similarity
    cosine_similarities = torch.sum(features0_normalized * features1_normalized, dim=1)
    return cosine_similarities


def printer(q, t):
    return f"{q[0]} {q[1]} {q[2]} {q[3]} {t[0][0]} {t[1][0]} {t[2][0]}"

def get_prediction(model_name, device, intrinsics, IMAGES, image_name, visualize=False): 

    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    images = load_images(IMAGES, size=512, verbose=False)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw mast3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    orig_W0, orig_H0 = (int(x) for x in view1['original_shape'])
    orig_W1, orig_H1 = (int(x) for x in view1['original_shape'])

    H0, W0 = (int(x) for x in view1['true_shape'][0])
    H1, W1 = (int(x) for x in view2['true_shape'][0])

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)


    
    # ignore small border around the edge
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0 - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0 - 3)    
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < W1 - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < H1 - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
    
    # DO DINO

    # Initialize processor and model
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dino_model = Dinov2Model.from_pretrained('facebook/dinov2-base')
    dino_model.eval()  # Set model to evaluation mode
    dino_model.to(device)

    # Get DINO patch grids
    dino_grids = get_dino_patch_grids(device, processor, dino_model, IMAGES, [H0, H1], [W0, W1]) 
    # For each match calculate the cosine similarity using corresponding DINO features
    cosine_similarities = get_similarity(dino_grids, matches_im0, matches_im1, H0, W0, H1, W1)
    # Thershold cosine similarities 
    valid_similarities = cosine_similarities[:] >= 0.6 # WIP: what threshold is best??? we will try 0.6 for now
    # Remove matches with too low similarities
    matches_im0, matches_im1 = matches_im0[valid_similarities], matches_im1[valid_similarities]
    

    # Convert point outputs to numpy arrays
    pts3d_im0 = pred1['pts3d'].squeeze(0).detach().cpu().numpy()
    pts3d_im1 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy()

    # Convert matches to numpy float32 arrays
    points1 = matches_im0.astype(np.float32)
    points2 = matches_im1.astype(np.float32)

    # If not enough matches, return empty string (this will skip the line in the result file, which is okay)
    if (len(points1) < 5) or (len(points2) < 5):
        return ""

    # Intrinsic matrices
    intrinsics_0 = intrinsics[0]
    intrinsics_1 = intrinsics[1]

    K1 = scale_intrinsics(intrinsics_0, orig_W0, orig_H0, W0, H0)
    K2 = scale_intrinsics(intrinsics_1, orig_W1, orig_H1, W1, H1)

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

    # Normalize threshold
    threshold = 0.75
    avg_diagonal = (K1[0][0] + K1[1][1] + K2[0][0] + K2[1][1]) / 4
    normalized_threshold = threshold / avg_diagonal

    ransac_options = poselib.RansacOptions()
    ransac_options['max_reproj_error'] = 4.0  # or your chosen threshold in pixels
    ransac_options['success_prob'] = 0.999

  
    # Get absolute pose
    pose, info = poselib.estimate_absolute_pose(matches_im1.astype(np.float32), pts3d_im0[matches_im0[:, 1], matches_im0[:, 0], :], camera1, {'max_reproj_error': 5, 'max_iterations': 10_000, 'success_prob': 0.9999}, {})
    Rt = pose.Rt  # (3x4)

    R = Rt[:,:3]    
    t = Rt[:,3:]
    
    quat = mat2quat(R)

    # Prepare result output string
    res = printer(quat, t)

    # Visualize matches if flag is set
    if visualize:
        mask = np.array(info["inliers"])

        mask_E = mask.ravel().astype(bool)
        points1_inliers = points1[mask_E]
        points2_inliers = points2[mask_E]

        # Convert inlier matches to integer indices
        indices_query = points1_inliers.astype(int)
        indices_map = points2_inliers.astype(int)

        visualize_all_features(points1_inliers, points2_inliers, IMAGES[0], IMAGES[1], 384, 512)

        visualize_all_features(points1_inliers, points2_inliers, IMAGES[0], IMAGES[1], 384, 512)

    return f"{image_name} {res} {len(valid_matches)}"

