import os
import numpy as np
from numpy.typing import NDArray  # Import NDArray
import cv2

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R

import poselib

# visualize a few matches
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl


def scale_intrinsics(K: NDArray, prev_w: float, prev_h: float, master_w: float, master_h: float) -> NDArray:
    """Scale the intrinsics matrix by a given factor .

    Args:
        K (NDArray): 3x3 intrinsics matrix
        scale (float): Scale factor

    Returns:
        NDArray: Scaled intrinsics matrix
    """
    #540 x 960 --> 288 x 512

    assert K.shape == (3, 3), f"Expected (3, 3), but got {K.shape=}"

    scale_w = master_w / prev_w  # sizes of the images in the Mast3r dataset
    scale_h = master_h / prev_h  # sizes of the images in the Mast3r dataset

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_w
    K_scaled[0, 2] *= scale_w
    K_scaled[1, 1] *= scale_h
    K_scaled[1, 2] *= scale_h

    return K_scaled

def run_pnp(pts2D, pts3D, K):
    """
    intrinsics= K

    mode='cv2'
    """
    success, r_pose, t_pose, _ = cv2.solvePnPRansac(pts3D, pts2D, K, None, flags=cv2.SOLVEPNP_SQPNP,
                                                    iterationsCount=10_000,
                                                    reprojectionError=5,
                                                    confidence=0.9999)
    if not success:
        return False, None
    r_pose = cv2.Rodrigues(r_pose)[0]  # world2cam == world2cam2
    RT = np.r_[np.c_[r_pose, t_pose], [(0,0,0,1)]] # world2cam2
    return True, np.linalg.inv(RT)  # cam2toworld

def run_poselib(pts2D, pts3D, K, width, height):
    confidence = 0.9999
    iterationsCount = 10_000

    camera = {'model': 'PINHOLE', 'width': width, 'height': height, 'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}

    pts2D = np.copy(pts2D)
    pts2D[:, 0] += 0.5
    pts2D[:, 1] += 0.5
    pose, info = poselib.estimate_absolute_pose(pts2D, pts3D, camera, {'max_reproj_error': 5, 'max_iterations': iterationsCount, 'success_prob': confidence}, {})

    RT = pose.Rt  # (3x4)
    RT = np.r_[RT, [(0,0,0,1)]]  # world2cam
    prediction = np.linalg.inv(RT)  # cam2toworld
    return ('success' in info and info['success']), prediction

def relative_transformation(pose0, pose1):
    return np.linalg.inv(pose0) @ pose1	

def load_images_from_folder(input_folder):
    input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    input_files = [f for f in input_files if os.path.splitext(f)[1].lower() in image_extensions]
    assert input_files, "No image files found in the specified folder."
    return input_files

def get_mast3r_output(MODEL_NAME, IMAGES, DEVICE, BORDER):
    # Load model, run inference
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)
    images = load_images(IMAGES, size=512, verbose=False)
    output = inference([tuple(images)], model, DEVICE, batch_size=1, verbose=False)

    # Raw predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1 = pred1['desc'].squeeze(0).detach()
    desc2 = pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=DEVICE, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= BORDER) & \
                        (matches_im0[:, 0] < int(W0) - BORDER) & \
                        (matches_im0[:, 1] >= BORDER) & \
                        (matches_im0[:, 1] < int(H0) - BORDER)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= BORDER) & \
                        (matches_im1[:, 0] < int(W1) - BORDER) & \
                        (matches_im1[:, 1] >= BORDER) & \
                        (matches_im1[:, 1] < int(H1) - BORDER)
    
    valid_matches = valid_matches_im0 & valid_matches_im1

    # matches are Nx2 image coordinates.
    matches_im0 = matches_im0[valid_matches]
    matches_im1 = matches_im1[valid_matches]

    # Convert the other outputs to numpy arrays
    pts3d_im0 = pred1['pts3d'].squeeze(0).detach().cpu().numpy()
    pts3d_im1 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy()

    conf_im0 = pred1['conf'].squeeze(0).detach().cpu().numpy()
    conf_im1 = pred2['conf'].squeeze(0).detach().cpu().numpy()

    desc_conf_im0 = pred1['desc_conf'].squeeze(0).detach().cpu().numpy()
    desc_conf_im1 = pred2['desc_conf'].squeeze(0).detach().cpu().numpy()

    return matches_im0, matches_im1, pts3d_im0, pts3d_im1, conf_im0, conf_im1, desc_conf_im0, desc_conf_im1, view1, view2

def visualize_few_matches(matches_im0, matches_im1, view1, view2, n_viz=20):
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)

def visualize_matches_on_images(matches_im0, matches_im1, path0, path1, w, h, n_viz=20):
    img0 = cv2.resize(cv2.cvtColor(cv2.imread(path0), cv2.COLOR_BGR2RGB), (w, h))
    img1 = cv2.resize(cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB), (w, h)) 

    H0, W0 = img0.shape[:2] 
    H1, W1 = img1.shape[:2]
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)

def load_image_pair(image_paths, size=512):
    """Loads and returns a pair of images from specified paths."""
    return load_images(image_paths, size=size)