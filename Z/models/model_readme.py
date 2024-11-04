import numpy as np
import sys
import tqdm
import os

Z_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if Z_PATH not in sys.path:
    sys.path.insert(0, Z_PATH)
from utils import path_to_dust3r_and_mast3r

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs


from dust3r.inference import inference
from dust3r.utils.image import load_images

from utils.utils import *

from transforms3d.quaternions import mat2quat


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

    return matches_im0, matches_im1, pts3d_im0, pts3d_im1, valid_matches

def get_prediction(model_name, device, intrinsics, imgs, image_name):
    matches_im0, matches_im1, pts3d_im0, pts3d_im1, valid_matches = get_mast3r_output(model_name, imgs, device, 3)

    scale_K = scale_intrinsics(np.array(intrinsics), 540, 720, 384, 512)

    # Predicted Transform copied from visloc.py
    ret_val, transform = run_poselib(matches_im1.astype(np.float32), pts3d_im0[matches_im0[:, 1], matches_im0[:, 0], :], scale_K, 288, 512)

    rot_quat = mat2quat(transform[:3, :3])
    trans = transform[:3, 3]

    prediction = f"{image_name} {rot_quat[0]} {rot_quat[1]} {rot_quat[2]} {rot_quat[3]} {trans[0]} {trans[1]} {trans[2]} {len(valid_matches)}"

    return prediction