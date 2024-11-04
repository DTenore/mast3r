import os
import numpy as np
import tempfile
import sys
import tqdm

Z_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if Z_PATH not in sys.path:
    sys.path.insert(0, Z_PATH)
from utils import path_to_dust3r_and_mast3r

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.model import AsymmetricMASt3R

from dust3r.utils.device import to_numpy
from transforms3d.quaternions import mat2quat

from utils.utils import load_images

def get_prediction(model_name, device, intrinsics, images, image_name):    
    # Initialize model
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    imgs = load_images(images, size=512, verbose=False)

    # Create pairs for sparse global alignment
    pairs = [(imgs[0], imgs[1]), (imgs[1], imgs[0])]
    
    # Set up cache directory
    tmp_path = tempfile.mkdtemp()
    cache_dir = os.path.join(tmp_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Perform sparse global alignment
    scene = sparse_global_alignment(
        images, pairs, cache_dir, model,
        lr1=0.07, niter1=500, lr2=0.014, niter2=200, device=device,
        opt_depth='depth' in 'refine+depth', shared_intrinsics=True, matching_conf_thr=5.0
    )

    # Extract and export 3D model from scene
    pts3d, depthmaps, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))
    msk = to_numpy([c > 1.5 for c in confs])
    focals = to_numpy(scene.get_focals().cpu())
    cams2world = to_numpy(scene.get_im_poses().cpu())
    imgs_np = to_numpy(scene.imgs)

    world_to_c0 = np.linalg.inv(cams2world[0])
    cams_to_c0 = [world_to_c0 @ cams2world[i] for i in range(len(pts3d))]

    transform = np.linalg.inv(cams_to_c0[0]) @ cams_to_c0[1]
    rot_quat = mat2quat(transform[:3, :3])

    # Format prediction string
    prediction = f"{image_name} {rot_quat[0]} {rot_quat[1]} {rot_quat[2]} {rot_quat[3]} {transform[0, 3]} {transform[1, 3]} {transform[2, 3]} {42}"
    return prediction
