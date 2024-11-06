# --------------------------------------------------------
# MASt3R w/o Gradio (Modified Version)
# --------------------------------------------------------
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

from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy

import matplotlib.pyplot as pl
import matplotlib.cm as cm
import warnings
from PIL import Image

from transforms3d.quaternions import mat2quat

from utils.utils import visualize_matches_on_images

#sys.stderr = open(os.devnull, 'w')

warnings.filterwarnings("ignore", category=FutureWarning)

device = 'cuda'
model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

FOLDER = "/home/dario/DATASETS/map-free-reloc/data/mapfree/val/s00460/"
IMAGE_0 = "seq0/frame_00000.jpg"

for i in tqdm.tqdm(range(5, 569, 5), file=sys.stdout):
    IMAGE_I = "seq1/frame_" + "{:05d}".format(i) + ".jpg"
    IMAGES = [FOLDER + IMAGE_0, FOLDER + IMAGE_I] # Image paths

    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    imgs = load_images(IMAGES, size=512)

    # Create pairs for the sparse global alignment
    pairs = [(imgs[0], imgs[1]),(imgs[1], imgs[0])]

    # Set up cache directory and parameters
    tmp_path = tempfile.mkdtemp()
    cache_dir = os.path.join(tmp_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Perform sparse global alignment
    scene = sparse_global_alignment(
        IMAGES, pairs, cache_dir, model,
        lr1=0.07, niter1=500, lr2=0.014, niter2=200, device=device,
        opt_depth='depth' in 'refine+depth', shared_intrinsics=True, matching_conf_thr=5.0
    )

    # Extract and export 3D model from scene
    pts3d, depthmaps, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))
    msk = to_numpy([c > 1.5 for c in confs])
    focals = to_numpy(scene.get_focals().cpu())
    cams2world = to_numpy(scene.get_im_poses().cpu())
    imgs_np = to_numpy(scene.imgs)

    world_to_c0 = np.linalg.inv(cams2world[0])  # world to cam0

    # Transform camera poses to cam0 (first is just identity)
    cams_to_c0 = []
    for i in range(len(pts3d)):
        ci_to_world = cams2world[i]  # cam_i to world
        ci_to_c0 = world_to_c0 @ ci_to_world  # cam_i to cam0
        cams_to_c0.append(ci_to_c0)
        #print(f"Camera {i} pose: \n {cams2world[i]}")

    # Get relative transformation between the two frames
    transform = np.linalg.inv(cams_to_c0[0]) @ cams_to_c0[1]

    rot_quat = mat2quat(transform[:3, :3])

    prediction = f"{IMAGE_I} {rot_quat[0]} {rot_quat[1]} {rot_quat[2]} {rot_quat[3]} {transform[0, 3]} {transform[1, 3]} {transform[2, 3]} {42}"
    
    #FIXME: Can you Visualize matches? How?

    with open("DEMO_pose_s00460.txt", "a") as file:
        file.write(prediction + "\n")

# New results
#{
#  "Average Median Translation Error": 1.6394335168756804,
#  "Average Median Rotation Error": 58.91280893713621,
#  "Average Median Reprojection Error": 354.11617415778255,
#  "Precision @ Pose Error < (25.0cm, 5deg)": 0.0,
#  "AUC @ Pose Error < (25.0cm, 5deg)": 0.0,
#  "Precision @ VCRE < 90px": 0.0,
#  "AUC @ VCRE < 90px": 0.0,
#  "Estimates for % of frames": 0.003097842476080818
#}

#{
#  "Average Median Translation Error": 2.5215314123662473,
#  "Average Median Rotation Error": 73.0139952540139,
#  "Average Median Reprojection Error": 285.5289114070251,
#  "Precision @ Pose Error < (25.0cm, 5deg)": 0.0,
#  "AUC @ Pose Error < (25.0cm, 5deg)": 0.0,
#  "Precision @ VCRE < 90px": 0.03508771929824561,
#  "AUC @ VCRE < 90px": 0.035087718372851344,
#  "Estimates for % of frames": 0.9912280701754386
#}