import numpy as np
import sys
import tqdm
import os

from ZZ_utils import *

from transforms3d.quaternions import mat2quat

sys.stderr = open(os.devnull, 'w')


MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
DEVICE = 'cuda'
BORDER = 3

FOLDER = "/home/dario/DATASETS/map-free-reloc/data/mapfree/val/s00460/"
IMAGE_0 = "seq1/frame_00000.jpg"

# Hardcoded MapFree Intrinsics
INTRINSICS = [[590.3821, 0, 269.6031], 
                [0, 590.3821, 270.2328], 
                [0, 0, 1]]
scale_K = scale_intrinsics(np.array(INTRINSICS), 540, 720, 384, 512)

for i in tqdm.tqdm(range(5, 569, 5), file=sys.stdout):
    IMAGE_I = "seq1/frame_" + "{:05d}".format(i) + ".jpg"
    IMAGES = [FOLDER + IMAGE_0, FOLDER + IMAGE_I] # Image paths

    #Main function
    matches_im0, matches_im1, pts3d_im0, pts3d_im1, conf_im0, conf_im1, desc_conf_im0, desc_conf_im1 = get_mast3r_output(MODEL_NAME, IMAGES, DEVICE, BORDER)

    # Predicted Transform copied from visloc.py
    ret_val, transform = run_poselib(matches_im1.astype(np.float32), pts3d_im0[matches_im0[:, 1], matches_im0[:, 0], :], scale_K, 288, 512)

    rot_quat = mat2quat(transform[:3, :3])
    trans = transform[:3, 3]

    prediction = f"{IMAGE_I} {rot_quat[0]} {rot_quat[1]} {rot_quat[2]} {rot_quat[3]} {trans[0]} {trans[1]} {trans[2]} 42"

    with open("pose_s00460.txt", "a") as file:
        file.write(prediction + "\n")

# CV2
#{
# "Average Median Translation Error": 2.3369172195007524,
# "Average Median Rotation Error": 70.91078456892517,
# "Average Median Reprojection Error": 289.9466352091038,
# "Precision @ Pose Error < (25.0cm, 5deg)": 0.0,
# "AUC @ Pose Error < (25.0cm, 5deg)": 0.0,
# "Precision @ VCRE < 90px": 0.14912280701754385,
# "AUC @ VCRE < 90px": 0.14912280308461823,
# "Estimates for % of frames": 0.9912280701754386
#}

# Poselib
#{
# "Average Median Translation Error": 2.341405540504688,
# "Average Median Rotation Error": 70.55557510772276,
# "Average Median Reprojection Error": 290.51465557925155,
# "Precision @ Pose Error < (25.0cm, 5deg)": 0.0,
# "AUC @ Pose Error < (25.0cm, 5deg)": 0.0,
# "Precision @ VCRE < 90px": 0.14035087719298245,
# "AUC @ VCRE < 90px": 0.14035087349140538,
# "Estimates for % of frames": 0.9912280701754386
#}