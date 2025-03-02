import os
import re
import sys
import tempfile
import tqdm
import zipfile
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoImageProcessor

# Adjust system path
Z_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if Z_PATH not in sys.path:
    sys.path.insert(0, Z_PATH)

# Import project-specific modules
from utils import path_to_dust3r_and_mast3r
from utils.file_io import write_prediction, load_intrinsics

from mast3r.model import AsymmetricMASt3R
from mast3r.model3 import DINOMASt3R

import importlib

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
# This can suppress all errors -> not recommended for now
#sys.stderr = open(os.devnull, 'w')

# Parameters
device = 'cuda'
data_folder = "/home/dario/DATASETS/map-free-reloc/data/mapfree/"
zip_output_path = "/home/dario/_MINE/mast3r/Z/_submissions"
image_0 = "seq0/frame_00000.jpg"

POSE_ESTIMATION_MODEL = "baseline_a"
SET = "test"
SIZE = "XL"
#MODEL = "MASt3R"
MODEL = "DINOMASt3R"
TOP_K = 1


def get_folder_range(directory):
    # List all subfolders in the directory
    subfolders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    
    # Extract numbers using regex (assuming folder names are in the format 'sXXXXX')
    numbers = [int(re.search(r"s(\d+)", folder).group(1)) for folder in subfolders if re.search(r"s(\d+)", folder)]
    
    return min(numbers), max(numbers) 

folder = os.path.join(data_folder, SET)
start, end = get_folder_range(folder)

scenes = {}
scenes["S"] = ["s00465"]
scenes["M"] = ["s00465", "s00475", "s00485", "s00495"]
scenes["L"] = [f"s{i:05d}" for i in range(start, end + 1, 5)]
scenes["XL"] = [f"s{i:05d}" for i in range(start, end + 1, 1)]

scenes["SP"] = [f"s{i:05d}" for i in range(20)]

SCENES = scenes[SIZE]

START = 0
VISUALIZE = False
VISUALIZE_INTERVAL = 5 # Visualize every 5 predictions NOT every 5 frames

def get_prediction_function(model_name):
    #Make model_name lowercase
    model_name = model_name.lower()
    
    try:
        # Dynamically import the model's module
        module = importlib.import_module(f"models.{model_name}")
        # Retrieve the `get_prediction` function from the module
        return getattr(module, "get_prediction")
    except ModuleNotFoundError:
        raise ValueError(f"Unknown model name: {model_name}")


def main():
    get_prediction = get_prediction_function(POSE_ESTIMATION_MODEL)

    # Create temporary folder for submission
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_dir = Path(__file__).parent / f"_tmp/submission-{POSE_ESTIMATION_MODEL}-{timestamp}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    if "dino" in POSE_ESTIMATION_MODEL:
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
        dino_model.eval()  # Set the model to evaluation mode
        dino_args = {'model': dino_model, 'processor': processor}
    else:
        dino_args = None

    if MODEL == "MASt3R":
        mast3r_model = AsymmetricMASt3R.from_pretrained("/home/dario/_MINE/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", verbose=False).to(device)
        if True:
            CKPT = "20250228_1153_MASt3R_D_e6_sc10_ac16_hb30"
            state_dict = torch.load("/home/dario/_MINE/mast3r/Z/checkpoints/"+CKPT + ".pth", map_location=device)
            mast3r_model.load_state_dict(state_dict, strict=False)
            model_name = f"MASt3R_{CKPT}"
        else:
            model_name = "MASt3R"
    else:
        mast3r_model = DINOMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)


        if True:
            CKPT = "20250302_1010_DINOMASt3R_D_NP_e6_sc100_ac16_hb30"
            state_dict = torch.load("/home/dario/_MINE/mast3r/Z/checkpoints/"+CKPT + ".pth", map_location=device)
            mast3r_model.load_state_dict(state_dict, strict=False)
            model_name = f"DINOMASt3R_{CKPT}"

            mast3r_model.topK = 0.75

        else:
            model_name = f"DINOMASt3R_topK{int(TOP_K*100)}"

    # Top-level progress bar for all scenes
    with tqdm.tqdm(SCENES, desc="Processing scenes", file=sys.stdout, dynamic_ncols=True) as scene_pbar:
        for SCENE in scene_pbar:
            intrinsics_dict, frame_width, frame_height = load_intrinsics(os.path.join(folder, SCENE, "intrinsics.txt"))
            intrincics_0 = intrinsics_dict[image_0]

            scene_file_path = temp_dir / f"pose_{SCENE}.txt"
            scene_pbar.set_description(f"Processing scene {SCENE[1:]}")

            # Count number of frames in the scene folder
            scene_folder = Path(folder) / SCENE / "seq1"
            num_frames = len(list(scene_folder.glob("frame_*.jpg")))

            # Open file for the current scene
            with scene_file_path.open("w") as scene_file:
                # Single progress bar for frames within the current scene
                with tqdm.tqdm(total=(num_frames-START) // 5, desc=f"Processing frames", leave=False, file=sys.stdout, dynamic_ncols=True) as frame_pbar:
                    # Process images and generate predictions
                    for i in range(START, num_frames, 5):
                        frame_pbar.set_description(f"Processing frame {i:05d}")

                        image_i = f"seq1/frame_{i:05d}.jpg"
                        images = [os.path.join(folder, SCENE, image_0), os.path.join(folder, SCENE, image_i)]
                        
                        if image_i not in intrinsics_dict:
                            #scene_pbar.write(f"Missing intrinsics for {image_i} in scene {SCENE}, skipping.")
                            continue

                        intrincics_i = intrinsics_dict[image_i]

                        # Set intrinsics if in README mode
                        intrinsics = [intrincics_0, intrincics_i]
                        
                        # Only visualize if flag is set and every n-th desired prediction
                        visualization = VISUALIZE and ((i/5) % VISUALIZE_INTERVAL == 0)

                        # Get prediction and write to output file
                        prediction = get_prediction(mast3r_model, device, intrinsics, images, image_i, 
                          visualize=visualization, dino_args=dino_args)
                        if(prediction):
                            scene_file.write(prediction + '\n')
                        
                        # Update frame progress bar
                        frame_pbar.update(1)

    
    # Ensure the output directory exists
    output_dir = Path(zip_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Zip the temporary folder and save to the predefined location
    zip_file_path = output_dir / f"{timestamp}_{SET}_{SIZE}_{model_name}.zip"
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in temp_dir.glob('*'):
            zipf.write(file, arcname=file.name)

    # Delete the temporary folder
    for file in temp_dir.glob('*'):
        file.unlink()
    temp_dir.rmdir()

if __name__ == "__main__":
    main()
