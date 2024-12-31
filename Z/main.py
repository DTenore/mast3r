import os
import sys
import tempfile
import tqdm
import zipfile
from datetime import datetime
from pathlib import Path

# Adjust system path
Z_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if Z_PATH not in sys.path:
    sys.path.insert(0, Z_PATH)

# Import project-specific modules
from utils import path_to_dust3r_and_mast3r
from utils.file_io import write_prediction, load_intrinsics
from utils.utils import load_image_pair

import importlib

#sys.stderr = open(os.devnull, 'w')

# Parameters
device = 'cuda'
model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
output_file = os.path.join(os.path.dirname(__file__), "pose_s00460.txt")
folder = "/home/dario/DATASETS/map-free-reloc/data/mapfree/val/"
image_0 = "seq0/frame_00000.jpg"

MODEL = "baseline"
SCENES = ["s00465"]
#SCENES = [f"s{i:05d}" for i in range(460, 525, 5)]
ZIP_OUTPUT_PATH = "/home/dario/_MINE/mast3r"

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
    get_prediction = get_prediction_function(MODEL)

    # Create temporary folder for submission
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_dir = Path(__file__).parent / f"_tmp/submission-{MODEL}-{timestamp}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Top-level progress bar for all scenes
    with tqdm.tqdm(SCENES, desc="Processing scenes", file=sys.stdout) as scene_pbar:
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
                with tqdm.tqdm(total=(num_frames-START) // 5, desc=f"Processing frames", leave=False, file=sys.stdout) as frame_pbar:
                    # Process images and generate predictions
                    for i in range(START, num_frames, 5):
                        frame_pbar.set_description(f"Processing frame {i:05d}")

                        image_i = f"seq1/frame_{i:05d}.jpg"
                        images = [os.path.join(folder, SCENE, image_0), os.path.join(folder, SCENE, image_i)]
                        
                        if image_i not in intrinsics_dict:
                            scene_pbar.write(f"Missing intrinsics for {image_i} in scene {SCENE}, skipping.")
                            continue

                        intrincics_i = intrinsics_dict[image_i]

                        # Set intrinsics if in README mode
                        intrinsics = [intrincics_0, intrincics_i]
                        
                        visualization = VISUALIZE and ((i/5) % VISUALIZE_INTERVAL == 0)

                        # Get prediction and write to output file
                        prediction = get_prediction(model_name, device, intrinsics, images, image_i, visualize=visualization)
                        if(prediction):
                            scene_file.write(prediction + '\n')
                        
                        # Update frame progress bar
                        frame_pbar.update(1)

    # Zip the temporary folder and save to the predefined location
    zip_file_path = Path(ZIP_OUTPUT_PATH) / f"submission-{MODEL}-{timestamp}.zip"
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in temp_dir.glob('*'):
            zipf.write(file, arcname=file.name)

    # Delete the temporary folder
    for file in temp_dir.glob('*'):
        file.unlink()
    temp_dir.rmdir()

if __name__ == "__main__":
    main()
