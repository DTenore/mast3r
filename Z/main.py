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
from utils.file_io import write_prediction
from utils.utils import load_image_pair

sys.stderr = open(os.devnull, 'w')

# Parameters
device = 'cuda'
model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
output_file = os.path.join(os.path.dirname(__file__), "pose_s00460.txt")
folder = "/home/dario/DATASETS/map-free-reloc/data/mapfree/val/"
image_0 = "seq0/frame_00000.jpg"

MODE = "VISLOC"
SCENES = ["s00460", "s00461"]
# SCENES = [f"s{i:05d}" for i in range(460, 525)]
ZIP_OUTPUT_PATH = "/home/dario/_MINE/mast3r"

# Conditional import based on mode
def get_prediction_function():
    if MODE == "DEMO":
        from models.model_demo import get_prediction
    elif MODE == "README":
        from models.model_readme import get_prediction
    elif MODE == "VISLOC":
        from models.model_visloc import get_prediction
    else:
        raise ValueError(f"Unknown MODE: {MODE}")
    return get_prediction

def main():
    get_prediction = get_prediction_function()

    # Create temporary folder for submission
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_dir = Path(__file__).parent / f"submission-{timestamp}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # **Top-level progress bar for all scenes**
    with tqdm.tqdm(SCENES, desc="Processing scenes", file=sys.stdout) as scene_pbar:
        for SCENE in scene_pbar:
            scene_file_path = temp_dir / f"{SCENE}.txt"
            scene_pbar.set_description(f"Processing scene {SCENE}")

            # Count number of frames in the scene folder
            scene_folder = Path(folder) / SCENE / "seq1"
            num_frames = len(list(scene_folder.glob("frame_*.jpg")))

            # Open file for the current scene
            with scene_file_path.open("w") as scene_file:
                # **Single progress bar for frames within the current scene**
                with tqdm.tqdm(total=num_frames // 5, desc=f"Processing frames for {SCENE}", leave=False, file=sys.stdout) as frame_pbar:
                    # Process images and generate predictions
                    for i in range(0, num_frames, 5):
                        image_i = f"seq1/frame_{i:05d}.jpg"
                        images = [os.path.join(folder, SCENE, image_0), os.path.join(folder, SCENE, image_i)]
                        
                        # TODO: Load intrinsics from txt files (MAYBE?)
                        # Set intrinsics if in README mode
                        intrinsics = None
                        if MODE == "README":
                            intrinsics = [[590.3821, 0, 269.6031], 
                                          [0, 590.3821, 270.2328], 
                                          [0, 0, 1]]
                        
                        # Get prediction and write to output file
                        prediction = get_prediction(model_name, device, intrinsics, images, image_i)
                        scene_file.write(prediction + '\n')
                        
                        # Update frame progress bar
                        frame_pbar.update(1)

    # Zip the temporary folder and save to the predefined location
    zip_file_path = Path(ZIP_OUTPUT_PATH) / f"submission-{timestamp}.zip"
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in temp_dir.glob('*'):
            zipf.write(file, arcname=file.name)

    # Delete the temporary folder
    for file in temp_dir.glob('*'):
        file.unlink()
    temp_dir.rmdir()

if __name__ == "__main__":
    main()
