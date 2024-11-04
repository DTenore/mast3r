import os
import sys
import tempfile
import tqdm

# Adjust system path
Z_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if Z_PATH not in sys.path:
    sys.path.insert(0, Z_PATH)

# Import project-specific modules
from utils import path_to_dust3r_and_mast3r
from utils.file_io import write_prediction
from utils.utils import load_image_pair

# Parameters
device = 'cuda'
model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
output_file = os.path.join(os.path.dirname(__file__), "pose_s00460.txt")
folder = "/home/dario/DATASETS/map-free-reloc/data/mapfree/val/s00460/"
image_0 = "seq0/frame_00000.jpg"
MODE = "VISLOC"

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

    # Initialize output file
    with open(output_file, "w") as f:
        pass

    # Process images and generate predictions
    for i in tqdm.tqdm(range(5, 10, 5)):
        image_i = f"seq1/frame_{i:05d}.jpg"
        images = [os.path.join(folder, image_0), os.path.join(folder, image_i)]
        
        # Set intrinsics if in README mode
        intrinsics = None
        if MODE == "README":
            intrinsics = [[590.3821, 0, 269.6031], 
                          [0, 590.3821, 270.2328], 
                          [0, 0, 1]]
        
        # Get prediction and write to output file
        prediction = get_prediction(model_name, device, intrinsics, images, image_i)
        write_prediction(output_file, prediction)

if __name__ == "__main__":
    main()
