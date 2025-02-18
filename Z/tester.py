import os
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
from utils.file_io import write_prediction, load_intrinsics, load_images

from mast3r.model import AsymmetricMASt3R, DinoMASt3R

from dust3r.inference import inference


def test_dino_mast3r():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
    dino_model.eval()  # Set the model to evaluation mode
    dino_model.to(device)  # Move the model to the device
    
    # Create DinoMASt3R instance with the pretrained model's parameters
    model = DinoMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        dino_model=dino_model
    ).to(device)

    # Load test images 
    test_images = load_images([
        '/home/dario/DATASETS/map-free-reloc/data/mapfree/val/s00465/seq0/frame_00000.jpg',
        '/home/dario/DATASETS/map-free-reloc/data/mapfree/val/s00465/seq1/frame_00000.jpg',
    ], size=512)

    # Run inference
    output = inference([tuple(test_images)], model, device, batch_size=1)

    #print(output.keys())

    # Access predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    # Access DINO features
    #dino_feat1, dino_feat2 = output['dino_feat1'], output['dino_feat2']

if __name__ == '__main__':
    test_dino_mast3r()