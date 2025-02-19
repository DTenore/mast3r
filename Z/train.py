import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tqdm
import warnings

from transforms3d.quaternions import quat2mat

# Adjust system path
Z_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if Z_PATH not in sys.path:
    sys.path.insert(0, Z_PATH)

# Import project-specific modules
from utils import path_to_dust3r_and_mast3r
from utils.file_io import load_images

from mast3r.model import DinoMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import loss_of_one_batch
from dust3r.utils.device import to_cpu, collate_with_cat
import time

# Suppress some warnings for clarity
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Dataset Definition ---
class MapFreeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = os.path.abspath(root_dir)
        self.frame_pairs = []
        
        # Find all scene directories: e.g. s00000, s00005, etc.
        scene_dirs = sorted(glob.glob(os.path.join(self.root_dir, "s*")))
        scene_dirs = scene_dirs[:20]  # for testing purposes

        for scene_dir in tqdm.tqdm(scene_dirs, desc="Loading scenes", dynamic_ncols=True):
            scene_dir = os.path.abspath(scene_dir)
            intrinsics_path = os.path.join(scene_dir, "intrinsics.txt")
            poses_path = os.path.join(scene_dir, "poses.txt")
            
            if (not os.path.exists(intrinsics_path)) or (not os.path.exists(poses_path)):
                continue
            
            intrinsics = self._load_intrinsics(intrinsics_path)
            poses      = self._load_poses(poses_path)
            
            frame1 = os.path.join(scene_dir, "seq0", "frame_00000.jpg")
            frames_seq1 = sorted(glob.glob(os.path.join(scene_dir, "seq1", "frame_*.jpg")))
            for frame2 in frames_seq1[::5]:
                self.frame_pairs.append({
                    'frame1': frame1,
                    'frame2': frame2,
                    'intrinsics1': intrinsics[frame1],
                    'intrinsics2': intrinsics[frame2],
                    'pose1': poses[frame1],
                    'pose2': poses[frame2]
                })


    def _load_intrinsics(self, intrinsics_path):
        base_dir = os.path.dirname(intrinsics_path)
        intrinsics = {}
        with open(intrinsics_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                rel_frame = parts[0]  # e.g. 'seq0/frame_00000.jpg'
                fx, fy, cx, cy, w, h = map(float, parts[1:])
                abs_frame_path = os.path.abspath(os.path.join(base_dir, rel_frame))
                K = torch.tensor([
                    [fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1 ]
                ], dtype=torch.float32)
                intrinsics[abs_frame_path] = {
                    'K': K,
                    'width':  int(w),
                    'height': int(h)
                }
        return intrinsics

    def _load_poses(self, poses_path):
        base_dir = os.path.dirname(poses_path)
        poses = {}
        with open(poses_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                rel_frame = parts[0]
                qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:])
                abs_frame_path = os.path.abspath(os.path.join(base_dir, rel_frame))
                R = torch.tensor(quat2mat([qw, qx, qy, qz]), dtype=torch.float32)
                t = torch.tensor([tx, ty, tz], dtype=torch.float32)
                poses[abs_frame_path] = {'R': R, 't': t}
        return poses

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        pair = self.frame_pairs[idx]
        data = {
            'img': [pair['frame1'], pair['frame2']],
            'K': [pair['intrinsics1']['K'], pair['intrinsics2']['K']],
            'R': [pair['pose1']['R'], pair['pose2']['R']],
            't': [pair['pose1']['t'], pair['pose2']['t']],
            'true_shape': [
                (pair['intrinsics1']['height'], pair['intrinsics1']['width']),
                (pair['intrinsics2']['height'], pair['intrinsics2']['width'])
            ]
        }
        return data

# --- Loss Functions and Helpers ---
def pose_to_matrix(R, t):
    """Convert a rotation matrix (3x3) and translation vector (3,) into a 4x4 homogeneous matrix."""
    T = torch.eye(4, device=R.device, dtype=R.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def reproject_points(pts3d, T_ref, T_query, K):
    """
    Reproject 3D points from the reference view into the query view.
    Args:
        pts3d (torch.Tensor): (N, 3) 3D points in ref frame.
        T_ref (torch.Tensor): (4,4) ref pose.
        T_query (torch.Tensor): (4,4) query pose.
        K (torch.Tensor): (3,3) intrinsics for query view.
    Returns:
        pts2d (torch.Tensor): (N,2) projected 2D points.
    """
    N = pts3d.shape[0]
    ones = torch.ones((N, 1), device=pts3d.device, dtype=pts3d.dtype)
    pts3d_h = torch.cat([pts3d, ones], dim=1)  # (N, 4)
    T_rel = T_query @ torch.inverse(T_ref)
    pts3d_query_h = (T_rel @ pts3d_h.t()).t()  # (N, 4)
    pts3d_query = pts3d_query_h[:, :3]
    pts2d_h = (K @ pts3d_query.t()).t()  # (N, 3)
    pts2d = pts2d_h[:, :2] / pts2d_h[:, 2:3]
    return pts2d

def reprojection_loss(pts3d_pred, pts2d_obs, T_ref, T_query, K, reduction='mean', use_huber=False, huber_beta=1.0):
    """
    Computes the reprojection loss.
    Args:
        pts3d_pred (torch.Tensor): (N, 3) predicted 3D points.
        pts2d_obs (torch.Tensor): (N, 2) observed 2D keypoints in query view.
        T_ref (torch.Tensor): (4,4) ref pose.
        T_query (torch.Tensor): (4,4) query pose.
        K (torch.Tensor): (3,3) intrinsics of query camera.
        reduction (str): 'mean' | 'sum' | 'none'
        use_huber (bool): If True, uses Huber (Smooth L1) loss
        huber_beta (float): Delta parameter for Huber (Smooth L1). 
                            PyTorch calls this 'beta' in Smooth L1.
    """
    # Reproject 3D points to the query view
    pts2d_proj = reproject_points(pts3d_pred, T_ref, T_query, K)  # shape (N, 2)

    if use_huber:
        # Smooth L1 operates element-wise (e.g. difference in x, difference in y).
        # We'll reduce across (x,y) by summing or (commonly) by default inside Smooth L1
        # but we still need to handle the 'reduction' logic ourselves if we want a 'sum' or 'mean' over all points.
        # You can do it in multiple ways; here is one typical approach:
        
        # Step 1: compute per-point, per-coordinate Smooth L1
        #   shape -> (N, 2) if reduction='none'
        per_point_loss = F.smooth_l1_loss(
            pts2d_proj, 
            pts2d_obs, 
            beta=huber_beta,        # 'beta' in PyTorch docs is the Huber delta threshold
            reduction='none'
        )
        
        # Step 2: sum over x,y so we have a single loss per point (N,)
        per_point_loss = per_point_loss.sum(dim=1)  # shape: (N,)
        
    else:
        # Original L2 norm approach
        diff = pts2d_proj - pts2d_obs   # shape (N, 2)
        per_point_loss = torch.norm(diff, dim=1)    # shape (N,)

    # final reduction
    if reduction == 'mean':
        return per_point_loss.mean()
    elif reduction == 'sum':
        return per_point_loss.sum()
    else:
        return per_point_loss

def scale_intrinsics(K, prev_w, prev_h, master_w, master_h):
    """Scale the intrinsics matrix by a given factor ."""

    assert K.shape == (3, 3), f"Expected (3, 3), but got {K.shape=}"

    scale_w = master_w / prev_w  # sizes of the images in the Mast3r dataset
    scale_h = master_h / prev_h  # sizes of the images in the Mast3r dataset

    K_scaled = K.clone()
    K_scaled[0, 0] *= scale_w
    K_scaled[0, 2] *= scale_w
    K_scaled[1, 1] *= scale_h
    K_scaled[1, 2] *= scale_h

    return K_scaled

# --- Inference Helper without no_grad ---
def inference(pairs, model, device, batch_size=8, verbose=True):
    result = []
    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=False)

    return result

# --- Main Training Script ---
def main():
    # Parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir = "/home/dario/DATASETS/map-free-reloc/data/mapfree/train"  # adjust to your dataset root
    num_epochs = 10
    lr = 1e-3
    batch_size = 1        # If you want to accumulate 10 samples, often you set batch_size=1
    accumulation_steps = 10  # Number of iterations to accumulate before stepping
    top_k = 0.75  # parameter passed to the DinoMASt3R constructor

    # Create the dataset and dataloader
    dataset = MapFreeDataset(root_dir)
    # WIP: collate used because of weird tuple stuff
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x[0])

    # --- Load Dino model and instantiate DinoMASt3R ---
    print("Loading DINO model ...")

    # Load dinov2 model via torch.hub
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
    dino_model.eval()  # evaluation mode
    dino_model.to(device)
    
    print("Instantiating DinoMASt3R model ...")
    
    # Instantiate the model using the pretrained weights.
    mast3r_model = DinoMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        dino_model=dino_model,
        top_k=top_k
    ).to(device)
    


    # Freeze all parameters except alpha.
    for param in mast3r_model.parameters():
        param.requires_grad = False
    mast3r_model.alpha.requires_grad = True
    mast3r_model.alpha = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float32))
    
    for name, param in mast3r_model.downstream_head1.head_local_features.named_parameters():
        param.requires_grad = True
    for name, param in mast3r_model.downstream_head2.head_local_features.named_parameters():
        param.requires_grad = True
    

    # Setup optimizer (only optimizing alpha)
    params_to_optimize = list(mast3r_model.downstream_head1.head_local_features.parameters()) \
                       + list(mast3r_model.downstream_head2.head_local_features.parameters()) \
                       + [mast3r_model.alpha]
    optimizer = optim.Adam(params_to_optimize, lr=lr)
    
    mast3r_model.train()

    # ---- Gradient Accumulation add-ons ----
    optimizer.zero_grad()
    running_loss = 0.0
    accumulation_count = 0  # Counts how many samples have been accumulated

    # Training loop with tqdm progress bar
    pbar = tqdm.tqdm(total=len(dataloader) * num_epochs, desc="Training", dynamic_ncols=True)
    for epoch in range(num_epochs):
        for sample in dataloader:
            # sample is a dict containing:
            #   'img': [img1, img2] as file paths
            #   'K': list of two dicts; each dict has key 'K' (tensor 3x3)
            #   'R': list of two tensors (3x3)
            #   't': list of two tensors (3,)
            #   'true_shape': list of tuples (H, W)
            imgs = sample['img']  

            images = [tuple(load_images(imgs, size=512, verbose=False))]
            output = inference(images, mast3r_model, device, batch_size=1, verbose=False)

            # Expected output: a dict with keys 'view1', 'pred1', 'view2', 'pred2'
            view1 = output['view1']
            pred1 = output['pred1']
            view2 = output['view2']
            pred2 = output['pred2']
            
            # Get descriptors from both views (do not detach in training)
            desc1 = pred1['desc'].squeeze(0)  # e.g. (H, W, C)
            desc2 = pred2['desc'].squeeze(0)
            
            # Compute 2D-2D matches using the fast NN routine.
            matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                           device=device, dist='dot', block_size=2**13)
            # Get image shapes from views (assumed to be stored in view['true_shape'])
            H0, W0 = (int(x) for x in view1['true_shape'][0])
            H1, W1 = (int(x) for x in view2['true_shape'][0])
            valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0 - 3) & \
                                (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0 - 3)
            valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < W1 - 3) & \
                                (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < H1 - 3)
            valid_matches = valid_matches_im0 & valid_matches_im1
            matches_im0 = matches_im0[valid_matches]
            matches_im1 = matches_im1[valid_matches]
            
            if matches_im0.shape[0] < 5 or matches_im1.shape[0] < 5:
                pbar.update(1)
                continue

            # Get predicted 3D points from view1 (assumed shape (1,H,W,3)); remove batch dim.
            pts3d_im0 = pred1['pts3d'].squeeze(0)  # shape (H, W, 3)
            pts3d_pred = pts3d_im0[matches_im0[:, 1], matches_im0[:, 0]].to(device)  # shape (N, 3)

            # Use the matched keypoints in view2 as the observed 2D points.
            points2d_obs = torch.from_numpy(matches_im1).to(device)  # shape (N, 2)
            
            # Get ground-truth poses from the sample.
            R_ref = sample['R'][0].squeeze(0).to(device)  # shape (3,3)
            t_ref = sample['t'][0].squeeze(0).to(device)    # shape (3,)
            R_query = sample['R'][1].squeeze(0).to(device)
            t_query = sample['t'][1].squeeze(0).to(device)
            T_ref = pose_to_matrix(R_ref, t_ref).to(device)  # shape (4,4)
            T_query = pose_to_matrix(R_query, t_query).to(device)
            
            # Get query camera intrinsics.
            K_query = sample['K'][1].to(device)
            K_rescaled = scale_intrinsics(K_query, sample['true_shape'][0][1], sample['true_shape'][0][0], W0, H0)


            # Compute the reprojection loss.
            loss = reprojection_loss(pts3d_pred, points2d_obs, T_ref, T_query, K_rescaled, use_huber=True, huber_beta=1.0)
            
            # ------------------------
            # Gradient Accumulation
            # ------------------------
            # We divide the loss by accumulation_steps so the gradients
            # end up being the average over these steps (like a bigger batch).
            (loss / accumulation_steps).backward()

            running_loss += loss.item()
            accumulation_count += 1

            # Only update the model after 'accumulation_steps' samples
            if accumulation_count % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                # Compute average loss for display
                avg_loss = running_loss / accumulation_steps
                running_loss = 0.0

                # Update progress bar description
                pbar.set_description(
                    f"Epoch {epoch+1}/{num_epochs} "
                    f"Loss: {avg_loss:.4f} "
                    f"| alpha: {mast3r_model.alpha.item():.4f}"
                )

            pbar.update(1)
    
    pbar.close()
    # Optionally, save the finetuned model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt_name = f"/home/dario/_MINE/mast3r/Z/checkpoints/finetuned_dinomast3r_{timestamp}.pth"
    torch.save(mast3r_model.state_dict(), ckpt_name)
    print("Training finished and model saved at", ckpt_name)

if __name__ == "__main__":
    main()
