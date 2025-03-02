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
import time
import shutil

from transforms3d.quaternions import quat2mat

# TensorBoard import
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

# Adjust system path
Z_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if Z_PATH not in sys.path:
    sys.path.insert(0, Z_PATH)

# Import project-specific modules
from utils import path_to_dust3r_and_mast3r
from utils.file_io import load_images

from mast3r.model3 import DINOMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import loss_of_one_batch
from dust3r.utils.device import to_cpu, collate_with_cat

# Suppress some warnings for clarity
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Dataset Definition ---
class MapFreeDataset(Dataset):
    def __init__(self, root_dir, num_scenes=460, frame_stride=20):
        self.root_dir = os.path.abspath(root_dir)
        self.frame_pairs = []
        
        # Find all scene directories: e.g. s00000, s00005, etc.
        scene_dirs = sorted(glob.glob(os.path.join(self.root_dir, "s*")))
        scene_dirs = scene_dirs[:num_scenes]  # for testing purposes

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
            for frame2 in frames_seq1[::frame_stride]:
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
    N = pts3d.shape[0]
    ones = torch.ones((N, 1), device=pts3d.device, dtype=pts3d.dtype)
    pts3d_h = torch.cat([pts3d, ones], dim=1)
    T_rel = T_query @ torch.inverse(T_ref)
    pts3d_query_h = (T_rel @ pts3d_h.t()).t()
    pts3d_query = pts3d_query_h[:, :3]
    pts2d_h = (K @ pts3d_query.t()).t()
    pts2d = pts2d_h[:, :2] / pts2d_h[:, 2:3]
    return pts2d

def reprojection_loss(pts3d_pred, pts2d_obs, T_ref, T_query, K, reduction='mean', huber_beta=40.0):
    pts2d_proj = reproject_points(pts3d_pred, T_ref, T_query, K)

    per_point_loss = F.smooth_l1_loss(
        pts2d_proj, 
        pts2d_obs, 
        beta=huber_beta,
        reduction='none'
    )
    per_point_loss = per_point_loss.sum(dim=1)
    
    if reduction == 'mean':
        return per_point_loss.mean()
    elif reduction == 'sum':
        return per_point_loss.sum()
    else:
        return per_point_loss

def scale_intrinsics(K, prev_w, prev_h, master_w, master_h):
    assert K.shape == (3, 3), f"Expected (3, 3), but got {K.shape=}"
    scale_w = master_w / prev_w
    scale_h = master_h / prev_h
    K_scaled = K.clone()
    K_scaled[0, 0] *= scale_w
    K_scaled[0, 2] *= scale_w
    K_scaled[1, 1] *= scale_h
    K_scaled[1, 2] *= scale_h
    return K_scaled

# --- Inference Helper without no_grad ---
def inference_no_gradless(pairs, model, device, batch_size=8, verbose=True):
    result = []
    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))
    result = collate_with_cat(result, lists=False)
    return result

# --- Main Training Script ---
def train():
    # Create TensorBoard log directory and clear if it exists
    log_dir = "/home/dario/_MINE/mast3r/Z/_runs"
    writer = SummaryWriter(log_dir=log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_root_dir = "/home/dario/DATASETS/map-free-reloc/data/mapfree/train" 
    val_root_dir = "/home/dario/DATASETS/map-free-reloc/data/mapfree/val"


    lr = 1e-4
    weight_decay_value = 1e-4 
    max_grad_norm = 2.0 # Or maybe 1 if no convergence, or try else 5.0

    num_epochs = 6
    train_num_scenes = 100  # Number of scenes to use for training
    val_num_scenes = 10  # Number of scenes to use for validation
    frame_stride = 20  # Stride between frames in a scene

    batch_size = 1        
    accumulation_steps = 16  

    huber_beta = 30.0
    topK = 1
    optimize_local_features = True  
    optimize_postprocess = False

    g = torch.Generator()
    g.manual_seed(42)

    train_dataset = MapFreeDataset(train_root_dir, num_scenes=train_num_scenes, frame_stride=frame_stride)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x[0], generator=g)

    val_dataset = MapFreeDataset(val_root_dir, num_scenes=val_num_scenes, frame_stride=frame_stride)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x[0], generator=g)

    mast3r_model = DINOMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", postprocess=optimize_postprocess).to(device)

    CKPT = "20250302_0059_DINOMASt3R_ND_P_e6_sc10_ac16_hb30"
    state_dict = torch.load("/home/dario/_MINE/mast3r/Z/checkpoints/"+CKPT + ".pth", map_location=device)
    mast3r_model.load_state_dict(state_dict, strict=False)

    #WIP: maybe train once with topK
    #mast3r_model.topK = 0.5???


    for name, param in mast3r_model.named_parameters():
        param.requires_grad = False

    params_to_optimize = []

    if optimize_local_features:    
        for name, param in mast3r_model.named_parameters():
            if "downstream_head" in name:
                param.requires_grad = True
                params_to_optimize.append(param)
        
    if optimize_postprocess:
        for name, param in mast3r_model.named_parameters():
            if "ncnet" in name:
                param.requires_grad = True
                params_to_optimize.append(param)


    optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=weight_decay_value)
    

    #optimizer.zero_grad()
    running_loss = 0.0
    accumulation_count = 0
    global_step = 0
    best_val_loss = float('inf')

    timestamp = time.strftime("%Y%m%d_%H%M")
    desc_opt_str = "D" if optimize_local_features else "ND"
    post_opt_str = "P" if optimize_postprocess else "NP"
    huber_beta_str = f"{huber_beta:.0f}"

    ckpt_name = (
        f"{timestamp}_"
        "DINOMASt3R_"
        f"{desc_opt_str}_"
        f"{post_opt_str}_"
        f"e{num_epochs}_sc{train_num_scenes}_"
        f"ac{accumulation_steps}_"
        f"hb{huber_beta_str}.pth"
    )

    ckpt_path = "/home/dario/_MINE/mast3r/Z/checkpoints"
    ckpt_out = os.path.join(ckpt_path, ckpt_name)

    # -------------------------
    # Evaluate once before training to get baseline
    initial_val_loss = evaluate(val_dataloader, mast3r_model, device)
    writer.add_scalar("Validation Loss", initial_val_loss, 0)
    print(f"Initial Validation Loss: {initial_val_loss:.2f}")
    best_val_loss = min(best_val_loss, initial_val_loss)
    # -------------------------

    # Create a tqdm progress bar for all epochs and iterations
    pbar = tqdm.tqdm(total=len(train_dataloader) * num_epochs, desc="Training", dynamic_ncols=True)

    for epoch in range(num_epochs):
        mast3r_model.train()

        # Divide lr for warmup epochs
        if epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / 100
        if epoch == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / 10

        # Initialize epoch-level loss accumulators
        epoch_loss_total = 0.0
        epoch_sample_count = 0
        
        for sample in train_dataloader:
            imgs = sample['img']  
            images = [tuple(load_images(imgs, size=512, verbose=False))]
            output = inference_no_gradless(images, mast3r_model, device, batch_size=1, verbose=False)
            view1 = output['view1']
            pred1 = output['pred1']
            view2 = output['view2']
            pred2 = output['pred2']
            
            desc1 = pred1['desc'].squeeze(0)
            desc2 = pred2['desc'].squeeze(0)
            matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                           device=device, dist='dot', block_size=2**13)
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

            pts3d_im0 = pred1['pts3d'].squeeze(0)
            pts3d_pred = pts3d_im0[matches_im0[:, 1], matches_im0[:, 0]].to(device)
            points2d_obs = torch.from_numpy(matches_im1).to(device)
            
            R_ref = sample['R'][0].squeeze(0).to(device)
            t_ref = sample['t'][0].squeeze(0).to(device)
            R_query = sample['R'][1].squeeze(0).to(device)
            t_query = sample['t'][1].squeeze(0).to(device)
            T_ref = pose_to_matrix(R_ref, t_ref).to(device)
            T_query = pose_to_matrix(R_query, t_query).to(device)
            
            K_query = sample['K'][1].to(device)
            K_rescaled = scale_intrinsics(K_query, sample['true_shape'][0][1], sample['true_shape'][0][0], W0, H0)

            loss = reprojection_loss(pts3d_pred, points2d_obs, T_ref, T_query, K_rescaled, huber_beta=huber_beta)
            
            # Accumulate epoch loss for epoch-average computation
            epoch_loss_total += loss.item()
            epoch_sample_count += 1

            (loss / accumulation_steps).backward()
            running_loss += loss.item()
            accumulation_count += 1

            if accumulation_count % accumulation_steps == 0:
                clip_grad_norm_(params_to_optimize, max_norm=max_grad_norm)


                optimizer.step()
                optimizer.zero_grad()
                avg_loss = running_loss / accumulation_steps
                running_loss = 0.0

                # Compute epoch average loss so far (as an int)
                epoch_avg_loss_so_far = epoch_loss_total / epoch_sample_count
                # Log instantaneous loss to TensorBoard
                writer.add_scalar("Loss", avg_loss, global_step)
                global_step += 1

                # Update tqdm description with both the accumulation loss and the epoch average loss (as ints)
                pbar.set_description(
                    f"Epoch: {epoch+1}/{num_epochs} | Avg: {int(epoch_avg_loss_so_far)} | Acc: {int(avg_loss)} Training"
                )

            pbar.update(1)
        
        # At the end of each epoch, log the epoch average loss to TensorBoard
        epoch_avg_loss = epoch_loss_total / epoch_sample_count if epoch_sample_count > 0 else 0.0
        writer.add_scalar("Epoch Train Loss", epoch_avg_loss, epoch+1)

        # Validate
        val_loss = evaluate(val_dataloader, mast3r_model, device)
        writer.add_scalar("Validation Loss", val_loss, epoch + 1)

        # Save checkpoint if best so far:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(mast3r_model.state_dict(), ckpt_out)

    pbar.close()
    writer.close()
    print("Training finished and best model saved at", ckpt_out)

def evaluate(dataloader, model, device, huber_beta=30.0):
    model.eval()
    total_loss = 0.0
    sample_count = 0
    
    with torch.no_grad():
        for sample in tqdm.tqdm(dataloader, desc="Evaluating", dynamic_ncols=True):
            imgs = sample['img']
            images = [tuple(load_images(imgs, size=512, verbose=False))]
            output = inference_no_gradless(images, model, device, batch_size=1, verbose=False)
            
            view1 = output['view1']
            pred1 = output['pred1']
            view2 = output['view2']
            pred2 = output['pred2']
            
            desc1 = pred1['desc'].squeeze(0)
            desc2 = pred2['desc'].squeeze(0)
            matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                           device=device, dist='dot', block_size=2**13)
            
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
                continue
            
            pts3d_im0 = pred1['pts3d'].squeeze(0)
            pts3d_pred = pts3d_im0[matches_im0[:, 1], matches_im0[:, 0]].to(device)
            points2d_obs = torch.from_numpy(matches_im1).to(device)
            
            R_ref = sample['R'][0].squeeze(0).to(device)
            t_ref = sample['t'][0].squeeze(0).to(device)
            R_query = sample['R'][1].squeeze(0).to(device)
            t_query = sample['t'][1].squeeze(0).to(device)
            T_ref = pose_to_matrix(R_ref, t_ref).to(device)
            T_query = pose_to_matrix(R_query, t_query).to(device)
            
            K_query = sample['K'][1].to(device)
            K_rescaled = scale_intrinsics(K_query, sample['true_shape'][0][1], sample['true_shape'][0][0], W0, H0)

            loss = reprojection_loss(pts3d_pred, points2d_obs, T_ref, T_query, K_rescaled, huber_beta=huber_beta)
            
            total_loss += loss.item()
            sample_count += 1
    
    return total_loss / sample_count if sample_count > 0 else float('inf')



if __name__ == "__main__":
    train()
