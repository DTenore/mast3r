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

# Suppress some warnings for clarity
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Dataset Definition ---
class MapFreeDataset(Dataset):
    def __init__(self, root_dir, num_scenes=460):
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
            for frame2 in frames_seq1[::20]: # FIXME: change back to [::5] 
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
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir = "/home/dario/DATASETS/map-free-reloc/data/mapfree/train"  # adjust to your dataset root


    lr = 1e-4
    num_epochs = 4
    num_scenes = 460  # Number of scenes to use for training

    batch_size = 1        
    accumulation_steps = 16  

    top_k = 0.8  
    huber_beta = 90.0
    optimize_local_features = True  
    optimize_postprocess = False

    dataset = MapFreeDataset(root_dir, num_scenes=num_scenes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x[0])

    # Loading DINO model ...
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
    dino_model.eval()
    dino_model.to(device)
    
    # Instantiating DinoMASt3R model ...
    mast3r_model = DinoMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        dino_model=dino_model,
        top_k=top_k
    ).to(device)
    
    for name, param in mast3r_model.named_parameters():
        param.requires_grad = False

    params_to_optimize = []

    if optimize_local_features:    
        for name, param in mast3r_model.downstream_head1.head_local_features.named_parameters():
            param.requires_grad = True
            params_to_optimize.append(param)
        for name, param in mast3r_model.downstream_head2.head_local_features.named_parameters():
            param.requires_grad = True
            params_to_optimize.append(param)
        for name, param in mast3r_model.named_parameters():
            if "downstream_head1.dpt.act_postprocess" in name or "downstream_head2.dpt.act_postprocess" in name:
                param.requires_grad = True
                params_to_optimize.append(param)

    if optimize_postprocess:
        for name, param in mast3r_model.named_parameters():
            if "sim_postprocess" in name:
                param.requires_grad = True
                params_to_optimize.append(param)

    optimizer = optim.Adam(params_to_optimize, lr=lr)
    mast3r_model.train()

    optimizer.zero_grad()
    running_loss = 0.0
    accumulation_count = 0
    global_step = 0

    # Create a tqdm progress bar for all epochs and iterations
    pbar = tqdm.tqdm(total=len(dataloader) * num_epochs, desc="Training", dynamic_ncols=True)

    for epoch in range(num_epochs):
        # Add 1 warmup epoch
        if epoch < 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / 100
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Initialize epoch-level loss accumulators
        epoch_loss_total = 0.0
        epoch_sample_count = 0
        
        for sample in dataloader:
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
        writer.add_scalar("Epoch Average Loss", epoch_avg_loss, epoch+1)
    
    pbar.close()
    writer.close()

    timestamp = time.strftime("%Y%m%d_%H%M")
    desc_opt_str = "D" if optimize_local_features else "ND"
    postprocess_str = "P" if optimize_postprocess else "NP"
    huber_beta_str = f"{huber_beta:.0f}"

    ckpt_name = (
        f"{timestamp}_"
        f"{desc_opt_str}_"
        f"{postprocess_str}_"
        f"e{num_epochs}_sc{num_scenes}_topK{int(top_k * 100)}_"
        f"ac{accumulation_steps}_"
        f"hb{huber_beta_str}.pth"
    )

    ckpt_path = "/home/dario/_MINE/mast3r/Z/checkpoints"
    ckpt_out = os.path.join(ckpt_path, ckpt_name)
    torch.save(mast3r_model.state_dict(), ckpt_out)
    print("Training finished and model saved at", ckpt_out)

if __name__ == "__main__":
    train()
