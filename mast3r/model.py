# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import os

from mast3r.catmlp_dpt_head import mast3r_head_factory

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape  # noqa

# ========================================================


from torchvision import transforms
from PIL import Image, ImageOps
from functools import partial
import torch.nn as nn 
from copy import deepcopy


import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa
from models.blocks import DinoDecoderBlock, DecoderBlock


inf = float('inf')


def load_model(model_path, device, verbose=False):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else: 
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

# --------------------------------------------------------
# MY MODEL CLASS WITH DINO
# --------------------------------------------------------

class DinoMASt3R(AsymmetricMASt3R):
    def __init__(self, dino_model, **kwargs):
        super().__init__(**kwargs)

        self.dino_model = dino_model

        if True:
            self.dec_blocks = nn.ModuleList([
                # NOTE: dec_depth = 12, which is equal to number of heads??? i.e., num_heads=12???
                DinoDecoderBlock(self.dec_embed_dim, self.dec_depth, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_mem=True, rope=self.rope)
                for i in range(self.dec_depth)])

            self.dec_blocks2 = deepcopy(self.dec_blocks)


    def _get_dino_features(self, view1, view2):
        """Extract DINO features and upsample to original resolution."""
        H1, W1 = view1['true_shape'][0]
        H2, W2 = view2['true_shape'][0]

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        results = []
        for img, h, w in [(view1['img'], H1, W1), (view2['img'], H2, W2)]:
            # Ensure dimensions are multiples of patch size
            new_width = (w // 14) * 14  
            new_height = (h // 14) * 14

            # Convert tensor to PIL for resize
            img_pil = transforms.ToPILImage()(img.squeeze(0))
            img_resized = img_pil.resize((new_width, new_height), Image.LANCZOS)
        

            # Preprocess and add batch dimension
            img_tensor = preprocess(img_resized).unsqueeze(0).to("cuda")


            # Extract features
            with torch.no_grad():
                outputs = self.dino_model.forward_features(img_tensor)


            # Get patch features (excluding CLS token)
            patch_features = outputs['x_norm_patchtokens']  # Shape: [1, N_patches, D]
            # Normalize patch features
            patch_features = torch.nn.functional.normalize(patch_features, p=2, dim=-1) # Now each feature has unit L2 norm

            # Reshape to grid
            num_patches_height = new_height // 14
            num_patches_width = new_width // 14 
            patch_grid = patch_features.reshape(1, num_patches_height, num_patches_width, -1)[0]

            # Reshape to match original image dimensions
            # Use nearest neighbor to maintain sharp patch boundaries
            feature_dim = patch_grid.shape[-1]
            upsampled_features = F.interpolate(
                patch_grid.permute(2,0,1).unsqueeze(0),  # [1, D, H, W]
                size=(h, w),
                mode='nearest'
            )

            upsampled_features = upsampled_features.squeeze(0).permute(1,2,0)  # [H, W, D]

            results.append(upsampled_features)

        return tuple(results)

    def _reshape_dino_features(self, dino_feats):
        #Reshapes dino features to match the dimensions of the attention layer
        # Permute to (D, H, W) so that we can apply pooling across H and W
        dino_feats = dino_feats.permute(2, 0, 1)  # (384, 512, 384)

        # Apply 2D average pooling (reducing H and W by a factor of 16)
        pooled = F.avg_pool2d(dino_feats, kernel_size=16, stride=16)  # (384, 32, 24)

        # Permute back to (H//16, W//16, D)
        pooled = pooled.permute(1, 2, 0)  # (32, 24, 384)

        return pooled

    def _create_adjacency_graphs(self, features1, features2, top_k=0.5):
        

        # Flatten spatial dimensions (32, 24, 384) → (768, 384)
        features1_flat = features1.reshape(features1.shape[0] * features1.shape[1], features1.shape[2])  # (768, 384)
        features2_flat = features2.reshape(features2.shape[0] * features2.shape[1], features2.shape[2])  # (768, 384)

        # Create distance matrix
        similarity_matrix = torch.matmul(features1_flat, features2_flat.T)  # (768, 768)

        # Get top-k by mutliplying values per row times percentage
        k1 = int(similarity_matrix.shape[0] * top_k)
        k2 = int(similarity_matrix.shape[1] * top_k)

        # WIP: maybe in the future will have to change to boolens for the mask

        # Select top-K neighbors for 1 → 2
        topk_values_1, topk_indices_1 = torch.topk(similarity_matrix, k1, dim=1)
        adj_1_to_2 = torch.full_like(similarity_matrix, -float("inf"))
        adj_1_to_2.scatter_(1, topk_indices_1, topk_values_1)

        # Create boolean mask for 1 → 2
        mask_1_to_2 = (adj_1_to_2 != -float("inf"))

        # Select top-K neighbors for 2 → 1
        topk_values_2, topk_indices_2 = torch.topk(similarity_matrix.T, k2, dim=1)  # Transpose for 2 → 1
        adj_2_to_1 = torch.full_like(similarity_matrix.T, -float("inf"))
        adj_2_to_1.scatter_(1, topk_indices_2, topk_values_2)
        
        # Create boolean mask for 2 → 1
        mask_2_to_1 = (adj_2_to_1 != -float("inf"))

        return adj_1_to_2, mask_1_to_2, adj_2_to_1, mask_2_to_1

    def _plot_mask(self, mask1, mask2, x, y):
        """
        Plots the masks of pixel at coordinates x, y side by side.
        
        Args:
            mask1: torch tensor of shape (768, 768) on cuda device
            mask2: torch tensor of shape (768, 768) on cuda device
            x: x-coordinate of the pixel
            y: y-coordinate of the pixel
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Move tensors to CPU and convert to numpy
        mask1 = mask1.cpu().numpy()
        mask2 = mask2.cpu().numpy()
        
        # Extract the rows corresponding to the pixel
        pixel_idx = x * 32 + y
        mask_row1 = mask1[pixel_idx]
        mask_row2 = mask2[pixel_idx]
        
        # Reshape both to 32x24
        mask_vis1 = mask_row1.reshape(32, 24)
        mask_vis2 = mask_row2.reshape(32, 24)
        
        # Scale up the visualizations
        mask_vis1 = np.repeat(np.repeat(mask_vis1, 14, axis=0), 14, axis=1)
        mask_vis2 = np.repeat(np.repeat(mask_vis2, 14, axis=1), 14, axis=0)
        
        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot with a purple-red-orange colormap
        cmap = plt.get_cmap('plasma')
        
        # Plot both masks
        im1 = ax1.imshow(mask_vis1, cmap=cmap)
        im2 = ax2.imshow(mask_vis2, cmap=cmap)
        
        # Add colorbars
        plt.colorbar(im1, ax=ax1, label='Similarity Score 1')
        plt.colorbar(im2, ax=ax2, label='Similarity Score 2')
        
        # Add titles and labels
        ax1.set_title(f'Mask 1 at Pixel ({x}, {y})')
        ax2.set_title(f'Mask 2 at Pixel ({x}, {y})')
        ax1.set_xlabel('Width (scaled)')
        ax1.set_ylabel('Height (scaled)')
        ax2.set_xlabel('Width (scaled)')
        ax2.set_ylabel('Height (scaled)')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Show the plot
        plt.show()


    # WIP: Accept masks and pass them to the blocks
    def _decoder(self, f1, pos1, mask_1_to_2, f2, pos2, mask_2_to_1):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2, mask_1_to_2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1, mask_2_to_1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        # Encodes the two images using ViT transformer and returns their shapes, patch features and geom infos about patch locations
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # Extract DINO features and upsample to original resolution
        dino_feat1, dino_feat2 = self._get_dino_features(view1, view2)

        dino_feat1 = self._reshape_dino_features(dino_feat1)
        dino_feat2 = self._reshape_dino_features(dino_feat2)

        # Get adjacency distances and masks (keep top 50% of neighbors)
        dist_1_to_2, mask_1_to_2, dist_2_to_1, mask_2_to_1 = self._create_adjacency_graphs(dino_feat1, dino_feat2, top_k=0.5)
        
        #self._plot_mask(dist_1_to_2, dist_2_to_1, 0, 0)

        # WIP: pass distances to decoder
        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, mask_1_to_2, feat2, pos2, mask_2_to_1)
        #dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.amp.autocast('cuda',enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
