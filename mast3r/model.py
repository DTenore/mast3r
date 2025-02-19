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
from models.blocks import Attention, DropPath, Mlp

import torch.nn as nn 


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
    def __init__(self, dino_model=None, top_k=0.75, 
                **kwargs):
        super().__init__(**kwargs)

        if dino_model is not None:
            self.dino_model = dino_model
        else:
            dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
            dino_model.eval()  # Set the model to evaluation mode
            self.dino_model = dino_model

        self.dec_blocks = nn.ModuleList([
            # NOTE: dec_depth = 12, which is equal to number of heads??? i.e., num_heads=12???
            DinoDecoderBlock(self.dec_embed_dim, self.dec_depth, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_mem=True, rope=self.rope)
            for i in range(self.dec_depth)])

        self.dec_blocks2 = deepcopy(self.dec_blocks)

        self.top_k = top_k

        # WIP:
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def _get_dino_features(self, view1, view2):
        """Extract DINO features and upsample to original resolution."""
        H1, W1 = view1['true_shape'][0]
        H2, W2 = view2['true_shape'][0]

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        results = []
        for img, h, w in [(view1['img'][0], H1, W1), (view2['img'][0], H2, W2)]:
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

    def _create_adjacency_graphs(self, features1, features2, top_k):
        # Flatten spatial dimensions (32, 24, 384) → (768, 384)
        features1_flat = features1.reshape(features1.shape[0] * features1.shape[1], features1.shape[2])  # (768, 384)
        features2_flat = features2.reshape(features2.shape[0] * features2.shape[1], features2.shape[2])  # (768, 384)

        # Create distance matrix
        similarity_matrix = torch.matmul(features1_flat, features2_flat.T)  # (768, 768)

        # Get top-k by mutliplying values per row times percentage
        k1 = int(similarity_matrix.shape[0] * top_k)
        k2 = int(similarity_matrix.shape[1] * top_k)

        # Select top-K neighbors for 1 → 2
        topk_values_1, topk_indices_1 = torch.topk(similarity_matrix, k1, dim=1)
        adj_1_to_2 = torch.full_like(similarity_matrix, -float("inf"))
        adj_1_to_2.scatter_(1, topk_indices_1, topk_values_1)

        # Select top-K neighbors for 2 → 1
        topk_values_2, topk_indices_2 = torch.topk(similarity_matrix.T, k2, dim=1)  # Transpose for 2 → 1
        adj_2_to_1 = torch.full_like(similarity_matrix.T, -float("inf"))
        adj_2_to_1.scatter_(1, topk_indices_2, topk_values_2)

        # Create boolean mask for 1 → 2
        mask_1_to_2 = (adj_1_to_2 != -float("inf"))
        # Create boolean mask for 2 → 1
        mask_2_to_1 = (adj_2_to_1 != -float("inf"))

        scaled_similarity_matrix = similarity_matrix * self.alpha

        return adj_1_to_2, mask_1_to_2, adj_2_to_1, mask_2_to_1, scaled_similarity_matrix

    def _plot_mask(self, mask1, mask2, x, y, orig_img1, orig_img2):
        """
        Plots four panels arranged as follows:
        Row 1: Left - Processed Image 1 (denormalized to [0,1] and resized),
                Right - Mask from Image 1 to Image 2 with the selected patch highlighted.
        Row 2: Left - Processed Image 2 (denormalized and resized),
                Right - Mask from Image 2 to Image 1 with the selected patch highlighted.

        The images are resized to the same dimensions as the mask visualizations 
        (i.e., width=24*14=336 and height=32*14=448).
        
        Args:
            mask1: torch tensor of shape (768, 768) on CUDA (mask for Image 1 → Image 2).
            mask2: torch tensor of shape (768, 768) on CUDA (mask for Image 2 → Image 1).
            x: row index (in the 32×24 grid) of the selected patch.
            y: column index (in the 32×24 grid) of the selected patch.
            orig_img1: original image 1 as a torch tensor in (H, W, 3) with values in [–1,1].
            orig_img2: original image 2 as a torch tensor in (H, W, 3) with values in [–1,1].
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle
        from torchvision import transforms
        from PIL import Image

        # Helper: Denormalize and resize image to mask size (336, 448)
        def process_orig_image(img_tensor, target_size=(336, 448)):
            # If values are in [-1,1], map them to [0,1]
            if img_tensor.min() < 0:
                img_tensor = (img_tensor + 1) / 2
            img_tensor = torch.clamp(img_tensor, 0, 1)
            # Ensure shape is (C, H, W) for ToPILImage (input is (H, W, 3))
            if img_tensor.ndimension() == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            pil_img = transforms.ToPILImage()(img_tensor)
            pil_img = pil_img.resize(target_size, Image.LANCZOS)
            return np.array(pil_img)

        # Process the original images.
        proc_img1 = process_orig_image(orig_img1)
        proc_img2 = process_orig_image(orig_img2)

        # Process masks: move to CPU and convert to numpy.
        mask1 = mask1.cpu().numpy()
        mask2 = mask2.cpu().numpy()

        # Compute the patch index for a grid of 32 rows x 24 columns.
        pixel_idx = x * 24 + y

        # Extract the corresponding row and reshape to (32, 24).
        mask_row1 = mask1[pixel_idx]
        mask_row2 = mask2[pixel_idx]
        mask_vis1 = mask_row1.reshape(32, 24)
        mask_vis2 = mask_row2.reshape(32, 24)

        # Scale up the mask visualization so that each patch becomes 14x14 pixels.
        patch_scale = 14
        mask_vis1 = np.repeat(np.repeat(mask_vis1, patch_scale, axis=0), patch_scale, axis=1)
        mask_vis2 = np.repeat(np.repeat(mask_vis2, patch_scale, axis=0), patch_scale, axis=1)

        # Define the rectangle parameters for the patch highlight.
        top_left = (y * patch_scale, x * patch_scale)
        patch_size = patch_scale

        # Create a 2x2 subplot layout:
        # Row 1: Left - Processed Image 1, Right - Mask from Image 1 → Image 2.
        # Row 2: Left - Processed Image 2, Right - Mask from Image 2 → Image 1.
        fig, axs = plt.subplots(2, 2, figsize=(20, 16))

        # Row 1, Column 1: Processed Original Image 1.
        axs[0, 0].imshow(proc_img1)
        axs[0, 0].set_title("Image 1")
        rect_img1 = Rectangle(top_left, patch_size, patch_size, linewidth=2,
                            edgecolor='white', facecolor='none')
        axs[0, 0].add_patch(rect_img1)
        axs[0, 0].set_xlabel("Width")
        axs[0, 0].set_ylabel("Height")

        # Row 1, Column 2: Mask from Image 1 to Image 2.
        im1 = axs[0, 1].imshow(mask_vis1, cmap='plasma')
        axs[0, 1].set_title(f"Mask from Image 1 to Image 2 (patch ({x}, {y})")
        rect_mask1 = Rectangle(top_left, patch_size, patch_size, linewidth=2,
                            edgecolor='white', facecolor='none')
        axs[0, 1].add_patch(rect_mask1)
        axs[0, 1].set_xlabel("Width (scaled)")
        axs[0, 1].set_ylabel("Height (scaled)")
        plt.colorbar(im1, ax=axs[0, 1], label='Similarity Score')

        # Row 2, Column 1: Processed Original Image 2.
        axs[1, 0].imshow(proc_img2)
        axs[1, 0].set_title("Image 2")
        rect_img2 = Rectangle(top_left, patch_size, patch_size, linewidth=2,
                            edgecolor='white', facecolor='none')
        axs[1, 0].add_patch(rect_img2)
        axs[1, 0].set_xlabel("Width")
        axs[1, 0].set_ylabel("Height")

        # Row 2, Column 2: Mask from Image 2 to Image 1.
        im2 = axs[1, 1].imshow(mask_vis2, cmap='plasma')
        axs[1, 1].set_title(f"Mask from Image 2 to Image 1 (patch ({x}, {y})")
        rect_mask2 = Rectangle(top_left, patch_size, patch_size, linewidth=2,
                            edgecolor='white', facecolor='none')
        axs[1, 1].add_patch(rect_mask2)
        axs[1, 1].set_xlabel("Width (scaled)")
        axs[1, 1].set_ylabel("Height (scaled)")
        plt.colorbar(im2, ax=axs[1, 1], label='Similarity Score')

        plt.tight_layout()
        plt.show()



    
    def _decoder(self, f1, pos1, mask_1_to_2, f2, pos2, mask_2_to_1, similarity_matrix=None):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2, mask_1_to_2, similarity_matrix)
            # img2 side
            
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1, mask_2_to_1, similarity_matrix)
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
        dist_1_to_2, mask_1_to_2, dist_2_to_1, mask_2_to_1, similarity_matrix = self._create_adjacency_graphs(dino_feat1, dino_feat2, self.top_k)
        
        # WIP: maybe turn this into parameter
        if False:
            for x in [0, 10, 21, 31]:
                for y in [0, 8, 15, 23]:
                    self._plot_mask(dist_1_to_2, dist_2_to_1, x, y, view1['img'][0].cpu().permute(1,2, 0), view2['img'][0].cpu().permute(1,2, 0))

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, mask_1_to_2, feat2, pos2, mask_2_to_1, similarity_matrix)
        #dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.amp.autocast('cuda',enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2

class DinoDecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = SparseCrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim)

    def forward(self, x, y, xpos, ypos, mask, similarities):
        x = x + self.attn(self.norm1(x), xpos)
        y_ = self.norm_y(y)
        x = x + self.cross_attn(self.norm2(x), y_, y_, xpos, ypos, mask, similarities)
        x = x + self.mlp(self.norm3(x))
        return x, y

class SparseCrossAttention(nn.Module):
    
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.rope = rope

    def forward(self, query, key, value, qpos, kpos, mask, similarities):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]
        
        q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        
        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)
            
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if False:
            # Use mask to set false values to -inf
            expanded_mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)   
            #print(f"Max attn: {torch.max(attn).item():.4f}")
            #print(f"Min attn: {torch.min(attn).item():.4f}")     
            attn = attn.masked_fill(~expanded_mask, float('-inf'))
        else:
            # WIP: make alpha trainable

            attn = attn + similarities
            expanded_mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)   
            attn = attn.masked_fill(~expanded_mask, float('-inf'))


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x