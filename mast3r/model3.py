# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DINOMASt3R model class
# --------------------------------------------------------
import torch
import torch.nn.functional as F

from mast3r.model import AsymmetricMASt3R

import mast3r.utils.path_to_dust3r  # noqa

from torchvision import transforms
from PIL import Image
from functools import partial
import torch.nn as nn 
from copy import deepcopy

import dust3r.utils.path_to_croco  # noqa: F401
from models.blocks import Attention, DropPath, Mlp

# ========================================================

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.autograd import Variable

n_inf = float('-inf')

class DINOMASt3R(AsymmetricMASt3R):
    def __init__(self, topK=1, postprocess=True, plot=False,**kwargs):
        super().__init__(**kwargs)

        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
        dino_model.eval()  # Set the model to evaluation mode
        self.dino_model = dino_model

        self.dec_blocks = nn.ModuleList([
            DinoDecoderBlock(self.dec_embed_dim, self.dec_depth, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_mem=True, rope=self.rope)
            for i in range(self.dec_depth)])

        self.dec_blocks2 = deepcopy(self.dec_blocks)

        self.topK = topK

        self.postprocess = postprocess

        self.feature_correlation_4d = FeatureCorrelation(shape='4D', normalization=False)
        self.ncnet_consensus = NeighConsensus(
            use_cuda=True, 
            kernel_sizes=[3,3,3],   # example 
            channels=[10,10,1],     # example 
            symmetric_mode=True
        )

        self.plot = plot


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
            patch_features = F.normalize(patch_features, p=2, dim=-1) # Now each feature has unit L2 norm

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

    def _create_adjacency_graphs(self, features1, features2, topK):
        """
        features1: [H1, W1, D]
        features2: [H2, W2, D]
        """
        # 1) Construct 4D correlation volume using official "FeatureCorrelation"
        #    shape=(B=1, C=D, H, W). Then correlation => (B=1,1,H1,W1,H2,W2)
        B = 1
        # Turn (H1, W1, D) => (B=1, D, H1, W1)
        fA = features1.permute(2,0,1).unsqueeze(0)
        fB = features2.permute(2,0,1).unsqueeze(0)

        # If you want them L2-normalized, do it here or rely on the official layers
        fA = featureL2Norm(fA)
        fB = featureL2Norm(fB)

        # correlation 4D => shape [1,1,H1,W1,H2,W2]
        corr4d = self.feature_correlation_4d(fA, fB)

        # 2) The official pipeline does:
        #    corr4d = MutualMatching(corr4d)
        #    corr4d = self.ncnet_consensus(corr4d)
        #    corr4d = MutualMatching(corr4d)
        corr4d = MutualMatching(corr4d)
        corr4d = self.ncnet_consensus(corr4d)
        corr4d = MutualMatching(corr4d)

        # 3) Flatten to get final similarity matrix => shape (N1, N2)
        #    where N1=H1*W1, N2=H2*W2
        B, _, H1, W1, H2, W2 = corr4d.shape  # typically B=1, channel=1
        # shape => [B, 1, H1, W1, H2, W2] -> [H1*W1, H2*W2]
        refined_sim = corr4d.view(B, 1, H1*W1, H2*W2).squeeze(0).squeeze(0)

        # 4) Then do topK on the refined similarity
        #    refined_sim is shape (N1, N2).
        N1, N2 = refined_sim.shape
        k1 = int(N2 * topK)
        k2 = int(N1 * topK)

        topk_values_1, topk_indices_1 = torch.topk(refined_sim, k1, dim=1)
        adj_1_to_2 = torch.full_like(refined_sim, float('-inf'))
        adj_1_to_2.scatter_(1, topk_indices_1, topk_values_1)

        topk_values_2, topk_indices_2 = torch.topk(refined_sim.transpose(0,1), k2, dim=1)
        adj_2_to_1 = torch.full_like(refined_sim.transpose(0,1), float('-inf'))
        adj_2_to_1.scatter_(1, topk_indices_2, topk_values_2)

        mask_1_to_2 = (adj_1_to_2 != float('-inf'))
        mask_2_to_1 = (adj_2_to_1 != float('-inf'))

        return adj_1_to_2, mask_1_to_2, adj_2_to_1, mask_2_to_1, refined_sim

    
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
        dist_1_to_2, mask_1_to_2, dist_2_to_1, mask_2_to_1, similarity_matrix = self._create_adjacency_graphs(dino_feat1, dino_feat2, self.topK)
        

        if self.plot:
            self._plot_mask(dist_1_to_2, dist_2_to_1, 10, 10, view1['img'][0].cpu().permute(1,2, 0), view2['img'][0].cpu().permute(1,2, 0))



        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, mask_1_to_2, feat2, pos2, mask_2_to_1, similarity_matrix)
        
        with torch.amp.autocast('cuda',enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2




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
        axs[0, 1].set_title(f"Mask from Image 1 to Image 2 (patch ({x}, {y}))")
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
        axs[1, 1].set_title(f"Mask from Image 2 to Image 1 (patch ({x}, {y}))")
        rect_mask2 = Rectangle(top_left, patch_size, patch_size, linewidth=2,
                            edgecolor='white', facecolor='none')
        axs[1, 1].add_patch(rect_mask2)
        axs[1, 1].set_xlabel("Width (scaled)")
        axs[1, 1].set_ylabel("Height (scaled)")
        plt.colorbar(im2, ax=axs[1, 1], label='Similarity Score')

        plt.tight_layout()
        plt.show()


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

        # FIXME: we center for now
        similarities_centered = similarities - similarities.mean(dim=1, keepdim=True)
        attn = attn + similarities_centered
        # NOTE: we dont use the mask
        if True: 
            expanded_mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)   
            attn = attn.masked_fill(~expanded_mask, n_inf)


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


######################################################
# NCNet Stuff

class FeatureCorrelation(torch.nn.Module):
    def __init__(self,shape='3D',normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape=shape
        self.ReLU = nn.ReLU()
    
    def forward(self, feature_A, feature_B):        
        if self.shape=='3D':
            b,c,h,w = feature_A.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
            feature_B = feature_B.view(b,c,h*w).transpose(1,2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B,feature_A)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        elif self.shape=='4D':
            b,c,hA,wA = feature_A.size()
            b,c,hB,wB = feature_B.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b,c,hA*wA).transpose(1,2) # size [b,c,h*w]
            feature_B = feature_B.view(b,c,hB*wB) # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A,feature_B)
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,hA,wA,hB,wB).unsqueeze(1)
        
        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
            
        return correlation_tensor

class NeighConsensus(torch.nn.Module):
    def __init__(self, use_cuda=True, kernel_sizes=[3,3,3], channels=[10,10,1], symmetric_mode=True):
        super(NeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i==0:
                ch_in = 1
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(Conv4d(in_channels=ch_in,out_channels=ch_out,kernel_size=k_size,bias=True))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)        
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = self.conv(x)+self.conv(x.permute(0,1,4,5,2,3)).permute(0,1,4,5,2,3)
            # because of the ReLU layers in between linear layers, 
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x = self.conv(x)
        return x

def MutualMatching(corr4d):
    # mutual matching
    batch_size,ch,fs1,fs2,fs3,fs4 = corr4d.size()

    corr4d_B=corr4d.view(batch_size,fs1*fs2,fs3,fs4) # [batch_idx,k_A,i_B,j_B]
    corr4d_A=corr4d.view(batch_size,fs1,fs2,fs3*fs4)

    # get max
    corr4d_B_max,_=torch.max(corr4d_B,dim=1,keepdim=True)
    corr4d_A_max,_=torch.max(corr4d_A,dim=3,keepdim=True)

    eps = 1e-5
    corr4d_B=corr4d_B/(corr4d_B_max+eps)
    corr4d_A=corr4d_A/(corr4d_A_max+eps)

    corr4d_B=corr4d_B.view(batch_size,1,fs1,fs2,fs3,fs4)
    corr4d_A=corr4d_A.view(batch_size,1,fs1,fs2,fs3,fs4)

    corr4d=corr4d*(corr4d_A*corr4d_B) # parenthesis are important for symmetric output 
        
    return corr4d



def conv4d(data, filters, bias=None, permute_filters=True, use_half=False):
    b, c, h, w, d, t = data.size()

    # permute to avoid making contiguous inside loop
    data = data.permute(2, 0, 1, 3, 4, 5).contiguous()

    # Same permutation is done with filters, unless already provided with permutation
    if permute_filters:
        # permute to avoid making contiguous inside loop
        filters = filters.permute(2, 0, 1, 3, 4, 5).contiguous()

    c_out = filters.size(1)
    if use_half:
        output = Variable(
            torch.HalfTensor(h, b, c_out, w, d, t), requires_grad=data.requires_grad
        )
    else:
        output = Variable(
            torch.zeros(h, b, c_out, w, d, t), requires_grad=data.requires_grad
        )

    padding = filters.size(0) // 2
    if use_half:
        Z = Variable(torch.zeros(padding, b, c, w, d, t).half())
    else:
        Z = Variable(torch.zeros(padding, b, c, w, d, t))

    if data.is_cuda:
        Z = Z.cuda(data.get_device())
        output = output.cuda(data.get_device())

    data_padded = torch.cat((Z, data, Z), 0)

    for i in range(output.size(0)):  # loop on first feature dimension
        # convolve with center channel of filter (at position=padding)
        output[i, :, :, :, :, :] = F.conv3d(
            data_padded[i + padding, :, :, :, :, :],
            filters[padding, :, :, :, :, :],
            bias=bias,
            stride=1,
            padding=padding,
        )
        # convolve with upper/lower channels of filter (at postions [:padding] [padding+1:])
        for p in range(1, padding + 1):
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(
                data_padded[i + padding - p, :, :, :, :, :],
                filters[padding - p, :, :, :, :, :],
                bias=None,
                stride=1,
                padding=padding,
            )
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(
                data_padded[i + padding + p, :, :, :, :, :],
                filters[padding + p, :, :, :, :, :],
                bias=None,
                stride=1,
                padding=padding,
            )

    output = output.permute(1, 2, 0, 3, 4, 5).contiguous()
    return output


class Conv4d(_ConvNd):
    """
    Applies a 4D convolution over an input signal composed of several input planes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        pre_permuted_filters=True,
    ):
        # stride, dilation and groups !=1 functionality not tested
        stride = 1
        dilation = 1
        groups = 1
        # zero padding is added automatically in conv4d function to preserve tensor size
        padding = 0
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)
        super(Conv4d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _quadruple(0),
            groups,
            bias,
            padding_mode="zeros",  # This fixes TypeError: __init__() missing 1 required positional argument: 'padding_mode' in Python 3.7
        )
        # weights will be sliced along one dimension during convolution loop
        # make the looping dimension to be the first one in the tensor,
        # so that we don't need to call contiguous() inside the loop
        self.pre_permuted_filters = pre_permuted_filters
        if self.pre_permuted_filters:
            self.weight.data = self.weight.data.permute(2, 0, 1, 3, 4, 5).contiguous()
        self.use_half = False

    def forward(self, input):
        # filters pre-permuted in constructor
        return conv4d(
            input,
            self.weight,
            bias=self.bias,
            permute_filters=not self.pre_permuted_filters,
            use_half=self.use_half,
        )

def Softmax1D(x,dim):
    x_k = torch.max(x,dim)[0].unsqueeze(dim)
    x -= x_k.expand_as(x)
    exp_x = torch.exp(x)
    return torch.div(exp_x,torch.sum(exp_x,dim).unsqueeze(dim).expand_as(x))

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm) 