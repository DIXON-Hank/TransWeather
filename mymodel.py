import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import to_2tuple,trunc_normal_

class DropPath(nn.Module):
    """DropPath (Stochastic Depth) Regularization"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x  # No drop in evaluation mode or if drop_prob=0

        # Generate a random mask at the batch level
        keep_prob = 1 - self.drop_prob
        random_tensor = torch.rand(x.shape, device=x.device, dtype=x.dtype) < keep_prob
        output = x * random_tensor
        return output

class DynamicSparseAttention(nn.Module):
    """Dynamic Sparse Attention"""

    def __init__(self, dim, num_heads, top_k, attn_drop=0., proj_drop=0.):
        super(DynamicSparseAttention, self).__init__()
        self.num_heads = num_heads
        self.top_k = top_k
        self.scale = (dim // num_heads) ** -0.5

        # Linear projections for Q, K, V
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Forward pass through Dynamic Sparse Attention.
        Args:
            x (Tensor): Input tensor of shape (B, N, C), where B=batch size, N=sequence length, C=feature dimension.

        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        topk_indices = attn.topk(self.top_k, dim=-1, sorted=False)[1]  # Select top-k
        sparse_mask = torch.zeros_like(attn).scatter_(-1, topk_indices, 1)
        sparse_attn = attn * sparse_mask
        sparse_attn = self.attn_drop(torch.softmax(sparse_attn, dim=-1))

        # Compute output
        out = (sparse_attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MSFN(nn.Module):
    """Mixed-Scale Feed-forward Network"""

    def __init__(self, dim, hidden_dim, drop=0.):
        super(MSFN, self).__init__()
        self.conv3 = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=dim)
        self.conv5 = nn.Conv2d(dim, hidden_dim, kernel_size=5, padding=2, groups=dim)
        self.conv1x1 = nn.Conv2d(2 * hidden_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, C).

        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        B, N, C = x.shape
        H = W = int(N ** 0.5)  # Assume square input
        x = x.transpose(1, 2).view(B, C, H, W)

        # Multi-scale convolutions
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = torch.cat([x3, x5], dim=1)  # Concatenate multi-scale features
        x = self.conv1x1(x)  # Reduce to original dimension
        x = x.flatten(2).transpose(1, 2)  # Restore shape (B, N, C)
        x = self.dropout(x)
        return x

class SparseTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, top_k, mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.):
        super(SparseTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DynamicSparseAttention(dim, num_heads, top_k, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MSFN(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape  # Input shape is [batch_size, channels, height, width]

        # Flatten spatial dimensions for LayerNorm
        x = x.flatten(2).transpose(1, 2)  # Shape becomes [batch_size, seq_len, channels]

        # Apply attention block with residual connection
        x = x + self.drop_path1(self.attn(self.norm1(x)))

        # Apply MLP block with residual connection
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        # Reshape back to original spatial dimensions
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class DAttentionPatchEmbed(nn.Module): #TODO: add padding to solve stride division
    """Fixed Patch Embedding with Dynamic Attention"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch Embedding Layer
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Density Predictor for Degradation Level
        self.density_predictor = nn.Sequential(
            nn.Conv2d(in_chans, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Attention Mechanism for Feature Weighting
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),  # Reduce dimension
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),  # Restore dimension
            nn.Sigmoid()  # Generate weights
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # Step 1: Predict degradation level
        density_map = self.density_predictor(x)  # Shape: (B, 1, H, W)
        density_avg = torch.mean(density_map, dim=(1, 2, 3))  # Shape: (B,)

        # Step 2: Apply convolution for patch embedding
        x = self.proj(x)  # Shape: (B, embed_dim, H', W')
        _, _, H_out, W_out = x.shape
        x = x.flatten(2).transpose(1, 2)  # Shape: (B, N_patches, embed_dim)
        x = self.norm(x)  # Normalize embedded features

        # Step 3: Compute dynamic weights (attention mechanism)
        global_features = x.mean(dim=1)  # Global feature summary, shape: (B, embed_dim)
        attention_weights = self.attention(global_features)  # Shape: (B, embed_dim)

        # Step 4: Enhance or suppress features based on weights
        x = x * attention_weights.unsqueeze(1)  # Apply weights to features

        return x, H_out, W_out

class SelfAttentionFusion(nn.Module):
    """Self-Attention Fusion for Input and Output"""

    def __init__(self, embed_dim):
        super(SelfAttentionFusion, self).__init__()
        self.query_conv = nn.Conv2d(embed_dim, embed_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(embed_dim, embed_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        """
        Args:
            x1: Input image features (B, C, H, W).
            x2: Decoder output features (B, C, H, W).
        Returns:
            Tensor: Fused features of shape (B, C, H, W).
        """
        B, C, H, W = x1.shape

        # Query, Key, Value projections
        query = self.query_conv(x1).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        key = self.key_conv(x2).view(B, -1, H * W)  # (B, C//8, H*W)
        value = self.value_conv(x2).view(B, -1, H * W)  # (B, C, H*W)

        # Attention map
        attention = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)

        # Fused features
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(B, C, H, W)

        # Residual connection
        out = self.gamma * out + x2
        return out

class UNetTransformerWithAttentionFusion(nn.Module):
    """U-Net with Sparse Transformer Blocks and Self-Attention Fusion"""

    def __init__(self, img_size=224, patch_size=7, in_chans=3, embed_dim=64, num_heads=4, top_k=8, mlp_ratio=4.0, blocks_num=[2,2,2,2]):
        super().__init__()
        
        self.blocks_num = blocks_num
        self.patch_embed = DAttentionPatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=patch_size // 2,
            in_chans=in_chans, embed_dim=embed_dim
        )

        # Encoder
        self.encoder1 = self._make_layer(self.blocks_num[0], embed_dim, num_heads, top_k, mlp_ratio)
        self.encoder2 = self._make_layer(self.blocks_num[1], embed_dim * 2, num_heads, top_k, mlp_ratio)
        self.encoder3 = self._make_layer(self.blocks_num[2], embed_dim * 4, num_heads, top_k, mlp_ratio)

        # Bottleneck
        self.bottleneck = self._make_layer(self.blocks_num[3], embed_dim * 8, num_heads, top_k, mlp_ratio)

        # Decoder
        self.decoder3 = self._make_layer(self.blocks_num[2], embed_dim * 4, num_heads, top_k, mlp_ratio)
        self.decoder2 = self._make_layer(self.blocks_num[1], embed_dim * 2, num_heads, top_k, mlp_ratio)
        self.decoder1 = self._make_layer(self.blocks_num[0], embed_dim, num_heads, top_k, mlp_ratio)

        # Upsampling layers
        self.upsample3 = nn.ConvTranspose2d(embed_dim * 8, embed_dim * 4, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)

        # Self-Attention Fusion
        self.attention_fusion = SelfAttentionFusion(embed_dim)

        # Output layer
        self.output_layer = nn.Conv2d(embed_dim, in_chans, kernel_size=1)

    def _make_layer(self, block_num, dim, num_heads, top_k, mlp_ratio):
        """Create a layer with multiple Sparse Transformer Blocks."""
        return nn.Sequential(*[
            SparseTransformerBlock(dim=dim, num_heads=num_heads, top_k=top_k, mlp_ratio=mlp_ratio)
            for _ in range(block_num)
        ])

    def forward(self, x):
        # Original input for skip connection
        input_image = x

        # Patch Embedding
        x, H, W = self.patch_embed(x)

        # Reshape for Transformer
        x = x.transpose(1, 2).reshape(-1, self.patch_embed.embed_dim, H, W)

        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc3, kernel_size=2))

        # Decoder with Skip Connections
        dec3 = self.upsample3(bottleneck) + enc3
        dec3 = self.decoder3(dec3)

        dec2 = self.upsample2(dec3) + enc2
        dec2 = self.decoder2(dec2)

        dec1 = self.upsample1(dec2) + enc1
        dec1 = self.decoder1(dec1)

        # Self-Attention Fusion between Input and Decoder Output
        fused_features = self.attention_fusion(input_image, dec1)

        # Output
        output = self.output_layer(fused_features)
        return output

device = torch.cuda.get_device_name(0)
# Instantiate the model
model = UNetTransformerWithAttentionFusion(
    img_size=224, patch_size=7, in_chans=3, embed_dim=64, num_heads=4, top_k=8, blocks_num=2
).to('cuda')  # 将模型加载到 CUDA 上

# Generate a random input tensor
x = torch.randn(1, 3, 224, 224).to('cuda')  # 将输入张量加载到 CUDA 上

# Forward pass
output = model(x)

# Output shape
print(f"Output Shape: {output.shape}")  # Expected: (16, 3, 224, 224)





