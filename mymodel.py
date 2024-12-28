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

    def __init__(self, dim, num_heads, top_k=128, attn_drop=0., proj_drop=0.):
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

class EncoderSTB(nn.Module):
    def __init__(self, dim, num_heads=8, top_k=128, mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.):
        super(EncoderSTB, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DynamicSparseAttention(dim, num_heads, top_k, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MSFN(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x): 
        # Apply attention block with residual connection
        x = x + self.drop_path1(self.attn(self.norm1(x)))

        # Apply MLP block with residual connection
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class DecoderSTB(nn.Module):
    def __init__(self, dim, num_heads, top_k=128, mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.global_attn = DynamicSparseAttention(dim, num_heads, top_k=top_k, attn_drop=attn_drop, proj_drop=drop)
        self.local_attn = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=num_heads)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MSFN(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # Apply global attention
        global_features = self.global_attn(self.norm1(x))

        # Apply local attention
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x_reshaped = x.transpose(1, 2).view(B, C, H, W)
        local_features = self.local_attn(x_reshaped).flatten(2).transpose(1, 2)

        # Combine global and local attention features
        attn_output = self.norm1(global_features + local_features)

        # Apply MLP with residual connection
        x = x + self.drop_path1(attn_output)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class DAttentionPatchEmbed(nn.Module): 
    """Fixed Patch Overlap Embedding with Dynamic Attention"""

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

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
        attention_weights = self.attention(global_features) * density_avg.unsqueeze(-1)  # Shape: (B, embed_dim)

        # Step 4: Enhance or suppress features based on weights
        x = x * attention_weights.unsqueeze(1)  # Apply weights to features

        return x, H_out, W_out

class PatchEmbed(nn.Module): 
    """Overlap Patch Embedding"""

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch Embedding Layer
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize Conv2d layer
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

        # Initialize LayerNorm layer
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)  # Shape: (B, embed_dim, H', W')
        _, _, H_out, W_out = x.shape
        x = x.flatten(2).transpose(1, 2)  # Shape: (B, N_patches, embed_dim)
        x = self.norm(x)  # Normalize embedded features

        return x, H_out, W_out

class SelfAttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttentionFusion, self).__init__()
        self.query_conv = nn.Conv2d(embed_dim, embed_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(embed_dim, embed_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习参数

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        _, _, H_actual, W_actual = x2.shape

        # Query, Key, Value 的计算
        query = self.query_conv(x1).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        key = self.key_conv(x2).view(B, -1, H_actual * W_actual)  # (B, C//8, H_actual*W_actual)
        value = self.value_conv(x2).view(B, -1, H_actual * W_actual)  # (B, C, H_actual*W_actual)

        # 计算注意力分布
        attention = torch.bmm(query, key)  # (B, H*W, H_actual*W_actual)
        attention = F.softmax(attention, dim=-1)

        # 融合特征
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(B, C, H, W)  # 还原到二维特征图格式

        # 加入残差连接
        out = self.gamma * out + x2
        return out

class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(OutputLayer, self).__init__()
        # 反卷积层
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,     # 输入通道数
            out_channels=out_channels,  # 输出通道数
            kernel_size=4,              # 卷积核大小
            stride=4,                   # 步幅
            padding=0,                  # 填充
            output_padding=0            # 输出填充
        )

    def forward(self, x):
        return self.deconv(x)

class UNetTransformerWithAttentionFusion(nn.Module):
    """U-Net with Sparse Transformer Blocks and Self-Attention Fusion"""

    def __init__(self, img_size=224, in_chans=3, embed_dims=[64,128,256,512], num_heads=[1,2,4,8], top_k=8, mlp_ratio=4.0, blocks_num=[1, 2, 4, 4]):
        super().__init__()

        self.attn_patch_embed = DAttentionPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])

        self.embed_layer1 = PatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.embed_layer2 = PatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.embed_layer3 = PatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Encoder
        self.encoder1 = self._make_encoder(blocks_num[0], embed_dims[0], num_heads[0], top_k, mlp_ratio)
        self.encoder2 = self._make_encoder(blocks_num[1], embed_dims[1], num_heads[1], top_k, mlp_ratio)
        self.encoder3 = self._make_encoder(blocks_num[2], embed_dims[2], num_heads[2], top_k, mlp_ratio)

        # Bottleneck
        self.bottleneck = self._make_encoder(blocks_num[3], embed_dims[3], num_heads[3], top_k, mlp_ratio)

        # Decoder
        self.decoder3 = self._make_decoder(1, embed_dims[2], num_heads[2], top_k, mlp_ratio)
        self.decoder2 = self._make_decoder(1, embed_dims[1], num_heads[1], top_k, mlp_ratio)
        self.decoder1 = self._make_decoder(1, embed_dims[0], num_heads[0], top_k, mlp_ratio) #TODO: add droppath diffrentials

        # Upsampling layers
        self.upsample3 = nn.ConvTranspose2d(embed_dims[3], embed_dims[2], kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(embed_dims[2], embed_dims[1], kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(embed_dims[1], embed_dims[0], kernel_size=2, stride=2)

        # Self-Attention Fusion
        self.attention_fusion = SelfAttentionFusion(embed_dims[0])
        self.feature_adapter = nn.Conv2d(in_channels=3, out_channels=embed_dims[0], kernel_size=7, stride=4, padding=3)
        # Output layer
        self.output_layer = OutputLayer(embed_dims[0], in_chans)

    def _make_encoder(self, block_num, dim, num_heads, top_k, mlp_ratio, attn_drop=0., drop_path=0.): # TODO: add droppath diffrentials 靠近中心的地方高，越来越低
        """Create a layer with multiple Sparse Transformer Blocks."""
        return nn.Sequential(*[
            EncoderSTB(dim=dim, num_heads=num_heads, top_k=top_k, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop_path=drop_path)
            for _ in range(block_num)
        ])
    
    def _make_decoder(self, block_num, dim, num_heads, top_k, mlp_ratio, attn_drop=0., drop_path=0.): # TODO: add droppath diffrentials 靠近中心的地方高，越来越低
        """Create a layer with multiple Sparse Transformer Blocks."""
        return nn.Sequential(*[
            DecoderSTB(dim=dim, num_heads=num_heads, top_k=top_k, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop_path=drop_path)
            for _ in range(block_num)
        ])

    def forward(self, x):
        # Original input for skip connection
        input_image = x #(1,3,224,224)

        # Patch Embedding
        x, H, W = self.attn_patch_embed(x)
        # Encoder
        enc1 = self.encoder1(x) 
        enc1_reshaped = enc1.transpose(1,2).reshape(-1, self.attn_patch_embed.embed_dim, H, W)
        # print(enc1_reshaped.shape) #1,64,56,56
        enc2_input, H, W = self.embed_layer1(enc1_reshaped)
        enc2 = self.encoder2(enc2_input)
        enc2_reshaped = enc2.transpose(1, 2).reshape(-1, self.embed_layer1.embed_dim, H, W)
        # print(enc2_reshaped.shape) #1，128，28，28
        enc3_input, H, W = self.embed_layer2(enc2_reshaped)
        enc3 = self.encoder3(enc3_input)
        
        enc3_reshaped = enc3.transpose(1, 2).reshape(-1, self.embed_layer2.embed_dim, H, W)
        # print(enc3_reshaped.shape) #1,256,14,14
        # Bottleneck
        bottleneck_input, H, W = self.embed_layer3(enc3_reshaped)
        # (1,49,512)
        bottleneck = self.bottleneck(bottleneck_input)
        bottleneck += bottleneck_input # skip connections
        # (1,49,512)
        # Decoder with Skip Connections
        bottleneck_reshaped = bottleneck.transpose(1, 2).reshape(-1, self.embed_layer3.embed_dim, H, W)
        dec3_input = self.upsample3(bottleneck_reshaped)
        # 1,256,14,14
        dec3_reshaped = dec3_input.view(dec3_input.shape[0], dec3_input.shape[1], -1).permute(0, 2, 1)
        dec3 = self.decoder3(dec3_reshaped)
        dec3 = dec3 + enc3
        # 1,196,256

        dec3_reshaped = dec3.transpose(1, 2).reshape(-1, self.embed_layer2.embed_dim, H * 2, W * 2)
        dec2_input = self.upsample2(dec3_reshaped)
        dec2_reshaped = dec2_input.view(dec3_input.shape[0], dec2_input.shape[1], -1).permute(0, 2, 1)
        dec2 = self.decoder2(dec2_reshaped)
        dec2 = dec2 + enc2
        # 1,784,128

        dec2_reshaped = dec2.transpose(1, 2).reshape(-1, self.embed_layer1.embed_dim, H * 4, W * 4)
        dec1_input = self.upsample1(dec2_reshaped)
        dec1_reshaped = dec1_input.view(dec3_input.shape[0], dec1_input.shape[1], -1).permute(0, 2, 1)
        dec1 = self.decoder1(dec1_reshaped)
        dec1= dec1 + enc1
        # 1,3136,64

        B1, N1, C1 = dec1.shape
        H0 = W0 = int(math.sqrt(N1))
        decoder_features = dec1.transpose(1, 2).view(B1, C1, H0, W0)
        # Self-Attention Fusion between Input and Decoder Output
        input_features = self.feature_adapter(input_image)
        fused_features = self.attention_fusion(input_features, decoder_features)

        # Output
        output = self.output_layer(fused_features)
        return output






