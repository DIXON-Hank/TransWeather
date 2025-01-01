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

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        dim: Number of input channels.
        window_size: Size of the window (H, W).
        num_heads: Number of attention heads.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (window_height, window_width)
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        x: Input feature of shape (B, N, C).
        mask: Attention mask of shape (num_windows, window_size * window_size, window_size * window_size).
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Make queries, keys and values

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BlockAttentionFusion(nn.Module):
    def __init__(self, embed_dim, window_size=16, num_heads=8):
        """
        embed_dim: Channel dimension of input.
        window_size: Size of the attention window.
        num_heads: Number of attention heads.
        """
        super(BlockAttentionFusion, self).__init__()
        self.window_size = window_size
        self.attention = WindowAttention(
            dim=embed_dim,
            window_size=(window_size, window_size),
            num_heads=num_heads
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable parameter

    def window_partition(self, x, window_size):
        """
        Partition input tensor into windows.
        Args:
            x: Input tensor of shape (B, H, W, C).
            window_size: Size of the window.
        Returns:
            windows: Tensor of shape (num_windows * B, window_size, window_size, C).
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        """
        Reverse the window partition to reconstruct the image.
        Args:
            windows: Windows tensor of shape (num_windows * B, window_size, window_size, C).
            window_size: Size of the window.
            H, W: Height and width of the original image.
        Returns:
            x: Reconstructed tensor of shape (B, H, W, C).
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
        return x
    
    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1_windows = self.window_partition(x1.permute(0, 2, 3, 1), self.window_size)  # (num_windows*B, window_size, window_size, C)
        x2_windows = self.window_partition(x2.permute(0, 2, 3, 1), self.window_size)  # Same shape

        x1_windows = x1_windows.view(-1, self.window_size**2, C)
        x2_windows = x2_windows.view(-1, self.window_size**2, C)

        # Compute attention
        fused_windows = self.attention(x1_windows + x2_windows)  # Add skip connection

        # Restore windows back to spatial dimensions
        fused_windows = fused_windows.view(-1, self.window_size, self.window_size, C)
        fused_features = self.window_reverse(fused_windows, self.window_size, H, W).permute(0, 3, 1, 2)

        return self.gamma * fused_features + x2  # Residual connection

class DynamicSparseAttention(nn.Module):
    """Dynamic Sparse Attention"""

    def __init__(self, dim, num_heads, top_k=16, attn_drop=0., proj_drop=0.):
        super(DynamicSparseAttention, self).__init__()
        self.num_heads = num_heads
        self.top_k = top_k
        self.scale = (dim // num_heads) ** -0.5

        # Linear projections for Q, K, V
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.activation = nn.ReLU()

        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self._init_weights()

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.activation(q)
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

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class MSFN(nn.Module):
    """Mixed-Scale Feed-forward Network"""

    def __init__(self, dim, hidden_dim, drop=0.):
        super(MSFN, self).__init__()
        self.conv3 = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=dim)
        self.conv5 = nn.Conv2d(dim, hidden_dim, kernel_size=5, padding=2, groups=dim)
        self.conv1x1 = nn.Conv2d(2 * hidden_dim, dim, kernel_size=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.conv1x1.weight)
        if self.conv3.bias is not None:
            nn.init.constant_(self.conv3.bias, 0)
        if self.conv5.bias is not None:
            nn.init.constant_(self.conv5.bias, 0)
        if self.conv1x1.bias is not None:
            nn.init.constant_(self.conv1x1.bias, 0)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        assert H * W ==N, f"H * W ({H} * {W}) must equal N ({N})"

        x = x.transpose(1, 2).view(B, C, H, W)

        # Multi-scale convolutions
        x3 = self.activation(self.conv3(x))
        x5 = self.activation(self.conv5(x))
        x = torch.cat([x3, x5], dim=1)  # Concatenate multi-scale features
        x = self.conv1x1(x)  # Reduce to original dimension

        x = x.flatten(2).transpose(1, 2)  # Restore back to shape (B, N, C)
        x = self.dropout(x)
        return x

class EncoderSTB(nn.Module):
    def __init__(self, dim, num_heads=8, top_k=8, mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.):
        super(EncoderSTB, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DynamicSparseAttention(dim, num_heads, top_k, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MSFN(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, H, W): 
        # Apply attention block with residual connection
        x = x + self.drop_path1(self.attn(self.norm1(x)))

        # Apply MLP block with residual connection
        x = x + self.drop_path2(self.mlp(self.norm2(x), H, W))
        return x

class DecoderSTB(nn.Module):
    def __init__(self, dim, num_heads, top_k=8, mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.global_attn = DynamicSparseAttention(dim, num_heads, top_k=top_k, attn_drop=attn_drop, proj_drop=drop)
        self.local_attn = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=num_heads)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MSFN(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.activation = nn.ReLU()

        self.weights = nn.Parameter(torch.tensor([0.5, 0.5])) # learnable weights for global/local attention fusion

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.global_attn.q.weight)
        nn.init.xavier_uniform_(self.global_attn.kv.weight)
        nn.init.xavier_uniform_(self.global_attn.proj.weight)
        if self.global_attn.q.bias is not None:
            nn.init.constant_(self.global_attn.q.bias, 0)
        if self.global_attn.kv.bias is not None:
            nn.init.constant_(self.global_attn.kv.bias, 0)
        if self.global_attn.proj.bias is not None:
            nn.init.constant_(self.global_attn.proj.bias, 0)

        nn.init.kaiming_normal_(self.local_attn.weight, mode='fan_out', nonlinearity='relu')
        if self.local_attn.bias is not None:
            nn.init.constant_(self.local_attn.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, H, W):
        global_features = self.global_attn(self.norm1(x))

        B, N, C = x.shape
        x_reshaped = x.transpose(1, 2).view(B, C, H, W)
        local_features = self.activation(self.local_attn(x_reshaped).flatten(2).transpose(1, 2))

        weights = F.softmax(self.weights, dim=0)
        global_weight, local_weight = weights[0], weights[1]
        attn_output = self.norm1(global_weight * global_features + local_weight * local_features)

        x = x + self.drop_path1(attn_output)
        x = x + self.drop_path2(self.mlp(self.norm2(x), H, W))
        return x

class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureAdapter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=4, padding=3)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.pointwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class upsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
      super(upsampleConv, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out
    
class OutputLayer(nn.Module):
    def __init__(self, input_channels=128, output_channels=3):
        super(OutputLayer, self).__init__()
        # 1,128,H/4,W/4
        self.convd1 = upsampleConv(input_channels, input_channels // 2, kernel_size=4, stride=2, padding=1)
        # 1,64,H/2,W/2
        self.dense1 = nn.Sequential(ResidualBlock(input_channels // 2))
        self.convd2 = upsampleConv(input_channels // 2, input_channels // 8, kernel_size=4, stride=2, padding=1)
        # 1,8,H,W
        self.dense2 = nn.Sequential(ResidualBlock(input_channels // 8))
        self.conv_output = nn.Conv2d(input_channels // 8, output_channels, kernel_size=3, stride=1, padding=1)
        self.active = nn.Tanh()
    
    def forward(self, x):
        # 第一层上采样 + 残差
        x_res = x  # 保留输入用于残差连接
        x = self.convd1(x)
        x = self.dense1(x) + self.convd1(x_res)  # 添加残差连接

        # 第二层上采样 + 残差
        x_res = x
        x = self.convd2(x)
        x = self.dense2(x) + self.convd2(x_res)  # 添加残差连接

        # 第三层上采样 + 输出
        x = self.conv_output(x)
        x = self.active(x)
        return x

class UNetTransformerWithAttentionFusion(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 4, 8], top_k=16,
                 mlp_ratio=4.0, blocks_num=[2, 2, 3, 4], drop_path_rate=0.1):
        super().__init__()
        # get drop_path_rate for each layer
        depth = sum(blocks_num)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        dpr1 = dpr[0:sum(blocks_num[:1])]
        dpr2 = dpr[sum(blocks_num[:1]):sum(blocks_num[:2])]
        dpr3 = dpr[sum(blocks_num[:2]):sum(blocks_num[:3])]
        dpr4 = dpr[sum(blocks_num[:3]):sum(blocks_num[:4])]

        self.embed_layer0 = PatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.embed_layer1 = PatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.embed_layer2 = PatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.embed_layer3 = PatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.encoder1 = self._make_encoder(blocks_num[0], embed_dims[0], num_heads[0], top_k, mlp_ratio, dpr1)
        self.encoder2 = self._make_encoder(blocks_num[1], embed_dims[1], num_heads[1], top_k, mlp_ratio, dpr2)
        self.encoder3 = self._make_encoder(blocks_num[2], embed_dims[2], num_heads[2], top_k, mlp_ratio, dpr3)

        self.bottleneck = self._make_encoder(blocks_num[3], embed_dims[3], num_heads[3], top_k, mlp_ratio, dpr4)

        self.decoder3 = self._make_decoder(blocks_num[2] , embed_dims[2], num_heads[2], top_k, mlp_ratio)
        self.decoder2 = self._make_decoder(blocks_num[1], embed_dims[1], num_heads[1], top_k, mlp_ratio)
        self.decoder1 = self._make_decoder(blocks_num[0], embed_dims[0], num_heads[0], top_k, mlp_ratio)

        self.upsample3 = upsampleConv(embed_dims[3], embed_dims[2], kernel_size=2, stride=2)
        self.upsample2 = upsampleConv(embed_dims[2], embed_dims[1], kernel_size=2, stride=2)
        self.upsample1 = upsampleConv(embed_dims[1], embed_dims[0], kernel_size=2, stride=2)
    
        # Self-Attention Fusion
        self.attention_fusion = BlockAttentionFusion(embed_dims[0], window_size=8, num_heads=8)
        self.feature_adapter = FeatureAdapter(in_channels=3, out_channels=embed_dims[0])
        # Output layer
        self.output_layer = OutputLayer(embed_dims[0] * 2, in_chans)

    def _make_encoder(self, block_num, dim, num_heads, top_k, mlp_ratio, drop_path, attn_drop=0.): # TODO: add droppath diffrentials 靠近中心的地方高，越来越低
        """Create a layer with multiple Sparse Transformer Blocks."""
        return nn.Sequential(*[
            EncoderSTB(dim=dim, num_heads=num_heads, top_k=top_k, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop_path=drop_path[i])
            for i in range(block_num)
        ])

    def _make_decoder(self, block_num, dim, num_heads, top_k, mlp_ratio, attn_drop=0., ):
        return nn.Sequential(*[
            DecoderSTB(dim=dim, num_heads=num_heads, top_k=top_k, mlp_ratio=mlp_ratio, attn_drop=attn_drop)
            for i in range(block_num)
        ])

    def forward(self, x):
        # Original input for skip connection
        input_image = x #(1,3,224,224)
        B, C, _, _ = x.shape

        # Patch Embedding
        x, H, W = self.embed_layer0(x)

        # ------ Encoder ------
        enc1_res = x # input of enc1, save for skip connections
        for blk in self.encoder1:
            x = blk(x, H, W)
        x = x.transpose(1,2).contiguous().view(-1, self.embed_layer0.embed_dim, H, W)
        x, H, W = self.embed_layer1(x)

        enc2_res = x # output of enc1/ input of enc2, save for skip connections
        for blk in self.encoder2:
            x = blk(x, H, W)
        x = x.transpose(1, 2).contiguous().view(-1, self.embed_layer1.embed_dim, H, W)
        x, H, W = self.embed_layer2(x)

        enc3_res = x
        for blk in self.encoder3:
            x = blk(x, H, W)        
        x = x.transpose(1, 2).contiguous().view(-1, self.embed_layer2.embed_dim, H, W)
        x, H, W = self.embed_layer3(x)

        bottleneck_res = x
        for blk in self.bottleneck:
            x = blk(x, H, W)

        # ------ Decoder with Skip Connections ------
        x += bottleneck_res # skip connections
        x = x.transpose(1, 2).contiguous().view(-1, self.embed_layer3.embed_dim, H, W)

        x = self.upsample3(x)

        H, W = x.shape[2], x.shape[3]
        x = x.view(B, x.shape[1], -1).permute(0, 2, 1)

        for blk in self.encoder3:
            x = blk(x, H, W)
        x = x + enc3_res
        # 1,196,256

        x = x.transpose(1, 2).contiguous().view(-1, self.embed_layer2.embed_dim, H, W)
        x = self.upsample2(x)
        H, W = x.shape[2], x.shape[3]
        x = x.view(B, x.shape[1], -1).permute(0, 2, 1)
        for blk in self.encoder2:
            x = blk(x, H, W)
        x = x + enc2_res
        # 1,784,128

        x = x.transpose(1, 2).contiguous().view(-1, self.embed_layer1.embed_dim, H, W)
        x = self.upsample1(x)
        H, W = x.shape[2], x.shape[3]
        x = x.view(B, x.shape[1], -1).permute(0, 2, 1)
        for blk in self.encoder1:
            x = blk(x, H, W)
        x = x + enc1_res
        # 1,3136,64
        _, _, C1 = x.shape
        decoder_features = x.transpose(1, 2).view(B, C1, H, W)
        # 1,64,56,56
        # Self-Attention Fusion between Input and Decoder Output
        input_features = self.feature_adapter(input_image)
        fused_features = self.attention_fusion(input_features, decoder_features)
        # Output
        concat_features = torch.concat([decoder_features, fused_features], dim=1)
        output = self.output_layer(concat_features)
        return output