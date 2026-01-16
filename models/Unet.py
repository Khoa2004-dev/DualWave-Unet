import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_
from .base_blocks import DoubleConv, Down2W, Up1W
from .attention_blocks import WaveBlock3D

class Unet(nn.Module):
    def __init__(self, in_channels=4, n_channels=64, n_classes=2, drop_path_rate=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels
        
        self.conv = DoubleConv(in_channels, 4*n_channels)
        self.enc1 = Down2W(4*n_channels, 6*n_channels)
        self.enc2 = Down2W(6*n_channels, 8*n_channels)
        self.enc3 = Down2W(8*n_channels, 12*n_channels)
        
        self.wave_enc = WaveBlock3D(
            in_dim=12*n_channels, out_dim=16*n_channels, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=drop_path_rate / 3, norm_layer=nn.GroupNorm, mode='fc'
        )
        self.wave_bottleneck = WaveBlock3D(
            in_dim=16*n_channels, out_dim=32*n_channels, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=drop_path_rate / 3, norm_layer=nn.GroupNorm, mode='fc'
        )
        self.bottleneck_norm = nn.GroupNorm(num_groups=32*n_channels, num_channels=32*n_channels)
        self.wave_dec = WaveBlock3D(
            in_dim=32*n_channels, out_dim=16*n_channels, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=drop_path_rate / 3, norm_layer=nn.GroupNorm, mode='fc'
        )
        
        self.dec1 = Up1W(16*n_channels, 8*n_channels)
        self.dec2 = Up1W(8*n_channels, 6*n_channels)
        self.dec3 = Up1W(6*n_channels, 4*n_channels)
        self.out = nn.Conv3d(in_channels=4*n_channels, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.conv(x)
        x2, subbands1 = self.enc1(x1)
        x3, subbands2 = self.enc2(x2)
        x4, subbands3 = self.enc3(x3)
        
        x4 = self.wave_enc(x4)
        x5 = self.wave_bottleneck(x4)
        x5 = self.bottleneck_norm(x5)
        x5 = self.wave_dec(x5)
        
        x_out = self.dec1(x5, x3, subbands3)
        x_out = self.dec2(x_out, x2, subbands2)
        x_out = self.dec3(x_out, x1, subbands1)
        out = self.out(x_out)
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()