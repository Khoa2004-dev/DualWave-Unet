import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from .wavelet_blocks import WavePool3D, WaveUnpool3D, WaveletAttentionFusion
from .attention_blocks import LKA, ProjectExciteLayer

class LayerNormProxy3d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b d h w c -> b c d h w')

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=16, kernel_size=3, padding=1, stride=1, bias=True):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down2W(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.wave_pool = WavePool3D(in_channels)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm3d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.wavelet_fusion = WaveletAttentionFusion(in_channels, out_channels)
        self.spatial_gating_unit = LKA(dim=out_channels, k_size=3)
        self.INorm = nn.InstanceNorm3d(num_features=out_channels)
        self.PE = ProjectExciteLayer(num_channels=out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        subbands = self.wave_pool(x)
        LLL = subbands[0]
        high_freq = subbands[1:]
        x_main = self.conv(LLL)
        x = self.wavelet_fusion(x_main, high_freq)
        x = self.spatial_gating_unit(x)
        x1 = self.INorm(x)
        x1 = self.PE(x1)
        x1 = self.sigmoid(x1)
        x = x1 * x
        return x, subbands

class Up1W(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()
        self.upT = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.wave_unpool = WaveUnpool3D(out_channels, option_unpool='cat')
        
        self.wavelet_proj = nn.Sequential(
            nn.Conv3d(out_channels * 7, out_channels, kernel_size=1, 
                      groups=out_channels, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # Simplified three-way fusion attention
        self.fusion_attention = nn.Sequential(
            nn.Conv3d(out_channels * 3, out_channels, kernel_size=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False), 
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        self.PE = ProjectExciteLayer(num_channels=out_channels)

    def forward(self, x1, x2, subbands):
        # Upsample and reconstruct
        up1 = self.upT(x1)
        wave_up = self.wave_unpool(subbands, include_low=False)
        wave_up = self.wavelet_proj(wave_up)
        
        # Three-way attention fusion
        fusion_input = torch.cat([up1, x2, wave_up], dim=1)
        attention_weights = self.fusion_attention(fusion_input)  # [B, 3, D, H, W]
        
        # Weighted fusion
        fused = (attention_weights[:, 0:1] * up1 + 
                attention_weights[:, 1:2] * x2 + 
                attention_weights[:, 2:3] * wave_up)
        
        # Apply PE enhancement and final processing
        pe_weights = torch.sigmoid(self.PE(fused))
        enhanced = fused * pe_weights
        
        # Final convolutions with residual
        output = self.conv_out(enhanced)
        return output + enhanced