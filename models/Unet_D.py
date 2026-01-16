import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
import math

def get_wav_3d(in_channels, pool=True):
    # 3D Haar wavelet filters
    s = 1 / np.sqrt(2)
    harr_wav_L = s * np.ones((1, 2))
    harr_wav_H = s * np.ones((1, 2))
    harr_wav_H[0, 0] = -harr_wav_H[0, 0]

    # 3D filter banks (2x2x2 kernels)
    LLL = np.kron(np.kron(harr_wav_L, harr_wav_L), harr_wav_L).reshape(1, 1, 2, 2, 2)
    LLH = np.kron(np.kron(harr_wav_L, harr_wav_L), harr_wav_H).reshape(1, 1, 2, 2, 2)
    LHL = np.kron(np.kron(harr_wav_L, harr_wav_H), harr_wav_L).reshape(1, 1, 2, 2, 2)
    LHH = np.kron(np.kron(harr_wav_L, harr_wav_H), harr_wav_H).reshape(1, 1, 2, 2, 2)
    HLL = np.kron(np.kron(harr_wav_H, harr_wav_L), harr_wav_L).reshape(1, 1, 2, 2, 2)
    HLH = np.kron(np.kron(harr_wav_H, harr_wav_L), harr_wav_H).reshape(1, 1, 2, 2, 2)
    HHL = np.kron(np.kron(harr_wav_H, harr_wav_H), harr_wav_L).reshape(1, 1, 2, 2, 2)
    HHH = np.kron(np.kron(harr_wav_H, harr_wav_H), harr_wav_H).reshape(1, 1, 2, 2, 2)

    filters = [LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH]
    filter_tensors = [torch.from_numpy(f).float() for f in filters]  # Shape: [1, 1, 2, 2, 2]

    if pool:
        net = nn.Conv3d
    else:
        net = nn.ConvTranspose3d

    subbands = []
    for f in filter_tensors:
        conv = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
        conv.weight.requires_grad = False
        # More efficient expansion - using repeat_interleave for grouped convolution
        f_expanded = f.repeat_interleave(in_channels, dim=0).view(in_channels, 1, 2, 2, 2)
        conv.weight.data = f_expanded
        subbands.append(conv)
    
    return subbands

class WavePool3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.LLL, self.LLH, self.LHL, self.LHH, self.HLL, self.HLH, self.HHL, self.HHH = get_wav_3d(in_channels, pool=True)

    def forward(self, x):
        return [self.LLL(x), self.LLH(x), self.LHL(x), self.LHH(x), self.HLL(x), self.HLH(x), self.HHL(x), self.HHH(x)]

class WaveUnpool3D(nn.Module):
    def __init__(self, in_channels, option_unpool='sum'):
        super().__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LLL, self.LLH, self.LHL, self.LHH, self.HLL, self.HLH, self.HHL, self.HHH = get_wav_3d(in_channels, pool=False)

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH, original=None):
        if self.option_unpool == 'sum':
            return self.LLL(LLL) + self.LLH(LLH) + self.LHL(LHL) + self.LHH(LHH) + \
                   self.HLL(HLL) + self.HLH(HLH) + self.HHL(HHL) + self.HHH(HHH)
        elif self.option_unpool == 'cat' and original is not None:
            return torch.cat([self.LLL(LLL), self.LLH(LLH), self.LHL(LHL), self.LHH(LHH),
                              self.HLL(HLL), self.HLH(HLH), self.HHL(HHL), self.HHH(HHH), original], dim=1)
        else:
            raise NotImplementedError


class WaveletAttentionFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Project high-frequency features to out_channels
        self.high_freq_proj = nn.Conv3d(in_channels * 7, out_channels, kernel_size=1, bias=False)
        # Attention mechanism for high-frequency features
        self.attn = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, 1, kernel_size=1),  # Spatial attention
            nn.Sigmoid()
        )
        # Fuse main and attended high-frequency features
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, high_freq):
        # high_freq: list of 7 tensors [B, in_channels, D, H, W]
        high_freq_cat = torch.cat(high_freq, dim=1)  # [B, in_channels*7, D, H, W]
        high_freq_feat = self.high_freq_proj(high_freq_cat)  # [B, out_channels, D, H, W]
        attn_map = self.attn(high_freq_feat)  # [B, 1, D, H, W]
        high_freq_attended = high_freq_feat * attn_map  # [B, out_channels, D, H, W]
        fused = torch.cat([x, high_freq_attended], dim=1)  # [B, out_channels*2, D, H, W]
        out = self.fuse(fused)
        return out

class LKA(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()
        self.k_size = k_size
        
        self.conv_w = nn.Conv3d(dim, dim, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2), groups=dim)   
        self.gn = nn.GroupNorm(num_groups=dim, num_channels=dim)
        self.irelu = nn.ReLU(inplace=True)
        self.conv_spatial = nn.Conv3d(dim, dim, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(9, 9, 9), groups=dim, dilation=(3, 3, 3))
        self.conv = nn.Conv3d(dim, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        attn = self.conv_w(x)
        attn = self.gn(attn)
        attn = self.irelu(attn)
        attn = self.conv_spatial(attn)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)

        attn = u*attn
        u = u + attn
        return u

class ProjectExciteLayer(nn.Module):
    """
    Project & Excite Module, with wave-modulated and adaptively weighted
    combination of spatially-squeezed features.
    """

    def __init__(self, num_channels, reduction_ratio=2, mlp_hidden_ratio=1):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much num_channels should be reduced for the excitation bottleneck
        :param mlp_hidden_ratio: Ratio for hidden dim in MLPs generating theta and mixing weights
        """
        super(ProjectExciteLayer, self).__init__()
        self.num_channels = num_channels
        num_channels_reduced = num_channels // reduction_ratio
        mlp_hidden_dim = num_channels * mlp_hidden_ratio

        self.theta_mlp = nn.Sequential(
            nn.Linear(num_channels, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim), # Common practice in "wave" examples
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, num_channels)
        )

        self.reweight_mlp = nn.Sequential(
            nn.Linear(num_channels, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim), # Common practice in "wave" examples
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, num_channels * 3) # 3 weights per channel
        )

        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # --- 1. Original Squeeze Operations ---
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W)) # (B, C, 1, 1, W)
        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1)) # (B, C, 1, H, 1)
        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1)) # (B, C, D, 1, 1)

        # --- 2. Generate Global Context for MLPs ---
        global_context = F.adaptive_avg_pool3d(input_tensor, (1, 1, 1)).view(batch_size, num_channels) # (B, C)

        # --- 3. Generate Theta for Wave Modulation ---
        # Theta has shape (B, C), will be reshaped for broadcasting with squeezed tensors
        theta = self.theta_mlp(global_context) # (B, C)
        # Reshape theta for broadcasting with (B, C, D, H, W) style features
        theta_bc111 = theta.view(batch_size, num_channels, 1, 1, 1)

        # --- 4. Apply Wave Modulation to Squeezed Tensors ---
        # s_mod = s * cos(theta) + s * sin(theta)
        # theta_bc111 will broadcast correctly with squeeze_tensor_w, _h, _d
        s_w_mod = squeeze_tensor_w * torch.cos(theta_bc111) + squeeze_tensor_w * torch.sin(theta_bc111)
        s_h_mod = squeeze_tensor_h * torch.cos(theta_bc111) + squeeze_tensor_h * torch.sin(theta_bc111)
        s_d_mod = squeeze_tensor_d * torch.cos(theta_bc111) + squeeze_tensor_d * torch.sin(theta_bc111)

        # --- 5. Generate Adaptive Mixing Weights ---
        # reweight_mlp outputs (B, 3*C) for per-channel weights for 3 directions
        mixing_weights_flat = self.reweight_mlp(global_context) # (B, 3*C)
        # Reshape to (B, C, 3) and apply softmax over the 3 directions for each channel
        mixing_weights_per_channel = mixing_weights_flat.view(batch_size, num_channels, 3).softmax(dim=2) # (B, C, 3)

        alpha_w = mixing_weights_per_channel[:, :, 0].view(batch_size, num_channels, 1, 1, 1) # (B, C, 1, 1, 1)
        alpha_h = mixing_weights_per_channel[:, :, 1].view(batch_size, num_channels, 1, 1, 1) # (B, C, 1, 1, 1)
        alpha_d = mixing_weights_per_channel[:, :, 2].view(batch_size, num_channels, 1, 1, 1) # (B, C, 1, 1, 1)

        # --- 6. Wave-Modulated and Weighted Combination ---
        # Each alpha (B,C,1,1,1) will multiply its corresponding modulated squeezed tensor (B,C,D',H',W')
        final_squeeze_tensor = (alpha_w * s_w_mod.view(batch_size, num_channels, 1, 1, W) +
                                alpha_h * s_h_mod.view(batch_size, num_channels, 1, H, 1) +
                                alpha_d * s_d_mod.view(batch_size, num_channels, D, 1, 1))
        # The .view() calls ensure the components are structured as in the original P&E's sum,
        # which is important for how self.conv_c processes them.

        # --- 7. Original Excitation Path ---
        excitation = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))

        # --- 8. Apply Excitation ---
        output_tensor = torch.mul(input_tensor, excitation)

        return output_tensor

class Mlp3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PATM3D(nn.Module):
    def __init__(self, in_dim, out_dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='fc'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc_h = nn.Conv3d(in_dim, out_dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv3d(in_dim, out_dim, 1, 1, bias=qkv_bias)
        self.fc_d = nn.Conv3d(in_dim, out_dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv3d(in_dim, out_dim, 1, 1, bias=qkv_bias)
        self.tfc_h = nn.Conv3d(2 * out_dim, out_dim, (1, 1, 7), stride=1, padding=(0, 0, 7//2), groups=out_dim, bias=False)
        self.tfc_w = nn.Conv3d(2 * out_dim, out_dim, (1, 7, 1), stride=1, padding=(0, 7//2, 0), groups=out_dim, bias=False)
        self.tfc_d = nn.Conv3d(2 * out_dim, out_dim, (7, 1, 1), stride=1, padding=(7//2, 0, 0), groups=out_dim, bias=False)
        self.reweight = Mlp3D(out_dim, out_dim // 4, out_dim * 4)
        self.proj = nn.Conv3d(out_dim, out_dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        if mode == 'fc':
            self.theta_h_conv = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, 1, 1, bias=True),
                nn.GroupNorm(num_groups=out_dim, num_channels=out_dim),
                nn.ReLU()
            )
            self.theta_w_conv = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, 1, 1, bias=True),
                nn.GroupNorm(num_groups=out_dim, num_channels=out_dim),
                nn.ReLU()
            )
            self.theta_d_conv = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, 1, 1, bias=True),
                nn.GroupNorm(num_groups=out_dim, num_channels=out_dim),
                nn.ReLU()
            )
        else:
            self.theta_h_conv = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, 3, stride=1, padding=1, groups=in_dim, bias=False),
                nn.GroupNorm(num_groups=out_dim, num_channels=out_dim),
                nn.ReLU()
            )
            self.theta_w_conv = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, 3, stride=1, padding=1, groups=in_dim, bias=False),
                nn.GroupNorm(num_groups=out_dim, num_channels=out_dim),
                nn.ReLU()
            )
            self.theta_d_conv = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, 3, stride=1, padding=1, groups=in_dim, bias=False),
                nn.GroupNorm(num_groups=out_dim, num_channels=out_dim),
                nn.ReLU()
            )

    def forward(self, x):
        B, C, D, H, W = x.shape
        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)
        theta_d = self.theta_d_conv(x)
        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        x_d = self.fc_d(x)
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)
        x_d = torch.cat([x_d * torch.cos(theta_d), x_d * torch.sin(theta_d)], dim=1)
        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        d = self.tfc_d(x_d)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool3d(h + w + d + c, output_size=1)
        a = self.reweight(a).reshape(B, self.out_dim, 4).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + d * a[2] + c * a[3]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WaveBlock3D(nn.Module):
    def __init__(self, in_dim, out_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.GroupNorm, mode='fc'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if norm_layer == nn.GroupNorm:
            self.norm1 = norm_layer(num_groups=in_dim, num_channels=in_dim)
            self.norm2 = norm_layer(num_groups=out_dim, num_channels=out_dim)
        else:
            self.norm1 = norm_layer(in_dim)
            self.norm2 = norm_layer(out_dim)
        self.attn = PATM3D(in_dim=in_dim, out_dim=out_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = Mlp3D(in_features=out_dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer)
        # Projection for residual connection if in_dim != out_dim
        self.residual_proj = nn.Conv3d(in_dim, out_dim, 1, 1) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        x = self.residual_proj(x) + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
          )

    def forward(self,x):
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

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class UnetBaseD(nn.Module):
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

        self.dec1 = Up(24 * n_channels, 8 * n_channels)
        self.dec2 = Up(14 * n_channels, 6 * n_channels)
        self.dec3 = Up(10 * n_channels, 4 * n_channels)
        self.out = Out(4*n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2, subbands1 = self.enc1(x1)
        x3, subbands2 = self.enc2(x2)
        x4, subbands3 = self.enc3(x3)
        
        x4 = self.wave_enc(x4)
        x5 = self.wave_bottleneck(x4)
        x5 = self.bottleneck_norm(x5)
        x5 = self.wave_dec(x5)
        
        mask = self.dec1(x5, x3)
        mask = self.dec2(mask, x2)
        mask = self.dec3(mask, x1)
        out = self.out(mask)
        return out