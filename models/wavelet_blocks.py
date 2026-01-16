import torch
import torch.nn as nn
import numpy as np

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

    def forward(self, subbands, include_low=True):
        # Unpack the list of subbands
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = subbands
        if self.option_unpool == 'sum':
            high_freq_sum = (self.LLH(LLH) + self.LHL(LHL) + self.LHH(LHH) + 
                             self.HLL(HLL) + self.HLH(HLH) + self.HHL(HHL) + 
                             self.HHH(HHH))
            if include_low:
                return self.LLL(LLL) + high_freq_sum
            else:
                return high_freq_sum
                
        elif self.option_unpool == 'cat':
            if include_low:
                return torch.cat([self.LLL(LLL), self.LLH(LLH), self.LHL(LHL), self.LHH(LHH),
                              self.HLL(HLL), self.HLH(HLH), self.HHL(HHL), self.HHH(HHH)], dim=1)
            else:
                return torch.cat([self.LLH(LLH), self.LHL(LHL), self.LHH(LHH),
                              self.HLL(HLL), self.HLH(HLH), self.HHL(HHL), self.HHH(HHH)], dim=1)
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