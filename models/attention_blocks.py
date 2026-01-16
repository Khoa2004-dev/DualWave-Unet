import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import math

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

        # --- MLPs for Theta and Mixing Weights ---
        # These will operate on a global context vector derived from the input

        # MLP to generate theta parameters for wave modulation (shared for w, h, d)
        # Outputs C parameters, which will be broadcasted
        self.theta_mlp = nn.Sequential(
            nn.Linear(num_channels, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim), # Common practice in "wave" examples
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, num_channels)
        )

        # MLP to generate mixing weights for the three directions (w, h, d)
        # Outputs 3*C parameters for per-channel weighting of each direction
        self.reweight_mlp = nn.Sequential(
            nn.Linear(num_channels, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim), # Common practice in "wave" examples
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, num_channels * 3) # 3 weights per channel
        )

        # --- Original P&E Components ---
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

# %%
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