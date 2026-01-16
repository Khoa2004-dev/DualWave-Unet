import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Down2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encode1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm3d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.spatial_gating_unit = LKA(dim = out_channels, k_size=3)
        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.INorm = nn.InstanceNorm3d(num_features=in_channels)
        self.LR = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.PE = ProjectExciteLayer(num_channels = out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.encode1(x)
        x = self.spatial_gating_unit(x)
        x1 = self.INorm(x)
        x1 = self.PE(x1)
        x1 = self.sigmoid(x1)
        x = x1 * x # element-wise multiplied
        return x


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


class UnetBaseB(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down2(n_channels, 2 * n_channels)
        self.enc2 = Down2(2 * n_channels, 4 * n_channels)
        self.enc3 = Down2(4 * n_channels, 8 * n_channels)
        self.enc4 = Down2(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask
