######################################## DA_Net IEEE Access 2024   by AI Little monster start  ########################################
import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.nn.init import _calculate_fan_in_and_fan_out

# https://blog.csdn.net/m0_63774211?type=lately
from ultralytics_local.nn.modules.block import C3k, C3k2

# This is the official code of DA-Net for haze removal in remote sensing images (RSI).
# DA-Net: Dual Attention Network for Haze Removal in Remote Sensing Image
# IEEE Access
# 09/12/2024
# Namwon Kim (namwon@korea.ac.kr)


class ChannelBranch(nn.Module):
    # Channel Branch
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        avg_pool = self.avg_pool(x).view(x.size(0), -1)
        channel_att_raw = self.fc(avg_pool)
        channel_att = torch.sigmoid(channel_att_raw).unsqueeze(-1).unsqueeze(-1)
        return x * channel_att


class SpatialBranch(nn.Module):
    # Spatial Branch
    def __init__(self, in_channels):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, padding_mode="reflect"), nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.spatial(x)
        return x * scale


# Channel Spatial Attention Module
class ChannelSpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = ChannelBranch(in_channels)
        self.spatial_attention = SpatialBranch(in_channels)

    def forward(self, x):
        out = self.channel_attention(x) + self.spatial_attention(x)
        return out


class LocalChannelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, padding_mode="reflect")

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode="reflect"), nn.Sigmoid()
        )

    def forward(self, x):
        N, C, _H, _W = x.shape
        att = self.GAP(x).reshape(N, 1, C)
        att = self.conv(att).sigmoid()
        att = att.reshape(N, C, 1, 1)
        out = ((x * att) + x) + (self.local(x) * x)
        return out


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1), nn.Mish(True), nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class DualAttentionBlock(nn.Module):
    def __init__(self, dim, network_depth=1):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.dim = dim

        # shallow feature extraction layer
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)  # main
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode="reflect")  # main

        self.attn = ChannelSpatialAttentionModule(dim)

        # Local Channel Attention
        self.gp = LocalChannelAttention(dim)

        # Global Channel Attention
        self.cam = GlobalChannelAttention(dim)

        # Spatial Attention
        self.pam = SpatialAttention(dim)

        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * 4.0), out_features=dim)
        self.mlp2 = Mlp(network_depth, dim * 3, hidden_features=int(dim * 4.0), out_features=dim)

    def forward(self, x):
        # Channel Spatial Attention Module
        identity = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.mlp(x)
        x = identity + x

        # Parallel Attention Module
        identity = x
        x = self.norm2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.gp(x), self.cam(x), self.pam(x)], dim=1)
        x = self.mlp2(x)
        x = identity + x

        return x


# Global Channel Attention
class GlobalChannelAttention(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.ca(x) * x


# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv2d(dim, 1, kernel_size=7, padding=3), nn.Sigmoid())

    def forward(self, x):
        return self.spatial(x) * x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, network_depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([DualAttentionBlock(dim=dim, network_depth=network_depth) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class C3k_DAB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DualAttentionBlock(c_) for _ in range(n)))


class C3k2_DAB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_DAB(self.c, self.c, 2, shortcut, g) if c3k else DualAttentionBlock(self.c) for _ in range(n)
        )


######################################## DA_Net IEEE Access 2024   by AI Little monster end  ########################################
