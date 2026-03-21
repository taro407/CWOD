import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# 🔸 基础卷积块
# ==========================================================
class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ==========================================================
# 🔸 RepGhostConv 模块（训练阶段）
# ==========================================================
class RepGhostConv(nn.Module):
    """
    Reference:
    "RepGhost: Integrating Re-parameterization and Ghost Convolution for Lightweight CNNs"
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, reparam=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reparam = reparam  # 是否为推理模式

        # 主分支: 1×1 conv + depthwise conv
        self.primary_conv = nn.Sequential(
            ConvBNAct(in_channels, out_channels, 1, 1),
            ConvBNAct(out_channels, out_channels, kernel_size, stride, groups=out_channels)
        )

        # 旁支: identity + BN
        self.shortcut = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

        # 重参数化后的单分支（仅在推理时使用）
        if reparam:
            self.reparam_conv = ConvBNAct(in_channels, out_channels, kernel_size, stride, groups=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.reparam:  # 推理阶段
            return self.relu(self.reparam_conv(x))
        else:  # 训练阶段
            out = self.primary_conv(x)
            if self.shortcut is not None:
                out = out + self.shortcut(x)
            return self.relu(out)

    # ==========================================================
    # 🔁 重参数化函数
    # ==========================================================
    def fuse_reparam(self):
        """
        将训练阶段的多分支结构融合为单一卷积。
        """
        if self.shortcut is None:
            identity_w = torch.zeros_like(self.primary_conv[1].conv.weight)
            identity_b = torch.zeros(self.primary_conv[1].bn.num_features, device=identity_w.device)
        else:
            identity_w, identity_b = self._fuse_bn_tensor(self.shortcut)

        conv1_w, conv1_b = self._fuse_bn_tensor(self.primary_conv[0].bn, self.primary_conv[0].conv)
        conv2_w, conv2_b = self._fuse_bn_tensor(self.primary_conv[1].bn, self.primary_conv[1].conv)

        # 合并两层卷积 (1x1 + DWConv)
        fused_weight = conv2_w
        fused_bias = conv2_b + conv1_b.mean()

        # 将 identity 分支合并进来
        fused_weight += F.pad(identity_w, [1, 1, 1, 1])  # 对齐通道与卷积尺寸
        fused_bias += identity_b

        # 创建融合后的卷积层
        fused_conv = ConvBNAct(self.in_channels, self.out_channels, 3, 1)
        fused_conv.conv.weight.data = fused_weight
        fused_conv.bn.bias.data = fused_bias

        self.reparam_conv = fused_conv
        self.reparam = True
        del self.primary_conv, self.shortcut  # 删除旧结构以节省显存

    def _fuse_bn_tensor(self, bn, conv=None):
        """BN 与 Conv 融合为单一卷积核"""
        if conv is None:
            # identity 分支
            w = torch.zeros(self.out_channels, self.in_channels, 3, 3, device=bn.weight.device)
            for i in range(self.in_channels):
                w[i, i, 1, 1] = 1.0
            b = torch.zeros_like(bn.bias)
        else:
            w = conv.weight
            b = torch.zeros(bn.weight.shape, device=w.device)

        # BN参数融合
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = (var + eps).sqrt()
        w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
        b_fused = beta - gamma * mean / std

        return w_fused, b_fused
