import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_tf_

__all__ = ["C3k2_MAKDF"]


# ------------------------------
# 初始化 & 激活（与你给的 MSCB 风格一致）
# ------------------------------
def _init_weights(module, name="", scheme=""):
    if isinstance(module, nn.Conv2d):
        if scheme == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif scheme == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif scheme == "trunc_normal":
            trunc_normal_tf_(module.weight, std=0.02)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act="silu", inplace=True):
    a = act.lower()
    if a == "relu":
        return nn.ReLU(inplace)
    if a == "relu6":
        return nn.ReLU6(inplace)
    if a == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace)
    if a == "gelu":
        return nn.GELU()
    return nn.SiLU(inplace)


# ------------------------------
# 基础卷积
# ------------------------------
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act is True else act_layer() if isinstance(act, str) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ------------------------------
# AKDC：K×K, 1×M, M×1（M=3K+2），Softmax 融合
# ------------------------------
class AKDC(nn.Module):
    def __init__(self, c, K=3, r=16):
        super().__init__()
        assert K in (1, 3, 5)
        M = 3 * K + 2
        self.sq = nn.Sequential(
            nn.Conv2d(c, c, K, 1, autopad(K), groups=c, bias=False), nn.BatchNorm2d(c), nn.SiLU(inplace=True)
        )
        self.h = nn.Sequential(
            nn.Conv2d(c, c, (1, M), 1, (0, M // 2), groups=c, bias=False), nn.BatchNorm2d(c), nn.SiLU(inplace=True)
        )
        self.v = nn.Sequential(
            nn.Conv2d(c, c, (M, 1), 1, (M // 2, 0), groups=c, bias=False), nn.BatchNorm2d(c), nn.SiLU(inplace=True)
        )
        hid = max(c // r, 8)
        self.w = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid, 3 * c, 1, bias=True),
        )
        self.pw = Conv(c, c, 1, 1, act=True)

    def forward(self, x):
        b, c, _, _ = x.shape
        y1, y2, y3 = self.sq(x), self.h(x), self.v(x)
        w = self.w(x).view(b, 3, c, 1, 1)
        w = torch.softmax(w, dim=1)
        y = w[:, 0] * y1 + w[:, 1] * y2 + w[:, 2] * y3
        return self.pw(y)


# ------------------------------
# MAKDF：三组通道 → AKDC(1/3/5) → Concat → 1×1
# 输出通道数 == 输入通道数（为残差/拼接服务）
# ------------------------------
class MAKDF(nn.Module):
    def __init__(self, c, e=1.0):
        super().__init__()
        c_mid = int(c * e)
        self.pre = Conv(c, c_mid, 1, 1, act=True)
        g1 = c_mid // 3
        g2 = (c_mid - g1) // 2
        g3 = c_mid - g1 - g2
        self.slices = (g1, g2, g3)
        # 防极端小通道
        assert min(self.slices) > 0, f"MAKDF channel groups too small: {self.slices}"
        self.a1 = AKDC(g1, K=1)
        self.a3 = AKDC(g2, K=3)
        self.a5 = AKDC(g3, K=5)
        self.post = Conv(c_mid, c, 1, 1, act=True)

    def forward(self, x):
        x = self.pre(x)
        g1, g2, g3 = self.slices
        x1, x2, x3 = torch.split(x, [g1, g2, g3], dim=1)
        y = torch.cat([self.a1(x1), self.a3(x2), self.a5(x3)], dim=1)
        return self.post(y)


# ------------------------------
# Bottleneck_MAKDF：按论文“替换最后一层卷积”
# 原 bottleneck：cv1(1×1) -> cv2(3×3)
# 本实现：cv1(1×1) -> MAKDF(c_)    （输出维度仍为 c_）
# ------------------------------
class Bottleneck_MAKDF(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=1.0):
        super().__init__()
        # 这里 c1==c2==c_（在 C2f/C3k2 内部使用）
        self.cv1 = Conv(c1, c2, k=1, s=1)
        self.makdf = MAKDF(c2, e=1.0)  # 替换“最后一层卷积”为 MAKDF
        self.add = shortcut and (c1 == c2)

    def forward(self, x):
        y = self.makdf(self.cv1(x))
        return x + y if self.add else y


# ------------------------------
# C3k2_MAKDF：保留 C2f 外壳，只把内部 m 列表换成 Bottleneck_MAKDF
# cv1(c1→2c_),  m: n×[Bottleneck_MAKDF(c_→c_)],  cv2((2+n)c_→c2)
# ------------------------------
class C3k2_MAKDF(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c1 = c1[0]
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_MAKDF(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))
        named_apply(partial(_init_weights, scheme=""), self)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------
# self-test
# ------------------------------
if __name__ == "__main__":
    x = torch.randn(1, 256, 80, 80)
    m = C3k2_MAKDF(256, 256, n=2, shortcut=True, e=0.5)
    y = m(x)
    print("in:", x.shape, "out:", y.shape)
