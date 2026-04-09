# ===============================================================
# Compact MSCB v2 with CrossLink
# 轻量双路径多尺度特征块（带 CrossLink 通道交互机制）
# 用于替换 C2f 内部 Bottleneck（兼顾轻量与协同表达）
# ===============================================================

import torch
import torch.nn as nn

__all__ = ["CMSCB", "CMSCB_V2_CL", "C3k2_CMSCB"]


# -----------------------------
# 基础组件
# -----------------------------
def autopad(k, p=None, d=1):
    """Pad to 'same' for k, with dilation d."""
    if isinstance(k, int):
        k_eff = d * (k - 1) + 1
        return k_eff // 2 if p is None else p
    else:
        return [autopad(ki, None, d) if p is None else p for ki in k]


class Conv(nn.Module):
    """Conv2d + BN + SiLU."""

    default_act = nn.SiLU(inplace=True)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise Conv(k×k) + BN + Act."""

    def __init__(self, c, k=3, s=1, d=1, act=True):
        super().__init__()
        self.conv = Conv(c, c, k=k, s=s, d=d, g=c, act=act)

    def forward(self, x):
        return self.conv(x)


# -----------------------------
# CMSCB_V2 with CrossLink
# -----------------------------
class CMSCB_V2_CL(nn.Module):
    """轻量双路径多尺度特征块（带 CrossLink 通道交互机制） - Path1：1×1 → DW 3×3（局部语义） - Path2：1×1 → DW 5×5 → DW 3×3(d=2)（大感受野） - Add 融合，内部可选
    CrossLink 交叉注入.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, inner_e=0.5, crosslink_ratio=0.25):
        super().__init__()
        assert 0 < inner_e <= 1.0, "inner_e must be in (0, 1]."
        self.add = shortcut and (c1 == c2)
        self.r = max(0.0, min(0.5, float(crosslink_ratio)))  # CrossLink 比例

        c_mid = max(1, int(c2 * inner_e))  # 两路径统一的轻量中间通道

        # Path 1：稳健局部语义
        self.p1_reduce = Conv(c1, c_mid, k=1, s=1)
        self.p1_dw3 = DWConv(c_mid, k=3, s=1, d=1)

        # Path 2：广感受野
        self.p2_reduce = Conv(c1, c_mid, k=1, s=1)
        self.p2_dw5 = DWConv(c_mid, k=5, s=1, d=1)
        self.p2_dil3 = DWConv(c_mid, k=3, s=1, d=2)

        # 融合后投影
        self.out_proj = Conv(c_mid, c2, k=1, s=1)

    # -------------------------
    # CrossLink 模块（无参数）
    # -------------------------
    @torch.no_grad()
    def _crosslink(self, a, b):
        """通道互换比例 crosslink_ratio（默认0.25） 作用：增强两路径信息流通，减少特征孤立。.
        """
        if self.r <= 0.0 or a.shape[1] != b.shape[1]:
            return a, b
        C = a.shape[1]
        n = int(C * self.r)
        if n == 0:
            return a, b
        a1, a2 = a[:, :n], a[:, n:]
        b1, b2 = b[:, :n], b[:, n:]
        a_new = torch.cat([b1, a2], dim=1)
        b_new = torch.cat([a1, b2], dim=1)
        return a_new, b_new

    # -------------------------
    # 前向传播
    # -------------------------
    def forward(self, x):
        # Path 1
        y1 = self.p1_dw3(self.p1_reduce(x))
        # Path 2
        y2 = self.p2_dil3(self.p2_dw5(self.p2_reduce(x)))

        # CrossLink：增强路径间交流（无参数）
        y1, y2 = self._crosslink(y1, y2)

        # Add 融合 + 输出映射
        y = self.out_proj(y1 + y2)
        return x + y if self.add else y


# -----------------------------
# 兼容 C2f 结构
# -----------------------------
class C2f(nn.Module):
    """最小化版本 CSP 模块."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, k=1, s=1)
        self.cv2 = Conv((2 + n) * self.c, c2, k=1, s=1)
        self.m = nn.ModuleList(nn.Identity() for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


# -----------------------------
# C3k2_CMSCB：YAML可调用接口
# -----------------------------
class C3k2_CMSCB(C2f):
    """直接替换 C2f 内部 Bottleneck 为 CMSCB_V2_CL."""

    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, inner_e=0.5, crosslink_ratio=0.25):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=1, e=e)
        self.m = nn.ModuleList(
            CMSCB_V2_CL(self.c, self.c, shortcut=True, e=1.0, inner_e=inner_e, crosslink_ratio=crosslink_ratio)
            for _ in range(n)
        )


# 兼容旧名
CMSCB = CMSCB_V2_CL


# -----------------------------
# 自检
# -----------------------------
if __name__ == "__main__":
    x = torch.randn(1, 256, 80, 80)
    m = C3k2_CMSCB(256, 256, n=2, shortcut=True, e=0.5, inner_e=0.5, crosslink_ratio=0.25)
    y = m(x)
    print("Output shape:", y.shape)
