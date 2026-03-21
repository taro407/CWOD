import torch
import torch.nn.functional as F
from torch import nn

from ultralytics_local.nn.modules.conv import Conv
from ultralytics_local.utils.torch_utils import fuse_conv_and_bn


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)


class DilatedBlock(nn.Module):
    """Standard bottleneck with dilated convolution."""

    def __init__(self, c, dilation, k, fuse="sum", shortcut=True):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        self.dilation = dilation
        self.k = k
        self.cv2 = Conv(c, c, k=1, s=1)
        self.add = shortcut

        self.fuse = fuse
        if fuse == "glu":
            self.conv_gating = Conv(c * len(self.dilation), c * len(self.dilation), k=1, s=1, g=c * len(self.dilation))
            self.conv1x1 = Conv(c * len(self.dilation), c, k=1, s=1, g=c)
        elif fuse == "sum":
            self.conv1x1 = Conv(c, c, k=1, s=1, g=c)

        self.dcv = Conv(c, c, k=self.k, s=1)

    def dilated_conv(self, x, dilation):
        act = self.dcv.act
        bn = self.dcv.bn
        weight = self.dcv.conv.weight
        padding = dilation * (self.k // 2)
        return act(bn(F.conv2d(x, weight, stride=1, padding=padding, dilation=dilation)))

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        dx = [self.dilated_conv(x, d) for d in self.dilation]
        dx = [self.cv2(_dx) for _dx in dx]
        if self.fuse == "glu":
            dx = torch.cat(dx, dim=1)
            G = torch.sigmoid(self.conv_gating(dx))
            dx = dx * G  # Element-wise multiplication
            dx = self.conv1x1(dx)
        elif self.fuse == "sum":
            dx = [_dx.unsqueeze(0) for _dx in dx]
            dx = torch.cat(dx, dim=0)
            dx = torch.sum(dx, dim=0)
            dx = self.conv1x1(dx)

        return x + dx if self.add else dx


class DilatedBottleneck(nn.Module):
    """Standard bottleneck with dilated convolution."""

    def __init__(self, c1, c2, shortcut=True, dilation=[1, 2, 3], block_k=3, fuse="sum", g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()

        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)

        self.dilated_block = DilatedBlock(c_, dilation, block_k, fuse)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.dilated_block(self.cv1(x))) if self.add else self.cv2(self.dilated_block(self.cv1(x)))


class G2L_CRM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1, c2, n=1, shortcut=False, use_dilated=False, dilation=[1, 2, 3], block_k=3, fuse="sum", g=1, e=0.5
    ):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        if use_dilated:
            self.m = nn.ModuleList(
                DilatedBottleneck(self.c, self.c, shortcut, dilation, block_k, fuse, g, k=((3, 3), (3, 3)), e=1.0)
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))
