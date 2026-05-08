import torch
import torch.nn as nn

__all__ = ["C3k2_BSD"]


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class BSD(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        kernel_sizes=(1, 3),
        expansion_factor=2,
        dw_parallel=True,
    ):
        super().__init__()

        assert stride in (1, 2)
        assert len(kernel_sizes) == 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.dw_parallel = dw_parallel
        self.use_residual = stride == 1

        hidden_channels = int(in_channels * expansion_factor)

        # Fexp = LeakyReLU(BN(Conv1x1(X)))
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # DWConv1x1 and DWConv3x3 branches
        self.dwconvs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for kernel_size in kernel_sizes
        )

        # BN(Conv1x1(ChannelShuffle(Fbs)))
        self.pconv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_residual and in_channels != out_channels:
            self.shortcut_conv = nn.Conv2d(
                in_channels,
                out_channels,
                1,
                1,
                0,
                bias=False,
            )

        self.apply(init_weights)

    def forward(self, x):
        f_exp = self.pconv1(x)

        # Fbs = DWConv1x1(Fexp) + DWConv3x3(Fexp)
        branch_outs = []
        for dwconv in self.dwconvs:
            branch_outs.append(dwconv(f_exp))

        f_bs = branch_outs[0]
        for branch_out in branch_outs[1:]:
            f_bs = f_bs + branch_out

        f_bs = channel_shuffle(
            f_bs,
            gcd(f_bs.shape[1], self.out_channels),
        )

        out = self.pconv2(f_bs)

        # Y = X + BN(Conv1x1(ChannelShuffle(Fbs)))
        if self.use_residual:
            if self.in_channels != self.out_channels:
                x = self.shortcut_conv(x)
            out = x + out

        return out


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [
            d * (x - 1) + 1 for x in k
        ]

    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]

    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()

        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()

        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            Bottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                k=((3, 3), (3, 3)),
                e=1.0,
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv2(torch.cat(y, 1))


class C3k2_BSD(C2f):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        g=1,
        shortcut=True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            BSD(self.c, self.c) for _ in range(n)
        )