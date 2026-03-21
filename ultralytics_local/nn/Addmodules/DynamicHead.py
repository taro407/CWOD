import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d

from ultralytics_local.utils.tal import dist2bbox, make_anchors

__all__ = ["DynamicHead"]


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class DYReLU(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        reduction=4,
        lambda_a=1.0,
        K2=True,
        use_bias=True,
        use_spatial=False,
        init_a=[1.0, 0.0],
        init_b=[0.0, 0.0],
    ):
        super().__init__()
        self.oup = oup
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b

        squeeze = inp // reduction if reduction == 4 else _make_divisible(inp // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze), nn.ReLU(inplace=True), nn.Linear(squeeze, oup * self.exp), h_sigmoid()
        )
        self.spa = (
            nn.Sequential(
                nn.Conv2d(inp, 1, kernel_size=1),
                nn.BatchNorm2d(1),
            )
            if use_spatial
            else None
        )

    def forward(self, x):
        if isinstance(x, list):
            x_in, x_out = x[0], x[1]
        else:
            x_in = x_out = x
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias:
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1
            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)
        else:
            a1 = (y - 0.5) * self.lambda_a + self.init_a[0]
            out = x_out * a1

        if self.spa:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
            ys = F.hardtanh(ys, 0, 3, inplace=True) / 3
            out = out * ys
        return out


class Conv3x3Norm(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)

    def forward(self, input, **kwargs):
        x = self.conv(input.contiguous(), **kwargs)
        x = self.bn(x)
        return x


class DyConvLevel(nn.Module):
    def __init__(self, c_prev, c_cur, c_next, conv_func=Conv3x3Norm):
        super().__init__()
        self.conv_same = conv_func(c_cur, c_cur, 1)
        self.conv_down = conv_func(c_prev, c_cur, 2) if c_prev else None
        self.conv_up = conv_func(c_next, c_cur, 1) if c_next else None

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_cur, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.act = DYReLU(c_cur, c_cur)
        self.offset = nn.Conv2d(c_cur, 27, kernel_size=3, stride=1, padding=1)

    def forward(self, feat_cur, feat_prev=None, feat_next=None):
        _B, _C, H, W = feat_cur.shape
        offset_mask = self.offset(feat_cur)
        offset = offset_mask[:, :18, :, :]
        mask = offset_mask[:, 18:, :, :].sigmoid()
        conv_args = dict(offset=offset, mask=mask)

        cand = [self.conv_same(feat_cur, **conv_args)]
        if self.conv_down is not None and feat_prev is not None:
            cand.append(self.conv_down(feat_prev, **conv_args))
        if self.conv_up is not None and feat_next is not None:
            up_in = F.interpolate(feat_next, size=(H, W), mode="nearest")
            cand.append(self.conv_up(up_in, **conv_args))

        attns = [self.attn(f) for f in cand]
        stack = torch.stack(cand)  # [K,B,C,H,W]
        attn_stack = h_sigmoid()(torch.stack(attns))  # [K,B,1,1,1]
        fused = torch.mean(stack * attn_stack, dim=0)
        return self.act(fused)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DynamicHead(nn.Module):
    """轻量 DyHead 检测头（P3~P5）： - 各层保留自身通道数 - 首次前向自动推断 stride=[8,16,32] 并只初始化一次 bias - 兼容 Ultralytics 的 m.strides 迁移逻辑.
    """

    # === 类属性：Ultralytics 在 ._apply() 里会访问，必须先有 ===
    dynamic = False
    export = False
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)  # <- 类属性存在
    strides = torch.empty(0)  # <- 类属性存在

    def __init__(self, nc=80, ch=()):
        super().__init__()
        assert isinstance(ch, (list, tuple)) and len(ch) > 0, f"DynamicHead requires ch list/tuple, got {ch}"
        self.nc = nc
        self.in_channels = list(ch)
        self.nl = len(self.in_channels)

        # DyHead 单轮
        levels = []
        for i in range(self.nl):
            c_prev = self.in_channels[i - 1] if i - 1 >= 0 else 0
            c_cur = self.in_channels[i]
            c_next = self.in_channels[i + 1] if i + 1 < self.nl else 0
            levels.append(DyConvLevel(c_prev, c_cur, c_next, conv_func=Conv3x3Norm))
        self.dyhead_levels = nn.ModuleList(levels)

        # 预测头
        self.reg_max = 16
        self.no = nc + self.reg_max * 4

        # === Instance buffers ===
        self.register_buffer("stride", torch.zeros(self.nl))  # 每层标量步长 8/16/32
        self.register_buffer("strides", torch.empty(0))  # 与 anchors 搭配使用的 per-location strides
        self.register_buffer("anchors", torch.empty(0))
        self._bias_inited = False

        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()
        for c in self.in_channels:
            c2 = max(16, c // 4, self.reg_max * 4)
            c3 = max(c, min(self.nc, 100))
            self.cv2.append(nn.Sequential(Conv(c, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)))
            self.cv3.append(
                nn.Sequential(
                    nn.Sequential(DWConv(c, c, 3), Conv(c, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
            )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        # 1) DyHead 聚合
        outs = []
        for i in range(self.nl):
            feat_cur = x[i]
            feat_prev = x[i - 1] if i - 1 >= 0 else None
            feat_next = x[i + 1] if i + 1 < self.nl else None
            outs.append(self.dyhead_levels[i](feat_cur, feat_prev, feat_next))

        # 2) 第一次前向兜底写 stride，并做一次 bias_init
        if (self.stride == 0).any():
            Hs = [t.shape[-2] for t in outs]  # 高从大到小对应 P3->P5
            order = torch.tensor(Hs, device=outs[0].device).argsort(descending=True)
            base = 8
            guessed = torch.tensor(
                [base * (2**i) for i in range(len(outs))], device=outs[0].device, dtype=torch.float32
            )
            self.stride[order] = guessed
            if not self._bias_inited:
                self.bias_init()
                self._bias_inited = True

        # 3) 预测头
        if self.end2end:
            return self.forward_end2end(outs)

        for i in range(self.nl):
            outs[i] = torch.cat((self.cv2[i](outs[i]), self.cv3[i](outs[i])), 1)

        if self.training:
            return outs

        y = self._inference(outs)
        return y if self.export else (y, outs)

    def forward_end2end(self, x):
        x_detach = [xi.detach() for xi in x]
        one2one = [torch.cat((self.cv2[i](x_detach[i]), self.cv3[i](x_detach[i])), 1) for i in range(self.nl)]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return {"one2many": x, "one2one": one2one}
        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        # anchors/strides（注意：这里写入的是 per-location 的张量，保存在 buffers 里，兼容 .to()）
        if self.dynamic or self.shape != shape:
            anchors, strides = make_anchors(x, self.stride, 0.5)
            self.anchors = anchors.transpose(0, 1)  # [2, A] or [nl, ...] -> 按 Ultralytics 用法
            self.strides = strides.transpose(0, 1)
            self.shape = shape

        # split box/cls
        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # decode
        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h, grid_w = shape[2], shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / (float(s) if float(s) > 0 else 8)) ** 2)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, self.stride):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / (float(s) if float(s) > 0 else 8)) ** 2)

    def decode_bboxes(self, bboxes, anchors):
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        batch_size, anchors, _ = preds.shape
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


if __name__ == "__main__":
    imgs = [torch.randn(1, 64, 80, 80), torch.randn(1, 128, 40, 40), torch.randn(1, 256, 20, 20)]
    head = DynamicHead(nc=1, ch=(64, 128, 256))
    head.train()
    outs = head(imgs)
    print([o.shape for o in outs])
    head.eval()
    y, raw = head(imgs)
    print(y.shape)
