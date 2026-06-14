import argparse
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_yolo_dataset, build_dataloader


def seed_everything(seed=9527):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_layers(value):
    if value is None or value.strip() == "":
        return tuple()
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


class ConvFeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


@torch.no_grad()
def normalize_map(x, height, width):
    batch = x.shape[0]
    x = x.view(batch, -1)
    mn = x.min(1, keepdim=True).values
    mx = x.max(1, keepdim=True).values
    x = (x - mn) / (mx - mn + 1e-6)
    return x.view(batch, 1, height, width)


@torch.no_grad()
def build_aux_feat_mask(img, conv_module):
    _, _, height, width = img.shape
    device = img.device

    gray = img.mean(1, keepdim=True)

    fft = torch.fft.fft2(gray)
    fft = torch.fft.fftshift(fft, dim=(-2, -1))
    magnitude = fft.abs()
    phase = torch.angle(fft)

    log_magnitude = torch.log1p(magnitude)
    kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
    spectrum_base = F.conv2d(log_magnitude, kernel, padding=1)

    spectrum = torch.exp(spectrum_base + 1j * phase)
    spectrum = torch.fft.ifftshift(spectrum, dim=(-2, -1))
    saliency = torch.fft.ifft2(spectrum).abs()

    conv_feat = conv_module(img).abs()
    conv_feat = conv_feat / (conv_feat.amax(dim=(2, 3), keepdim=True) + 1e-6)

    saliency = normalize_map(saliency, height, width)
    conv_feat = normalize_map(conv_feat, height, width)

    mask = saliency * (0.5 + 0.5 * conv_feat)
    mask = normalize_map(mask, height, width)
    return torch.nan_to_num(mask)


@torch.no_grad()
def build_fruit_mask_from_batch(batch, img_shape, device):
    batch_size, _, height, width = img_shape
    fruit_mask = torch.zeros((batch_size, 1, height, width), device=device)

    if "bboxes" not in batch or "batch_idx" not in batch:
        return fruit_mask

    bboxes = batch["bboxes"].to(device)
    batch_idx = batch["batch_idx"].long().to(device)

    if bboxes.numel() == 0:
        return fruit_mask

    if bboxes.max() <= 1.5:
        cx = bboxes[:, 0] * width
        cy = bboxes[:, 1] * height
        bw = bboxes[:, 2] * width
        bh = bboxes[:, 3] * height
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
    else:
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    x1 = x1.clamp(0, width - 1)
    x2 = x2.clamp(0, width - 1)
    y1 = y1.clamp(0, height - 1)
    y2 = y2.clamp(0, height - 1)

    for i in range(bboxes.shape[0]):
        b = int(batch_idx[i].item())
        if b < 0 or b >= batch_size:
            continue

        xi1 = int(x1[i].item())
        yi1 = int(y1[i].item())
        xi2 = int(x2[i].item())
        yi2 = int(y2[i].item())

        if xi2 <= xi1 or yi2 <= yi1:
            continue

        fruit_mask[b, 0, yi1:yi2, xi1:xi2] = 1.0

    kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
    fruit_mask = F.conv2d(fruit_mask, kernel, padding=1)
    return fruit_mask.clamp(0.0, 1.0)


@torch.no_grad()
def combine_aux_fruit_mask(aux_mask, fruit_mask, fruit_weight=1.0, bg_weight=0.1):
    batch_size, _, height, width = aux_mask.shape

    if fruit_mask is None or fruit_mask.shape != aux_mask.shape:
        base = aux_mask
    else:
        fruit_mask = fruit_mask.clamp(0.0, 1.0)
        weight = bg_weight + (fruit_weight - bg_weight) * fruit_mask
        base = aux_mask * weight

    mask = base.view(batch_size, -1)
    mn = mask.min(1, keepdim=True).values
    mx = mask.max(1, keepdim=True).values
    mask = (mask - mn) / (mx - mn + 1e-6)
    mask = mask.view(batch_size, 1, height, width)

    return torch.nan_to_num(mask)


class CustomTrainer(DetectionTrainer):
    def __init__(
        self,
        target_domain_data_cfg,
        hook_cmm=(16, 19),
        alpha_cmm=0.1,
        fruit_weight=1.0,
        bg_weight=0.1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.cmm_layers = list(hook_cmm)
        self.alpha_cmm = alpha_cmm
        self.fruit_weight = fruit_weight
        self.bg_weight = bg_weight

        self.conv_feat = ConvFeat().to(self.device).eval()
        for param in self.conv_feat.parameters():
            param.requires_grad_(False)

        self.t_data = check_det_dataset(target_domain_data_cfg)
        self.t_loader = None
        self.t_iter = None
        self.aux_mask_full = None

        self.add_callback("on_train_start", self._init_cmm)
        self.add_callback("on_train_batch_start", self._prepare_target_mask)

    def _init_cmm(self, *args, **kwargs):
        self._register_cmm_hooks()

        stride = max(int(self.model.stride.max()), 32)
        target_dataset = build_yolo_dataset(
            self.args,
            img_path=self.t_data["train"],
            batch=self.batch_size,
            data=self.t_data,
            mode="train",
            stride=stride,
        )
        self.t_loader = build_dataloader(
            target_dataset,
            batch=self.batch_size,
            workers=self.args.workers,
            shuffle=True,
            rank=-1,
        )
        self.t_iter = iter(self.t_loader)

        print(
            f"[CMM] target loader ready: {len(self.t_loader)} batches | "
            f"layers={self.cmm_layers} | alpha={self.alpha_cmm}"
        )

    def _prepare_target_mask(self, *args, **kwargs):
        if self.t_loader is None:
            return

        try:
            target_batch = next(self.t_iter)
        except StopIteration:
            self.t_iter = iter(self.t_loader)
            target_batch = next(self.t_iter)

        target_batch = self.preprocess_batch(target_batch)
        target_imgs = target_batch["img"]

        with torch.no_grad():
            aux_mask = build_aux_feat_mask(target_imgs, self.conv_feat)
            fruit_mask = build_fruit_mask_from_batch(target_batch, target_imgs.shape, self.device)
            self.aux_mask_full = combine_aux_fruit_mask(
                aux_mask,
                fruit_mask,
                fruit_weight=self.fruit_weight,
                bg_weight=self.bg_weight,
            )

    def _register_cmm_hooks(self):
        for layer_idx in self.cmm_layers:
            self.model.model[layer_idx].register_forward_hook(self._make_cmm_hook(layer_idx))

    def _make_cmm_hook(self, layer_idx):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor) or output.dim() != 4 or self.aux_mask_full is None:
                return output

            batch_size, channels, height, width = output.shape
            mask = self.aux_mask_full.to(device=output.device, dtype=output.dtype)
            target_batch_size = mask.shape[0]

            if target_batch_size == 1:
                mask = mask.expand(batch_size, -1, -1, -1)
            elif target_batch_size >= batch_size:
                mask = mask[:batch_size]
            else:
                indices = torch.randint(0, target_batch_size, (batch_size,), device=output.device)
                mask = mask[indices]

            mask = F.interpolate(mask, (height, width), mode="bilinear", align_corners=False)
            mask = mask - mask.mean(dim=(2, 3), keepdim=True)
            mask = mask / (mask.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)
            mask = mask.expand(-1, channels, -1, -1)

            return output * (1 + self.alpha_cmm * mask)

        return hook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cfg", type=str, required=True)
    parser.add_argument("--source-data", type=str, required=True)
    parser.add_argument("--target-data", type=str, required=True)
    parser.add_argument("--name", type=str, default="YOLO-CMM")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9527)
    parser.add_argument("--cmm-layers", type=parse_layers, default=(16, 19))
    parser.add_argument("--alpha-cmm", type=float, default=0.1)
    parser.add_argument("--fruit-weight", type=float, default=1.0)
    parser.add_argument("--bg-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    model = YOLO(args.model_cfg)

    trainer = partial(
        CustomTrainer,
        target_domain_data_cfg=args.target_data,
        hook_cmm=args.cmm_layers,
        alpha_cmm=args.alpha_cmm,
        fruit_weight=args.fruit_weight,
        bg_weight=args.bg_weight,
    )

    model.train(
        trainer,
        data=args.source_data,
        name=args.name,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        val=args.val,
        patience=args.patience,
        cache=args.cache,
        amp=args.amp,
    )


if __name__ == "__main__":
    main()
