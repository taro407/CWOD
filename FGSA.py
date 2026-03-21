import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics_local import YOLO
from ultralytics_local.data import build_dataloader, build_yolo_dataset
from ultralytics_local.data.utils import check_det_dataset
from ultralytics_local.models.yolo.detect.train import DetectionTrainer


# =========================
# 固定随机种子
# =========================
def seed_everything(seed=9527):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 轻量 Conv 空间补偿模块（无监督）
# =========================
class ConvFeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


# =========================
# FFT + Conv 生成无监督辅助 mask
# =========================
@torch.no_grad()
def build_aux_feat_mask(img, conv_module):
    """img: (B, 3, H, W) target-domain images return: (B, 1, H, W) normalized spatial mask.
    """
    _B, _C, H, W = img.shape
    device = img.device

    # ---- FFT 显著性（仅图像先验）----
    gray = img.mean(1, keepdim=True)

    F_fft = torch.fft.fft2(gray)
    F_shift = torch.fft.fftshift(F_fft, dim=(-2, -1))
    mag = F_shift.abs()
    phase = torch.angle(F_shift)

    log_mag = torch.log1p(mag)

    kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
    log_mag_smooth = F.conv2d(log_mag, kernel, padding=1)

    spectrum = torch.exp(log_mag_smooth + 1j * phase)
    spectrum = torch.fft.ifftshift(spectrum, dim=(-2, -1))
    saliency = torch.fft.ifft2(spectrum).abs()

    # ---- Conv 空间补偿 ----
    conv_feat = conv_module(img).abs()
    conv_feat = conv_feat / (conv_feat.amax(dim=(2, 3), keepdim=True) + 1e-6)

    # ---- 归一化 ----
    def norm(x):
        b = x.shape[0]
        x = x.view(b, -1)
        x = (x - x.min(1, keepdim=True).values) / (x.max(1, keepdim=True).values - x.min(1, keepdim=True).values + 1e-6)
        return x.view(b, 1, H, W)

    saliency = norm(saliency)
    conv_feat = norm(conv_feat)

    mask = saliency * (0.5 + 0.5 * conv_feat)
    mask = norm(mask)

    return torch.nan_to_num(mask)


# =========================
# 自定义 Trainer（严格无监督 target 域）
# =========================
class CustomTrainer(DetectionTrainer):
    def __init__(self, target_domain_data_cfg, hook_film=(16, 19, 22), alpha_film=0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.film_layers = list(hook_film)
        self.alpha_film = alpha_film

        self.conv_feat = ConvFeat().to(self.device)

        self.t_data = check_det_dataset(target_domain_data_cfg)
        self.t_loader = None
        self.t_iter = None

        self._collect_target = False
        self.aux_mask_full = None

        self.add_callback("on_train_start", self._init)
        self.add_callback("on_train_batch_start", self._prepare_target_batch)

    def _init(self, *args, **kwargs):
        self._register_hooks()

        gs = max(int(self.model.stride.max()), 32)
        t_dataset = build_yolo_dataset(
            self.args,
            img_path=self.t_data["train"],
            batch=self.batch_size,
            data=self.t_data,
            mode="train",
            stride=gs,
        )
        self.t_loader = build_dataloader(
            t_dataset,
            batch=self.batch_size,
            workers=self.args.workers,
            shuffle=True,
            rank=-1,
        )
        self.t_iter = iter(self.t_loader)

    def _prepare_target_batch(self, *args, **kwargs):
        try:
            batch_t = next(self.t_iter)
        except StopIteration:
            self.t_iter = iter(self.t_loader)
            batch_t = next(self.t_iter)

        # ⚠️ 只取 image，完全不碰任何 annotation
        batch_t = self.preprocess_batch(batch_t)
        imgs_t = batch_t["img"]

        with torch.no_grad():
            self.aux_mask_full = build_aux_feat_mask(imgs_t, self.conv_feat)

        # forward 一次 target，仅用于触发 hook（不反传）
        self._collect_target = True
        with torch.no_grad():
            _ = self.model(imgs_t)
        self._collect_target = False

    def _register_hooks(self):
        for lid in self.film_layers:
            self.model.model[lid].register_forward_hook(self._make_film_hook())

    def _make_film_hook(self):
        def hook(m, inp, out):
            if self._collect_target:
                return out
            if out.dim() != 4 or self.aux_mask_full is None:
                return out

            B, _C, H, W = out.shape
            mask = F.interpolate(
                self.aux_mask_full[:B],
                (H, W),
                mode="bilinear",
                align_corners=False,
            )

            # zero-mean + scale
            mask = mask - mask.mean(dim=(2, 3), keepdim=True)
            mask = mask / (mask.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)

            return out * (1 + self.alpha_film * mask)

        return hook


# =========================
# main
# =========================
def main():
    seed_everything(9527)

    model = YOLO("ultralytics_local/cfg/models/v3/yolov3.yaml")

    trainer = partial(
        CustomTrainer,
        target_domain_data_cfg="/root/autodl-tmp/pear_split_filtered_1600/target.yaml",
        hook_film=(16, 19, 22),
        alpha_film=0.001,
    )

    model.train(
        trainer,
        data="/root/autodl-tmp/pear_split_filtered_1600/data.yaml",
        name="FiLM_AuxFFT_UNSUP",
        imgsz=640,
        epochs=150,
        batch=64,
        workers=16,
        val=True,
        patience=0,
        cache=True,
        amp=True,
    )


if __name__ == "__main__":
    main()
