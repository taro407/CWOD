import argparse
import os
import random
from functools import partial

import numpy as np
import torch
from ultralytics_local import YOLO
from ultralytics_local.data import build_dataloader, build_yolo_dataset
from ultralytics_local.data.utils import check_det_dataset
from ultralytics_local.models.yolo.detect.train import DetectionTrainer


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
    return [int(x.strip()) for x in value.split(",") if x.strip()]


class CustomTrainer(DetectionTrainer):
    def __init__(
        self,
        target_domain_data_cfg,
        fsc_alpha=100.0,
        ema_momentum=0.9,
        fsc_layers=(4,),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._target_cfg_path = target_domain_data_cfg
        self.t_data = check_det_dataset(target_domain_data_cfg)
        self.t_iter, self.t_loader = None, None

        self.fsc_alpha = fsc_alpha
        self.ema_momentum = ema_momentum
        self.model_hook_layer_idx = list(fsc_layers)
        self.model_hook_handler = []
        self.running_mu_t = None
        self._collecting_target = False
        self._fsc_logged = False

        self.add_callback("on_train_start", self._init_fsc)
        self.add_callback("on_train_batch_start", self._collect_target_stats_before_source)

    def _init_fsc(self, *args, **kwargs):
        self._activate_hooks(register_only=True)
        dummy = torch.zeros(1, 3, self.args.imgsz, self.args.imgsz, device=self.device)
        _ = self.model(dummy)

        self.running_mu_t = [None for _ in self.model_hook_layer_idx]

        self._deactivate_hooks()
        self._activate_hooks(register_only=False)

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
            rank=-1,
            shuffle=True,
        )
        self.t_iter = iter(self.t_loader)
        print(
            f"[FSC] target loader ready: {len(self.t_loader)} batches | "
            f"layers={self.model_hook_layer_idx} | alpha={self.fsc_alpha}"
        )

    def _collect_target_stats_before_source(self, *args, **kwargs):
        if self.t_loader is None:
            return

        try:
            t_batch = next(self.t_iter)
        except StopIteration:
            self.t_iter = iter(self.t_loader)
            t_batch = next(self.t_iter)

        t_batch = self.preprocess_batch(t_batch)

        self._collecting_target = True
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=self.amp):
            _ = self.model(t_batch)
        self._collecting_target = False

        if not self._fsc_logged:
            print("[FSC] collecting target stats each step. Source features will be corrected online.")
            self._fsc_logged = True

    def _activate_hooks(self, register_only: bool):
        self.model_hook_handler = []
        for i, layer in enumerate(self.model_hook_layer_idx):
            if register_only:
                hook = self.model.model[layer].register_forward_hook(lambda m, inp, out: out)
            else:
                hook = self.model.model[layer].register_forward_hook(self._make_fsc_hook(i))
            self.model_hook_handler.append(hook)

    def _deactivate_hooks(self):
        for hook in self.model_hook_handler:
            hook.remove()
        self.model_hook_handler = []

    def _make_fsc_hook(self, layer_idx: int):
        def _hook(m, inp, out):
            if not isinstance(out, torch.Tensor) or out.dim() != 4:
                return out

            if self._collecting_target:
                with torch.no_grad():
                    mu = out.mean(dim=(0, 2, 3))
                    prev = self.running_mu_t[layer_idx]
                    self.running_mu_t[layer_idx] = (
                        mu.detach()
                        if prev is None
                        else self.ema_momentum * prev + (1.0 - self.ema_momentum) * mu.detach()
                    )
                return out

            mu_t = self.running_mu_t[layer_idx]
            if mu_t is None:
                return out

            mu_s = out.mean(dim=(0, 2, 3))
            delta = (mu_s - mu_t.to(device=out.device, dtype=out.dtype)).view(1, -1, 1, 1)
            return out - self.fsc_alpha * delta

        return _hook


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cfg", required=True)
    parser.add_argument("--source-data", required=True)
    parser.add_argument("--target-data", required=True)
    parser.add_argument("--name", default="DAYOLO-FSC")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9527)
    parser.add_argument("--fsc-alpha", type=float, default=100.0)
    parser.add_argument("--ema-momentum", type=float, default=0.9)
    parser.add_argument("--fsc-layers", type=parse_layers, default=parse_layers("4"))
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)

    model = YOLO(args.model_cfg)
    custom_trainer = partial(
        CustomTrainer,
        target_domain_data_cfg=args.target_data,
        fsc_alpha=args.fsc_alpha,
        ema_momentum=args.ema_momentum,
        fsc_layers=args.fsc_layers,
    )

    model.train(
        custom_trainer,
        data=args.source_data,
        name=args.name,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        val=not args.no_val,
        patience=args.patience,
        cache=not args.no_cache,
        amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
