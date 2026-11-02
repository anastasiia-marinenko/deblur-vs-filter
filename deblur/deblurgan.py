"""DeblurGAN-v2 wrapper.

Official code: https://github.com/VITA-Group/DeblurGANv2 (cloned by the notebook),
authors' weights: fpn_inception.h5 (heavy, 61 M params) and fpn_mobilenet.h5 (light, 3.3 M).
We do not train the network, only run it as published.

Differences from their predict.py (no effect on the output):
  - runs on CPU and GPU (the original always calls .cuda())
  - does not download ImageNet backbone weights (they are overwritten by the .h5 anyway)
  - takes and returns a grayscale image, copied to 3 channels
model.train(True) is kept as in the original: normalisation uses statistics of the current image.
"""
from __future__ import annotations

import functools
import os
import sys

import numpy as np


class DeblurGANv2:
    def __init__(self, repo_dir: str, weights_path: str, backbone: str = "fpn_inception",
                 device: str | None = None):
        import torch
        import torch.nn as nn

        # so we can import the authors' models
        repo_dir = os.path.abspath(repo_dir)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        # InstanceNorm, as in the authors' config
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)

        if backbone == "fpn_inception":
            # heavy: Inception-ResNet-v2 backbone + FPN
            import models.fpn_inception as fi
            from pretrainedmodels import inceptionresnetv2 as _irv2
            # build the backbone without ImageNet weights, the .h5 replaces them anyway
            fi.inceptionresnetv2 = lambda num_classes=1000, pretrained="imagenet": _irv2(
                num_classes=num_classes, pretrained=None)
            model = fi.FPNInception(norm_layer=norm_layer)
        elif backbone == "fpn_mobilenet":
            # light: MobileNet-v2 backbone
            from models.fpn_mobilenet import FPNMobileNet
            model = FPNMobileNet(norm_layer=norm_layer, pretrained=False)
        else:
            raise ValueError(backbone)

        if weights_path:
            # authors' weights; drop the "module." prefix left by multi-GPU training
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = ckpt["model"] if "model" in ckpt else ckpt
            state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
            model.load_state_dict(state)

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.train(True)
        self.backbone = backbone

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def __call__(self, gray: np.ndarray) -> np.ndarray:
        torch = self.torch
        # trained on colour photos: copy gray to 3 channels, scale 0..255 to -1..1
        x = np.repeat(gray.astype(np.float32)[..., None], 3, axis=2) / 255.0
        x = (x - 0.5) / 0.5
        # sizes must be multiples of 32, pad with zeros
        h, w, _ = x.shape
        H = (h // 32 + 1) * 32
        W = (w // 32 + 1) * 32
        x = np.pad(x, ((0, H - h), (0, W - w), (0, 0)), mode="constant")
        # (1, 3, H, W)
        t = torch.from_numpy(np.transpose(x, (2, 0, 1))[None]).to(self.device)
        with torch.no_grad():
            y = self.model(t)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()             # wait for the GPU, for honest timing
        y = y[0].detach().cpu().float().numpy()
        # back to 0..255, cut the padding, average the 3 channels
        y = (np.transpose(y, (1, 2, 0)) + 1.0) / 2.0 * 255.0
        return np.clip(y[:h, :w, :].mean(axis=2), 0, 255)