from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


class ToyColorDescriptor(nn.Module):
    """Sanity-check-only deterministic model. NOT a VPR baseline."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(2, 3))


class TinyCNN(nn.Module):
    """Sanity-check-only trainable model. NOT competitive VPR."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class DescriptorAdapter(nn.Module):
    """Convert a model's BxD tensor / dict / tuple output to an L2 normalized BxD tensor."""

    def __init__(self, backbone: nn.Module, output_key: str | None = None):
        super().__init__()
        self.backbone = backbone
        self.output_key = output_key

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raw = self.backbone(images)
        if isinstance(raw, dict):
            key = self.output_key or "descriptor"
            if key not in raw:
                raise KeyError(f"Model output dict has keys {list(raw)}; set model.output_key")
            raw = raw[key]
        elif isinstance(raw, (tuple, list)):
            raw = raw[0]
        if not torch.is_tensor(raw) or raw.ndim != 2 or raw.shape[0] != images.shape[0]:
            raise ValueError("VPR model must return a BxD descriptor Tensor")
        return torch.nn.functional.normalize(raw.float(), p=2, dim=1, eps=1e-8)


def _factory_from_string(spec: str):
    if ":" not in spec:
        raise ValueError("model.factory format must be 'module.path:callable_name'")
    module, name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module), name)


def build_model(config: dict[str, Any]) -> DescriptorAdapter:
    spec = config.get("model", {})
    kind = spec.get("type", "existing")
    if spec.get("pythonpath"):
        sys.path.insert(0, str(Path(spec["pythonpath"]).expanduser().resolve()))
    if kind == "existing":
        if not spec.get("factory"):
            raise ValueError("For model.type=existing, provide model.factory, e.g. my_model:build_model")
        raw = _factory_from_string(spec["factory"])(**spec.get("factory_kwargs", {}))
    elif kind == "salad_hub":
        # First call downloads the official code and pretrained weights. Requires network.
        raw = torch.hub.load("serizba/salad", "dinov2_salad", pretrained=True, trust_repo=True)
    elif kind == "toy_color":
        raw = ToyColorDescriptor()
    elif kind == "tiny_cnn":
        raw = TinyCNN(dim=int(spec.get("dim", 32)))
    else:
        raise ValueError(f"Unknown model.type '{kind}'")

    if not isinstance(raw, nn.Module):
        raise TypeError("model.factory must return torch.nn.Module")
    if spec.get("checkpoint"):
        # Load only trusted weights. weights_only avoids arbitrary pickle object unpickling.
        loaded = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            key = spec.get("state_dict_key")
            if key:
                loaded = loaded[key]
            elif "model_state_dict" in loaded:
                loaded = loaded["model_state_dict"]
            elif "state_dict" in loaded:
                loaded = loaded["state_dict"]
        if not isinstance(loaded, dict):
            raise TypeError("Checkpoint must contain a state dictionary")
        prefix = spec.get("strip_prefix", "")
        if prefix:
            loaded = {k.removeprefix(prefix): v for k, v in loaded.items()}
        raw.load_state_dict(loaded, strict=spec.get("strict", True))
    return DescriptorAdapter(raw, output_key=spec.get("output_key"))
