"""Shared image preprocessing and tensor-safe diffusion utilities.

Heavy Diffusers imports are intentionally lazy; Stage-1 does not need them.
"""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

DEFAULT_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_CONTROLNET = "lllyasviel/sd-controlnet-canny"
DEFAULT_PROMPT = (
    "realistic photograph of the same urban street at night, natural street lighting, "
    "dark sky, real nighttime illumination, buildings and road unchanged, documentary photograph"
)
DEFAULT_NEGATIVE = "different buildings, warped geometry, extra buildings, surreal, cartoon, daytime, oversaturated"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_edge(image: Image.Image, width: int, height: int, low: int = 100, high: int = 200) -> Image.Image:
    """RGB Canny control image; Pillow fallback allows CPU unit tests without OpenCV."""
    rgb = image.convert('RGB').resize((width, height), Image.Resampling.BILINEAR)
    try:
        import cv2
        array = np.asarray(rgb)
        edges = cv2.Canny(cv2.cvtColor(array, cv2.COLOR_RGB2GRAY), low, high)
    except ImportError:
        edges = np.asarray(rgb.convert('L').filter(ImageFilter.FIND_EDGES))
        edges = np.where(edges > high, 255, 0).astype(np.uint8)
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))


def edge_iou(first: Image.Image, second: Image.Image, low: int = 100, high: int = 200) -> float:
    """Heuristic only: nighttime lighting changes edges; do not use as place ground truth."""
    size = first.size
    a = np.asarray(pil_edge(first, *size, low=low, high=high).convert('L')) > 0
    b = np.asarray(pil_edge(second, *size, low=low, high=high).convert('L')) > 0
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def tensor_canny(images: torch.Tensor, low: int = 100, high: int = 200) -> torch.Tensor:
    """Convert unnormalized Bx3xHxW RGB in [0,1] into frozen ControlNet conditions."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError('Expected Bx3xHxW')
    device = images.device
    data = []
    for i in range(images.shape[0]):
        im = Image.fromarray((images[i].detach().float().clamp(0,1).cpu().permute(1,2,0).numpy()*255).astype('uint8'))
        edge = pil_edge(im, im.width, im.height, low, high)
        data.append(torch.from_numpy(np.asarray(edge).copy()).permute(2,0,1).float()/255)
    return torch.stack(data).to(device)


def differentiable_vpr_input(image: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Differentiable RGB [0,1] -> existing VPR preprocessing (resizing and normalization)."""
    size = cfg.get('data', {}).get('image_size', [322,322])
    norm = cfg.get('data', {}).get('normalize', 'imagenet')
    image = F.interpolate(image.float(), size=tuple(int(x) for x in size), mode='bilinear', align_corners=False)
    if norm == 'imagenet':
        mean = image.new_tensor([.485,.456,.406]).view(1,3,1,1)
        std = image.new_tensor([.229,.224,.225]).view(1,3,1,1)
        return (image-mean)/std
    if norm == 'none':
        return image
    raise ValueError('data.normalize must be imagenet or none')


def predict_x0(noisy: torch.Tensor, noise_pred: torch.Tensor, alpha_cumprod: torch.Tensor) -> torch.Tensor:
    """DDPM epsilon prediction: x0=(xt-sqrt(1-alpha)*eps)/sqrt(alpha). Keeps gradients."""
    a = alpha_cumprod.to(device=noisy.device, dtype=noisy.dtype)
    while a.ndim < noisy.ndim:
        a = a.unsqueeze(-1)
    return (noisy - (1-a).sqrt()*noise_pred)/a.sqrt().clamp(min=1e-5)


def soft_edge_map(image: torch.Tensor) -> torch.Tensor:
    """Lighting-robust (not lighting-invariant) differentiable image gradient magnitude."""
    grey = image.float().mean(dim=1, keepdim=True)
    kernel_x = grey.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3)/8
    kernel_y = kernel_x.transpose(2,3)
    dx = F.conv2d(grey, kernel_x, padding=1)
    dy = F.conv2d(grey, kernel_y, padding=1)
    edge = (dx.square()+dy.square()+1e-6).sqrt()
    return edge / edge.mean(dim=(-2,-1), keepdim=True).clamp(min=1e-4)


def structure_loss(day: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    ref = F.interpolate(day.float(), size=generated.shape[-2:], mode='bilinear', align_corners=False)
    return F.smooth_l1_loss(soft_edge_map(ref), soft_edge_map(generated))


def illumination_loss(generated: torch.Tensor, real_night: torch.Tensor) -> torch.Tensor:
    """Global night color/contrast statistics (no assumption of pixel-perfect registration)."""
    mean_g, mean_n = generated.mean(dim=(-1,-2)), real_night.mean(dim=(-1,-2))
    std_g, std_n = generated.std(dim=(-1,-2)), real_night.std(dim=(-1,-2))
    return F.l1_loss(mean_g,mean_n)+F.l1_loss(std_g,std_n)


def save_lora_weights(unet, path: str | Path) -> None:
    """Save standard Diffusers pipeline-loadable PEFT UNet LoRA weights."""
    from peft import get_peft_model_state_dict
    from diffusers.utils import convert_state_dict_to_diffusers
    from diffusers import StableDiffusionControlNetImg2ImgPipeline
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionControlNetImg2ImgPipeline.save_lora_weights(
        str(path), unet_lora_layers=state, safe_serialization=True)
