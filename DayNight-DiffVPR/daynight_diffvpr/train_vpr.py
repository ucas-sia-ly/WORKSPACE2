"""Optional Stage-1 VPR fine-tuning on supervised (anchor,positive,negative) triples.

NOT diffusion training. For 24GB cards start with freeze_modules=["backbone"].
"""
from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

from .images import TripletDataset, build_transform
from .models import build_model


def train(config: dict, output: Path, device: torch.device) -> Path:
    cfg = config.get("training", {})
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_model(config).to(device)
    for module_name in cfg.get("freeze_modules", ["backbone"]):
        module = model.backbone
        for component in module_name.split("."):
            if not hasattr(module, component):
                raise AttributeError(f"Cannot freeze '{module_name}': missing '{component}'. Edit training.freeze_modules")
            module = getattr(module, component)
        for p in module.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters remain: change training.freeze_modules")
    data = config["data"]
    dataset = TripletDataset(cfg["triplets_csv"], data["image_root"], build_transform(config))
    loader = DataLoader(dataset, shuffle=True, batch_size=int(cfg.get("batch_size", 2)),
                        num_workers=int(cfg.get("num_workers", 4)),
                        pin_memory=device.type == "cuda", drop_last=False)
    optimizer = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-4)),
                                  weight_decay=float(cfg.get("weight_decay", 1e-4)))
    epochs = int(cfg.get("epochs", 2))
    accum = int(cfg.get("accumulate", 4))
    if accum < 1 or epochs < 1:
        raise ValueError("epochs and accumulate must be >=1")
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.jsonl"
    for epoch in range(epochs):
        model.train()
        # Freeze BN/dropout in frozen submodules for consistency.
        for module_name in cfg.get("freeze_modules", ["backbone"]):
            module = model.backbone
            for component in module_name.split("."):
                module = getattr(module, component)
            module.eval()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, (anchor, positive, negative) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")):
            batch = torch.cat((anchor, positive, negative), dim=0).to(device, non_blocking=True)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
            with ctx:
                features = model(batch)
                a, p, n = features.chunk(3, dim=0)
                # Larger positive cosine similarity and lower negative similarity.
                sim_pos = (a * p).sum(dim=1)
                sim_neg = (a * n).sum(dim=1)
                loss = F.softplus(sim_neg - sim_pos + float(cfg.get("margin", 0.2))).mean()
            (loss / accum).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(cfg.get("grad_clip", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch + 1, "triplet_loss": float(np.mean(losses)), "num_triplets": len(dataset)}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(row)
        # Save ONLY the native model weights, not the DescriptorAdapter's 'backbone.' prefix.
        checkpoint = output / "vpr_last.pth"
        torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
                    "epoch": epoch + 1}, checkpoint)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Optional Stage-1 triplet VPR fine-tuning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/stage1_train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print("Saved checkpoint:", train(config, Path(args.output), torch.device(args.device)))


if __name__ == "__main__":
    main()
