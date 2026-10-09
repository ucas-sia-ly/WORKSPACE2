from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .images import ImageDataset
from .manifest import Sample


@torch.inference_mode()
def extract_descriptors(model, samples: list[Sample], transform, *, device: torch.device,
                        batch_size: int, num_workers: int, output_file: Path) -> np.ndarray:
    """Write descriptor rows to a .npy memmap so large galleries don't exhaust RAM."""
    loader = DataLoader(ImageDataset(samples, transform), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    output = None
    for images, indices in tqdm(loader, desc=f"Extract {output_file.stem}", leave=False):
        images = images.to(device, non_blocking=True)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext()
        with ctx:
            vectors = model(images)
        vectors = vectors.detach().float().cpu().numpy()
        if output is None:
            output = np.lib.format.open_memmap(output_file, mode="w+", dtype="float32",
                                               shape=(len(samples), vectors.shape[1]))
        output[indices.numpy()] = vectors
    if output is None:
        raise ValueError("No images to extract")
    output.flush()
    return output


@torch.inference_mode()
def chunked_topk(query: np.ndarray, database: np.ndarray, *, k: int = 10,
                 query_chunk: int = 16, database_chunk: int = 8192, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine top-k, streaming gallery chunks with bounded GPU memory."""
    if not (len(database) and len(query)):
        raise ValueError("query/database must be nonempty")
    if query.shape[1] != database.shape[1]:
        raise ValueError("Query/database descriptor dimensions differ")
    if min(k, query_chunk, database_chunk) < 1:
        raise ValueError("k and chunk sizes must be positive")
    k = min(k, len(database))
    all_ix, all_scores = [], []
    for qs in tqdm(range(0, len(query), query_chunk), desc="Search", leave=False):
        q = torch.from_numpy(np.asarray(query[qs:qs + query_chunk])).to(device=device, dtype=torch.float32)
        best_scores = torch.full((len(q), k), -float("inf"), device=device)
        best_indices = torch.full((len(q), k), -1, device=device, dtype=torch.long)
        for ds in range(0, len(database), database_chunk):
            db = torch.from_numpy(np.asarray(database[ds:ds + database_chunk])).to(device=device, dtype=torch.float32)
            score = q @ db.T
            local_k = min(k, len(db))
            s, ix = torch.topk(score, k=local_k, dim=1)
            ix += ds
            candidate_scores = torch.cat((best_scores, s), dim=1)
            candidate_indices = torch.cat((best_indices, ix), dim=1)
            best_scores, positions = candidate_scores.topk(k, dim=1)
            best_indices = candidate_indices.gather(1, positions)
        all_ix.append(best_indices.cpu().numpy())
        all_scores.append(best_scores.cpu().numpy())
    return np.concatenate(all_ix), np.concatenate(all_scores)


def compute_recalls(indices: np.ndarray, positives: list[set[int]], domains: list[str],
                    ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    if len(indices) != len(positives) or len(domains) != len(indices):
        raise ValueError("Indices/positives/domains count mismatch")
    # For exact Recall@K must retrieve at least max(K) gallery items.
    if indices.shape[1] < max(ks):
        raise ValueError("Not enough retrieved columns for requested K")
    if any(not p for p in positives):
        raise ValueError("Cannot evaluate query with no positive")
    groups = {"all": list(range(len(indices)))}
    for i, domain in enumerate(domains):
        groups.setdefault(domain, []).append(i)
    results = {}
    for name, members in groups.items():
        recalls = {}
        for k in ks:
            success = sum(bool(set(indices[i, :k]) & positives[i]) for i in members)
            recalls[f"R@{k}"] = round(100.0 * success / len(members), 3)
        results[name] = {"queries": len(members), **recalls}
    return results


def save_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
