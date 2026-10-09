from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .images import build_transform
from .manifest import load_positives, load_samples
from .models import build_model
from .retrieval import chunked_topk, compute_recalls, extract_descriptors, save_json


def evaluate(config: dict, out: Path, device: torch.device, *, batch_size: int | None = None) -> dict:
    data = config["data"]
    image_root = data["image_root"]
    database = load_samples(data["database_csv"], image_root)
    queries = load_samples(data["queries_csv"], image_root)
    positives = load_positives(data["positives_csv"], queries, database)
    model = build_model(config).to(device).eval()
    transform = build_transform(config)
    infer = config.get("inference", {})
    bs = batch_size or int(infer.get("batch_size", 16))
    db_desc = extract_descriptors(model, database, transform, device=device, batch_size=bs,
                                  num_workers=int(infer.get("num_workers", 4)), output_file=out / "database.npy")
    q_desc = extract_descriptors(model, queries, transform, device=device, batch_size=bs,
                                 num_workers=int(infer.get("num_workers", 4)), output_file=out / "queries.npy")
    ks = tuple(sorted(set(map(int, infer.get("recall_ks", [1, 5, 10])))))
    # Even galleries with < max(K) images can be tested: R@10 effectively R@N when N<10.
    effective_ks = tuple(min(k, len(database)) for k in ks)
    topk, score = chunked_topk(q_desc, db_desc, k=max(effective_ks),
                      query_chunk=int(infer.get("query_chunk", 16)),
                      database_chunk=int(infer.get("database_chunk", 8192)), device=device)
    metrics_raw = compute_recalls(topk, positives, [s.domain for s in queries], ks=effective_ks)
    metrics = {domain: {"queries": v["queries"], **{f"R@{req_k}": v[f"R@{eff_k}"]
               for req_k, eff_k in zip(ks, effective_ks)}} for domain, v in metrics_raw.items()}
    with (out / "retrieval_topk.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query_domain", "rank", "db_id", "cosine_similarity", "is_positive"])
        for i, q in enumerate(queries):
            for rank, j in enumerate(topk[i], start=1):
                writer.writerow([q.id, q.domain, rank, database[int(j)].id, float(score[i, rank-1]), int(int(j) in positives[i])])
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "model": config["model"], "database_count": len(database), "query_count": len(queries),
              "descriptor_dimension": int(db_desc.shape[1]), "metrics": metrics,
              "notes": "Recall@K defined by positives.csv; K>N_database is treated as K=N_database."}
    save_json(out / "metrics.json", result)
    print(yaml.safe_dump(metrics, sort_keys=False, allow_unicode=True))
    return result


def main():
    parser = argparse.ArgumentParser(description="Stage-1 exact VPR Recall@K evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/stage1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    evaluate(config, out, torch.device(args.device), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
