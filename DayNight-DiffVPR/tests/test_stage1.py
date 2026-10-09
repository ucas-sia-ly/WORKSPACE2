from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from daynight_diffvpr.evaluate import evaluate
from daynight_diffvpr.manifest import load_positives, load_samples
from daynight_diffvpr.retrieval import chunked_topk, compute_recalls
from daynight_diffvpr.train_vpr import train
from scripts.create_demo import create


def _demo_config(tmp_path):
    root = tmp_path / "demo"
    create(root)
    cfg = yaml.safe_load((Path(__file__).parent.parent / "configs/stage1_demo.yaml").read_text())
    cfg["data"]["image_root"] = str(root)
    for name in ("database_csv", "queries_csv", "positives_csv"):
        cfg["data"][name] = str(root / f"{name.removesuffix('_csv')}.csv")
    return cfg


def test_end_to_end_demo_recall(tmp_path):
    cfg = _demo_config(tmp_path)
    result = evaluate(cfg, tmp_path / "eval", torch.device("cpu"))
    assert result["metrics"]["all"]["R@1"] == 100.0
    assert result["metrics"]["night"]["R@1"] == 100.0
    assert result["metrics"]["day"]["R@1"] == 100.0
    assert (tmp_path / "eval/metrics.json").is_file()
    assert len((tmp_path / "eval/retrieval_topk.csv").read_text().splitlines()) == 8 * 4 + 1


def test_recall_has_denominator_all_queries():
    indices = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    result = compute_recalls(indices, [{0}, {0}], ["night", "night"], ks=(1, 3))
    assert result["night"]["R@1"] == 50.0
    assert result["night"]["R@3"] == 100.0


def test_exact_topk_chunks():
    q = np.array([[1., 0.], [0., 1.]], dtype="float32")
    db = np.array([[1., 0.], [0., 1.], [-1., 0.]], dtype="float32")
    idx, score = chunked_topk(q, db, device=torch.device("cpu"), k=2,
                               query_chunk=1, database_chunk=1)
    assert idx[:, 0].tolist() == [0, 1]
    assert np.allclose(score[:, 0], 1.0)


def test_missing_positives_fail(tmp_path):
    cfg = _demo_config(tmp_path)
    p = Path(cfg["data"]["positives_csv"])
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n")
    db = load_samples(cfg["data"]["database_csv"], cfg["data"]["image_root"])
    q = load_samples(cfg["data"]["queries_csv"], cfg["data"]["image_root"])
    with pytest.raises(ValueError, match="no valid positive"):
        load_positives(p, q, db)


def test_triplet_train_and_load_checkpoint(tmp_path):
    from daynight_diffvpr.models import build_model
    cfg = _demo_config(tmp_path)
    cfg["model"] = {"type": "tiny_cnn", "dim": 16}
    cfg["training"] = {
        "triplets_csv": str(tmp_path / "demo/triplets.csv"),
        "epochs": 1, "batch_size": 2, "accumulate": 2,
        "num_workers": 0, "freeze_modules": [], "lr": 0.001,
    }
    saved = train(cfg, tmp_path / "train", torch.device("cpu"))
    assert saved.is_file()
    cfg["model"]["checkpoint"] = str(saved)
    output = build_model(cfg)(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 16)
    assert torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-4)
