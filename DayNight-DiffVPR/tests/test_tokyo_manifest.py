from pathlib import Path

from PIL import Image

from scripts.prepare_tokyo247 import prepare, utm_from_name


def test_parse_utm_and_prepare(tmp_path):
    root = tmp_path / "tokyo247"
    db = root / "images/test/database"; db.mkdir(parents=True)
    q = root / "images/test/queries"; q.mkdir(parents=True)
    d = db / "@300000@3900000@54@S@a.png"
    query = q / "@300010@3900010@54@S@b.png"
    Image.new("RGB", (8, 8)).save(d)
    Image.new("RGB", (8, 8)).save(query)
    assert utm_from_name(query) == (300010., 3900010., "54S")
    out = tmp_path / "manifests"
    prepare(root, out, 25)
    assert (out / "positives.csv").read_text().strip().splitlines()[1] == "q_000000,db_000000"
    assert "unknown" in (out / "queries.csv").read_text()
