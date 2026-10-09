"""Generate synthetic colored squares for pipeline sanity checks (NOT VPR research data)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = [(220, 35, 35), (25, 220, 35), (35, 35, 230), (225, 160, 35)]


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(headers); writer.writerows(rows)


def create(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    db, q, pos, triplets = [], [], [], []
    for i, col in enumerate(COLORS):
        name = f"place_{i}"
        day = f"{name}_day.png"
        night = f"{name}_night.png"
        for filename, rgb, domain in [
            (day, col, "day"),
            (night, tuple(max(1, int(c * .34)) for c in col), "night")
        ]:
            im = Image.new("RGB", (128, 96), rgb)
            ImageDraw.Draw(im).rectangle([15, 10, 30, 25], outline="white" if domain == "day" else (65, 65, 65))
            im.save(root / filename)
        db.append([name, day, "day"])
        for domain, file in [("day", day), ("night", night)]:
            query_id = f"{name}_{domain}"
            q.append([query_id, file, domain])
            pos.append([query_id, name])
        triplets.append([night, day, f"place_{(i + 1) % len(COLORS)}_day.png"])
    write_csv(root / "database.csv", ["id", "path", "domain"], db)
    write_csv(root / "queries.csv", ["id", "path", "domain"], q)
    write_csv(root / "positives.csv", ["query_id", "db_id"], pos)
    write_csv(root / "triplets.csv", ["anchor", "positive", "negative"], triplets)
    print(f"Synthetic demo images + CSV ready: {root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/demo")
    args = p.parse_args()
    create(Path(args.output))
