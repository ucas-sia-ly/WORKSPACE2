"""Manifest builder for ALREADY FORMATTED Tokyo24/7 from VPR-datasets-downloader.

This does NOT download the dataset. Images require @EASTING@NORTHING@ZONE@ZONELETTER... names.
All positive references are within radius_m of a query in the same UTM zone.
Query domain labels are explicit from an optional CSV; otherwise 'unknown'.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def utm_from_name(image: Path) -> tuple[float, float, str]:
    fields = image.stem.split("@")
    if len(fields) < 5 or fields[0] != "":
        raise ValueError(f"UTM-formatted image name expected: {image.name}")
    try:
        easting, northing = float(fields[1]), float(fields[2])
        zone = f"{int(fields[3])}{fields[4].upper()}"
    except (IndexError, ValueError) as ex:
        raise ValueError(f"Bad UTM fields in {image.name}") from ex
    return easting, northing, zone


def discover(folder: Path) -> list[Path]:
    images = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images in {folder}")
    return images


def load_domain_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"filename", "domain"}.issubset(reader.fieldnames or []):
            raise ValueError("domain CSV needs filename,domain")
        mapping = {}
        for r in reader:
            domain = r["domain"].strip().lower()
            if domain not in {"day", "night", "evening", "unknown"}:
                raise ValueError(f"Unexpected domain label: {domain}")
            mapping[r["filename"]] = domain
        return mapping


def prepare(root: Path, out: Path, radius: float, domain_csv: str | None = None):
    db_images = discover(root / "images/test/database")
    q_images = discover(root / "images/test/queries")
    db_utm = [utm_from_name(p) for p in db_images]
    q_utm = [utm_from_name(p) for p in q_images]
    domain_map = load_domain_map(domain_csv)
    out.mkdir(parents=True, exist_ok=True)
    def write(name, header, rows):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)

    write("database.csv", ["id", "path", "domain"],
          [[f"db_{i:06d}", str(p.relative_to(root)), "day"] for i,p in enumerate(db_images)])
    write("queries.csv", ["id", "path", "domain"],
          [[f"q_{i:06d}", str(p.relative_to(root)), domain_map.get(p.name, "unknown")]
           for i,p in enumerate(q_images)])
    locations = np.array([[e, n] for e, n, _ in db_utm], dtype=np.float64)
    zones = np.array([z for _, _, z in db_utm])
    positives = []
    missing = []
    for i, (e, n, z) in enumerate(q_utm):
        squared = ((locations - np.array([e, n])) ** 2).sum(axis=1)
        hits = np.flatnonzero((squared <= radius**2) & (zones == z))
        if len(hits) == 0:
            missing.append(q_images[i].name)
        for j in hits:
            positives.append([f"q_{i:06d}", f"db_{j:06d}"])
    if missing:
        # Never fabricate GT or silently exclude hard queries.
        raise ValueError(f"{len(missing)} queries have no db image within {radius}m; first {missing[:3]}. "
                         "Inspect dataset coordinates/ground truth. Partial CSVs were generated but cannot be used as-is.")
    write("positives.csv", ["query_id", "db_id"], positives)
    print(f"Ready: {len(db_images)} database, {len(q_images)} query images, {len(positives)} positive pairs")
    print(f"CSV directory: {out}")
    if not domain_map:
        print("NOTE: all query domains = 'unknown'; provide --query-domains CSV to get night-only results")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Tokyo247 root with images/test/database and images/test/queries")
    p.add_argument("--output", default="data/tokyo247_manifests")
    p.add_argument("--radius-m", type=float, default=25.0)
    p.add_argument("--query-domains", help="CSV: filename,domain; explicit day/night labels")
    a = p.parse_args()
    prepare(Path(a.root).resolve(), Path(a.output), a.radius_m, a.query_domains)
