from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    id: str
    path: Path
    domain: str = "unknown"


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} needs columns {sorted(required)}; got {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def load_samples(csv_path: str | Path, image_root: str | Path) -> list[Sample]:
    image_root = Path(image_root).expanduser().resolve()
    result = []
    seen: set[str] = set()
    for row in _read_csv(Path(csv_path), {"id", "path"}):
        sid = row["id"].strip()
        if not sid or sid in seen:
            raise ValueError(f"Duplicate or empty sample id '{sid}' in {csv_path}")
        seen.add(sid)
        rel = Path(row["path"].strip()).expanduser()
        path = (image_root / rel).resolve() if not rel.is_absolute() else rel.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image for id={sid} not found: {path}")
        result.append(Sample(sid, path, (row.get("domain") or "unknown").strip().lower()))
    return result


def load_positives(path: str | Path, queries: list[Sample], database: list[Sample]) -> list[set[int]]:
    db_map = {s.id: i for i, s in enumerate(database)}
    q_map = {s.id: i for i, s in enumerate(queries)}
    positives: list[set[int]] = [set() for _ in queries]
    for row in _read_csv(Path(path), {"query_id", "db_id"}):
        qid, dbid = row["query_id"].strip(), row["db_id"].strip()
        if qid not in q_map or dbid not in db_map:
            raise ValueError(f"Invalid positive pair ({qid}, {dbid}): unknown image id")
        positives[q_map[qid]].add(db_map[dbid])
    missing = [queries[i].id for i, pos in enumerate(positives) if not pos]
    if missing:
        raise ValueError(f"{len(missing)} queries have no valid positive in the database; first: {missing[:5]}")
    return positives
