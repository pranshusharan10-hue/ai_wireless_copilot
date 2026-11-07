"""Validate and mirror Phase 5 benchmark dataset (78 cases across 8 categories).

Verifies integrity against BenchmarkDataset schema and ensures tests/fixtures
remains synchronized with data/benchmark/phase5_benchmark.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.schema import BenchmarkDataset  # noqa: E402

CANONICAL_PATH = ROOT_DIR / "data" / "benchmark" / "phase5_benchmark.json"
FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "phase5_benchmark.json"


def validate_and_sync_benchmark() -> BenchmarkDataset:
    """Load canonical benchmark dataset, validate schema, and mirror to fixtures."""
    if not CANONICAL_PATH.is_file():
        raise FileNotFoundError(f"Canonical benchmark missing at {CANONICAL_PATH}")

    with CANONICAL_PATH.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset = BenchmarkDataset.model_validate(raw_data)

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    return dataset


if __name__ == "__main__":
    ds = validate_and_sync_benchmark()
    print(f"Validated {ds.total_cases} cases across {len(ds.categories)} categories.")
    cat_counts: dict[str, int] = {}
    for case in ds.cases:
        cat_counts[case.category] = cat_counts.get(case.category, 0) + 1

    for cat in sorted(cat_counts.keys()):
        print(f"  {cat}: {cat_counts[cat]} cases")
