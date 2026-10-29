"""Fail CI when the committed question bank loses required records or fields."""

import csv
from pathlib import Path


def main():
    path = Path(__file__).resolve().parents[1] / "data/benchmark/golden_dataset.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = ("id", "question", "ground_truth", "question_type")
    if len(rows) != 30:
        raise ValueError(f"Expected the documented 30-question bank, got {len(rows)}")
    if any(not str(row.get(key, "")).strip() for row in rows for key in required):
        raise ValueError(
            "Each question requires id, question, ground_truth and question_type"
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Question IDs must be unique")
    print(f"Validated {len(rows)} questions; no answer accuracy measured.")


if __name__ == "__main__":
    main()
