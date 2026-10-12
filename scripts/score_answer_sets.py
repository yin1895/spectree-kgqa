#!/usr/bin/env python3
"""Compute set-Hit, macro-F1 and set-EM from canonical answer-ID sets.

JSONL input: {"question_id": "q1", "prediction": ["entity1"],
              "reference": ["entity1", "entity2"]}.
A failed prediction is represented by [] (or null), never by a missing row.
Benchmark-specific alias/entity normalization must be performed upstream.
Reads canonical answer sets and reports aggregate set-level metrics.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_set(value: Any, field: str, *, allow_null: bool = False) -> set[str]:
    if value is None and allow_null:
        return set()
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"{field} must be a list of nonempty canonical string IDs")
    return set(value)


def score_records(records: Iterable[dict[str, Any]], expected_ids: set[str] | None = None) -> dict[str, Any]:
    seen: set[str] = set()
    hit_count = exact_count = 0
    f1_sum = 0.0
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Each JSONL row must be an object")
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError("Missing, invalid or duplicate question_id")
        if "prediction" not in row or "reference" not in row:
            raise ValueError(f"{qid}: both prediction and reference fields are required")
        prediction = canonical_set(row["prediction"], "prediction", allow_null=True)
        reference = canonical_set(row["reference"], "reference")
        if not reference:
            raise ValueError(f"{qid}: this protocol requires nonempty reference answers")
        seen.add(qid)
        overlap = len(prediction & reference)
        hit_count += int(overlap > 0)
        exact_count += int(prediction == reference)
        f1_sum += 2.0 * overlap / (len(prediction) + len(reference))
    if not seen:
        raise ValueError("No question records supplied")
    if expected_ids is not None and seen != expected_ids:
        missing = sorted(expected_ids - seen)
        extra = sorted(seen - expected_ids)
        raise ValueError(f"Question coverage mismatch: {len(missing)} missing, {len(extra)} extra")
    n = len(seen)
    return {"n": n, "hit_count": hit_count, "exact_count": exact_count,
            "set_hit_pct": 100.0 * hit_count / n,
            "macro_f1_pct": 100.0 * f1_sum / n,
            "set_exact_match_pct": 100.0 * exact_count / n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--expected-ids", required=True, type=Path,
                        help="JSON array of every expected test question ID")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected = json.loads(args.expected_ids.read_text(encoding="utf-8"))
    if not isinstance(expected, list) or any(not isinstance(x, str) or not x for x in expected):
        parser.error("Expected IDs must be a JSON array of nonempty strings")
    if len(set(expected)) != len(expected):
        parser.error("Expected ID manifest contains duplicates")
    try:
        with args.predictions.open(encoding="utf-8") as file:
            rows = (json.loads(line) for line in file if line.strip())
            result = score_records(rows, set(expected))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
