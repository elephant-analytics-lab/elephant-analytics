from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict, key: str) -> float | None:
    val = payload.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two benchmark JSON reports.")
    parser.add_argument("--base", required=True, help="Baseline report path.")
    parser.add_argument("--candidate", required=True, help="Candidate report path.")
    args = parser.parse_args()

    base = _load(Path(args.base).resolve())
    cand = _load(Path(args.candidate).resolve())

    base_ratio = _metric(base, "mean_processing_ratio")
    cand_ratio = _metric(cand, "mean_processing_ratio")
    base_sec = _metric(base, "mean_processing_seconds")
    cand_sec = _metric(cand, "mean_processing_seconds")

    verdict = "inconclusive"
    if base_ratio is not None and cand_ratio is not None:
        if cand_ratio < base_ratio:
            verdict = "candidate_faster"
        elif cand_ratio > base_ratio:
            verdict = "candidate_slower"
        else:
            verdict = "same_speed"

    out = {
        "base_report": str(Path(args.base).resolve()),
        "candidate_report": str(Path(args.candidate).resolve()),
        "base_mean_processing_ratio": base_ratio,
        "candidate_mean_processing_ratio": cand_ratio,
        "base_mean_processing_seconds": base_sec,
        "candidate_mean_processing_seconds": cand_sec,
        "delta_ratio": (cand_ratio - base_ratio) if (base_ratio is not None and cand_ratio is not None) else None,
        "delta_seconds": (cand_sec - base_sec) if (base_sec is not None and cand_sec is not None) else None,
        "verdict": verdict,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
