"""Generation stability: same input, same model — how much does quality swing?

Production buyers care about consistent delivery, not just the median.
With ``--repeats N`` the harness generates each case multiple times; this
script groups scores by (provider, case_id) and reports the within-case
spread (max - min production score across repeats). A model that swings
40 points on the same reference image is a lottery, whatever its median.

Usage:
    python -m src.analysis.stability results/scores.json [more ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import median


def run(paths: list[str]) -> dict[str, dict]:
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in paths:
        with open(path) as fh:
            for r in json.load(fh):
                if r.get("production_score") is not None:
                    by[(r["provider"], r["case_id"])].append(float(r["production_score"]))
    spreads: dict[str, list[float]] = defaultdict(list)
    for (prov, _case), vals in by.items():
        if len(vals) >= 2:
            spreads[prov].append(max(vals) - min(vals))
    out: dict[str, dict] = {}
    for prov, sp in sorted(spreads.items()):
        out[prov] = {"n_cases_with_repeats": len(sp),
                     "median_spread": round(median(sp), 1),
                     "max_spread": round(max(sp), 1),
                     "frac_spread_gt10": round(sum(1 for s in sp if s > 10) / len(sp), 2)}
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: stability <scores.json> [...]")
    print(json.dumps(run(sys.argv[1:]), indent=2))


if __name__ == "__main__":
    main()
