"""Repair-cost anchor: turn measured tiers into expected repair hours & cost.

The external-framework review's strongest idea: weights (and buying
decisions) should be anchored to what failure actually costs to fix. Our
measured tiers ARE a repair-time table — each tier corresponds to a known
DCC task with a defensible hour range:

  bind: welded tier        0.0h  (binds as-is)
        remeshed tier      3.0h  (manual retopo 2-4h; voxel remesh loses UVs)
        failed             8.0h  (effectively rebuild)
  watertight: welded       0.0h  (weld is a click)
        repaired tier      0.5h  (hole-fill + check)
        unrepairable       2.0h  (manual patching)
  uv missing               1.0h  (unwrap + pack)

Expected repair hours per asset = sum of tier probabilities x hours, and
full cost per asset = generation price + hours x TA rate (default $30/h,
parametrizable). This is a PARALLEL economic reading, not a new score; its
job is to make the weights defensible ("watertight over topology because
closure failures cost more to fix") and falsifiable (change the hour table,
the conclusion recomputes).

Usage:
    python -m src.analysis.repair_cost results/scores.json \
        --rig results/rig_scores.json --metrics results/metrics.json \
        [--rate 30]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

HOURS = {
    "bind_remeshed": 3.0,
    "bind_failed": 8.0,
    "wt_repaired": 0.5,
    "wt_unrepairable": 2.0,
    "uv_missing": 1.0,
}
DEFAULT_RATE = 30.0  # USD per TA hour; conservative mid freelance rate


def expected_hours(provider_rows: dict[str, Any]) -> float:
    """Expected repair hours for one provider from tier fractions."""
    rig = provider_rows.get("rig", [])
    h = 0.0
    if rig:
        n = len(rig)
        remeshed = sum(1 for r in rig if r.get("bind_tier") == "remeshed")
        failed = sum(1 for r in rig if not r.get("bind_ok"))
        h += HOURS["bind_remeshed"] * remeshed / n
        h += HOURS["bind_failed"] * failed / n
    mets = provider_rows.get("metrics", [])
    if mets:
        n = len(mets)
        repaired = sum(1 for m in mets
                       if not m.get("is_watertight")
                       and (m.get("extra") or {}).get("watertight_after_repair"))
        unrep = sum(1 for m in mets
                    if not m.get("is_watertight")
                    and not (m.get("extra") or {}).get("watertight_after_repair"))
        h += HOURS["wt_repaired"] * repaired / n
        h += HOURS["wt_unrepairable"] * unrep / n
    scores = provider_rows.get("scores", [])
    if scores:
        no_uv = sum(1 for s in scores if s.get("uv_score", 0) <= 0)
        h += HOURS["uv_missing"] * no_uv / len(scores)
    return h


def run(scores_path: str, rig_path: str | None, metrics_path: str | None,
        prices: dict[str, float] | None = None,
        rate: float = DEFAULT_RATE) -> dict[str, dict[str, Any]]:
    by: dict[str, dict[str, list]] = defaultdict(lambda: {"scores": [], "rig": [], "metrics": []})
    with open(scores_path) as fh:
        for r in json.load(fh):
            by[r["provider"]]["scores"].append(r)
    if rig_path:
        with open(rig_path) as fh:
            for r in json.load(fh):
                by[r["provider"]]["rig"].append(r)
    if metrics_path:
        with open(metrics_path) as fh:
            for r in json.load(fh):
                if r.get("ok"):
                    by[r["provider"]]["metrics"].append(r)
    out: dict[str, dict[str, Any]] = {}
    for p, rows in by.items():
        h = expected_hours(rows)
        price = (prices or {}).get(p)
        out[p] = {
            "expected_repair_hours": round(h, 2),
            "repair_cost_usd": round(h * rate, 2),
            "full_cost_per_asset_usd":
                round(price + h * rate, 2) if price is not None else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--rig", default=None)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE)
    args = ap.parse_args()
    out = run(args.scores, args.rig, args.metrics, rate=args.rate)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
