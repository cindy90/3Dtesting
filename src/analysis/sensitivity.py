"""Weight-sensitivity analysis: is the tier structure a weights artifact?

The production profile weights (watertight .40 / topology .35 / riggability
.25) are reasoned but hand-set. This script answers the standard objection
("your ranking is whatever your weights say") empirically: perturb each
weight by a uniform factor in [0.5, 1.5], renormalize, recompute every
model's median production score, and measure how often pairwise orderings
flip across K draws. Orderings that survive arbitrary +/-50% re-weighting
are structural; those that flip are weight-artifacts and must be reported
as ties.

Usage:
    python -m src.analysis.sensitivity results/scores.json [more_scores.json ...]

Deterministic (seeded); no network, no cost.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from statistics import median

BASE = {"watertight": 0.40, "topology": 0.35, "riggability": 0.25}
K = 2000
_SEED = 0


def load_scores(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p) as fh:
            rows.extend(json.load(fh))
    return [r for r in rows if r.get("watertight_score") is not None]


def median_scores(rows: list[dict], w: dict[str, float]) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        s = (w["watertight"] * r["watertight_score"]
             + w["topology"] * r["topology_score"]
             + w["riggability"] * r["riggability_score"])
        by[r["provider"]].append(s)
    return {p: median(v) for p, v in by.items()}


def run(paths: list[str], k: int = K) -> dict:
    rows = load_scores(paths)
    rng = random.Random(_SEED)
    base_med = median_scores(rows, BASE)
    provs = sorted(base_med, key=lambda p: -base_med[p])
    pairs = [(a, b) for i, a in enumerate(provs) for b in provs[i + 1:]]
    keep = {pr: 0 for pr in pairs}
    for _ in range(k):
        w = {kk: v * rng.uniform(0.5, 1.5) for kk, v in BASE.items()}
        tot = sum(w.values())
        w = {kk: v / tot for kk, v in w.items()}
        med = median_scores(rows, w)
        for a, b in pairs:
            if med[a] >= med[b]:
                keep[(a, b)] += 1
    stability = {f"{a} > {b}": round(keep[(a, b)] / k, 3) for a, b in pairs}
    return {"base_median": {p: round(base_med[p], 1) for p in provs},
            "n_meshes": len(rows), "k_draws": k,
            "pair_stability": stability}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: sensitivity <scores.json> [...]")
    out = run(sys.argv[1:])
    print(json.dumps(out, indent=2, ensure_ascii=False))
    stable = sum(1 for v in out["pair_stability"].values() if v >= 0.95 or v <= 0.05)
    print(f"\n{stable}/{len(out['pair_stability'])} pairwise orderings are "
          f"stable under +/-50% weight perturbation (>=95% of draws agree). "
          f"Pairs between 5% and 95% are weight-sensitive -> report as TIES.")


if __name__ == "__main__":
    main()
