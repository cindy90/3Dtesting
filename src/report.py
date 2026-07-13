"""Median-aggregate scores per model and emit a markdown report.

The report leads with the production headline (median production score +
completion rate), then breaks the three sub-dimensions out, then a
per-category view because "riggable" only matters where a rig is expected.
Everything is median per the protocol; means are deliberately avoided.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from statistics import median
from typing import Any

from .config import RunConfig, load_cases


def _load(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _med(vals: list[float]) -> float | None:
    v = [x for x in vals if x is not None]
    return round(median(v), 1) if v else None


def _fmt(x: Any) -> str:
    return "—" if x is None else f"{x:.1f}" if isinstance(x, float) else str(x)


def build_report(cfg: RunConfig) -> str:
    scores = _load(os.path.join(cfg.out_dir, "scores.json"))
    gen = _load(os.path.join(cfg.out_dir, "generation.json"))
    cases = {c.id: c for c in load_cases(cfg.cases_file)}

    providers = sorted({r["provider"] for r in gen})

    # completion: attempted vs produced
    attempted = defaultdict(int)
    produced = defaultdict(int)
    for r in gen:
        attempted[r["provider"]] += 1
        if r.get("ok"):
            produced[r["provider"]] += 1

    by_prov: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in scores:
        by_prov[s["provider"]].append(s)

    # headline aggregate (median per model, per protocol)
    from .analysis.scoring import aggregate_model
    agg = {p: aggregate_model(by_prov.get(p, [])) for p in providers}

    lines: list[str] = []
    lines.append(f"# Production-usability blind test — `{cfg.run_id}`\n")
    lines.append("Protocol: identical input set, official APIs, blind human "
                 "spot-check, **median** aggregation. Headline score weights "
                 "watertight 0.40 / topology 0.35 / riggability 0.25 and "
                 "deliberately excludes texture prettiness (asset completeness "
                 "is shown but not ranked).\n")

    # --- headline table ---
    lines.append("## Headline (median production score)\n")
    lines.append("| Model | Completion | Median production | Watertight | Topology | Riggability | Asset* |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    ranking = sorted(providers,
                     key=lambda p: (agg[p]["median_production_score"] or -1),
                     reverse=True)
    for p in ranking:
        a = agg[p]
        comp = f"{produced[p]}/{attempted[p]}"
        lines.append(
            f"| **{p}** | {comp} | {_fmt(a['median_production_score'])} | "
            f"{_fmt(a['median_watertight_score'])} | {_fmt(a['median_topology_score'])} | "
            f"{_fmt(a['median_riggability_score'])} | {_fmt(a['median_asset_completeness'])} |"
        )
    lines.append("\n*Asset completeness (UV/normals/material) is reported for "
                 "reference only and does not affect rank — this is the exact "
                 "dimension public arenas over-weight.*\n")

    # --- watertight pass-rate: a blunt, decision-useful number ---
    metrics = _load(os.path.join(cfg.out_dir, "metrics.json"))
    wt = defaultdict(lambda: [0, 0])  # provider -> [watertight, total_ok]
    for m in metrics:
        if not m.get("ok"):
            continue
        wt[m["provider"]][1] += 1
        if m.get("is_watertight"):
            wt[m["provider"]][0] += 1
    lines.append("## Watertight pass-rate (share of produced meshes that are closed solids)\n")
    lines.append("| Model | Watertight | Rate |")
    lines.append("|---|---:|---:|")
    for p in ranking:
        w, t = wt[p]
        rate = f"{100*w/t:.0f}%" if t else "—"
        lines.append(f"| {p} | {w}/{t} | {rate} |")
    lines.append("")

    # --- per-category median production score ---
    cat_of = {cid: c.category for cid, c in cases.items()}
    cats = sorted({cat_of.get(s["case_id"], "?") for s in scores})
    catmed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in scores:
        catmed[s["provider"]][cat_of.get(s["case_id"], "?")].append(s["production_score"])
    lines.append("## Median production score by category\n")
    lines.append("| Model | " + " | ".join(cats) + " |")
    lines.append("|---|" + "---:|" * len(cats))
    for p in ranking:
        row = [f"| {p} "]
        for c in cats:
            row.append(f"| {_fmt(_med(catmed[p].get(c, [])))} ")
        lines.append("".join(row) + "|")
    lines.append("")

    lines.append("## Notes\n")
    lines.append("- Median over ~40 cases; a single catastrophic case cannot "
                 "swing rank. Low completion counts weaken a model's median — "
                 "read the two together.\n")
    lines.append("- `character`/`hardsurface` cases carry the riggability signal "
                 "(symmetry + single-shell); `organic` cases are where "
                 "splat-derived meshes most often fail watertightness.\n")

    report = "\n".join(lines)
    out_path = os.path.join(cfg.out_dir, "report.md")
    with open(out_path, "w") as fh:
        fh.write(report)
    _dump_summary(cfg, agg, produced, attempted)
    print(f"report -> {out_path}")
    return report


def _dump_summary(cfg: RunConfig, agg: dict, produced: dict, attempted: dict) -> None:
    summary = {p: {**agg[p], "completion": f"{produced[p]}/{attempted[p]}"}
               for p in agg}
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
