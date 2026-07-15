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

    # --- second profile: game-asset readiness (budget fit + UVs) ---
    lines.append("## Game-asset profile (median; budget-fit 0.15 + UV 0.10, "
                 "softer watertightness)\n")
    lines.append("| Model | Median game-ready | Budget fit | UV | Watertight | Topology |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    game_rank = sorted(providers,
                       key=lambda p: (agg[p].get("median_game_ready_score") or -1),
                       reverse=True)
    for p in game_rank:
        a = agg[p]
        lines.append(
            f"| **{p}** | {_fmt(a.get('median_game_ready_score'))} | "
            f"{_fmt(a.get('median_budget_fit_score'))} | {_fmt(a.get('median_uv_score'))} | "
            f"{_fmt(a.get('median_watertight_score'))} | {_fmt(a.get('median_topology_score'))} |"
        )
    lines.append("\n*The print/sim headline rewards dense watertight sculpts; "
                 "this profile asks whether the mesh fits a real-time budget "
                 "(1.5k-150k tris) and ships UVs. Read both — they answer "
                 "different pipelines.*\n")

    # --- semantic fidelity (if the VLM judge or human sheet has been run) ---
    sem_path = os.path.join(cfg.out_dir, "semantic_scores.json")
    if os.path.exists(sem_path):
        sem = _load(sem_path)
        by_p: dict[str, list[float]] = defaultdict(list)
        for s in sem:
            v = s.get("vlm_match_1to5")
            if v is not None:
                by_p[s["provider"]].append(float(v))
        if by_p:
            lines.append("## Semantic fidelity (VLM judge, median 1-5 — does "
                         "the mesh match the reference?)\n")
            lines.append("| Model | Median match | n |")
            lines.append("|---|---:|---:|")
            for p in sorted(by_p, key=lambda k: -median(by_p[k])):
                lines.append(f"| {p} | {median(by_p[p]):.1f} | {len(by_p[p])} |")
            lines.append("")

    # --- watertight pass-rate, THREE tiers (raw / welded / after auto-repair).
    # Vendors' self-reported "watertight rates" are incomparable because they
    # never state the processing tier; the same file can show 300k open edges
    # on raw buffers and 0 after a position weld. We report all three.
    metrics = _load(os.path.join(cfg.out_dir, "metrics.json"))
    wt3 = defaultdict(lambda: [0, 0, 0, 0])  # raw, welded, repaired, total
    for m in metrics:
        if not m.get("ok"):
            continue
        extra = m.get("extra", {}) or {}
        c = wt3[m["provider"]]
        c[3] += 1
        welded = bool(m.get("is_watertight"))
        raw_closed = welded and extra.get("raw_boundary_edges", 0) == 0
        repaired = welded or bool(extra.get("watertight_after_repair"))
        c[0] += raw_closed
        c[1] += welded
        c[2] += repaired
    lines.append("## Watertight pass-rate — three tiers\n")
    lines.append("Raw = closed on the exported buffers as-is; Welded = closed "
                 "after a position weld (any pipeline's first step); Repaired = "
                 "closed after weld + one-click hole fill. Vendor-quoted rates "
                 "that don't state their tier are not comparable to any column.\n")
    lines.append("| Model | Raw | Welded | +Auto-repair |")
    lines.append("|---|---:|---:|---:|")
    for p in ranking:
        r, w, rep, t = wt3[p]
        if not t:
            lines.append(f"| {p} | — | — | — |")
            continue
        lines.append(f"| {p} | {r}/{t} | {w}/{t} | {rep}/{t} |")
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
