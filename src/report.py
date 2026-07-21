"""Median-aggregate scores per model and emit a markdown report.

The report leads with the production headline (median production score +
completion rate), then breaks the three sub-dimensions out, then a
per-category view because "riggable" only matters where a rig is expected.
Everything is median per the protocol; means are deliberately avoided.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from statistics import median
from typing import Any

from .config import RunConfig, load_cases, stable_seed


def _load(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _med(vals: list[float]) -> float | None:
    v = [x for x in vals if x is not None]
    return round(median(v), 1) if v else None


def _fmt(x: Any) -> str:
    return "—" if x is None else f"{x:.1f}" if isinstance(x, float) else str(x)


def _bootstrap_ci(rows: list[dict[str, Any]], key: str = "production_score",
                  iters: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """95% CI of the median score via CLUSTER bootstrap (resample cases).

    Repeat generations of the same case are correlated (same prompt, same
    reference pixels), so resampling individual meshes would fake power.
    We resample case_ids with replacement and carry each sampled case's
    repeats along whole. Needs >=5 distinct cases to say anything.
    """
    by_case: dict[str, list[float]] = defaultdict(list)
    for s in rows:
        v = s.get(key)
        if v is not None:
            by_case[s["case_id"]].append(float(v))
    case_ids = sorted(by_case)
    if len(case_ids) < 5:
        return None
    rng = random.Random(seed)
    meds: list[float] = []
    for _ in range(iters):
        pool: list[float] = []
        for _ in case_ids:
            pool.extend(by_case[rng.choice(case_ids)])
        meds.append(median(pool))
    meds.sort()
    return (round(meds[int(0.025 * (iters - 1))], 1),
            round(meds[int(0.975 * (iters - 1))], 1))


def build_report(cfg: RunConfig) -> str:
    scores = _load(os.path.join(cfg.out_dir, "scores.json"))
    gen = _load(os.path.join(cfg.out_dir, "generation.json"))
    cases = {c.id: c for c in load_cases(cfg.cases_file)}

    providers = sorted({r["provider"] for r in gen})

    # completion: attempted vs produced; latency: median seconds per success
    attempted = defaultdict(int)
    produced = defaultdict(int)
    lat = defaultdict(list)
    for r in gen:
        attempted[r["provider"]] += 1
        if r.get("ok"):
            produced[r["provider"]] += 1
            if r.get("latency_s"):
                lat[r["provider"]].append(float(r["latency_s"]))

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
    lines.append("| Model | Completion | Median production | 95% CI | Watertight | Topology | Riggability | Asset* | Latency |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    ranking = sorted(providers,
                     key=lambda p: (agg[p]["median_production_score"] or -1),
                     reverse=True)
    ci_seed = stable_seed(cfg.run_id)
    cis = {p: _bootstrap_ci(by_prov.get(p, []), seed=ci_seed) for p in providers}
    # providers whose product positioning is low-poly/game-first: this
    # print/sim-weighted headline is NOT their target scenario, so mark the
    # row and point readers at the game profile as their primary reading.
    game_first = {p for p in providers
                  if (cfg.providers.get(p) or {}).get("primary_profile") == "game"}
    # measured-rig blending happened if any score row carries the component
    rig_blended = any("riggability_measured_score" in s for s in scores)
    for p in ranking:
        a = agg[p]
        comp = f"{produced[p]}/{attempted[p]}"
        lt = f"{median(lat[p]):.0f}s" if lat.get(p) else "—"
        ci = cis.get(p)
        ci_s = f"{ci[0]:.0f}–{ci[1]:.0f}" if ci else "—"
        mark = "†" if p in game_first else ""
        lines.append(
            f"| **{p}**{mark} | {comp} | {_fmt(a['median_production_score'])} | {ci_s} | "
            f"{_fmt(a['median_watertight_score'])} | {_fmt(a['median_topology_score'])} | "
            f"{_fmt(a['median_riggability_score'])} | {_fmt(a['median_asset_completeness'])} | {lt} |"
        )
    lines.append("\n*Asset completeness (UV/normals/material) is reported for "
                 "reference only and does not affect rank — this is the exact "
                 "dimension public arenas over-weight.*\n")
    if game_first:
        lines.append("*† Low-poly / game-first product line: this print/sim-"
                     "weighted table is not its target scenario — read its "
                     "**game-asset profile** below as the primary number.*\n")
    if rig_blended:
        lines.append("*Riggability = 50/50 blend of the structural proxy and "
                     "the MEASURED Blender bind (rig smoke test) where rig "
                     "data exists; per-mesh components are kept in "
                     "scores.json (errata #8).*\n")
    lines.append("*95% CI: cluster bootstrap over cases (repeat generations of "
                 "a case travel together, so repeats don't fake power). "
                 "**Models whose CIs overlap are statistically tied** — treat "
                 "them as one tier, not a ranking. Shown only with ≥5 cases.*\n")

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

    # --- rig smoke test (Blender bone-heat bind + 45-degree bend) ---
    rig_path = os.path.join(cfg.out_dir, "rig_scores.json")
    if os.path.exists(rig_path):
        rig = _load(rig_path)
        by_r: dict[str, list[dict]] = defaultdict(list)
        for s in rig:
            by_r[s["provider"]].append(s)
        if by_r:
            lines.append("## Rig smoke test (Blender auto-weights + 45° bend — "
                         "measured, not proxied)\n")
            lines.append("Bind (weld) = auto-weights succeed after a standard "
                         "cleanup pass (position weld + degenerate dissolve + "
                         "debris removal). +Remesh = only binds after a voxel "
                         "remesh, which discards UVs and topology — a real "
                         "extra DCC step, score capped at 65.\n")
            lines.append("| Model | Bind (weld) | Bind (+remesh) | Median smoke score | Median volume kept | Stretch p99 |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            def _rmed(rows, key):
                vals=[x.get(key) for x in rows if isinstance(x.get(key),(int,float))]
                return median(vals) if vals else None
            for p in sorted(by_r, key=lambda k: -( _rmed(by_r[k],"rig_smoke_score") or 0)):
                rows=by_r[p]
                weld=sum(1 for x in rows if x.get("bind_tier")=="welded")
                anyb=sum(1 for x in rows if x.get("bind_ok"))
                sm=_rmed(rows,"rig_smoke_score"); vr=_rmed(rows,"volume_ratio"); st=_rmed(rows,"edge_stretch_p99")
                vr_s = "—" if vr is None else f"{vr:.2f}"
                lines.append(f"| {p} | {weld}/{len(rows)} | {anyb}/{len(rows)} | "
                             f"{_fmt(sm)} | {vr_s} | {_fmt(st)} |")
            lines.append("")

    # --- slicer ground truth (PrusaSlicer) — the print-scenario referee ---
    slc_path = os.path.join(cfg.out_dir, "slicer_scores.json")
    if os.path.exists(slc_path):
        slc = _load(slc_path)
        by_s: dict[str, list[dict]] = defaultdict(list)
        for s in slc:
            if not s.get("error"):
                by_s[s["provider"]].append(s)
        if by_s:
            lines.append("## Slicer ground truth (PrusaSlicer — the referee "
                         "real print users run)\n")
            lines.append("Manifold = strict ADMesh verdict on the exported STL "
                         "(the tier vendor pass-rates quote). Sliced = draft "
                         "g-code export succeeded — LENIENT, PrusaSlicer "
                         "auto-repairs silently, so read both columns.\n")
            lines.append("| Model | Manifold | Sliced | Median open edges | Median shells |")
            lines.append("|---|---:|---:|---:|---:|")
            def _smed(rows, key):
                vals = [x.get(key) for x in rows if isinstance(x.get(key), (int, float))]
                return median(vals) if vals else None
            def _srate(rows, key):
                return sum(1 for x in rows if x.get(key))
            for p in sorted(by_s, key=lambda k: (-_srate(by_s[k], "slicer_manifold"),
                                                 -_srate(by_s[k], "gcode_ok"))):
                rows = by_s[p]
                oe = _smed(rows, "open_edges")
                sh = _smed(rows, "n_parts")
                lines.append(
                    f"| {p} | {_srate(rows, 'slicer_manifold')}/{len(rows)} | "
                    f"{_srate(rows, 'gcode_ok')}/{len(rows)} | "
                    f"{'—' if oe is None else f'{oe:.0f}'} | "
                    f"{'—' if sh is None else f'{sh:.0f}'} |")
            lines.append("")

    # --- scenario gates + cost per usable asset ---
    # Gates are procurement pass/fail views, NOT score inputs: print gate =
    # strict slicer manifold; game gate = ships UVs AND lands in the real-time
    # budget band. Cost-per-usable divides the actual-channel price by the
    # end-to-end usable rate (completion x gate) — the single number that
    # matters for buying decisions ("cheap but rarely usable" gets exposed).
    slc_by: dict[str, list[dict]] = defaultdict(list)
    if os.path.exists(slc_path):
        for s in _load(slc_path):
            if not s.get("error"):
                slc_by[s["provider"]].append(s)
    gate_by: dict[str, list[dict]] = defaultdict(list)
    for s in scores:
        gate_by[s["provider"]].append(s)
    prices = {p: (cfg.providers.get(p) or {}).get("price_usd") for p in providers}

    def _rate(num: int, den: int) -> float | None:
        return (num / den) if den else None

    lines.append("## Scenario gates & cost per usable asset\n")
    lines.append("Gates are pass/fail procurement views and do not enter any "
                 "score. Print gate = strict slicer manifold; game gate = "
                 "ships UVs and lands in the 1.5k-150k budget band. "
                 "**$/usable = actual-channel price ÷ gate rate among "
                 "generated meshes** — what a buyer pays per asset that needs "
                 "no rescue (failed submits are typically unbilled, so "
                 "completion is reported in the headline but kept out of the "
                 "cost divisor).\n")
    lines.append("| Model | $/gen | Print gate | $/usable (print) | Game gate | $/usable (game) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for p in ranking:
        srows = slc_by.get(p, [])
        pg = _rate(sum(1 for x in srows if x.get("slicer_manifold")), len(srows))
        grows = gate_by.get(p, [])
        gg = _rate(sum(1 for x in grows
                       if x.get("uv_score", 0) > 0 and x.get("budget_fit_score", 0) >= 99),
                   len(grows))
        pr = prices.get(p)

        def _cost(gate: float | None) -> str:
            if pr is None or gate is None:
                return "—"
            return f"${pr / gate:.2f}" if gate > 0 else "∞"
        pg_s = f"{sum(1 for x in srows if x.get('slicer_manifold'))}/{len(srows)}" if srows else "—"
        gg_s = f"{sum(1 for x in grows if x.get('uv_score',0)>0 and x.get('budget_fit_score',0)>=99)}/{len(grows)}" if grows else "—"
        pr_s = f"${pr:.2f}" if pr is not None else "—"
        lines.append(f"| {p} | {pr_s} | {pg_s} | {_cost(pg)} | {gg_s} | {_cost(gg)} |")
    lines.append("\n*∞ = no generated asset passed the gate in this run; the "
                 "true cost is finite but unbounded by this sample.*\n")

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
