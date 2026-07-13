# 3D generation blind test — production-usability edition

A small, reproducible blind-test harness that lines up **Tripo 3.x, Meshy 6,
Rodin Gen-2.x, Hunyuan3D, and TRELLIS 2** on an identical 30–50 case input set
and ranks them on **production usability — watertight, topology, riggable — not
on how pretty they look.**

## Why this exists

Public 3D "arenas" rank generators by human *preference*. Preference is
dominated by texture and Gaussian-splat prettiness, so the arenas have a
**systematic blind spot** in exactly the dimension that people *pay* for: a mesh
that survives a production pipeline. That production dimension — closed solids,
clean topology, riggable bodies — is the logic behind a company like VAST
turning 3D generation into real ARR, and, unlike "beauty," it is **objectively
measurable from the geometry alone.**

So this harness keeps the arena's procedural rigor (same inputs, official APIs,
blind eval, median aggregation — see [`PROTOCOL.md`](PROTOCOL.md)) and swaps the
scoring for an automated, reproducible production-usability engine.

> **Status.** This repo is the harness. Running the 5-model sweep needs your own
> API keys and spends real credits — it is **not** run here, and no numbers are
> fabricated. The reproducible half (analysis → scoring → blind → report) is
> fully implemented and covered by an offline self-test that proves the ranking
> separates good geometry from bad.

## Quickstart

```bash
pip install -r requirements.txt

# 1. prove the analysis/scoring/report path works — no keys, no network
python -m src.selftest

# 2. add your keys
cp .env.example .env && $EDITOR .env
set -a && source .env && set +a

# 3. run the sweep (pin model versions in config.yaml first)
python -m src.cli generate     # submit every case to every model, download meshes
python -m src.cli analyze      # objective geometry metrics + production scores
python -m src.cli blind        # anonymised human spot-check set + scoresheet
python -m src.cli report       # median-aggregated results/report.md
```

`python -m src.cli generate --only tripo meshy` limits a run to some providers.

## What comes out

`results/report.md` — the headline. Median production score per model, the
watertight/topology/riggability breakdown, a watertight pass-rate, and a
per-category view (riggability only counts where a rig is expected). Plus
machine-readable `metrics.json`, `scores.json`, `summary.json`.

Example shape (numbers illustrative — from the self-test, not real models):

```
| Model | Completion | Median production | Watertight | Topology | Riggability | Asset* |
|-------|-----------:|------------------:|-----------:|---------:|------------:|-------:|
| alpha |        3/3 |              99.3 |      100.0 |     98.4 |        99.5 |   30.0 |
```

`*Asset completeness (UV/normals/material) is reported but never ranked — it is
the exact dimension arenas over-weight.*`

## The scoring engine (the point of the repo)

Every metric is derived from mesh geometry via `trimesh`, no human in the loop:

- **Watertight** — `is_watertight`, winding consistency, positive volume, open
  boundary-edge count, connected-component count.
- **Topology** — non-manifold edges, degenerate & duplicate faces, sliver
  fraction, 95th-pct triangle aspect ratio, triangle-density bloat.
- **Riggability** — single closed shell, fragmentation, estimated interior
  faces (ray-cast probe), and a scale-invariant **bilateral-symmetry residual**
  for characters (mirror-and-measure).

Weights (watertight 0.40 / topology 0.35 / riggability 0.25) and all thresholds
sit at the top of `src/analysis/scoring.py` and `src/analysis/geometry.py` —
tune them in one place, diff them in review. See [`PROTOCOL.md`](PROTOCOL.md)
for the full rationale.

## Layout

```
config.yaml              run id, model versions, provider toggles, timeouts
cases/cases.yaml         the 40-case identical input set (4 categories)
src/
  cli.py                 generate | analyze | blind | report
  selftest.py            offline end-to-end pipeline test (no keys)
  providers/             one file per model, all behind a common submit/poll API
    tripo, meshy, rodin, fal_base -> hunyuan3d, trellis
  analysis/
    geometry.py          objective per-mesh metrics  (the heart)
    scoring.py           metrics -> sub-scores -> median aggregation
  eval/blind.py          anonymise + sealed manifest + score join
  report.py              median tables -> results/report.md
```

## Honest limitations

- **Costs credits.** A full 40×5 sweep is 200 generations; budget accordingly.
- **APIs move.** Endpoint paths and model-version strings are pinned from docs
  current at authoring time — verify them (each provider file names the
  endpoints it hits) before a run.
- **Hunyuan3D / TRELLIS have no first-party REST API.** They are called through
  fal.ai's hosted deployments (`src/providers/fal_base.py`); swap in a
  Tencent-Cloud or self-hosted client by subclassing — the analysis layer is
  provider-agnostic.
- **Riggability is a proxy.** No automated metric fully predicts a clean rig;
  single-shell + symmetry + no-interior-geometry are strong *necessary*
  signals, not a guarantee.
- **Text-only by default.** Image-to-3D is supported per provider, but a fair
  image run needs a shared reference-image set you supply (`mode: image`).

## Extending

Add a provider: subclass `Provider` (or `FalProvider`), implement
`submit/status/asset_url`, register it in `src/providers/__init__.py`, add a
block to `config.yaml`. Add cases: append to `cases/cases.yaml`. Re-weight
production priorities: edit `WEIGHTS` in `src/analysis/scoring.py`.
