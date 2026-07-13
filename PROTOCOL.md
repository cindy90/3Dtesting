# Blind-test protocol (HackerNoon-aligned)

This harness reproduces the discipline of the HackerNoon-style 3D generator
blind test, but re-points the *scoring* at production usability instead of
visual preference. The four procedural rules are kept verbatim; the fifth
(what we measure) is the intentional change.

## The five rules

1. **Identical input set.** Every model receives the exact same cases from
   `cases/cases.yaml` — same prompts, same modes, same order. No per-model
   prompt tuning. (40 text cases ship by default.)

   **Mixed cohorts (text-native + image-to-3D).** Most open-source generators
   (SF3D, TripoSR, InstantMesh, TRELLIS, Hunyuan3D) are *image-to-3D*, while the
   commercial ones also take text. To compare them fairly the harness supports
   an **image mode**: it generates ONE shared reference image per case from a
   fixed text-to-image model (`refimg` → FLUX on fal) and feeds that identical
   image to every model (Tripo included, via its image endpoint). This upholds
   the identical-input rule at the *pixel* level — a stronger guarantee than
   identical text, because text-to-image variance is removed from the compare.
   Run `python -m src.cli refimg` once, then `generate --mode image`.

2. **Official APIs, pinned versions.** Each model is called through its
   first-party (or, for open-weights models, the canonical hosted) API, at a
   model version pinned in `config.yaml`. A version bump is a one-line diff so
   any run is reproducible and re-runnable.

3. **Blind evaluation.** The objective geometry metrics need no blinding —
   geometry cannot be biased. The optional human spot-check (prompt match /
   gross-artefact catch) is fully blinded: `src/eval/blind.py` copies every mesh
   to a shuffled anonymous id and seals the id→model manifest until scoring is
   done.

4. **Median aggregation.** Every model's headline number is the **median** of
   its per-case scores, not the mean. A single catastrophic or lucky case
   cannot move a rank. Completion rate is reported alongside so a high median
   built on few successes is visible.

5. **Production-usability scoring (the deliberate departure).** Public arenas
   rank by human preference, which is dominated by texture and Gaussian-splat
   prettiness — a systematic blind spot for anyone who has to *ship* the mesh.
   We instead score three geometrically-measurable dimensions that map to what
   paying customers need, and we exclude prettiness from the rank.

## What "production usability" means, precisely

Headline production score = weighted sum of three 0–100 sub-scores:

| Dimension | Weight | Measures | Why it's the paid dimension |
|---|---:|---|---|
| **Watertight** | 0.40 | closed manifold, consistent winding, positive volume, zero boundary edges | printing, simulation, boolean ops, CAD |
| **Topology** | 0.35 | non-manifold edges, degenerate/duplicate faces, sliver ratio, aspect ratio, bloat | re-mesh, LODs, clean deformation, engine import |
| **Riggability** | 0.25 | single closed shell, low fragmentation, no interior geometry, bilateral symmetry (characters) | skinning + animation |

Reported but **not** ranked: `asset_completeness` (UV / normals / material
presence). It is shown so you can see the texture-side story that arenas
over-weight — it just never inflates the production rank.

All thresholds and weights live at the top of `src/analysis/geometry.py` and
`src/analysis/scoring.py` so the ranking is fully auditable and tunable in one
place.

## Sample size

30–50 cases is the sweet spot: enough for a stable median across the four
content categories (character / prop / hard-surface / organic), few enough to
keep a full 5-model sweep affordable. The default set is 40.

## Reproducibility

The blind shuffle is seeded from `run_id` via a deterministic hash — no
wall-clock randomness — so the same run id reproduces the same anonymised
ordering. Model versions are pinned in config. Re-running a benchmark is
`generate → analyze → report` with the same config.
