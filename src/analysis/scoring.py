"""Turn raw geometric metrics into production-usability scores.

Design choices, kept explicit so the ranking is auditable:

* Three sub-scores mirror the three things paying customers actually need:
  ``watertight``, ``topology``, ``riggability``. Each is a 0-100 number.
* We deliberately do **not** reward texture / PBR prettiness in the headline
  score — that is the exact dimension public arenas over-weight. Asset
  completeness is reported separately as ``asset_completeness`` so it is
  visible but never inflates production rank.
* Aggregation across cases uses the **median**, per the protocol, because a
  single catastrophic (or lucky) case should not move a model's rank.

All thresholds live in ``WEIGHTS``/``_curve`` constants at the top so they can
be tuned in one place and diffed in review.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Iterable

# Sub-score weights inside the headline production score.
WEIGHTS = {
    "watertight": 0.40,   # printability / sim / boolean-ability
    "topology": 0.35,     # re-mesh, LOD, clean deformation
    "riggability": 0.25,  # animation readiness
}


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _decay(value: float, half: float) -> float:
    """Map a non-negative 'badness' count to 100..0, halving every ``half``."""
    if value <= 0:
        return 100.0
    return 100.0 * (0.5 ** (value / half))


def watertight_score(m: dict[str, Any]) -> float:
    """Closed, single-shell, correctly-oriented solid == 100."""
    if not m.get("ok"):
        return 0.0
    score = 0.0
    if m.get("is_watertight"):
        score += 55.0
    if m.get("is_winding_consistent"):
        score += 15.0
    if m.get("is_volume"):  # watertight + consistent + positive volume
        score += 15.0
    # penalise open boundary edges relative to nothing-open ideal
    be = m.get("n_boundary_edges", 0)
    score += 15.0 * (0.5 ** (be / 20.0)) if be else 15.0
    return _clamp(score)


def topology_score(m: dict[str, Any]) -> float:
    """Manifold, sliver-free, non-degenerate, non-bloated triangulation."""
    if not m.get("ok") or m.get("n_faces", 0) == 0:
        return 0.0
    nonmanifold = _decay(m.get("n_nonmanifold_edges", 0), half=10.0)
    degenerate = 100.0 * (1.0 - _clamp(m.get("frac_degenerate_faces", 0.0) * 100, 0, 1))
    slivers = 100.0 * (1.0 - _clamp(m.get("sliver_frac", 0.0), 0, 1))
    # aspect ratio p95: 1==perfect, >=6 is bad
    ar = m.get("aspect_ratio_p95", 1.0)
    aspect = _clamp(100.0 - (ar - 1.5) * 18.0)
    dupes = _decay(m.get("n_duplicate_faces", 0), half=25.0)
    return _clamp(
        0.30 * nonmanifold
        + 0.20 * degenerate
        + 0.20 * slivers
        + 0.20 * aspect
        + 0.10 * dupes
    )


def riggability_score(m: dict[str, Any], *, expect_symmetry: bool) -> float:
    """Single clean shell, no interior geometry, symmetric if a character."""
    if not m.get("ok"):
        return 0.0
    score = 0.0
    # one closed shell is the single biggest riggability prerequisite
    score += 40.0 if m.get("single_shell") else 0.0
    # fragmentation penalty (many components == fused/floating parts)
    comps = m.get("n_connected_components", 1)
    score += 20.0 * (0.5 ** (max(0, comps - 1) / 3.0))
    # interior geometry wrecks skin weights; -1 means "not measured" (too dense
    # for the ray probe) -> stay neutral rather than penalise.
    internal = m.get("n_internal_faces_est", 0)
    if internal is None or internal < 0:
        score += 20.0
    else:
        faces = max(1, m.get("n_faces", 1))
        score += 20.0 * (1.0 - _clamp(internal / faces, 0, 1))
    # symmetry (characters only); non-character cases get the credit for free
    if expect_symmetry:
        res = m.get("symmetry_residual")
        if res is None:
            score += 10.0  # could not measure; neutral-ish
        else:
            # residual 0 -> 20, 0.05 -> ~0
            score += 20.0 * (0.5 ** (res / 0.015))
    else:
        score += 20.0
    return _clamp(score)


def asset_completeness(m: dict[str, Any]) -> float:
    """Reported-only: UV + normals + material presence. Never in headline."""
    if not m.get("ok"):
        return 0.0
    return _clamp(
        40.0 * bool(m.get("has_uv"))
        + 30.0 * bool(m.get("has_vertex_normals"))
        + 30.0 * bool(m.get("has_texture_or_material"))
    )


def score_mesh(metrics: dict[str, Any], *, expect_symmetry: bool = False) -> dict[str, Any]:
    """Compute sub-scores + weighted production score for one mesh."""
    wt = watertight_score(metrics)
    topo = topology_score(metrics)
    rig = riggability_score(metrics, expect_symmetry=expect_symmetry)
    production = (
        WEIGHTS["watertight"] * wt
        + WEIGHTS["topology"] * topo
        + WEIGHTS["riggability"] * rig
    )
    return {
        "watertight_score": round(wt, 2),
        "topology_score": round(topo, 2),
        "riggability_score": round(rig, 2),
        "asset_completeness": round(asset_completeness(metrics), 2),
        "production_score": round(production, 2),
    }


def _median(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(median(vals), 2) if vals else None


def aggregate_model(per_case_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Median-aggregate a model's per-case scores per the protocol.

    ``per_case_scores`` is a list of the dicts returned by ``score_mesh`` (one
    per successfully generated case). We report the median of each sub-score
    plus a completion rate, since failing to produce a mesh at all is itself a
    production signal that a mean would hide.
    """
    keys = [
        "production_score",
        "watertight_score",
        "topology_score",
        "riggability_score",
        "asset_completeness",
    ]
    out = {f"median_{k}": _median(s[k] for s in per_case_scores) for k in keys}
    out["n_scored"] = len(per_case_scores)
    return out
