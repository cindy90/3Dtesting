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

# Sub-score weights inside the headline production score (print/sim profile).
WEIGHTS = {
    "watertight": 0.40,   # printability / sim / boolean-ability
    "topology": 0.35,     # re-mesh, LOD, clean deformation
    "riggability": 0.25,  # animation readiness
}

# Second, separately-reported profile: GAME-ASSET readiness. The print/sim
# profile rewards dense watertight sculpts and ignores polygon budgets & UVs —
# fine for printing, wrong for engines. This profile adds budget fit and UV
# presence and softens watertightness (engines tolerate some openness).
# Both leaderboards are reported; neither replaces the other.
GAME_WEIGHTS = {
    "watertight": 0.25,
    "topology": 0.30,
    "riggability": 0.20,
    "budget_fit": 0.15,   # faces within a real-time asset budget
    "uv": 0.10,           # UVs present = texturable without unwrap work
}

# Real-time budget band: full credit inside, log2 half-life outside. Wide on
# purpose — hero assets legitimately reach ~150k tris; film/print density
# (500k+) or sub-1k blobs both need rework.
_BUDGET_LO, _BUDGET_HI = 1_500, 150_000


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _decay(value: float, half: float) -> float:
    """Map a non-negative 'badness' count to 100..0, halving every ``half``."""
    if value <= 0:
        return 100.0
    return 100.0 * (0.5 ** (value / half))


def watertight_score(m: dict[str, Any]) -> float:
    """Closure scored by DEFECT MAGNITUDE, not a binary cliff. (scoring v2)

    v1 scored ``is_watertight`` as a 55-point binary + 15-point is_volume,
    so a 1.5M-face mesh with TWO defective edges (0.0002% — invisible,
    auto-repairable in any pipeline) lost ~70 points and ranked with genuinely
    holed meshes. Forensics on real outputs showed exactly this pattern
    (multi-part game-style assets fail on 2-6 seam edges). v2 uses a smooth
    curve over the defective-edge ratio, credits per-part closure
    (closed_face_fraction) and one-click repairability.
    """
    if not m.get("ok"):
        return 0.0
    import math
    if m.get("is_watertight"):
        base = 100.0 if m.get("is_volume") else 94.0
    else:
        edges = m.get("extra", {}).get("n_edges") or (m.get("n_faces", 0) * 1.5) or 1
        defects = m.get("n_boundary_edges", 0) + m.get("n_nonmanifold_edges", 0)
        if defects <= 0:
            base = 90.0  # closure broken only by winding/orientation
        else:
            x = math.log10(defects / edges)
            #        ratio:  1e-7  1e-6  1e-5  1e-4  1e-3  1e-2  1e-1   1
            pts = [(-7, 95), (-6, 88), (-5, 78), (-4, 62), (-3, 45),
                   (-2, 25), (-1, 10), (0, 5)]
            if x <= pts[0][0]:
                base = pts[0][1]
            elif x >= pts[-1][0]:
                base = pts[-1][1]
            else:
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                    if x0 <= x <= x1:
                        base = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
                        break
        # most faces live in closed parts -> the asset is largely solid
        cff = m.get("extra", {}).get("closed_face_fraction")
        if cff is not None:
            base = max(base, 85.0 * float(cff))
        # a plain fill-holes pass closes it -> production cost is one click
        if m.get("extra", {}).get("watertight_after_repair"):
            base = max(base, 72.0)
    if not m.get("is_winding_consistent"):
        base -= 8.0
    return _clamp(base)


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
    # closure of the parts (v2): a multi-part asset whose parts are each
    # closed (eyes/teeth/props — standard game construction) rigs fine;
    # only the open fraction hurts. Falls back to the v1 single-shell binary
    # when the fraction wasn't recorded.
    cff = m.get("extra", {}).get("closed_face_fraction") if isinstance(m.get("extra"), dict) else None
    if cff is None:
        cff = 1.0 if m.get("single_shell") else (1.0 if m.get("is_watertight") else 0.0)
    score += 40.0 * float(cff)
    # fragmentation penalty, softened (parts are a style, dozens are a mess)
    comps = m.get("n_connected_components", 1)
    score += 20.0 * (0.5 ** (max(0, comps - 1) / 6.0))
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


def budget_fit_score(m: dict[str, Any]) -> float:
    """100 inside the real-time band, halving per doubling outside it."""
    import math
    faces = m.get("n_faces", 0) or 0
    if faces <= 0:
        return 0.0
    if _BUDGET_LO <= faces <= _BUDGET_HI:
        return 100.0
    ratio = _BUDGET_LO / faces if faces < _BUDGET_LO else faces / _BUDGET_HI
    return _clamp(100.0 * (0.5 ** math.log2(ratio)))


def score_mesh(metrics: dict[str, Any], *, expect_symmetry: bool = False) -> dict[str, Any]:
    """Compute sub-scores + both profile scores for one mesh.

    ``production_score`` (print/sim profile) keeps its frozen definition for
    continuity with earlier runs; ``game_ready_score`` is the second,
    engine-oriented profile (budget fit + UVs, softer watertightness).
    """
    wt = watertight_score(metrics)
    topo = topology_score(metrics)
    rig = riggability_score(metrics, expect_symmetry=expect_symmetry)
    budget = budget_fit_score(metrics)
    uv = 100.0 * bool(metrics.get("has_uv"))
    production = (
        WEIGHTS["watertight"] * wt
        + WEIGHTS["topology"] * topo
        + WEIGHTS["riggability"] * rig
    )
    game = (
        GAME_WEIGHTS["watertight"] * wt
        + GAME_WEIGHTS["topology"] * topo
        + GAME_WEIGHTS["riggability"] * rig
        + GAME_WEIGHTS["budget_fit"] * budget
        + GAME_WEIGHTS["uv"] * uv
    )
    return {
        "watertight_score": round(wt, 2),
        "topology_score": round(topo, 2),
        "riggability_score": round(rig, 2),
        "budget_fit_score": round(budget, 2),
        "uv_score": round(uv, 2),
        "asset_completeness": round(asset_completeness(metrics), 2),
        "production_score": round(production, 2),
        "game_ready_score": round(game, 2),
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
        "game_ready_score",
        "watertight_score",
        "topology_score",
        "riggability_score",
        "budget_fit_score",
        "uv_score",
        "asset_completeness",
    ]
    out = {f"median_{k}": _median(s.get(k) for s in per_case_scores) for k in keys}
    out["n_scored"] = len(per_case_scores)
    return out
