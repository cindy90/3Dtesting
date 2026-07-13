"""Offline end-to-end self-test of the analyze -> blind -> report pipeline.

No API keys, no network. It fabricates a handful of meshes of deliberately
different production quality (a clean watertight solid, a holed/open shell, a
fragmented multi-shell, a bloated sliver-ridden mesh), pretends three
"providers" produced them, then runs the exact same analyze/score/report code
path the real harness uses. Purpose: prove the reproducible half of the harness
is correct and that ranking actually separates good geometry from bad — before
any money is spent on generation.

    python -m src.selftest
"""

from __future__ import annotations

import json
import os

import numpy as np
import trimesh

from .config import RunConfig, stable_seed
from .analysis.geometry import analyze_mesh
from .analysis.scoring import score_mesh
from .eval.blind import build_blind_set
from . import report as report_mod


def _clean_solid() -> trimesh.Trimesh:
    """Watertight, symmetric, regular — a rigger's dream."""
    return trimesh.creation.icosphere(subdivisions=3)


def _holed_shell() -> trimesh.Trimesh:
    """Open mesh (missing faces) — not watertight, not printable."""
    m = trimesh.creation.icosphere(subdivisions=3)
    keep = np.ones(len(m.faces), dtype=bool)
    keep[:40] = False  # punch a hole
    return trimesh.Trimesh(m.vertices, m.faces[keep], process=False)


def _fragmented() -> trimesh.Trimesh:
    """Two disjoint shells (floating parts) — bad for rigging."""
    a = trimesh.creation.icosphere(subdivisions=2)
    b = trimesh.creation.icosphere(subdivisions=2)
    b.apply_translation([3.0, 0, 0])
    return trimesh.util.concatenate([a, b])


def _bloated_slivers() -> trimesh.Trimesh:
    """Watertight but noisy, asymmetric, sliver-heavy topology."""
    m = trimesh.creation.icosphere(subdivisions=4)
    rng = np.random.default_rng(1)
    v = m.vertices + rng.normal(0, 0.03, m.vertices.shape)
    v[v[:, 0] > 0, 0] *= 1.5  # break symmetry
    return trimesh.Trimesh(v, m.faces, process=False)


# provider -> case_id -> mesh factory (quality profile per "provider")
_PROFILES = {
    "alpha":   {"g1": _clean_solid,  "g2": _clean_solid,    "g3": _holed_shell},
    "bravo":   {"g1": _bloated_slivers, "g2": _clean_solid, "g3": _fragmented},
    "charlie": {"g1": _holed_shell,  "g2": _fragmented,     "g3": _bloated_slivers},
}

# g1/g3 are "characters" (symmetry matters), g2 a prop
_EXPECT_SYM = {"g1": True, "g2": False, "g3": True}
_CATEGORY = {"g1": "character", "g2": "prop", "g3": "character"}


def main() -> None:
    out_dir = os.path.join("results", "_selftest")
    os.makedirs(out_dir, exist_ok=True)

    generation = []
    for prov, cases in _PROFILES.items():
        for cid, factory in cases.items():
            mesh = factory()
            path = os.path.join(out_dir, prov, f"{cid}.glb")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            mesh.export(path)
            generation.append({"provider": prov, "case_id": cid, "ok": True,
                               "mesh_path": path, "task_id": None,
                               "latency_s": 1.0, "error": None})
    with open(os.path.join(out_dir, "generation.json"), "w") as fh:
        json.dump(generation, fh, indent=2)

    # analyze + score
    metrics, scores = [], []
    for r in generation:
        sym = _EXPECT_SYM[r["case_id"]]
        m = analyze_mesh(r["mesh_path"], expect_symmetry=sym).to_dict()
        m.update({"provider": r["provider"], "case_id": r["case_id"]})
        metrics.append(m)
        s = score_mesh(m, expect_symmetry=sym)
        s.update({"provider": r["provider"], "case_id": r["case_id"]})
        scores.append(s)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(out_dir, "scores.json"), "w") as fh:
        json.dump(scores, fh, indent=2)

    # blind set
    build_blind_set(generation, out_dir, seed=stable_seed("selftest"))

    # report — needs a cases file; write a tiny one matching the fake cases
    cases_file = os.path.join(out_dir, "cases.yaml")
    with open(cases_file, "w") as fh:
        fh.write("version: 1\ncases:\n")
        for cid in ("g1", "g2", "g3"):
            fh.write(f"  - {{id: {cid}, category: {_CATEGORY[cid]}, "
                     f"expect_symmetry: {str(_EXPECT_SYM[cid]).lower()}, "
                     f"prompt: selftest}}\n")
    cfg = RunConfig(run_id="selftest", providers={}, out_dir=out_dir,
                    cases_file=cases_file)
    report_mod.build_report(cfg)

    # assertions: ranking must reflect designed quality (alpha > bravo > charlie)
    with open(os.path.join(out_dir, "summary.json")) as fh:
        summary = json.load(fh)
    order = sorted(summary, key=lambda p: summary[p]["median_production_score"],
                   reverse=True)
    print("\nself-test ranking (best->worst):", order)
    print("median production scores:",
          {p: summary[p]["median_production_score"] for p in order})
    assert order[0] == "alpha", f"expected alpha best, got {order}"
    assert order[-1] == "charlie", f"expected charlie worst, got {order}"
    # clean solid must beat holed shell on watertightness
    wt = {(m["provider"], m["case_id"]): m for m in metrics}
    assert wt[("alpha", "g1")]["is_watertight"] is True
    assert wt[("charlie", "g1")]["is_watertight"] is False
    print("\nSELF-TEST PASSED — pipeline separates production quality correctly.")
    print(f"artefacts in {out_dir}/  (report.md, scores.json, blind_scoresheet.csv)")


if __name__ == "__main__":
    main()
