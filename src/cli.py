"""Command-line entry point tying the harness together.

Subcommands (run in order):

  generate  submit every case to every configured provider, download meshes,
            write results/generation.json
  analyze   run the objective geometry metrics + scoring on every mesh,
            write results/metrics.json and results/scores.json
  blind     build the anonymised human-spot-check set + blank scoresheet
  report    median-aggregate per model and emit results/report.md

Typical:
    python -m src.cli generate
    python -m src.cli analyze
    python -m src.cli report
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .config import load_config, load_cases, stable_seed, RunConfig
from .providers import build_provider, Case
from .analysis.geometry import analyze_mesh
from .analysis.scoring import score_mesh, aggregate_model
from .eval.blind import build_blind_set


def _sample_cases(cases: list[Case], n: int) -> list[Case]:
    """Take at most ``n`` cases, spread evenly across categories.

    Round-robin over the per-category buckets (each kept in file order) so a
    2-case smoke test still touches two different content types rather than two
    characters. Deterministic — no randomness.
    """
    if n <= 0 or n >= len(cases):
        return cases
    buckets: dict[str, list[Case]] = {}
    for c in cases:
        buckets.setdefault(c.category, []).append(c)
    order = sorted(buckets)  # stable category order
    picked: list[Case] = []
    i = 0
    while len(picked) < n:
        cat = order[i % len(order)]
        if buckets[cat]:
            picked.append(buckets[cat].pop(0))
        i += 1
        if all(not b for b in buckets.values()):
            break
    return picked[:n]


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def cmd_refimg(cfg: RunConfig, cases: list[Case], force: bool) -> None:
    """Generate one shared reference image per case (for image-to-3D models)."""
    from .refimg import generate_reference_images
    print(f"generating reference images for {len(cases)} cases -> {cfg.refs_dir}")
    made = generate_reference_images(cases, cfg.refs_dir, cfg.reference_images,
                                     force=force)
    print(f"reference images: {len(made)}/{len(cases)} ready in {cfg.refs_dir}")
    if cases and not made:
        # fail loudly: image-mode generate is pointless without refs, and a
        # silent 0/N here previously surfaced as confusing downstream errors.
        raise SystemExit("refimg produced 0 images — fix this before generate "
                         "(check FAL_KEY and the FLUX endpoint above)")


def _apply_image_mode(cfg: RunConfig, cases: list[Case]) -> list[Case]:
    """Point every case at its shared reference image and switch to image mode.

    Enforces the identical-input rule for a mixed cohort: every model consumes
    the same pixels. Missing refs are left as-is so the provider fails loudly
    for that case rather than silently comparing different inputs.
    """
    out = []
    for c in cases:
        ref = os.path.join(cfg.refs_dir, f"{c.id}.png")
        c.mode = "image"
        c.image_path = ref
        if not os.path.exists(ref):
            print(f"  ! missing reference image for {c.id}: {ref} "
                  f"(run `refimg` first)")
        out.append(c)
    return out


def cmd_generate(cfg: RunConfig, cases: list[Case], only: list[str] | None,
                 repeats: int = 1) -> None:
    """Submit every case to every provider, ``repeats`` times each.

    Repeats share the case's identity (same prompt/reference image/case_id)
    but write distinct mesh files; downstream medians pool them and the
    report's bootstrap CI resamples by case (cluster bootstrap) so correlated
    repeats don't fake extra statistical power.
    """
    import dataclasses
    import threading
    results: list[dict[str, Any]] = []
    provider_keys = [k for k in cfg.providers if not only or k in only]
    repeats = max(1, repeats)
    print(f"generating {len(cases)} cases x {len(provider_keys)} providers "
          f"x {repeats} repeats [mode={cfg.mode}]")

    # Per-BACKEND concurrency cap. tripo / tripo_h31 / tripo_p1 / tripo_p1d are
    # four provider keys but ONE Tripo account — 24-way fan-out slammed that
    # single backend into 429 rate-limits and burned the run. Group providers by
    # their credential env (the real backend) and cap simultaneous in-flight
    # tasks per backend; distinct backends still run fully in parallel.
    per_backend = int(cfg.extra.get("generate", {}).get("per_backend_concurrency", 4))
    backend_of: dict[str, str] = {}
    for pkey in provider_keys:
        try:
            backend_of[pkey] = build_provider(pkey, cfg.providers[pkey]).api_key_env or pkey
        except Exception:
            backend_of[pkey] = pkey
    sems = {b: threading.Semaphore(per_backend) for b in set(backend_of.values())}

    def _one(pkey: str, case: Case, rep: int) -> dict[str, Any]:
        with sems[backend_of[pkey]]:                 # cap load on the real backend
            prov = build_provider(pkey, cfg.providers[pkey])
            # per-provider timeout override (e.g. seed3d regularly exceeds 900s)
            prov.timeout_s = float((cfg.providers[pkey] or {}).get("timeout_s",
                                                                   cfg.timeout_s))
            prov.poll_interval_s = cfg.poll_interval_s
            res = prov.run(case, cfg.out_dir)
        status = "ok" if res.ok else f"FAIL({res.error})"
        rep_tag = f" (rep {rep})" if rep > 1 else ""
        print(f"  [{pkey}] {case.id}{rep_tag}: {status}")
        return {
            "provider": res.provider, "case_id": res.case_id, "rep": rep,
            "ok": res.ok,
            "mesh_path": res.mesh_path, "mesh_url": res.mesh_url,
            "task_id": res.task_id,
            "latency_s": res.latency_s, "error": res.error,
        }

    # distinct providers/backends run concurrently; the per-backend semaphore
    # above keeps any single API under its rate limit
    jobs = [(p, c if r == 1 else dataclasses.replace(c, out_name=f"{c.id}__r{r}"), r)
            for p in provider_keys for c in cases for r in range(1, repeats + 1)]
    with ThreadPoolExecutor(max_workers=min(24, max(1, len(jobs)))) as ex:
        futs = {ex.submit(_one, p, c, r): (p, c.id, r) for p, c, r in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())

    _write_json(os.path.join(cfg.out_dir, "generation.json"), results)
    ok = sum(1 for r in results if r["ok"])
    print(f"done: {ok}/{len(results)} meshes generated -> "
          f"{cfg.out_dir}/generation.json")


def cmd_analyze(cfg: RunConfig, cases: list[Case]) -> None:
    gen_path = os.path.join(cfg.out_dir, "generation.json")
    if not os.path.exists(gen_path):
        raise SystemExit("run `generate` first (no generation.json)")
    with open(gen_path) as fh:
        gen = json.load(fh)
    case_by_id = {c.id: c for c in cases}

    metrics_out: list[dict[str, Any]] = []
    scores_out: list[dict[str, Any]] = []
    for r in gen:
        if not r.get("ok") or not r.get("mesh_path") or not os.path.exists(r["mesh_path"]):
            metrics_out.append({"provider": r["provider"], "case_id": r["case_id"],
                                "ok": False, "error": r.get("error", "no mesh")})
            continue
        case = case_by_id.get(r["case_id"])
        expect_sym = bool(case and case.expect_symmetry)
        print(f"  [{r['provider']}] {r['case_id']}: analyzing...", flush=True)
        m = analyze_mesh(r["mesh_path"], expect_symmetry=expect_sym).to_dict()
        m.update({"provider": r["provider"], "case_id": r["case_id"]})
        metrics_out.append(m)
        s = score_mesh(m, expect_symmetry=expect_sym)
        s.update({"provider": r["provider"], "case_id": r["case_id"]})
        scores_out.append(s)
        print(f"  [{r['provider']}] {r['case_id']}: production={s['production_score']}")

    _write_json(os.path.join(cfg.out_dir, "metrics.json"), metrics_out)
    _write_json(os.path.join(cfg.out_dir, "scores.json"), scores_out)
    print(f"analyzed {len(scores_out)} meshes -> {cfg.out_dir}/scores.json")


def cmd_semantic(cfg: RunConfig) -> None:
    """Render meshes, build the blind semantic gallery; VLM-judge if configured."""
    from .eval.semantic import build_semantic_gallery, vlm_judge
    gen_path = os.path.join(cfg.out_dir, "generation.json")
    if not os.path.exists(gen_path):
        raise SystemExit("run `generate` first (no generation.json)")
    with open(gen_path) as fh:
        gen = json.load(fh)
    manifest = build_semantic_gallery(gen, cfg.refs_dir, cfg.out_dir,
                                      seed=stable_seed(cfg.run_id))
    print(f"semantic gallery: {len(manifest['items'])} meshes rendered -> "
          f"{cfg.out_dir}/semantic_gallery.html (+ scoresheet CSV)")
    vlm_model = (cfg.extra.get("semantic", {}) or {}).get("vlm_model") \
        or os.environ.get("ARK_VLM_MODEL", "").strip()
    if vlm_model:
        api_key = os.environ.get("ARK_API_KEY", "").strip()
        if not api_key:
            print("semantic: ARK_VLM_MODEL set but ARK_API_KEY missing — skipping judge")
            return
        scores = vlm_judge(manifest, cfg.refs_dir, model=vlm_model, api_key=api_key)
        _write_json(os.path.join(cfg.out_dir, "semantic_scores.json"), scores)
        print(f"semantic: VLM scores -> {cfg.out_dir}/semantic_scores.json")
    else:
        print("semantic: no VLM model configured (set ARK_VLM_MODEL or "
              "config semantic.vlm_model) — human scoresheet only")


def cmd_rig(cfg: RunConfig) -> None:
    """Blender rig smoke test: bind + 45° bend on every generated mesh.

    Each mesh runs in a fresh subprocess (bpy state is global; this gives
    clean state, a hard per-mesh timeout, and crash isolation). Skips
    gracefully when bpy isn't installed.
    """
    import subprocess, sys as _sys
    try:
        import bpy  # noqa: F401
    except Exception:
        print("rig: bpy not installed (pip install bpy) — skipping smoke test")
        return
    gen_path = os.path.join(cfg.out_dir, "generation.json")
    if not os.path.exists(gen_path):
        raise SystemExit("run `generate` first (no generation.json)")
    with open(gen_path) as fh:
        gen = json.load(fh)
    results = []
    for r in gen:
        if not (r.get("ok") and r.get("mesh_path") and os.path.exists(r["mesh_path"])):
            continue
        try:
            proc = subprocess.run(
                [_sys.executable, "-m", "src.eval.rig_probe", r["mesh_path"]],
                capture_output=True, text=True, timeout=300)
            line = (proc.stdout or "").strip().splitlines()
            data = json.loads(line[-1]) if line else {"error": "no output"}
        except subprocess.TimeoutExpired:
            data = {"bind_ok": False, "rig_smoke_score": 0.0, "error": "timeout 300s"}
        except Exception as exc:  # noqa: BLE001
            data = {"bind_ok": False, "rig_smoke_score": 0.0,
                    "error": f"{type(exc).__name__}: {exc}"}
        data.update({"provider": r["provider"], "case_id": r["case_id"]})
        results.append(data)
        print(f"  [rig] {r['provider']}/{r['case_id']}: "
              f"score={data.get('rig_smoke_score')} bind={data.get('bind_ok')} "
              f"{('ERR ' + str(data['error'])) if data.get('error') else ''}")
    _write_json(os.path.join(cfg.out_dir, "rig_scores.json"), results)
    print(f"rig smoke: {len(results)} meshes -> {cfg.out_dir}/rig_scores.json")


def cmd_slice(cfg: RunConfig) -> None:
    """PrusaSlicer ground truth: --info manifold verdict + draft gcode export.

    Two tiers per mesh, both from the referee real print users run: the
    strict ADMesh manifold verdict (the tier vendor pass-rates quote) and the
    lenient "did it slice" (PrusaSlicer auto-repairs silently, so this passes
    more). Skips gracefully when prusa-slicer isn't installed.
    """
    from .eval.slicer_probe import find_slicer, probe
    if not find_slicer():
        print("slice: prusa-slicer not found (apt-get install prusa-slicer) "
              "— skipping slicer ground truth")
        return
    gen_path = os.path.join(cfg.out_dir, "generation.json")
    if not os.path.exists(gen_path):
        raise SystemExit("run `generate` first (no generation.json)")
    with open(gen_path) as fh:
        gen = json.load(fh)
    workdir = os.path.join(cfg.out_dir, "_slice_tmp")
    os.makedirs(workdir, exist_ok=True)
    results = []
    for r in gen:
        if not (r.get("ok") and r.get("mesh_path") and os.path.exists(r["mesh_path"])):
            continue
        data = probe(r["mesh_path"], workdir)
        data.update({"provider": r["provider"], "case_id": r["case_id"]})
        results.append(data)
        print(f"  [slice] {r['provider']}/{r['case_id']}: "
              f"manifold={data.get('slicer_manifold')} gcode={data.get('gcode_ok')} "
              f"open_edges={data.get('open_edges')} "
              f"{('ERR ' + str(data['error'])) if data.get('error') else ''}")
    _write_json(os.path.join(cfg.out_dir, "slicer_scores.json"), results)
    print(f"slicer: {len(results)} meshes -> {cfg.out_dir}/slicer_scores.json")


def cmd_blind(cfg: RunConfig) -> None:
    gen_path = os.path.join(cfg.out_dir, "generation.json")
    with open(gen_path) as fh:
        gen = json.load(fh)
    manifest = build_blind_set(gen, cfg.out_dir, seed=stable_seed(cfg.run_id))
    print(f"blind set: {len(manifest['items'])} meshes anonymised -> "
          f"{cfg.out_dir}/blind/  (score {cfg.out_dir}/blind_scoresheet.csv, "
          f"keep blind_manifest.json sealed)")


def cmd_report(cfg: RunConfig) -> None:
    from .report import build_report
    build_report(cfg)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="blind3d")
    ap.add_argument("command",
                    choices=["refimg", "generate", "analyze", "semantic", "rig", "slice", "blind", "report", "all"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", nargs="*", help="limit generate to these providers")
    ap.add_argument("--max-cases", type=int, default=None,
                    help="cost cap: use at most N cases, sampled evenly across "
                         "categories (for cheap smoke tests)")
    ap.add_argument("--mode", choices=["text", "image"], default=None,
                    help="input mode; 'image' runs every model off the shared "
                         "reference images (overrides config)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="generate each case N times per provider (repeat "
                         "samples share the case's prompt/reference image; "
                         "the report's CI resamples by case, not by repeat)")
    ap.add_argument("--force", action="store_true",
                    help="refimg: regenerate reference images even if cached")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.mode:
        cfg.mode = args.mode
    cases = load_cases(cfg.cases_file)
    if args.max_cases is not None:
        cases = _sample_cases(cases, args.max_cases)
        print(f"cost cap: using {len(cases)} case(s): "
              f"{', '.join(c.id for c in cases)}")

    if args.command == "refimg":
        cmd_refimg(cfg, cases, args.force)
        return

    # image mode: `all` mints refs first; then repoint every case at its shared
    # image so all models consume identical pixels.
    if cfg.mode == "image":
        if args.command == "all":
            cmd_refimg(cfg, cases, args.force)
        if args.command in ("generate", "all"):
            cases = _apply_image_mode(cfg, cases)

    if args.command in ("generate", "all"):
        cmd_generate(cfg, cases, args.only, repeats=args.repeats)
    if args.command in ("analyze", "all"):
        cmd_analyze(cfg, cases)
    if args.command in ("semantic", "all"):
        cmd_semantic(cfg)
    if args.command in ("rig", "all"):
        cmd_rig(cfg)
    if args.command in ("slice", "all"):
        cmd_slice(cfg)
    if args.command in ("blind", "all"):
        cmd_blind(cfg)
    if args.command in ("report", "all"):
        cmd_report(cfg)


if __name__ == "__main__":
    main()
