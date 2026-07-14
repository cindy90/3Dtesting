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


def cmd_generate(cfg: RunConfig, cases: list[Case], only: list[str] | None) -> None:
    results: list[dict[str, Any]] = []
    provider_keys = [k for k in cfg.providers if not only or k in only]
    print(f"generating {len(cases)} cases x {len(provider_keys)} providers "
          f"[mode={cfg.mode}]")

    def _one(pkey: str, case: Case) -> dict[str, Any]:
        prov = build_provider(pkey, cfg.providers[pkey])
        # per-provider timeout override (e.g. seed3d regularly exceeds 900s)
        prov.timeout_s = float((cfg.providers[pkey] or {}).get("timeout_s",
                                                               cfg.timeout_s))
        prov.poll_interval_s = cfg.poll_interval_s
        res = prov.run(case, cfg.out_dir)
        status = "ok" if res.ok else f"FAIL({res.error})"
        print(f"  [{pkey}] {case.id}: {status}")
        return {
            "provider": res.provider, "case_id": res.case_id, "ok": res.ok,
            "mesh_path": res.mesh_path, "mesh_url": res.mesh_url,
            "task_id": res.task_id,
            "latency_s": res.latency_s, "error": res.error,
        }

    # modest parallelism: distinct providers run concurrently, cases serial
    # within a provider is unnecessary — the APIs are async — so fan out all.
    jobs = [(p, c) for p in provider_keys for c in cases]
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(jobs)))) as ex:
        futs = {ex.submit(_one, p, c): (p, c.id) for p, c in jobs}
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
                    choices=["refimg", "generate", "analyze", "semantic", "blind", "report", "all"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", nargs="*", help="limit generate to these providers")
    ap.add_argument("--max-cases", type=int, default=None,
                    help="cost cap: use at most N cases, sampled evenly across "
                         "categories (for cheap smoke tests)")
    ap.add_argument("--mode", choices=["text", "image"], default=None,
                    help="input mode; 'image' runs every model off the shared "
                         "reference images (overrides config)")
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
        cmd_generate(cfg, cases, args.only)
    if args.command in ("analyze", "all"):
        cmd_analyze(cfg, cases)
    if args.command in ("semantic", "all"):
        cmd_semantic(cfg)
    if args.command in ("blind", "all"):
        cmd_blind(cfg)
    if args.command in ("report", "all"):
        cmd_report(cfg)


if __name__ == "__main__":
    main()
