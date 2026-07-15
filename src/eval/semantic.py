"""Semantic-fidelity check: does the mesh depict what was asked?

Geometry metrics are blind to content — this closes that hole two ways:

1. **Blind human gallery** (always): every successful mesh is rendered to
   multi-view PNGs, shuffled under anonymous ids (same seeded shuffle as the
   mesh blind set), and written into a single self-contained HTML page showing
   reference image vs renders side by side, plus a CSV scoresheet keyed only by
   blind id. A human can grade prompt-match in minutes without knowing which
   model made what.

2. **VLM judge** (optional): if a vision model is configured (Volcengine Ark
   chat-completions, reuses ARK_API_KEY), each mesh's views are sent with the
   reference image and scored 1-5 for match; results land in
   ``semantic_scores.json`` and the report can join them. Off by default —
   enable via config ``semantic: {vlm_model: "<ark-vision-model-id>"}``.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

from ..render import render_views
from .blind import _seeded_order


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _img_tag(path: str, width: int = 200) -> str:
    return (f'<img src="data:image/png;base64,{_b64(path)}" '
            f'width="{width}" loading="lazy">')


def build_semantic_gallery(results: list[dict[str, Any]], refs_dir: str,
                           out_dir: str, *, seed: int) -> dict[str, Any]:
    """Render every successful mesh and emit blind gallery + scoresheet."""
    ok = [r for r in results if r.get("ok") and r.get("mesh_path")
          and os.path.exists(r["mesh_path"])]
    order = _seeded_order(len(ok), seed)
    render_dir = os.path.join(out_dir, "renders")
    manifest: dict[str, Any] = {"seed": seed, "items": []}
    rows_html, rows_csv = [], []

    for blind_idx, src_idx in enumerate(order):
        r = ok[src_idx]
        blind_id = f"sem_{blind_idx:04d}"
        try:
            views = render_views(r["mesh_path"], render_dir, blind_id)
        except Exception as exc:  # noqa: BLE001 - render failure shouldn't kill the batch
            print(f"  [semantic] render failed for {r['provider']}/{r['case_id']}: {exc}")
            continue
        ref = os.path.join(refs_dir, f"{r['case_id']}.png")
        ref_html = _img_tag(ref) if os.path.exists(ref) else "<i>no ref</i>"
        views_html = "".join(_img_tag(v, 170) for v in views)
        rows_html.append(
            f"<tr><td><b>{blind_id}</b></td><td>{ref_html}</td>"
            f"<td>{views_html}</td>"
            f"<td>1&nbsp;2&nbsp;3&nbsp;4&nbsp;5</td></tr>")
        rows_csv.append({"blind_id": blind_id, "match_1to5": "", "notes": ""})
        manifest["items"].append({"blind_id": blind_id, "provider": r["provider"],
                                  "case_id": r["case_id"], "views": views})

    html = ("<!doctype html><meta charset='utf-8'>"
            "<title>Semantic blind review</title>"
            "<style>table{border-collapse:collapse}td{border:1px solid #ccc;"
            "padding:6px;vertical-align:top}</style>"
            "<h2>Blind semantic review — does the mesh match the reference?</h2>"
            "<p>Grade each row 1-5 in the CSV (semantic_scoresheet.csv). "
            "Do not open semantic_manifest.json until done.</p>"
            "<table><tr><th>id</th><th>reference</th><th>mesh views</th>"
            "<th>match?</th></tr>" + "".join(rows_html) + "</table>")
    with open(os.path.join(out_dir, "semantic_gallery.html"), "w") as fh:
        fh.write(html)
    with open(os.path.join(out_dir, "semantic_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    import csv
    with open(os.path.join(out_dir, "semantic_scoresheet.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["blind_id", "match_1to5", "notes"])
        w.writeheader()
        w.writerows(rows_csv)
    return manifest


# ---------------------------------------------------------------------------
# optional VLM judge (Volcengine Ark chat completions, vision model)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = (
    "第一张图是参考图,后面几张是某个3D模型的多视角渲染(灰色素模,无纹理)。"
    "只根据形状/结构判断:这个3D模型是否忠实还原了参考图中的物体?"
    "1=完全不相关, 2=类别对但结构错, 3=大体相似但缺关键部件, "
    "4=结构忠实有小瑕疵, 5=高度忠实。只回答一个数字。")


def vlm_judge(manifest: dict[str, Any], refs_dir: str, *, model: str,
              api_key: str, base_url: str = "https://ark.cn-beijing.volces.com",
              session=None, dedupe_by_case: bool = True) -> list[dict[str, Any]]:
    """Score each manifest item 1-5 with an Ark vision model.

    ``dedupe_by_case`` judges only the first mesh per (provider, case_id):
    semantic fidelity is a property of the prompt-to-shape match, not of
    which repeat generation we happened to render, so scoring every repeat
    just multiplies the Ark bill (which shares the account balance with
    Seed3D generation) for no extra signal.
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    sess = session or requests.Session()

    # de-dupe up front so the parallel pool sizes to the real work
    todo = []
    seen: set[tuple[str, str]] = set()
    for item in manifest["items"]:
        if dedupe_by_case:
            key = (item["provider"], item["case_id"])
            if key in seen:
                continue
            seen.add(key)
        todo.append(item)

    def _judge(item: dict[str, Any]) -> dict[str, Any]:
        ref = os.path.join(refs_dir, f"{item['case_id']}.png")
        content: list[dict[str, Any]] = [{"type": "text", "text": _JUDGE_PROMPT}]
        for p in [ref] + item["views"]:
            if os.path.exists(p):
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{_b64(p)}"}})
        try:
            resp = sess.post(
                f"{base_url}/api/v3/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [{"role": "user", "content": content}]},
                timeout=60)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            m = re.search(r"[1-5]", text)
            score = int(m.group(0)) if m else None
        except Exception as exc:  # noqa: BLE001
            # surface the Ark error body: a bare 404 is undiagnosable, the
            # body says whether the model id is wrong vs not activated
            body = ""
            r = getattr(exc, "response", None)
            if r is not None:
                body = f" | body: {(r.text or '')[:300]}"
            print(f"  [semantic] vlm failed for {item['blind_id']}: {exc}{body}")
            score = None
        print(f"  [semantic] {item['provider']}/{item['case_id']}: vlm={score}")
        return {**{k: item[k] for k in ("blind_id", "provider", "case_id")},
                "vlm_match_1to5": score}

    # network-bound; fan out so a 40-case cohort isn't 160 serial round-trips.
    # requests.Session is thread-safe for concurrent requests to the same host.
    out = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(todo)))) as ex:
        futs = [ex.submit(_judge, it) for it in todo]
        for fut in as_completed(futs):
            out.append(fut.result())
    return out
