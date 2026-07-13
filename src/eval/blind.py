"""Blind-evaluation support.

The objective metrics need no blinding — geometry can't be biased. But the
protocol also allows a light human spot-check (does the mesh actually match the
prompt? any gross artefact the metrics miss?). For that pass we must prevent the
"it's the model I like" bias, so this module:

  1. copies every generated mesh to an anonymised id (``blind_0007.glb``) with a
     shuffled order, writing a sealed manifest that maps blind id -> (provider,
     case);
  2. emits a blank scoresheet keyed only by blind id;
  3. after a human fills it in, joins the scores back to providers.

The shuffle is seeded from the run id (passed in) so it is reproducible without
using wall-clock randomness.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from typing import Any


def _seeded_order(n: int, seed: int) -> list[int]:
    """Deterministic Fisher-Yates shuffle of range(n) from an integer seed."""
    order = list(range(n))
    state = (seed ^ 0x9E3779B9) & 0xFFFFFFFF
    for i in range(n - 1, 0, -1):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF  # LCG
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def build_blind_set(results: list[dict[str, Any]], out_dir: str, *, seed: int) -> dict[str, Any]:
    """Copy successful meshes under anonymised ids and write the sealed manifest.

    ``results`` items are GenResult dicts with at least ``provider``,
    ``case_id``, ``ok``, ``mesh_path``. Returns the manifest dict.
    """
    ok = [r for r in results if r.get("ok") and r.get("mesh_path")]
    order = _seeded_order(len(ok), seed)
    blind_dir = os.path.join(out_dir, "blind")
    os.makedirs(blind_dir, exist_ok=True)

    manifest: dict[str, Any] = {"seed": seed, "items": []}
    scoresheet_rows = []
    for blind_idx, src_idx in enumerate(order):
        r = ok[src_idx]
        ext = os.path.splitext(r["mesh_path"])[1] or ".glb"
        blind_id = f"blind_{blind_idx:04d}"
        dest = os.path.join(blind_dir, f"{blind_id}{ext}")
        shutil.copyfile(r["mesh_path"], dest)
        manifest["items"].append({
            "blind_id": blind_id,
            "file": os.path.basename(dest),
            "provider": r["provider"],
            "case_id": r["case_id"],
        })
        scoresheet_rows.append({"blind_id": blind_id, "file": os.path.basename(dest),
                                "prompt_match_1to5": "", "gross_artefact_y_n": "",
                                "notes": ""})

    # sealed manifest (do NOT open before scoring)
    with open(os.path.join(out_dir, "blind_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    # blank scoresheet for the human rater
    sheet_path = os.path.join(out_dir, "blind_scoresheet.csv")
    with open(sheet_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scoresheet_rows[0].keys())
                           if scoresheet_rows else
                           ["blind_id", "file", "prompt_match_1to5",
                            "gross_artefact_y_n", "notes"])
        w.writeheader()
        w.writerows(scoresheet_rows)
    return manifest


def join_blind_scores(manifest_path: str, scoresheet_path: str) -> list[dict[str, Any]]:
    """De-anonymise a filled scoresheet back to (provider, case) rows."""
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    lookup = {it["blind_id"]: it for it in manifest["items"]}
    joined = []
    with open(scoresheet_path, newline="") as fh:
        for row in csv.DictReader(fh):
            meta = lookup.get(row["blind_id"])
            if not meta:
                continue
            joined.append({
                "provider": meta["provider"],
                "case_id": meta["case_id"],
                "blind_id": row["blind_id"],
                "prompt_match_1to5": row.get("prompt_match_1to5", ""),
                "gross_artefact_y_n": row.get("gross_artefact_y_n", ""),
                "notes": row.get("notes", ""),
            })
    return joined
