"""Seed3D (ByteDance Doubao Seed team) via Volcengine Ark.

Seed3D was briefly hosted on fal but is no longer supported there; the official
channel is the Volcengine Ark platform (ark.cn-beijing.volces.com). Auth is a
plain ``Authorization: Bearer <ARK_API_KEY>``.

The 3D generation API follows Ark's async task pattern (same family as the
Seedance video API):

  create: POST {base}/api/v3/contents/generations/tasks
          {"model": "<model_id>",
           "content": [{"type": "text", "text": "--fileformat glb ..."},
                       {"type": "image_url", "image_url": {"url": <url|data-uri>}}]}
  query:  GET  {base}/api/v3/contents/generations/tasks/{task_id}

Statuses: queued / running / succeeded / failed / cancelled. The succeeded
payload carries the generated file URL; we scan the result for a .glb URL
rather than assuming one exact field name.

NOTE: the exact create/query paths come from Ark's task-API family plus mirror
examples (the official doc page 82379/1856293 is not fetchable from this
sandbox). If Ark answers 404 on the create path, check that doc and adjust
``_TASKS_PATH``. Model id is pinned in config (default Seed3D 1.0:
``doubao-seed3d-1-0-250928``; set the 2.0 id there when you have access).
Requires the model to be ENABLED for your Ark account (方舟控制台 → 开通管理).
"""

from __future__ import annotations

import json
from typing import Any

from .base import Provider, Case, ProviderError, image_to_data_uri, find_mesh_url

_BASE = "https://ark.cn-beijing.volces.com"
_TASKS_PATH = "/api/v3/contents/generations/tasks"

_STATUS_MAP = {
    "queued": "pending", "pending": "pending",
    "running": "running", "processing": "running",
    "succeeded": "succeeded", "success": "succeeded", "completed": "succeeded",
    "failed": "failed", "cancelled": "failed", "canceled": "failed",
    "expired": "failed",
}


class Seed3DProvider(Provider):
    name = "seed3d"
    api_key_env = "ARK_API_KEY"

    def _model_id(self) -> str:
        return self.config.get("model_id", "doubao-seed3d-1-0-250928")

    def _base(self) -> str:
        return self.config.get("base_url", _BASE).rstrip("/")

    def submit(self, case: Case) -> str:
        if case.mode != "image":
            raise ProviderError(
                "seed3d is image-to-3D only — run in image mode (`--mode image`)")
        src = case.image_path or case.notes
        if not src:
            raise ProviderError("seed3d: image mode needs a reference image "
                                "(run `refimg` first)")
        if not (src.startswith("http://") or src.startswith("https://")):
            src = image_to_data_uri(src)
        # generation options ride in a --flag string, Ark task-API style
        opts = self.config.get("options", "--fileformat glb")
        payload = {
            "model": self._model_id(),
            "content": [
                {"type": "text", "text": opts},
                {"type": "image_url", "image_url": {"url": src}},
            ],
        }
        resp = self._request(
            "POST", f"{self._base()}{_TASKS_PATH}",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json=payload)
        body = resp.json()
        task_id = body.get("id") or body.get("task_id") or (body.get("data") or {}).get("id")
        if not task_id:
            raise ProviderError(f"seed3d submit: no task id in {resp.text[:200]}")
        return str(task_id)

    def status(self, task_id: str) -> tuple[str, dict[str, Any]]:
        resp = self._request(
            "GET", f"{self._base()}{_TASKS_PATH}/{task_id}",
            headers={"Authorization": f"Bearer {self.api_key}"})
        data = resp.json()
        raw_status = str(data.get("status", "")).lower()
        return _STATUS_MAP.get(raw_status, "running"), data

    def asset_url(self, raw: dict[str, Any]) -> str | None:
        found = find_mesh_url(raw)
        if found:
            return found
        # Ark hands back presigned TOS URLs that may carry no file extension —
        # run 12 generated (and billed) successfully but the extension-based
        # deep scan missed the URL. Fall back to the first http(s) URL under a
        # file-ish key, then to any http(s) URL in the payload.
        def _any_url(obj: Any, keyed: bool) -> str | None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and v.startswith("http") and (
                            not keyed or any(t in k.lower() for t in
                                             ("file", "url", "model", "result"))):
                        return v
                for v in obj.values():
                    got = _any_url(v, keyed)
                    if got:
                        return got
            if isinstance(obj, list):
                for v in obj:
                    got = _any_url(v, keyed)
                    if got:
                        return got
            return None
        return _any_url(raw, keyed=True) or _any_url(raw, keyed=False)
