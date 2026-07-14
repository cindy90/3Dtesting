"""Meshy 6 official API (api.meshy.ai).

Text-to-3D is a two-stage flow: a ``preview`` task builds base geometry, then a
``refine`` task adds texture. We run both so the scored asset matches what a
customer would ship. Verify endpoints/model id against current docs:
  submit preview: POST https://api.meshy.ai/openapi/v2/text-to-3d
  refine:         POST https://api.meshy.ai/openapi/v2/text-to-3d {mode:refine, preview_task_id}
  status:         GET  https://api.meshy.ai/openapi/v2/text-to-3d/{id}
  image-to-3d:    POST https://api.meshy.ai/openapi/v1/image-to-3d
Auth: ``Authorization: Bearer <MESHY_API_KEY>``.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .base import Provider, Case, ProviderError, image_to_data_uri

_BASE = "https://api.meshy.ai/openapi"

_STATUS_MAP = {
    "PENDING": "pending", "IN_PROGRESS": "running",
    "SUCCEEDED": "succeeded", "FAILED": "failed", "CANCELED": "failed", "EXPIRED": "failed",
}


class MeshyProvider(Provider):
    name = "meshy"
    api_key_env = "MESHY_API_KEY"

    def _ai_model(self) -> str:
        return self.config.get("ai_model", "meshy-6")

    def _submit_preview(self, case: Case) -> str:
        if case.mode == "image":
            src = case.image_path or case.notes
            if not src:
                raise ProviderError("image mode requires a reference image "
                                    "(run `refimg` first)")
            # Meshy image-to-3D accepts a public URL or a base64 data URI; encode
            # the local shared reference image so every model gets identical bytes.
            if src.startswith("http://") or src.startswith("https://"):
                image_url = src
            elif os.path.exists(src):
                image_url = image_to_data_uri(src)
            else:
                raise ProviderError(f"meshy: reference image not found: {src}")
            payload = {"image_url": image_url, "ai_model": self._ai_model(),
                       "enable_pbr": True}
            url = f"{_BASE}/v1/image-to-3d"
        else:
            payload = {
                "mode": "preview",
                "prompt": case.prompt,
                "ai_model": self._ai_model(),
                "topology": "quad",       # request quad topology when available
                "target_polycount": self.config.get("target_polycount", 30000),
            }
            url = f"{_BASE}/v2/text-to-3d"
        resp = self._request("POST", url,
                             headers={**self._headers(), "Content-Type": "application/json"},
                             json=payload)
        tid = resp.json().get("result")
        if not tid:
            raise ProviderError(f"meshy submit: no result id in {resp.text[:200]}")
        return tid

    def _wait(self, task_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            state, raw = self.status(task_id)
            if state == "succeeded":
                return raw
            if state == "failed":
                raise ProviderError(f"meshy task {task_id} failed")
            time.sleep(self.poll_interval_s)
        raise ProviderError(f"meshy task {task_id} timed out")

    def submit(self, case: Case) -> str:
        preview_id = self._submit_preview(case)
        if case.mode == "image":
            return preview_id  # image flow is single-stage
        # wait for preview, then kick off refine (texture) and return its id
        self._wait(preview_id)
        resp = self._request("POST", f"{_BASE}/v2/text-to-3d",
                             headers={**self._headers(), "Content-Type": "application/json"},
                             json={"mode": "refine", "preview_task_id": preview_id,
                                   "enable_pbr": True})
        refine_id = resp.json().get("result")
        if not refine_id:
            raise ProviderError("meshy refine: no result id")
        return refine_id

    def status(self, task_id: str) -> tuple[str, dict[str, Any]]:
        # try v2 then v1 endpoint (image tasks live under v1)
        for ver in ("v2/text-to-3d", "v1/image-to-3d"):
            try:
                resp = self._request("GET", f"{_BASE}/{ver}/{task_id}", headers=self._headers())
                data = resp.json()
                raw_status = str(data.get("status", "")).upper()
                if raw_status:
                    return _STATUS_MAP.get(raw_status, "running"), data
            except ProviderError:
                continue
        raise ProviderError(f"meshy status: task {task_id} not found")

    def asset_url(self, raw: dict[str, Any]) -> str | None:
        urls = raw.get("model_urls", {}) or {}
        return urls.get("glb") or urls.get("fbx") or urls.get("obj")
