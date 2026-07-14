"""Tripo 3.x official API (api.tripo3d.ai).

Docs move fast — verify the model_version string and endpoints against the
current Tripo API reference before a run. As of writing:
  submit:  POST https://api.tripo3d.ai/v2/openapi/task
  status:  GET  https://api.tripo3d.ai/v2/openapi/task/{task_id}
  upload:  POST https://api.tripo3d.ai/v2/openapi/upload   (image mode)
Auth is ``Authorization: Bearer <TRIPO_API_KEY>``.
"""

from __future__ import annotations

import os
from typing import Any

from .base import Provider, Case, ProviderError

_BASE = "https://api.tripo3d.ai/v2/openapi"

_STATUS_MAP = {
    "queued": "pending", "waiting": "pending",
    "running": "running", "processing": "running",
    "success": "succeeded",
    "failed": "failed", "cancelled": "failed", "expired": "failed", "banned": "failed",
}


class TripoProvider(Provider):
    name = "tripo"
    api_key_env = "TRIPO_API_KEY"

    def _model_version(self) -> str:
        # e.g. "v2.5-20250123" — pin the 3.x model you want to benchmark.
        return self.config.get("model_version", "v2.5-20250123")

    def _upload_image(self, path: str) -> str:
        with open(path, "rb") as fh:
            resp = self._request("POST", f"{_BASE}/upload",
                                  headers=self._headers(),
                                  files={"file": fh})
        data = resp.json().get("data", {})
        token = data.get("image_token") or data.get("token")
        if not token:
            raise ProviderError(f"tripo upload returned no token: {data}")
        return token

    def submit(self, case: Case) -> str:
        if case.mode == "image":
            if not case.image_path:
                raise ProviderError("image mode requires image_path")
            token = self._upload_image(case.image_path)
            # file type must match the uploaded bytes (our shared refs are PNG)
            ext = os.path.splitext(case.image_path)[1].lstrip(".").lower() or "png"
            payload: dict[str, Any] = {
                "type": "image_to_model",
                "file": {"type": {"jpeg": "jpg"}.get(ext, ext), "file_token": token},
            }
        else:
            payload = {"type": "text_to_model", "prompt": case.prompt}
        payload["model_version"] = self._model_version()
        # ask for a clean, textured, quad-friendly asset where supported
        payload.setdefault("texture", True)
        payload.setdefault("pbr", True)
        resp = self._request("POST", f"{_BASE}/task",
                             headers={**self._headers(), "Content-Type": "application/json"},
                             json=payload)
        body = resp.json()
        task_id = (body.get("data") or {}).get("task_id")
        if not task_id:
            raise ProviderError(f"tripo submit: no task_id in {body}")
        return task_id

    def status(self, task_id: str) -> tuple[str, dict[str, Any]]:
        resp = self._request("GET", f"{_BASE}/task/{task_id}", headers=self._headers())
        data = resp.json().get("data", {})
        raw_status = str(data.get("status", "")).lower()
        return _STATUS_MAP.get(raw_status, "running"), data

    def asset_url(self, raw: dict[str, Any]) -> str | None:
        out = raw.get("output", {}) or {}
        # prefer the PBR/textured model, fall back to base model
        for key in ("pbr_model", "model", "base_model"):
            val = out.get(key)
            if isinstance(val, str) and val:
                return val
            if isinstance(val, dict) and val.get("url"):
                return val["url"]
        return None
