"""Shared base for models without a first-party hosted REST API.

Hunyuan3D and TRELLIS are open-weights models. To keep the protocol's "official
API" spirit while staying runnable, we call them through fal.ai's hosted
endpoints (the canonical managed deployment for both). A self-hosted or
Replicate deployment can be dropped in by subclassing and overriding
``model_id`` / ``_result_url``.

fal queue API:
  submit: POST https://queue.fal.run/{model_id}                 -> {request_id}
  status: GET  https://queue.fal.run/{model_id}/requests/{id}/status
  result: GET  https://queue.fal.run/{model_id}/requests/{id}
Auth: ``Authorization: Key <FAL_KEY>``.
"""

from __future__ import annotations

from typing import Any

from .base import Provider, Case, ProviderError

_STATUS_MAP = {
    "IN_QUEUE": "pending", "IN_PROGRESS": "running",
    "COMPLETED": "succeeded", "FAILED": "failed", "ERROR": "failed",
}


class FalProvider(Provider):
    api_key_env = "FAL_KEY"
    model_id = ""  # e.g. "fal-ai/hunyuan3d-v2" — set by subclass/config

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self.api_key}"}

    def _endpoint(self) -> str:
        mid = self.config.get("model_id", self.model_id)
        if not mid:
            raise ProviderError(f"{self.name}: model_id not configured")
        return f"https://queue.fal.run/{mid}"

    def _payload(self, case: Case) -> dict[str, Any]:
        """Build the model input. Override per model if field names differ."""
        if case.mode == "image":
            if not case.image_path and not case.notes:
                raise ProviderError("image mode requires image_path or image url in notes")
            return {"image_url": case.notes or case.image_path}
        return {"prompt": case.prompt}

    def submit(self, case: Case) -> str:
        resp = self._request("POST", self._endpoint(),
                             headers={**self._headers(), "Content-Type": "application/json"},
                             json=self._payload(case))
        rid = resp.json().get("request_id")
        if not rid:
            raise ProviderError(f"{self.name} submit: no request_id in {resp.text[:200]}")
        return rid

    def status(self, task_id: str) -> tuple[str, dict[str, Any]]:
        base = self._endpoint()
        resp = self._request("GET", f"{base}/requests/{task_id}/status",
                             headers=self._headers())
        data = resp.json()
        norm = _STATUS_MAP.get(str(data.get("status", "")).upper(), "running")
        if norm == "succeeded":
            # fetch full result payload which carries the asset url
            r2 = self._request("GET", f"{base}/requests/{task_id}", headers=self._headers())
            data = r2.json()
        return norm, data

    def asset_url(self, raw: dict[str, Any]) -> str | None:
        # common fal shapes: {"model_mesh":{"url":..}} or {"mesh":{"url":..}}
        for key in ("model_mesh", "mesh", "model_glb", "glb"):
            val = raw.get(key)
            if isinstance(val, dict) and val.get("url"):
                return val["url"]
            if isinstance(val, str) and val.startswith("http"):
                return val
        # sometimes nested under "output"
        out = raw.get("output") or {}
        for key in ("model_mesh", "mesh", "glb"):
            val = out.get(key)
            if isinstance(val, dict) and val.get("url"):
                return val["url"]
        return None
