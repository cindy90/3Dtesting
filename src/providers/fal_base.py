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

import os
from typing import Any

from .base import Provider, Case, ProviderError, image_to_data_uri

_STATUS_MAP = {
    "IN_QUEUE": "pending", "IN_PROGRESS": "running",
    "COMPLETED": "succeeded", "FAILED": "failed", "ERROR": "failed",
}


class FalProvider(Provider):
    api_key_env = "FAL_KEY"
    model_id = ""  # e.g. "fal-ai/hunyuan3d/v2" — set by subclass/config
    #: input field name for the reference image (overridable per model)
    image_field = "image_url"
    #: input field name for a text prompt (few fal 3D models are text-native)
    text_field = "prompt"
    #: fallback image field names to try if the primary is rejected (422).
    #: fal endpoints vary — some use image_url, Hunyuan uses input_image_url —
    #: and the exact schema isn't always publicly fetchable, so we self-heal.
    image_field_candidates = ("image_url", "input_image_url", "image", "images")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self.api_key}"}

    def _endpoint(self) -> str:
        mid = self.config.get("model_id", self.model_id)
        if not mid:
            raise ProviderError(f"{self.name}: model_id not configured")
        return f"https://queue.fal.run/{mid}"

    def _image_ref(self, case: Case) -> str:
        """Resolve the case's image input to a URL or data URI for fal."""
        src = case.image_path or case.notes
        if not src:
            raise ProviderError(
                f"{self.name}: image mode needs a reference image "
                f"(run `refimg` first, or set image_path/notes on the case)")
        if src.startswith("http://") or src.startswith("https://"):
            return src
        if not os.path.exists(src):
            raise ProviderError(f"{self.name}: reference image not found: {src}")
        return image_to_data_uri(src)

    def _extra_params(self) -> dict[str, Any]:
        """Per-model extra input params, supplied via config['params']."""
        return dict(self.config.get("params", {}) or {})

    def _payload(self, case: Case, image_field: str | None = None) -> dict[str, Any]:
        """Build the model input. Override per model if field names differ."""
        if case.mode == "image":
            field = image_field or self.image_field
            return {field: self._image_ref(case), **self._extra_params()}
        return {self.text_field: case.prompt, **self._extra_params()}

    def _post(self, payload: dict[str, Any]) -> str:
        resp = self._request("POST", self._endpoint(),
                             headers={**self._headers(), "Content-Type": "application/json"},
                             json=payload)
        rid = resp.json().get("request_id")
        if not rid:
            raise ProviderError(f"{self.name} submit: no request_id in {resp.text[:200]}")
        return rid

    def submit(self, case: Case) -> str:
        # text mode (or non-image): single shot
        if case.mode != "image":
            return self._post(self._payload(case))
        # image mode: try the configured field, then alternates on a 422/400
        # schema-validation error, so a wrong field-name guess self-heals.
        primary = self.image_field
        candidates = [primary] + [f for f in self.image_field_candidates if f != primary]
        last: ProviderError | None = None
        for field in candidates:
            try:
                return self._post(self._payload(case, image_field=field))
            except ProviderError as exc:
                msg = str(exc)
                if "client error 4" in msg or "422" in msg or "400" in msg:
                    last = exc
                    continue  # likely wrong field name — try the next
                raise  # auth / network / other — don't mask it
        raise last or ProviderError(f"{self.name}: submit failed for all image fields")

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
