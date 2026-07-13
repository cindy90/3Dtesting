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

import base64
import mimetypes
import os
from typing import Any

from .base import Provider, Case, ProviderError

_STATUS_MAP = {
    "IN_QUEUE": "pending", "IN_PROGRESS": "running",
    "COMPLETED": "succeeded", "FAILED": "failed", "ERROR": "failed",
}


def image_to_data_uri(path: str) -> str:
    """Base64-encode a local image as a data URI.

    Most fal image-to-3D endpoints accept a data URI in the ``image_url`` field,
    which lets us feed a locally-generated *shared reference image* without a
    separate upload round-trip — keeping the input identical across every
    image-to-3D model in the cohort.
    """
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class FalProvider(Provider):
    api_key_env = "FAL_KEY"
    model_id = ""  # e.g. "fal-ai/hunyuan3d/v2" — set by subclass/config
    #: input field name for the reference image (overridable per model)
    image_field = "image_url"
    #: input field name for a text prompt (few fal 3D models are text-native)
    text_field = "prompt"

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

    def _payload(self, case: Case) -> dict[str, Any]:
        """Build the model input. Override per model if field names differ."""
        if case.mode == "image":
            return {self.image_field: self._image_ref(case), **self._extra_params()}
        return {self.text_field: case.prompt, **self._extra_params()}

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
