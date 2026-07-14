"""Common provider interface + submit/poll/download plumbing.

Each concrete provider (Tripo, Meshy, Rodin, Hunyuan3D, TRELLIS) subclasses
``Provider`` and fills in four things: how to submit a job, how to read job
status, how to find the finished asset URL, and any auth header quirks. The
polling loop, backoff, and download are shared so the per-provider files stay
small and the *protocol* (same input, official API, one asset per case) is
enforced in one place.

Nothing here is model-specific on purpose — swapping a model version is a
config change, not a code change.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from dataclasses import dataclass, field
from typing import Any


def image_to_data_uri(path: str) -> str:
    """Base64-encode a local image as a ``data:`` URI.

    Shared by every provider that consumes the local *shared reference image*
    in image mode (fal cohort, Meshy, …). Most image-to-3D APIs accept a data
    URI in their image field, so we avoid a separate upload round-trip and keep
    the input byte-identical across models.
    """
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

import requests


class ProviderError(RuntimeError):
    pass


@dataclass
class Case:
    """One protocol input, identical across every model."""

    id: str
    category: str            # character | prop | organic | hardsurface
    prompt: str
    mode: str = "text"       # text | image
    image_path: str | None = None
    expect_watertight: bool = True
    expect_symmetry: bool = False   # true for bilateral characters
    notes: str = ""


@dataclass
class GenResult:
    """Outcome of running one case through one provider."""

    provider: str
    case_id: str
    ok: bool
    mesh_path: str | None = None
    task_id: str | None = None
    latency_s: float | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class Provider:
    """Base class. Subclasses set ``name`` and implement the hooks below."""

    name: str = "base"
    #: env var holding the API key
    api_key_env: str = ""
    #: output format we request from the API (glb preferred: single-file, PBR)
    out_format: str = "glb"

    # polling controls
    poll_interval_s: float = 6.0
    timeout_s: float = 900.0

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.api_key = os.environ.get(self.api_key_env, "").strip()
        self.session = requests.Session()

    # --- hooks each provider implements ---------------------------------
    def submit(self, case: Case) -> str:
        """Submit a generation job, return a task id."""
        raise NotImplementedError

    def status(self, task_id: str) -> tuple[str, dict[str, Any]]:
        """Return (normalised_status, raw_payload).

        normalised_status is one of: 'pending', 'running', 'succeeded',
        'failed'.
        """
        raise NotImplementedError

    def asset_url(self, raw: dict[str, Any]) -> str | None:
        """Extract the downloadable mesh URL from a succeeded payload."""
        raise NotImplementedError

    # --- shared plumbing -------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _request(self, method: str, url: str, **kw) -> requests.Response:
        """HTTP with bounded exponential backoff on *transient* failures only.

        Terminal (never retried), raised immediately with an actionable message:
          * 401/403 — bad/inactive key or no billing
          * other 4xx except 429 — e.g. 422 schema validation (callers may catch
            this to try an alternate request shape)
        Retried with backoff: network errors, 429, and 5xx.
        """
        delay = 2.0
        last: Exception | None = None
        for attempt in range(5):
            try:
                resp = self.session.request(method, url, timeout=60, **kw)
            except Exception as exc:  # noqa: BLE001 - network error, retry
                last = exc
            else:
                sc = resp.status_code
                if sc == 401:
                    raise ProviderError(
                        f"auth failed (401) — {self.api_key_env} is invalid or "
                        f"rejected. Server said: {resp.text[:250]}")
                if sc == 403:
                    # 403 covers both permission and (for Tripo) out-of-credit,
                    # so surface the body verbatim rather than guessing.
                    raise ProviderError(
                        f"forbidden (403) — key valid but request denied "
                        f"(often: account out of credit/billing). "
                        f"Server said: {resp.text[:250]}")
                if 400 <= sc < 500 and sc != 429:
                    hint = ""
                    if sc == 405:  # method not allowed — usually a redirect ate the POST
                        hint = (f" [Allow={resp.headers.get('Allow','?')}, "
                                f"Location={resp.headers.get('Location','none')}]")
                    raise ProviderError(f"client error {sc}: {resp.text[:250]}{hint}")
                if sc in (429, 500, 502, 503, 504):
                    last = ProviderError(f"transient {sc}: {resp.text[:200]}")
                else:
                    return resp
            if attempt == 4:
                break
            time.sleep(delay)
            delay *= 2
        raise ProviderError(f"{self.name}: request failed: {last}")

    def download(self, url: str, dest: str) -> str:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        resp = self._request("GET", url, stream=True)
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)
        return dest

    def run(self, case: Case, out_dir: str) -> GenResult:
        """Full lifecycle for one case: submit -> poll -> download."""
        if not self.api_key:
            return GenResult(self.name, case.id, ok=False,
                             error=f"missing API key ({self.api_key_env})")
        start = time.monotonic()
        try:
            task_id = self.submit(case)
        except Exception as exc:  # noqa: BLE001
            return GenResult(self.name, case.id, ok=False,
                             error=f"submit failed: {exc}")

        deadline = start + self.timeout_s
        raw: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                state, raw = self.status(task_id)
            except Exception as exc:  # noqa: BLE001
                return GenResult(self.name, case.id, ok=False, task_id=task_id,
                                 error=f"status failed: {exc}")
            if state == "succeeded":
                url = self.asset_url(raw)
                if not url:
                    return GenResult(self.name, case.id, ok=False, task_id=task_id,
                                     error="succeeded but no asset url", raw=raw)
                dest = os.path.join(out_dir, self.name, f"{case.id}.{self.out_format}")
                try:
                    path = self.download(url, dest)
                except Exception as exc:  # noqa: BLE001
                    return GenResult(self.name, case.id, ok=False, task_id=task_id,
                                     error=f"download failed: {exc}", raw=raw)
                return GenResult(self.name, case.id, ok=True, mesh_path=path,
                                 task_id=task_id, latency_s=time.monotonic() - start,
                                 raw=raw)
            if state == "failed":
                return GenResult(self.name, case.id, ok=False, task_id=task_id,
                                 error="generation failed", raw=raw)
            time.sleep(self.poll_interval_s)

        return GenResult(self.name, case.id, ok=False, task_id=task_id,
                         error=f"timeout after {self.timeout_s:.0f}s", raw=raw)
