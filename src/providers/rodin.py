"""Rodin Gen-2.x official API (Deemos Hyperhuman).

Rodin's flow differs from the others: submit returns a job ``uuid`` plus a
``subscription_key``; you poll a status endpoint with that key, then fetch the
asset list with the uuid. Verify against current docs:
  submit:  POST https://hyperhuman.deemos.com/api/v2/rodin   (multipart)
  status:  POST https://hyperhuman.deemos.com/api/v2/status  {subscription_key}
  result:  POST https://hyperhuman.deemos.com/api/v2/download {task_uuid}
Auth: ``Authorization: Bearer <RODIN_API_KEY>``.
"""

from __future__ import annotations

from typing import Any

from .base import Provider, Case, ProviderError

_BASE = "https://hyperhuman.deemos.com/api/v2"

_STATUS_MAP = {
    "Waiting": "pending", "Pending": "pending", "Queued": "pending",
    "Generating": "running", "Running": "running",
    "Done": "succeeded", "Succeeded": "succeeded", "Completed": "succeeded",
    "Failed": "failed", "Error": "failed",
}


class RodinProvider(Provider):
    name = "rodin"
    api_key_env = "RODIN_API_KEY"

    def _tier(self) -> str:
        # Gen-2 tier / quality knob, e.g. "Regular" or "Sketch"
        return self.config.get("tier", "Regular")

    def submit(self, case: Case) -> str:
        fields: dict[str, Any] = {
            "tier": self._tier(),
            "geometry_file_format": "glb",
            "material": "PBR",
            "quality": self.config.get("quality", "high"),
            "mesh_mode": self.config.get("mesh_mode", "Quad"),
        }
        files = None
        if case.mode == "image":
            if not case.image_path:
                raise ProviderError("image mode requires image_path")
            files = {"images": open(case.image_path, "rb")}
        else:
            fields["prompt"] = case.prompt
        resp = self._request("POST", f"{_BASE}/rodin",
                             headers=self._headers(), data=fields, files=files)
        body = resp.json()
        uuid = body.get("uuid") or (body.get("data") or {}).get("uuid")
        sub_key = ((body.get("jobs") or {}).get("subscription_key")
                   or body.get("subscription_key"))
        if not uuid:
            raise ProviderError(f"rodin submit: no uuid in {body}")
        # stash the subscription key on the task id (uuid|subkey) so status()
        # is stateless like the other providers.
        return f"{uuid}|{sub_key or ''}"

    def status(self, task_id: str) -> tuple[str, dict[str, Any]]:
        uuid, _, sub_key = task_id.partition("|")
        resp = self._request("POST", f"{_BASE}/status",
                             headers={**self._headers(), "Content-Type": "application/json"},
                             json={"subscription_key": sub_key})
        data = resp.json()
        jobs = data.get("jobs", data.get("list", []))
        # overall status = worst of sub-jobs; simplest: use top-level if present
        raw_status = str(data.get("status") or (jobs[0].get("status") if jobs else "")).strip()
        norm = _STATUS_MAP.get(raw_status, "running")
        data["_uuid"] = uuid
        return norm, data

    def asset_url(self, raw: dict[str, Any]) -> str | None:
        uuid = raw.get("_uuid")
        if not uuid:
            return None
        # fetch the download manifest for the finished job
        resp = self._request("POST", f"{_BASE}/download",
                             headers={**self._headers(), "Content-Type": "application/json"},
                             json={"task_uuid": uuid})
        items = resp.json().get("list", [])
        for it in items:
            name = str(it.get("name", "")).lower()
            if name.endswith(".glb"):
                return it.get("url")
        # fall back to first url
        return items[0].get("url") if items else None
