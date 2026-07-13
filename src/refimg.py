"""Shared reference-image generation for image-to-3D models.

Most open-source generators on fal (SF3D, TripoSR, InstantMesh, TRELLIS, …) are
*image-to-3D*, while our case set is text prompts. To compare a mixed cohort
fairly we generate ONE reference image per case from a fixed text-to-image model
and feed that identical image to every image-to-3D model. This keeps the
protocol's "identical input set" rule intact at the pixel level — a stronger
guarantee than identical text, since text-to-image variance is factored out.

The reference model (default FLUX schnell on fal) and prompt suffix are pinned
in config so refs are reproducible. Images are cached under ``cases/refs/`` and
only regenerated when missing (or with ``--force``).
"""

from __future__ import annotations

import os
from typing import Any

from .providers.fal_base import FalProvider
from .providers.base import Case

# Nudges the T2I model toward clean, single-object, full-view images that
# image-to-3D models reconstruct well (centered, plain background, no crop).
_REF_SUFFIX = (
    ", single object, centered, full object fully visible, plain light-grey "
    "background, soft even studio lighting, product photograph, no shadows cropped"
)


class RefImageProvider(FalProvider):
    """Text-to-image on fal, used only to mint shared reference images."""

    name = "refimg"
    model_id = "fal-ai/flux/schnell"   # confirmed/overridden via config
    out_format = "png"

    def _payload(self, case: Case) -> dict[str, Any]:
        return {
            "prompt": case.prompt + _REF_SUFFIX,
            "image_size": self.config.get("image_size", "square_hd"),
            "num_images": 1,
            "enable_safety_checker": False,
            **self._extra_params(),
        }

    def asset_url(self, raw: dict[str, Any]) -> str | None:
        imgs = raw.get("images") or (raw.get("output") or {}).get("images")
        if isinstance(imgs, list) and imgs:
            first = imgs[0]
            return first.get("url") if isinstance(first, dict) else first
        # some flux variants return {"image": {"url": ...}}
        img = raw.get("image")
        if isinstance(img, dict):
            return img.get("url")
        return None


def generate_reference_images(cases: list[Case], out_dir: str,
                              config: dict[str, Any] | None = None,
                              *, force: bool = False) -> dict[str, str]:
    """Generate/refresh one reference image per case. Returns {case_id: path}."""
    os.makedirs(out_dir, exist_ok=True)
    prov = RefImageProvider(config or {})
    made: dict[str, str] = {}
    for case in cases:
        dest = os.path.join(out_dir, f"{case.id}.png")
        if os.path.exists(dest) and not force:
            made[case.id] = dest
            print(f"  [refimg] {case.id}: cached")
            continue
        # RefImageProvider.run downloads into <out_dir>/refimg/<id>.png; we want
        # a flat cases/refs/<id>.png, so drive submit/poll/download directly.
        if not prov.api_key:
            print(f"  [refimg] {case.id}: FAIL (missing FAL_KEY)")
            continue
        try:
            task_id = prov.submit(case)
            import time
            deadline = time.monotonic() + prov.timeout_s
            url = None
            while time.monotonic() < deadline:
                state, raw = prov.status(task_id)
                if state == "succeeded":
                    url = prov.asset_url(raw)
                    break
                if state == "failed":
                    break
                time.sleep(prov.poll_interval_s)
            if not url:
                print(f"  [refimg] {case.id}: FAIL (no image url)")
                continue
            prov.download(url, dest)
            made[case.id] = dest
            print(f"  [refimg] {case.id}: ok")
        except Exception as exc:  # noqa: BLE001
            print(f"  [refimg] {case.id}: FAIL ({exc})")
    return made
