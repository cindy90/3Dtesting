"""Rodin (影眸/Deemos Hyper3D) via fal.ai hosted endpoint.

Alternative to the direct Deemos API (`rodin` provider, RODIN_API_KEY): fal
hosts Rodin at `fal-ai/hyper3d/rodin` with versioned subpaths — we default to
the newest, `fal-ai/hyper3d/rodin/v2.5`. Uses the shared FAL_KEY, ~$0.4/gen.

Schema quirks vs the other fal models (confirmed by the live 422 validation
error from Rodin Gen-2.5 itself: ``loc: ["body","image_urls"]``):
  * image input is ``image_urls`` — an ARRAY of urls/data-URIs
  * knobs: geometry_file_format (glb), material (PBR), quality
Output GLB at ``model_mesh.url`` (handled by the base asset_url).
"""

from __future__ import annotations

from typing import Any

from .fal_base import FalProvider
from .base import Case, ProviderError


class RodinFalProvider(FalProvider):
    name = "rodin_fal"
    model_id = "fal-ai/hyper3d/rodin/v2.5"

    def _payload(self, case: Case, image_field: str | None = None) -> dict[str, Any]:
        if case.mode != "image":
            raise ProviderError(
                "rodin_fal is image-to-3D — run in image mode (`--mode image`)")
        return {
            "image_urls": [self._image_ref(case)],  # array, not a string
            "geometry_file_format": self.config.get("geometry_file_format", "glb"),
            "material": self.config.get("material", "PBR"),
            "quality": self.config.get("quality", "medium"),
            **self._extra_params(),
        }

    def submit(self, case: Case) -> str:
        # documented schema — skip the image-field self-heal loop entirely
        return self._remember(self._post(self._payload(case)))
