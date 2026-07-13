"""InstantMesh (open-source) via fal.ai hosted endpoint.

Multi-view diffusion -> mesh reconstruction. Endpoint `fal-ai/instant-mesh`,
image input `image_url`, GLB at `model_mesh.url` (medium confidence on the exact
schema — the submit path self-heals the image field name via 422 fallback).
"""

from __future__ import annotations

from .fal_base import FalProvider


class InstantMeshProvider(FalProvider):
    name = "instantmesh"
    model_id = "fal-ai/instant-mesh"
    image_field = "image_url"
