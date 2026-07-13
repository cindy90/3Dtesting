"""TRELLIS (Microsoft) via fal.ai hosted endpoint.

TRELLIS is image-first and outputs both Gaussians and an extracted mesh. We
score the *mesh* only — the Gaussian/splat output is exactly the "pretty but
not production" artefact that public arenas over-reward.

Endpoint id and image field confirmed against fal.ai: `fal-ai/trellis`,
image input `image_url`, GLB at `model_mesh.url`. Mesh/format knobs go through
config `params` (e.g. mesh_simplify), so no per-model payload override needed.
"""

from __future__ import annotations

from .fal_base import FalProvider


class TrellisProvider(FalProvider):
    name = "trellis"
    model_id = "fal-ai/trellis"
    image_field = "image_url"
