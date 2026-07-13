"""TripoSR (Stability AI + Tripo AI, MIT) via fal.ai hosted endpoint.

Open-source Large Reconstruction Model: single image -> mesh in <1s. A useful
low-end anchor in the cohort — fast and free-weights, but typically lower
production quality, which the objective metrics should surface.

Confirmed against fal.ai: endpoint `fal-ai/triposr`, image input `image_url`,
GLB at `model_mesh.url`. Useful config params: mc_resolution (marching-cubes
resolution, default 256), foreground_ratio, do_remove_background.
"""

from __future__ import annotations

from .fal_base import FalProvider


class TripoSRProvider(FalProvider):
    name = "triposr"
    model_id = "fal-ai/triposr"
    image_field = "image_url"
