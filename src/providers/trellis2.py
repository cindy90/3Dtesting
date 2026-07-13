"""TRELLIS 2 (Microsoft, MIT) via fal.ai hosted endpoint.

Microsoft's 4B-parameter successor to TRELLIS — the model called out explicitly
in the original comparison. Confirmed against fal.ai: endpoint `fal-ai/trellis-2`,
image input `image_url` (also supports `image_urls` array for multi-view), GLB
at `model_glb.url`. Useful config params: decimation_target (target vertex count
of the exported mesh — set ~30000 for a fair production-poly comparison),
texture_size, ss_guidance_strength.
"""

from __future__ import annotations

from .fal_base import FalProvider


class Trellis2Provider(FalProvider):
    name = "trellis2"
    model_id = "fal-ai/trellis-2"
    image_field = "image_url"
