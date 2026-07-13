"""Hunyuan3D v2 (Tencent) via fal.ai hosted endpoint.

Confirmed against fal.ai: endpoint `fal-ai/hunyuan3d/v2`, image input field
`input_image_url` (NOT image_url), GLB output at `model_mesh.url`. Extra params
(num_inference_steps, guidance_scale, octree_resolution, textured_mesh) go
through config `params`.

If you have Tencent Cloud AI3D access instead (ai3d.tencentcloudapi.com,
TC3-HMAC-SHA256 signing), write a sibling provider — the analysis/scoring
layers are provider-agnostic.
"""

from __future__ import annotations

from .fal_base import FalProvider


class Hunyuan3DProvider(FalProvider):
    name = "hunyuan3d"
    model_id = "fal-ai/hunyuan3d/v2"
    image_field = "input_image_url"
