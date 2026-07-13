"""Hunyuan3D v2.1 (Tencent) via fal.ai hosted endpoint.

Newer Hunyuan3D generation. Confirmed against fal.ai: endpoint
`fal-ai/hunyuan3d-v21`, image input field `input_image_url`, GLB output at
`model_glb.url` (the base FalProvider.asset_url already handles model_glb).
Set config `params: {textured_mesh: true}` to get PBR texture (≈3x price).
"""

from __future__ import annotations

from .fal_base import FalProvider


class Hunyuan3Dv21Provider(FalProvider):
    name = "hunyuan3d_v21"
    model_id = "fal-ai/hunyuan3d-v21"
    image_field = "input_image_url"
