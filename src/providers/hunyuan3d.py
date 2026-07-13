"""Hunyuan3D (Tencent) via fal.ai hosted endpoint.

Default target is the fal deployment of Hunyuan3D. If you have Tencent Cloud
AI3D access instead (ai3d.tencentcloudapi.com, TC3-HMAC-SHA256 signing), write
a sibling provider — the analysis/scoring layers are provider-agnostic.

Override the model id in config.yaml, e.g.:
    hunyuan3d: {model_id: "fal-ai/hunyuan3d/v2"}
"""

from __future__ import annotations

from .fal_base import FalProvider


class Hunyuan3DProvider(FalProvider):
    name = "hunyuan3d"
    model_id = "fal-ai/hunyuan3d/v2"
