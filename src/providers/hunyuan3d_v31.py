"""Hunyuan3D v3.1 Pro (Tencent) via fal.ai — Tencent's current flagship.

Confirmed on fal: endpoint ``fal-ai/hunyuan-3d/v3.1/pro/image-to-3d``
($0.375/gen; the /rapid sibling is $0.225). Input field ``input_image_url``
(JPG/PNG/WEBP, <=8MB). Options include enable_pbr and a geometry-only mode;
Pro supports custom polygon counts (40K-1.5M faces).

This is the head-to-head "latest" entry for Tencent; v2 / v2.1 stay in the
cohort as generation references.
"""

from __future__ import annotations

from .fal_base import FalProvider


class Hunyuan3Dv31Provider(FalProvider):
    name = "hunyuan3d_v31"
    model_id = "fal-ai/hunyuan-3d/v3.1/pro/image-to-3d"
    image_field = "input_image_url"
