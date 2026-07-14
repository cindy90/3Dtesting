"""Tripo (VAST) hosted on fal.ai — v2.5 and the new P-series.

fal hosts Tripo under the partner namespace ``tripo3d/``:

  * ``tripo3d/tripo/v2.5/image-to-3d`` — Tripo v2.5 ($0.2 untextured, $0.3
    standard texture, $0.4 HD; +$0.05 each for style/quad options)
  * ``tripo3d/p1/image-to-3d``        — Tripo P1.0 "Smart Mesh" (GDC 2026),
    Tripo's native 3D diffusion model marketed explicitly as production-grade
    (clean topology, engine-ready) — the exact claim this benchmark measures.

Both use the standard fal queue API with ``image_url`` input; poll URLs come
from the submit response (subpath-safe). Using these instead of the direct
Tripo API lets the whole cohort ride one FAL_KEY.
"""

from __future__ import annotations

from .fal_base import FalProvider


class TripoFalProvider(FalProvider):
    """Tripo v2.5 via fal (image-to-3D)."""

    name = "tripo_fal"
    model_id = "tripo3d/tripo/v2.5/image-to-3d"
    image_field = "image_url"


class TripoP1Provider(FalProvider):
    """Tripo P1.0 (Smart Mesh, production-grade claim) via fal."""

    name = "tripo_p1"
    model_id = "tripo3d/p1/image-to-3d"
    image_field = "image_url"
