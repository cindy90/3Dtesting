"""Provider registry. Map a config key to a Provider subclass."""

from __future__ import annotations

from typing import Any

from .base import Provider, Case, GenResult, ProviderError
from .tripo import TripoProvider
from .meshy import MeshyProvider
from .rodin import RodinProvider
from .hunyuan3d import Hunyuan3DProvider
from .hunyuan3d_v21 import Hunyuan3Dv21Provider
from .trellis import TrellisProvider
from .trellis2 import Trellis2Provider
from .triposr import TripoSRProvider
from .instantmesh import InstantMeshProvider
from .seed3d import Seed3DProvider
from .rodin_fal import RodinFalProvider

REGISTRY: dict[str, type[Provider]] = {
    # commercial APIs
    "tripo": TripoProvider,
    "meshy": MeshyProvider,
    "rodin": RodinProvider,       # 影眸/Deemos Rodin (needs RODIN_API_KEY)
    "seed3d": Seed3DProvider,     # ByteDance via Volcengine Ark (needs ARK_API_KEY)
    # hosted-on-fal commercial (uses FAL_KEY, no separate account)
    "rodin_fal": RodinFalProvider,
    # open-source cohort (image-to-3D, via fal.ai)
    "hunyuan3d": Hunyuan3DProvider,
    "hunyuan3d_v21": Hunyuan3Dv21Provider,
    "trellis": TrellisProvider,
    "trellis2": Trellis2Provider,
    "triposr": TripoSRProvider,
    "instantmesh": InstantMeshProvider,
}


def build_provider(key: str, config: dict[str, Any] | None = None) -> Provider:
    if key not in REGISTRY:
        raise KeyError(f"unknown provider '{key}'. known: {sorted(REGISTRY)}")
    return REGISTRY[key](config or {})


__all__ = [
    "Provider", "Case", "GenResult", "ProviderError",
    "REGISTRY", "build_provider",
]
