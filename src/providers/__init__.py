"""Provider registry. Map a config key to a Provider subclass."""

from __future__ import annotations

from typing import Any

from .base import Provider, Case, GenResult, ProviderError
from .tripo import TripoProvider
from .meshy import MeshyProvider
from .rodin import RodinProvider
from .hunyuan3d import Hunyuan3DProvider
from .trellis import TrellisProvider

REGISTRY: dict[str, type[Provider]] = {
    "tripo": TripoProvider,
    "meshy": MeshyProvider,
    "rodin": RodinProvider,
    "hunyuan3d": Hunyuan3DProvider,
    "trellis": TrellisProvider,
}


def build_provider(key: str, config: dict[str, Any] | None = None) -> Provider:
    if key not in REGISTRY:
        raise KeyError(f"unknown provider '{key}'. known: {sorted(REGISTRY)}")
    return REGISTRY[key](config or {})


__all__ = [
    "Provider", "Case", "GenResult", "ProviderError",
    "REGISTRY", "build_provider",
]
