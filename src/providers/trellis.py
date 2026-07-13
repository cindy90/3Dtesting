"""TRELLIS (Microsoft) via fal.ai hosted endpoint.

TRELLIS is image-first and outputs both Gaussians and an extracted mesh. We
score the *mesh* only — the Gaussian/splat output is exactly the "pretty but
not production" artefact that public arenas over-reward. For text cases the fal
endpoint runs an internal text->image step; verify the field names for the
deployment you target.

Override the model id in config.yaml, e.g.:
    trellis: {model_id: "fal-ai/trellis"}
"""

from __future__ import annotations

from typing import Any

from .fal_base import FalProvider
from .base import Case


class TrellisProvider(FalProvider):
    name = "trellis"
    model_id = "fal-ai/trellis"

    def _payload(self, case: Case) -> dict[str, Any]:
        payload = super()._payload(case)
        # ask TRELLIS to export a mesh (glb), not only the Gaussian splat
        payload.setdefault("output_format", "glb")
        payload.setdefault("mesh_simplify", self.config.get("mesh_simplify", 0.95))
        return payload
