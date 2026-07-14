"""Headless multi-view mesh rendering for the semantic-fidelity check.

Geometry metrics can't tell whether the mesh actually depicts what the prompt
asked for — a watertight "potato" scores like a watertight knight. This module
renders each generated mesh to a few silhouette views (matplotlib Agg backend,
CI-safe, no GPU/EGL) so a human or a VLM judge can compare them against the
shared reference image. Dense meshes are decimated to a render budget first
(fast_simplification), which changes appearance only marginally at silhouette
scale and keeps the whole sweep to seconds per mesh.
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import trimesh  # noqa: E402

_RENDER_BUDGET_FACES = 12_000
_VIEWS = [(15, 0), (15, 90), (15, 200), (70, 30)]  # (elev, azim): front/side/back-ish/top


def _prepare(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Decimate to the render budget and centre/normalise for framing."""
    v, f = mesh.vertices, mesh.faces
    if len(f) > _RENDER_BUDGET_FACES:
        try:
            import fast_simplification
            v, f = fast_simplification.simplify(
                np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int64),
                target_reduction=1.0 - _RENDER_BUDGET_FACES / len(f))
        except Exception:
            pass  # render dense; slower but correct
    v = np.asarray(v, dtype=np.float64)
    centre = (v.max(axis=0) + v.min(axis=0)) / 2.0
    scale = float(np.max(v.max(axis=0) - v.min(axis=0))) or 1.0
    return (v - centre) / scale, np.asarray(f)


def render_views(mesh_path: str, out_dir: str, stem: str,
                 views: list[tuple[int, int]] | None = None) -> list[str]:
    """Render ``mesh_path`` to one PNG per view. Returns the file paths."""
    os.makedirs(out_dir, exist_ok=True)
    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("no geometry to render")
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = loaded
    v, f = _prepare(mesh)

    out_paths = []
    for i, (elev, azim) in enumerate(views or _VIEWS):
        fig = plt.figure(figsize=(3.2, 3.2), dpi=90)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2],
                        color="#b9c0c8", edgecolor="none", shade=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        path = os.path.join(out_dir, f"{stem}_v{i}.png")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        out_paths.append(path)
    return out_paths
