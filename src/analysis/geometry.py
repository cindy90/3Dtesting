"""Objective production-usability metrics for a single generated mesh.

This module is the technical heart of the harness. Public 3D "arenas" rank
outputs by human *preference*, which is dominated by texture / Gaussian-splat
prettiness. Paying customers (the logic behind VAST's ARR) instead need meshes
that survive a production pipeline: watertight for printing/sim, clean topology
for LODs and re-meshing, and riggable for animation.

Those three properties are *geometrically measurable* and reproducible, so we
compute them here with no human in the loop. Every metric is derived from the
mesh geometry alone via ``trimesh`` and returned as a flat, JSON-serialisable
dict. Scoring / weighting lives in ``scoring.py`` so this file stays a pure,
auditable measurement layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover - guarded for import-time clarity
    raise ImportError(
        "trimesh is required for geometry analysis. Install with "
        "`pip install -r requirements.txt`."
    ) from exc


# A "degenerate" triangle has an area below this fraction of the median face
# area; such faces are numerically meaningless and break most DCC importers.
_DEGENERATE_AREA_RATIO = 1e-4

# Symmetry probe: a character is expected to be roughly bilaterally symmetric
# about the X axis. We mirror and measure the residual as a fraction of the
# bounding-box diagonal, so the number is scale-invariant.
_SYMMETRY_SAMPLES = 4000


@dataclass
class MeshMetrics:
    """Flat container of objective per-mesh measurements."""

    # --- load / identity ---
    ok: bool = False
    error: str | None = None
    n_meshes_in_scene: int = 0

    # --- counts ---
    n_vertices: int = 0
    n_faces: int = 0

    # --- watertightness / solidity ---
    is_watertight: bool = False
    is_winding_consistent: bool = False
    n_boundary_edges: int = 0            # open edges; 0 == closed surface
    n_connected_components: int = 0      # fragmentation; 1 == single shell
    euler_number: int = 0
    genus: int | None = None             # topological handles; low is riggable
    volume: float = 0.0                  # signed; negative => inverted normals
    is_volume: bool = False              # trimesh: watertight + consistent + +vol

    # --- topology quality ---
    n_nonmanifold_edges: int = 0         # edges shared by >2 faces
    n_degenerate_faces: int = 0
    frac_degenerate_faces: float = 0.0
    n_duplicate_faces: int = 0
    sliver_frac: float = 0.0             # fraction of very thin triangles
    aspect_ratio_p95: float = 0.0        # 95th pct triangle aspect ratio
    valence_std: float = 0.0             # vertex-degree irregularity
    tri_density: float = 0.0             # faces per unit surface area (bloat)

    # --- riggability proxies ---
    symmetry_residual: float | None = None   # 0 == perfectly bilateral
    single_shell: bool = False               # one closed component
    n_internal_faces_est: int = 0            # hidden interior geometry (bad)

    # --- asset completeness (reported, lightly weighted for "production") ---
    has_uv: bool = False
    has_vertex_normals: bool = False
    has_texture_or_material: bool = False

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_scene(path: str) -> trimesh.Trimesh:
    """Load a mesh file and concatenate everything into a single Trimesh.

    Generators emit glb/obj/ply; a scene may hold several primitives. We keep a
    count of the original sub-mesh number (fragmentation signal) but analyse the
    concatenated geometry so watertightness is judged on the whole asset.
    """
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        n_sub = len(loaded.geometry)
        if n_sub == 0:
            raise ValueError("scene contains no geometry")
        mesh = trimesh.util.concatenate(
            [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    else:
        n_sub = 1
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
        raise ValueError("no triangle faces after load")
    mesh.metadata["_n_sub"] = n_sub
    return mesh


def _triangle_aspect_ratios(mesh: trimesh.Trimesh) -> np.ndarray:
    """Longest-edge / (2*inradius) per triangle. 1.0 == equilateral (best)."""
    tris = mesh.triangles
    a = np.linalg.norm(tris[:, 0] - tris[:, 1], axis=1)
    b = np.linalg.norm(tris[:, 1] - tris[:, 2], axis=1)
    c = np.linalg.norm(tris[:, 2] - tris[:, 0], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0, None))
    longest = np.maximum.reduce([a, b, c])
    with np.errstate(divide="ignore", invalid="ignore"):
        inradius = np.where(s > 0, area / s, 0.0)
        ar = np.where(inradius > 0, longest / (2.0 * inradius), np.inf)
    return ar


def _count_nonmanifold_edges(mesh: trimesh.Trimesh) -> int:
    """Edges incident to more than two faces (non-manifold)."""
    edges = mesh.edges_sorted
    # unique rows + counts
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts > 2))


def _symmetry_residual(mesh: trimesh.Trimesh) -> float | None:
    """Bilateral-symmetry residual about the X axis, scale-normalised.

    We centre the mesh, mirror a point sample across X=0, and measure the mean
    nearest-surface distance of the mirrored points back to the mesh, divided by
    the bounding-box diagonal. ~0 means clean left/right symmetry (riggable
    character); large means asymmetric or lopsided reconstruction.
    """
    try:
        diag = float(np.linalg.norm(mesh.bounding_box.extents))
        if diag <= 0:
            return None
        samples, _ = trimesh.sample.sample_surface(mesh, _SYMMETRY_SAMPLES)
        centroid = samples.mean(axis=0)
        centred = samples - centroid
        mirrored = centred.copy()
        mirrored[:, 0] *= -1.0
        # nearest distance from mirrored points to the (centred) surface sample
        # cloud via a KD-tree over the original samples.
        tree = trimesh.proximity.ProximityQuery(mesh)
        # move mirrored points back into mesh frame before querying
        dist = tree.signed_distance(mirrored + centroid)
        return float(np.mean(np.abs(dist)) / diag)
    except Exception:  # proximity can fail on pathological meshes
        return None


def _estimate_internal_faces(mesh: trimesh.Trimesh) -> int:
    """Rough count of inward-facing / occluded interior faces.

    Interior geometry (a second shell inside the body, or inverted caps) wrecks
    skinning and bloats poly count. We approximate by casting a ray from each
    face centroid along its normal: if it immediately re-enters solid geometry,
    the face is likely interior. Sampled to stay cheap on dense meshes.
    """
    try:
        n = len(mesh.faces)
        if n == 0:
            return 0
        idx = np.arange(n)
        if n > 3000:
            idx = np.random.default_rng(0).choice(n, 3000, replace=False)
        origins = mesh.triangles_center[idx] + mesh.face_normals[idx] * 1e-4
        hits = mesh.ray.intersects_any(origins, mesh.face_normals[idx])
        # a face whose *outward* normal ray immediately hits more surface is
        # facing into a cavity -> interior.
        frac = float(np.count_nonzero(hits)) / len(idx)
        return int(round(frac * n))
    except Exception:
        return 0


def analyze_mesh(path: str, *, expect_symmetry: bool = False) -> MeshMetrics:
    """Compute all objective metrics for the mesh at ``path``.

    ``expect_symmetry`` (set for character cases) enables the more expensive
    bilateral-symmetry probe; skipped otherwise since it is meaningless for,
    say, a chair or a rock.
    """
    m = MeshMetrics()
    try:
        mesh = _load_scene(path)
    except Exception as exc:
        m.ok = False
        m.error = f"{type(exc).__name__}: {exc}"
        return m

    m.ok = True
    m.n_meshes_in_scene = int(mesh.metadata.get("_n_sub", 1))
    m.n_vertices = int(len(mesh.vertices))
    m.n_faces = int(len(mesh.faces))

    # --- watertightness / solidity ---
    m.is_watertight = bool(mesh.is_watertight)
    m.is_winding_consistent = bool(mesh.is_winding_consistent)
    m.euler_number = int(mesh.euler_number)
    try:
        components = mesh.split(only_watertight=False)
        m.n_connected_components = int(len(components))
    except Exception:
        m.n_connected_components = 1
    # open (boundary) edges appear exactly once in the sorted edge list
    edges = mesh.edges_sorted
    _, counts = np.unique(edges, axis=0, return_counts=True)
    m.n_boundary_edges = int(np.count_nonzero(counts == 1))
    m.volume = float(mesh.volume) if mesh.is_watertight else 0.0
    m.is_volume = bool(mesh.is_volume)
    if m.is_watertight and m.n_connected_components > 0:
        # genus per component: g = (2C - (V - E + F)) / 2 for closed orientable
        m.genus = max(0, int((2 * m.n_connected_components - m.euler_number) // 2))

    # --- topology quality ---
    m.n_nonmanifold_edges = _count_nonmanifold_edges(mesh)
    areas = mesh.area_faces
    med_area = float(np.median(areas)) if len(areas) else 0.0
    if med_area > 0:
        degen = areas < (_DEGENERATE_AREA_RATIO * med_area)
        m.n_degenerate_faces = int(np.count_nonzero(degen))
        m.frac_degenerate_faces = float(m.n_degenerate_faces / len(areas))
    try:
        m.n_duplicate_faces = int(len(mesh.faces) - len(mesh.unique_faces))
    except Exception:
        m.n_duplicate_faces = 0
    ar = _triangle_aspect_ratios(mesh)
    finite = ar[np.isfinite(ar)]
    if len(finite):
        m.aspect_ratio_p95 = float(np.percentile(finite, 95))
        m.sliver_frac = float(np.count_nonzero(finite > 8.0) / len(finite))
    # vertex valence irregularity
    try:
        valence = np.bincount(mesh.faces.reshape(-1), minlength=len(mesh.vertices))
        m.valence_std = float(np.std(valence))
    except Exception:
        m.valence_std = 0.0
    if mesh.area > 0:
        m.tri_density = float(m.n_faces / mesh.area)

    # --- riggability ---
    m.single_shell = bool(m.is_watertight and m.n_connected_components == 1)
    if expect_symmetry:
        m.symmetry_residual = _symmetry_residual(mesh)
    m.n_internal_faces_est = _estimate_internal_faces(mesh)

    # --- asset completeness ---
    try:
        m.has_uv = bool(getattr(mesh.visual, "uv", None) is not None
                        and len(getattr(mesh.visual, "uv", []) or []) > 0)
    except Exception:
        m.has_uv = False
    m.has_vertex_normals = bool(mesh.vertex_normals is not None
                                and len(mesh.vertex_normals) == len(mesh.vertices))
    try:
        vis = mesh.visual
        m.has_texture_or_material = bool(
            getattr(vis, "material", None) is not None
            or getattr(getattr(vis, "material", None), "image", None) is not None
        )
    except Exception:
        m.has_texture_or_material = False

    return m


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys
    import json

    if len(sys.argv) < 2:
        print("usage: python -m src.analysis.geometry <mesh_path> [--symmetry]")
        raise SystemExit(2)
    sym = "--symmetry" in sys.argv
    result = analyze_mesh(sys.argv[1], expect_symmetry=sym)
    print(json.dumps(result.to_dict(), indent=2))
