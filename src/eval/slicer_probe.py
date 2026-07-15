"""PrusaSlicer ground truth — the 3D-print-scenario referee.

Vendors quote "slicer pass rates" (Meshy self-reports 97%) without stating
the tier. This probe produces both tiers with the same referee everyone uses:

  slicer_manifold   PrusaSlicer --info verdict on the exported STL (ADMesh
                    stats: manifold flag, open edges, degenerate facets,
                    facets auto-removed, shell count). STL is a positional
                    triangle soup, so this is the "welded" tier by
                    construction — directly comparable to vendor claims.
  gcode_ok          --export-gcode succeeded (draft profile). LENIENT:
                    PrusaSlicer silently auto-repairs while slicing, so a
                    broken mesh can still pass. Report both, never just one.

The mesh is exported exactly as generated (no weld, no repair on our side),
scaled to 50 mm longest side and dropped onto the plate — the same "download
and slice" workflow a real customer runs.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time

_TARGET_MM = 50.0          # longest side after scaling; a typical desk print
_INFO_TIMEOUT_S = 120
_GCODE_TIMEOUT_S = 300
# draft settings keep dense sculpts affordable: thick layers, sparse infill
_GCODE_ARGS = ["--layer-height", "0.3", "--fill-density", "5%"]

_INT_FIELDS = {
    "open_edges": "open_edges",
    "degenerate_facets": "degenerate_facets",
    "facets_removed": "facets_removed",
    "number_of_parts": "n_parts",
    "number_of_facets": "n_facets",
}


def find_slicer() -> str | None:
    """Locate the PrusaSlicer binary (env override, then PATH)."""
    env = os.environ.get("PRUSA_SLICER")
    if env and os.path.exists(env):
        return env
    return shutil.which("prusa-slicer") or shutil.which("prusaslicer")


def export_stl(mesh_path: str, stl_path: str) -> None:
    """GLB/OBJ -> print-ready STL: as-generated geometry, 50 mm, on-plate."""
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    ext = max(mesh.extents) if mesh.extents is not None else 0
    if ext <= 0:
        raise ValueError("degenerate bounds")
    mesh.apply_scale(_TARGET_MM / ext)
    mesh.apply_translation(-mesh.bounds[0])
    mesh.export(stl_path)


def slicer_info(slicer: str, stl_path: str) -> dict:
    """Parse `prusa-slicer --info` (ADMesh stats) into a flat dict."""
    proc = subprocess.run([slicer, "--info", stl_path],
                          capture_output=True, text=True,
                          timeout=_INFO_TIMEOUT_S)
    out: dict = {"slicer_manifold": None}
    for line in (proc.stdout or "").splitlines():
        m = re.match(r"\s*(\w+)\s*=\s*(\S+)", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if key == "manifold":
            out["slicer_manifold"] = (val == "yes")
        elif key in _INT_FIELDS:
            try:
                out[_INT_FIELDS[key]] = int(val)
            except ValueError:
                pass
        elif key == "volume":
            try:
                out["volume_mm3"] = round(float(val), 1)
            except ValueError:
                pass
    if out["slicer_manifold"] is None:
        raise RuntimeError(f"--info gave no manifold verdict: "
                           f"{(proc.stderr or proc.stdout or '')[:200]}")
    # a manifold verdict with zero open edges reported means clean pass;
    # ADMesh omits the error lines entirely when there is nothing to fix
    out.setdefault("open_edges", 0)
    out.setdefault("degenerate_facets", 0)
    out.setdefault("facets_removed", 0)
    return out


def slice_gcode(slicer: str, stl_path: str, gcode_path: str) -> dict:
    """Draft-profile slice; pass = exit 0 and a non-trivial gcode file."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [slicer, "--export-gcode", stl_path, "--output", gcode_path,
             *_GCODE_ARGS],
            capture_output=True, text=True, timeout=_GCODE_TIMEOUT_S)
        ok = (proc.returncode == 0 and os.path.exists(gcode_path)
              and os.path.getsize(gcode_path) > 1000)
        err = None if ok else (proc.stderr or proc.stdout or "")[-200:].strip()
    except subprocess.TimeoutExpired:
        ok, err = False, f"timeout {_GCODE_TIMEOUT_S}s"
    return {"gcode_ok": ok, "slice_time_s": round(time.time() - t0, 1),
            "gcode_error": err}


def probe(mesh_path: str, workdir: str) -> dict:
    """Full probe for one mesh. Never raises; errors land in the dict."""
    slicer = find_slicer()
    if not slicer:
        return {"error": "prusa-slicer not found"}
    base = os.path.splitext(os.path.basename(mesh_path))[0]
    stl = os.path.join(workdir, base + ".stl")
    gcode = os.path.join(workdir, base + ".gcode")
    out: dict = {"error": None}
    try:
        export_stl(mesh_path, stl)
        out.update(slicer_info(slicer, stl))
        out.update(slice_gcode(slicer, stl, gcode))
    except Exception as exc:  # noqa: BLE001 - one bad mesh must not kill the run
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for p in (stl, gcode):
            if os.path.exists(p):
                os.remove(p)
    return out
