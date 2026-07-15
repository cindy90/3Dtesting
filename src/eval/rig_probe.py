"""Blender headless rig smoke test — the game-scenario ground truth.

Our riggability sub-score is built from proxies (closure, fragmentation,
symmetry). This probe measures the thing itself: import the mesh into Blender,
bind it to a minimal two-bone armature with AUTOMATIC WEIGHTS (bone heat — the
exact step that fails on junk topology), bend the upper bone 45°, and measure
whether the mesh deforms sanely or collapses.

The mesh gets a rigger's standard cleanup first (position weld — UV-seam
duplicate vertices reliably kill the bone-heat solve on every vendor's real
GLB — plus degenerate-face dissolve and loose-debris removal). Binding is
then tiered: "welded" (binds after cleanup alone), "remeshed" (needs a voxel
remesh, i.e. throws away UVs/topology — score capped at 65), or "failed".

Reported per mesh (JSON on the last stdout line):
  bind_ok            bone-heat produced usable weights (at any tier)
  bind_tier          welded | remeshed | failed
  unweighted_frac    vertices no bone influences (won't follow the rig)
  volume_ratio       |deformed| / |rest| volume (candy-wrapper collapse -> <<1)
  edge_stretch_p99   99th pct deformed/rest edge length (explosion -> >>1)
  rig_smoke_score    0-100 composite
  decimated          probe ran on a decimated copy (very dense input)
  verts_welded / debris_faces_removed   cleanup magnitude

Run one mesh per process (bpy state is global; a fresh interpreter per mesh
gives clean state, hard timeouts, and crash isolation):

    python -m src.eval.rig_probe <mesh_path>
"""

from __future__ import annotations

import json
import math
import sys

# meshes above this vertex count are decimated before the probe: bone heat is a
# global solve and multi-million-vertex sculpts would blow the time budget.
_DECIMATE_ABOVE_VERTS = 400_000
_BEND_DEG = 45.0


def _tri_volume(verts, tris) -> float:
    """|signed volume| via divergence theorem; robust to open meshes."""
    total = 0.0
    for a, b, c in tris:
        va, vb, vc = verts[a], verts[b], verts[c]
        total += (va[0] * (vb[1] * vc[2] - vb[2] * vc[1])
                  - va[1] * (vb[0] * vc[2] - vb[2] * vc[0])
                  + va[2] * (vb[0] * vc[1] - vb[1] * vc[0]))
    return abs(total / 6.0)


def probe(mesh_path: str) -> dict:
    import bpy
    from mathutils import Vector

    out: dict = {"bind_ok": False, "bind_tier": "failed", "unweighted_frac": 1.0,
                 "volume_ratio": 0.0, "edge_stretch_p99": 0.0,
                 "rig_smoke_score": 0.0, "decimated": False, "error": None}

    bpy.ops.wm.read_factory_settings(use_empty=True)

    low = mesh_path.lower()
    if low.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif low.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=mesh_path)
    else:
        raise ValueError(f"unsupported mesh format: {mesh_path}")

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise ValueError("no mesh objects after import")
    for o in bpy.context.scene.objects:
        o.select_set(o.type == "MESH")
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # standard rigger cleanup before binding, mirroring the benchmark's
    # "welded tier": (1) position weld — real GLBs carry UV-seam duplicate
    # vertices and Blender's bone-heat solver reliably fails on them
    # ("failed to find solution for one or more bones", every vendor mesh);
    # (2) dissolve degenerate zero-area faces; (3) drop tiny loose debris
    # (a 2-face sliver island kills the solve as surely as a real defect,
    # and zeroing the score for it would be the binary cliff all over
    # again). Bone-heat failure AFTER this cleanup is a genuine signal.
    import bmesh
    diag = math.dist(ob.bound_box[0], ob.bound_box[6]) or 1.0
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    n_before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6 * diag)
    bmesh.ops.dissolve_degenerate(bm, dist=2e-6 * diag, edges=bm.edges)

    # connected components over the welded mesh (union-find on edges)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    parent = list(range(len(bm.verts)))
    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for e in bm.edges:
        ra, rb = _find(e.verts[0].index), _find(e.verts[1].index)
        if ra != rb:
            parent[ra] = rb
    comp_faces: dict[int, int] = {}
    for f in bm.faces:
        comp_faces[_find(f.verts[0].index)] = \
            comp_faces.get(_find(f.verts[0].index), 0) + 1
    n_faces = len(bm.faces)
    debris_cut = max(8, int(0.0005 * n_faces))
    debris_roots = {r for r, n in comp_faces.items() if n < debris_cut}
    out["debris_faces_removed"] = sum(comp_faces[r] for r in debris_roots)
    if debris_roots:
        kill = [v for v in bm.verts if _find(v.index) in debris_roots]
        bmesh.ops.delete(bm, geom=kill, context="VERTS")

    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    out["verts_welded"] = n_before - len(ob.data.vertices)

    if len(ob.data.vertices) > _DECIMATE_ABOVE_VERTS:
        out["decimated"] = True
        mod = ob.modifiers.new("dec", "DECIMATE")
        mod.ratio = _DECIMATE_ABOVE_VERTS / len(ob.data.vertices)
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # --- two-bone armature through the vertical extent of the mesh ---
    def _extent():
        vs = [v.co for v in ob.data.vertices]
        return (sum(v.x for v in vs) / len(vs), sum(v.y for v in vs) / len(vs),
                min(v.z for v in vs), max(v.z for v in vs))
    cx, cy, z0, z1 = _extent()
    zm = (z0 + z1) / 2.0

    arm_data = bpy.data.armatures.new("rig")
    arm = bpy.data.objects.new("rig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    b1 = arm_data.edit_bones.new("lower")
    b1.head, b1.tail = (cx, cy, z0), (cx, cy, zm)
    b2 = arm_data.edit_bones.new("upper")
    b2.head, b2.tail = (cx, cy, zm), (cx, cy, z1)
    b2.parent, b2.use_connect = b1, True
    bpy.ops.object.mode_set(mode="OBJECT")

    def _bind() -> float:
        """ARMATURE_AUTO bind; returns unweighted fraction (1.0 = no weights)."""
        for o in bpy.context.scene.objects:
            o.select_set(False)
        ob.select_set(True)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        try:
            bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        except RuntimeError:
            return 1.0
        groups = {g.index for g in ob.vertex_groups}
        bad = sum(1 for v in ob.data.vertices
                  if sum(g.weight for g in v.groups if g.group in groups) < 1e-6)
        return bad / max(1, len(ob.data.vertices))

    def _unbind() -> None:
        ob.parent = None
        for m in list(ob.modifiers):
            if m.type == "ARMATURE":
                ob.modifiers.remove(m)
        ob.vertex_groups.clear()

    # --- tiered bind: welded -> voxel-remeshed -> failed ---
    # Bone heat failing on the cleaned mesh is a real production signal
    # (open SparseFlex surfaces do this; watertight meshes don't), but a
    # rigger's next move is a voxel remesh — which rescues the bind at the
    # cost of throwing away UVs and topology. Measure both tiers.
    frac = _bind()
    tier = "welded"
    if frac >= 0.999:
        _unbind()
        try:
            bpy.context.view_layer.objects.active = ob
            ob.data.remesh_voxel_size = diag / 150.0
            bpy.ops.object.voxel_remesh()
            if len(ob.data.vertices) > _DECIMATE_ABOVE_VERTS:
                mod = ob.modifiers.new("dec", "DECIMATE")
                mod.ratio = _DECIMATE_ABOVE_VERTS / len(ob.data.vertices)
                bpy.ops.object.modifier_apply(modifier=mod.name)
            frac = _bind()
            tier = "remeshed"
        except Exception as exc:  # noqa: BLE001 - remesh itself can fail
            out["error"] = f"voxel-remesh: {exc}"
            frac, tier = 1.0, "failed"
    if frac >= 0.999:
        tier = "failed"
    out["bind_tier"] = tier
    out["unweighted_frac"] = round(frac, 4)
    out["bind_ok"] = frac < 0.5  # majority of mesh follows rig
    if not out["bind_ok"]:
        out["rig_smoke_score"] = 0.0
        return out

    # --- rest-state geometry (after whatever tier bound) ---
    me = ob.data
    me.calc_loop_triangles()
    rest_verts = [tuple(v.co) for v in me.vertices]
    tris = [tuple(t.vertices) for t in me.loop_triangles]
    rest_vol = _tri_volume(rest_verts, tris)

    # sample edges for the stretch metric
    edges = [(e.vertices[0], e.vertices[1]) for e in me.edges]
    if len(edges) > 20000:
        step = len(edges) // 20000
        edges = edges[::step]
    rest_len = [math.dist(rest_verts[a], rest_verts[b]) for a, b in edges]

    # --- bend the upper bone and measure the deformed mesh ---
    pb = arm.pose.bones["upper"]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (math.radians(_BEND_DEG), 0.0, 0.0)
    deps = bpy.context.evaluated_depsgraph_get()
    ob_eval = ob.evaluated_get(deps)
    me_eval = ob_eval.to_mesh()
    def_verts = [tuple(v.co) for v in me_eval.vertices]
    if len(def_verts) == len(rest_verts):
        def_vol = _tri_volume(def_verts, tris)
        out["volume_ratio"] = round(def_vol / rest_vol, 4) if rest_vol > 0 else 0.0
        stretch = sorted(
            (math.dist(def_verts[a], def_verts[b]) / rl) if rl > 1e-12 else 1.0
            for (a, b), rl in zip(edges, rest_len))
        out["edge_stretch_p99"] = round(stretch[int(0.99 * (len(stretch) - 1))], 3)
    ob_eval.to_mesh_clear()

    # --- composite score ---
    score = 100.0
    score -= 200.0 * out["unweighted_frac"]                      # dead vertices
    vr = out["volume_ratio"]
    if vr > 0:
        score -= max(0.0, abs(1.0 - vr) - 0.15) * 200.0          # collapse/inflate
    else:
        score -= 40.0
    s99 = out["edge_stretch_p99"]
    if s99 > 2.0:
        score -= (s99 - 2.0) * 30.0                              # explosion
    if out["bind_tier"] == "remeshed":
        # bound, but only after destroying UVs/topology — a real extra DCC
        # step; cap below any weld-tier bind (mirrors the repair-tier floor
        # in the watertight score)
        score = min(score, 65.0)
    out["rig_smoke_score"] = round(max(0.0, min(100.0, score)), 2)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: rig_probe <mesh>"}))
        sys.exit(2)
    try:
        result = probe(sys.argv[1])
    except Exception as exc:  # noqa: BLE001 - report, don't crash the driver
        result = {"bind_ok": False, "rig_smoke_score": 0.0,
                  "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
