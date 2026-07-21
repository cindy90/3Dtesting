"""Khronos glTF-Validator gate — the objective floor for "can engines import it".

Runs the official validator (npm package ``gltf-validator``, a JS library)
via a tiny node wrapper. numErrors == 0 is the ENGINE GATE: a glb that
fails spec validation may still open in forgiving viewers, but it is not a
defensible "engine-ready" claim. Warnings/infos are reported for context.

Requires node + `npm install -g gltf-validator` (the workflow installs it
non-fatally; probe() degrades to a skip when missing).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

_JS = """
const v = require('gltf-validator');
const fs = require('fs');
v.validateBytes(new Uint8Array(fs.readFileSync(process.argv[2])))
 .then(r => {
   const i = r.issues || {};
   console.log(JSON.stringify({numErrors: i.numErrors, numWarnings: i.numWarnings,
     numInfos: i.numInfos,
     codes: (i.messages || []).slice(0, 8).map(m => m.code)}));
 })
 .catch(e => { console.log(JSON.stringify({error: String(e)})); process.exit(1); });
"""


def _node_path() -> str | None:
    env = os.environ.get("GLTF_VALIDATOR_NODE_PATH")
    if env:
        return env
    try:
        proc = subprocess.run(["npm", "root", "-g"], capture_output=True,
                              text=True, timeout=30)
        root = (proc.stdout or "").strip()
        if root and os.path.isdir(os.path.join(root, "gltf-validator")):
            return root
    except Exception:  # noqa: BLE001
        pass
    return None


def available() -> bool:
    return shutil.which("node") is not None and _node_path() is not None


def probe(mesh_path: str, workdir: str) -> dict:
    """Validate one glb; never raises — errors land in the dict."""
    if not mesh_path.lower().endswith((".glb", ".gltf")):
        return {"skipped": f"not glTF: {os.path.basename(mesh_path)}"}
    node_path = _node_path()
    if not node_path or not shutil.which("node"):
        return {"skipped": "gltf-validator not installed"}
    os.makedirs(workdir, exist_ok=True)
    js = os.path.join(workdir, "_gltf_check.js")
    if not os.path.exists(js):
        with open(js, "w") as fh:
            fh.write(_JS)
    try:
        proc = subprocess.run(
            ["node", js, mesh_path],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "NODE_PATH": node_path})
        line = (proc.stdout or "").strip().splitlines()
        return json.loads(line[-1]) if line else {"error": "no output"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
