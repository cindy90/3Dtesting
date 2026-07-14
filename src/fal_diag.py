"""Zero-cost fal.ai connectivity diagnostics.

Run this ON THE RUNNER when fal submits fail mysteriously (e.g. the
self-contradictory ``405 Allow=POST`` we hit). It probes a matrix of
URL/method/client variants against the FLUX schnell endpoint using an EMPTY
JSON body — a healthy route answers ``422 validation error`` (auth + routing +
method all fine, nothing billed), so every probe is free:

  * 422        -> route works; the real payload/headers were the problem
  * 401        -> key rejected
  * 405/404    -> routing/gateway problem for THAT variant
  * 403        -> WAF/proxy block (check User-Agent / egress)

Prints status + salient response headers (Server, Allow, cf-ray, x-fal-*) and a
body snippet for each variant, plus environment facts (requests version, proxy
env vars). Never raises; purely informational.

    python -m src.fal_diag
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

MODEL = os.environ.get("FAL_DIAG_MODEL", "fal-ai/flux/schnell")
_INTERESTING = ("server", "allow", "location", "cf-ray", "content-type",
                "x-fal-request-id", "via", "x-cache")


def _show(label: str, status: int | str, headers: dict, body: str) -> None:
    hs = {k.lower(): v for k, v in headers.items()}
    picked = {k: hs[k] for k in _INTERESTING if k in hs}
    print(f"\n--- {label}")
    print(f"    status={status} headers={picked}")
    print(f"    body[:200]={body[:200]!r}")


def _probe_requests(label: str, method: str, url: str, key: str) -> None:
    try:
        import requests
        resp = requests.request(
            method, url, timeout=30, allow_redirects=False,
            headers={"Authorization": f"Key {key}",
                     "Content-Type": "application/json"},
            data=b"{}")
        _show(f"requests {label}", resp.status_code, dict(resp.headers), resp.text)
    except Exception as exc:  # noqa: BLE001
        print(f"\n--- requests {label}\n    EXC {type(exc).__name__}: {exc}")


def _probe_urllib(label: str, url: str, key: str) -> None:
    """Same POST via stdlib urllib — isolates requests-specific behaviour."""
    req = urllib.request.Request(
        url, data=b"{}", method="POST",
        headers={"Authorization": f"Key {key}",
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            _show(f"urllib {label}", resp.status, dict(resp.headers),
                  resp.read(300).decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        _show(f"urllib {label}", e.code, dict(e.headers or {}),
              (e.read(300) or b"").decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001
        print(f"\n--- urllib {label}\n    EXC {type(exc).__name__}: {exc}")


def main() -> None:
    key = os.environ.get("FAL_KEY", "").strip()
    print("==================== FAL DIAGNOSTICS ====================")
    print(f"python={sys.version.split()[0]} model={MODEL}")
    try:
        import requests
        print(f"requests={requests.__version__}")
    except Exception:
        print("requests=NOT INSTALLED")
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "NO_PROXY"):
        if os.environ.get(var):
            print(f"proxy env: {var}={os.environ[var]}")
    if not key:
        print("FAL_KEY is EMPTY — every probe below will 401; fix the secret first.")
    else:
        print(f"FAL_KEY present (len={len(key)}, has_colon={':' in key})")

    q = f"https://queue.fal.run/{MODEL}"
    s = f"https://fal.run/{MODEL}"
    # empty-body probes: a healthy route answers 422 (validation), never billed
    _probe_requests("POST queue", "POST", q, key)
    _probe_requests("POST queue trailing/", "POST", q + "/", key)
    _probe_requests("GET  queue (expect 405: proves gateway sees route)", "GET", q, key)
    _probe_requests("POST sync fal.run", "POST", s, key)
    _probe_urllib("POST queue (stdlib, curl UA)", q, key)
    print("\nInterpretation: 422=route+auth OK (payload rejected, free); "
          "401=bad key; 405/404=routing; 403=WAF/egress.")
    print("==================== END DIAGNOSTICS ====================")


if __name__ == "__main__":
    main()
