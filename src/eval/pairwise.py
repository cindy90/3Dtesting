"""Dimension-scoped pairwise blind voting + ELO — the human eval harness.

Public arenas vote "which looks better" on textured renders and drift toward
texture/splat preference (we quantified +144 ELO for textured). This harness
keeps the pairwise-ELO *form* but fixes the two bias vectors: votes happen on
UNTEXTURED geometry renders, and each vote is scoped to ONE dimension
(geometry quality / shape fidelity to the reference) instead of a global
"better".

  build: pair up same-case meshes from the semantic manifest, emit a single
         self-contained HTML voting page (anonymous left/right, keyboard
         voting, CSV export) + a pairs manifest for de-anonymisation.
  elo:   aggregate one or more vote CSVs into per-dimension ELO ratings.

    python -m src.eval.pairwise build results/
    python -m src.eval.pairwise elo results/pairwise_pairs.json votes.csv
"""

from __future__ import annotations

import base64
import csv
import json
import os
import random
import sys
from collections import defaultdict
from itertools import combinations

DIMENSIONS = ["geometry", "fidelity"]
_K = 32.0


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def build(out_dir: str, seed: int = 0) -> None:
    with open(os.path.join(out_dir, "semantic_manifest.json")) as fh:
        manifest = json.load(fh)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for it in manifest["items"]:
        if it.get("views"):
            by_case[it["case_id"]].append(it)
    rng = random.Random(seed)
    pairs, blocks = [], []
    for case_id in sorted(by_case):
        for a, b in combinations(by_case[case_id], 2):
            if rng.random() < 0.5:
                a, b = b, a  # randomize left/right
            pid = f"pw_{len(pairs):04d}"
            pairs.append({"pair_id": pid, "case_id": case_id,
                          "left": a["provider"], "right": b["provider"]})
            la, lb = _b64(a["views"][0]), _b64(b["views"][0])
            ref = os.path.join(os.path.dirname(out_dir) or ".", "cases", "refs",
                               f"{case_id}.png")
            ref_html = (f'<img class="ref" src="data:image/png;base64,{_b64(ref)}">'
                        if os.path.exists(ref) else "")
            blocks.append(f"""
<div class="pair" id="{pid}">
  <h3>{pid} <span class="dim"></span></h3>{ref_html}
  <div class="row"><img src="data:image/png;base64,{la}">
  <img src="data:image/png;base64,{lb}"></div>
  <div class="btns">
    <button onclick="vote('{pid}','L')">← Left (A)</button>
    <button onclick="vote('{pid}','T')">Tie (S)</button>
    <button onclick="vote('{pid}','R')">Right (D)</button>
  </div>
</div>""")
    html = f"""<meta charset="utf-8"><title>Pairwise blind vote</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:auto}}
.pair{{display:none;border:1px solid #ccc;padding:1em;margin:1em 0}}
.pair.active{{display:block}} .row img{{width:46%;margin:1%}}
.ref{{width:20%;display:block;margin:0 auto}} button{{font-size:1.2em;margin:0 1em;padding:.4em 1.4em}}</style>
<h1>Pairwise blind vote — untextured geometry renders</h1>
<p>Two dimensions per pair, asked one after the other:
<b>geometry</b> = which mesh is cleaner/more production-ready, ignoring shape;
<b>fidelity</b> = which matches the reference image's shape better.
Keys: A = left, S = tie, D = right. Votes download as CSV at the end.</p>
<div id="pairs">{''.join(blocks)}</div>
<script>
const DIMS={json.dumps(DIMENSIONS)};
const pairs=[...document.querySelectorAll('.pair')];
let pi=0, di=0, votes=[];
function show(){{
  pairs.forEach(p=>p.classList.remove('active'));
  if(pi>=pairs.length){{return done();}}
  pairs[pi].classList.add('active');
  pairs[pi].querySelector('.dim').textContent='— vote: '+DIMS[di];
}}
function vote(pid,w){{
  votes.push([pid,DIMS[di],w]);
  if(++di>=DIMS.length){{di=0;pi++;}}
  show();
}}
document.addEventListener('keydown',e=>{{
  const k={{a:'L',s:'T',d:'R'}}[e.key];
  if(k&&pi<pairs.length) vote(pairs[pi].id,k);
}});
function done(){{
  const csv='pair_id,dimension,winner\\n'+votes.map(v=>v.join(',')).join('\\n');
  const a=document.createElement('a');
  a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
  a.download='pairwise_votes.csv';
  document.body.innerHTML='<h1>Done — CSV downloading.</h1>';a.click();
}}
show();
</script>"""
    with open(os.path.join(out_dir, "pairwise_vote.html"), "w") as fh:
        fh.write(html)
    with open(os.path.join(out_dir, "pairwise_pairs.json"), "w") as fh:
        json.dump(pairs, fh, indent=2)
    print(f"pairwise: {len(pairs)} pairs x {len(DIMENSIONS)} dims -> "
          f"{out_dir}/pairwise_vote.html (+ pairs manifest)")


def elo(pairs_path: str, vote_csvs: list[str]) -> dict:
    with open(pairs_path) as fh:
        pairs = {p["pair_id"]: p for p in json.load(fh)}
    rating: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 1000.0))
    n = 0
    for path in vote_csvs:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                p = pairs.get(row["pair_id"])
                if not p:
                    continue
                dim, w = row["dimension"], row["winner"]
                a, b = p["left"], p["right"]
                ra, rb = rating[dim][a], rating[dim][b]
                ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
                sa = {"L": 1.0, "R": 0.0, "T": 0.5}.get(w, 0.5)
                rating[dim][a] = ra + _K * (sa - ea)
                rating[dim][b] = rb + _K * ((1.0 - sa) - (1.0 - ea))
                n += 1
    out = {dim: dict(sorted(((k, round(v, 1)) for k, v in d.items()),
                            key=lambda kv: -kv[1]))
           for dim, d in rating.items()}
    out["_n_votes"] = n
    return out


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: pairwise build <out_dir> | "
                         "pairwise elo <pairs.json> <votes.csv> [...]")
    if sys.argv[1] == "build":
        build(sys.argv[2])
    elif sys.argv[1] == "elo":
        print(json.dumps(elo(sys.argv[2], sys.argv[3:]), indent=2))
    else:
        raise SystemExit(f"unknown subcommand {sys.argv[1]}")


if __name__ == "__main__":
    main()
