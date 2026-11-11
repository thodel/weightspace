#!/usr/bin/env python3
"""Cosine between the displacements of two fine-tunes (stage 1): do two moves go the same way?

    python scripts/trocr_displacement.py docs/trocr/displacement.json

For a pair of moves (o1 -> m1) and (o2 -> m2) the displacements are m1 - o1 and m2 - o2, pooled over a group of tensors
(cosine of the concatenated vectors). Shared origin: do two children of one parent move alike? Sequential: does the second
step continue or undo the first? Same data from different origins: does the same training text move the weights the same way?
Also the pooler identity across all seven models (the unused ViT pooler as a lineage marker). Reads the files, writes nothing else.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safetensors import safe_open  # noqa: E402

from wsp.trocr_family import CACHE, MODELS  # noqa: E402
from wsp.trocr_subject import canonical_name, component  # noqa: E402

MOVES = {"base->kurrent": ("base", "kurrent"), "kurrent->xvi-xvii": ("kurrent", "xvi-xvii"),
         "xvi-xvii->from-kurrent-v2": ("xvi-xvii", "from-kurrent-v2"), "xvi-xvii->full-v1": ("xvi-xvii", "full-v1"),
         "base->v2": ("base", "v2"), "base->medieval": ("base", "medieval")}
PAIRS = [("base->kurrent", "base->v2"), ("base->kurrent", "base->medieval"), ("base->v2", "base->medieval"),
         ("xvi-xvii->from-kurrent-v2", "xvi-xvii->full-v1"),
         ("base->kurrent", "kurrent->xvi-xvii"), ("kurrent->xvi-xvii", "xvi-xvii->from-kurrent-v2"),
         ("kurrent->xvi-xvii", "xvi-xvii->full-v1"), ("base->kurrent", "xvi-xvii->from-kurrent-v2"),
         ("xvi-xvii->from-kurrent-v2", "base->v2")]
GROUPS = {"encoder": lambda c: c.startswith("enc."), "decoder without embedding": lambda c: c.startswith("dec.") and c != "dec.embed",
          "decoder embedding (tied)": lambda c: c == "dec.embed"}

h = {k: safe_open(str(CACHE / v / "model.safetensors"), framework="np") for k, v in MODELS.items()}
names = {k: {canonical_name(n): n for n in x.keys()} for k, x in h.items()}
pooler = {}
acc = {((p, q), g): [0.0, 0.0, 0.0] for (p, q) in PAIRS for g in GROUPS}
for cn in sorted(names["kurrent"]):
    comp = component(cn)
    if comp == "enc.pooler":
        pooler[cn] = {k: h[k].get_tensor(names[k][cn]) for k in MODELS if cn in names[k]}
        continue
    cache = {}
    def get(k, cn=cn, cache=cache):
        if k not in cache:
            cache[k] = h[k].get_tensor(names[k][cn]).astype(np.float64).ravel()
        return cache[k]
    disp = {m: get(b) - get(a) for m, (a, b) in MOVES.items()}
    for (p, q) in PAIRS:
        for g, f in GROUPS.items():
            if f(comp):
                v = acc[((p, q), g)]
                v[0] += float(disp[p] @ disp[q]); v[1] += float(disp[p] @ disp[p]); v[2] += float(disp[q] @ disp[q])
res = {"cosines": {f"{p} | {q}": {g: acc[((p, q), g)][0] / np.sqrt(acc[((p, q), g)][1] * acc[((p, q), g)][2]) for g in GROUPS}
                   for (p, q) in PAIRS},
       "norm_of_move": {f"{p}": {g: float(np.sqrt(acc[((p, q), g)][1])) for g in GROUPS} for (p, q) in PAIRS}}
pool = {}
ks = [k for k in MODELS if all(k in v for v in pooler.values())]
for a in ks:
    for b in ks:
        if a < b:
            pool[f"{a}|{b}"] = float(np.mean([np.mean(v[a] == v[b]) for v in pooler.values()]))
res["pooler_identical_share"] = pool
res["pooler_present_in"] = ks
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
for k, v in res["cosines"].items():
    print(k, {g: round(x, 3) for g, x in v.items()})
print(pool)
