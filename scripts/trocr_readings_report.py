#!/usr/bin/env python3
"""Tables of docs/TROCR_READINGS.md from docs/trocr/readings.json (every number in the tables is computed here).

    python scripts/trocr_readings_report.py docs/trocr/readings.json > docs/trocr/readings_tables.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp.trocr_family import dist_matrix, mst, pooled  # noqa: E402

res = json.load(open(sys.argv[1]))
M = list(res["models"])
PAR = res["parent"]
meta = res["tensors"]
rows = res["pairs"]
out = []
P = out.append


def sel_forward(r, m):          # everything the forward uses; weights, biases, norms and embeddings pooled
    return not m["unused"]


def sel_weights(r, m):
    return not m["unused"] and m["kind"] == "weight"


D_all = dist_matrix(res, sel_forward)
D_w = dist_matrix(res, sel_weights)

P("### T1. Distance between every pair, all tensors the forward uses (pooled ‖a−b‖ / mean(‖a‖, ‖b‖))\n")
P("| | " + " | ".join(f"`{m}`" for m in M) + " |")
P("|---|" + "---:|" * len(M))
for a in M:
    P(f"| `{a}` | " + " | ".join("—" if a == b else f"{D_all[(a, b)]:.3f}" for b in M) + " |")

P("\n### T2. The same, weight matrices only (biases, norms, embeddings and positions left out)\n")
P("| | " + " | ".join(f"`{m}`" for m in M) + " |")
P("|---|" + "---:|" * len(M))
for a in M:
    P(f"| `{a}` | " + " | ".join("—" if a == b else f"{D_w[(a, b)]:.3f}" for b in M) + " |")

P("\n### T3. Lineage from the weights alone: minimum spanning tree of T1 against the documented tree\n")
tree = mst(M, D_all)
doc = {frozenset((c, p)) for c, p in PAR.items()}
P("| MST edge | distance | documented parent–child edge |")
P("|---|---:|---|")
hit = 0
for a, b, d in tree:
    ok = frozenset((a, b)) in doc
    hit += ok
    P(f"| `{a}` – `{b}` | {d:.3f} | {'yes' if ok else 'no'} |")
P(f"\n{hit} of {len(tree)} MST edges are documented parent–child edges; documented edges: {len(doc)}.\n")
P("Nearest neighbour of each model (all tensors; weights only):\n")
P("| model | nearest (T1) | d | nearest (T2) | d | documented parent |")
P("|---|---|---:|---|---:|---|")
for a in M:
    n1 = min((D_all[(a, b)], b) for b in M if b != a)
    n2 = min((D_w[(a, b)], b) for b in M if b != a)
    P(f"| `{a}` | `{n1[1]}` | {n1[0]:.3f} | `{n2[1]}` | {n2[0]:.3f} | `{PAR.get(a, '—')}` |")

# ---------------------------------------------------------------------------------------------- parts
PARTS = [("enc.embed", "encoder embeddings (patch, cls, positions)"), ("enc.attn", "encoder attention"), ("enc.ffn", "encoder FFN"),
         ("enc.norm", "encoder norms"), ("dec.embed", "decoder embeddings (tokens, positions, norm; tied output)"),
         ("dec.self", "decoder self-attention"), ("dec.cross", "decoder cross-attention"), ("dec.ffn", "decoder FFN"),
         ("dec.norm", "decoder norms")]
CH = [(c, p) for c, p in PAR.items()]


def part_dist(a, b, comp):
    sub = [r for r in rows if r["a"] == a and r["b"] == b and meta[r["t"]]["component"] == comp and not meta[r["t"]]["unused"]]
    if not sub:
        sub = [r for r in rows if r["a"] == b and r["b"] == a and meta[r["t"]]["component"] == comp and not meta[r["t"]]["unused"]]
    sq = sum(r["sq"] for r in sub); n1 = sum(r["na2"] for r in sub); n2 = sum(r["nb2"] for r in sub)
    return float(np.sqrt(sq) / ((np.sqrt(n1) + np.sqrt(n2)) / 2)) if sub else float("nan")


def get(a, b):
    return (a, b) if (a, b) in {(r["a"], r["b"]) for r in rows[:300]} or True else (b, a)


P("\n### T4. Where a child moved from its documented parent, per part (pooled relative L2; weights and biases of the part together)\n")
P("| part | " + " | ".join(f"`{c}`←`{p}`" for c, p in CH) + " |")
P("|---|" + "---:|" * len(CH))
for comp, label in PARTS:
    P(f"| {label} | " + " | ".join(f"{part_dist(c, p, comp):.3f}" for c, p in CH) + " |")


def group(pred):
    return lambda r, m: (not m["unused"]) and pred(m)


for label, pred in (("**encoder, all**", lambda m: m["component"].startswith("enc.")),
                    ("**decoder, all**", lambda m: m["component"].startswith("dec.")),
                    ("decoder without the embedding", lambda m: m["component"].startswith("dec.") and m["component"] != "dec.embed")):
    D = dist_matrix(res, group(pred))
    P(f"| {label} | " + " | ".join(f"{D[(c, p)]:.3f}" for c, p in CH) + " |")

# ---------------------------------------------------------------------------------------------- depth
P("\n### T5. Depth: pooled relative L2 per layer, weight matrices, child ← parent\n")
for side, n in (("enc", 12), ("dec", 12)):
    P(f"\n{'Encoder' if side == 'enc' else 'Decoder'} layers 00–11:\n")
    P("| pair | " + " | ".join(f"{i:02d}" for i in range(n)) + " |")
    P("|---|" + "---:|" * n)
    for c, p in CH:
        vals = []
        for i in range(n):
            D = dist_matrix(res, lambda r, m, i=i: (not m["unused"]) and m["kind"] == "weight" and m["layer"] == f"{i:02d}" and m["component"].startswith(side))
            vals.append(D[(c, p)])
        P(f"| `{c}`←`{p}` | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

# ---------------------------------------------------------------------------------------------- additivity and the sibling pair
P("\n### T6. Along the chain `base → kurrent → xvi-xvii → from-kurrent-v2`: is the path straight?\n")
P("d(a,c) / (d(a,b) + d(b,c)); 1 = b lies on the straight line from a to c (all tensors the forward uses; the triangle inequality bounds it by 1).\n")
P("| a | b | c | d(a,b) | d(b,c) | d(a,c) | ratio |")
P("|---|---|---|---:|---:|---:|---:|")
for a, b, c in [("base", "kurrent", "xvi-xvii"), ("kurrent", "xvi-xvii", "from-kurrent-v2"), ("base", "kurrent", "from-kurrent-v2"),
                ("base", "xvi-xvii", "from-kurrent-v2"), ("kurrent", "xvi-xvii", "full-v1"), ("base", "kurrent", "full-v1")]:
    P(f"| `{a}` | `{b}` | `{c}` | {D_all[(a, b)]:.3f} | {D_all[(b, c)]:.3f} | {D_all[(a, c)]:.3f} | {D_all[(a, c)] / (D_all[(a, b)] + D_all[(b, c)]):.3f} |")

P("\n### T7. The sibling pair: same data and recipe, different parent\n")
P("| pair | all | encoder | decoder |")
P("|---|---:|---:|---:|")
De = dist_matrix(res, lambda r, m: (not m["unused"]) and m["component"].startswith("enc."))
Dd = dist_matrix(res, lambda r, m: (not m["unused"]) and m["component"].startswith("dec."))
for a, b, label in [("from-kurrent-v2", "v2", "`from-kurrent-v2` – `v2` (same data, parents `xvi-xvii` and `base`)"),
                    ("from-kurrent-v2", "full-v1", "`from-kurrent-v2` – `full-v1` (same parent, other data size/epochs)"),
                    ("from-kurrent-v2", "xvi-xvii", "`from-kurrent-v2` – its parent"), ("v2", "base", "`v2` – its parent"),
                    ("full-v1", "xvi-xvii", "`full-v1` – its parent"), ("v2", "xvi-xvii", "`v2` – `xvi-xvii` (not related by the documented tree)")]:
    P(f"| {label} | {D_all[(a, b)]:.3f} | {De[(a, b)]:.3f} | {Dd[(a, b)]:.3f} |")

# ---------------------------------------------------------------------------------------------- frozen parts, pooler
P("\n### T8. Parts that did not move at all (share of bit-identical values), child ← parent\n")
P("| tensor group | " + " | ".join(f"`{c}`←`{p}`" for c, p in CH) + " |")
P("|---|" + "---:|" * len(CH))
for comp in ["enc.pooler"] + [c for c, _ in PARTS]:
    cells = []
    for c, p in CH:
        sub = [r for r in rows if {r["a"], r["b"]} == {c, p} and meta[r["t"]]["component"] == comp]
        cells.append(f"{sum(r['identical'] * r['n'] for r in sub) / sum(r['n'] for r in sub):.4f}" if sub else "—")
    P(f"| {comp}{' (unused by the forward)' if comp == 'enc.pooler' else ''} | " + " | ".join(cells) + " |")
P("\nTensors with at least one identical value (share above 0) between parent and child, any group:\n")
for c, p in CH:
    t = sorted({r["t"] for r in rows if {r["a"], r["b"]} == {c, p} and r["identical"] > 0})
    P(f"- `{c}`←`{p}`: " + (", ".join(f"`{x}` ({max(r['identical'] for r in rows if r['t'] == x and {r['a'], r['b']} == {c, p}):.3f})" for x in t[:12]) + (f" … ({len(t)} tensors)" if len(t) > 12 else "") if t else "none"))

# ---------------------------------------------------------------------------------------------- token rows
P("\n### T9. Token rows of the tied embedding: mean relative movement per decile of token id (0 = the first, most merged, ids)\n")
P("Token id order in the RoBERTa/GPT-2 BPE roughly follows merge order, so low ids are common pieces; this is a proxy only (the training text is not known), the frequency test is stage 2.\n")
tr = meta["dec.embed.tokens"]["token_rows"]
P("| pair | rows moved by exactly 0 | " + " | ".join(f"d{i}" for i in range(10)) + " |")
P("|---|---:|" + "---:|" * 10)
for k, v in tr.items():
    a, b = k.split("|")
    if PAR.get(a) == b or PAR.get(b) == a or {a, b} == {"from-kurrent-v2", "v2"}:
        P(f"| `{a}` – `{b}` | {v['rows_identical']} | " + " | ".join(f"{x:.3f}" for x in v["rel_decile_mean"]) + " |")

print("\n".join(out))
