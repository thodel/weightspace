#!/usr/bin/env python3
"""Tables for the third-party models (docs/TROCR_READINGS.md section 'Third-party models') from docs/trocr/readings_all.json.

    python scripts/trocr_extras_report.py docs/trocr/readings_all.json > docs/trocr/extras_tables.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp.trocr_family import dist_matrix  # noqa: E402

res = json.load(open(sys.argv[1]))
M = list(res["models"])
meta = res["tensors"]
out = []
P = out.append

core = [m for m in M if m in ("base", "kurrent", "xvi-xvii", "from-kurrent-v2", "v2", "full-v1", "medieval")]
extra = [m for m in M if m not in core]

P("### E1. Tensors that are not in every model (all 13)\n")
P("| tensor | present in | missing in |")
P("|---|---|---|")
for t, m in sorted(meta.items()):
    if len(m["present_in"]) != len(M):
        P(f"| `{t}` | {', '.join(f'`{x}`' for x in m['present_in'])} | {', '.join(f'`{x}`' for x in M if x not in m['present_in'])} |")

# "other.*" is the sinusoidal position buffer of the stage-1 architecture (``embed_positions._float_tensor``): not a parameter, stored with
# uninitialised values of the order 1e37 that would swamp every norm, so it is left out of every distance
sel_all = lambda r, m: not m["unused"] and m["component"] != "other"
sel_w = lambda r, m: (not m["unused"]) and m["component"] != "other" and m["kind"] == "weight"
Da, Dw = dist_matrix(res, sel_all), dist_matrix(res, sel_w)
P("\n### E2. Distance of each third-party model to every other model (pooled over the tensors both have; weight matrices only, then all tensors)\n")
for label, D in (("weight matrices only", Dw), ("all tensors the forward uses", Da)):
    P(f"\n{label}:\n")
    P("| | " + " | ".join(f"`{m}`" for m in M) + " |")
    P("|---|" + "---:|" * len(M))
    for a in extra:
        P(f"| `{a}` | " + " | ".join("—" if a == b else f"{D[(a, b)]:.3f}" for b in M) + " |")

P("\n### E3. Nearest model and the two roots\n")
P("| model | documented parent (hub card) | nearest (weights only) | d | d to `base` | d to `stage1` | nearest (all tensors) | d |")
P("|---|---|---|---:|---:|---:|---:|---:|")
doc = {"swe": "base", "lamop": "base", "stage1": "(pre-training root)", "catmus": "not stated", "ru": "base (by the card's name)", "hanse": "not stated"}
for a in extra:
    n1 = min((Dw[(a, b)], b) for b in M if b != a)
    n2 = min((Da[(a, b)], b) for b in M if b != a)
    P(f"| `{a}` | {doc[a]} | `{n1[1]}` | {n1[0]:.3f} | {Dw[(a, 'base')]:.3f} | {Dw[(a, 'stage1')] if a != 'stage1' else float('nan'):.3f} | `{n2[1]}` | {n2[0]:.3f} |")
print("\n".join(out))
