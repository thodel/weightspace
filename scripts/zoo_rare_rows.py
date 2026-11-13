"""Observation 1 with measured frequencies: output-row change by how often the character occurs in the child's own practice lines.

    python scripts/zoo_rare_rows.py <models/kraken> docs/zoo/rare_rows.json > docs/zoo/rare_rows.md

Only the in-house pairs have training-side text (data/practice/*.jsonl, 300 lines each, seed 20261106); no text is available for the zoo's
fine-tunes, so the zoo pairs stay with the proxy of ZOO_SURVEY.md. Rows are paired by character; the blank is not a character.
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from wsp import align
from wsp.data import load
from wsp.kraken_subject import KrakenSubject

models = Path(sys.argv[1])
PAIRS = [("cursive_inzigkofen", "bern-hertwig-v1", "bern-hertwig-train-300"), ("cursive_inzigkofen", "bern-hertwig-v2", "bern-hertwig-train-300"),
         ("bern-hertwig-v1", "bern-hertwig-v2", "bern-hertwig-train-300"), ("bern-hertwig-v2", "bern-schlussredaktion-v1", "bern-train-300"),
         ("bern-hertwig-v2", "bern-combined-v1", "bern-train-300"), ("german-xix-v1", "german-xix-blb-v1", "blb-train-300")]


def sub(mid):
    return KrakenSubject(models / f"kraken-{mid}.mlmodel")


out, md = [], []
cache = {}
for pa, ch, ds in PAIRS:
    a = cache.setdefault(pa, sub(pa)); b = cache.setdefault(ch, sub(ch))
    counts = Counter("".join(x["text"] for x in load(Path("data/practice") / f"{ds}.jsonl")))
    ta, tb = a.tensors(), b.tensors()
    wa = next(n for n in ta if n.startswith("O_") and n.endswith("lin.weight")); ba_ = wa.replace("weight", "bias")
    Wa, Wb, ba, bb = (ta[wa].astype(float), tb[wa].astype(float), ta[ba_].astype(float), tb[ba_].astype(float))
    ca, cb = a.codec_map(), b.codec_map()
    rm = align.codec_row_map(ca, cb)
    rows = []
    for c, ia in ca.items():
        if c not in cb:
            continue
        ib = rm[ia]; na, nb = np.linalg.norm(Wa[ia]), np.linalg.norm(Wb[ib])
        rows.append({"char": c, "count": counts.get(c, 0), "rel": float(np.linalg.norm(Wb[ib] - Wa[ia]) / na), "ratio": float(nb / na), "dbias": float(bb[ib] - ba[ia])})
    cls = {"absent (0)": lambda n: n == 0, "rare (1–9)": lambda n: 1 <= n < 10, "frequent (≥10)": lambda n: n >= 10}
    rec = {"parent": pa, "child": ch, "set": ds, "rows": len(rows), "classes": {}}
    for k, f in cls.items():
        r = [x for x in rows if f(x["count"])]
        rec["classes"][k] = {"n": len(r), "median_rel": float(np.median([x["rel"] for x in r])) if r else None,
                             "median_norm_ratio": float(np.median([x["ratio"] for x in r])) if r else None,
                             "median_dbias": float(np.median([x["dbias"] for x in r])) if r else None,
                             "norm_up": sum(x["ratio"] > 1 for x in r), "bias_down": sum(x["dbias"] < 0 for x in r)}
    lc = np.log1p([x["count"] for x in rows])
    for key in ("rel", "ratio", "dbias"):
        rho, pv = spearmanr(lc, [x[key] for x in rows])
        rec[f"spearman_{key}"] = {"rho": float(rho), "p": float(pv)}
    out.append(rec)
json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)

md.append("| pair | practice set | class | rows | median rel. move | median norm ratio | median bias Δ | norm up | bias down |")
md.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
for r in out:
    for k, v in r["classes"].items():
        if v["n"]:
            md.append(f"| `{r['parent']}` → `{r['child']}` | {r['set']} | {k} | {v['n']} | {v['median_rel']:.2f} | {v['median_norm_ratio']:.2f} | {v['median_dbias']:+.2f} | {v['norm_up']} | {v['bias_down']} |")
md.append("\n| pair | ρ(log count, rel. move) | ρ(log count, norm ratio) | ρ(log count, bias Δ) |")
md.append("|---|---:|---:|---:|")
for r in out:
    md.append(f"| `{r['parent']}` → `{r['child']}` | " + " | ".join(f"{r[f'spearman_{k}']['rho']:+.2f} (p={r[f'spearman_{k}']['p']:.3f})" for k in ("rel", "ratio", "dbias")) + " |")
print("\n".join(md))
