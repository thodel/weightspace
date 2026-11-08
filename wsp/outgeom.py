"""Output-layer geometry, parent and child together (epic thodel/weightspace#3, sub-task 5).

    python -m wsp.outgeom --parent models/kraken/kraken-german-xix-v1.mlmodel \\
        --child models/kraken/kraken-german-xix-blb-v1.mlmodel --practice data/practice/blb-train-300.jsonl \\
        --out docs/viz

The output layer ``O_*`` has one row (weights + bias) per character and one for the CTC blank. Rows of the
two models are paired **by character** (``wsp.align.codec_row_map``), never by index. Two panels:

1. **Projection.** The rows of both models projected to 2-D with one PCA fitted on parent and child rows
   together (so both live in one coordinate system), each shared character drawn as an arrow parent → child.
   Rows only the child has (its grown alphabet) are drawn as points.
2. **Size and bias.** Each row's weight norm and bias, parent against child, on the identity line: a row
   above it grew in the fine-tune. This is where the phase-0 readings ("rare rows inflated, blank weakest")
   can be checked by eye.

Classes from the practice lines (counts of the reference characters, NFD): blank; frequent (the 20 most
frequent characters); rare (fewer than 10 occurrences, or absent); other (the rest); new (the child's rows
the parent lacks). No forward pass; writes ``outgeom-<child>.svg`` / ``-dark.svg`` and a JSON with every row.
Colours Okabe–Ito with one marker shape per class (two colour pairs fall below 8 OKLab ΔE under CVD, so the
shape is the second cue).
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from wsp.align import codec_row_map
from wsp.viz import render

CLASSES = ["frequent", "other", "rare", "new", "blank"]
STYLE = {"frequent": ("C", "o"), "other": ("muted", "s"), "rare": ("L", "D"), "new": ("O", "^"), "blank": ("hi", "X")}


def _out(subject):
    t = subject.tensors()
    w = next(n for n in t if n.startswith("O_") and n.endswith("weight"))
    return t[w].astype(np.float64), t[w.replace("weight", "bias")].astype(np.float64)


def geometry(parent, child, counts: Counter) -> dict:
    wp, bp = _out(parent); wc, bc = _out(child)
    cp, cc = parent.codec_map(), child.codec_map()
    rm = codec_row_map(cc, cp)                                    # child row -> parent row, shared characters
    inv = {r: ch for ch, r in cc.items()}
    freq = {ch for ch, _ in counts.most_common(20)}
    rows = []
    for r in range(wc.shape[0]):
        ch = "" if r == 0 else inv.get(r, "?")
        pr = 0 if r == 0 else rm.get(r)
        cls = ("blank" if r == 0 else "new" if pr is None else "frequent" if ch in freq
               else "rare" if counts.get(ch, 0) < 10 else "other")
        rows.append({"row": r, "char": ch, "name": "blank" if r == 0 else unicodedata.name(ch, f"U+{ord(ch):04X}") if len(ch) == 1 else ch,
                     "class": cls, "count": counts.get(ch, 0), "parent_row": pr,
                     "norm_child": float(np.linalg.norm(wc[r])), "bias_child": float(bc[r]),
                     "norm_parent": None if pr is None else float(np.linalg.norm(wp[pr])),
                     "bias_parent": None if pr is None else float(bp[pr]),
                     "moved": None if pr is None else float(np.linalg.norm(wc[r] - wp[pr]))})
    shared = [x for x in rows if x["parent_row"] is not None]
    X = np.vstack([wc[[x["row"] for x in rows]], wp[[x["parent_row"] for x in shared]]])
    mu = X.mean(0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    P = vt[:2]
    for x in rows:
        x["xy_child"] = [float(v) for v in (wc[x["row"]] - mu) @ P.T]
        if x["parent_row"] is not None:
            x["xy_parent"] = [float(v) for v in (wp[x["parent_row"]] - mu) @ P.T]
    var = (s ** 2) / (s ** 2).sum()
    summary = {}
    for cls in CLASSES:
        xs = [x for x in rows if x["class"] == cls]
        sh = [x for x in xs if x["norm_parent"] is not None]
        summary[cls] = {"rows": len(xs), "median_norm_child": float(np.median([x["norm_child"] for x in xs])) if xs else None,
                        "median_norm_ratio": float(np.median([x["norm_child"] / x["norm_parent"] for x in sh])) if sh else None,
                        "share_grown": float(np.mean([x["norm_child"] > x["norm_parent"] for x in sh])) if sh else None,
                        "median_bias_change": float(np.median([x["bias_child"] - x["bias_parent"] for x in sh])) if sh else None,
                        "median_moved": float(np.median([x["moved"] for x in sh])) if sh else None}
    return {"rows": rows, "explained": [float(var[0]), float(var[1])], "shared": len(shared), "new": len(rows) - len(shared),
            "by_class": summary}


def plot(data, th):
    import matplotlib.pyplot as plt
    g, title = data
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 5.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    col = lambda c: th[STYLE[c][0]]
    for cls in CLASSES:
        xs = [x for x in g["rows"] if x["class"] == cls]
        if not xs:
            continue
        for x in xs:
            if "xy_parent" in x:
                (px, py), (cx, cy) = x["xy_parent"], x["xy_child"]
                a1.annotate("", xy=(cx, cy), xytext=(px, py),
                            arrowprops=dict(arrowstyle="-|>", color=col(cls), lw=0.6, alpha=0.6, shrinkA=0, shrinkB=0, mutation_scale=6))
        a1.scatter([x["xy_child"][0] for x in xs], [x["xy_child"][1] for x in xs], s=26 if cls != "blank" else 70,
                   c=col(cls), marker=STYLE[cls][1], edgecolors=th["bg"], linewidths=0.6, zorder=3,
                   label=f"{cls} ({len(xs)})")
        sh = [x for x in xs if x["norm_parent"] is not None]
        a2.scatter([x["norm_parent"] for x in sh], [x["norm_child"] for x in sh], s=24 if cls != "blank" else 70,
                   c=col(cls), marker=STYLE[cls][1], edgecolors=th["bg"], linewidths=0.6, zorder=3)
        a3.scatter([x["bias_parent"] for x in sh], [x["bias_child"] for x in sh], s=24 if cls != "blank" else 70,
                   c=col(cls), marker=STYLE[cls][1], edgecolors=th["bg"], linewidths=0.6, zorder=3)
    for ax in (a2, a3):
        lo, hi = ax.get_xlim(); lo2, hi2 = ax.get_ylim(); lo, hi = min(lo, lo2), max(hi, hi2)
        ax.plot([lo, hi], [lo, hi], color=th["muted"], lw=0.9, zorder=1); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.grid(lw=0.5); ax.set_axisbelow(True)
    # labels: blank, the 6 rows that moved most, the 4 most frequent
    lab = sorted((x for x in g["rows"] if x["moved"] is not None), key=lambda x: -x["moved"])[:6]
    lab += [x for x in g["rows"] if x["class"] == "blank"]
    seen = set()
    for x in lab:
        if x["row"] in seen:
            continue
        seen.add(x["row"])
        t = "blank" if x["class"] == "blank" else ("␣" if x["char"] == " " else ("◌" + x["char"] if unicodedata.category(x["char"][:1] or "a").startswith("M") else x["char"]))
        a1.annotate(t, x["xy_child"], xytext=(4, 3), textcoords="offset points", fontsize=8, color=th["fg"])
        if x["norm_parent"] is not None:
            a2.annotate(t, (x["norm_parent"], x["norm_child"]), xytext=(4, 3), textcoords="offset points", fontsize=8, color=th["fg"])
            a3.annotate(t, (x["bias_parent"], x["bias_child"]), xytext=(4, 3), textcoords="offset points", fontsize=8, color=th["fg"])
    a1.set_xlabel(f"PC 1 ({100 * g['explained'][0]:.0f} % of variance)"); a1.set_ylabel(f"PC 2 ({100 * g['explained'][1]:.0f} %)")
    a1.set_title("rows projected: arrow parent → child", loc="left", fontsize=9)
    a2.set_xlabel("row weight norm, parent"); a2.set_ylabel("row weight norm, child")
    a2.set_title("size: above the line = grew", loc="left", fontsize=9)
    a3.set_xlabel("bias, parent"); a3.set_ylabel("bias, child")
    a3.set_title("bias: above the line = favoured more", loc="left", fontsize=9)
    a1.legend(loc="best", fontsize=7.5, frameon=False, title="row class", title_fontsize=8)
    fig.suptitle(title, x=0.0, ha="left", fontsize=9.5)
    fig.tight_layout()
    return fig


def main(argv=None):
    from wsp.data import load
    from wsp.kraken_subject import KrakenSubject
    from wsp.metrics import normalise
    ap = argparse.ArgumentParser(prog="wsp.outgeom")
    ap.add_argument("--parent", required=True); ap.add_argument("--child", required=True)
    ap.add_argument("--practice", required=True); ap.add_argument("--out", default="docs/viz")
    a = ap.parse_args(argv)
    if "heldout" in Path(a.practice).parts:
        raise SystemExit("practice lines only")
    parent, child = KrakenSubject(a.parent), KrakenSubject(a.child)
    counts = Counter(ch for x in load(a.practice) for ch in normalise(x["text"]))
    g = geometry(parent, child, counts)
    g.update({"parent": Path(a.parent).stem, "child": Path(a.child).stem, "practice": Path(a.practice).stem})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = f"outgeom-{Path(a.child).stem}"
    (out / f"{stem}.json").write_text(json.dumps(g, indent=1, ensure_ascii=False))
    pr = lambda s: s.removeprefix("kraken-")
    title = (f"Output layer {pr(g['parent'])} → {pr(g['child'])}: {g['shared']} rows paired by character, {g['new']} new; "
             f"classes from {g['practice']} (frequent = 20 most common, rare = under 10 occurrences)")
    for p in render(plot, (g, title), out, stem):
        print(p)


if __name__ == "__main__":
    main()
