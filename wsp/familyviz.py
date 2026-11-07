"""Figures for Track C (docs/MODEL_FAMILY.md): where the served kraken models differ, and what stays alike.

    python -m wsp.familyviz --out docs/family/figures

Reads only ``docs/family/*.json`` (no weights, no forward pass). Style and themes are
``wsp.viz``'s: every figure as ``<name>.svg`` (light) and ``<name>-dark.svg``, Okabe–Ito colours.
The two series colours (same lineage: blue; different lineages: orange) were checked for
colour-vision separation (OKLab ΔE×100 ≥ 23 under protan/deutan/tritan in both themes); orange on
white is low-contrast (2.25 : 1), so every series also has its own marker shape and a direct label,
and docs/family/tables.md carries the numbers as tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wsp.family import BLOCKS, LINEAGE_COS, OUT, _pair
from wsp.viz import render

LAYER_KIND = {"C": "convolution", "L": "BiLSTM", "O": "output"}
# direct fine-tune edges as recorded (docs/MODELS.md) and confirmed by the weights (docs/MODEL_FAMILY.md)
EDGES = [("kraken-catmus_medieval", "kraken-cursive_inzigkofen"), ("kraken-cursive_inzigkofen", "kraken-bern-hertwig-v1"),
         ("kraken-bern-hertwig-v1", "kraken-bern-hertwig-v2"), ("kraken-bern-hertwig-v2", "kraken-bern-schlussredaktion-v1"),
         ("kraken-bern-hertwig-v2", "kraken-bern-combined-v1"), ("kraken-bohemian_19th", "kraken-german-xix-v1"),
         ("kraken-german-xix-v1", "kraken-german-xix-blb-v1")]


def _s(x: str) -> str:
    return x.replace("kraken-", "")


# position inside its lineage, base first (from EDGES); unknown models after
_RANK = {"kraken-catmus_medieval": 0, "kraken-cursive_inzigkofen": 1, "kraken-bern-hertwig-v1": 2, "kraken-bern-hertwig-v2": 3,
         "kraken-bern-schlussredaktion-v1": 4, "kraken-bern-combined-v1": 5, "kraken-bohemian_19th": 0,
         "kraken-german-xix-v1": 1, "kraken-german-xix-blb-v1": 2}


def lineage_order(w: dict, keep) -> list[str]:
    return [m for g in w["lineages"] for m in sorted((m for m in g if m in keep), key=lambda m: (_RANK.get(m, 9), m))]


def _seq_cmap(th: dict):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", [th["bg"], th["soft"] if th["bg"] == "#ffffff" else "#0072B2", th["C"]]
                                             if th["bg"] == "#ffffff" else [th["bg"], "#0b4a73", th["C"]])


def _block_axis(ax, th):
    ax.set_xticks(range(len(BLOCKS)))
    ax.set_xticklabels([f"{b}\n{LAYER_KIND[b[0]]}" for b in BLOCKS], fontsize=7.5)
    ax.axvspan(-0.5, 3.5, color=th["grid"], alpha=0.35, linewidth=0)       # convolutional front end
    ax.text(1.5, 1.02, "convolutional front end", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.5, color=th["muted"])
    ax.text(5.0, 1.02, "recurrent stack", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.5, color=th["muted"])


def plot_cka_profile(data, th):
    """Per-layer CKA for every pair (thin) and the two group means (thick), by depth."""
    import matplotlib.pyplot as plt
    f, w = data
    rel = lambda k: (_pair(w["pairs"], *k.split("|")) or {}).get("body", {}).get("cosine", 0) > LINEAGE_COS
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(BLOCKS))
    groups = {True: ("same lineage", th["C"], "o"), False: ("different lineages", th["L"], "s")}
    for k in f["cka"]["C_0"]:
        _, col, _ = groups[rel(k)]
        ax.plot(x, [f["cka"][b][k] for b in BLOCKS], color=col, linewidth=0.8, alpha=0.28, zorder=1)
    for g, (label, col, mk) in groups.items():
        ks = [k for k in f["cka"]["C_0"] if rel(k) == g]
        mean = [np.mean([f["cka"][b][k] for k in ks]) for b in BLOCKS]
        ax.plot(x, mean, color=col, linewidth=2.2, marker=mk, markersize=8, markeredgecolor=th["bg"], markeredgewidth=1.5,
                zorder=3, label=f"{label} — mean of {len(ks)} pairs")
        ax.text(len(BLOCKS) - 1 + 0.15, mean[-1], f"{label}", color=th["fg"], fontsize=8, va="center")
        for xi in (0, 6):
            ax.text(xi, mean[xi] + (0.045 if g else -0.07), f"{mean[xi]:.2f}", ha="center", fontsize=7.5, color=th["fg"])
    _block_axis(ax, th)
    ax.set_ylim(-0.02, 1.05); ax.set_xlim(-0.5, len(BLOCKS) + 0.9)
    ax.set_ylabel("linear CKA (1 = same features up to unit order and scale)")
    ax.grid(axis="y", linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title(f"Where the served kraken models agree, layer by layer ({len(f['models'])} models, {len(f['cka']['C_0'])} pairs)\n"
                 f"same {f['probe']['lines']} probe lines for every model; thin lines: single pairs", loc="left", pad=22, fontsize=9.5)
    return fig


def plot_cka_matrices(data, th):
    """Small multiples: model × model CKA per block, models ordered by lineage."""
    import matplotlib.pyplot as plt
    f, w = data
    order = lineage_order(w, f["models"])
    bounds = np.cumsum([len([m for m in g if m in f["models"]]) for g in w["lineages"]])[:-1]
    n = len(order)
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.2), constrained_layout=True)
    cmap = _seq_cmap(th)
    for ax, b in zip(axes.ravel(), BLOCKS):
        M = np.eye(n)
        for i, a in enumerate(order):
            for j, c in enumerate(order):
                if i != j:
                    M[i, j] = _pair(f["cka"][b], a, c)
        im = ax.imshow(M, vmin=0, vmax=1, cmap=cmap)
        for t in bounds:
            ax.axhline(t - 0.5, color=th["fg"], linewidth=0.8); ax.axvline(t - 0.5, color=th["fg"], linewidth=0.8)
        ax.set_title(f"{b} ({LAYER_KIND[b[0]]}) — mean {np.mean(M[~np.eye(n, dtype=bool)]):.2f}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0, 0].set_yticks(range(n)); axes[0, 0].set_yticklabels([_s(m) for m in order], fontsize=6.5)
    axes[1, 0].set_yticks(range(n)); axes[1, 0].set_yticklabels([_s(m) for m in order], fontsize=6.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("linear CKA", fontsize=8); cb.outline.set_visible(False)
    fig.suptitle("Per-layer similarity of every pair, models grouped by lineage (lines separate lineages)",
                 x=0.0, ha="left", fontsize=9.5)
    return fig


def plot_weight_drift(data, th):
    """Heat strip: how far each fine-tune moved each block (relative distance), with the values written in."""
    import matplotlib.pyplot as plt
    w = data
    rows = [(a, b, _pair(w["pairs"], a, b)) for a, b in EDGES if _pair(w["pairs"], a, b)]
    M = np.array([[p[blk]["rel_dist"] for blk in BLOCKS] for _, _, p in rows])
    fig, ax = plt.subplots(figsize=(9, 0.48 * len(rows) + 1.7))
    vmax = float(np.ceil(M.max() * 10) / 10)
    im = ax.imshow(M, vmin=0, vmax=vmax, cmap=_seq_cmap(th), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color=th["bg"] if M[i, j] > 0.55 * vmax else th["fg"])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{_s(a)} → {_s(b)}" for a, b, _ in rows], fontsize=8)
    ax.set_xticks(range(len(BLOCKS))); ax.set_xticklabels([f"{b}\n{LAYER_KIND[b[0]]}" for b in BLOCKS], fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01); cb.outline.set_visible(False)
    cb.set_label("‖child − parent‖ / mean norm", fontsize=8)
    ax.set_title("How far each fine-tune moved each layer (weight space, parent → child)\n"
                 "lighter = barely moved; the output layer is compared on the characters both alphabets share",
                 loc="left", fontsize=9.5)
    return fig


def plot_dead_filters(data, th):
    """Dead first-layer filters per model, in lineage order: what never fires, and so never changes."""
    import matplotlib.pyplot as plt
    f, w = data
    order = lineage_order(w, f["dead_filters"])
    lin_of = {m: i for i, g in enumerate(w["lineages"]) for m in g}
    vals = [f["dead_filters"][m]["C_0"]["dead"] for m in order]
    tot = f["dead_filters"][order[0]]["C_0"]["filters"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(order))
    ax.bar(x, vals, width=0.62, color=th["C"], edgecolor=th["bg"], linewidth=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.25, f"{v}", ha="center", fontsize=8, color=th["fg"])
    prev = None
    for xi, m in zip(x, order):
        if prev is not None and lin_of[m] != lin_of[prev]:
            ax.axvline(xi - 0.5, color=th["muted"], linewidth=0.8)
        prev = m
    ax.set_xticks(x); ax.set_xticklabels([_s(m) for m in order], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel(f"dead C_0 filters (of {tot})"); ax.set_ylim(0, max(vals) + 2.5)
    ax.grid(axis="y", linewidth=0.5); ax.set_axisbelow(True)
    ax.set_title("Dead first-layer filters per model, lineage by lineage (base on the left of each group)\n"
                 "a dead filter gets no gradient: these are the parameters that stay bit-identical in a fine-tune",
                 loc="left", fontsize=9.5)
    return fig


FILE_TO_ID = {"catmus-medieval-1.6.0": "catmus_medieval", "cursive_johannes_jaeck": "cursive_inzigkofen",
              "rec_bohemica": "bohemian_19th", "bifrost_01": "bifrost_old_norse"}


def _mid(stem: str) -> str:
    return FILE_TO_ID.get(stem, _s(stem))


def plot_exchange(data, th):
    """RQ3: ΔCER per block when that block moves β of the way to the general model's, per set, one panel per pair."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    exs, beta = data
    fig, axes = plt.subplots(1, len(exs), figsize=(3.4 * len(exs) + 0.6, 4.9), sharey=True, squeeze=False)
    marks = ["o", "s", "D"]; cols = [th["C"], th["L"], th["O"]]
    for ax, e in zip(axes[0], exs):
        y = np.arange(len(e["blocks"]))
        for si, s in enumerate(e["sets"]):
            for r in e["rows"]:
                if r["beta"] != beta:
                    continue
                v = r["sets"][s]["vs_S"]; d = 100 * v["delta_cer"]; lo, hi = (100 * t for t in v["ci95"])
                yi = e["blocks"].index(r["block"]) + (si - 1) * 0.22
                sig = lo > 0 or hi < 0
                ax.hlines(yi, lo, hi, color=cols[si], linewidth=1.3)
                ax.plot(d, yi, marker=marks[si], color=cols[si], markersize=6, markerfacecolor=cols[si] if sig else th["bg"],
                        markeredgewidth=1.3)
        ax.axvline(0, color=th["muted"], linewidth=0.9)
        ax.set_yticks(y); ax.set_yticklabels(e["blocks"]); ax.invert_yaxis()
        ax.grid(axis="x", linewidth=0.5); ax.set_axisbelow(True)
        ax.set_title(f"{_mid(e['specific'])}\n← {_mid(e['general'])}", fontsize=8.5, loc="left")
        ax.set_xlabel("ΔCER (points; < 0 better)", fontsize=8)
        handles = [Line2D([], [], marker=marks[i], color=cols[i], linestyle="-", markersize=5) for i in range(len(e["sets"]))]
        labels = [f"{s} {100 * e['c0'][s]['cer']:.0f} %" + (" (1st)" if i == 0 else "") for i, s in enumerate(e["sets"])]
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=7, ncol=1)
    what = "transplanted" if beta == 1.0 else f"moved {beta:g} of the way"
    fig.suptitle(f"RQ3: one block of the specific model {what} to the general model's (β = {beta:g})\n"
                 "per set: S's CER unedited in the legend; filled = paired 95 % interval excludes 0, hollow = includes 0",
                 x=0.0, y=1.04, ha="left", fontsize=9.5)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.familyviz")
    ap.add_argument("--out", default=str(OUT / "figures")); ap.add_argument("--format", default="svg")
    a = ap.parse_args(argv)
    out = Path(a.out)
    w = json.loads((OUT / "weights.json").read_text()); f = json.loads((OUT / "functional.json").read_text())
    written = []
    written += render(plot_cka_profile, (f, w), out, "family-cka-profile", a.format)
    written += render(plot_cka_matrices, (f, w), out, "family-cka-matrices", a.format)
    written += render(plot_weight_drift, w, out, "family-weight-drift", a.format)
    written += render(plot_dead_filters, (f, w), out, "family-dead-filters", a.format)
    exs = [json.loads(p.read_text()) for p in sorted(OUT.glob("exchange-*.json"))]
    if exs:
        for beta in sorted({r["beta"] for e in exs for r in e["rows"]}):
            written += render(plot_exchange, (exs, beta), out, f"family-exchange-b{beta:g}", a.format)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
