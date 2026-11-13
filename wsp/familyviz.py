"""Figures for Track C (docs/MODEL_FAMILY.md): where the served kraken models differ, and what stays alike.

    python -m wsp.familyviz --out docs/family/figures

Reads only ``docs/family/*.json`` (no weights, no forward pass). Style and themes are
``wsp.viz``'s: every figure as ``<name>.svg`` (light) and ``<name>-dark.svg``, Okabe–Ito colours.
Relation classes are coloured same lineage = blue (circle), distant relatives = green (diamond),
unrelated = orange (square). Colour-vision separation was computed (OKLab ΔE×100, Machado 2009
matrices): worst pair 8.6 (blue–green, tritan, light theme), 11.2 in the dark theme; orange on white is
low-contrast (2.25 : 1), so every class also has its own marker shape and a direct label, and
docs/family/tables.md carries the numbers as tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wsp.family import DISTANT, LINEAGE_COS, OUT, _pair
from wsp.viz import render

LAYER_KIND = {"C": "convolution", "G": "group norm", "L": "BiLSTM", "O": "output"}
SHORT_KIND = {"C": "conv", "G": "norm", "L": "LSTM", "O": "out"}
# direct fine-tune edges as recorded for the in-house models (docs/MODELS.md), confirmed by the weights
EDGES = [("kraken-catmus_medieval", "kraken-cursive_inzigkofen"), ("kraken-cursive_inzigkofen", "kraken-bern-hertwig-v1"),
         ("kraken-bern-hertwig-v1", "kraken-bern-hertwig-v2"), ("kraken-bern-hertwig-v2", "kraken-bern-schlussredaktion-v1"),
         ("kraken-bern-hertwig-v2", "kraken-bern-combined-v1"), ("kraken-bohemian_19th", "kraken-german-xix-v1"),
         ("kraken-german-xix-v1", "kraken-german-xix-blb-v1")]
FILE_TO_ID = {"catmus-medieval-1.6.0": "catmus_medieval", "cursive_johannes_jaeck": "cursive_inzigkofen",
              "rec_bohemica": "bohemian_19th", "bifrost_01": "bifrost_old_norse"}


def _s(x: str) -> str:
    return x.replace("kraken-", "")


def _mid(stem: str) -> str:
    return FILE_TO_ID.get(stem, _s(stem))


def learn_names(w: dict) -> None:
    """File stem → gateway id for every model in weights.json (Zenodo files are named by their authors)."""
    for mid, info in w["models"].items():
        FILE_TO_ID.setdefault(Path(info["path"]).stem, _s(mid))


def edges(w: dict) -> list[tuple[str, str]]:
    """Parent → child: the recorded in-house fine-tunes plus every stated base the weights confirm."""
    e = list(EDGES)
    for r in w.get("stated_bases", []):
        if r["verdict"] == "confirmed" and (r["base_model"], r["model"]) not in e:
            e.append((r["base_model"], r["model"]))
    return e


def lineage_order(w: dict, keep) -> list[str]:
    """Models of ``keep`` grouped by lineage (largest first), base first inside a lineage, singletons last."""
    parent = {c: p for p, c in edges(w)}

    def depth(m, seen=()):
        return 0 if m not in parent or m in seen else 1 + depth(parent[m], seen + (m,))
    out = []
    for g in w["lineages"]:
        out += sorted((m for m in g if m in keep), key=lambda m: (depth(m), m))
    return out


def relation(w, a, b) -> str:
    c = ((_pair(w["pairs"], a, b) or {}).get("body") or {}).get("cosine") or 0
    return "same" if c > LINEAGE_COS else "distant" if c > DISTANT[0] else "unrelated"


def _seq_cmap(th: dict):
    from matplotlib.colors import LinearSegmentedColormap
    stops = [th["bg"], th["soft"], th["C"]] if th["bg"] == "#ffffff" else [th["bg"], "#0b4a73", th["C"]]
    return LinearSegmentedColormap.from_list("seq", stops)


def _kind(b: str) -> str:
    return LAYER_KIND.get(b[0], b)


def plot_cka_profile(data, th):
    """Per-layer CKA by depth, one panel per architecture group: mean and 10–90 % band per relation class."""
    import matplotlib.pyplot as plt
    f, w = data
    groups = [g for g in f["groups"] if g["cka"] and len(g["models"]) >= 3]
    fig, axes = plt.subplots(1, len(groups), figsize=(6.2 * len(groups), 4.6), squeeze=False,
                             gridspec_kw={"width_ratios": [len(g["blocks"]) for g in groups]})
    cls = {"same": ("same lineage", th["C"], "o"), "distant": ("distant relatives", th["O"], "D"),
           "unrelated": ("unrelated", th["L"], "s")}
    for ax, g in zip(axes[0], groups):
        B = g["blocks"]; x = np.arange(len(B))
        ends = []
        for key, (label, col, mk) in cls.items():
            ks = [k for k in g["cka"][B[0]] if relation(w, *k.split("|")) == key]
            if not ks:
                continue
            M = np.array([[g["cka"][b][k] for b in B] for k in ks])
            mean = M.mean(0)
            if len(ks) >= 5:
                ax.fill_between(x, np.percentile(M, 10, 0), np.percentile(M, 90, 0), color=col, alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=col, linewidth=2.2, marker=mk, markersize=7.5, markeredgecolor=th["bg"],
                    markeredgewidth=1.4, label=f"{label} ({len(ks)} pairs)")
            ends.append([mean[-1], f"{label} {mean[-1]:.2f}"])
        ends.sort()
        for i in range(1, len(ends)):                       # keep end labels at least 0.05 apart
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.05)
        for y_, txt in ends:
            ax.text(x[-1] + 0.18, y_, txt, color=th["fg"], fontsize=7.5, va="center")
        n_conv = sum(b[0] in "CG" for b in B)
        ax.axvspan(-0.5, n_conv - 0.5, color=th["grid"], alpha=0.35, linewidth=0)
        ax.set_xticks(x); ax.set_xticklabels([f"{b}\n{SHORT_KIND.get(b[0], '')}" for b in B], fontsize=7.5)
        ax.set_ylim(-0.02, 1.05); ax.set_xlim(-0.5, len(B) + 1.3)
        ax.grid(axis="y", linewidth=0.5); ax.set_axisbelow(True)
        ax.set_title(f"group {g['group']}: {len(g['models'])} models of one architecture", loc="left", fontsize=9)
        ax.legend(loc="lower left", frameon=False, fontsize=7.5)
    axes[0, 0].set_ylabel("linear CKA (1 = same features up to unit order and scale)")
    fig.suptitle(f"Where the served kraken models agree, layer by layer — mean and 10–90 % band per relation; "
                 f"same {f['probe']['lines']} probe lines for every model; grey: convolutional front end",
                 x=0.0, ha="left", fontsize=9.5)
    return fig


def plot_cka_matrices(data, th):
    """Small multiples: model × model CKA per block for the largest architecture group, ordered by lineage."""
    import matplotlib.pyplot as plt
    f, w = data
    g = max((g for g in f["groups"] if g["cka"]), key=lambda g: len(g["models"]))
    B = g["blocks"]
    order = lineage_order(w, g["models"])
    lin_of = {m: i for i, l in enumerate(w["lineages"]) for m in l}
    size = {i: len(l) for i, l in enumerate(w["lineages"])}
    bounds = [i for i in range(1, len(order)) if lin_of[order[i]] != lin_of[order[i - 1]]
              and (size[lin_of[order[i]]] > 1 or size[lin_of[order[i - 1]]] > 1)]     # frame multi-model lineages only
    n = len(order)
    fig, axes = plt.subplots(2, (len(B) + 1) // 2, figsize=(14, 7.8), constrained_layout=True)
    cmap = _seq_cmap(th)
    for ax, b in zip(axes.ravel(), B):
        M = np.eye(n)
        for i, a in enumerate(order):
            for j, c in enumerate(order):
                if i != j:
                    M[i, j] = _pair(g["cka"][b], a, c)
        im = ax.imshow(M, vmin=0, vmax=1, cmap=cmap)
        for t in bounds:
            ax.axhline(t - 0.5, color=th["fg"], linewidth=0.35); ax.axvline(t - 0.5, color=th["fg"], linewidth=0.35)
        ax.set_title(f"{b} ({_kind(b)}) — mean {np.mean(M[~np.eye(n, dtype=bool)]):.2f}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for row in axes:
        row[0].set_yticks(range(n)); row[0].set_yticklabels([_s(m) for m in order], fontsize=5)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("linear CKA", fontsize=8); cb.outline.set_visible(False)
    fig.suptitle(f"Group {g['group']} ({n} models): per-layer similarity of every pair, models grouped by lineage "
                 "(thin lines frame lineages of 2+ models; unrelated models follow, one by one)", x=0.0, ha="left", fontsize=9.5)
    return fig


def plot_weight_drift(data, th):
    """Heat strips: how far each recorded or confirmed fine-tune moved each block, one strip per architecture group."""
    import matplotlib.pyplot as plt
    w = data
    by_group = {}
    gname = {m: g for g in w["groups"] for m in g["models"]}
    for a, b in edges(w):
        p = _pair(w["pairs"], a, b)
        if p:
            by_group.setdefault(gname[a]["name"], (gname[a], []))[1].append((a, b, p))
    strips = sorted(by_group.values(), key=lambda x: x[0]["name"])
    heights = [len(rows) + 1.2 for _, rows in strips]
    fig, axes = plt.subplots(len(strips), 1, figsize=(10, 0.42 * sum(heights) + 1.2), squeeze=False,
                             gridspec_kw={"height_ratios": heights}, constrained_layout=True)
    vmax = 1.0
    for ax, (g, rows) in zip(axes[:, 0], strips):
        B = g["blocks"]
        M = np.array([[p[b]["rel_dist"] if p.get(b, {}).get("rel_dist") is not None else np.nan for b in B] for _, _, p in rows])
        im = ax.imshow(M, vmin=0, vmax=vmax, cmap=_seq_cmap(th), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color=th["bg"] if M[i, j] > 0.55 * vmax else th["fg"])
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{_s(a)} → {_s(b)}" for a, b, _ in rows], fontsize=7.5)
        ax.set_xticks(range(len(B))); ax.set_xticklabels([f"{b}\n{_kind(b)}" for b in B], fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"group {g['name']}", loc="left", fontsize=8.5)
    cb = fig.colorbar(im, ax=axes[:, 0], shrink=0.5, pad=0.01); cb.outline.set_visible(False)
    cb.set_label("‖child − parent‖ / mean norm", fontsize=8)
    fig.suptitle("How far each fine-tune moved each layer (weight space, parent → child; recorded in-house fine-tunes and every "
                 "stated base the weights confirm)\nlighter = barely moved; output layer on the characters both alphabets share",
                 x=0.0, ha="left", fontsize=9.5)
    return fig


def plot_dead_filters(data, th):
    """Dead first-layer filters per model, by architecture group and lineage."""
    import matplotlib.pyplot as plt
    f, w = data
    dead = {m: d for g in f["groups"] for m, d in g["dead_filters"].items() if d}
    gname = {m: g["name"] for g in w["groups"] for m in g["models"]}
    order = sorted(lineage_order(w, dead), key=lambda m: gname[m])        # stable: lineage order inside a group
    first = {m: next(iter(dead[m])) for m in order}
    vals = [dead[m][first[m]]["dead"] / dead[m][first[m]]["filters"] * 100 for m in order]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    x = np.arange(len(order))
    ax.bar(x, vals, width=0.7, color=th["C"], edgecolor=th["bg"], linewidth=1.5)
    for xi, m in zip(x, order):
        d = dead[m][first[m]]
        ax.text(xi, vals[xi] + 0.8, f"{d['dead']}", ha="center", fontsize=6.5, color=th["fg"])
    for i in range(1, len(order)):
        if gname[order[i]] != gname[order[i - 1]]:
            ax.axvline(i - 0.5, color=th["muted"], linewidth=0.9)
            ax.text(i - 0.4, max(vals) + 4, gname[order[i]], fontsize=8, color=th["muted"])
    ax.text(-0.4, max(vals) + 4, gname[order[0]], fontsize=8, color=th["muted"])
    ax.set_xticks(x); ax.set_xticklabels([_s(m) for m in order], rotation=55, ha="right", fontsize=6.5)
    ax.set_ylabel("dead first-layer filters (% of the layer)"); ax.set_ylim(0, max(vals) + 7)
    ax.grid(axis="y", linewidth=0.5); ax.set_axisbelow(True)
    ax.set_title("Dead first-layer filters per model (number on the bar), by architecture group A–E, lineages kept together\n"
                 "a dead filter outputs 0 on every probe frame and gets no gradient: the parameters that stay bit-identical in a fine-tune",
                 loc="left", fontsize=9.5)
    return fig


def plot_exchange(data, th):
    """RQ3: ΔCER per block when that block moves β of the way to the general model's, per set, one panel per pair."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    exs, beta = data
    exs = [e for e in exs if any(r["beta"] == beta for r in e["rows"])]
    ncol = min(4, len(exs)); nrow = (len(exs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol + 0.6, 5.6 * nrow), squeeze=False)
    for ax in axes.ravel()[len(exs):]:
        ax.set_visible(False)
    marks = ["o", "s", "D"]; cols = [th["C"], th["L"], th["O"]]
    for ax, e in zip(axes.ravel(), exs):
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
                 x=0.0, y=1.0 + 0.04 / nrow, ha="left", fontsize=9.5)
    fig.tight_layout()
    return fig


def plot_combine(e, th):
    """Combined moves: ΔCER per combination, practice lines (filled/hollow by interval) next to disjoint confirmation lines."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sources = []
    for s in e["sets"]:
        base = s.replace("-conf", "")
        if base not in sources:
            sources.append(base)
    cols = {src: c for src, c in zip(sources, [th["C"], th["L"], th["O"]])}
    marks = {src: m for src, m in zip(sources, ["o", "s", "D"])}
    combos = [r["combo"] for r in e["rows"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(combos) + 2.2), sharey=True, sharex=True)
    for ax, conf in zip(axes, (False, True)):
        for r in e["rows"]:
            y = combos.index(r["combo"])
            for si, src in enumerate(sources):
                s = src.replace("-300", "-conf-300") if conf else src
                if s not in r["sets"]:
                    continue
                v = r["sets"][s]["vs_S"]; d = 100 * v["delta_cer"]; lo, hi = (100 * t for t in v["ci95"])
                yi = y + (si - 1) * 0.22; sig = lo > 0 or hi < 0
                ax.hlines(yi, lo, hi, color=cols[src], linewidth=1.3)
                ax.plot(d, yi, marker=marks[src], color=cols[src], markersize=6.5, markeredgewidth=1.3,
                        markerfacecolor=cols[src] if sig else th["bg"])
        ax.axvline(0, color=th["muted"], linewidth=0.9)
        moves = {r["combo"]: " + ".join(f"{b}{'' if beta == 1 else ' ½'}" for b, beta in r["moves"]) for r in e["rows"]}
        ax.set_yticks(range(len(combos))); ax.set_yticklabels([moves[c] for c in combos], fontsize=7.5); ax.invert_yaxis()
        ax.grid(axis="x", linewidth=0.5); ax.set_axisbelow(True)
        ax.set_xlabel("ΔCER against the unedited model (points; < 0 better)", fontsize=8)
        ax.set_title("confirmation lines (disjoint, chosen after)" if conf else "practice lines (moves chosen here)",
                     loc="left", fontsize=8.5)
    handles = [Line2D([], [], marker=marks[s], color=cols[s], linestyle="-", markersize=6) for s in sources]
    labels = [f"{s} (S {100 * e['c0'][s]['cer']:.1f} %)" + (" — own material" if i == 0 else "") for i, s in enumerate(sources)]
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Combined moves: {_mid(e['specific'])} ← {_mid(e['general'])} (block = transplanted, ½ = half-way); filled = paired 95 % interval excludes 0",
                 x=0.0, y=1.09, ha="left", fontsize=9.5)
    fig.tight_layout()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.familyviz")
    ap.add_argument("--out", default=str(OUT / "figures")); ap.add_argument("--format", default="svg")
    a = ap.parse_args(argv)
    out = Path(a.out)
    w = json.loads((OUT / "weights.json").read_text()); f = json.loads((OUT / "functional.json").read_text())
    learn_names(w)
    written = []
    written += render(plot_cka_profile, (f, w), out, "family-cka-profile", a.format)
    written += render(plot_cka_matrices, (f, w), out, "family-cka-matrices", a.format)
    written += render(plot_weight_drift, w, out, "family-weight-drift", a.format)
    written += render(plot_dead_filters, (f, w), out, "family-dead-filters", a.format)
    exs = [json.loads(p.read_text()) for p in sorted(OUT.glob("exchange-*.json"))]
    for beta in sorted({r["beta"] for e in exs for r in e["rows"]}):
        written += render(plot_exchange, (exs, beta), out, f"family-exchange-b{beta:g}", a.format)
    for p in sorted(OUT.glob("combine-*.json")):
        e = json.loads(p.read_text())
        written += render(plot_combine, e, out, f"family-{p.stem}", a.format)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
