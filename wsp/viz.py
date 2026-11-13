"""Figures for the research protocol: layer map, lineage drift, output rows, α-sweep, Cr255 padding.

    python -m wsp.viz all --out protocol/docs/figures [--models models/kraken] [--format svg|png]
    python -m wsp.viz arch|drift|rows|alpha|padding --out <dir>

The numbers come from the functions that wrote ``docs/readings/``:
``wsp.stats.describe`` / ``lineage_diff`` / ``kernel_row_l2`` on weights read
through ``KrakenSubject``; output rows are paired **by character**
(``wsp.align.codec_row_map``), never by index. A model that is not on this
machine falls back to the committed ``docs/readings/*.json`` where that is
enough (layer map, drift); the α-sweep figure needs no weights at all
(``docs/baselines/alpha-*.json``). No forward pass, no held-out data.

Every figure is written twice, ``<name>.svg`` on a light and ``<name>-dark.svg``
on a dark ground, for MkDocs Material's ``#only-light`` / ``#only-dark``.
Colours are Okabe–Ito (colour-blind safe); glyphs are drawn as paths so rare
characters survive any viewer. ``all`` also writes ``README.md`` next to the
figures: the command, every input file's sha256 against ``models/DIGESTS.md``,
and the data sources.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from wsp import align, diff as diffmod, stats

REPO = Path(__file__).resolve().parent.parent
READINGS = REPO / "docs" / "readings"

# parent → child pairs per family, by model id (file kraken-<id>.mlmodel)
FAMILY_BERN = [("cursive_inzigkofen", "bern-hertwig-v1"), ("bern-hertwig-v1", "bern-hertwig-v2"),
               ("bern-hertwig-v2", "bern-schlussredaktion-v1"), ("bern-hertwig-v2", "bern-combined-v1")]
FAMILY_XIX = [("german-xix-v1", "german-xix-blb-v1")]
# output-row figures: pair, practice set whose text counts as "how often the child sees this character"
ROW_PAIRS = [(("german-xix-v1", "german-xix-blb-v1"), "blb-train-300"),
             (("bern-hertwig-v2", "bern-combined-v1"), "bern-hertwig-train-300")]
ARCH_MODEL = "german-xix-blb-v1"
PADDING_MODEL = "collapse-probe-131-cr255"
ALPHA_JSON = REPO / "docs" / "baselines" / "alpha-xix-blb-v1.json"
RARE_BELOW = 10          # occurrences in the 300 practice lines

# Okabe–Ito; the dark theme swaps the deep blue for sky blue
LIGHT = {"bg": "#ffffff", "fg": "#1b1b1b", "muted": "#6e6e6e", "grid": "#e2e2e2",
         "C": "#0072B2", "L": "#E69F00", "O": "#009E73", "hi": "#D55E00", "alt": "#CC79A7", "soft": "#56B4E9"}
DARK = {"bg": "#16181d", "fg": "#e6e6e6", "muted": "#a3a3a3", "grid": "#33363d",
        "C": "#56B4E9", "L": "#E69F00", "O": "#009E73", "hi": "#E8743B", "alt": "#CC79A7", "soft": "#0072B2"}
THEMES = {"": LIGHT, "-dark": DARK}

_LSTM_ORDER = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"]
_OFFSETS = [(10, 10), (12, -14), (-20, 10), (-22, -12), (16, 0), (-26, 0)]


# -- small helpers ---------------------------------------------------------------

def tensor_key(name: str) -> tuple:
    """state_dict order: block number, forward before reverse, weight before bias (ih, hh, b_ih, b_hh for LSTMs)."""
    blk, *rest = name.split(".")
    leaf = rest[-1] if rest else ""
    rev = leaf.endswith("_reverse")
    base = leaf[: -len("_reverse")] if rev else leaf
    rank = _LSTM_ORDER.index(base) if base in _LSTM_ORDER else (0 if base == "weight" else 1)
    return int(blk.split("_")[1]), rev, rank


def short_name(name: str) -> str:
    """``L_12.layer.weight_hh_l0_reverse`` → ``L_12 hh←``; ``C_3.co.bias`` → ``C_3 b``."""
    blk, *rest = name.split(".")
    leaf = rest[-1]
    rev = leaf.endswith("_reverse")
    base = leaf[: -len("_reverse")] if rev else leaf
    lab = {"weight": "w", "bias": "b", "weight_ih_l0": "ih", "weight_hh_l0": "hh",
           "bias_ih_l0": "b_ih", "bias_hh_l0": "b_hh"}.get(base, base)
    return f"{blk} {lab}" + ("←" if rev else ("→" if base in _LSTM_ORDER else ""))


def glyph(ch: str) -> str:
    """A printable label for a codec entry: combining marks on a dotted circle, space as ␣."""
    if ch == "<blank>":
        return "blank"
    if ch == " ":
        return "␣"
    if all(unicodedata.combining(c) for c in ch):
        return "◌" + ch
    return ch


def model_path(models: Path, mid: str) -> Path:
    return Path(models) / f"kraken-{mid}.mlmodel"


def known_digests(path: Path = REPO / "models" / "DIGESTS.md") -> dict[str, str]:
    """id → sha256 from the tables in models/DIGESTS.md."""
    out = {}
    if Path(path).exists():
        for m in re.finditer(r"^\| (kraken-[^ |]+) \| ([0-9a-f]{64}) \|", Path(path).read_text(), re.M):
            out[m.group(1)] = m.group(2)
    return out


# -- data: weights or committed readings ------------------------------------------------

class Inputs:
    """Loads each model once and records which files a figure read."""

    def __init__(self, models: Path):
        self.models = Path(models)
        self._subjects: dict[str, object] = {}
        self.used: dict[str, str] = {}          # id → sha256 of the file read
        self.sources: set[str] = set()          # committed JSON / practice files read

    def has(self, mid: str) -> bool:
        return model_path(self.models, mid).exists()

    def subject(self, mid: str):
        if mid not in self._subjects:
            from wsp.kraken_subject import KrakenSubject
            from wsp.data import sha256_file
            p = model_path(self.models, mid)
            self._subjects[mid] = KrakenSubject(p)
            self.used[f"kraken-{mid}"] = sha256_file(p)
        return self._subjects[mid]

    def reading(self, name: str) -> dict:
        p = READINGS / f"{name}.json"
        self.sources.add(str(p.relative_to(REPO)))
        return json.loads(p.read_text())

    def describe(self, mid: str) -> dict:
        if self.has(mid):
            return stats.describe(self.subject(mid))
        return self.reading(f"describe_{mid}")

    def drift(self, parent: str, child: str) -> dict:
        """``stats.lineage_diff`` plus ``aligned_output``: the output layer over the rows both codecs share."""
        if self.has(parent) and self.has(child):
            p, c = self.subject(parent), self.subject(child)
            d = stats.lineage_diff(p, c)
            d["aligned_output"] = aligned_output_drift(p, c)
            return d
        return self.reading(f"diff_{parent}__{child}")

    def practice_counts(self, name: str) -> Counter:
        from wsp.data import load
        p = REPO / "data" / "practice" / f"{name}.jsonl"
        self.sources.add(str(p.relative_to(REPO)))
        return Counter("".join(x["text"] for x in load(p)))


def aligned_output_drift(parent, child) -> dict:
    """``diff.compare`` on the output layer restricted to shared characters, rows paired by character."""
    a, b = parent.tensors(), child.tensors()
    wname = next(n for n in a if n.startswith("O_") and n.endswith("lin.weight"))
    bname = wname.replace("weight", "bias")
    rm = align.codec_row_map(parent.codec_map(), child.codec_map()); rm[0] = 0     # blank ↔ blank
    ia = sorted(rm); ib = [rm[i] for i in ia]
    out = diffmod.compare({wname: a[wname][ia], bname: a[bname][ia]}, {wname: b[wname][ib], bname: b[bname][ib]}, row_top=0)
    for e in out.values():
        e["rows"] = len(ia)
    return out


def drift_records(d: dict) -> list[dict]:
    """One record per tensor in network order: rel L2, cosine, and how it was measured."""
    recs = []
    al = d.get("aligned_output", {})
    for name in sorted(d["tensors"], key=tensor_key):
        e = d["tensors"][name]
        r = {"name": name, "kind": name[0], "rel": float("nan"), "cos": float("nan"), "how": "tensor"}
        if name in al:
            r.update(rel=al[name]["rel_l2"], cos=al[name]["cosine"], how="aligned")
        elif "rel_l2" in e:
            zero = e["l2"] == 0 and e["rel_l2"] == 0 and e["cosine"] == 1 and e.get("changed", 1) == 0
            r.update(rel=e["rel_l2"], cos=e["cosine"], how="zero" if zero and "bias_ih" in name else "tensor")
        elif name.endswith("lin.weight") and "output_rows" in d:     # readings JSON, codec grew
            r.update(rel=d["output_rows"]["mean_rel"], how="rowmean")
        recs.append(r)
    return recs


def row_table(parent_desc: dict, child_desc: dict, counts: Counter | None = None) -> list[dict]:
    """Output rows of parent and child joined by character (the blank by position 0); child-only rows have no parent."""
    pr = {r["char"]: r for r in parent_desc["output_layer"]["rows"]}
    out = []
    for r in child_desc["output_layer"]["rows"]:
        p = pr.get(r["char"])
        out.append({"char": r["char"], "norm_child": r["norm"], "bias_child": r["bias"],
                    "norm_parent": p["norm"] if p else None, "bias_parent": p["bias"] if p else None,
                    "bias_delta": (r["bias"] - p["bias"]) if p else None,
                    "count": None if counts is None or r["char"] == "<blank>" else counts.get(r["char"], 0)})
    return out


# -- rendering --------------------------------------------------------------------

def _rc(th: dict) -> dict:
    return {"figure.facecolor": th["bg"], "axes.facecolor": th["bg"], "savefig.facecolor": th["bg"],
            "text.color": th["fg"], "axes.labelcolor": th["fg"], "axes.edgecolor": th["muted"],
            "xtick.color": th["fg"], "ytick.color": th["fg"], "grid.color": th["grid"],
            "legend.facecolor": th["bg"], "legend.edgecolor": th["grid"], "legend.labelcolor": th["fg"],
            "font.family": ["DejaVu Sans", "Noto Sans", "Noto Sans Symbols", "Noto Sans Symbols2"],
            "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
            "svg.fonttype": "path", "svg.hashsalt": "wsp.viz"}


def render(plot, data, out: Path, name: str, fmt: str = "svg") -> list[Path]:
    """Draw ``plot(data, theme)`` once per theme and save it; returns the written paths."""
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)     # bold falls back to regular in the symbol fonts
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, th in THEMES.items():
        with plt.rc_context(_rc(th)):
            fig = plot(data, th)
            p = out / f"{name}{suffix}.{fmt}"
            meta = {"Date": None, "Creator": "wsp.viz"} if fmt == "svg" else {"Software": "wsp.viz"}
            fig.savefig(p, format=fmt, dpi=200, bbox_inches="tight", metadata=meta)
            plt.close(fig)
            paths.append(p)
    return paths


# -- figures ------------------------------------------------------------------------

def plot_architecture(desc: dict, th: dict):
    """The VGSL stack as a strip, width ∝ parameters; the conv stack magnified underneath."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, ConnectionPatch
    tensors = sorted(desc["tensors"], key=lambda r: tensor_key(r["name"]))
    total = sum(r["numel"] for r in tensors)
    blocks: dict[str, list[dict]] = {}
    for r in tensors:
        blocks.setdefault(r["name"].split(".")[0], []).append(r)
    conv = [b for b in blocks if b.startswith("C_")]
    conv_total = sum(r["numel"] for b in conv for r in blocks[b])
    out = desc.get("output_layer") or {}

    fig, (ax, az) = plt.subplots(2, 1, figsize=(11, 3.4), gridspec_kw={"height_ratios": [1, 1], "hspace": 1.25})

    def strip(axis, names, scale, label_min):
        x = 0.0
        spans = {}
        for b in names:
            x0 = x
            for i, r in enumerate(blocks[b]):
                w = r["numel"] * scale
                axis.add_patch(Rectangle((x, 0), w, 1, facecolor=th[b[0]], alpha=1.0 if r["name"].split(".")[-1].startswith("weight") else 0.55,
                                         edgecolor=th["bg"], linewidth=0.6))
                if w > label_min:
                    axis.text(x + w / 2, 0.5, short_name(r["name"]).split(" ")[1], ha="center", va="center",
                              fontsize=7.5, color=th["bg"])
                x += w
            spans[b] = (x0, x)
        axis.set_xlim(0, x); axis.set_ylim(-0.05, 1.05)
        axis.set_yticks([])
        for s in ("left", "bottom"):
            axis.spines[s].set_visible(False)
        return spans

    spans = strip(ax, list(blocks), 1 / 1e6, 0.12)
    for b, (x0, x1) in spans.items():
        n = sum(r["numel"] for r in blocks[b])
        if b.startswith("C_"):
            continue
        ax.text((x0 + x1) / 2, 1.12, f"{b}  {n / 1e6:.2f} M" if n >= 1e6 else f"{b}  {n / 1e3:.0f} k",
                ha="center", va="bottom", fontsize=8.5)
    c0, c1 = spans[conv[0]][0], spans[conv[-1]][1]
    ax.text(c0, 1.12, f"{conv[0]}–{conv[-1]}", ha="left", va="bottom", fontsize=8.5)
    ax.set_xticks(np.arange(0, total / 1e6 + 1e-9, 0.5))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("parameters (millions), cumulative in network order")
    ax.spines["bottom"].set_visible(True)
    ax.set_title(f"{len(tensors)} tensors, {total:,} parameters — width ∝ parameter count "
                 f"(dark: weights, light: biases)", loc="left", pad=24)

    zspans = strip(az, conv, 1 / 1e3, 1.5)
    for b, (x0, x1) in zspans.items():
        n = sum(r["numel"] for r in blocks[b])
        w0 = next(r for r in blocks[b] if r["name"].endswith("weight"))["shape"]
        wide = x1 - x0 > 4
        az.text((x0 + x1) / 2 if wide else x0, 1.12 if wide else 1.75,
                f"{b}  {n:,}\n{w0[3]}×{w0[2]}×{w0[0]}" if wide else f"{b}  {n:,}",
                ha="center" if wide else "left", va="bottom", fontsize=8)
    az.set_xticks(np.arange(0, conv_total / 1e3 + 1e-9, 10))
    az.tick_params(axis="x", labelsize=8)
    az.spines["bottom"].set_visible(True)
    az.set_xlabel(f"conv stack magnified ×{total / conv_total:.0f} — parameters (thousands); {conv_total:,} = {100 * conv_total / total:.1f} % of the network")
    for xa, xb in ((c0, 0), (c1, zspans[conv[-1]][1])):
        fig.add_artist(ConnectionPatch((xa, 0), (xb, 1.0), "data", "data", axesA=ax, axesB=az,
                                       color=th["muted"], linewidth=0.7, linestyle=":"))
    legend = [Rectangle((0, 0), 1, 1, color=th[k]) for k in "CLO"]
    labels = ["convolution (3×13 / 3×9 kernels, max-pool 2×2)", "bidirectional LSTM, 200 units per direction",
              f"output: {out.get('classes', '?')} classes (CTC blank + codec) × {out.get('hidden', 400)}"]
    fig.legend(legend, labels, loc="lower left", bbox_to_anchor=(0.0, -0.13), ncol=3, frameon=False, fontsize=8)
    fig.text(0, -0.2, "VGSL [1,120,0,1 Cr3,13,32 Do Mp2,2 Cr3,13,32 Do Mp2,2 Cr3,9,64 Do Mp2,2 Cr3,9,64 Do "
             "S1(1x0)1,3 Lbx200 Do Lbx200 Do Lbx200 Do O1c<N>]; → forward, ← reverse direction",
             fontsize=7.5, color=th["muted"], family="DejaVu Sans Mono")
    return fig


def plot_drift(pairs: list[tuple[str, dict]], th: dict):
    """Relative L2 and cosine per tensor, one panel per parent→child pair, shared scales."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    recs = [(t, drift_records(d)) for t, d in pairs]
    names = [r["name"] for r in recs[0][1]]
    x = np.arange(len(names))
    rmax = max(np.nanmax([r["rel"] for r in rs]) for _, rs in recs)
    cmin = min(np.nanmin([r["cos"] for r in rs if r["how"] != "zero"] or [1]) for _, rs in recs)
    n = len(recs)
    H = 2.6 * n + 1.4
    fig, grid = plt.subplots(3 * n - 1, 1, figsize=(11, H), sharex=True,
                             gridspec_kw={"height_ratios": ([2, 1, 0.55] * n)[:-1], "hspace": 0.12})
    for spacer in grid[2::3]:
        spacer.set_visible(False)
    axes = [a for i, a in enumerate(grid) if i % 3 != 2]
    fig.subplots_adjust(top=1 - 1.0 / H)
    for k, (title, rs) in enumerate(recs):
        ar, ac = axes[2 * k], axes[2 * k + 1]
        for i, r in enumerate(rs):
            col = th[r["kind"]]
            if r["how"] == "zero":
                ar.plot(i, 0, marker="x", color=th["muted"], markersize=4)
                ac.plot(i, 1, marker="x", color=th["muted"], markersize=4)
                continue
            hatch = "////" if r["how"] in ("aligned", "rowmean") else None
            ar.bar(i, r["rel"], color=col if not hatch else th["bg"], edgecolor=col, hatch=hatch, linewidth=0.8, width=0.75)
            if not np.isnan(r["cos"]):
                ac.vlines(i, r["cos"], 1, color=col, linewidth=0.8)
                ac.plot(i, r["cos"], marker="o", color=col, markersize=3.2)
        ar.set_ylim(0, rmax * 1.12)
        ar.set_ylabel("‖Δ‖ / ‖parent‖", fontsize=8)
        ar.grid(axis="y", linewidth=0.5); ar.set_axisbelow(True)
        ar.set_title(title, loc="left", fontsize=9.5, pad=3)
        ac.set_ylim(cmin - (1 - cmin) * 0.1, 1.0 + (1 - cmin) * 0.04)
        ac.set_ylabel("cosine", fontsize=8)
        ac.grid(axis="y", linewidth=0.5); ac.set_axisbelow(True)
        ac.tick_params(axis="y", labelsize=7); ar.tick_params(axis="y", labelsize=7)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([short_name(nm) for nm in names], rotation=90, fontsize=7)
    handles = [Patch(color=th[k]) for k in "CLO"] + [
        Patch(facecolor=th["bg"], edgecolor=th["O"], hatch="////"),
        Line2D([], [], marker="x", color=th["muted"], linestyle="")]
    labels = ["convolution", "BiLSTM", "output layer", "output layer, rows paired by character (shared characters only)",
              "all-zero in both (kraken trains only b_hh)"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 1 - 0.3 / H), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Parent → child drift per tensor: relative L2 of the difference (bars) and cosine parent·child (dots); "
                 "→ forward, ← reverse LSTM direction", x=0.06, ha="left", y=1 - 0.05 / H, fontsize=10)
    return fig


def plot_rows(data: dict, th: dict):
    """Output rows paired by character: row norm parent→child, and bias Δ against frequency in the practice text."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows, title = data["rows"], data["title"]
    shared = [r for r in rows if r["norm_parent"] is not None and r["char"] != "<blank>"]
    new = [r for r in rows if r["norm_parent"] is None]
    blank = next(r for r in rows if r["char"] == "<blank>")
    has_counts = any(r["count"] is not None for r in shared)

    def cls(r):
        if not has_counts:
            return ("shared character", th["C"])
        if r["count"] == 0:
            return ("absent from practice text", th["alt"])
        if r["count"] < RARE_BELOW:
            return (f"rare (1–{RARE_BELOW - 1}×)", th["L"])
        return (f"frequent (≥ {RARE_BELOW}×)", th["C"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.22})
    hi = max(max(r["norm_child"] for r in rows), max(r["norm_parent"] for r in shared)) * 1.05
    a.plot([0, hi], [0, hi], color=th["muted"], linewidth=0.8, linestyle="--")
    a.text(hi * 0.97, hi * 0.04, "dashed: unchanged (child = parent)", ha="right", color=th["muted"], fontsize=7.5)
    for r in shared:
        lab, col = cls(r)
        a.scatter(r["norm_parent"], r["norm_child"], s=14, color=col, alpha=0.85, linewidths=0)
    a.scatter([0] * len(new), [r["norm_child"] for r in new], s=16, facecolors="none", edgecolors=th["muted"], linewidths=0.8)
    a.scatter(blank["norm_parent"], blank["norm_child"], s=70, marker="D", color=th["hi"], zorder=5)
    a.annotate("blank", (blank["norm_parent"], blank["norm_child"]), xytext=(8, -12), textcoords="offset points",
               color=th["hi"], fontsize=9, fontweight="bold")
    movers = sorted(shared, key=lambda r: -abs(r["norm_child"] - r["norm_parent"]) / r["norm_parent"])[:6]
    for k, r in enumerate(movers):
        a.annotate(glyph(r["char"]), (r["norm_parent"], r["norm_child"]), xytext=_OFFSETS[k % len(_OFFSETS)],
                   textcoords="offset points", fontsize=10, arrowprops={"arrowstyle": "-", "color": th["muted"], "lw": 0.5})
    a.set_xlim(-hi * 0.03, hi); a.set_ylim(0, hi)
    a.set_xlabel("row norm in parent  (x = 0: row new in child)")
    a.set_ylabel("row norm in child")
    a.grid(linewidth=0.5); a.set_axisbelow(True)
    a.set_title(f"row norm, {len(shared) + 1} shared rows + {len(new)} new", loc="left")

    xs = [(r["count"] if has_counts else abs(r["norm_child"] - r["norm_parent"]) / r["norm_parent"]) for r in shared]
    for r, xv in zip(shared, xs):
        lab, col = cls(r)
        b.scatter(xv, r["bias_delta"], s=14, color=col, alpha=0.85, linewidths=0)
    b.axhline(0, color=th["muted"], linewidth=0.8)
    bx = (max(xs) * 1.6 if has_counts else 0)
    if has_counts:
        b.set_xscale("symlog", linthresh=1)
        b.set_xlabel("occurrences in the practice text (300 training-side lines), symlog")
        b.scatter(bx, blank["bias_delta"], s=70, marker="D", color=th["hi"], zorder=5)
        b.annotate("blank\n(not in text)", (bx, blank["bias_delta"]), xytext=(-10, 8), textcoords="offset points",
                   color=th["hi"], fontsize=8.5, fontweight="bold", ha="right")
    else:
        b.set_xlabel("relative row change ‖Δrow‖ / ‖parent row‖")
    seen_in_text = [(r, xv) for r, xv in zip(shared, xs) if not has_counts or r["count"] > 0]
    for k, (r, xv) in enumerate(sorted(seen_in_text, key=lambda t: t[0]["bias_delta"])[:5]):
        b.annotate(glyph(r["char"]), (xv, r["bias_delta"]), xytext=_OFFSETS[k % len(_OFFSETS)], textcoords="offset points",
                   fontsize=10, arrowprops={"arrowstyle": "-", "color": th["muted"], "lw": 0.5})
    b.set_ylabel("bias Δ (child − parent)")
    b.grid(linewidth=0.5); b.set_axisbelow(True)
    b.set_title("bias change per character", loc="left")

    seen = {}
    for r in shared:
        lab, col = cls(r)
        seen.setdefault(lab, col)
    handles = [Line2D([], [], marker="o", linestyle="", color=c, markersize=5) for c in seen.values()]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=th["muted"], markersize=5),
                Line2D([], [], marker="D", linestyle="", color=th["hi"], markersize=6)]
    fig.legend(handles, list(seen) + ["new in child (no parent row)", "CTC blank (row 0)"],
               loc="lower left", bbox_to_anchor=(0.0, -0.06), ncol=5, frameon=False, fontsize=8)
    fig.suptitle(f"{title} — output-layer rows paired by character, never by index; "
                 f"labelled: largest relative norm change (left), largest bias drop among characters in the text (right)",
                 x=0.0, ha="left", y=1.0, fontsize=10)
    return fig


def plot_alpha(sweep: dict, th: dict):
    """C3: ΔCER against the unedited child per block, α = 0.5 and 1.5, paired bootstrap 95 % intervals."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    blocks = sweep["blocks"]
    alphas = sweep["alphas"]
    style = {alphas[0]: ("o", th["C"]), alphas[-1]: ("s", th["L"])} if len(alphas) > 1 else {alphas[0]: ("o", th["C"])}
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(blocks) + 1.6))
    offs = np.linspace(-0.17, 0.17, len(alphas)) if len(alphas) > 1 else [0.0]
    rel = {}
    for r in sweep["rows"]:
        y = blocks.index(r["block"]) + offs[alphas.index(r["alpha"])]
        d, (lo, hi) = 100 * r["vs_c0"]["delta_cer"], (100 * v for v in r["vs_c0"]["ci95"])
        m, col = style[r["alpha"]]
        sig = lo > 0 or hi < 0
        ax.hlines(y, lo, hi, color=col, linewidth=1.6)
        ax.plot(d, y, marker=m, markersize=6.5, color=col, markerfacecolor=col if sig else th["bg"], markeredgewidth=1.4)
        rel[r["block"]] = r["block_rel_l2"]
    ax.axvline(0, color=th["muted"], linewidth=0.9)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels(blocks)
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.5); ax.set_axisbelow(True)
    ax.set_xlabel("ΔCER against the unedited child (percentage points; < 0 is better)")
    x1 = ax.get_xlim()[1]
    ax.set_xlim(ax.get_xlim()[0], x1 + (x1 - ax.get_xlim()[0]) * 0.18)
    for b in blocks:
        if b in rel:
            ax.text(1.0, blocks.index(b), f"‖Δ‖/‖p‖ {rel[b]:.3f}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=7.5, color=th["muted"])
    handles = [Line2D([], [], marker=style[a][0], color=style[a][1], linestyle="-", markersize=6) for a in alphas]
    handles.append(Line2D([], [], marker="o", color=th["muted"], markerfacecolor=th["bg"], linestyle="", markersize=6))
    labels = [f"α = {a}" + (" (halfway back to parent)" if a == 0.5 else " (extrapolated past child)" if a == 1.5 else "")
              for a in alphas] + ["hollow: 95 % interval includes 0"]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8)
    c0 = sweep["c0"]["cer"]; pc = (sweep.get("parent_c1") or {}).get("cer")
    ax.set_title(f"C3 per-block α-sweep: {sweep['child'].removeprefix('kraken-')} = "
                 f"{sweep['parent'].removeprefix('kraken-')} + α·Δ on one block at a time\n"
                 f"{sweep['lines']} practice lines ({sweep['practice']}); child CER {100 * c0:.2f} %"
                 + (f", parent {100 * pc:.2f} %" if pc else "")
                 + f"; paired bootstrap, {sweep['rows'][0]['vs_c0']['resamples']} resamples",
                 loc="left", pad=26, fontsize=9.5)
    return fig


def plot_padding(data: dict, th: dict):
    """Kernel-row L2 of the Cr255 conv: one live row among padding rows."""
    import matplotlib.pyplot as plt
    rows, keep = np.asarray(data["rows"]), data["keep"]
    other = np.delete(rows, keep)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axhspan(other.mean() - other.std(), other.mean() + other.std(), color=th["soft"], alpha=0.18, linewidth=0)
    ax.axhline(other.mean(), color=th["soft"], linewidth=0.8)
    ax.plot(np.arange(len(rows)), rows, color=th["C"], linewidth=0.9, marker=".", markersize=2.5)
    ax.plot(keep, rows[keep], marker="o", markersize=7, color=th["hi"], zorder=5)
    ax.annotate(f"kept row {keep}, the only row that receives gradient\nL2 {rows[keep]:.2f} "
                f"(+{100 * (rows[keep] / other.mean() - 1):.1f} % over the padding mean)",
                (keep, rows[keep]), xytext=(14, -4), textcoords="offset points", color=th["hi"], fontsize=8.5, va="top")
    ax.text(0, rows.max(), f"band: the other {len(other)} rows, {other.mean():.2f} ± {other.std():.3f} (mean ± sd) — "
            f"still at their initialisation norm; y axis does not start at 0",
            ha="left", va="top", fontsize=8, color=th["muted"])
    ax.set_xlim(-2, len(rows) + 1)
    ax.set_xlabel("kernel row (height index of the 255-row kernel)")
    ax.set_ylabel("row L2 over out × in")
    ax.grid(linewidth=0.5); ax.set_axisbelow(True)
    s = data["shape"]
    ax.set_title(f"{data['model']}: {data['tensor']} {s[0]}×{s[1]}×{s[2]}×{s[3]} — "
                 f"{(len(rows) - 1) * s[0] * s[1] * s[3]:,} weights only ever multiply padding", loc="left")
    return fig


# -- figure builders: data → files ----------------------------------------------------------

def _pair_title(parent: str, child: str, d: dict) -> str:
    c = d["codec"]
    grew = f"codec {c['parent']} → {c['child']}" + (f" (+{len(c['added'])})" if c["added"] else "")
    return f"{parent} → {child}   ({grew})"


def figure_architecture(inp: Inputs, out: Path, fmt: str) -> list[Path]:
    return render(plot_architecture, inp.describe(ARCH_MODEL), out, "architecture", fmt)


def figure_drift(inp: Inputs, out: Path, fmt: str) -> list[Path]:
    paths = []
    for name, fam in (("drift-bern", FAMILY_BERN), ("drift-xix", FAMILY_XIX)):
        pairs = [(_pair_title(p, c, d), d) for p, c in fam for d in [inp.drift(p, c)]]
        paths += render(plot_drift, pairs, out, name, fmt)
    return paths


def figure_rows(inp: Inputs, out: Path, fmt: str) -> list[Path]:
    paths = []
    for (p, c), practice in ROW_PAIRS:
        if not (inp.has(p) and inp.has(c)):
            print(f"skip rows {p}→{c}: weights not here (readings JSON hold only the top rows)")
            continue
        counts = inp.practice_counts(practice)
        data = {"title": f"{p} → {c}", "rows": row_table(inp.describe(p), inp.describe(c), counts)}
        paths += render(plot_rows, data, out, f"output-rows-{c}", fmt)
    return paths


def figure_alpha(inp: Inputs, out: Path, fmt: str, path: Path = ALPHA_JSON) -> list[Path]:
    inp.sources.add(str(Path(path).resolve().relative_to(REPO)))
    return render(plot_alpha, json.loads(Path(path).read_text()), out, "alpha-sweep-xix-blb-v1", fmt)


def figure_padding(inp: Inputs, out: Path, fmt: str, keep: int = 127) -> list[Path]:
    if not inp.has(PADDING_MODEL):
        print(f"skip padding: kraken-{PADDING_MODEL} not here")
        return []
    t = inp.subject(PADDING_MODEL).tensors()
    name = stats.padding_tensor(t)
    data = {"model": f"kraken-{PADDING_MODEL}", "tensor": name, "shape": list(t[name].shape),
            "rows": stats.kernel_row_l2(t[name]).tolist(), "keep": keep}
    return render(plot_padding, data, out, "padding-cr255", fmt)


# -- zoo survey figures (docs/zoo/survey.json only; no weights are read here) ---------------------------

ZOO_JSON = REPO / "docs" / "zoo" / "survey.json"


def _zoo_short(mid: str) -> str:
    return mid.removeprefix("kraken-")


def plot_zoo_blank(data: dict, th: dict):
    """Blank-row bias of every surveyed model: raw, and relative to the mean of the model's other biases."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ms = sorted(data["models"].values(), key=lambda m: m["output"]["blank_bias_rel"])
    y = np.arange(len(ms))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 0.21 * len(ms) + 1.6), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, key, lab in ((a, "blank_bias", "blank bias (raw)"), (b, "blank_bias_rel", "blank bias − mean of the other biases")):
        for i, m in enumerate(ms):
            col = th["hi"] if m["origin"] == "in-house" else th["C"]
            ax.scatter(m["output"][key], i, s=22, color=col, marker="D" if m["origin"] == "in-house" else "o", linewidths=0)
        ax.axvline(0, color=th["muted"], linewidth=0.8)
        ax.set_xlabel(lab); ax.grid(axis="x", linewidth=0.5); ax.set_axisbelow(True)
    a.set_yticks(y); a.set_yticklabels([_zoo_short(m["id"]) for m in ms], fontsize=7)
    a.set_title("raw: the softmax ignores a constant added to all biases", loc="left", fontsize=9)
    b.set_title("relative: the quantity the softmax can see", loc="left", fontsize=9)
    fig.legend([Line2D([], [], marker="o", linestyle="", color=th["C"]), Line2D([], [], marker="D", linestyle="", color=th["hi"])],
               ["model zoo (Zenodo)", "in-house"], loc="lower right", bbox_to_anchor=(0.99, 0.0), frameon=False, fontsize=8)
    fig.suptitle(f"Blank-row bias of {len(ms)} kraken models, sorted by the relative value", x=0.0, ha="left", fontsize=10)
    return fig


def zoo_order(entry: dict) -> list[str]:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    ids = entry["ids"]
    n = len(ids); D = np.zeros((n, n))
    for p, v in entry["pairs"].items():
        a, b = p.split("|"); D[ids.index(a), ids.index(b)] = D[ids.index(b), ids.index(a)] = v["rel"]
    return [ids[i] for i in leaves_list(linkage(squareform(D, checks=False), "average"))]


def plot_zoo_distance(data: dict, th: dict):
    """Symmetric relative L2 between all non-output weights, standard stack; clustered order."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    entry = max(data["distances"].values(), key=lambda e: len(e["ids"]))
    order = zoo_order(entry); n = len(order)
    D = np.full((n, n), np.nan)
    for p, v in entry["pairs"].items():
        a, b = p.split("|"); i, j = order.index(a), order.index(b); D[i, j] = D[j, i] = v["rel"]
    fig, ax = plt.subplots(figsize=(9.2, 8.4))
    cm = LinearSegmentedColormap.from_list("w", [th["O"], th["bg"], th["muted"]])
    im = ax.imshow(D, cmap=cm, vmin=0, vmax=1.5)
    ax.set_xticks(range(n)); ax.set_xticklabels([_zoo_short(i) for i in order], rotation=90, fontsize=6.5)
    ax.set_yticks(range(n)); ax.set_yticklabels([_zoo_short(i) for i in order], fontsize=6.5)
    for k, i in enumerate(order):
        if data["models"][i]["origin"] == "in-house":
            for t in (ax.get_xticklabels()[k], ax.get_yticklabels()[k]):
                t.set_color(th["hi"])
    for i in range(n):
        for j in range(n):
            if i != j and D[i, j] < 1.0:
                ax.text(j, i, f"{D[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5, color=th["fg"])
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("‖a − b‖ / mean(‖a‖, ‖b‖), non-output weights", fontsize=8)
    ax.set_title(f"{n} models with the standard stack: unrelated pairs sit near √2 ≈ 1.41; labelled cells < 1.0 "
                 "(in-house names in orange)", loc="left", fontsize=9)
    return fig


def plot_zoo_pairs(data: dict, th: dict):
    """Observation 3 and the blank bias on the tested parent→child pairs."""
    import matplotlib.pyplot as plt
    rows = [p for p in data["documented_pairs"] if p["status"] == "tested" and p["reading"]["blocks"]["conv_w"] < 1.0]
    lab = [f"{_zoo_short(p['parent'])} → {_zoo_short(p['child'])}" for p in rows]
    y = np.arange(len(rows))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 0.55 * len(rows) + 1.8), sharey=True, gridspec_kw={"wspace": 0.05, "width_ratios": [1.25, 1]})
    h = 0.26
    for k, (key, col, name) in enumerate((("conv_w", th["C"], "conv stack"), ("lstm_w", th["L"], "BiLSTM stack"), ("out_rows", th["O"], "output rows (aligned)"))):
        a.barh(y + (1 - k) * h, [p["reading"]["blocks"][key] for p in rows], height=h, color=col, label=name)
    a.set_yticks(y); a.set_yticklabels(lab, fontsize=7.5)
    a.set_xlabel("‖child − parent‖ / ‖parent‖, weights only (pooled over the stack)")
    a.legend(frameon=False, fontsize=8, loc="lower right"); a.grid(axis="x", linewidth=0.5); a.set_axisbelow(True)
    a.set_title("which stack moved", loc="left", fontsize=9)
    for yi, p in zip(y, rows):
        k = p["reading"]["blank"]
        b.scatter(k["delta_raw"], yi + 0.07, s=30, color=th["muted"], label="raw Δ bias[blank]")
        b.scatter(k["delta_rel"], yi - 0.07, s=30, color=th["hi"], marker="D", label="Δ (blank − mean of other biases)")
    h_, l_ = b.get_legend_handles_labels(); u = dict(zip(l_, h_)); b.legend(u.values(), u.keys(), frameon=False, fontsize=8, loc="lower right")
    b.axvline(0, color=th["muted"], linewidth=0.8); b.grid(axis="x", linewidth=0.5); b.set_axisbelow(True)
    b.set_xlabel("change of the blank bias, child − parent"); b.set_title("blank bias", loc="left", fontsize=9)
    fig.suptitle("Parent → child pairs of the zoo and the in-house family (pairs whose weights are related, distance < 1.0)",
                 x=0.0, ha="left", fontsize=10)
    return fig


LINEAGE_JSON = REPO / "docs" / "zoo" / "lineage.json"

# node positions of the lineage drawing (x = generation, y = row); a model not listed here has no relation to any other surveyed model
_LIN_NODES = {
    "kraken-catmus_medieval": (0, 8.0), "kraken-bastarda_inzigkofen": (1.6, 9.2), "kraken-cursive_inzigkofen": (1.6, 8.0),
    "bern-hertwig-v1": (3.2, 8.9), "bern-hertwig-v2": (3.2, 7.6), "bern-schlussredaktion-v1": (4.8, 8.2), "bern-combined-v1": (4.8, 7.0),
    "kraken-bifrost_old_norse": (1.6, 6.6), "kraken-catmus_gothic_print": (1.6, 5.6), "kraken-old_norse_am305": (1.6, 4.5),
    "kraken-tridis_v2": (0, 3.0), "kraken-textualis_inzigkofen": (1.6, 3.0),
    "kraken-catmus_print_large": (0, 1.2), "kraken-french_vietnamese": (1.6, 1.7), "kraken-latin_incunabula": (1.6, 0.7),
    "kraken-mccatmus": (0, -0.6), "kraken-prima": (1.6, -0.6),
    "kraken-bohemian_19th": (0, -2.0), "german-xix-v1": (1.6, -2.0), "german-xix-blb-v1": (3.2, -2.0),
    "kraken-openiti_arabic_print": (0, -3.9), "kraken-openiti_persian_print": (1.6, -3.4), "kraken-openiti_ottoman_print": (1.6, -4.4),
    "kraken-hebrew_sephardi": (3.6, -3.3), "kraken-glagolitic_print": (5.6, -3.3), "kraken-samaritan": (4.6, -4.4),
}
# (parent, child, kind, label override); kind: ok = documented and the weights agree, bad = documented and the weights do not support it,
# weights = suggested by weights and codec only, cont = continuation read in READINGS
_LIN_EDGES = [
    ("kraken-catmus_medieval", "kraken-bastarda_inzigkofen", "ok"), ("kraken-catmus_medieval", "kraken-cursive_inzigkofen", "ok"),
    ("kraken-cursive_inzigkofen", "bern-hertwig-v1", "ok"), ("kraken-cursive_inzigkofen", "bern-hertwig-v2", "ok"),
    ("bern-hertwig-v1", "bern-hertwig-v2", "cont"),
    ("bern-hertwig-v2", "bern-schlussredaktion-v1", "ok"), ("bern-hertwig-v2", "bern-combined-v1", "ok"),
    ("kraken-catmus_medieval", "kraken-bifrost_old_norse", "bad"), ("kraken-catmus_medieval", "kraken-catmus_gothic_print", "arch"),
    ("kraken-catmus_medieval", "kraken-old_norse_am305", "weights"),
    ("kraken-tridis_v2", "kraken-textualis_inzigkofen", "ok"),
    ("kraken-catmus_print_large", "kraken-french_vietnamese", "ok"), ("kraken-catmus_print_large", "kraken-latin_incunabula", "ok"),
    ("kraken-mccatmus", "kraken-prima", "weights"),
    ("kraken-bohemian_19th", "german-xix-v1", "weights"), ("german-xix-v1", "german-xix-blb-v1", "ok"),
    ("kraken-openiti_arabic_print", "kraken-openiti_persian_print", "ok"), ("kraken-openiti_arabic_print", "kraken-openiti_ottoman_print", "ok"),
    ("kraken-hebrew_sephardi", "kraken-glagolitic_print", "pair"), ("kraken-hebrew_sephardi", "kraken-samaritan", "pair"),
    ("kraken-glagolitic_print", "kraken-samaritan", "pair"),
]
# documented parents that are not among the surveyed files: (label, child, why not tested)
_LIN_EXTERNAL = [("Chat model\n(external)", "kraken-kuzushiji", "parent not in set"),
                 ("German manuscripts\nbase (eScriptorium)", "kraken-mendelssohn_letters", "parent not in set"),
                 ("Manu McFrench\n(Zenodo 6657809)", "kraken-peraire_french", "parent not in set")]


def plot_zoo_lineage(data: dict, th: dict):
    """Lineage drawing: documented edges coloured by what the weights say, edges only the weights suggest, and the models without relation."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sv, ln = data["survey"], data["lineage"]
    dist = {frozenset((p["a"], p["b"])): p["distance"] for p in ln["pairs"]}
    M = sv["models"]
    fig, ax = plt.subplots(figsize=(11.5, 10.6))
    ax.set_axis_off(); ax.set_xlim(-0.9, 6.9); ax.set_ylim(-7.6, 10.2)
    style = {"ok": (th["O"], "-", 2.0), "cont": (th["O"], "-", 1.0), "bad": (th["hi"], (0, (4, 3)), 1.8), "arch": (th["hi"], (0, (4, 3)), 1.8),
             "weights": (th["C"], (0, (1, 2)), 2.0), "pair": (th["C"], (0, (1, 2)), 1.4)}
    for pa, ch, kind in _LIN_EDGES:
        (x0, y0), (x1, y1) = _LIN_NODES[pa], _LIN_NODES[ch]
        col, ls, lw = style[kind]
        if kind == "pair":
            ax.plot([x0, x1], [y0, y1], color=col, ls=ls, lw=lw, zorder=1)
            d = dist.get(frozenset((pa, ch)))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, f"{d:.2f}", color=col, fontsize=7.5, ha="center", va="bottom", bbox={"fc": th["bg"], "ec": "none", "pad": 0.5}, zorder=4)
            continue
        ax.annotate("", xy=(x1 - 0.42, y1), xytext=(x0 + 0.42, y0), arrowprops={"arrowstyle": "-" if (pa, ch) == ("kraken-catmus_medieval", "kraken-old_norse_am305") or kind == "pair" else "-|>", "color": col, "lw": lw, "linestyle": ls,
                                                                              "shrinkA": 0, "shrinkB": 0, "mutation_scale": 11})
        d = dist.get(frozenset((pa, ch)))
        lab = "different architecture" if kind == "arch" else (f"{d:.2f}" if d is not None else "")
        if lab and ch in ("kraken-bifrost_old_norse", "kraken-catmus_gothic_print", "kraken-old_norse_am305"):
            dd = "" if d is None else f"{d:.2f}"
            note = {"kraken-bifrost_old_norse": f"distance {dd}: weights unrelated to the documented base",
                    "kraken-catmus_gothic_print": "documented base has another architecture",
                    "kraken-old_norse_am305": f"distance {dd}: close by weights, no direction known"}[ch]
            ax.text(x1 + 0.72, y1, note, color=col, fontsize=7.5, ha="left", va="center")
        elif lab:
            ax.text(x0 + (x1 - x0) * 0.38 + 0.05, y0 + (y1 - y0) * 0.38 + 0.08, lab, color=col, fontsize=7.5, ha="left", va="bottom",
                    bbox={"fc": th["bg"], "ec": "none", "pad": 0.5})
    involved = set(_LIN_NODES)
    for mid, (x, y) in _LIN_NODES.items():
        inh = M[mid]["origin"] == "in-house"
        ax.text(x, y, mid.removeprefix("kraken-"), ha="center", va="center", fontsize=8, color=th["fg"], zorder=3,
                bbox={"boxstyle": "round,pad=0.35", "fc": th["bg"], "ec": th["hi"] if inh else th["muted"], "lw": 1.6 if inh else 1.0})
    # external and unfetched parents
    ax.text(5.35, 5.9, "documented parents that are not in the surveyed set", fontsize=8.5, color=th["muted"], ha="center")
    for k, (lab, child, why) in enumerate(_LIN_EXTERNAL):
        y = 5.0 - 1.5 * k
        ax.text(4.35, y, lab, ha="center", va="center", fontsize=7.5, color=th["muted"],
                bbox={"boxstyle": "round,pad=0.3", "fc": th["bg"], "ec": th["muted"], "lw": 0.9, "ls": "--"})
        ax.annotate("", xy=(5.55, y), xytext=(5.0, y), arrowprops={"arrowstyle": "-|>", "color": th["muted"], "lw": 1.0, "linestyle": (0, (3, 3))})
        ax.text(5.65, y, child.removeprefix("kraken-"), ha="left", va="center", fontsize=7.5)
    # the rest
    rest = [m for m in sorted(M) if m not in involved and m not in {c for _, c, _ in [(0, e[1], 0) for e in _LIN_EXTERNAL]}
            and not any(m == e[1].split("\n")[0] for e in _LIN_EXTERNAL) and M[m]["origin"] == "zoo"
]
    rest = [m for m in rest if m not in ("kraken-kuzushiji", "kraken-mendelssohn_letters", "kraken-peraire_french")]
    ax.text(-0.85, -5.45, f"No relation found to any other surveyed model ({len(rest)} zoo models; nearest neighbour ≥ 1.0, mostly 1.3–1.5):", fontsize=8.5, color=th["muted"], ha="left")
    ax.text(-0.85, -5.9, ", ".join(m.removeprefix("kraken-") for m in rest[:5]) + ",\n" + ", ".join(m.removeprefix("kraken-") for m in rest[5:9]) + ",\n" + ", ".join(m.removeprefix("kraken-") for m in rest[9:]),
            fontsize=7.5, ha="left", va="top", color=th["fg"], linespacing=1.5)
    handles = [Line2D([], [], color=th["O"], lw=2), Line2D([], [], color=th["hi"], lw=1.8, ls=(0, (4, 3))), Line2D([], [], color=th["C"], lw=2, ls=(0, (1, 2))),
               Line2D([], [], marker="s", ls="", mfc=th["bg"], mec=th["hi"], mew=1.6), Line2D([], [], marker="s", ls="", mfc=th["bg"], mec=th["muted"])]
    ax.legend(handles, ["documented, and the weights agree (distance printed)", "documented, but the weights do not support it",
                        "suggested by weights and codec only (direction from the codec)", "in-house model", "zoo model"],
              loc="center right", bbox_to_anchor=(1.0, 0.405), frameon=False, fontsize=7.2)
    fig.suptitle("Lineage of the surveyed models: what the records say and what the weights say  (distance 1.41 = unrelated)", x=0.02, ha="left", y=0.93, fontsize=10)
    return fig


def plot_zoo_appearance(data: dict, th: dict):
    """What looks related (codec overlap, name, architecture) against what is related (weights), one dot per pair of models with equal architecture."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sv, ln = data["survey"], data["lineage"]
    pc = {frozenset((p["child"], p["parent"])) for p in sv["documented_pairs"] if p["status"] == "tested" and p["reading"]["blocks"]["conv_w"] < 1.0}
    pc |= {frozenset((p["child"], p["parent"])) for p in sv["inferred_pairs"]}
    bad = {frozenset(("kraken-bifrost_old_norse", "kraken-catmus_medieval"))}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 5.4), gridspec_kw={"wspace": 0.62, "width_ratios": [1.5, 1]})
    cats = {"other": [], "kin": [], "pc": [], "name": [], "bad": []}
    for p in ln["pairs"]:
        if p["distance"] is None:
            continue
        k = frozenset((p["a"], p["b"]))
        c = "pc" if k in pc else "bad" if k in bad else "kin" if p["distance"] < 1.0 else "name" if p["shared_name_family"] else "other"
        cats[c].append(p)
    spec = {"other": (th["grid"], 10, "any other pair, same architecture"), "kin": (th["soft"], 22, "related through a shared ancestor, not parent and child"),
            "name": (th["L"], 24, "same project or name, unrelated weights"), "pc": (th["O"], 34, "parent → child (documented and tested, or inferred)"),
            "bad": (th["hi"], 70, "documented base, weights unrelated")}
    for c in ("other", "kin", "name", "pc", "bad"):
        col, sz, _ = spec[c]
        a.scatter([p["codec_jaccard"] for p in cats[c]], [p["distance"] for p in cats[c]], s=sz, color=col, linewidths=0,
                  marker="D" if c == "bad" else "o", alpha=0.9 if c != "other" else 0.8, zorder=3 if c != "other" else 2)
    a.axhline(2 ** 0.5, color=th["muted"], lw=0.8, ls="--"); a.text(0.0, 1.425, "independent networks √2", color=th["muted"], fontsize=7.5)
    a.set_xlabel("codec overlap (Jaccard of the character sets)"); a.set_ylabel("weight distance (non-output weights)")
    a.grid(linewidth=0.5); a.set_axisbelow(True); a.set_ylim(-0.62, 1.65); a.set_yticks(np.arange(0, 1.7, 0.2)); a.set_xlim(-0.02, 1.02)
    notes = {("kraken-bifrost_old_norse", "kraken-catmus_medieval"): "bifrost ~ catmus_medieval:\ncodec reused, weights not",
             ("kraken-latin_expanded_11c", "kraken-latin_graphematic_11c"): "the two Latin 11th c.\nmodels (same project)",
             ("kraken-catmus_print_large", "kraken-latin_incunabula"): "incunabula ← print_large:\ncodec rebuilt, weights kept",
             ("kraken-catmus_medieval", "kraken-old_norse_am305"): "am305 ~ catmus_medieval",
             ("kraken-bastarda_inzigkofen", "kraken-cursive_inzigkofen"): "siblings: bastarda ~ cursive",
             ("kraken-german_print", "kraken-mendelssohn_letters"): "german_print ~ mendelssohn",
             ("kraken-catmus_medieval", "kraken-mccatmus"): "catmus_medieval ~ mccatmus",
             ("kraken-mccatmus", "kraken-prima"): "mccatmus → prima", ("german-xix-v1", "kraken-bohemian_19th"): "bohemian → xix-v1"}
    for p in ln["pairs"]:
        for (u, v), t in notes.items():
            if {p["a"], p["b"]} == {u, v} and p["distance"] is not None:
                off = {"bifrost": (-8, -28), "Latin 11th": (-30, 34), "german_print": (6, 22), "catmus_medieval ~ mccatmus": (-14, -26),
                       "siblings": (-100, 14), "mccatmus →": (6, -22), "incunabula": (14, 14)}
                off = next((v for k_, v in off.items() if k_ in t), (6, 6))
                a.annotate(t, (p["codec_jaccard"], p["distance"]), xytext=off, textcoords="offset points", fontsize=7, color=th["fg"],
                           ha="left" if off[0] > 0 else "right", arrowprops={"arrowstyle": "-", "color": th["muted"], "lw": 0.5})
    a.legend([Line2D([], [], marker="D" if c == "bad" else "o", ls="", color=spec[c][0], markersize=6 if c != "other" else 4) for c in ("pc", "kin", "name", "bad", "other")],
             [spec[c][2] for c in ("pc", "kin", "name", "bad", "other")], loc="lower left", fontsize=7.3, frameon=True, framealpha=0.9)
    a.set_title("A model that shares a codec, a name or a project need not share weights", loc="left", fontsize=9.5)
    # panel b: pairs that are not same-architecture cannot be related; show the share that name families would have predicted
    fam = [p for p in ln["pairs"] if p["shared_name_family"] and p["a"] in sv["models"] and p["b"] in sv["models"]
           and not (sv["models"][p["a"]]["origin"] == "in-house" and sv["models"][p["b"]]["origin"] == "in-house")]
    rel = [p for p in fam if p["distance"] is not None and p["distance"] < 1.0]
    diff = [p for p in fam if not p["same_architecture"]]
    unrel = [p for p in fam if p["distance"] is not None and p["distance"] >= 1.0]
    vals = [len(rel), len(unrel), len(diff)]
    b.barh([2, 1, 0], vals, color=[th["O"], th["L"], th["muted"]], height=0.6)
    for y, v in zip([2, 1, 0], vals):
        b.text(v + 0.3, y, str(v), va="center", fontsize=9)
    b.set_yticks([2, 1, 0]); b.set_yticklabels(["related", "same architecture,\nunrelated", "different\narchitecture"], fontsize=8)
    b.set_xlabel("pairs (zoo and in-house models, excluding in-house pairs)")
    b.set_title(f"{len(fam)} pairs that share a project or name", loc="left", fontsize=9.5); b.grid(axis="x", linewidth=0.5); b.set_axisbelow(True)
    b.set_xlim(0, max(vals) * 1.15)
    fig.suptitle("Close by appearance, close by weights: not the same thing", x=0.0, ha="left", fontsize=10.5)
    return fig


def figure_zoo(out: Path, fmt: str, path: Path = ZOO_JSON) -> list[Path]:
    data = json.loads(Path(path).read_text())
    paths = []
    if LINEAGE_JSON.exists():
        both = {"survey": data, "lineage": json.loads(LINEAGE_JSON.read_text())}
        paths += render(plot_zoo_lineage, both, out, "zoo-lineage", fmt)
        paths += render(plot_zoo_appearance, both, out, "zoo-appearance", fmt)
    for fn, name in ((plot_zoo_blank, "zoo-blank-bias"), (plot_zoo_distance, "zoo-distance-standard"), (plot_zoo_pairs, "zoo-pairs")):
        paths += render(fn, data, out, name, fmt)
    return paths


FIGURES = {"arch": figure_architecture, "drift": figure_drift, "rows": figure_rows,
           "alpha": figure_alpha, "padding": figure_padding}

CAPTIONS = {
    "architecture": "The VGSL stack of the in-house models, one rectangle per tensor, width ∝ parameter count; "
                    "the conv stack (5 % of the parameters) magnified underneath. Read from `kraken-german-xix-blb-v1` "
                    "(all seven in-house models share the stack; only the output rows differ).",
    "drift-bern": "Bern family, every parent→child pair: relative L2 of child − parent per tensor and the cosine "
                  "between them. The output layer is compared over the characters both codecs share, rows paired by character.",
    "drift-xix": "German XIX family: `german-xix-v1` → `german-xix-blb-v1` (the C3 subject), same reading.",
    "output-rows-german-xix-blb-v1": "Output rows of `german-xix-v1` → `german-xix-blb-v1` paired by character: "
                                     "row norm before/after and bias change against the character's frequency in `blb-train-300`.",
    "output-rows-bern-combined-v1": "Same for `bern-hertwig-v2` → `bern-combined-v1` (target 1), frequency from `bern-hertwig-train-300`.",
    "alpha-sweep-xix-blb-v1": "C3 per-block α-sweep on `german-xix-blb-v1`: one block set to parent + α(child − parent), "
                              "ΔCER against the unedited child with paired-bootstrap 95 % intervals (filled: excludes 0).",
    "padding-cr255": "Kernel-row L2 of the `Cr255,1,85` conv of the collapse probe: row 127 is the only row that "
                     "receives gradient; the other 254 rows only ever multiply padding and stay at their initialisation norm.",
}


def write_readme(out: Path, written: list[Path], inp: Inputs, argv: list[str]) -> Path:
    known = known_digests()
    names = sorted({p.stem.removesuffix("-dark") for p in written}, key=lambda s: list(CAPTIONS).index(s) if s in CAPTIONS else 99)
    ext = written[0].suffix if written else ".svg"
    L = ["# Figures", "",
         "Generated by `wsp.viz` — do not edit by hand. Regenerate with", "",
         "```bash", "python -m wsp.viz " + " ".join(argv), "```", "",
         "Each figure exists twice: `<name>" + ext + "` (light) and `<name>-dark" + ext + "` (dark). In MkDocs Material:", "",
         "```markdown", f"![caption](figures/<name>{ext}#only-light)", f"![caption](figures/<name>-dark{ext}#only-dark)", "```", "",
         "Colours are Okabe–Ito (colour-blind safe). No forward pass and no held-out data went into any figure.", "",
         "| figure | what it shows |", "|---|---|"]
    for n in names:
        L.append(f"| `{n}{ext}` | {CAPTIONS.get(n, '')} |")
    L += ["", "## Inputs", "", "Weights (sha256 of the file read, checked against `models/DIGESTS.md`):", "",
          "| model | sha256 | DIGESTS.md |", "|---|---|---|"]
    for mid, h in sorted(inp.used.items()):
        L.append(f"| `{mid}` | `{h}` | {'match' if known.get(mid) == h else ('MISMATCH' if mid in known else 'not listed')} |")
    if not inp.used:
        L.append("| — | no weights read; figures from committed JSON only | |")
    L += ["", "Committed data:", ""] + [f"- `{s}`" for s in sorted(inp.sources)]
    if any(s.startswith("docs/baselines/alpha-") for s in inp.sources):
        sw = json.loads(ALPHA_JSON.read_text())
        L += ["", f"The α-sweep JSON records tensor digests (`Subject.digest()`, not file sha256): "
              f"child `{sw['child_digest']}`, parent `{sw['parent_digest']}`; practice list sha256 `{sw['practice_sha256']}`."]
    L += ["", "Code: `wsp/viz.py` (plots), `wsp/stats.py` (`describe`, `lineage_diff`, `kernel_row_l2`), "
          "`wsp/diff.py` (`compare`), `wsp/align.py` (`codec_row_map`).", ""]
    p = Path(out) / "README.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="wsp.viz")
    ap.add_argument("figure", choices=["all", "zoo", *FIGURES])
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=str(REPO / "models" / "kraken"), help="directory of kraken-<id>.mlmodel")
    ap.add_argument("--format", default="svg", choices=["svg", "png"])
    a = ap.parse_args(argv)
    if a.figure == "zoo":
        for p in figure_zoo(Path(a.out), a.format):
            print(p)
        return
    inp = Inputs(Path(a.models))
    todo = FIGURES if a.figure == "all" else {a.figure: FIGURES[a.figure]}
    written = []
    for key, fn in todo.items():
        written += fn(inp, Path(a.out), a.format)
    for p in written:
        print(p)
    if a.figure == "all":
        cmd = ["all", "--out", a.out] + (["--format", a.format] if a.format != "svg" else [])    # no local paths
        print(write_readme(Path(a.out), written, inp, cmd))


if __name__ == "__main__":
    main()
