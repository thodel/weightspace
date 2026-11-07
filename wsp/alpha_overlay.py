"""The α-sweep per line and per character (epic thodel/weightspace#3, sub-task 4).

    python -m wsp.alpha_overlay --child models/kraken/kraken-german-xix-blb-v1.mlmodel \\
        --parent models/kraken/kraken-german-xix-v1.mlmodel --practice data/practice/blb-train-300.jsonl \\
        --out docs/viz [--lines KEY ...] [--views 3] [--check docs/baselines/alpha-xix-blb-v1.json]

``wsp.alpha_sweep`` gives one CER per (block, α). This keeps what it throws away: for every block set to
``parent + α(child − parent)`` (the output layer with rows paired by character, as there), every line's
CTC strip (P(blank) per frame) and best path, and per reference character whether it is emitted,
replaced or deleted. Frames do not move with α (the conv strides are fixed), so the strips line up
with the child's forced alignment of the reference.

Output: ``<out>/alpha-overlay-<child>__<set>.json`` (per block and α: CER and the edit counts by kind —
letters and spaces deleted, substituted, inserted — and per line the status string), the summary
figure ``alpha-overlay-<child>__<set>.svg`` and per-line views under ``lines/`` (manuscript crops:
git-ignored). ``--check`` compares the CERs with a committed ``wsp.alpha_sweep`` run.
A control, not a session: practice lines only, nothing counted.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from wsp.alpha_sweep import DEFAULT_BLOCKS, aligned_parent_output, block_names
from wsp.layerview import greedy, line_image, read_line
from wsp.metrics import align, normalise, score

ALPHAS = [0.0, 0.5, 1.5]          # 1.0 is the child itself
CODE = {"ok": "o", "sub": "s", "del": "d"}


def block_plan(child, parent, blocks=DEFAULT_BLOCKS) -> dict[str, dict[str, np.ndarray]]:
    """block → {tensor: parent value}, the output layer in the child's row order (``alpha_sweep``'s pairing)."""
    pt = parent.tensors()
    plan = {b: {n: pt[n] for n in block_names(child.tensors(), b)} for b in blocks}
    plan["O_18"] = {n: v for n, v in aligned_parent_output(parent, child).items() if not n.startswith("_")}
    return plan


@contextmanager
def interpolated(child, par: dict[str, np.ndarray], alpha: float):
    ct = child.tensors()
    orig = {n: ct[n].copy() for n in par}
    try:
        for n in par:
            child.set_tensor(n, par[n] + alpha * (orig[n] - par[n]))
        yield
    finally:
        for n, v in orig.items():
            child.set_tensor(n, v)


def logits_all(subject, items) -> list[np.ndarray | None]:
    out = []
    for it in items:
        a = subject.activations([it], ["O_18"])["O_18"]
        out.append(a[0].astype(np.float64) if a else None)
    return out


def edit_counts(hyps, refs) -> dict:
    """Edits by kind over all lines, spaces apart from other characters."""
    c = dict.fromkeys(["del_letter", "del_space", "ins_letter", "ins_space", "sub"], 0)
    for h, r in zip(hyps, refs):
        for a, b in align(normalise(r), normalise(h)):
            if a and not b:
                c["del_space" if a == " " else "del_letter"] += 1
            elif b and not a:
                c["ins_space" if b == " " else "ins_letter"] += 1
            elif a != b:
                c["sub"] += 1
    return c


def _decode(lg, l2c):
    if lg is None:
        return "", []
    runs = greedy(lg)
    return "".join(l2c.get(l, "") for l, _, _ in runs), [(l2c.get(l, "?"), a, b) for l, a, b in runs]


def run(child, parent, items, blocks=DEFAULT_BLOCKS, alphas=ALPHAS, log=print) -> dict:
    l2c = {v: k for k, v in child.codec_map().items()}
    refs = [x["text"] for x in items]
    t0 = time.time()
    base = logits_all(child, items)
    lines = []
    for it, lg in zip(items, base):
        rd = read_line(child, it, {"logp": lg - np.logaddexp.reduce(lg, axis=1, keepdims=True), "frames": len(lg)}) \
            if lg is not None else None
        lines.append({"key": it["key"], "ref": rd["ref"] if rd else normalise(it["text"]), "runs": rd["runs"] if rd else None,
                      "states": {}})
    def record(tag, logits):
        hyps = []
        for ln, lg in zip(lines, logits):
            text, chars = _decode(lg, l2c)
            hyps.append(text)
            p_blank = None if lg is None else np.exp(lg[:, 0] - np.logaddexp.reduce(lg, axis=1))
            from wsp.layerview import char_status
            ln["states"][tag] = {"hyp": normalise(text), "status": "".join(CODE[s] for s in char_status(ln["ref"], normalise(text))),
                                 "p_blank": None if p_blank is None else np.round(p_blank, 4).tolist(), "chars": chars}
        m = score(hyps, refs)
        return {"cer": m.cer, "wer": m.wer, "collapsed": m.collapsed, "edits": edit_counts(hyps, refs)}
    summary = {"child": record("child", base)}
    log(f"child {summary['child']['cer']:.5f}  {time.time() - t0:.0f}s")
    plan = block_plan(child, parent, blocks)
    for b, par in plan.items():
        for a in alphas:
            with interpolated(child, par, a):
                lg = logits_all(child, items)
            tag = f"{b}@{a:g}"
            summary[tag] = {"block": b, "alpha": a, **record(tag, lg)}
            e, e0 = summary[tag]["edits"], summary["child"]["edits"]
            log(f"{tag:9s} CER {summary[tag]['cer']:.5f}  Δdel letters {e['del_letter'] - e0['del_letter']:+4d}  "
                f"Δdel spaces {e['del_space'] - e0['del_space']:+4d}  Δsub {e['sub'] - e0['sub']:+4d}  "
                f"Δins {e['ins_letter'] + e['ins_space'] - e0['ins_letter'] - e0['ins_space']:+4d}  {time.time() - t0:.0f}s")
    return {"summary": summary, "lines": lines, "blocks": list(plan), "alphas": alphas}


def check(summary: dict, sweep: dict) -> list[dict]:
    """CER of this run against a committed wsp.alpha_sweep (real passes, the same interpolation)."""
    out = []
    for r in sweep["rows"]:
        tag = f"{r['block']}@{r['alpha']:g}"
        if tag in summary:
            out.append({"tag": tag, "here": summary[tag]["cer"], "committed": r["metrics"]["cer"],
                        "equal": abs(summary[tag]["cer"] - r["metrics"]["cer"]) < 1e-9})
    return out


# -- figures ---------------------------------------------------------------------------------

KINDS = [("del_letter", "letters deleted"), ("del_space", "spaces deleted"), ("sub", "substituted"),
         ("ins_letter", "letters inserted"), ("ins_space", "spaces inserted")]


def plot_summary(data: dict, th: dict):
    """Per block and α: how many more (or fewer) edits of each kind than the child makes."""
    import matplotlib.pyplot as plt
    s, blocks, alphas = data["summary"], data["blocks"], data["alphas"]
    e0 = s["child"]["edits"]
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
    cols = [th["hi"], th["L"], th["alt"], th["C"], th["soft"]]
    hatch = [None, "////", None, None, "////"]
    w = 0.8 / len(KINDS)
    for ax, a in zip(np.atleast_1d(axes), alphas):
        x = np.arange(len(blocks))
        for k, ((kind, lab), col, h) in enumerate(zip(KINDS, cols, hatch)):
            v = [s[f"{b}@{a:g}"]["edits"][kind] - e0[kind] for b in blocks]
            ax.bar(x - 0.4 + w * (k + 0.5), v, width=w, color=col if not h else th["bg"], edgecolor=col, hatch=h,
                   linewidth=0.8, label=lab)
        for i, b in enumerate(blocks):
            d = 100 * (s[f"{b}@{a:g}"]["cer"] - s["child"]["cer"])
            ax.text(i, 1.0, f"{d:+.2f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7,
                    color=th["muted"])
        ax.axhline(0, color=th["muted"], linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(blocks, fontsize=8, rotation=45)
        ax.grid(axis="y", linewidth=0.5); ax.set_axisbelow(True)
        what = {0.0: "block set back to the parent", 0.5: "halfway back to the parent", 1.5: "extrapolated past the child"}
        ax.set_title(f"α = {a:g}: {what.get(a, '')}", loc="left", fontsize=9.5, pad=14)
    np.atleast_1d(axes)[0].set_ylabel("edits against the unedited child")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.suptitle(f"{data['child']} ← {data['parent']} on {data['practice']}: one block at parent + α·(child − parent), "
                 f"edits by kind against the child (child: CER {100 * s['child']['cer']:.2f} %, "
                 f"{e0['del_letter']} letters and {e0['del_space']} spaces deleted); grey: ΔCER in points",
                 x=0.0, ha="left", y=1.04, fontsize=9.5)
    return fig


def plot_line(d: dict, th: dict):
    """One line: the image, and per block the CTC strip 1 − P(blank) under every α, with each reference
    character's fate (emitted / replaced / deleted) per α underneath."""
    import matplotlib.pyplot as plt
    from wsp.viz import glyph
    blocks, tags = d["blocks"], d["tags"]                  # tags: [(label, key)] in α order, child included
    T = len(d["states"]["child"]["p_blank"])
    H = 1.25 * len(blocks) + 3.2
    fig, axes = plt.subplots(len(blocks) + 2, 1, figsize=(max(9, min(T / 14, 24)), H), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 0.7] + [1.15] * len(blocks), "hspace": 0.15})
    fig.subplots_adjust(top=1 - 0.45 / H)
    img = d["image"]
    axes[0].imshow(img, cmap="gray", aspect="auto", extent=(0, img.shape[1] / d["px_per_frame"], img.shape[0], 0), vmin=0, vmax=1)
    axes[0].set_yticks([])
    ref_ax = axes[1]
    for ch, (s, e) in zip(d["ref"], d["runs"]):
        ref_ax.text((s + e + 1) / 2, 0.5, glyph(ch), ha="center", va="center", fontsize=8)
    ref_ax.set_yticks([0.5]); ref_ax.set_yticklabels(["reference"], fontsize=8); ref_ax.set_ylim(0, 1)
    cmap = plt.get_cmap("cividis")
    shade = {lab: cmap(0.8 * i / max(len(tags) - 1, 1)) for i, (lab, _) in enumerate(tags)}   # stop before the pale yellow end
    stc = {"o": th["fg"], "s": th["alt"], "d": th["hi"]}
    t = np.arange(T) + 0.5
    for ax, b in zip(axes[2:], blocks):
        for j, (lab, key) in enumerate(tags):
            st = d["states"][key if key == "child" else f"{b}@{key}"]
            ax.step(t, 1 - np.asarray(st["p_blank"]), where="mid", color=shade[lab], linewidth=1.5 if key == "child" else 0.9,
                    label=f"α = {lab}")
            y = -0.25 - 0.22 * j
            for (s, e), code in zip(d["runs"], st["status"]):
                ax.plot((s + e + 1) / 2, y, marker="s", markersize=3.2, linestyle="", markeredgewidth=0,
                        color=stc[code] if code != "o" else th["grid"])
        ax.set_ylim(-0.25 - 0.22 * len(tags), 1.05)
        ax.set_yticks([0.5]); ax.set_yticklabels([b], fontsize=8)
        ax.axhline(0, color=th["grid"], linewidth=0.5)
    h, lab = axes[2].get_legend_handles_labels()
    axes[1].legend(h, lab, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    axes[-1].set_xlim(0, T)
    axes[-1].set_xlabel(f"output frame ({d['px_per_frame']} px each); curves: 1 − P(blank); squares per α (top row α = "
                        f"{tags[0][0]}): grey emitted, purple replaced, vermillion deleted")
    fig.suptitle(f"{d['child']} ← {d['parent']}  {d['key']}: each block alone at parent + α·(child − parent)",
                 x=0.0, ha="left", y=0.995, fontsize=9.5)
    return fig


def pick_lines(data: dict, n: int, block: str = "L_16", alpha: float = 0.5) -> list[str]:
    """Lines where one block's move changes the most characters' fate (default: L_16 halfway back), the
    child reading them at CER ≤ 20 % — so the view shows the block's work, not a defective crop."""
    from wsp.metrics import levenshtein
    tag = f"{block}@{alpha:g}"
    cand = []
    for ln in data["lines"]:
        c, x = ln["states"]["child"], ln["states"].get(tag)
        if ln["runs"] is None or x is None or len(ln["ref"]) < 15 or levenshtein(ln["ref"], c["hyp"]) > 0.2 * len(ln["ref"]):
            continue
        cand.append((sum(a != b for a, b in zip(c["status"], x["status"])), ln["key"]))
    return [k for _, k in sorted(cand, reverse=True)[:n]]


def main(argv=None):
    from wsp.data import load, sha256_file
    from wsp.kraken_subject import KrakenSubject
    from wsp.viz import render
    ap = argparse.ArgumentParser(prog="wsp.alpha_overlay")
    ap.add_argument("--child", required=True); ap.add_argument("--parent", required=True)
    ap.add_argument("--practice", required=True); ap.add_argument("--out", default="docs/viz")
    ap.add_argument("--lines", nargs="*"); ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--check"); ap.add_argument("--format", default="svg", choices=["svg", "png"])
    a = ap.parse_args(argv)
    if "heldout" in Path(a.practice).parts:
        raise SystemExit("practice lines only")
    child, parent = KrakenSubject(a.child), KrakenSubject(a.parent)
    items = load(a.practice)
    res = run(child, parent, items)
    cs, ps, pset = Path(a.child).stem, Path(a.parent).stem, Path(a.practice).stem
    res.update(child=cs, parent=ps, practice=pset,
               meta={"child_sha256": sha256_file(Path(a.child)), "parent_sha256": sha256_file(Path(a.parent)),
                     "practice_sha256": sha256_file(Path(a.practice)), "child_digest": child.digest()})
    if a.check:
        res["check"] = check(res["summary"], json.loads(Path(a.check).read_text()))
        print(f"check against {a.check}: {sum(c['equal'] for c in res['check'])}/{len(res['check'])} CERs equal")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jp = out / f"alpha-overlay-{cs}__{pset}.json"
    slim = {**res, "lines": [{**ln, "states": {t: {k: v for k, v in st.items() if k in ("hyp", "status")}
                                               for t, st in ln["states"].items()}} for ln in res["lines"]]}
    jp.write_text(json.dumps(slim, ensure_ascii=False))          # strips and best-path frames stay out of git
    print(jp)
    for p in render(plot_summary, res, out, f"alpha-overlay-{cs}__{pset}", a.format):
        print(p)
    by_key = {x["key"]: x for x in items}
    lines = {ln["key"]: ln for ln in res["lines"]}
    tags = [("0", "0"), ("0.5", "0.5"), ("1", "child"), ("1.5", "1.5")]
    for key in a.lines if a.lines is not None else pick_lines(res, a.views):
        ln = lines[key]
        img = line_image(child, by_key[key])
        d = {"key": key, "child": cs, "parent": ps, "blocks": res["blocks"], "tags": tags, "ref": ln["ref"], "runs": ln["runs"],
             "image": img, "px_per_frame": img.shape[1] // len(ln["states"]["child"]["p_blank"]), "states": ln["states"]}
        for p in render(plot_line, d, out / "lines", f"alpha-{cs}__{key.replace('/', '_')}", a.format):
            print(p)


if __name__ == "__main__":
    main()
