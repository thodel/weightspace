"""Readings without a forward pass: tensor statistics and lineage diffs.

    python -m wsp.stats describe <model.mlmodel>                 per-tensor norms, output-row energies
    python -m wsp.stats diff <parent.mlmodel> <child.mlmodel>    lineage diff, codec rows aligned by character
    python -m wsp.stats padding <cr255.mlmodel>                   kernel-row energies of the Cr255 conv

Everything here reads ``KrakenSubject.tensors()`` and the codec; nothing is
counted against a budget. Output is JSON (``--json``) or markdown tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wsp import align, diff as diffmod


def _blocks(names: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(n.split(".")[0], []).append(n)
    return out


def describe(subject) -> dict:
    t = subject.tensors()
    rows = []
    for name, w in t.items():
        w64 = w.astype(np.float64)
        rows.append({"name": name, "shape": list(w.shape), "numel": int(w.size),
                     "l2": float(np.linalg.norm(w64)), "rms": float(np.sqrt((w64 ** 2).mean())),
                     "absmax": float(np.abs(w64).max()), "zeros": int((w == 0).sum())})
    out_name = next((n for n in t if n.endswith("lin.weight") and n.startswith("O_")), None)
    out = {"tensors": rows, "parameters": int(sum(r["numel"] for r in rows)), "output_layer": None}
    if out_name:
        W = t[out_name].astype(np.float64); b = t[out_name.replace("weight", "bias")].astype(np.float64)
        energy = np.linalg.norm(W, axis=1)
        l2c = {v: k for k, v in subject.codec_map().items()}
        out["output_layer"] = {
            "name": out_name, "classes": int(W.shape[0]), "hidden": int(W.shape[1]),
            "blank": {"row_norm": float(energy[0]), "bias": float(b[0])},
            "row_norm_mean": float(energy[1:].mean()), "row_norm_std": float(energy[1:].std()),
            "bias_mean": float(b[1:].mean()), "bias_std": float(b[1:].std()),
            "rows": [{"row": int(i), "char": l2c.get(int(i), "<blank>" if i == 0 else "?"),
                      "norm": float(energy[i]), "bias": float(b[i])} for i in range(W.shape[0])],
        }
        order = np.argsort(energy)
        out["output_layer"]["weakest"] = [out["output_layer"]["rows"][int(i)] for i in order[:10]]
        out["output_layer"]["strongest"] = [out["output_layer"]["rows"][int(i)] for i in order[::-1][:10]]
        out["output_layer"]["bias_extremes"] = sorted(out["output_layer"]["rows"], key=lambda r: r["bias"])[:5] + \
            sorted(out["output_layer"]["rows"], key=lambda r: -r["bias"])[:5]
    return out


def lineage_diff(parent, child, row_top: int = 12) -> dict:
    """Per-tensor diff parent→child; the output layer is compared row-by-character."""
    a, b = parent.tensors(), child.tensors()
    ca, cb = parent.codec_map(), child.codec_map()
    cmp = diffmod.compare(a, b, row_top=row_top)
    out_w = next((n for n in a if n.endswith("lin.weight") and n.startswith("O_")), None)
    out_b = out_w.replace("weight", "bias") if out_w else None
    result = {"tensors": cmp, "codec": {"parent": len(ca), "child": len(cb),
                                        "added": sorted(set(cb) - set(ca)), "removed": sorted(set(ca) - set(cb)),
                                        "same_order": [c for c in ca if c in cb] == [c for c in cb if c in ca]}}
    if out_w:
        rm = align.codec_row_map(ca, cb); rm[0] = 0    # blank ↔ blank
        Wa, Wb = a[out_w].astype(np.float64), b[out_w].astype(np.float64)
        ba, bb = a[out_b].astype(np.float64), b[out_b].astype(np.float64)
        rows = []
        for ia, ib in rm.items():
            d = Wb[ib] - Wa[ia]
            rows.append({"char": "<blank>" if ia == 0 else next(k for k, v in ca.items() if v == ia),
                         "row_parent": int(ia), "row_child": int(ib),
                         "l2": float(np.linalg.norm(d)), "rel": float(np.linalg.norm(d) / (np.linalg.norm(Wa[ia]) or 1)),
                         "cos": float(Wa[ia] @ Wb[ib] / ((np.linalg.norm(Wa[ia]) * np.linalg.norm(Wb[ib])) or 1)),
                         "bias_delta": float(bb[ib] - ba[ia]), "norm_parent": float(np.linalg.norm(Wa[ia])),
                         "norm_child": float(np.linalg.norm(Wb[ib]))})
        rows.sort(key=lambda r: -r["l2"])
        new_rows = [{"char": c, "row_child": cb[c], "norm": float(np.linalg.norm(Wb[cb[c]])), "bias": float(bb[cb[c]])}
                    for c in sorted(set(cb) - set(ca))]
        result["output_rows"] = {"aligned": len(rm), "moved_most": rows[:row_top], "moved_least": rows[-5:],
                                 "mean_rel": float(np.mean([r["rel"] for r in rows])),
                                 "new_rows": new_rows,
                                 "blank": next(r for r in rows if r["row_parent"] == 0)}
        if a[out_w].shape != b[out_w].shape:
            result["tensors"][out_w]["note"] = "shape differs (codec grew); rows compared by character above"
    # block summary
    blocks = {}
    for name, e in cmp.items():
        if "rel_l2" in e:
            blk = name.split(".")[0]
            blocks.setdefault(blk, []).append(e["rel_l2"])
    result["blocks"] = {k: {"tensors": len(v), "mean_rel_l2": float(np.mean(v)), "max_rel_l2": float(np.max(v))} for k, v in blocks.items()}
    return result


def padding_rows(subject, keep: int = 127) -> dict:
    """Kernel-row energies of a (out, in, kH, 1) conv: the kept row against the padding rows."""
    t = subject.tensors()
    name = next(n for n in t if n.endswith("co.weight") and t[n].ndim == 4 and t[n].shape[2] > 100)
    w = t[name].astype(np.float64)
    rows = np.sqrt((w ** 2).sum(axis=(0, 1, 3)))      # per kernel row
    other = np.delete(rows, keep)
    return {"tensor": name, "shape": list(w.shape), "kept_row": keep, "kept_row_l2": float(rows[keep]),
            "padding_rows_l2_mean": float(other.mean()), "padding_rows_l2_std": float(other.std()),
            "padding_rows_l2_min": float(other.min()), "padding_rows_l2_max": float(other.max()),
            "ratio_kept_to_padding_mean": float(rows[keep] / other.mean()),
            "padding_elements": int(other.size * w.shape[0] * w.shape[1] * w.shape[3]),
            "rows_sorted_top5": [[int(i), float(rows[i])] for i in np.argsort(rows)[::-1][:5]]}


# -- markdown ------------------------------------------------------------------

def md_describe(d: dict, title: str) -> str:
    L = [f"### {title}", "", f"{d['parameters']:,} parameters, {len(d['tensors'])} tensors", "",
         "| tensor | shape | L2 | rms | abs max |", "|---|---|---:|---:|---:|"]
    for r in d["tensors"]:
        L.append(f"| `{r['name']}` | {r['shape']} | {r['l2']:.2f} | {r['rms']:.4f} | {r['absmax']:.3f} |")
    o = d.get("output_layer")
    if o:
        L += ["", f"Output layer `{o['name']}`: {o['classes']} rows × {o['hidden']}; blank row norm {o['blank']['row_norm']:.3f}, "
              f"bias {o['blank']['bias']:+.3f}; other rows norm {o['row_norm_mean']:.3f} ± {o['row_norm_std']:.3f}, "
              f"bias {o['bias_mean']:+.3f} ± {o['bias_std']:.3f}.", "",
              "weakest rows: " + ", ".join(f"`{r['char']}` {r['norm']:.2f}" for r in o["weakest"]), "",
              "strongest rows: " + ", ".join(f"`{r['char']}` {r['norm']:.2f}" for r in o["strongest"]), "",
              "bias extremes: " + ", ".join(f"`{r['char']}` {r['bias']:+.2f}" for r in o["bias_extremes"])]
    return "\n".join(L)


def md_diff(d: dict, title: str) -> str:
    L = [f"### {title}", "",
         f"Codec: parent {d['codec']['parent']}, child {d['codec']['child']}, added {len(d['codec']['added'])} "
         f"{''.join(d['codec']['added'])[:80]!r}, removed {len(d['codec']['removed'])}; "
         f"shared order preserved: {d['codec']['same_order']}.", "",
         "| block | tensors | mean rel L2 | max rel L2 |", "|---|---:|---:|---:|"]
    for k, v in d["blocks"].items():
        L.append(f"| {k} | {v['tensors']} | {v['mean_rel_l2']:.4f} | {v['max_rel_l2']:.4f} |")
    L += ["", "| tensor | shape | rel L2 | cosine | changed |", "|---|---|---:|---:|---:|"]
    for name, e in d["tensors"].items():
        if "rel_l2" in e:
            L.append(f"| `{name}` | {e['shape']} | {e['rel_l2']:.4f} | {e['cosine']:.5f} | {e['changed']}/{e['numel']} |")
        else:
            L.append(f"| `{name}` | {e} | | | |")
    o = d.get("output_rows")
    if o:
        L += ["", f"Output rows aligned by character: {o['aligned']}; mean relative change {o['mean_rel']:.4f}; "
              f"blank row rel {o['blank']['rel']:.4f}, bias Δ {o['blank']['bias_delta']:+.3f}.", "",
              "| moved most | rel L2 | cos | bias Δ | norm parent → child |", "|---|---:|---:|---:|---|"]
        for r in o["moved_most"]:
            L.append(f"| `{r['char']}` | {r['rel']:.3f} | {r['cos']:.3f} | {r['bias_delta']:+.3f} | {r['norm_parent']:.2f} → {r['norm_child']:.2f} |")
        if o["new_rows"]:
            L += ["", "new rows (child only): " + ", ".join(f"`{r['char']}` norm {r['norm']:.2f} bias {r['bias']:+.2f}" for r in o["new_rows"][:20])]
    return "\n".join(L)


def md_padding(d: dict, title: str) -> str:
    return "\n".join([f"### {title}", "",
                      f"`{d['tensor']}` {d['shape']}: kept row {d['kept_row']} L2 {d['kept_row_l2']:.3f}; the other "
                      f"{d['shape'][2] - 1} rows L2 {d['padding_rows_l2_mean']:.3f} ± {d['padding_rows_l2_std']:.3f} "
                      f"(min {d['padding_rows_l2_min']:.3f}, max {d['padding_rows_l2_max']:.3f}); ratio {d['ratio_kept_to_padding_mean']:.2f}; "
                      f"{d['padding_elements']:,} padding-only weights.",
                      "", "top rows by L2: " + ", ".join(f"{i}: {v:.3f}" for i, v in d["rows_sorted_top5"])])


def main(argv=None):
    from wsp.kraken_subject import KrakenSubject
    ap = argparse.ArgumentParser(prog="wsp.stats")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe"); d.add_argument("model"); d.add_argument("--json", action="store_true")
    f = sub.add_parser("diff"); f.add_argument("parent"); f.add_argument("child"); f.add_argument("--json", action="store_true")
    p = sub.add_parser("padding"); p.add_argument("model"); p.add_argument("--keep", type=int, default=127); p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "describe":
        r = describe(KrakenSubject(a.model)); print(json.dumps(r, indent=1) if a.json else md_describe(r, Path(a.model).stem))
    elif a.cmd == "diff":
        r = lineage_diff(KrakenSubject(a.parent), KrakenSubject(a.child))
        print(json.dumps(r, indent=1) if a.json else md_diff(r, f"{Path(a.parent).stem} → {Path(a.child).stem}"))
    elif a.cmd == "padding":
        r = padding_rows(KrakenSubject(a.model), a.keep); print(json.dumps(r, indent=1) if a.json else md_padding(r, Path(a.model).stem))


if __name__ == "__main__":
    main()
