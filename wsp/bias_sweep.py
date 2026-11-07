"""Trade-off curve of one output-row bias: ``O_18.lin.bias[row] += δ`` over a grid, scored on practice.

    python -m wsp.bias_sweep --model models/kraken/kraken-german-xix-blb-v1.mlmodel \\
        --practice data/practice/blb-train-300.jsonl --out docs/baselines/bias-xix-blb-v1.json \\
        [--probes docs/viz/kraken-german-xix-blb-v1__blb-train-300.json] [--verify]

A bias edit adds δ to one logit column and nothing else, and kraken reads the best path, so the
edited model's text is ``argmax(logits + δ·e_row)`` decoded greedily: the curve is computed exactly
from **one** capture of the logits, not one pass per δ. ``--verify`` checks that claim against real
edited forward passes (``KrakenSubject.set_tensor`` + ``forward``) at two grid points.

A control, not a session: no budget, practice lines only. Rows: the space and the CTC blank (the
lever phase 1 used). Per δ: CER/WER/collapsed (``wsp.metrics.score``), space deletions, insertions and
substitutions from the edit alignment, the paired bootstrap against δ = 0, and — with ``--probes`` —
how many of the deleted spaces ``wsp.layerview`` found still decodable at ``O_18`` are emitted.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wsp.layerview import greedy, lost_at
from wsp.metrics import align, bootstrap_delta, normalise, per_line_edits, score

GRID = {"space": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
        "blank": [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]}


def capture_logits(subject, items) -> list[np.ndarray | None]:
    """(frames, classes) logits per line; None where the input transform cannot scale the crop (read as '')."""
    out = []
    for it in items:
        a = subject.activations([it], ["O_18"])["O_18"]
        out.append(a[0].astype(np.float64) if a else None)
    return out


def decode(logits: np.ndarray | None, l2c: dict[int, str], row: int, delta: float) -> str:
    if logits is None:
        return ""
    x = logits.copy()
    x[:, row] += delta
    return "".join(l2c.get(l, "") for l, _, _ in greedy(x))


def space_edits(hyps, refs) -> dict:
    """Space deletions, insertions and substitutions (either side a space) over all lines."""
    d = i = s = 0
    for h, r in zip(hyps, refs):
        for a, b in align(normalise(r), normalise(h)):
            if a == " " and b == "":
                d += 1
            elif a == "" and b == " ":
                i += 1
            elif a != b and " " in (a, b):
                s += 1
    return {"deleted": d, "inserted": i, "substituted": s}


def lost_spaces(probes: dict) -> dict[str, list[int]]:
    """key → positions (in the aligned reference) of deleted spaces whose O_18 probes still had them."""
    out = {}
    for ln in probes["lines"]:
        pos = [i for i, (c, st) in enumerate(zip(ln["ref"], ln["status"])) if c == " " and st == "del"
               and lost_at(ln, i) == "O_18"]
        if pos:
            out[ln["key"]] = pos
    return out


def recovered(lost: dict[str, list[int]], keys, refs_aligned: dict[str, str], hyps) -> int:
    """How many of the probe-flagged spaces the edited hypothesis now emits (aligned as 'ok')."""
    from wsp.layerview import char_status
    n = 0
    for k, h in zip(keys, hyps):
        if k in lost:
            st = char_status(refs_aligned[k], normalise(h))
            n += sum(st[i] == "ok" for i in lost[k])
    return n


def sweep(logits, items, l2c, rows: dict[str, int], grid=GRID, probes: dict | None = None, log=print) -> list[dict]:
    refs = [x["text"] for x in items]
    keys = [x["key"] for x in items]
    base = [decode(lg, l2c, 0, 0.0) for lg in logits]
    base_lines = per_line_edits(base, refs)
    lost = lost_spaces(probes) if probes else {}
    ref_al = {ln["key"]: ln["ref"] for ln in probes["lines"]} if probes else {}
    out = []
    for name, row in rows.items():
        for dlt in grid[name]:
            hyps = [decode(lg, l2c, row, dlt) for lg in logits]
            m = score(hyps, refs)
            r = {"row": name, "index": row, "delta": dlt, "metrics": {k: v for k, v in m.as_dict().items() if k != "confusions"},
                 "spaces": space_edits(hyps, refs), "vs_delta0": bootstrap_delta(base_lines, per_line_edits(hyps, refs))}
            if lost:
                r["probe_flagged_spaces"] = {"total": sum(map(len, lost.values())), "emitted": recovered(lost, keys, ref_al, hyps)}
            out.append(r)
            b = r["vs_delta0"]
            log(f"{name:5s} δ={dlt:+5.2f}  CER {m.cer:.5f}  Δ {100 * b['delta_cer']:+.3f} pt "
                f"[{100 * b['ci95'][0]:+.3f}, {100 * b['ci95'][1]:+.3f}]  spaces del {r['spaces']['deleted']} "
                f"ins {r['spaces']['inserted']}  collapsed {m.collapsed}"
                + (f"  flagged emitted {r['probe_flagged_spaces']['emitted']}/{r['probe_flagged_spaces']['total']}" if lost else ""))
    return out


def verify(subject, items, logits, l2c, checks: list[tuple[int, float]]) -> list[dict]:
    """Real edited forward passes at a few (row, δ) against the exact re-decode."""
    name = next(n for n in subject.tensors() if n.startswith("O_") and n.endswith("lin.bias"))
    orig = subject.tensors()[name].copy()
    res = []
    for row, dlt in checks:
        b = orig.copy(); b[row] += dlt
        subject.set_tensor(name, b)
        real = [normalise(p.text) for p in subject.forward(items)]
        subject.set_tensor(name, orig)
        mine = [normalise(decode(lg, l2c, row, dlt)) for lg in logits]
        res.append({"row": row, "delta": dlt, "lines": len(items), "identical": sum(a == c for a, c in zip(real, mine))})
    return res


def plot_curve(data: dict, th: dict):
    """Left: ΔCER against δ with the bootstrap band, per row. Right: space deletions against insertions."""
    import matplotlib.pyplot as plt
    rows = data["rows"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.28})
    style = {"space": (th["C"], "o"), "blank": (th["L"], "s")}
    for name in dict.fromkeys(r["row"] for r in rows):
        rr = [r for r in rows if r["row"] == name]
        col, m = style.get(name, (th["alt"], "D"))
        x = [r["delta"] for r in rr]
        y = [100 * r["vs_delta0"]["delta_cer"] for r in rr]
        lo = [100 * r["vs_delta0"]["ci95"][0] for r in rr]; hi = [100 * r["vs_delta0"]["ci95"][1] for r in rr]
        lab = "space row bias" if name == "space" else "blank bias (phase 1's lever)"
        a.fill_between(x, lo, hi, color=col, alpha=0.15, linewidth=0)
        a.plot(x, y, color=col, marker=m, markersize=4.5, label=lab)
        sp = [(r["spaces"]["deleted"], r["spaces"]["inserted"], r["delta"]) for r in rr]
        b.plot([s[0] for s in sp], [s[1] for s in sp], color=col, marker=m, markersize=4.5, label=lab)
        for d_, i_, dl in sp:
            if dl in (rr[0]["delta"], 0.0, rr[-1]["delta"]) or (name == "space" and dl in (1.0, 2.0, 3.0)):
                b.annotate(f"{dl:+g}", (d_, i_), xytext=(4, 3), textcoords="offset points", fontsize=7, color=col)
    a.axhline(0, color=th["muted"], linewidth=0.8)
    a.set_xlabel("δ added to the row's bias")
    a.set_ylabel("ΔCER against the unedited model (points)")
    a.grid(linewidth=0.5); a.set_axisbelow(True)
    a.legend(loc="upper left", fontsize=8, frameon=False)
    a.set_title("CER, paired bootstrap 95 % band", loc="left", fontsize=9.5)
    b0 = next(r for r in rows if r["delta"] == 0.0)["spaces"]
    b.plot(b0["deleted"], b0["inserted"], marker="*", markersize=11, color=th["hi"], linestyle="", zorder=5, label="unedited")
    b.set_xlabel("spaces deleted"); b.set_ylabel("spaces inserted")
    b.grid(linewidth=0.5); b.set_axisbelow(True)
    b.legend(loc="upper right", fontsize=8, frameon=False)
    b.set_title("the space trade: δ labelled on the curve", loc="left", fontsize=9.5)
    m = data.get("meta", {})
    fl = next((r["probe_flagged_spaces"] for r in rows if r["row"] == "space" and r["delta"] == 0.0
               and "probe_flagged_spaces" in r), None)
    fig.suptitle(f"{data['model']} on {data['practice']} ({m.get('lines', '?')} practice lines): one output-row bias "
                 f"moved by δ, text re-decoded exactly from the captured logits"
                 + (f"; {fl['total']} deleted spaces the O_18 probes still had" if fl else ""),
                 x=0.0, ha="left", y=1.02, fontsize=9.5)
    return fig


def main(argv=None):
    from wsp.data import load, sha256_file
    from wsp.kraken_subject import KrakenSubject
    from wsp.viz import render
    ap = argparse.ArgumentParser(prog="wsp.bias_sweep")
    ap.add_argument("--model", required=True); ap.add_argument("--practice", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--probes")
    ap.add_argument("--figures", default="docs/viz")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if "heldout" in Path(a.practice).parts:
        raise SystemExit("practice lines only")
    subject = KrakenSubject(a.model)
    items = load(a.practice)
    c2l = subject.codec_map(); l2c = {v: k for k, v in c2l.items()}
    t0 = time.time()
    logits = capture_logits(subject, items)
    print(f"logits captured {time.time() - t0:.0f}s")
    probes = json.loads(Path(a.probes).read_text()) if a.probes else None
    rows = sweep(logits, items, l2c, {"space": c2l[" "], "blank": 0}, probes=probes)
    model, pset = Path(a.model).stem, Path(a.practice).stem
    res = {"control": "output-row bias trade-off (exact re-decode of captured logits)", "model": model, "practice": pset,
           "meta": {"lines": len(items), "unreadable_crops": sum(lg is None for lg in logits),
                    "model_sha256": sha256_file(Path(a.model)), "tensor_digest": subject.digest(),
                    "practice_sha256": sha256_file(Path(a.practice)), "probes": a.probes},
           "rows": rows}
    if a.verify:
        res["verify"] = verify(subject, items, logits, l2c, [(c2l[" "], 1.0), (0, -1.0)])
        print(res["verify"])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(out)
    for p in render(plot_curve, res, Path(a.figures), f"bias-{model}__{pset}"):
        print(p)


if __name__ == "__main__":
    main()
