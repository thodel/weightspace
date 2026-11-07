"""RQ3 of Track C: exchange a block of a specific model for the same block of a more general one.

    python -m wsp.exchange --specific <S.mlmodel> --general <G.mlmodel> \\
        --sets <in-domain practice set> <other set> ... --out docs/family/exchange-<S>-from-<G>.json

For every block ``C_0 … O_18`` and every β, the block becomes ``S + β (G − S)`` (β = 1: G's block
transplanted; β = 0.5: half way), the rest stays S, and the model reads every set. The first set is
S's own material (train side, as every practice set here); the others are material neither model
was trained on, or G's. Each reading is compared, paired per line, with S's unedited reading of the
same set: corpus CER with its interval, and the block table of ``wsp.blocks`` (docs/BLOCK_STATS.md).

What it answers (docs/MODEL_FAMILY.md):

* **stabilisation** — a block exchange that leaves S's own set inside its interval and lowers CER on
  the other sets makes S more robust at no cost;
* **freezability (RQ1, functional)** — when G is S's parent, β = 1 undoes the fine-tune in that block
  alone; if S's own set does not move, that block's training could have been frozen.

The output layer is exchanged on the characters both alphabets share (rows paired by character,
``wsp.alpha_sweep.aligned_parent_output``); S's characters that G lacks keep S's rows. Body blocks
need one lineage (same initialisation): across lineages hidden units are in arbitrary order and an
element-wise exchange is meaningless, which ``--allow-unrelated`` must be passed to override.
Practice sets only: no held-out set is read here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wsp.alpha_sweep import aligned_parent_output, block_names
from wsp.blocks import bootstrap_blocks, per_line_counts
from wsp.data import load, sha256_file
from wsp.metrics import bootstrap_delta, per_line_edits, score

REPO = Path(__file__).resolve().parent.parent
BLOCKS = ["C_0", "C_3", "C_6", "C_9", "L_12", "L_14", "L_16", "O_18"]


def body_cosine(a: dict, b: dict) -> float:
    """Cosine of all tensors outside the output layer that both models have with equal shapes."""
    names = [n for n in a if not n.startswith("O_") and n in b and a[n].shape == b[n].shape]
    x = np.concatenate([a[n].ravel().astype(np.float64) for n in names]); y = np.concatenate([b[n].ravel().astype(np.float64) for n in names])
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def read(subject, items, refs):
    preds = subject.forward(items)
    hyps = [p.text for p in preds]
    return [p.key for p in preds], hyps, score(hyps, refs), per_line_edits(hyps, refs), per_line_counts(hyps, refs)


def save_readings(path: Path, keys, hyps, refs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"key": k, "hyp": h, "ref": r}, ensure_ascii=False) + "\n"
                            for k, h, r in zip(keys, hyps, refs)), encoding="utf-8")


def exchange(specific: Path, general: Path, sets: list[str], blocks: list[str], betas: list[float],
             readings_root: Path, allow_unrelated: bool = False, log=print) -> dict:
    from wsp.kraken_subject import KrakenSubject
    S, G = KrakenSubject(specific), KrakenSubject(general)
    st, gt = S.tensors(), G.tensors()
    cos = body_cosine(st, gt)
    if not blocks:                                   # every parameter block of S's architecture, in order
        blocks = [n for n, m in S.model.nn.named_children() if any(True for _ in m.parameters())]
    if cos < 0.5 and not allow_unrelated and any(not b.startswith("O_") for b in blocks):
        raise SystemExit(f"body cosine {cos:.3f}: not one lineage, an element-wise block exchange is meaningless")
    out_al = aligned_parent_output(G, S)
    data = {s: load(REPO / "data" / "practice" / f"{s}.jsonl") for s in sets}
    refs = {s: [x["text"] for x in data[s]] for s in sets}
    c0, g0, rows = {}, {}, []
    for s in sets:
        k, h, m, pl, pb = read(S, data[s], refs[s]); c0[s] = (pl, pb, m)
        save_readings(readings_root / s / "S.jsonl", k, h, refs[s])
        k, h, mg, _, _ = read(G, data[s], refs[s]); g0[s] = mg.cer
        save_readings(readings_root / s / "G.jsonl", k, h, refs[s])
        log(f"{s:28s} S CER {m.cer:.4f}   G CER {mg.cer:.4f}")
    for b in blocks:
        target = ({n: v for n, v in out_al.items() if not n.startswith("_")} if b.startswith("O_")
                  else {n: gt[n] for n in block_names(st, b)})
        orig = {n: st[n].copy() for n in target}
        for beta in betas:
            for n in target:
                S.set_tensor(n, orig[n] + beta * (target[n] - orig[n]))
            row = {"block": b, "beta": beta, "digest": S.digest(), "sets": {}}
            for s in sets:
                t = time.time()
                k, h, m, pl, pb = read(S, data[s], refs[s])
                save_readings(readings_root / s / f"{b}-b{beta}.jsonl", k, h, refs[s])
                boot = bootstrap_delta(c0[s][0], pl)
                blk = bootstrap_blocks(c0[s][1], pb)
                row["sets"][s] = {"cer": m.cer, "wer": m.wer, "collapsed": m.collapsed, "vs_S": boot,
                                  "blocks_vs_S": {g: v for g, v in blk.items() if v["ref"] >= 20 or max(v["errors_before"], v["errors_after"]) >= 10},
                                  "wall_s": round(time.time() - t, 1)}
                log(f"{b:5s} β={beta:<4} {s:28s} CER {m.cer:.4f}  Δ {100*boot['delta_cer']:+.2f} pt "
                    f"[{100*boot['ci95'][0]:+.2f}, {100*boot['ci95'][1]:+.2f}]")
            rows.append(row)
        for n, v in orig.items():
            S.set_tensor(n, v)
    return {"specific": specific.stem, "general": general.stem,
            "specific_sha256": sha256_file(specific), "general_sha256": sha256_file(general),
            "body_cosine": cos, "sets": sets, "own_set": sets[0], "betas": betas, "blocks": blocks,
            "output_rows_shared": out_al["_rows_shared"], "output_rows_new": out_al["_rows_new"],
            "c0": {s: {"cer": c0[s][2].cer, "wer": c0[s][2].wer, "collapsed": c0[s][2].collapsed} for s in sets},
            "general_cer": g0, "rows": rows, "when": time.strftime("%Y-%m-%d %H:%M")}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.exchange")
    ap.add_argument("--specific", required=True); ap.add_argument("--general", required=True)
    ap.add_argument("--sets", nargs="+", required=True, help="practice sets; the first is the specific model's own material")
    ap.add_argument("--blocks", nargs="*", default=None, help="default: every parameter block of the specific model")
    ap.add_argument("--betas", nargs="*", type=float, default=[0.5, 1.0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-unrelated", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    res = exchange(Path(a.specific), Path(a.general), a.sets, a.blocks, a.betas,
                   out.parent / "readings" / out.stem, a.allow_unrelated)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
