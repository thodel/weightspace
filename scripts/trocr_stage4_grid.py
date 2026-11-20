#!/usr/bin/env python3
"""Stage 4, exchange grid on the practice set (docs/TROCR_STAGE4_PREREG.md section 6; plan section 5). No programmer, no held-out read.

    PYTHONPATH=. ~/atr-cache/trocr/venv/bin/python scripts/trocr_stage4_grid.py [--pairs A<-D B<-I ...] [--device cuda:0]

For each directed pair RECIPIENT<-SOURCE the recipient reads the 1,194-line practice set unedited (baseline), then once per cell: the cell's tensors become
``recipient + f * (source - recipient)`` for f in {1.0, 0.5}, everything else stays the recipient's. Cells: whole encoder, whole decoder, each of the 12 encoder
and 12 decoder layers, the tied embedding with its norm, all layer norms (28). Shape of every exchanged tensor and the tokenizer vocabulary are asserted
(a cell with a shape or vocabulary mismatch is skipped and recorded). Each reading is compared, paired per line with the recipient's baseline, with the
page-clustered 95 % interval (wsp.metrics.bootstrap_delta). Results append to docs/trocr/stage4_grid.jsonl; a finished (pair, cell, f) is skipped on a rerun.
The ranking and the held-out reads of the top cells are NOT done here."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from wsp import data, metrics
from wsp.trocr_subject import TrocrSubject

REPO = Path(__file__).resolve().parent.parent
AC = Path.home() / "atr-cache/trocr"
MODELS = {"A": AC / "dh-unibe__trocr-german-kurrent-xvi-xix-from-kurrent-v2", "D": AC / "dh-unibe__trocr-kurrent-XVI-XVII",
          "B": AC / "dh-unibe__trocr-kurrent", "I": AC / "microsoft__trocr-base-handwritten"}
PAIRS = ["A<-D", "B<-I", "D<-A", "I<-B"]                  # subject first, then the reverse directions
PRACTICE = REPO / "data/practice/trocr-blb-train-1200-seen.jsonl"
OUT = REPO / "docs/trocr/stage4_grid.jsonl"
FACTORS = (1.0, 0.5)


def cells(names: list[str]) -> dict[str, list[str]]:
    enc = [n for n in names if n.startswith("enc.") and not n.startswith("enc.pooler")]
    dec = [n for n in names if n.startswith("dec.")]
    out = {"enc_all": enc, "dec_all": dec}
    for i in range(12):
        out[f"enc.{i:02d}"] = [n for n in names if n.startswith(f"enc.{i:02d}.")]
    for i in range(12):
        out[f"dec.{i:02d}"] = [n for n in names if n.startswith(f"dec.{i:02d}.")]
    out["embed"] = [n for n in names if n == "dec.embed.tokens" or n.startswith("dec.embed.norm.")]
    out["norms"] = [n for n in names if ".norm" in n or n.startswith("dec.embed.norm.")]
    return out


def done_keys() -> set:
    if not OUT.exists():
        return set()
    return {(r["pair"], r["cell"], r["f"]) for r in map(json.loads, OUT.read_text().splitlines()) if r.get("kind") == "cell"}


def read(sub, rows):
    import time as _t
    while True:
        try:
            return sub.forward(rows)
        except RuntimeError as e:                          # wsp.card.ready refused: wait, do not force the card
            if "no pass" not in str(e):
                raise
            print("card not ready:", e, flush=True)
            _t.sleep(120)


def edits(preds, rows):
    return metrics.per_line_edits([p.text for p in preds], [r["text"] for r in rows])


def log(rec):
    with OUT.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=PAIRS)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    rows = data.load(str(PRACTICE))
    clusters = [metrics.page_of(r["key"]) for r in rows]
    done = done_keys()
    for pair in a.pairs:
        rk, sk = pair.split("<-")
        t0 = time.time()
        sub = TrocrSubject(MODELS[rk], device=a.device, batch=8)
        src = TrocrSubject(MODELS[sk], device="cpu", batch=8)
        mine, theirs = sub.tensors(), src.tensors()
        orig = {n: v.copy() for n, v in mine.items()}
        same_vocab = sub.tokenizer.get_vocab() == src.tokenizer.get_vocab()
        grid = cells(sorted(mine))
        prior = [r for r in (map(json.loads, OUT.read_text().splitlines()) if OUT.exists() else []) if r["kind"] == "baseline" and r["pair"] == pair]
        if prior:
            base = [tuple(x) for x in prior[0]["edits"]]
        else:
            t1 = time.time()
            base = edits(read(sub, rows), rows)
            cer = sum(e for e, _ in base) / sum(c for _, c in base)
            log({"kind": "baseline", "pair": pair, "recipient": rk, "cer": cer, "seconds": round(time.time() - t1, 1), "edits": base})
            print(pair, "baseline CER", round(100 * cer, 2), "in", round(time.time() - t1), "s", flush=True)
        for cell, names in grid.items():
            bad = [n for n in names if n not in theirs or theirs[n].shape != mine[n].shape]
            if not names or bad or (cell in ("embed", "dec_all") and not same_vocab):
                if not any(r.get("pair") == pair and r.get("cell") == cell and r["kind"] == "skip" for r in (map(json.loads, OUT.read_text().splitlines()) if OUT.exists() else [])):
                    log({"kind": "skip", "pair": pair, "cell": cell, "why": f"no tensors" if not names else f"shape/name mismatch {bad[:3]}" if bad else "tokenizer vocabularies differ"})
                continue
            for f in FACTORS:
                if (pair, cell, f) in done:
                    continue
                for n in names:
                    sub.set_tensor(n, orig[n] + f * (theirs[n] - orig[n]))
                t1 = time.time()
                ed = edits(read(sub, rows), rows)
                for n in names:
                    sub.set_tensor(n, orig[n])
                b = metrics.bootstrap_delta(base, ed, clusters=clusters, resamples=1000)
                cer = sum(e for e, _ in ed) / sum(c for _, c in ed)
                log({"kind": "cell", "pair": pair, "cell": cell, "f": f, "tensors": len(names), "cer": cer, "delta_cer": b["delta_cer"],
                     "ci95_cluster": b["ci95_cluster"], "seconds": round(time.time() - t1, 1), "edits": ed})
                print(pair, cell, f, "CER", round(100 * cer, 2), "delta", round(100 * b["delta_cer"], 2), "cluster CI", [round(100 * x, 2) for x in b["ci95_cluster"]],
                      round(time.time() - t1), "s", flush=True)
        print(pair, "done in", round((time.time() - t0) / 60), "min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
