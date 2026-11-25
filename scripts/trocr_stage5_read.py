#!/usr/bin/env python3
"""Stage 5 readings (docs/TROCR_STAGE5_PREREG.md, sections 4 and 6). One reading of one model, appended to docs/trocr/stage5_reads.jsonl.

    PYTHONPATH=. ~/atr-cache/trocr/venv/bin/python scripts/trocr_stage5_read.py SUBJECT NAME {practice|discovery} (--dir MODEL_DIR | --variant SPEC) [--device cuda:0]

SUBJECT is A or B. ``--dir``: a saved hub-format model (a C4 final or a zero-step copy). ``--variant`` builds a model in memory from the subject and other
checkpoints: ``sibpair:SIB:ALPHA`` (subject + ALPHA*(SIB - subject), SIB in s1..s5,s42), ``sibmean:ALPHA`` (toward the uniform mean of the six siblings),
``sibavg7`` (uniform average of the recorded model and the six siblings), ``toward:KEY:ALPHA`` (subject toward the model KEY in {A,D,I}; the ceiling
comparators of B and the arm-I grid). Reads the practice set (1,194 lines) or the discovery half of blb-eval-hand (never the confirmation half; there is
no option for it here). Each row: subject, name, what, cer, delta against the unedited subject on the same lines, page-clustered 95 % interval, edits.
The practice baseline of the subject is the cached one of docs/trocr/stage4_grid.jsonl, the discovery baseline that of stage4_discovery.jsonl.
A reading on cuda waits for wsp.card.ready (>= 12 GB free, no trainer job waiting); card 1 is never used."""
import argparse
import json
import sys
import time
from pathlib import Path

from wsp import data, metrics, stage5
from wsp.trocr_subject import TrocrSubject

REPO = Path(__file__).resolve().parent.parent
AC = Path.home() / "atr-cache/trocr"
MODELS = {"A": AC / "dh-unibe__trocr-german-kurrent-xvi-xix-from-kurrent-v2", "B": AC / "dh-unibe__trocr-kurrent",
          "D": AC / "dh-unibe__trocr-kurrent-XVI-XVII", "I": AC / "microsoft__trocr-base-handwritten"}
SIBS = {k: AC / "seed-null" / k for k in ("s1", "s2", "s3", "s4", "s5", "s42")}
PRACTICE = REPO / "data/practice/trocr-blb-train-1200-seen.jsonl"
OUT = REPO / "docs/trocr/stage5_reads.jsonl"
GRID = REPO / "docs/trocr/stage4_grid.jsonl"
DISC = REPO / "docs/trocr/stage4_discovery.jsonl"
PAIR = {"A": "A<-D", "B": "B<-I"}


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []


def lines(what):
    if what == "practice":
        return data.load(str(PRACTICE))
    split = json.loads((REPO / "data/heldout/blb-eval-hand.split.json").read_text())
    disc = set(split["objects"]["discovery"])
    rows = [r for r in data.load(str(REPO / "data/heldout/blb-eval-hand.jsonl")) if r["doc"] in disc]
    assert len(rows) == split["lines"]["discovery"], (len(rows), split["lines"])
    return rows


def baseline(subject, what):
    if what == "practice":
        return [tuple(x) for x in next(r for r in jl(GRID) if r["kind"] == "baseline" and r["pair"] == PAIR[subject])["edits"]]
    return [tuple(x) for x in next(r for r in jl(DISC) if r["subject"] == subject and r["what"] == "baseline")["edits"]]


def build(subject, variant, device):
    """A TrocrSubject holding the variant: the subject's tokenizer and config with the tensors of the variant."""
    sub = TrocrSubject(MODELS[subject], device=device, batch=8)
    kind, *rest = variant.split(":")
    mine = {n: v.copy() for n, v in sub.tensors().items()}
    def other(path):
        o = TrocrSubject(path, device="cpu", batch=8)
        if o.tokenizer.get_vocab() != sub.tokenizer.get_vocab():
            raise ValueError(f"{path}: tokenizer vocabulary differs from the subject's")
        return o.tensors()
    if kind == "sibpair":
        new = stage5.toward(mine, other(SIBS[rest[0]]), float(rest[1]))
    elif kind == "toward":
        new = stage5.toward(mine, other(MODELS[rest[0]]), float(rest[1]))
    elif kind in ("sibmean", "sibavg7"):
        sibs = [other(p) for p in SIBS.values()]
        if kind == "sibmean":
            mean, _ = stage5.average(sibs)
            new = stage5.toward(mine, mean, float(rest[0]))
        else:
            new, _ = stage5.average([mine] + sibs)
    else:
        raise SystemExit(f"unknown variant {variant}")
    for n, v in new.items():
        sub.set_tensor(n, v)
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", choices=["A", "B"])
    ap.add_argument("name")
    ap.add_argument("what", choices=["practice", "discovery"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir")
    g.add_argument("--variant")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.device != "cpu" and not a.device.endswith(":0"):
        raise SystemExit("card 0 only")
    rows = lines(a.what)
    clusters = [metrics.page_of(r["key"]) for r in rows]
    base = baseline(a.subject, a.what)
    t0 = time.time()
    sub = TrocrSubject(a.dir, device=a.device, batch=8) if a.dir else build(a.subject, a.variant, a.device)
    while True:
        try:
            preds = sub.forward(rows)
            break
        except RuntimeError as e:
            if "no pass" not in str(e):
                raise
            print("card not ready:", e, flush=True)
            time.sleep(120)
    ed = metrics.per_line_edits([p.text for p in preds], [r["text"] for r in rows])
    b = metrics.bootstrap_delta(base, ed, clusters=clusters, resamples=2000 if a.what == "discovery" else 1000)
    cer = sum(e for e, _ in ed) / sum(c for _, c in ed)
    rec = {"subject": a.subject, "name": a.name, "what": a.what, "source": a.dir or a.variant, "lines": len(rows), "cer": cer, "delta_cer": b["delta_cer"],
           "ci95_cluster": b["ci95_cluster"], "seconds": round(time.time() - t0, 1), "edits": ed}
    with OUT.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(a.subject, a.name, a.what, "CER", round(100 * cer, 2), "delta", round(100 * b["delta_cer"], 2), [round(100 * x, 2) for x in b["ci95_cluster"]], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
