#!/usr/bin/env python3
"""Stage 4, subject A: shams (arm S) and margins (arm M) of the kept edits of a session, on practice and on the DISCOVERY half
(docs/TROCR_STAGE4_PREREG.md section 6, amendment 4). The confirmation half is not read.

    ... trocr_stage4_A_followup.py RUN_DIR [RUN_DIR ...]      # RUN_DIR holds final.trocr
The 'edit' of a session is its net change, final state minus the subject, on the tensors that differ. Five shams (seed 20261119, wsp.stage4.sham_deltas):
same tensors, i.i.d. Gaussian, same Frobenius norm per tensor. Margins: the net change at 0.5 and 2 times its size."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trocr_stage4_discovery as DS
import trocr_stage4_grid as G
from wsp import data, metrics, stage4
from wsp.trocr_subject import TrocrSubject

REPO = G.REPO
OUT = REPO / "docs/trocr/stage4_A_followup.jsonl"
SEED = 20261119


def main(runs):
    prac = data.load(str(G.PRACTICE))
    pcl = [metrics.page_of(r["key"]) for r in prac]
    pbase = [tuple(x) for x in next(r["edits"] for r in DS.jl(G.OUT) if r["kind"] == "baseline" and r["pair"] == "A<-D")]
    split = json.loads(DS.SPLIT.read_text())
    disc = set(split["objects"]["discovery"])
    drows = [r for r in data.load(str(DS.HELD)) if r["doc"] in disc]
    dcl = [metrics.page_of(r["key"]) for r in drows]
    dbase = [tuple(x) for x in next(r["edits"] for r in DS.jl(DS.DISC) if r["subject"] == "A" and r["what"] == "baseline")]
    sub = TrocrSubject(G.MODELS["A"], device="cuda:0", batch=8)
    orig = {n: v.copy() for n, v in sub.tensors().items()}
    done = {(r["run"], r["what"]) for r in DS.jl(OUT)}

    def read(rows, base, cl):
        ed = G.edits(G.read(sub, rows), rows)
        b = metrics.bootstrap_delta(base, ed, clusters=cl, resamples=2000)
        return {"cer": sum(e for e, _ in ed) / sum(c for _, c in ed), "delta_cer": b["delta_cer"], "ci95_cluster": b["ci95_cluster"], "edits": ed}

    for rd in runs:
        run = Path(rd)
        final = TrocrSubject(run / "final.trocr", device="cpu", check_card=False).tensors()
        delta = stage4.edit_delta(orig, final)
        variants = {f"sham{i}": d for i, d in enumerate(stage4.sham_deltas(delta, SEED, 5))}
        variants.update({f"margin{f}": {n: delta[n] * f for n in delta} for f in stage4.MARGIN_FACTORS})
        for what, d in variants.items():
            if (run.name, what) in done:
                continue
            names = list(d)
            for n in names:
                sub.set_tensor(n, (orig[n].astype("float64") + d[n]).astype("float32"))
            rec = {"run": run.name, "what": what, "tensors": len(names), "norm": float(sum((d[n] ** 2).sum() for n in names) ** 0.5)}
            rec["practice"] = read(prac, pbase, pcl)
            rec["discovery"] = read(drows, dbase, dcl)
            for n in names:
                sub.set_tensor(n, orig[n])
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(run.name, what, "practice", round(100 * rec["practice"]["delta_cer"], 2), "discovery", round(100 * rec["discovery"]["delta_cer"], 2),
                  [round(100 * x, 2) for x in rec["discovery"]["ci95_cluster"]], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
