#!/usr/bin/env python3
"""Stage 4, subject B: the one confirmation-half read approved by Tobias on 25.11.2026 (docs/TROCR_STAGE4_PREREG.md, amendment 6).

    ... trocr_stage4_confirm.py FINAL_DIR      # FINAL_DIR = final.trocr of a B session that ended in dec.NN.cross.o x 1.15 (P2, P3-Fable, R share one digest)

Reads on the 12 CONFIRMATION objects (1,125 lines of blb-eval-hand), each exactly once: the unedited subject, the frozen edit, five shams of its net
change (same tensors, same Frobenius norm, seed 20261119, wsp.stage4.sham_deltas) and arm I at alpha 0.3 (the B best on practice). Nothing else is read."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trocr_stage4_discovery as DS
import trocr_stage4_grid as G
from wsp import data, metrics, stage4
from wsp.trocr_subject import TrocrSubject

OUT = G.REPO / "docs/trocr/stage4_B_confirmation.jsonl"
SEED, ALPHA = 20261119, 0.3


def main(final_dir):
    split = json.loads(DS.SPLIT.read_text())
    conf = set(split["objects"]["confirmation"])
    rows = [r for r in data.load(str(DS.HELD)) if r["doc"] in conf]
    assert len(rows) == split["lines"]["confirmation"], (len(rows), split["lines"])
    cl = [metrics.page_of(r["key"]) for r in rows]
    sub = TrocrSubject(G.MODELS["B"], device="cuda:0", batch=8)
    don = TrocrSubject(G.MODELS["I"], device="cpu", batch=8, check_card=False)
    orig = {n: v.copy() for n, v in sub.tensors().items()}
    final = TrocrSubject(Path(final_dir), device="cpu", check_card=False).tensors()
    delta = stage4.edit_delta(orig, final)
    assert delta and all(".cross.o." in n for n in delta) and len(delta) == 24, sorted(delta)[:5]
    done = {r["what"]: r for r in DS.jl(OUT)}

    def read(what, apply, undo):
        if what in done:
            return done[what]
        apply()
        ed = G.edits(G.read(sub, rows), rows)
        undo()
        rec = {"what": what, "lines": len(rows), "cer": sum(e for e, _ in ed) / sum(c for _, c in ed), "edits": ed}
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(what, "confirmation CER", round(100 * rec["cer"], 2), flush=True)
        return rec

    def setd(d):
        for n in d:
            sub.set_tensor(n, (orig[n].astype("float64") + d[n]).astype("float32"))

    def back(names):
        for n in names:
            sub.set_tensor(n, orig[n])

    names_i = DS.shared(sub, don)
    base = read("baseline", lambda: None, lambda: None)
    reads = [("edit", lambda: setd(delta), lambda: back(delta))]
    reads += [(f"sham{i}", (lambda d=d: setd(d)), (lambda d=d: back(d))) for i, d in enumerate(stage4.sham_deltas(delta, SEED, 5))]
    reads += [("armI", lambda: DS.interp(sub, don, orig, names_i, ALPHA), lambda: DS.restore(sub, orig, names_i))]
    res = {"baseline": base}
    for w, a, u in reads:
        res[w] = read(w, a, u)
    b0 = [tuple(x) for x in base["edits"]]
    for w, r in res.items():
        if w == "baseline":
            continue
        d = metrics.bootstrap_delta(b0, [tuple(x) for x in r["edits"]], clusters=cl, resamples=2000)
        print(w, "delta", round(100 * d["delta_cer"], 2), [round(100 * x, 2) for x in d["ci95_cluster"]])
    ai = [tuple(x) for x in res["armI"]["edits"]]
    d = metrics.bootstrap_delta(ai, [tuple(x) for x in res["edit"]["edits"]], clusters=cl, resamples=2000)
    print("edit vs armI", round(100 * d["delta_cer"], 2), [round(100 * x, 2) for x in d["ci95_cluster"]])


if __name__ == "__main__":
    main(sys.argv[1])
