#!/usr/bin/env python3
"""Stage 4, arm I and the discovery-half reads (docs/TROCR_STAGE4_PREREG.md section 6, amendments 1 and 2).

    ... trocr_stage4_discovery.py armI        # global interpolation toward the donor, alpha 0.1..0.9, practice set, both subjects
    ... trocr_stage4_discovery.py discovery   # the top six grid cells per subject (ranking of amendment 2) and the best alpha of arm I, DISCOVERY half only

The confirmation half is never read here: the object list of the split file is the only thing used, and the rows of the confirmation objects are dropped
before any model sees them."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trocr_stage4_grid as G
from wsp import data, metrics, stage4
from wsp.trocr_subject import TrocrSubject

REPO = G.REPO
GRID = REPO / "docs/trocr/stage4_grid.jsonl"
ARMI = REPO / "docs/trocr/stage4_armI.jsonl"
DISC = REPO / "docs/trocr/stage4_discovery.jsonl"
SPLIT = REPO / "data/heldout/blb-eval-hand.split.json"
HELD = REPO / "data/heldout/blb-eval-hand.jsonl"
SUBJECTS = {"A": "A<-D", "B": "B<-I"}
TOP = 6


def jl(p):
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def shared(sub, don):
    m, d = sub.tensors(), don.tensors()
    return [n for n in m if n in d and m[n].shape == d[n].shape and not n.startswith("enc.pooler")]


def interp(sub, don, orig, names, alpha):
    t = don.tensors()
    for n in names:
        sub.set_tensor(n, ((1 - alpha) * orig[n].astype(float) + alpha * t[n].astype(float)).astype("float32"))


def restore(sub, orig, names):
    for n in names:
        sub.set_tensor(n, orig[n])


def ranking():
    """Amendment 2: cells of the subject's own direction (A<-D, B<-I), both sizes pooled, ordered by the practice delta (lowest first); ties by the lower f."""
    out = {}
    for s, pair in SUBJECTS.items():
        C = [r for r in jl(GRID) if r["kind"] == "cell" and r["pair"] == pair]
        out[s] = sorted(C, key=lambda r: (r["delta_cer"], r["f"]))[:TOP]
    return out


def arm_i():
    rows = data.load(str(G.PRACTICE))
    clusters = [metrics.page_of(r["key"]) for r in rows]
    for s, pair in SUBJECTS.items():
        rk, sk = pair.split("<-")
        base = [tuple(x) for x in next(r["edits"] for r in jl(GRID) if r["kind"] == "baseline" and r["pair"] == pair)]
        have = {r["alpha"] for r in jl(ARMI) if r["pair"] == pair}
        sub, don = TrocrSubject(G.MODELS[rk], device="cuda:0", batch=8), TrocrSubject(G.MODELS[sk], device="cpu", batch=8)
        orig = {n: v.copy() for n, v in sub.tensors().items()}
        names = shared(sub, don)
        for a in stage4.ALPHAS:
            if a in have:
                continue
            interp(sub, don, orig, names, a)
            ed = G.edits(G.read(sub, rows), rows)
            restore(sub, orig, names)
            b = metrics.bootstrap_delta(base, ed, clusters=clusters, resamples=2000)
            cer = sum(e for e, _ in ed) / sum(c for _, c in ed)
            with ARMI.open("a") as f:
                f.write(json.dumps({"pair": pair, "alpha": a, "tensors": len(names), "cer": cer, "delta_cer": b["delta_cer"], "ci95_cluster": b["ci95_cluster"], "edits": ed}) + "\n")
            print(pair, "alpha", a, "CER", round(100 * cer, 2), "delta", round(100 * b["delta_cer"], 2), [round(100 * x, 2) for x in b["ci95_cluster"]], flush=True)


def best_alpha(pair):
    R = [r for r in jl(ARMI) if r["pair"] == pair]
    return min(R, key=lambda r: (r["cer"], r["alpha"]))


def discovery():
    split = json.loads(SPLIT.read_text())
    disc = set(split["objects"]["discovery"])
    rows = [r for r in data.load(str(HELD)) if r["doc"] in disc]
    assert len(rows) == split["lines"]["discovery"], (len(rows), split["lines"])
    clusters = [metrics.page_of(r["key"]) for r in rows]
    done = {(r["subject"], r["what"]) for r in jl(DISC)}
    rk_by = ranking()
    for s, pair in SUBJECTS.items():
        rk, sk = pair.split("<-")
        sub, don = TrocrSubject(G.MODELS[rk], device="cuda:0", batch=8), TrocrSubject(G.MODELS[sk], device="cpu", batch=8)
        orig = {n: v.copy() for n, v in sub.tensors().items()}
        base = [x for x in jl(DISC) if x["subject"] == s and x["what"] == "baseline"]
        if base:
            base = [tuple(x) for x in base[0]["edits"]]
        else:
            base = G.edits(G.read(sub, rows), rows)
            with DISC.open("a") as f:
                f.write(json.dumps({"subject": s, "what": "baseline", "cer": sum(e for e, _ in base) / sum(c for _, c in base), "lines": len(rows), "edits": base}) + "\n")
            print(s, "baseline discovery CER", round(100 * sum(e for e, _ in base) / sum(c for _, c in base), 2), flush=True)
        theirs = don.tensors()
        todo = [(f"cell:{r['cell']}@{r['f']}", r) for r in rk_by[s]]
        for what, r in todo + [("armI", best_alpha(pair))]:
            if (s, what) in done:
                continue
            if what == "armI":
                names = shared(sub, don)
                interp(sub, don, orig, names, r["alpha"])
            else:
                _, names = next((c, v) for c, v in G.cells(sorted(orig)).items() if c == r["cell"]), G.cells(sorted(orig))[r["cell"]]
                for n in names:
                    sub.set_tensor(n, orig[n] + r["f"] * (theirs[n] - orig[n]))
            ed = G.edits(G.read(sub, rows), rows)
            restore(sub, orig, names)
            b = metrics.bootstrap_delta(base, ed, clusters=clusters, resamples=2000)
            cer = sum(e for e, _ in ed) / sum(c for _, c in ed)
            with DISC.open("a") as f:
                f.write(json.dumps({"subject": s, "what": what, "practice_delta": r["delta_cer"], "practice_ci95_cluster": r["ci95_cluster"], "alpha": r.get("alpha"), "cer": cer,
                                    "delta_cer": b["delta_cer"], "ci95_cluster": b["ci95_cluster"], "clusters": b["clusters"], "edits": ed}) + "\n")
            print(s, what, "discovery CER", round(100 * cer, 2), "delta", round(100 * b["delta_cer"], 2), [round(100 * x, 2) for x in b["ci95_cluster"]], flush=True)


if __name__ == "__main__":
    {"armI": arm_i, "discovery": discovery}[sys.argv[1]]()
