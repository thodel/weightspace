"""Three programmers on kraken session 2 (bern-combined-v1): Opus 5.5 (20261106-…-b10-1), Fable 5.1 (20261111-…-f1-b10-1),
Fable 5.2 (20261115-…-f2-b10-1). Paired deltas vs C0 and between programmers, line and page-clustered (A4),
practice and held-out, overall and per hand; cosines of the kept diffs. CPU.

    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/compare_combined_f2.py

Practice per-line errors: Fable runs from their last kept pass's readings; Opus rescored from its final.mlmodel.
Frozen models and the f1 readings are outside git; their paths are below. Writes runs/…-f2-b10-1/comparison.json.
"""
import json, os
from pathlib import Path
import numpy as np
from wsp.metrics import bootstrap_delta, per_line_edits
from wsp.data import load
from wsp.kraken_subject import KrakenSubject

REPO = Path(__file__).resolve().parent.parent
HOME = Path.home() / "Repo/weightspace"
F2 = REPO / "runs/20261115-bern-combined-v1-f2-b10-1"
F1 = REPO / "runs/20261111-bern-combined-v1-f1-b10-1"
F1_LOCAL = HOME / ".claude/worktrees/kraken-s2-fable-e2803a/runs/20261111-bern-combined-v1-f1-b10-1"  # readings, final.mlmodel
OP = REPO / "runs/20261106-bern-combined-v1-b10-1"
OP_LOCAL = HOME / "runs/20261106-bern-combined-v1-b10-1"  # final.mlmodel
SUBJ = HOME / "models/kraken/kraken-bern-combined-v1.mlmodel"

ho = load(REPO / "data/heldout/bern-eval.jsonl"); pr = load(REPO / "data/practice/bern-train-300.jsonl")
pages = lambda it: [x["key"].rsplit("/", 1)[0] for x in it]
hand = lambda it: ["H" if "Hertwig" in x["key"] else "S" for x in it]
cer = lambda r: sum(a for a, _ in r) / sum(c for _, c in r)
T = lambda r: [tuple(x) for x in r]


def last_kept_pass(run):
    p = 0; k = 1; last = None
    for l in open(run / "log.jsonl"):
        e = json.loads(l)
        if e["kind"] == "pass": p += 1; last = p
        if e["kind"] == "decision" and e.get("kept"): k = last
    return k


def pass_lines(readings, n):
    rd = [json.loads(l) for l in open(readings / f"pass_{n:03d}.jsonl")]
    return per_line_edits([r["text"] for r in rd], [x["text"] for x in pr])


out = {}


def rep(name, a, b, items):
    d = bootstrap_delta(a, b, clusters=pages(items))
    out[name] = {"cer_a": cer(a), "cer_b": cer(b), **{k: d[k] for k in ("delta_cer", "ci95", "ci95_cluster")}}
    p = lambda v: f"{100 * v:+.2f}"
    print(f"{name:34s} {cer(a):.4f} -> {cer(b):.4f}  Δ {p(d['delta_cer'])}  line [{p(d['ci95'][0])}, {p(d['ci95'][1])}]  page [{p(d['ci95_cluster'][0])}, {p(d['ci95_cluster'][1])}]")


def by_hand(prefix, a, b, items):
    rep(prefix, a, b, items)
    H = hand(items)
    for k, nm in (("H", "Hertwig"), ("S", "Schlussred")):
        idx = [i for i, h in enumerate(H) if h == k]
        rep(f"{prefix} {nm}", [a[i] for i in idx], [b[i] for i in idx], [items[i] for i in idx])


# held-out
c0h = T(next(m for m in json.load(open(REPO / "docs/baselines/heldout-bern-eval.json"))["models"] if m["model"] == "kraken-bern-combined-v1")["per_line"])
hl = {nm: T(json.load(open(r / "summary.json"))["heldout"]["per_line"]) for nm, r in (("opus", OP), ("f1", F1), ("f2", F2))}
print("== held-out (bern-eval, 545 lines, 14 pages)")
for nm in ("opus", "f1", "f2"):
    by_hand(f"heldout {nm} vs C0", c0h, hl[nm], ho)
by_hand("heldout f2 vs f1", hl["f1"], hl["f2"], ho)
by_hand("heldout f2 vs opus", hl["opus"], hl["f2"], ho)

# practice
print("== practice (bern-train-300, training side)")
c0p = pass_lines(F2 / "readings", 1)
pl = {"f2": pass_lines(F2 / "readings", last_kept_pass(F2)), "f1": pass_lines(F1_LOCAL / "readings", last_kept_pass(F1))}
op_model = KrakenSubject(OP_LOCAL / "final.mlmodel")
assert op_model.digest() == json.load(open(OP / "summary.json"))["final_digest"]
pl["opus"] = per_line_edits([p.text for p in op_model.forward(pr)], [x["text"] for x in pr])
for nm in ("opus", "f1", "f2"):
    by_hand(f"practice {nm} vs C0", c0p, pl[nm], pr)
by_hand("practice f2 vs f1", pl["f1"], pl["f2"], pr)

# kept diffs: cosine and norms
print("== kept diffs (final − subject), all tensors, output rows by index (same codec)")
s = KrakenSubject(SUBJ).tensors()
diffs = {}
for nm, path in (("opus", OP_LOCAL / "final.mlmodel"), ("f1", F1_LOCAL / "final.mlmodel"), ("f2", F2 / "final.mlmodel")):
    t = KrakenSubject(path).tensors()
    diffs[nm] = np.concatenate([(t[n].astype(np.float64) - s[n].astype(np.float64)).ravel() for n in sorted(s)])
cos = {}
for a, b in (("f2", "f1"), ("f2", "opus"), ("f1", "opus")):
    c = float(diffs[a] @ diffs[b] / (np.linalg.norm(diffs[a]) * np.linalg.norm(diffs[b])))
    cos[f"{a}·{b}"] = c
    print(f"cos({a}, {b}) = {c:+.3f}")
norms = {k: float(np.linalg.norm(v)) for k, v in diffs.items()}
print("norms", {k: round(v, 2) for k, v in norms.items()})
out["kept_diff_cosine"] = cos; out["kept_diff_norm"] = norms
json.dump(out, open(F2 / "comparison.json", "w"), indent=1)
