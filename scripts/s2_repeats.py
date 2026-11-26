"""Run-to-run variation on kraken session 2 (bern-combined-v1): four runs of Claude Fable 5.2
(b10-1 = #44, and the repeats b10-2..4) against the five programmers (one run each).
Pre-registration: docs/S2_REPEATS_PREREG.md. CPU.

    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/s2_repeats.py            # all runs whose folder has a held-out score
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/s2_repeats.py --across   # the five programmers only

Per run: held-out Δ vs C0 (line and page-clustered, overall and per hand), the kept edits, the step toward the
parent (least-squares fit of the kept diff on the parent and sibling directions, output rows by character, the blank
bias element excluded and reported alone), and whether the page interval lies below 0. Across the four Fable 5.2 runs:
range and SD of held-out Δ against the same for the five programmers. Writes docs/s2_repeats.json.
Frozen models are outside git; their paths are below.
"""
import json, sys
from pathlib import Path
import numpy as np
from wsp.metrics import bootstrap_delta, per_line_edits
from wsp.data import load
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map

REPO = Path(__file__).resolve().parent.parent
HOME = Path.home() / "Repo/weightspace"
WT = HOME / ".claude/worktrees"
MODELS = HOME / "models/kraken"
SUBJ, PARENT, SIB = (MODELS / f"kraken-bern-{m}.mlmodel" for m in ("combined-v1", "hertwig-v2", "schlussredaktion-v1"))

# label -> (run folder in git, folder that holds final.mlmodel and readings/)
ACROSS = {
    "Opus 5.5": ("20261106-bern-combined-v1-b10-1", HOME / "runs/20261106-bern-combined-v1-b10-1"),
    "Fable 5.1": ("20261111-bern-combined-v1-f1-b10-1", WT / "kraken-s2-fable-e2803a/runs/20261111-bern-combined-v1-f1-b10-1"),
    "Fable 5.2": ("20261115-bern-combined-v1-f2-b10-1", WT / "kraken-s2-fable52-923f0b/runs/20261115-bern-combined-v1-f2-b10-1"),
    "Gemini 4 Argon": ("20261119-bern-combined-v1-g1-b10-1", HOME / "runs/20261119-bern-combined-v1-g1-b10-1"),
    "GPT-6.2 Astra": ("20261119-bern-combined-v1-o1-b10-1", HOME / "runs/20261119-bern-combined-v1-o1-b10-1"),
}
REPEAT_DATE = "20261124"
WITHIN = {"f2 b10-1": ACROSS["Fable 5.2"]}
for k in (2, 3, 4):
    name = f"{REPEAT_DATE}-bern-combined-v1-f2-b10-{k}"
    WITHIN[f"f2 b10-{k}"] = (name, REPO / "runs" / name)

ho = load(REPO / "data/heldout/bern-eval.jsonl")
pr = load(REPO / "data/practice/bern-train-300.jsonl")
pages = lambda it: [x["key"].rsplit("/", 1)[0] for x in it]
hand = lambda it: ["H" if "Hertwig" in x["key"] else "S" for x in it]
T = lambda r: [tuple(x) for x in r]
c0h = T(next(m for m in json.load(open(REPO / "docs/baselines/heldout-bern-eval.json"))["models"] if m["model"] == "kraken-bern-combined-v1")["per_line"])


def delta(a, b, items):
    d = bootstrap_delta(a, b, clusters=pages(items))
    return {"delta": 100 * d["delta_cer"], "line": [100 * v for v in d["ci95"]], "page": [100 * v for v in d["ci95_cluster"]]}


def by_hand(a, b, items):
    out = {"all": delta(a, b, items)}
    H = hand(items)
    for k, nm in (("H", "Hertwig"), ("S", "Schlussredaktion")):
        idx = [i for i, h in enumerate(H) if h == k]
        out[nm] = delta([a[i] for i in idx], [b[i] for i in idx], [items[i] for i in idx])
    return out


def kept_edits(run):
    """(pass label, op, tensors) of every kept decision, in order; and passes used."""
    log = [json.loads(l) for l in open(run / "log.jsonl")]
    last_edit, last_pass, out, used, n = None, None, [], 0, 0
    for e in log:
        if e["kind"] == "edit": last_edit = e
        if e["kind"] == "pass": last_pass = e["label"]; used = e["budget_used"]
        if e["kind"] == "decision" and e.get("kept") and last_edit is not None:
            out.append({"pass": last_pass, "op": last_edit["op"], "n_tensors": len(last_edit["names"])})
        if e["kind"] == "edit": n += 1
    return out, used, n


def last_kept_pass(run):
    p = 0; k = 1; last = None
    for l in open(run / "log.jsonl"):
        e = json.loads(l)
        if e["kind"] == "pass": p += 1; last = p
        if e["kind"] == "decision" and e.get("kept"): k = last
    return k


_dirs = None


def directions():
    """Subject tensors and the parent and sibling directions in the subject's layout (rows by character).
    Output rows the donor lacks are masked out of every vector; the blank bias element is masked out too."""
    global _dirs
    if _dirs is None:
        S = KrakenSubject(SUBJ); s = {n: v.astype(np.float64) for n, v in S.tensors().items()}
        cs = S.codec_map(); out = {}
        for nm, path in (("P", PARENT), ("S", SIB)):
            D = KrakenSubject(path); t = D.tensors(); rm = codec_row_map(cs, D.codec_map())
            d, mask = {}, {}
            for n, v in s.items():
                if n.startswith("O_18.lin."):
                    g = np.zeros_like(v); m = np.zeros(v.shape, bool)
                    for ia, ib in rm.items():
                        g[ia] = t[n][ib]; m[ia] = True
                    if n == "O_18.lin.bias":
                        m[0] = False
                    d[n] = np.where(m, g - v, 0.0); mask[n] = m
                else:
                    d[n] = t[n].astype(np.float64) - v; mask[n] = np.ones(v.shape, bool)
            out[nm] = (d, mask)
        # one common mask: rows both donors have
        mask = {n: out["P"][1][n] & out["S"][1][n] for n in s}
        _dirs = (s, mask, out["P"][0], out["S"][0])
    return _dirs


def step(final_path):
    s, mask, dP, dS = directions()
    f = {n: v.astype(np.float64) for n, v in KrakenSubject(final_path).tensors().items()}
    names = sorted(s)
    vec = lambda D: np.concatenate([D[n][mask[n]].ravel() for n in names])
    d = vec({n: f[n] - s[n] for n in names}); p = vec(dP); q = vec(dS)
    blank = float(f["O_18.lin.bias"][0] - s["O_18.lin.bias"][0])
    if not np.any(d):
        return {"a_parent": 0.0, "b_sibling": 0.0, "r2": None, "cos_parent": None, "norm": 0.0, "blank_step": blank, "parent_step": False}
    A = np.stack([p, q], 1)
    (a, b), *_ = np.linalg.lstsq(A, d, rcond=None)
    r2 = 1 - float(np.sum((d - A @ np.array([a, b])) ** 2) / np.sum(d ** 2))
    cosP = float(d @ p / (np.linalg.norm(d) * np.linalg.norm(p)))
    return {"a_parent": float(a), "b_sibling": float(b), "r2": r2, "cos_parent": cosP,
            "alpha_parent_only": float(d @ p / (p @ p)), "norm": float(np.linalg.norm(d)), "norm_parent_dir": float(np.linalg.norm(p)),
            "blank_step": blank, "parent_step": bool(a >= 0.05 and r2 >= 0.5)}


def run_row(label, name, local):
    git = REPO / "runs" / name
    summ = json.load(open(git / "summary.json"))
    final = local / "final.mlmodel"
    assert KrakenSubject(final).digest() == summ["final_digest"], label
    kept, used, n_edits = kept_edits(git)
    row = {"label": label, "run": name, "passes": used, "edits": n_edits, "kept": kept,
           "heldout": by_hand(c0h, T(summ["heldout"]["per_line"]), ho),
           "heldout_cer": summ["heldout"]["metrics"]["cer"], "heldout_collapsed": summ["heldout"]["metrics"]["collapsed"],
           "step": step(final)}
    row["page_below_0"] = bool(row["heldout"]["all"]["page"][1] < 0)
    rd = local / "readings"
    if rd.exists():
        rl = lambda n: per_line_edits([json.loads(l)["text"] for l in open(rd / f"pass_{n:03d}.jsonl")], [x["text"] for x in pr])
        row["practice"] = by_hand(rl(1), rl(last_kept_pass(git)), pr)
    return row


def spread(rows):
    v = np.array([r["heldout"]["all"]["delta"] for r in rows])
    return {"n": len(v), "values": v.tolist(), "range": float(v.max() - v.min()), "sd": float(v.std(ddof=1)), "mean": float(v.mean())}


def show(r):
    f = lambda x: f"{x:+.2f}"
    h = r["heldout"]; st = r["step"]
    iv = lambda d: f"{f(d['delta'])} line [{f(d['line'][0])}, {f(d['line'][1])}] page [{f(d['page'][0])}, {f(d['page'][1])}]"
    print(f"{r['label']:15s} passes {r['passes']} edits {r['edits']} kept {[k['pass'] for k in r['kept']]}")
    print(f"   held-out all {iv(h['all'])}  page<0: {r['page_below_0']}")
    print(f"   Hertwig {iv(h['Hertwig'])} | Schlussred {iv(h['Schlussredaktion'])}")
    if "practice" in r:
        print(f"   practice all {iv(r['practice']['all'])}")
    print(f"   step: a_parent {st['a_parent']:+.3f} b_sibling {st['b_sibling']:+.3f} R2 {st['r2'] if st['r2'] is None else round(st['r2'], 3)} "
          f"cos_parent {st['cos_parent'] if st['cos_parent'] is None else round(st['cos_parent'], 3)} |d| {st['norm']:.2f} blank {st['blank_step']:+.3f} parent_step {st['parent_step']}")


if __name__ == "__main__":
    across = [run_row(k, *v) for k, v in ACROSS.items()]
    out = {"across": across, "across_spread": spread(across),
           "across_spread_parent_steppers": spread([r for r in across if r["step"]["parent_step"]])}
    for r in across: show(r)
    print("across five:", {k: round(v, 3) if isinstance(v, float) else v for k, v in out["across_spread"].items()})
    print("across, parent-step runs:", {k: round(v, 3) if isinstance(v, float) else v for k, v in out["across_spread_parent_steppers"].items()})
    if "--across" not in sys.argv:
        within = [run_row(k, *v) for k, v in WITHIN.items() if (REPO / "runs" / v[0] / "summary.json").exists()
                  and json.load(open(REPO / "runs" / v[0] / "summary.json")).get("heldout")]
        out["within"] = within; out["within_spread"] = spread(within)
        print("== four Fable 5.2 runs")
        for r in within: show(r)
        print("within:", {k: round(v, 3) if isinstance(v, float) else v for k, v in out["within_spread"].items()})
        json.dump(out, open(REPO / "docs/s2_repeats.json", "w"), indent=1)
    else:
        json.dump(out, open(REPO / "docs/s2_repeats_across.json", "w"), indent=1)
