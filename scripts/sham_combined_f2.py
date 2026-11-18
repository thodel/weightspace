"""C5 sham and a component split for the Fable 5.2 replicate of kraken session 2
(runs/20261115-bern-combined-v1-f2-b10-1); adapted from scripts/sham_combined_f1.py, scored on practice (bern-train-300) and held-out (bern-eval), CPU.

    .venv/bin/python scripts/sham_combined_f2.py run       # 4 workers
    .venv/bin/python scripts/sham_combined_f2.py report

Variants (each against the unedited subject, C0):
  final      the frozen model (the session's kept diff)
  blank      only H3's blank-bias step (O_18.lin.bias[0] -0.30)
  interp     only H2's interpolation (the kept diff minus that step; it includes the interpolation's own -0.030 on the blank)
  sham00-04  C5: every tensor gets Gaussian noise of the same Frobenius norm as the kept diff of that tensor
             (O_18.lin.bias: the noise vector has the norm of the whole bias diff, blank element included). Seeds 20261111+k.
The keep rule of PROTOCOL §3 (paired line interval vs C0 below 0) is applied to each variant on practice.
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "runs/20261115-bern-combined-v1-f2-b10-1"
SUBJ = Path.home() / "Repo/weightspace/models/kraken/kraken-bern-combined-v1.mlmodel"
OUT = Path.home() / "atr-cache/wsp/controls/combined-f2"
SEEDS = [20261111 + k for k in range(5)]


def specs():
    return [{"id": "final"}, {"id": "blank"}, {"id": "interp"}] + [{"id": f"sham{k:02d}", "seed": s} for k, s in enumerate(SEEDS)]


def build(sp):
    from wsp.kraken_subject import KrakenSubject
    S = KrakenSubject(SUBJ); F = KrakenSubject(RUN / "final.mlmodel")
    t, f = S.tensors(), F.tensors()
    t0 = {n: v.copy() for n, v in t.items()}
    d = {n: f[n].astype(np.float64) - t0[n].astype(np.float64) for n in t0 if not np.array_equal(f[n], t0[n])}
    if sp["id"] == "final":
        return F, d
    if sp["id"] in ("blank", "interp"):
        for n, dv in d.items():
            b = np.zeros_like(dv)
            if n == "O_18.lin.bias":
                b[0] = -0.30  # H3, applied after H2 (log.jsonl)
            part = b if sp["id"] == "blank" else dv - b
            if np.any(part):
                S.set_tensor(n, (t0[n] + part).astype(np.float32))
        return S, d
    rng = np.random.default_rng(sp["seed"])
    for n, dv in d.items():
        z = rng.standard_normal(dv.shape)
        S.set_tensor(n, (t0[n] + z * (np.linalg.norm(dv) / np.linalg.norm(z))).astype(np.float32))
    return S, d


def work(sp):
    out = OUT / f"{sp['id']}.json"
    if out.exists():
        return sp["id"]
    from wsp.data import load
    from wsp.metrics import per_line_edits, score
    S, d = build(sp)
    res = {"id": sp["id"], "seed": sp.get("seed"), "digest": S.digest(),
           "diff_norm": float(np.sqrt(sum(np.linalg.norm(v) ** 2 for v in d.values())))}
    for name, path in [("practice", REPO / "data/practice/bern-train-300.jsonl"), ("heldout", REPO / "data/heldout/bern-eval.jsonl")]:
        items = load(path); hyps = [p.text for p in S.forward(items)]; refs = [x["text"] for x in items]
        m = score(hyps, refs)
        res[name] = {"cer": m.cer, "collapsed": m.collapsed, "per_line": per_line_edits(hyps, refs)}
    out.write_text(json.dumps(res))
    return sp["id"]


def report():
    from wsp.data import load
    from wsp.metrics import bootstrap_delta
    base = json.loads((REPO / "docs/baselines/bern-train-300.json").read_text())["models"][0]["per_line"]
    hbase = json.loads((REPO / "docs/baselines/heldout-bern-eval.json").read_text())["models"][0]["per_line"]
    pg = lambda f: [x["key"].rsplit("/", 1)[0] for x in load(REPO / f)]
    cp, ch = pg("data/practice/bern-train-300.jsonl"), pg("data/heldout/bern-eval.jsonl")
    rows = {}
    for sp in specs():
        r = json.loads((OUT / f"{sp['id']}.json").read_text())
        p = bootstrap_delta([tuple(x) for x in base], [tuple(x) for x in r["practice"]["per_line"]], clusters=cp)
        h = bootstrap_delta([tuple(x) for x in hbase], [tuple(x) for x in r["heldout"]["per_line"]], clusters=ch)
        kept = p["ci95"][1] < 0 and p["delta_cer"] < 0
        rows[sp["id"]] = {"diff_norm": r["diff_norm"], "practice": {"cer": r["practice"]["cer"], "collapsed": r["practice"]["collapsed"], **{k: p[k] for k in ("delta_cer", "ci95", "ci95_cluster")}},
                          "heldout": {"cer": r["heldout"]["cer"], "collapsed": r["heldout"]["collapsed"], **{k: h[k] for k in ("delta_cer", "ci95", "ci95_cluster")}},
                          "keep_rule_vs_C0": kept}
        f = lambda v: f"{v:+.4f}"
        print(f"{sp['id']:8s} |d| {r['diff_norm']:7.2f}  practice {r['practice']['cer']:.4f} {f(p['delta_cer'])} line [{f(p['ci95'][0])},{f(p['ci95'][1])}] page [{f(p['ci95_cluster'][0])},{f(p['ci95_cluster'][1])}] coll {r['practice']['collapsed']} kept={kept} | "
              f"held-out {r['heldout']['cer']:.4f} {f(h['delta_cer'])} line [{f(h['ci95'][0])},{f(h['ci95'][1])}] page [{f(h['ci95_cluster'][0])},{f(h['ci95_cluster'][1])}] coll {r['heldout']['collapsed']}")
    (RUN / "controls.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "run":
        with Pool(4) as pool:
            for i in pool.imap_unordered(work, specs()):
                print("done", i, flush=True)
    report()
