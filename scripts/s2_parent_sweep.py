"""The parent step of kraken session 2 at fixed sizes, with no programmer (PROJECT_STATE open item 13).
Pre-registration: docs/S2_PARENT_SWEEP_PREREG.md. CPU.

    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/s2_parent_sweep.py run      # 5 workers, 4 OMP threads each
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/s2_parent_sweep.py report   # writes docs/s2_parent_sweep.json

For each alpha in 0.10, 0.15, ..., 0.30 the subject kraken-bern-combined-v1 is moved alpha of the way toward the parent
hertwig-v2 in every tensor (w <- (1-alpha) w + alpha parent). Output rows are paired by character, the blank weight row
and blank bias with row 0;
rows the parent lacks keep the subject's values. This is the edit the programmers kept (H1 of b10-1..4); at alpha 0.25
and 0.30 it has the digests of the split's part1 in b10-2/-4 and b10-3.
Each model is scored on practice (bern-train-300) and held-out (bern-eval) against C0 (the unedited subject).
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
MOD = Path.home() / "Repo/weightspace/models/kraken"
SUBJ, PARENT = MOD / "kraken-bern-combined-v1.mlmodel", MOD / "kraken-bern-hertwig-v2.mlmodel"
OUT = Path.home() / "atr-cache/wsp/parent_sweep"
ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30]
W, B = "O_18.lin.weight", "O_18.lin.bias"


def model(alpha):
    from wsp.kraken_subject import KrakenSubject
    S = KrakenSubject(SUBJ); P = KrakenSubject(PARENT)
    t, p = S.tensors(), P.tensors()
    cs, cp = S.codec_map(), P.codec_map()
    other = {n: p[n] for n in t if n not in (W, B)}
    w, b = t[W].copy(), t[B].copy()
    w[0], b[0] = p[W][0], p[B][0]
    for ch, i in cs.items():
        j = cp.get(ch)
        if j is not None:
            w[i], b[i] = p[W][j], p[B][j]
    other[W], other[B] = w, b
    for n, v in t.items():
        S.set_tensor(n, ((1.0 - alpha) * v + alpha * other[n]).astype(v.dtype))
    return S


def work(alpha):
    out = OUT / f"alpha_{alpha:.2f}.json"
    if out.exists():
        return alpha
    from wsp.data import load
    from wsp.metrics import per_line_edits, score
    S = model(alpha)
    res = {"alpha": alpha, "digest": S.digest()}
    want = {0.25: "343e03bd086ee217", 0.30: "f8968caf8b0575ab"}.get(alpha)
    assert want is None or res["digest"].startswith(want), (alpha, res["digest"])
    for name, path in [("practice", REPO / "data/practice/bern-train-300.jsonl"), ("heldout", REPO / "data/heldout/bern-eval.jsonl")]:
        items = load(path); hyps = [x.text for x in S.forward(items)]; refs = [x["text"] for x in items]
        m = score(hyps, refs)
        res[name] = {"cer": m.cer, "collapsed": m.collapsed, "per_line": per_line_edits(hyps, refs)}
    out.write_text(json.dumps(res))
    return alpha


def report():
    sys.path.insert(0, str(REPO / "scripts"))
    from s2_repeats import by_hand, ho, pr, T, c0h
    c0p = T(next(m for m in json.load(open(REPO / "docs/baselines/bern-train-300.json"))["models"] if m["model"] == "kraken-bern-combined-v1")["per_line"])
    rows = []
    f = lambda x: f"{x:+.2f}"
    iv = lambda d: f"{f(d['delta'])} line [{f(d['line'][0])}, {f(d['line'][1])}] page [{f(d['page'][0])}, {f(d['page'][1])}]"
    for a in ALPHAS:
        r = json.loads((OUT / f"alpha_{a:.2f}.json").read_text())
        row = {"alpha": a, "digest": r["digest"],
               "practice": {"cer": r["practice"]["cer"], "collapsed": r["practice"]["collapsed"], **by_hand(c0p, T(r["practice"]["per_line"]), pr)},
               "heldout": {"cer": r["heldout"]["cer"], "collapsed": r["heldout"]["collapsed"], **by_hand(c0h, T(r["heldout"]["per_line"]), ho)}}
        row["practice_keep_rule"] = bool(row["practice"]["all"]["line"][1] < 0)
        row["heldout_page_below_0"] = bool(row["heldout"]["all"]["page"][1] < 0)
        rows.append(row)
        print(f"alpha {a:.2f}  practice {r['practice']['cer']:.4f} {iv(row['practice']['all'])} keep={row['practice_keep_rule']} coll {r['practice']['collapsed']}")
        print(f"            held-out {r['heldout']['cer']:.4f} {iv(row['heldout']['all'])} coll {r['heldout']['collapsed']}")
        print(f"            Hertwig {iv(row['heldout']['Hertwig'])} | Schlussred {iv(row['heldout']['Schlussredaktion'])}")
    json.dump({"alphas": ALPHAS, "rows": rows}, open(REPO / "docs/s2_parent_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "run":
        with Pool(len(ALPHAS)) as pool:
            for a in pool.imap_unordered(work, ALPHAS):
                print("done", a, flush=True)
    report()
