"""Controls for the one held-out gain of the project (docs/CONTROLS_REPORT.md §6): combined-v1 + (blank -0.3, 25 % toward the sibling).

    .venv/bin/python scripts/sham_alpha_combined.py run      # scores everything, 6 workers, CPU
    .venv/bin/python scripts/sham_alpha_combined.py report   # tables

Series, each scored on bern-val (practice, 513 lines) and bern-eval (held-out, 545 lines):
  alpha   non-output tensors alpha x toward schlussredaktion-v1, alpha in 0..1, with and without the blank -0.3 (C3 on the edit's own axis)
  avg     C2: uniform average of combined-v1 and schlussredaktion-v1 over every tensor, output rows aligned by character
  sham    C5: 20 seeds; each non-output tensor gets Gaussian noise of the same Frobenius norm as the real alpha = 0.25 interpolation delta
          of that tensor, on top of blank -0.3. The keep rule of PROTOCOL §3 is applied to each sham exactly as to the real edit
          (against the blank -0.3 state, on the 297 sibling-unseen lines and on all 513).
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

M = Path.home() / "Repo/weightspace/models/kraken"
OUT = Path.home() / "atr-cache/wsp/controls/combined"
W, B = "O_18.lin.weight", "O_18.lin.bias"
ALPHAS = [0.1, 0.25, 0.4, 0.5, 0.75, 1.0]
N_SHAM = 20


def specs():
    s = [{"id": "base", "kind": "base"}, {"id": "blank", "kind": "alpha", "alpha": 0.0, "blank": True}]
    for a in ALPHAS:
        s.append({"id": f"alpha{a}", "kind": "alpha", "alpha": a, "blank": False})
        s.append({"id": f"alpha{a}+blank", "kind": "alpha", "alpha": a, "blank": True})
    s.append({"id": "avg", "kind": "avg"})
    s += [{"id": f"sham{k:02d}", "kind": "sham", "seed": 20261107 + k} for k in range(N_SHAM)]
    return s


def build(sp):
    from wsp.kraken_subject import KrakenSubject
    S = KrakenSubject(M / "kraken-bern-combined-v1.mlmodel"); D = KrakenSubject(M / "kraken-bern-schlussredaktion-v1.mlmodel")
    t, d = S.tensors(), D.tensors()
    body = [n for n in t if not n.startswith("O_") and n in d and d[n].shape == t[n].shape]
    if sp["kind"] == "base":
        return S
    if sp["kind"] == "avg":
        cc, dc = S.codec_map(), D.codec_map()
        for n in body:
            S.set_tensor(n, 0.5 * t[n] + 0.5 * d[n])
        w, b = t[W].copy(), t[B].copy()
        w[0], b[0] = 0.5 * w[0] + 0.5 * d[W][0], 0.5 * b[0] + 0.5 * d[B][0]
        for ch, r in cc.items():
            if ch in dc:
                w[r] = 0.5 * w[r] + 0.5 * d[W][dc[ch]]; b[r] = 0.5 * b[r] + 0.5 * d[B][dc[ch]]
        S.set_tensor(W, w); S.set_tensor(B, b)
        return S
    base_t = {n: t[n].copy() for n in body}
    if sp["kind"] == "alpha":
        for n in body:
            S.set_tensor(n, base_t[n] + sp["alpha"] * (d[n] - base_t[n]))
    elif sp["kind"] == "sham":
        rng = np.random.default_rng(sp["seed"])
        for n in body:
            real = 0.25 * (d[n] - base_t[n]); nr = float(np.linalg.norm(real.astype(np.float64)))
            if nr == 0:
                continue
            z = rng.standard_normal(base_t[n].shape)
            S.set_tensor(n, (base_t[n] + z * (nr / float(np.linalg.norm(z)))).astype(np.float32))
        sp = {**sp, "blank": True}
    if sp.get("blank", sp["kind"] == "sham"):
        bb = S.tensors()[B].copy(); bb[0] -= 0.3; S.set_tensor(B, bb)
    return S


def work(sp):
    out = OUT / f"{sp['id']}.json"
    if out.exists():
        return sp["id"]
    from wsp.data import load
    from wsp.metrics import per_line_edits
    S = build(sp)
    res = {"spec": sp, "digest": S.digest()}
    for name, path in (("practice", "data/practice/bern-val.jsonl"), ("heldout", "data/heldout/bern-eval.jsonl")):
        items = load(path)
        hyps = [p.text for p in S.forward(items)]
        res[name] = {"per_line": per_line_edits(hyps, [i["text"] for i in items])}
    out.write_text(json.dumps(res))
    return sp["id"]


def report():
    from wsp.data import load
    from wsp.metrics import bootstrap_delta, page_of
    prac = load("data/practice/bern-val.jsonl"); held = load("data/heldout/bern-eval.jsonl")
    pc = [page_of(i["key"]) for i in prac]; hc = [page_of(i["key"]) for i in held]
    unseen = [i for i, it in enumerate(prac) if "kraken-bern-schlussredaktion-v1" not in it["seen_by"]]
    R = {p.stem: json.loads(p.read_text()) for p in OUT.glob("*.json")}
    cer = lambda rows: sum(e for e, _ in rows) / sum(c for _, c in rows)
    L = lambda r, k: [tuple(x) for x in r[k]["per_line"]]
    base, blank = R["base"], R["blank"]
    def row(name):
        r = R[name]
        d_o = bootstrap_delta(L(base, "heldout"), L(r, "heldout"), 2000, 20261106, clusters=hc)
        d_b = bootstrap_delta(L(blank, "heldout"), L(r, "heldout"), 2000, 20261106, clusters=hc)
        pk = bootstrap_delta([L(blank, "practice")[i] for i in unseen], [L(r, "practice")[i] for i in unseen], 2000, 20261106)
        pa = bootstrap_delta(L(blank, "practice"), L(r, "practice"), 2000, 20261106)
        return {"id": name, "practice": 100 * cer(L(r, "practice")), "held": 100 * cer(L(r, "heldout")),
                "d_orig": 100 * d_o["delta_cer"], "ci_orig_lines": [100 * x for x in d_o["ci95"]], "ci_orig_pages": [100 * x for x in d_o["ci95_cluster"]],
                "d_blank": 100 * d_b["delta_cer"], "keep_unseen": pk["ci95"][1] < 0, "keep_all": pa["ci95"][1] < 0}
    rows = {n: row(n) for n in R}
    real = rows["alpha0.25+blank"]
    sham = [rows[n] for n in sorted(R) if n.startswith("sham")]
    out = {"rows": rows, "base": {"practice": 100 * cer(L(base, "practice")), "held": 100 * cer(L(base, "heldout"))},
           "sham": {"n": len(sham), "held_d_orig": [s["d_orig"] for s in sham], "practice": [s["practice"] for s in sham],
                    "false_keep_unseen": sum(s["keep_unseen"] for s in sham), "false_keep_all": sum(s["keep_all"] for s in sham),
                    "real_rank_held": sum(1 for s in sham if s["d_orig"] <= real["d_orig"])}}
    Path("docs/controls/combined_controls.json").write_text(json.dumps(out, indent=1))
    print("base", out["base"]); 
    for n in ["blank"] + [f"alpha{a}" for a in ALPHAS] + [f"alpha{a}+blank" for a in ALPHAS] + ["avg"]:
        r = rows[n]; print(f"| {n} | {r['practice']:.2f} | {r['held']:.2f} | {r['d_orig']:+.2f} [{r['ci_orig_lines'][0]:+.2f}, {r['ci_orig_lines'][1]:+.2f}] | [{r['ci_orig_pages'][0]:+.2f}, {r['ci_orig_pages'][1]:+.2f}] |")
    ds = np.array(out["sham"]["held_d_orig"]); ps = np.array(out["sham"]["practice"])
    print("sham held d_orig mean %.3f sd %.3f min %.3f max %.3f; practice mean %.2f min %.2f; false keep unseen %d/%d all %d/%d; shams as good as real (held): %d" % (
        ds.mean(), ds.std(), ds.min(), ds.max(), ps.mean(), ps.min(), out["sham"]["false_keep_unseen"], len(ds), out["sham"]["false_keep_all"], len(ds), out["sham"]["real_rank_held"]))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "run":
        with Pool(6) as p:
            for i in p.imap_unordered(work, specs()):
                print("done", i, flush=True)
    else:
        report()
