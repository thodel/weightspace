"""Global interpolation and additivity for the Kurrent pair xix-v1 -> xix-blb-v1 (docs/CONTROLS_REPORT.md §7).

    .venv/bin/python scripts/global_interp.py run|report

W(alpha) = parent + alpha (child - parent) on a set of blocks, the rest left as the child (alpha = 1 is the child, 0 the parent on those blocks,
1.5 extrapolates). Blocks: C_0 C_3 C_6 C_9 L_12 L_14 L_16 O_18; the output layer's rows are aligned by character, the child's 72 new rows keep
their values. Specs: all eight blocks jointly at alpha 0..1.5 in steps of 0.25; each single block at the same alphas; all 28 pairs at alpha 0.5.
Scored on blb-val-300 (practice side: eval-side objects the model was not trained on, labelled `practice`) and blb-eval-hand-b (held-out side,
labelled `heldout`). Additivity: sum of the single-block costs against the joint cost at the same alpha, and for pairs against the sum of the two.
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

M = Path.home() / "Repo/weightspace/models/kraken"
OUT = Path.home() / "atr-cache/wsp/controls/interp"
BLOCKS = ["C_0", "C_3", "C_6", "C_9", "L_12", "L_14", "L_16", "O_18"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.25, 1.5]          # 1.0 is the child itself
SETS = (("practice", "data/practice/blb-val-300.jsonl"), ("heldout", "data/heldout/blb-eval-hand-b.jsonl"))


def specs():
    s = [{"id": "child", "blocks": [], "alpha": 1.0}]
    for a in ALPHAS:
        s.append({"id": f"joint_a{a}", "blocks": BLOCKS, "alpha": a})
        for b in BLOCKS:
            s.append({"id": f"single_{b}_a{a}", "blocks": [b], "alpha": a})
    for x, y in itertools.combinations(BLOCKS, 2):
        s.append({"id": f"pair_{x}_{y}_a0.5", "blocks": [x, y], "alpha": 0.5})
    return s


def build(sp):
    from wsp.kraken_subject import KrakenSubject
    from wsp.alpha_sweep import aligned_parent_output
    C = KrakenSubject(M / "kraken-german-xix-blb-v1.mlmodel"); P = KrakenSubject(M / "kraken-german-xix-v1.mlmodel")
    ct, pt = C.tensors(), P.tensors()
    out_al = aligned_parent_output(P, C)
    a = sp["alpha"]
    for b in sp["blocks"]:
        names = [n for n in ct if n.split(".")[0] == b]
        for n in names:
            par = {k: v for k, v in out_al.items() if not k.startswith("_")}[n] if b == "O_18" else pt[n]
            C.set_tensor(n, (par + a * (ct[n] - par)).astype(np.float32))
    return C


def work(sp):
    out = OUT / f"{sp['id']}.json"
    if out.exists():
        return sp["id"]
    from wsp.data import load
    from wsp.metrics import per_line_edits
    C = build(sp)
    res = {"spec": sp, "digest": C.digest()}
    for name, path in SETS:
        items = load(path)
        hyps = [p.text for p in C.forward(items)]
        res[name] = {"per_line": per_line_edits(hyps, [i["text"] for i in items])}
    out.write_text(json.dumps(res))
    return sp["id"]


def report():
    from wsp.data import load
    from wsp.metrics import page_of
    R = {p.stem: json.loads(p.read_text()) for p in OUT.glob("*.json")}
    keys = {n: [i["key"] for i in load(p)] for n, p in SETS}
    cl = {n: np.array([hash(page_of(k)) for k in ks]) for n, ks in keys.items()}
    def arr(r, s): return np.array(r[s]["per_line"], dtype=float)
    def cer(a): return a[:, 0].sum() / a[:, 1].sum()
    def boot(stat, s, n=2000, seed=20261106):
        """stat(idx_weights) -> float, over a two-stage page bootstrap; returns 95 % interval."""
        g = np.random.default_rng(seed); pages = {}
        for i, c in enumerate(cl[s]): pages.setdefault(c, []).append(i)
        arrs = [np.array(v) for v in pages.values()]; out = []
        for _ in range(n):
            idx = np.concatenate([a[g.integers(0, len(a), len(a))] for a in (arrs[k] for k in g.integers(0, len(arrs), len(arrs)))])
            out.append(stat(idx))
        out.sort(); return [out[int(0.025 * n)], out[int(0.975 * n)]]
    res = {}
    for s in ("practice", "heldout"):
        child = arr(R["child"], s); c0 = cer(child)
        d = lambda r: arr(r, s)
        tab = {"child_cer": 100 * c0, "joint": {}, "single_sum": {}, "additivity": {}, "pairs": {}}
        for a in ALPHAS:
            j = d(R[f"joint_a{a}"]); singles = [d(R[f"single_{b}_a{a}"]) for b in BLOCKS]
            dj = (j[:, 0].sum() - child[:, 0].sum()) / child[:, 1].sum()
            ds = sum((x[:, 0].sum() - child[:, 0].sum()) for x in singles) / child[:, 1].sum()
            inter = lambda idx, j=j, singles=singles: ((j[idx, 0] - child[idx, 0]).sum() - sum((x[idx, 0] - child[idx, 0]).sum() for x in singles)) / child[idx, 1].sum()
            tab["joint"][a] = {"cer": 100 * cer(j), "delta_pts": 100 * dj}
            tab["single_sum"][a] = {"singles_delta_pts": {b: 100 * (cer(x) - c0) for b, x in zip(BLOCKS, singles)}, "sum_pts": 100 * ds}
            tab["additivity"][a] = {"joint_minus_sum_pts": 100 * (dj - ds), "ci95_pages": [100 * x for x in boot(inter, s)]}
        pairs = []
        for x, y in itertools.combinations(BLOCKS, 2):
            p = d(R[f"pair_{x}_{y}_a0.5"]); ax, ay = d(R[f"single_{x}_a0.5"]), d(R[f"single_{y}_a0.5"])
            dp = (p[:, 0].sum() - child[:, 0].sum()) / child[:, 1].sum()
            dsum = ((ax[:, 0].sum() - child[:, 0].sum()) + (ay[:, 0].sum() - child[:, 0].sum())) / child[:, 1].sum()
            pairs.append({"pair": f"{x}+{y}", "pair_delta_pts": 100 * dp, "sum_of_singles_pts": 100 * dsum, "interaction_pts": 100 * (dp - dsum)})
        tab["pairs"] = pairs
        res[s] = tab
    Path("docs/controls/global_interp.json").write_text(json.dumps(res, indent=1))
    for s in res:
        print("##", s, "child CER %.2f" % res[s]["child_cer"])
        print("| alpha | joint CER | joint Δ (pt) | sum of single-block Δ (pt) | joint − sum (pt) [page 95 %] |")
        for a in ALPHAS:
            print(f"| {a} | {res[s]['joint'][a]['cer']:.2f} | {res[s]['joint'][a]['delta_pts']:+.2f} | {res[s]['single_sum'][a]['sum_pts']:+.2f} | {res[s]['additivity'][a]['joint_minus_sum_pts']:+.2f} [{res[s]['additivity'][a]['ci95_pages'][0]:+.2f}, {res[s]['additivity'][a]['ci95_pages'][1]:+.2f}] |")
        ps = res[s]["pairs"]; x = np.array([p["sum_of_singles_pts"] for p in ps]); y = np.array([p["pair_delta_pts"] for p in ps])
        print("pairs at alpha 0.5: corr(sum, pair) %.3f, mean interaction %+.3f, max |interaction| %.3f (%s)" % (
            np.corrcoef(x, y)[0, 1], np.mean(y - x), np.max(np.abs(y - x)), max(ps, key=lambda p: abs(p["interaction_pts"]))["pair"]))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "run":
        with Pool(6) as p:
            for i in p.imap_unordered(work, specs()):
                print("done", i, flush=True)
    else:
        report()
