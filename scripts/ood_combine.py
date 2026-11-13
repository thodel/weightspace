"""Combined block moves within a lineage, read in parallel CPU processes, with line-level and page-clustered
paired intervals (docs/MODEL_FAMILY.md, 08.11.2026; the same moves as ``wsp.exchange --combo``).

    CUDA_VISIBLE_DEVICES="" python scripts/ood_combine.py --specific S.mlmodel --general G.mlmodel \\
        --sets <own set> [<own set 2>] <ood set> ... --own 2 \\
        --combo c6h=C_6:0.5 --combo c0c6h=C_0:0.5,C_6:0.5 ... --out docs/family/ood-combine-<S>.json

Each block of a combination becomes ``S + β (G − S)`` (output rows paired by character, as in
``wsp.exchange``); every combination reads every set and is compared, paired per line, with S unedited:
corpus Δ CER, a line-level 95 % interval (2,000 resamples, seed 20261106) and a two-stage page-clustered one.
The page of a line is ``wsp.metrics.page_of(key)``, or the ``page_<id>`` in the key where the source stores one line
per image (federal minutes). Practice and confirmation lists only, except with ``--heldout`` (the one
pre-registered held-out score, docs/family/PREREG-ood-exchange.md).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(REPO))          # this checkout's wsp, not the venv's editable install
LISTS = [REPO / "data" / "practice", REPO / "data" / "heldout"]


def find_list(name: str) -> Path:
    for d in LISTS:
        p = d / f"{name}.jsonl"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def cluster_of(key: str) -> str:
    from wsp.metrics import page_of
    m = re.search(r"page_(\d+)", key)
    return m.group(1) if m else page_of(key)


def parse_combo(spec: str):
    name, _, moves = spec.partition("=")
    return name, [(b, float(beta)) for b, beta in (m.split(":") for m in moves.split(",") if m)]


def _task(args):
    specific, general, moves, set_name, threads, readings, allow_heldout, name = args
    import torch
    torch.set_num_threads(threads)
    import sys
    sys.path.insert(0, str(REPO))
    from wsp.alpha_sweep import aligned_parent_output, block_names
    from wsp.data import load
    from wsp.kraken_subject import KrakenSubject
    from wsp.metrics import per_line_edits
    lst = find_list(set_name)
    if lst.parent.name == "heldout" and not allow_heldout:
        raise SystemExit(f"{set_name} is a held-out list")
    items = load(lst)
    S = KrakenSubject(specific)
    if moves:
        G = KrakenSubject(general)
        st, gt = S.tensors(), G.tensors()
        out_al = aligned_parent_output(G, S)
        orig = {n: v.copy() for n, v in st.items()}
        for b, beta in moves:
            target = ({n: v for n, v in out_al.items() if not n.startswith("_")} if b.startswith("O_")
                      else {n: gt[n] for n in block_names(st, b)})
            if not target:
                raise KeyError(b)
            for n in target:
                S.set_tensor(n, orig[n] + beta * (target[n] - orig[n]))
    preds = S.forward(items)
    hyps = [p.text for p in preds]; refs = [x["text"] for x in items]
    if readings:
        out = Path(readings) / set_name / f"{name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps({"key": p.key, "hyp": h, "ref": r}, ensure_ascii=False) + "\n"
                               for p, h, r in zip(preds, hyps, refs)), encoding="utf-8")
    return {"keys": [x["key"] for x in items], "pl": per_line_edits(hyps, refs), "digest": S.digest(), "hyps": hyps, "refs": refs}


def run(specific, general, sets, combos, out, procs=12, threads=3, allow_heldout=False, log=print):
    from wsp.data import sha256_file
    from wsp.metrics import bootstrap_delta, score
    readings = REPO / "docs" / "family" / "readings" / Path(out).stem
    jobs = [("S", [], s) for s in sets] + [("G", None, s) for s in sets] + [(n, mv, s) for n, mv in combos for s in sets]
    tasks = []
    for name, mv, s in jobs:
        if name == "G":                     # the general model unedited: every block transplanted is not the same thing,
            tasks.append((general, general, [], s, threads, str(readings), allow_heldout, name))   # read G itself
        else:
            tasks.append((specific, general, mv, s, threads, str(readings), allow_heldout, name))
    with ProcessPoolExecutor(procs) as ex:
        results = list(ex.map(_task, tasks))
    by = {(t[-1], t[3]): r for t, r in zip(tasks, results)}
    res = {"kind": "combinations (parallel)", "specific": Path(specific).stem, "general": Path(general).stem,
           "specific_sha256": sha256_file(Path(specific)), "general_sha256": sha256_file(Path(general)),
           "sets": sets, "set_sha256": {s: sha256_file(find_list(s)) for s in sets}, "c0": {}, "general_cer": {}, "rows": [],
           "bootstrap": {"resamples": 2000, "seed": 20261106, "cluster": "page (wsp.metrics.page_of, or page_<id> in the key)"}}
    for s in sets:
        r = by[("S", s)]; m = score(r["hyps"], r["refs"], blocks=False)
        res["c0"][s] = {"cer": m.cer, "wer": m.wer, "collapsed": m.collapsed, "lines": m.lines,
                        "pages": len(set(cluster_of(k) for k in r["keys"]))}
        res["general_cer"][s] = score(by[("G", s)]["hyps"], by[("G", s)]["refs"], blocks=False).cer
        log(f"{s:28s} S {100*m.cer:6.2f} %   G {100*res['general_cer'][s]:6.2f} %")
    for name, mv in combos:
        row = {"combo": name, "moves": [[b, beta] for b, beta in mv], "sets": {}}
        for s in sets:
            r0, r = by[("S", s)], by[(name, s)]
            row["digest"] = r["digest"]
            m = score(r["hyps"], r["refs"], blocks=False)
            boot = bootstrap_delta(r0["pl"], r["pl"], clusters=[cluster_of(k) for k in r0["keys"]])
            row["sets"][s] = {"cer": m.cer, "collapsed": m.collapsed, "vs_S": boot}
            log(f"{name:14s} {s:28s} Δ {100*boot['delta_cer']:+6.2f}  line [{100*boot['ci95'][0]:+.2f}, {100*boot['ci95'][1]:+.2f}]"
                f"  page [{100*boot['ci95_cluster'][0]:+.2f}, {100*boot['ci95_cluster'][1]:+.2f}]")
        res["rows"].append(row)
    res["when"] = time.strftime("%Y-%m-%d %H:%M")
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--specific", required=True); ap.add_argument("--general", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--combo", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12); ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--heldout", action="store_true", help="the one pre-registered held-out score: allows data/heldout lists")
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run(a.specific, a.general, a.sets, [parse_combo(c) for c in a.combo], a.out, a.procs, a.threads, allow_heldout=a.heldout)


if __name__ == "__main__":
    main()
